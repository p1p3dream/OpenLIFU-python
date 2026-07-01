from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import pandas as pd
import xarray as xa
from scipy.ndimage import zoom as scipy_zoom

from openlifu.seg.material import CORTICAL_BONE, MATERIALS, MATERIALS_TWO_CLASS_BONE, TRABECULAR_BONE, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER
from openlifu.util.annotations import OpenLIFUFieldData

logger = logging.getLogger(__name__)

# Patch sizes per model type, matching the nnU-Net training configuration.
PATCH_SIZES: dict[str, tuple[int, int, int]] = {
    "skull": (128, 128, 128),
    "fullhead": (128, 160, 112),
}

# Mapping from nnU-Net integer label to material key, per model type.
LABEL_MAP_FULLHEAD: dict[int, str] = {
    0: "water",
    1: "air",
    2: "csf",
    3: "gray_matter",
    4: "white_matter",
    5: "skull",
    6: "tissue",
}

LABEL_MAP_SKULL: dict[int, str] = {
    0: "water",
    1: "skull",
}

# Two-class bone label map: splits skull into cortical and trabecular bone.
# Used when bone_model="two_class" with fullhead segmentation. The nnU-Net
# fullhead model outputs label 5 as "skull", but when fed pre-segmented
# volumes (e.g. SimNIBS CHARM output remapped with --two-class-bone),
# label 5 = cortical_bone and label 7 = trabecular_bone.
LABEL_MAP_FULLHEAD_TWO_CLASS_BONE: dict[int, str] = {
    0: "water",
    1: "air",
    2: "csf",
    3: "gray_matter",
    4: "white_matter",
    5: "cortical_bone",
    6: "tissue",
    7: "trabecular_bone",
}

# Material key sets for validation (parallel to ThresholdMRI pattern).
_BASE_MATERIAL_KEYS = frozenset({"water", "skull", "air"})
_BASE_MATERIAL_KEYS_TWO_CLASS = frozenset({"water", "cortical_bone", "trabecular_bone", "air"})
_FULLHEAD_EXTRA_KEYS = frozenset({"csf", "gray_matter", "white_matter"})
_SKULL_EXTRA_KEYS = frozenset({"tissue"})

# Canonical material key order for single-bone NNUNetSegmentation. Skull is
# forced to index 1 so that saved label volumes are interoperable with the
# project convention (label 1 == skull/bone) used by ThresholdMRI, the N=180
# validation producer, and the tissue-masking recompute. Without this, NNUNet
# would emit skull at index 2 because MATERIALS keeps "tissue" before "skull".
# Only the KEY ORDER is constrained here; the Material definitions are taken
# verbatim from the caller/default factory. Acoustic-param lookup in
# SegmentationMethod._map_params uses the same _material_indices(), so this is
# self-consistent within the OpenLIFU pipeline. Two-class bone mode keeps its
# own order (cortical/trabecular) and is not used for live ONNX inference.
_SINGLE_BONE_MATERIAL_ORDER: tuple[str, ...] = (
    "water", "skull", "air", "standoff", "tissue",
    "csf", "gray_matter", "white_matter",
)


def _default_materials_fullhead() -> dict[str, Material]:
    """Default materials dict for fullhead mode (adds brain subtypes, keeps tissue for scalp)."""
    m = MATERIALS.copy()
    # Keep "tissue" for label 6 (soft tissue/scalp). Add brain subtypes.
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


def _default_materials_fullhead_two_class_bone() -> dict[str, Material]:
    """Default materials dict for fullhead mode with two-class bone model.

    Replaces the single "skull" material with cortical_bone and trabecular_bone,
    using ITRUSST benchmark acoustic properties. Keeps tissue for scalp and
    adds brain subtypes.
    """
    m = MATERIALS_TWO_CLASS_BONE.copy()
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


@dataclass
class NNUNetSegmentation(SegmentationMethod):
    """ONNX-based segmentation using a nnU-Net model exported to ONNX format.

    Performs sliding-window inference on a 3-D MRI volume to produce a label map
    of skull, brain tissue, and surrounding regions. Two model types are
    supported:

    - ``"skull"``: Binary segmentation producing skull vs. non-skull labels.
      The brain interior is assigned a single "tissue" material.
    - ``"fullhead"``: Seven-class segmentation producing water, air, CSF,
      gray matter, white matter, skull, and soft tissue labels.

    The inference pipeline resamples the input to 1 mm isotropic spacing,
    crops to the foreground, normalizes intensities via z-score, runs
    sliding-window inference with Gaussian-weighted patch blending, and maps
    the resulting labels back to the original volume geometry.

    This class depends only on ``onnxruntime`` for inference (no PyTorch or
    nnU-Net installation required). The ONNX model file can be supplied via
    ``model_path`` or auto-downloaded when left empty.

    Each axis of the input volume must have at least 2 coordinate values
    so that voxel spacing can be computed. Coordinate spacing is assumed
    to be uniform along each axis.

    Limitations (status: unverified prototype, GLADYS minimal correctness pass):

    - Preprocessing (resample, foreground crop, z-score, sliding-window blend)
      is a hand-rolled reimplementation, NOT a bit-faithful copy of nnU-Net v2's
      ``predict_from_raw_data`` pipeline. It has not been checked against the
      nnU-Net runtime output. Treat results as unverified until a
      CT-ground-truth comparison reproduces the N=180 accuracy.
    - This fullhead ONNX path has not been validated against CT ground truth.
      (A separate skull-only nnU-Net model, Dataset001, has been CT-validated on
      held-out SynthRAD with skull Dice 0.903; that is a different code path, not
      this class.) For transcranial FUS, ThresholdMRI is the validated default
      (see ``openlifu.gladys.config.default_seg_method``) and remains the
      upstream-bound path.
    - Test-time augmentation (``use_mirroring``) is OFF by default to keep CPU
      inference around 2 min instead of around 13 min. Set ``use_mirroring=True``
      when accuracy matters more than latency.
    - Auto-download of the ONNX model is not implemented; supply ``model_path``
      explicitly.
    """

    model_path: Annotated[
        str,
        OpenLIFUFieldData(
            "ONNX model path",
            "Path to the .onnx model file. If empty, the model will be "
            "auto-downloaded on first use.",
        ),
    ] = ""
    """Path to the .onnx model file. Leave empty for auto-download."""

    model_type: Annotated[
        str,
        OpenLIFUFieldData(
            "Model type",
            'Either "skull" for binary skull segmentation or "fullhead" '
            "for 7-class head segmentation.",
        ),
    ] = "fullhead"
    """Model type: 'skull' for binary or 'fullhead' for 7-class segmentation."""

    use_gpu: Annotated[
        bool,
        OpenLIFUFieldData(
            "Use GPU",
            "If True, run ONNX inference on GPU via CUDAExecutionProvider. "
            "Falls back to CPU if CUDA is not available.",
        ),
    ] = False
    """If True, prefer GPU execution via CUDAExecutionProvider."""

    use_mirroring: Annotated[
        bool,
        OpenLIFUFieldData(
            "Use mirroring (TTA)",
            "If True, apply test-time augmentation by averaging predictions "
            "across axis-flipped versions of each patch. TTA multiplies CPU "
            "inference time by ~8, so it is OFF by default; enable it when "
            "accuracy matters more than latency.",
        ),
    ] = False
    """If True, apply test-time augmentation via axis mirroring. Off by default."""

    bone_model: Annotated[
        str,
        OpenLIFUFieldData(
            "Bone model",
            'Either "single" for one skull material or "two_class" to split '
            "skull into cortical bone (outer/inner table) and trabecular bone "
            "(diploe). Two-class mode uses ITRUSST benchmark properties and "
            "requires label maps with separate cortical/trabecular labels.",
        ),
    ] = "single"
    """Bone model: 'single' for one skull material, 'two_class' for cortical + trabecular."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.model_type not in PATCH_SIZES:
            valid = ", ".join(sorted(PATCH_SIZES.keys()))
            msg = f"model_type must be one of [{valid}], got '{self.model_type}'."
            raise ValueError(msg)

        if self.bone_model not in ("single", "two_class"):
            msg = f"bone_model must be 'single' or 'two_class', got '{self.bone_model}'."
            raise ValueError(msg)

        # Auto-add brain tissue materials for fullhead mode.
        # Unlike ThresholdMRI, we keep "tissue" because the fullhead model
        # uses it for label 6 (soft tissue/scalp). We only ADD the brain
        # subtypes if they are missing.
        if self.model_type == "fullhead" and "csf" not in self.materials:
            self.materials = dict(self.materials)
            self.materials.setdefault("csf", CSF)
            self.materials.setdefault("gray_matter", GRAY_MATTER)
            self.materials.setdefault("white_matter", WHITE_MATTER)

        # Auto-add two-class bone materials when bone_model="two_class".
        if self.bone_model == "two_class":
            self.materials = dict(self.materials)
            self.materials.setdefault("cortical_bone", CORTICAL_BONE)
            self.materials.setdefault("trabecular_bone", TRABECULAR_BONE)

        # Validate that all required material keys are present.
        if self.bone_model == "two_class":
            if self.model_type == "fullhead":
                required = _BASE_MATERIAL_KEYS_TWO_CLASS | _FULLHEAD_EXTRA_KEYS | {"tissue"}
            else:
                required = _BASE_MATERIAL_KEYS_TWO_CLASS | _SKULL_EXTRA_KEYS
        else:
            if self.model_type == "fullhead":
                required = _BASE_MATERIAL_KEYS | _FULLHEAD_EXTRA_KEYS | {"tissue"}
            else:
                required = _BASE_MATERIAL_KEYS | _SKULL_EXTRA_KEYS
        missing = required - set(self.materials.keys())
        if missing:
            msg = (
                f"NNUNetSegmentation (model_type='{self.model_type}', "
                f"bone_model='{self.bone_model}') "
                f"requires material keys {required}, missing: {missing}."
            )
            raise ValueError(msg)

        # Force the canonical single-bone material index order so the skull
        # label matches the project convention (label 1 == skull). See
        # _SINGLE_BONE_MATERIAL_ORDER. Only key order changes; Material values
        # are preserved. Two-class bone keeps its own order.
        if self.bone_model == "single":
            ordered = {
                key: self.materials[key]
                for key in _SINGLE_BONE_MATERIAL_ORDER
                if key in self.materials
            }
            for key, value in self.materials.items():
                if key not in ordered:
                    ordered[key] = value
            self.materials = ordered

        # The ONNX session is lazily initialized and cached.
        self._session: Any = None

    # ------------------------------------------------------------------
    # ONNX session management
    # ------------------------------------------------------------------

    def _get_session(self) -> Any:
        """Return the cached ONNX InferenceSession, creating it on first call.

        The session is stored on ``self._session`` so subsequent calls to
        ``_segment`` reuse it without reloading the model.

        :returns: An ``onnxruntime.InferenceSession`` instance.
        :raises FileNotFoundError: If ``model_path`` does not point to an
            existing file and auto-download is not yet implemented.
        :raises ImportError: If ``onnxruntime`` is not installed.
        """
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        model_path = self._resolve_model_path()

        providers: list[str] = []
        if self.use_gpu:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            else:
                logger.warning(
                    "CUDAExecutionProvider not available; falling back to CPU."
                )
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Limit intra-op parallelism to avoid contention on multi-socket systems.
        sess_opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers,
        )
        return self._session

    def _resolve_model_path(self) -> str:
        """Resolve the ONNX model path.

        If ``model_path`` is set and the file exists, return it directly.
        If ``model_path`` is empty, attempt auto-download (placeholder for
        future asset management). Raises ``FileNotFoundError`` if the model
        cannot be located.

        :returns: Absolute path string to the .onnx file.
        """
        if self.model_path:
            from pathlib import Path

            p = Path(self.model_path)
            if not p.exists():
                msg = f"ONNX model not found at '{self.model_path}'."
                raise FileNotFoundError(msg)
            return str(p.resolve())

        # Auto-download placeholder: in production this would call
        # openlifu.util.assets.install_asset with the appropriate URL.
        msg = (
            "model_path is empty and auto-download is not yet configured. "
            "Please provide an explicit path to the .onnx model file."
        )
        raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # Gaussian blending kernel
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gaussian_kernel(
        patch_shape: tuple[int, ...],
        sigma_scale: float = 0.125,
    ) -> np.ndarray:
        """Build a 3-D Gaussian importance-weighting kernel for patch blending.

        The kernel is constructed as the outer product of three 1-D Gaussians,
        one per axis. The sigma for each axis is ``sigma_scale * axis_length``,
        matching the nnU-Net default weighting scheme.

        :param patch_shape: Spatial dimensions of the inference patch.
        :param sigma_scale: Fraction of each axis length used as the Gaussian
            sigma. Default 0.125 matches nnU-Net internal blending.
        :returns: Kernel array with the same shape as ``patch_shape``, values
            in (0, 1] with peak at center.
        """
        kernels_1d = []
        for s in patch_shape:
            ax = np.arange(s, dtype=np.float64) - (s - 1) / 2.0
            sigma = max(s * sigma_scale, 1.0)
            k = np.exp(-0.5 * (ax / sigma) ** 2)
            kernels_1d.append(k)
        kernel = (
            kernels_1d[0][:, None, None]
            * kernels_1d[1][None, :, None]
            * kernels_1d[2][None, None, :]
        )
        kernel = kernel.astype(np.float32)
        # Clamp very small values to avoid near-zero weights at patch edges.
        kernel = np.clip(kernel, a_min=1e-5, a_max=None)
        return kernel

    # ------------------------------------------------------------------
    # Sliding window inference
    # ------------------------------------------------------------------

    def _run_model(self, patch: np.ndarray) -> np.ndarray:
        """Run the ONNX model on a single 5-D patch (1, 1, D, H, W).

        :param patch: Input array with shape ``(1, 1, D, H, W)``, float32.
        :returns: Raw model output array with shape ``(1, C, D, H, W)``.
        """
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: patch})
        return outputs[0]

    def _sliding_window_inference(
        self,
        volume: np.ndarray,
        patch_size: tuple[int, int, int],
        num_classes: int,
    ) -> np.ndarray:
        """Run sliding-window inference over a 3-D volume.

        Patches are extracted with 50% overlap (step = patch_size // 2).
        Each patch is blended into the output using a Gaussian importance
        kernel. When ``use_mirroring`` is True, predictions are averaged
        over all 8 axis-flip combinations (test-time augmentation).

        :param volume: Preprocessed 3-D array (D, H, W), float32.
        :param patch_size: Spatial patch dimensions (D, H, W).
        :param num_classes: Number of output classes from the model.
        :returns: Aggregated prediction array (C, D, H, W), after softmax
            and Gaussian-weighted blending.
        """
        spatial_shape = volume.shape
        gaussian_kernel = self._build_gaussian_kernel(patch_size)

        # Accumulator for weighted predictions and weight map.
        aggregated = np.zeros((num_classes, *spatial_shape), dtype=np.float64)
        weight_map = np.zeros(spatial_shape, dtype=np.float64)

        # Step size is 50% of patch size along each axis.
        steps = tuple(max(1, ps // 2) for ps in patch_size)

        # Compute starting positions for each axis. The last position is
        # adjusted so the final patch always reaches the volume boundary.
        def _start_positions(total: int, patch_len: int, step: int) -> list[int]:
            positions = list(range(0, max(total - patch_len + 1, 1), step))
            if not positions or positions[-1] + patch_len < total:
                positions.append(max(0, total - patch_len))
            return sorted(set(positions))

        starts_d = _start_positions(spatial_shape[0], patch_size[0], steps[0])
        starts_h = _start_positions(spatial_shape[1], patch_size[1], steps[1])
        starts_w = _start_positions(spatial_shape[2], patch_size[2], steps[2])

        # All axis-flip combinations for test-time augmentation.
        # Axes 2, 3, 4 in the 5-D tensor (1, 1, D, H, W) correspond to D, H, W.
        if self.use_mirroring:
            mirror_axes_list: list[tuple[int, ...]] = [()]
            for d_ax in [(), (2,)]:
                for h_ax in [(), (3,)]:
                    for w_ax in [(), (4,)]:
                        combo = d_ax + h_ax + w_ax
                        if combo:
                            mirror_axes_list.append(combo)
        else:
            mirror_axes_list = [()]

        n_mirrors = len(mirror_axes_list)
        total_patches = len(starts_d) * len(starts_h) * len(starts_w)
        logger.info(
            "Sliding window: %d patches, %d mirror(s) each, patch size %s",
            total_patches, n_mirrors, patch_size,
        )

        patch_count = 0
        for d0 in starts_d:
            for h0 in starts_h:
                for w0 in starts_w:
                    d1 = d0 + patch_size[0]
                    h1 = h0 + patch_size[1]
                    w1 = w0 + patch_size[2]

                    patch_data = volume[d0:d1, h0:h1, w0:w1]
                    patch_5d = patch_data[np.newaxis, np.newaxis].astype(np.float32)

                    # Accumulate predictions, optionally with TTA mirroring.
                    pred_accum = np.zeros(
                        (1, num_classes, *patch_size), dtype=np.float64,
                    )
                    for axes in mirror_axes_list:
                        if axes:
                            flipped = np.flip(patch_5d, axis=axes).copy()
                        else:
                            flipped = patch_5d
                        raw_pred = self._run_model(flipped)
                        # Flip the prediction back to the original orientation.
                        if axes:
                            raw_pred = np.flip(raw_pred, axis=axes).copy()
                        pred_accum += raw_pred.astype(np.float64)

                    pred_accum /= n_mirrors

                    # Apply softmax to convert logits to probabilities.
                    pred_softmax = _softmax(pred_accum[0])  # (C, D, H, W)

                    # Blend into the aggregated output.
                    for c in range(num_classes):
                        aggregated[c, d0:d1, h0:h1, w0:w1] += (
                            pred_softmax[c] * gaussian_kernel
                        )
                    weight_map[d0:d1, h0:h1, w0:w1] += gaussian_kernel

                    patch_count += 1
                    if patch_count % 50 == 0:
                        logger.debug(
                            "Processed %d / %d patches", patch_count, total_patches,
                        )

        # Normalize by accumulated weights.
        weight_map = np.maximum(weight_map, 1e-8)
        for c in range(num_classes):
            aggregated[c] /= weight_map

        return aggregated.astype(np.float32)

    # ------------------------------------------------------------------
    # Core segmentation pipeline
    # ------------------------------------------------------------------

    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        """Segment an MRI volume using ONNX-based nnU-Net inference.

        The pipeline:
        1. Extract voxel spacing from xarray coordinates.
        2. Resample to 1 mm isotropic (cubic interpolation).
        3. Crop to the foreground bounding box (Otsu threshold).
        4. Z-score normalize using foreground statistics.
        5. Pad so each dimension is divisible by the patch size.
        6. Run sliding-window inference with Gaussian blending.
        7. Argmax to produce integer label map.
        8. Unpad, uncrop, resample to original spacing (nearest neighbor).
        9. Map model labels to material indices.

        :param volume: An xarray DataArray containing the MRI volume data.
        :returns: An xarray DataArray with integer labels matching the ordering
            from ``self._material_indices()``.
        """
        data: np.ndarray = volume.to_numpy().astype(np.float32)
        material_idx = self._material_indices()
        water_label = material_idx["water"]

        # Handle NaN values.
        if np.isnan(data).any():
            nan_count = int(np.isnan(data).sum())
            logger.warning(
                "Volume contains %d NaN voxels; replacing with 0 for segmentation.",
                nan_count,
            )
            data = data.copy()
            data[np.isnan(data)] = 0.0

        # Each axis must have at least 2 coordinates for spacing computation.
        for dim in volume.dims:
            if len(volume.coords[dim]) < 2:
                msg = (
                    f"Axis '{dim}' has fewer than 2 coordinates; "
                    "cannot compute voxel spacing for segmentation."
                )
                raise ValueError(msg)

        # Extract voxel spacing (assumes uniform spacing per axis).
        original_spacing = np.array([
            float(np.abs(
                volume.coords[dim].to_numpy()[1] - volume.coords[dim].to_numpy()[0]
            ))
            for dim in volume.dims
        ])
        original_shape = data.shape

        # If the volume has no intensity variation, return all-water.
        if float(data.max() - data.min()) == 0:
            seg = np.full(original_shape, water_label, dtype=int)
            return xa.DataArray(seg, coords=volume.coords, dims=volume.dims)

        # --- Step 1: Resample to 1 mm isotropic ---
        target_spacing = np.array([1.0, 1.0, 1.0])
        zoom_factors = original_spacing / target_spacing
        resampled = scipy_zoom(data, zoom_factors, order=3, mode="nearest")
        resampled_shape = resampled.shape
        logger.debug(
            "Resampled from %s (spacing %s) to %s (1mm iso)",
            original_shape, original_spacing, resampled_shape,
        )

        # --- Step 2: Crop to foreground ---
        foreground_mask = _otsu_foreground_mask(resampled)
        if not foreground_mask.any():
            seg = np.full(original_shape, water_label, dtype=int)
            return xa.DataArray(seg, coords=volume.coords, dims=volume.dims)

        bbox_slices, _bbox_starts = _crop_to_foreground(foreground_mask, margin=8)
        cropped = resampled[bbox_slices]
        cropped_fg = foreground_mask[bbox_slices]
        logger.debug(
            "Cropped to foreground: %s -> %s", resampled_shape, cropped.shape,
        )

        # --- Step 3: Z-score normalize using foreground statistics ---
        fg_values = cropped[cropped_fg]
        fg_mean = float(np.mean(fg_values))
        fg_std = float(np.std(fg_values))
        if fg_std < 1e-8:
            fg_std = 1.0
        normalized = ((cropped - fg_mean) / fg_std).astype(np.float32)

        # --- Step 4: Pad to be divisible by patch size ---
        patch_size = PATCH_SIZES[self.model_type]
        padded, pad_widths = _pad_to_divisible(normalized, patch_size)
        logger.debug(
            "Padded from %s to %s for patch size %s",
            normalized.shape, padded.shape, patch_size,
        )

        # --- Step 5: Sliding window inference ---
        if self.bone_model == "two_class":
            raise ValueError(
                "bone_model='two_class' is not supported for live ONNX inference "
                "because the model outputs 7 classes (single skull). Use "
                "PreSegmented with pre-split two-class label NIfTIs instead."
            )
        if self.model_type == "fullhead":
            label_map = LABEL_MAP_FULLHEAD
        else:
            label_map = LABEL_MAP_SKULL
        num_classes = len(label_map)

        aggregated = self._sliding_window_inference(padded, patch_size, num_classes)

        # --- Step 6: Argmax ---
        predicted_labels = np.argmax(aggregated, axis=0).astype(np.int16)

        # --- Step 7: Unpad ---
        unpadded = _unpad(predicted_labels, pad_widths)

        # --- Step 8: Uncrop (place back into resampled-space volume) ---
        uncropped = np.full(resampled_shape, 0, dtype=np.int16)  # 0 = background/water
        uncropped[bbox_slices] = unpadded

        # For the skull model, assign "tissue" to the brain interior.
        # The skull model only outputs labels 0 (water) and 1 (skull).
        # Anything inside the foreground that is not skull becomes tissue.
        if self.model_type == "skull":
            skull_model_label = 1
            interior_mask = foreground_mask & (uncropped != skull_model_label)
            # Use a virtual label index for tissue (the skull model does not
            # produce one natively). This is mapped to the tissue material
            # index below.
            _SKULL_TISSUE_VIRTUAL_LABEL: np.int16 = np.int16(2)
            uncropped[interior_mask] = _SKULL_TISSUE_VIRTUAL_LABEL

        # --- Step 9: Resample back to original spacing (nearest neighbor) ---
        inverse_zoom = np.array(original_shape) / np.array(resampled_shape)
        final_labels = scipy_zoom(
            uncropped.astype(np.float32), inverse_zoom, order=0, mode="nearest",
        ).astype(np.int16)

        # Ensure shape matches exactly (zoom can be off by 1 voxel).
        final_labels = _match_shape(final_labels, original_shape)

        # --- Step 10: Map model labels to material indices ---
        output = np.full(original_shape, water_label, dtype=int)

        if self.model_type == "fullhead":
            for model_label, mat_key in label_map.items():
                if mat_key in material_idx:
                    output[final_labels == model_label] = material_idx[mat_key]
                else:
                    logger.debug(
                        "Model label %d ('%s') has no matching material; "
                        "defaulting to water.",
                        model_label, mat_key,
                    )
        else:
            # Skull model mapping:
            #   0 = water (already filled)
            #   1 = skull
            #   2 = tissue (virtual label assigned above)
            output[final_labels == 1] = material_idx["skull"]
            output[final_labels == 2] = material_idx["tissue"]

        return xa.DataArray(output, coords=volume.coords, dims=volume.dims)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, excluding the cached ONNX session."""
        d = super().to_dict()
        d.pop("_session", None)
        return d

    def to_table(self) -> pd.DataFrame:
        """Get a table of the segmentation method parameters.

        :returns: Pandas DataFrame of the segmentation method parameters.
        """
        records = [
            {"Name": "Type", "Value": "nnU-Net Segmentation", "Unit": ""},
            {"Name": "Model Path", "Value": self.model_path or "(auto)", "Unit": ""},
            {"Name": "Model Type", "Value": self.model_type, "Unit": ""},
            {"Name": "Bone Model", "Value": self.bone_model, "Unit": ""},
            {"Name": "Use GPU", "Value": self.use_gpu, "Unit": ""},
            {"Name": "Use Mirroring (TTA)", "Value": self.use_mirroring, "Unit": ""},
            {"Name": "Reference Material", "Value": self.ref_material, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)


# ======================================================================
# Module-level utility functions
# ======================================================================


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis 0 (class dimension).

    :param logits: Array of shape ``(C, D, H, W)``.
    :returns: Softmax probabilities with the same shape.
    """
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=0, keepdims=True)


def _otsu_foreground_mask(data: np.ndarray) -> np.ndarray:
    """Compute a binary foreground mask using Otsu thresholding.

    :param data: 3-D intensity array.
    :returns: Boolean mask where True indicates foreground.
    """
    import skimage.filters

    nonzero_vals = data[data > 0]
    if nonzero_vals.size == 0:
        return np.zeros(data.shape, dtype=bool)
    try:
        threshold = skimage.filters.threshold_otsu(nonzero_vals)
    except ValueError:
        return np.zeros(data.shape, dtype=bool)
    return data > threshold


def _crop_to_foreground(
    mask: np.ndarray,
    margin: int = 0,
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    """Compute bounding box slices for the foreground region of a binary mask.

    :param mask: Boolean 3-D mask.
    :param margin: Number of voxels to pad around the bounding box.
    :returns: Tuple of (slices, start_indices) where slices can index the
        original array and start_indices records the offset for uncropping.
    """
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    slices = []
    starts = []
    for i in range(3):
        lo = max(0, int(mins[i]) - margin)
        hi = min(mask.shape[i], int(maxs[i]) + 1 + margin)
        slices.append(slice(lo, hi))
        starts.append(lo)

    return tuple(slices), tuple(starts)


def _pad_to_divisible(
    data: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Pad a 3-D array so each dimension is at least the patch size and divisible by it.

    Padding is applied symmetrically (split between before and after).
    Uses constant padding with value 0 (the z-score normalized background).

    :param data: 3-D array to pad.
    :param patch_size: Target divisibility for each dimension.
    :returns: Tuple of (padded_array, pad_widths) where pad_widths is a list
        of (before, after) tuples per dimension.
    """
    pad_widths: list[tuple[int, int]] = []
    for i in range(3):
        current = data.shape[i]
        target = patch_size[i]
        if current < target:
            # Volume smaller than patch: pad to exactly the patch size.
            total_pad = target - current
        else:
            remainder = current % target
            total_pad = (target - remainder) if remainder != 0 else 0
        before = total_pad // 2
        after = total_pad - before
        pad_widths.append((before, after))

    padded = np.pad(data, pad_widths, mode="constant", constant_values=0)
    return padded, pad_widths


def _unpad(data: np.ndarray, pad_widths: list[tuple[int, int]]) -> np.ndarray:
    """Remove padding applied by ``_pad_to_divisible``.

    :param data: Padded 3-D array.
    :param pad_widths: List of (before, after) padding per dimension.
    :returns: Unpadded array.
    """
    slices = []
    for i, (before, after) in enumerate(pad_widths):
        end = data.shape[i] - after if after > 0 else data.shape[i]
        slices.append(slice(before, end))
    return data[tuple(slices)]


def _match_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Crop or pad an array to exactly match ``target_shape``.

    Handles the +/- 1 voxel discrepancy that can arise from floating-point
    zoom factor computation.

    :param arr: Input array.
    :param target_shape: Desired output shape.
    :returns: Array with shape ``target_shape``.
    """
    slices = []
    pad_widths = []
    for i in range(len(target_shape)):
        if arr.shape[i] > target_shape[i]:
            slices.append(slice(0, target_shape[i]))
            pad_widths.append((0, 0))
        elif arr.shape[i] < target_shape[i]:
            slices.append(slice(None))
            pad_widths.append((0, target_shape[i] - arr.shape[i]))
        else:
            slices.append(slice(None))
            pad_widths.append((0, 0))

    result = arr[tuple(slices)]
    if any(p != (0, 0) for p in pad_widths):
        result = np.pad(result, pad_widths, mode="edge")
    return result
