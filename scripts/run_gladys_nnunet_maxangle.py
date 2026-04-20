#!/usr/bin/env python3
"""GLADYS Simulation with nnU-Net segmentation + MaxAngle apodization.

Forked from run_gladys_nnunet.py to test whether down-weighting
normal-incidence elements reduces the near-aperture impedance peak in
skull sims. Only difference is the apodization weighting.

Outputs:
  - pmax NIfTIs: ~/Data/openlifu-validation/results/gladys_nnunet_maxangle45_{corrected,geometric,water}_pmax.nii.gz
  - analysis PNG: ~/Data/openlifu-validation/results/gladys_nnunet_maxangle45_pmax_analysis_2026-04-18.png
"""

import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Workaround for a bug in kwave v3's helper files that call
# `logging.log(level, "msg", arg, ...)` with positional args instead of
# using %-formatting. Python's logging then raises
# `TypeError: not all arguments converted during string formatting`
# The sim itself still runs, but we flood the log. Patch `logging.log`
# to concatenate extra positional args into the message string.
# ---------------------------------------------------------------------------
_orig_log = logging.log

def _patched_log(level, msg, *args, **kwargs):
    try:
        if args and isinstance(msg, str) and "%" not in msg:
            msg = msg + " " + " ".join(str(a) for a in args)
            args = ()
    except Exception:
        pass
    return _orig_log(level, msg, *args, **kwargs)

logging.log = _patched_log
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import xarray as xa
from scipy.ndimage import map_coordinates

# Ensure local openlifu is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.bf.apod_methods.maxangle import MaxAngle
from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.seg_methods.nnunet_seg import LABEL_MAP_FULLHEAD
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER
from openlifu.sim.kwave_if import run_simulation
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MRI_PATH = Path.home() / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)
LABEL_NIFTI_PATH = Path.home() / "Data/openlifu-validation/results/GU008_nnunet_labels.nii.gz"

N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
GRID_SPACING_MM = 0.5
CFL = 0.1
CYCLES = 5
AMPLITUDE = 1.0
C0 = 1500.0
T_END_SAFETY = 2.0
GRID_MARGIN_MM = 10.0

# MaxAngle apodization config
MAX_ANGLE_DEG = 45.0
MAX_ANGLE_FALLBACK_DEG = 70.0
APERTURE_BAND_MM = 10.0  # band around aperture center for focal-gain denominator


# ---------------------------------------------------------------------------
# PreSegmented: wrap a pre-computed label NIfTI as a SegmentationMethod
# ---------------------------------------------------------------------------
def _default_fullhead_materials() -> dict[str, Material]:
    m = MATERIALS.copy()
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


@dataclass
class PreSegmented(SegmentationMethod):
    """A SegmentationMethod that returns labels from a pre-computed NIfTI."""

    label_nifti_path: str = ""
    nnunet_label_map: dict[int, str] = field(default_factory=lambda: dict(LABEL_MAP_FULLHEAD))
    materials: dict[str, Material] = field(default_factory=_default_fullhead_materials)

    def __post_init__(self):
        super().__post_init__()
        if not self.label_nifti_path:
            raise ValueError("label_nifti_path is required")
        img = nib.load(self.label_nifti_path)
        data = np.asarray(img.dataobj).astype(np.int16)
        affine = img.affine
        dim_names = ("x", "y", "z")
        coords = {}
        for axis, dim in enumerate(dim_names):
            origin = float(affine[axis, 3])
            spacing = float(affine[axis, axis])
            coord_values = origin + np.arange(data.shape[axis]) * spacing
            coords[dim] = xa.Variable(dim, coord_values, attrs={"units": "mm"})
        self._labels = xa.DataArray(data, dims=dim_names, coords=coords)

        mat_idx = self._material_indices()
        for label_int, mat_key in self.nnunet_label_map.items():
            if mat_key not in mat_idx:
                raise ValueError(
                    f"nnunet_label_map references material '{mat_key}' "
                    f"which is not in self.materials (for label {label_int})."
                )

    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        mat_idx = self._material_indices()
        src = self._labels
        src_dims = list(src.dims)

        src_origin = {d: float(src.coords[d].to_numpy()[0]) for d in src_dims}
        src_spacing = {}
        for d in src_dims:
            cs = src.coords[d].to_numpy()
            src_spacing[d] = float(cs[1] - cs[0]) if len(cs) > 1 else 1.0

        tgt_dims = list(volume.dims)
        assert set(tgt_dims) == set(src_dims), (
            f"Dim mismatch: src={src_dims}, tgt={tgt_dims}"
        )

        tgt_coord_arrays = [volume.coords[d].to_numpy() for d in src_dims]
        mg = np.meshgrid(*tgt_coord_arrays, indexing="ij")
        frac_idx = []
        for i, d in enumerate(src_dims):
            fi = (mg[i] - src_origin[d]) / src_spacing[d]
            frac_idx.append(fi)
        frac_stack = np.stack(frac_idx, axis=0)

        resampled = map_coordinates(
            src.to_numpy().astype(np.float32),
            frac_stack,
            order=0,
            mode="constant",
            cval=0.0,
        ).astype(np.int16)

        water_idx = mat_idx["water"]
        output = np.full(resampled.shape, water_idx, dtype=int)
        for nn_label, mat_key in self.nnunet_label_map.items():
            output[resampled == nn_label] = mat_idx[mat_key]

        labels_da = xa.DataArray(
            output,
            dims=src_dims,
            coords={d: volume.coords[d] for d in src_dims},
        )
        return labels_da.transpose(*tgt_dims)

    def to_table(self) -> pd.DataFrame:
        records = [
            {"Name": "Type", "Value": "PreSegmented (nnU-Net label NIfTI)", "Unit": ""},
            {"Name": "Label NIfTI", "Value": self.label_nifti_path, "Unit": ""},
            {"Name": "Reference Material", "Value": self.ref_material, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Hemispherical array generator (verbatim from run_gladys_simulation.py)
# ---------------------------------------------------------------------------
def create_hemispherical_array(
    n_elements: int = 64,
    radius_mm: float = 90.0,
    aperture_mm: float = 80.0,
    freq_hz: float = 500e3,
    element_size_mm: float = 5.0,
) -> Transducer:
    half_aperture = aperture_mm / 2.0
    if half_aperture > radius_mm:
        raise ValueError(
            f"Aperture radius ({half_aperture}mm) exceeds sphere radius ({radius_mm}mm)"
        )
    theta_max = np.arcsin(half_aperture / radius_mm)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    elements = []
    for i in range(n_elements):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        x = radius_mm * np.sin(theta) * np.cos(phi)
        y = radius_mm * np.sin(theta) * np.sin(phi)
        z = radius_mm * np.cos(theta)
        nx, ny, nz = -x, -y, -z
        az = np.arctan2(nx, nz)
        el = -np.arctan2(ny, np.sqrt(nx**2 + nz**2))
        elements.append(Element(
            index=i + 1, pin=i + 1,
            position=np.array([x, y, z]),
            orientation=np.array([az, el, 0.0]),
            size=np.array([element_size_mm, element_size_mm]),
            units="mm",
        ))
    return Transducer(
        id="hemi64",
        name=f"Hemispherical {n_elements}-element array",
        elements=elements, frequency=freq_hz, units="mm",
    )


def load_nifti_as_xarray(nifti_path: Path) -> xa.DataArray:
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    dim_names = ("x", "y", "z")
    coords = {}
    for axis, dim in enumerate(dim_names):
        origin = float(affine[axis, 3])
        spacing = float(affine[axis, axis])
        coord_values = origin + np.arange(data.shape[axis]) * spacing
        coords[dim] = xa.Variable(dim, coord_values, attrs={"units": "mm"})
    return xa.DataArray(data, dims=dim_names, coords=coords)


def extract_focal_stats(
    result: xa.Dataset,
    target_mm: np.ndarray,
    label: str,
    element_positions_mm: np.ndarray | None = None,
    exclusion_radius_mm: float = 3.0,
) -> dict:
    p_max = result["p_max"].to_numpy()
    dims = list(result["p_max"].dims)
    coord_arrays = {d: result.coords[d].to_numpy() for d in dims}

    raw_idx = np.unravel_index(p_max.argmax(), p_max.shape)
    raw_focal_mm = np.array([
        float(coord_arrays[d][raw_idx[i]]) for i, d in enumerate(dims)
    ])
    raw_error = float(np.linalg.norm(raw_focal_mm - target_mm))
    raw_max_p = float(p_max.max())

    target_idx = tuple(
        int(np.argmin(np.abs(coord_arrays[dims[ax]] - target_mm[ax])))
        for ax in range(3)
    )
    p_at_target = float(p_max[target_idx])

    masked_focal_mm = None
    masked_error = None
    masked_max_p = None
    if element_positions_mm is not None and exclusion_radius_mm > 0:
        mg = np.meshgrid(*[coord_arrays[d] for d in dims], indexing="ij")
        coord_stack = np.stack([m for m in mg], axis=-1)
        mask = np.ones(p_max.shape, dtype=bool)
        for pos in element_positions_mm:
            d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
            mask &= d2 > exclusion_radius_mm ** 2
        if mask.any():
            masked_p = np.where(mask, p_max, -np.inf)
            m_idx = np.unravel_index(masked_p.argmax(), masked_p.shape)
            masked_focal_mm = np.array([
                float(coord_arrays[d][m_idx[i]]) for i, d in enumerate(dims)
            ])
            masked_error = float(np.linalg.norm(masked_focal_mm - target_mm))
            masked_max_p = float(p_max[m_idx])

    logger.info(
        "%s: raw max_p=%.4g Pa @ (%s) mm, raw_err=%.2f mm; p@target=%.4g Pa",
        label, raw_max_p,
        ", ".join(f"{v:.1f}" for v in raw_focal_mm),
        raw_error, p_at_target,
    )
    if masked_focal_mm is not None:
        logger.info(
            "  masked max_p=%.4g Pa @ (%s) mm, masked_err=%.2f mm "
            "(excl radius %.1f mm around %d sources)",
            masked_max_p,
            ", ".join(f"{v:.1f}" for v in masked_focal_mm),
            masked_error, exclusion_radius_mm, len(element_positions_mm),
        )

    return {
        "label": label,
        "max_pressure": raw_max_p,
        "focal_mm": raw_focal_mm,
        "focal_error": raw_error,
        "p_at_target": p_at_target,
        "masked_max_pressure": masked_max_p,
        "masked_focal_mm": masked_focal_mm,
        "masked_focal_error": masked_error,
    }


def extract_masked_argmax(
    result: xa.Dataset,
    target_mm: np.ndarray,
    element_positions_mm: np.ndarray,
    exclusion_radius_mm: float = 20.0,
) -> dict:
    p_max = result["p_max"].to_numpy()
    dims = list(result["p_max"].dims)
    coord_arrays = {d: result.coords[d].to_numpy() for d in dims}
    mg = np.meshgrid(*[coord_arrays[d] for d in dims], indexing="ij")
    coord_stack = np.stack([m for m in mg], axis=-1)
    mask = np.ones(p_max.shape, dtype=bool)
    for pos in element_positions_mm:
        d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
        mask &= d2 > exclusion_radius_mm ** 2
    if not mask.any():
        return {"max_p": None, "focal_mm": None, "error": None}
    masked_p = np.where(mask, p_max, -np.inf)
    m_idx = np.unravel_index(masked_p.argmax(), masked_p.shape)
    focal_mm = np.array([float(coord_arrays[d][m_idx[i]]) for i, d in enumerate(dims)])
    return {
        "max_p": float(p_max[m_idx]),
        "focal_mm": focal_mm,
        "error": float(np.linalg.norm(focal_mm - target_mm)),
    }


def compute_aperture_band_mean(
    result: xa.Dataset,
    element_positions_mm: np.ndarray,
    band_mm: float,
) -> float:
    """Mean pmax inside a `band_mm` spherical shell-of-influence of any element."""
    p_max = result["p_max"].to_numpy()
    dims = list(result["p_max"].dims)
    coord_arrays = {d: result.coords[d].to_numpy() for d in dims}
    mg = np.meshgrid(*[coord_arrays[d] for d in dims], indexing="ij")
    coord_stack = np.stack([m for m in mg], axis=-1)
    mask = np.zeros(p_max.shape, dtype=bool)
    band2 = band_mm ** 2
    for pos in element_positions_mm:
        d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
        mask |= d2 <= band2
    if not mask.any():
        return float("nan")
    return float(p_max[mask].mean())


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_total = time.time()

    print("=" * 72)
    print("GLADYS Simulation: Transcranial FUS (nnU-Net + MaxAngle apodization)")
    print("=" * 72)

    # -------------------------------------------------------------------
    # 1. Load MRI + load pre-computed nnU-Net labels
    # -------------------------------------------------------------------
    print(f"\n[1] Loading MRI: {MRI_PATH}")
    if not MRI_PATH.exists():
        print(f"ERROR: MRI file not found at {MRI_PATH}")
        sys.exit(1)
    if not LABEL_NIFTI_PATH.exists():
        print(f"ERROR: nnU-Net labels not found at {LABEL_NIFTI_PATH}")
        sys.exit(1)

    volume = load_nifti_as_xarray(MRI_PATH)
    print(f"    MRI shape: {volume.shape}, dims={volume.dims}")
    for dim in volume.dims:
        c = volume.coords[dim].to_numpy()
        print(f"    {dim}: [{c[0]:.1f}, {c[-1]:.1f}] mm (spacing={abs(c[1]-c[0]):.2f} mm, N={len(c)})")

    print(f"\n    Loading nnU-Net labels: {LABEL_NIFTI_PATH}")
    seg_method = PreSegmented(label_nifti_path=str(LABEL_NIFTI_PATH))
    print(f"    Label shape: {seg_method._labels.shape}")
    lab_arr = seg_method._labels.to_numpy()
    print(f"    nnU-Net label histogram:")
    for nn_label, mat_key in sorted(seg_method.nnunet_label_map.items()):
        n = int((lab_arr == nn_label).sum())
        pct = 100.0 * n / lab_arr.size
        print(f"      {nn_label} ({mat_key}): {n:,d} voxels ({pct:.2f}%)")

    # -------------------------------------------------------------------
    # 2. Segment MRI volume (for target/approach determination)
    # -------------------------------------------------------------------
    print("\n[2] Computing segmentation on MRI grid (for target geometry)...")
    t0 = time.time()
    seg_labels = seg_method._segment(volume)
    print(f"    Segmentation (resample + remap) complete in {time.time()-t0:.1f}s")
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()
    for name, idx in material_idx.items():
        count = int(np.sum(seg_arr == idx))
        pct = 100.0 * count / seg_arr.size
        print(f"    {name:>14s} (idx={idx}): {count:>10,d} voxels ({pct:.2f}%)")

    brain_keys = [k for k in ("csf", "gray_matter", "white_matter") if k in material_idx]
    brain_mask = np.zeros(seg_arr.shape, dtype=bool)
    for k in brain_keys:
        brain_mask |= (seg_arr == material_idx[k])
    if brain_mask.sum() == 0:
        print("    No brain voxels found; falling back to 'tissue' centroid.")
        brain_mask = seg_arr == material_idx["tissue"]
    print(f"    Brain mask: {int(brain_mask.sum()):,d} voxels ({100.0*brain_mask.sum()/seg_arr.size:.2f}%)")

    dim_names = list(volume.dims)
    coord_arrays = {d: volume.coords[d].to_numpy() for d in dim_names}
    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"    Brain center (target): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])
    max_skull_dist_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax])
        for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_dist_per_axis))
    approach_dim = dim_names[approach_axis]
    skull_top_val = float(skull_mm[approach_axis].max())
    print(f"    Approach axis: {approach_dim} (axis {approach_axis})")
    print(f"    Skull top along {approach_dim}: {skull_top_val:.1f} mm")

    xdc_center_mm = target_mm.copy()
    xdc_center_mm[approach_axis] = target_mm[approach_axis] + RADIUS_MM
    print(f"    Transducer center: ({xdc_center_mm[0]:.1f}, {xdc_center_mm[1]:.1f}, {xdc_center_mm[2]:.1f}) mm")

    # -------------------------------------------------------------------
    # 3. Array + bake world positions
    # -------------------------------------------------------------------
    print(f"\n[3] Creating {N_ELEMENTS}-element hemispherical array (R={RADIUS_MM}mm)")
    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )
    transform = np.eye(4)
    transform[:3, 3] = target_mm
    if approach_axis == 0:
        angle = np.pi / 2
        transform[:3, :3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
    elif approach_axis == 1:
        angle = -np.pi / 2
        transform[:3, :3] = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
    transform[:3, 3] = target_mm

    arr = deepcopy(arr_local)
    for el in arr.elements:
        world_pos = el.get_position(units="mm", matrix=transform)
        el.position = world_pos
        direction = target_mm - world_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            n = direction / dist
            az = np.arctan2(n[0], n[2])
            el_angle = -np.arctan2(n[1], np.sqrt(n[0]**2 + n[2]**2))
            el.orientation = np.array([az, el_angle, 0.0])

    positions = arr.get_positions(units="mm")
    for ax, dim in enumerate(["x", "y", "z"]):
        print(f"      {dim}: [{positions[:, ax].min():.1f}, {positions[:, ax].max():.1f}] mm")
    dists_to_target = np.linalg.norm(positions - target_mm[np.newaxis, :], axis=1)
    print(f"    Element-to-target distance: mean={dists_to_target.mean():.1f}, "
          f"std={dists_to_target.std():.2f}, range=[{dists_to_target.min():.1f}, {dists_to_target.max():.1f}] mm")

    # -------------------------------------------------------------------
    # 4. Build sim grid
    # -------------------------------------------------------------------
    print(f"\n[4] Building simulation grid (spacing={GRID_SPACING_MM}mm)...")
    all_points = np.vstack([positions, target_mm[np.newaxis, :]])
    grid_min = all_points.min(axis=0) - GRID_MARGIN_MM
    grid_max = all_points.max(axis=0) + GRID_MARGIN_MM
    grid_min = np.floor(grid_min / GRID_SPACING_MM) * GRID_SPACING_MM
    grid_max = np.ceil(grid_max / GRID_SPACING_MM) * GRID_SPACING_MM

    sim_coords = {}
    for ax, dim in enumerate(["x", "y", "z"]):
        n_pts = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        sim_coords[dim] = xa.Variable(
            dim, np.linspace(grid_min[ax], grid_max[ax], n_pts),
            attrs={"units": "mm"},
        )
    grid_shape = tuple(len(sim_coords[d]) for d in ["x", "y", "z"])
    print(f"    Grid shape: {grid_shape} ({int(np.prod(grid_shape)):,d} voxels)")

    from scipy.interpolate import RegularGridInterpolator
    orig_coords = [volume.coords[d].to_numpy() for d in volume.dims]
    interp = RegularGridInterpolator(
        orig_coords, volume.to_numpy(),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    sim_coord_arrays = [sim_coords[d].data for d in ["x", "y", "z"]]
    mg = np.meshgrid(*sim_coord_arrays, indexing="ij")
    query_pts = np.stack([m.ravel() for m in mg], axis=-1)
    resampled_data = interp(query_pts).reshape(grid_shape).astype(np.float32)
    sim_volume = xa.DataArray(
        resampled_data, dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )

    print("    Segmenting sim grid via PreSegmented (resample nnU-Net labels)...")
    t0 = time.time()
    sim_params = seg_method.seg_params(sim_volume)
    print(f"    seg_params complete in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------
    # 5. Replace air voxels with water
    # -------------------------------------------------------------------
    print("\n[5] Replacing air voxels with water in medium...")
    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask = sim_seg_arr == material_idx["air"]
    skull_mask_sim = sim_seg_arr == material_idx["skull"]
    water_mat = seg_method.materials["water"]
    n_air = int(air_mask.sum())
    n_skull = int(skull_mask_sim.sum())
    total_vox = sim_seg_arr.size
    pct_skull = 100.0 * n_skull / total_vox
    print(f"    Air voxels: {n_air:,d} ({100.0*n_air/total_vox:.2f}%)")
    print(f"    SKULL voxels (bone fraction): {n_skull:,d} ({pct_skull:.2f}%)")

    if n_air > 0:
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation
        print("    Replaced air with water.")

    print("    Sim-grid material histogram:")
    for name, idx in material_idx.items():
        count = int((sim_seg_arr == idx).sum())
        pct = 100.0 * count / total_vox
        print(f"      {name:>14s}: {count:>10,d} voxels ({pct:.2f}%)")

    # -------------------------------------------------------------------
    # 5b. Skull path length along target-to-aperture axis (diagnostic)
    # -------------------------------------------------------------------
    print("\n[5b] Measuring skull path length along target->aperture-center axis...")
    aperture_center_mm = positions.mean(axis=0)
    ray_dir = aperture_center_mm - target_mm
    ray_len_mm = float(np.linalg.norm(ray_dir))
    skull_path_near_mm = float("nan")
    skull_path_far_mm = float("nan")
    if ray_len_mm > 1e-6:
        ray_unit = ray_dir / ray_len_mm
        probe_len_mm = 1.3 * RADIUS_MM
        step_mm = GRID_SPACING_MM / 2.0
        sim_origins = np.array([sim_coord_arrays[i][0] for i in range(3)])
        sim_specs = np.array([sim_coord_arrays[i][1] - sim_coord_arrays[i][0] for i in range(3)])
        def _skull_samples_along(direction):
            n_samples = int(np.ceil(probe_len_mm / step_mm)) + 1
            ts = np.linspace(0.0, probe_len_mm, n_samples)
            pts = target_mm[None, :] + ts[:, None] * direction[None, :]
            frac = ((pts - sim_origins[None, :]) / sim_specs[None, :]).T
            sampled = map_coordinates(
                sim_seg_arr.astype(np.float32), frac, order=0,
                mode="constant", cval=-1.0,
            ).astype(np.int16)
            is_skull = sampled == material_idx["skull"]
            n_skull = int(is_skull.sum())
            return n_skull * step_mm, n_samples, ts, is_skull
        skull_path_near_mm, n_near, ts_near, mask_near = _skull_samples_along(ray_unit)
        skull_path_far_mm, n_far, ts_far, mask_far = _skull_samples_along(-ray_unit)
        print(f"    NEAR-side skull: {skull_path_near_mm:.1f} mm, FAR-side skull: {skull_path_far_mm:.1f} mm")
    skull_path_mm = skull_path_near_mm

    # -------------------------------------------------------------------
    # 6. Target, delays, t_end, sim kwargs, APOD (MaxAngle)
    # -------------------------------------------------------------------
    target = Point(position=target_mm.copy(), id="brain_center", name="Brain Center Target", units="mm")
    print("\n[6] Computing geometric delays...")
    direct = Direct(c0=C0)
    delays_geo = direct.calc_delays(arr, target, sim_params)
    print(f"    Delay range: {delays_geo.min()*1e6:.1f} to {delays_geo.max()*1e6:.1f} us")

    print("\n[7] Calculating t_end...")
    from openlifu.sim.sim_setup import SimSetup
    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=CFL,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    max_dist_m = max_dist_mm * 1e-3
    t_end = max_dist_m / C0 * T_END_SAFETY
    c_max = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max
    n_timesteps = int(np.ceil(t_end / dt))
    print(f"    Max element->corner: {max_dist_mm:.1f} mm | t_end={t_end*1e6:.1f} us | dt={dt*1e9:.2f} ns | N={n_timesteps:,d}")

    print("\n[8] SimulationCorrected (phase correction)...")
    sim_corrected = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
    t0 = time.time()
    delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
    print(f"    Phase correction done in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------
    # APOD: MaxAngle (PRIMARY CHANGE vs run_gladys_nnunet.py)
    # -------------------------------------------------------------------
    print("\n[APOD] MaxAngle apodization")
    # Since we already baked element positions/orientations into world frame
    # above, pass transform=None.
    apod_method = MaxAngle(max_angle=MAX_ANGLE_DEG, units="deg")
    apod = apod_method.calc_apodization(arr, target, sim_params, transform=None)
    # Diagnostic: compute the angles themselves so we can see the distribution.
    angles_deg = np.array([
        el.angle_to_point(target.get_position(units="m"), units="m",
                          matrix=np.eye(4), return_as="deg")
        for el in arr.elements
    ])
    print(f"    Element-to-target angle distribution (deg):")
    print(f"      min={angles_deg.min():.2f}, max={angles_deg.max():.2f}, "
          f"mean={angles_deg.mean():.2f}, median={np.median(angles_deg):.2f}")
    print(f"    max_angle={MAX_ANGLE_DEG} deg -> {int((apod > 0).sum())} / {arr.numelements()} active elements")

    if int((apod > 0).sum()) == 0:
        print(f"    WARNING: ALL elements zeroed at max_angle={MAX_ANGLE_DEG} deg."
              f" Falling back to max_angle={MAX_ANGLE_FALLBACK_DEG} deg.")
        apod_method = MaxAngle(max_angle=MAX_ANGLE_FALLBACK_DEG, units="deg")
        apod = apod_method.calc_apodization(arr, target, sim_params, transform=None)
        print(f"    Fallback: {int((apod > 0).sum())} / {arr.numelements()} active elements")

    active_elements = int((apod > 0).sum())
    total_elements = int(arr.numelements())

    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, source_method="point_source",
    )

    # -------------------------------------------------------------------
    # 9-11. Sims A, B, C
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[SIM A] Phase-corrected + heterogeneous skull (nnU-Net) + MaxAngle")
    print("=" * 72)
    t0 = time.time()
    result_a = run_simulation(params=sim_params, delays=delays_corrected, ref_values_only=False, **common_kwargs)
    print(f"    Completed in {time.time()-t0:.1f}s")
    stats_a = extract_focal_stats(result_a, target_mm, "SIM A (corrected+hetero+MaxAngle)", element_positions_mm=positions)
    masked_a_20 = extract_masked_argmax(result_a, target_mm, positions, 20.0)
    aperture_mean_a = compute_aperture_band_mean(result_a, positions, APERTURE_BAND_MM)

    print("\n" + "=" * 72)
    print("[SIM B] Geometric + heterogeneous skull (nnU-Net) + MaxAngle")
    print("=" * 72)
    t0 = time.time()
    result_b = run_simulation(params=sim_params, delays=delays_geo, ref_values_only=False, **common_kwargs)
    print(f"    Completed in {time.time()-t0:.1f}s")
    stats_b = extract_focal_stats(result_b, target_mm, "SIM B (geometric+hetero+MaxAngle)", element_positions_mm=positions)
    masked_b_20 = extract_masked_argmax(result_b, target_mm, positions, 20.0)
    aperture_mean_b = compute_aperture_band_mean(result_b, positions, APERTURE_BAND_MM)

    print("\n" + "=" * 72)
    print("[SIM C] Geometric + homogeneous water (ref_values_only) + MaxAngle")
    print("=" * 72)
    t0 = time.time()
    result_c = run_simulation(params=sim_params, delays=delays_geo, ref_values_only=True, **common_kwargs)
    print(f"    Completed in {time.time()-t0:.1f}s")
    stats_c = extract_focal_stats(result_c, target_mm, "SIM C (geometric+water+MaxAngle)", element_positions_mm=positions)
    masked_c_20 = extract_masked_argmax(result_c, target_mm, positions, 20.0)
    aperture_mean_c = compute_aperture_band_mean(result_c, positions, APERTURE_BAND_MM)

    # -------------------------------------------------------------------
    # 12. Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY (nnU-Net + MaxAngle apodization)")
    print("=" * 72)
    print(f"Apodization              : MaxAngle (max_angle={MAX_ANGLE_DEG} deg -> "
          f"fallback possibly applied)")
    print(f"Active elements          : {active_elements} / {total_elements}")
    print(f"Element-target angles    : min={angles_deg.min():.2f}, max={angles_deg.max():.2f}, "
          f"mean={angles_deg.mean():.2f} deg")
    print(f"Bone fraction in sim grid: {pct_skull:.2f}%")
    print(f"Skull path NEAR side     : {skull_path_near_mm:.1f} mm")
    print(f"Skull path FAR  side     : {skull_path_far_mm:.1f} mm")
    print(f"Aperture-band radius     : {APERTURE_BAND_MM} mm")
    print()
    header = (
        f"{'SIM':<28s} {'p@target':>10s} {'ap_mean':>10s} {'focal_gain':>11s} "
        f"{'raw_err':>9s} {'mask20_err':>11s}"
    )
    print(header)
    print("-" * len(header))

    def _fg(p_at_t, ap_mean):
        if ap_mean is None or ap_mean <= 0 or not np.isfinite(ap_mean):
            return float("nan")
        return p_at_t / ap_mean

    focal_gain_a = _fg(stats_a["p_at_target"], aperture_mean_a)
    focal_gain_b = _fg(stats_b["p_at_target"], aperture_mean_b)
    focal_gain_c = _fg(stats_c["p_at_target"], aperture_mean_c)

    for label, stats, masked20, ap_mean, fg in [
        ("A: Corrected + skull", stats_a, masked_a_20, aperture_mean_a, focal_gain_a),
        ("B: Geometric + skull", stats_b, masked_b_20, aperture_mean_b, focal_gain_b),
        ("C: Geometric + water", stats_c, masked_c_20, aperture_mean_c, focal_gain_c),
    ]:
        m20_err = masked20["error"] if masked20["error"] is not None else float("nan")
        print(
            f"{label:<28s} {stats['p_at_target']:>10.4g} {ap_mean:>10.4g} "
            f"{fg:>11.4g} {stats['focal_error']:>9.2f} {m20_err:>11.2f}"
        )
    print()

    # -------------------------------------------------------------------
    # 13. Save pmax NIfTIs
    # -------------------------------------------------------------------
    results_dir = Path.home() / "Data/openlifu-validation/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for sim_label, result in [("corrected", result_a), ("geometric", result_b), ("water", result_c)]:
        p_max_data = result["p_max"].to_numpy()
        x_coords = result.coords["x"].to_numpy()
        y_coords = result.coords["y"].to_numpy()
        z_coords = result.coords["z"].to_numpy()
        out_affine = np.diag([
            float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0,
            float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0,
            float(z_coords[1] - z_coords[0]) if len(z_coords) > 1 else 1.0,
            1.0,
        ])
        out_affine[0, 3] = float(x_coords[0])
        out_affine[1, 3] = float(y_coords[0])
        out_affine[2, 3] = float(z_coords[0])
        out_path = results_dir / f"gladys_nnunet_maxangle45_{sim_label}_pmax.nii.gz"
        nib.save(nib.Nifti1Image(p_max_data.astype(np.float32), out_affine), str(out_path))
        print(f"    Saved: {out_path}")

    # -------------------------------------------------------------------
    # 14. Plot: 2D slice + axial profile per sim
    # -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = results_dir / "gladys_nnunet_maxangle45_pmax_analysis_2026-04-18.png"

    sim_x = np.asarray(sim_coord_arrays[0])
    sim_y = np.asarray(sim_coord_arrays[1])
    sim_z = np.asarray(sim_coord_arrays[2])
    axes_coords = [sim_x, sim_y, sim_z]
    target_idx_sim = tuple(
        int(np.argmin(np.abs(axes_coords[ax] - target_mm[ax])))
        for ax in range(3)
    )
    non_approach = [ax for ax in range(3) if ax != approach_axis]
    slice_axis = non_approach[0]

    def slice_2d(vol):
        other_axes = [ax for ax in range(3) if ax != slice_axis]
        idx = target_idx_sim[slice_axis]
        sl = [slice(None), slice(None), slice(None)]
        sl[slice_axis] = idx
        return vol[tuple(sl)], tuple("xyz"[ax] for ax in other_axes)

    def axial_profile(vol):
        idx_list = list(target_idx_sim)
        idx_list[approach_axis] = slice(None)
        profile = vol[tuple(idx_list)]
        coord = axes_coords[approach_axis]
        return coord, np.asarray(profile)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    sims = [("A: corrected+skull", result_a), ("B: geometric+skull", result_b), ("C: geometric+water", result_c)]
    all_pmax = [np.log10(np.maximum(r["p_max"].to_numpy(), 1e-10)) for _, r in sims]
    vmin = min(a.min() for a in all_pmax)
    vmax_global = max(a.max() for a in all_pmax)

    for i, (title, result) in enumerate(sims):
        pmax = result["p_max"].to_numpy()
        sl2d, (u_label, v_label) = slice_2d(pmax)
        log_sl = np.log10(np.maximum(sl2d, 1e-10))
        ax_top = axes[0, i]
        im = ax_top.imshow(
            log_sl.T, origin="lower",
            extent=[axes_coords[[ax for ax in range(3) if ax != slice_axis][0]][0],
                    axes_coords[[ax for ax in range(3) if ax != slice_axis][0]][-1],
                    axes_coords[[ax for ax in range(3) if ax != slice_axis][1]][0],
                    axes_coords[[ax for ax in range(3) if ax != slice_axis][1]][-1]],
            vmin=vmin, vmax=vmax_global, cmap="inferno", aspect="equal",
        )
        other_axes = [ax for ax in range(3) if ax != slice_axis]
        ax_top.plot(target_mm[other_axes[0]], target_mm[other_axes[1]], "wx", ms=10, mew=2)
        ax_top.set_title(f"{title}\nlog10(p_max) slice @{'xyz'[slice_axis]}={target_mm[slice_axis]:.1f}mm")
        ax_top.set_xlabel(f"{u_label} (mm)")
        ax_top.set_ylabel(f"{v_label} (mm)")
        plt.colorbar(im, ax=ax_top, label="log10(Pa)")

        coord, profile = axial_profile(pmax)
        ax_bot = axes[1, i]
        ax_bot.semilogy(coord, np.maximum(profile, 1e-10), "b-", lw=1.5)
        ax_bot.axvline(target_mm[approach_axis], color="r", ls="--", alpha=0.7, label="target")
        ax_bot.axvline(positions[:, approach_axis].mean(), color="g", ls=":", alpha=0.7, label="aperture ctr")
        ax_bot.set_xlabel(f"{'xyz'[approach_axis]} (mm) [approach axis]")
        ax_bot.set_ylabel("p_max (Pa)")
        ax_bot.set_title(f"Axial profile through target")
        ax_bot.legend(loc="best", fontsize=8)
        ax_bot.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"GLADYS nnU-Net + MaxAngle={MAX_ANGLE_DEG}deg  |  "
        f"active={active_elements}/{total_elements}, "
        f"bone={pct_skull:.1f}%, skull path={skull_path_mm:.1f}mm",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print(f"    Saved plot: {plot_path}")

    t_elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")
    print("Done.")


if __name__ == "__main__":
    main()
