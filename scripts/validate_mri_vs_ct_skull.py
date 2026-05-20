#!/usr/bin/env python3
"""Validate MRI-derived skull segmentation against CT-derived skull for FUS simulation.

For each SynthRAD2023 subject with paired CT + MRI data, this script:
  1. Runs ThresholdMRI on the MRI (with classify_brain_tissues=True) for soft tissue labels
  2. Runs skull-only nnU-Net (Dataset001_SkullSeg, binary) on the MRI for the skull mask
  3. Builds a CT-derived skull mask (HU thresholding within body mask)
  4. Creates two label volumes that share IDENTICAL soft tissue labels:
     - MRI-skull: ThresholdMRI soft tissue + nnU-Net skull mask
     - CT-skull:  ThresholdMRI soft tissue + CT-thresholded skull mask
  5. Runs k-Wave FDTD simulation for each label set
  6. Compares focal metrics (position, pressure, attenuation)

The skull-only model (Dataset001_SkullSeg) was trained on 251 subjects including
180 SynthRAD subjects and achieves Dice 0.903 on held-out SynthRAD data. For
SynthRAD subjects, the script auto-detects the out-of-fold number so that
inference uses the fold where the subject was held out from training.

Usage:
    source ~/openlifu-env/bin/activate
    PYTHONPATH=~/OpenLIFU-python/src:$PYTHONPATH python3 -u scripts/validate_mri_vs_ct_skull.py \
        --subject 1BA001 \
        --synthrad-dir ~/Data/openlifu-validation/datasets/synthrad2023-paired/Task1/brain \
        --output-dir ~/Data/openlifu-validation/results/mri_vs_ct
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import pathlib
import sys
import time

# ---------------------------------------------------------------------------
# Workaround for kwave v3 logging bug (same as other scripts)
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
import xarray as xa
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load GPU flock helper (same pattern as run_gladys_nnunet_subject.py)
import importlib.util as _ilu

_gpu_flock_spec = _ilu.spec_from_file_location(
    "_gpu_flock",
    os.path.join(os.path.dirname(__file__), "_gpu_flock.py"),
)
_gpu_flock_mod = _ilu.module_from_spec(_gpu_flock_spec)
_gpu_flock_spec.loader.exec_module(_gpu_flock_mod)
gpu_flock = _gpu_flock_mod.gpu_flock

from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER, ThresholdMRI
from openlifu.sim.kwave_if import (
    get_kgrid,
    get_medium,
    get_point_source,
    run_simulation,
)
from openlifu.sim.sim_setup import SimSetup
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulation constants (matching run_gladys_nnunet_subject.py)
# ---------------------------------------------------------------------------
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
GRID_MARGIN_MM = 25.0


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------
def _default_threshold_mri_materials() -> dict[str, Material]:
    """Materials dict matching ThresholdMRI with classify_brain_tissues=True."""
    m = MATERIALS.copy()
    m.pop("tissue", None)
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


# ---------------------------------------------------------------------------
# Out-of-fold lookup for skull-only model
# ---------------------------------------------------------------------------
def lookup_oof_fold(subject: str, splits_json_path: str | None = None) -> int | None:
    """Find the fold where this subject was held out from training.

    Reads splits_final.json from the nnU-Net preprocessed directory for
    Dataset001_SkullSeg. SynthRAD subjects are prefixed with 'srad_' in
    the splits file (e.g., 'srad_1BA001').

    Returns the fold number (0-4) if found, or None if the subject is not
    in any validation split.
    """
    if splits_json_path is None:
        splits_json_path = os.path.expanduser(
            "~/nnUNet_preprocessed/Dataset001_SkullSeg/splits_final.json"
        )
    if not os.path.exists(splits_json_path):
        logger.warning("splits_final.json not found at %s", splits_json_path)
        return None

    with open(splits_json_path) as f:
        splits = json.load(f)

    # SynthRAD subjects are prefixed with srad_ in the splits
    srad_id = f"srad_{subject}"

    for fold_idx, fold in enumerate(splits):
        val_keys = fold.get("val", [])
        if srad_id in val_keys or subject in val_keys:
            logger.info(
                "Subject %s found in fold %d validation set (as '%s')",
                subject, fold_idx,
                srad_id if srad_id in val_keys else subject,
            )
            return fold_idx

    logger.warning(
        "Subject %s (or %s) not found in any validation split",
        subject, srad_id,
    )
    return None


# ---------------------------------------------------------------------------
# nnU-Net skull-only prediction (Dataset001_SkullSeg)
# ---------------------------------------------------------------------------
def run_or_load_skull_prediction(
    mr_path: Path,
    output_dir: Path,
    subject: str,
    fold: int,
) -> Path:
    """Run nnU-Net skull-only model on MRI, or load cached prediction.

    Uses Dataset001_SkullSeg (binary output: 0=background, 1=skull).
    The fold argument specifies which trained fold to use for inference,
    typically the out-of-fold number for the subject.

    Returns the path to the skull-only label NIfTI.
    """
    pred_path = output_dir / f"{subject}_skull_only_labels.nii.gz"
    if pred_path.exists():
        logger.info("Using cached skull-only prediction: %s", pred_path)
        return pred_path

    # Check for nnU-Net environment variables
    nnunet_results = os.environ.get("nnUNet_results", "")
    if not nnunet_results:
        home = Path.home()
        candidates = [
            home / "nnUNet_results",
            home / "Data" / "nnUNet_results",
        ]
        for c in candidates:
            if c.exists():
                nnunet_results = str(c)
                os.environ["nnUNet_results"] = nnunet_results
                break

    import shutil
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="nnunet_input_") as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        # nnU-Net expects files named <case>_0000.nii.gz
        shutil.copy2(str(mr_path), str(input_dir / f"{subject}_0000.nii.gz"))

        pred_output_dir = Path(tmpdir) / "output"
        pred_output_dir.mkdir()

        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(pred_output_dir),
            "-d", "1",  # Dataset001_SkullSeg
            "-c", "3d_fullres",
            "-f", str(fold),
            "--disable_tta",
        ]
        logger.info("Running nnU-Net skull-only prediction: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # nnU-Net outputs <case>.nii.gz
        nnunet_output = pred_output_dir / f"{subject}.nii.gz"
        if not nnunet_output.exists():
            outputs = list(pred_output_dir.glob("*.nii.gz"))
            if outputs:
                nnunet_output = outputs[0]
            else:
                raise FileNotFoundError(
                    f"nnU-Net produced no output in {pred_output_dir}"
                )

        shutil.copy2(str(nnunet_output), str(pred_path))
        logger.info("Saved skull-only prediction: %s", pred_path)

    return pred_path


# ---------------------------------------------------------------------------
# Array geometry (same golden-angle spiral as run_gladys_nnunet_subject.py)
# ---------------------------------------------------------------------------
def create_hemispherical_array(
    n_elements=64, radius_mm=90.0, aperture_mm=80.0,
    freq_hz=500e3, element_size_mm=5.0,
) -> Transducer:
    half_aperture = aperture_mm / 2.0
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
        id="hemi_validate", name=f"Hemispherical {n_elements}-element array",
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


# ---------------------------------------------------------------------------
# CT skull mask from Hounsfield units
# ---------------------------------------------------------------------------
def ct_skull_mask(ct_path: Path, mask_path: Path, hu_threshold: float = 300.0) -> np.ndarray:
    """Threshold CT at HU > hu_threshold within the body mask.

    Returns a boolean array in the CT voxel grid: True = skull bone.
    """
    ct_img = nib.load(str(ct_path))
    ct_data = np.asarray(ct_img.dataobj, dtype=np.float32)
    mask_img = nib.load(str(mask_path))
    mask_data = np.asarray(mask_img.dataobj, dtype=np.float32) > 0.5
    bone = (ct_data > hu_threshold) & mask_data
    return bone


# ---------------------------------------------------------------------------
# Resample a binary mask from one voxel grid to another (nearest-neighbor)
# ---------------------------------------------------------------------------
def resample_mask_to_sim_grid(
    mask: np.ndarray,
    mask_affine: np.ndarray,
    mask_shape: tuple,
    sim_coord_arrays: list[np.ndarray],
) -> np.ndarray:
    """Resample a binary mask into the simulation grid using nearest-neighbor.

    Args:
        mask: Boolean array in the source voxel grid.
        mask_affine: 4x4 affine of the source volume.
        mask_shape: Shape of the source volume (for building coordinates).
        sim_coord_arrays: List of [x_coords, y_coords, z_coords] for the sim grid.

    Returns:
        Boolean array on the simulation grid.
    """
    # Build coordinate grids for the source volume
    src_coords = []
    for axis in range(3):
        origin = float(mask_affine[axis, 3])
        spacing = float(mask_affine[axis, axis])
        src_coords.append(origin + np.arange(mask_shape[axis]) * spacing)

    # Interpolate into sim grid
    interp = RegularGridInterpolator(
        src_coords, mask.astype(np.float32),
        method="nearest", bounds_error=False, fill_value=0.0,
    )
    mg = np.meshgrid(*sim_coord_arrays, indexing="ij")
    query_pts = np.stack([m.ravel() for m in mg], axis=-1)
    sim_shape = tuple(len(c) for c in sim_coord_arrays)
    return interp(query_pts).reshape(sim_shape) > 0.5


# ---------------------------------------------------------------------------
# Build label volumes by overlaying skull mask onto ThresholdMRI base labels
# ---------------------------------------------------------------------------
def build_labels_with_skull_overlay(
    base_labels: np.ndarray,
    skull_mask_sim: np.ndarray,
    material_idx: dict[str, int],
) -> np.ndarray:
    """Replace ThresholdMRI's skull estimate with the given skull mask.

    Where the skull mask says skull: set to skull material index.
    Where ThresholdMRI said skull but the mask disagrees: set to water
    (soft tissue fallback, since the external mask disagrees about bone there).

    Only applies the skull mask where the head is present (i.e., where the
    base labels are not water), to avoid labeling outside-head voxels as skull.

    Args:
        base_labels: Integer label array from ThresholdMRI (on sim grid).
        skull_mask_sim: Boolean skull mask on the sim grid.
        material_idx: Mapping from material name to integer label index.

    Returns:
        New integer label array with skull replaced.
    """
    skull_idx = material_idx["skull"]
    water_idx = material_idx["water"]

    output = base_labels.copy()

    # Where ThresholdMRI said skull but new mask disagrees: fall back to water
    threshold_skull = base_labels == skull_idx
    output[threshold_skull & ~skull_mask_sim] = water_idx

    # Where new mask says skull and the head is present: override to skull
    head_present = base_labels != water_idx
    output[skull_mask_sim & head_present] = skull_idx

    return output


# ---------------------------------------------------------------------------
# Save a label array as NIfTI (for debugging / inspection)
# ---------------------------------------------------------------------------
def save_labels_nifti(labels: np.ndarray, affine: np.ndarray, path: Path):
    img = nib.Nifti1Image(labels.astype(np.int16), affine)
    nib.save(img, str(path))
    logger.info("Saved labels: %s", path)


# ---------------------------------------------------------------------------
# Time-gated sparse sensor probe (simplified from run_gladys_nnunet_subject.py)
# ---------------------------------------------------------------------------
def _run_sparse_sensor_sim(
    arr: Transducer,
    params: xa.Dataset,
    delays: np.ndarray,
    apod: np.ndarray,
    sensor_mask_params_order: np.ndarray,
    freq: float,
    cycles: int,
    amplitude: float,
    dt: float,
    t_end: float,
    cfl: float,
    gpu: bool,
    ref_values_only: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end, cfl=cfl)
    if t_end == 0:
        _coord_units = [params[dim].attrs["units"] for dim in params.dims]
        _scl_to_m = getunitconversion(_coord_units[0], "m")
        _c_ref = float(params["sound_speed"].attrs.get("ref_value", 1500.0))
        _max_delay = float(np.max(np.abs(delays)))
        _extents_sq = 0.0
        for dim in params.dims:
            cv = params.coords[dim].to_numpy()
            _extents_sq += ((float(cv[-1]) - float(cv[0])) * _scl_to_m) ** 2
        _grid_diagonal = float(np.sqrt(_extents_sq))
        _signal_duration = cycles / freq
        _t_end_needed = (_max_delay + _grid_diagonal / _c_ref + _signal_duration) * 1.1
        _auto_t_end = float(kgrid.Nt * kgrid.dt)
        if _auto_t_end < _t_end_needed:
            kgrid = get_kgrid(params.coords, dt=float(kgrid.dt), t_end=_t_end_needed, cfl=cfl)

    t = np.arange(
        0,
        np.min([cycles / freq, (kgrid.Nt - np.ceil(max(delays) / kgrid.dt)) * kgrid.dt]),
        kgrid.dt,
    )
    input_signal = amplitude * np.sin(2 * np.pi * freq * t)

    medium = get_medium(params, ref_values_only=ref_values_only)
    source_mat = arr.calc_output(input_signal, kgrid.dt, delays, apod)
    source = get_point_source(arr, params, source_mat)

    dim_names = list(params.dims)
    _dim_order = {"x": 0, "y": 1, "z": 2}
    perm = [_dim_order[d] for d in dim_names]
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    sensor_mask_xyz = np.transpose(sensor_mask_params_order, inv_perm)

    nz = np.nonzero(sensor_mask_xyz)
    lin = (
        nz[0].astype(np.int64)
        + nz[1].astype(np.int64) * sensor_mask_xyz.shape[0]
        + nz[2].astype(np.int64) * sensor_mask_xyz.shape[0] * sensor_mask_xyz.shape[1]
    )
    order = np.argsort(lin)
    xyz_sensor_indices = np.stack([nz[0][order], nz[1][order], nz[2][order]], axis=-1)

    sensor = kSensor(sensor_mask_xyz, record=["p"])
    simulation_options = SimulationOptions(
        pml_auto=True, pml_inside=False, save_to_disk=True, data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=gpu)
    inputs = {
        "kgrid": kgrid, "source": source, "sensor": sensor, "medium": medium,
        "simulation_options": simulation_options, "execution_options": execution_options,
    }
    logger.info(
        "Running sparse-sensor probe (%d sensor voxels, ref_values_only=%s)...",
        int(xyz_sensor_indices.shape[0]), ref_values_only,
    )
    try:
        with gpu_flock():
            output = kspaceFirstOrder3D(**deepcopy(inputs))
    finally:
        for fpath in [simulation_options.input_filename, simulation_options.output_filename]:
            with contextlib.suppress(OSError):
                pathlib.Path(fpath).unlink(missing_ok=True)

    p_sensor = np.asarray(output["p"])
    if p_sensor.ndim == 1:
        p_sensor = p_sensor.reshape(-1, 1)
    return p_sensor, float(kgrid.dt), xyz_sensor_indices


def _build_target_sensor_mask(
    params: xa.Dataset,
    target_mm: np.ndarray,
    aperture_center_mm: np.ndarray,
) -> tuple[np.ndarray, list[str], list[tuple[int, int, int]], list[np.ndarray]]:
    """Build a small sensor mask with just the target and a few reference points."""
    axis_vec = aperture_center_mm - target_mm
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 1e-6:
        raise ValueError("aperture center coincides with target")
    axis_unit = axis_vec / axis_len

    # Find a perpendicular direction
    perp = None
    for cand in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        cross = np.cross(axis_unit, cand)
        if np.linalg.norm(cross) > 0.5:
            p = cand - axis_unit * np.dot(cand, axis_unit)
            p_norm = np.linalg.norm(p)
            if p_norm > 1e-6:
                perp = p / p_norm
                break
    if perp is None:
        perp = np.array([1.0, 0.0, 0.0])

    sensor_world_mm = [
        ("target",        target_mm.copy()),
        ("aperture_ctr",  aperture_center_mm.copy()),
        ("midway",        target_mm + 0.5 * axis_vec),
    ]

    sim_dim_names = list(params.dims)
    sim_coord_params = {d: params.coords[d].to_numpy() for d in sim_dim_names}
    mask = np.zeros(tuple(len(sim_coord_params[d]) for d in sim_dim_names), dtype=np.int32)

    names: list[str] = []
    xyz_idx_list: list[tuple[int, int, int]] = []
    actual_list: list[np.ndarray] = []
    seen_xyz: set[tuple[int, int, int]] = set()
    for name, pos_mm in sensor_world_mm:
        idx_in_params = []
        idx_in_xyz = [0, 0, 0]
        actual = np.zeros(3)
        for _params_ax, dim in enumerate(sim_dim_names):
            cv = sim_coord_params[dim]
            xyz_ax = {"x": 0, "y": 1, "z": 2}[dim]
            val = pos_mm[xyz_ax]
            i = int(np.argmin(np.abs(cv - val)))
            idx_in_params.append(i)
            idx_in_xyz[xyz_ax] = i
            actual[xyz_ax] = float(cv[i])
        xyz_tuple = tuple(idx_in_xyz)
        if xyz_tuple in seen_xyz:
            continue
        seen_xyz.add(xyz_tuple)
        mask[tuple(idx_in_params)] = 1
        names.append(name)
        xyz_idx_list.append(xyz_tuple)
        actual_list.append(actual)
    return mask, names, xyz_idx_list, actual_list


def _run_timegated_probe(
    *,
    arr: Transducer,
    params: xa.Dataset,
    delays: np.ndarray,
    apod: np.ndarray,
    target_mm: np.ndarray,
    aperture_center_mm: np.ndarray,
    common_kwargs: dict,
    ref_values_only: bool,
    sim_label: str,
    target_gate_center_s: float | None = None,
) -> dict | None:
    """Run sparse-sensor probe and compute time-gated metrics."""
    try:
        mask, names, xyz_idx_list, actual_list = _build_target_sensor_mask(
            params, target_mm, aperture_center_mm,
        )
        probe_kwargs = dict(common_kwargs)
        probe_kwargs.pop("arr", None)
        probe_kwargs.pop("apod", None)
        probe_kwargs.pop("source_method", None)

        t0 = time.time()
        p_sensor, dt_probe, xyz_order = _run_sparse_sensor_sim(
            arr=arr,
            params=params,
            delays=delays,
            apod=apod,
            sensor_mask_params_order=mask,
            ref_values_only=ref_values_only,
            **probe_kwargs,
        )
        probe_secs = time.time() - t0
        logger.info(
            "    [%s] probe done in %.1fs (p shape=%s, dt=%.2f ns)",
            sim_label, probe_secs, p_sensor.shape, dt_probe * 1e9,
        )

        pulse_dur = CYCLES / FREQ_HZ
        Nt = p_sensor.shape[0]
        t_axis = np.arange(Nt) * dt_probe

        _col_index: dict[tuple[int, int, int], int] = {}
        for _ci in range(xyz_order.shape[0]):
            _key = (int(xyz_order[_ci, 0]), int(xyz_order[_ci, 1]), int(xyz_order[_ci, 2]))
            if _key not in _col_index:
                _col_index[_key] = _ci

        per_sensor = {}
        for name, xyz_idx, actual in zip(names, xyz_idx_list, actual_list):
            col = _col_index.get(tuple(int(v) for v in xyz_idx))
            d_m = float(np.linalg.norm(actual - aperture_center_mm)) * 1e-3
            tof_s = d_m / C0
            rec: dict[str, Any] = {
                "sensor_world_mm": actual.tolist(),
                "tof_s": tof_s,
                "tof_us": tof_s * 1e6,
            }
            if col is None:
                rec.update({
                    "p_focal_window_Pa": float("nan"),
                    "p_focal_window_water_gate_Pa": float("nan"),
                    "p_allt_Pa": float("nan"),
                    "peak_time_s": float("nan"),
                })
            else:
                p_t = p_sensor[:, col]
                p_abs = np.abs(p_t)
                p_allt = float(p_abs.max())
                peak_time_s = float(t_axis[int(np.argmax(p_abs))]) if Nt > 0 else float("nan")

                # Geometric gate
                t_lo = tof_s - 2.0 * pulse_dur
                t_hi = tof_s + 2.0 * pulse_dur
                in_win = (t_axis >= t_lo) & (t_axis <= t_hi)
                p_fw_geom = float(p_abs[in_win].max()) if in_win.any() else float("nan")

                # Water-calibrated gate
                p_fw_water = float("nan")
                if target_gate_center_s is not None and name == "target":
                    t_lo_w = target_gate_center_s - 2.0 * pulse_dur
                    t_hi_w = target_gate_center_s + 2.0 * pulse_dur
                    in_win_w = (t_axis >= t_lo_w) & (t_axis <= t_hi_w)
                    p_fw_water = float(p_abs[in_win_w].max()) if in_win_w.any() else float("nan")

                rec.update({
                    "p_focal_window_Pa": p_fw_geom,
                    "p_focal_window_water_gate_Pa": p_fw_water,
                    "p_allt_Pa": p_allt,
                    "peak_time_s": peak_time_s,
                })
            per_sensor[name] = rec

        tgt = per_sensor.get("target", {})
        result = {
            "sim_label": sim_label,
            "p_focal_window_at_target_Pa": tgt.get("p_focal_window_Pa", float("nan")),
            "p_focal_window_water_gate_at_target_Pa": tgt.get("p_focal_window_water_gate_Pa", float("nan")),
            "p_allt_at_target_Pa": tgt.get("p_allt_Pa", float("nan")),
            "target_peak_time_s": tgt.get("peak_time_s", float("nan")),
            "target_tof_s": tgt.get("tof_s", float("nan")),
            "probe_runtime_s": probe_secs,
            "sensors": per_sensor,
        }
        return result
    except Exception as exc:
        logger.warning(
            "Time-gated probe failed for %s (%s); continuing without it.",
            sim_label, exc, exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate MRI-derived skull vs CT-derived skull for FUS simulation."
    )
    ap.add_argument("--subject", required=True,
                    help="Subject ID, e.g., 1BA001")
    ap.add_argument("--synthrad-dir", required=True,
                    help="Path to SynthRAD2023 Task1/brain directory")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for output results")
    ap.add_argument("--hu-threshold", type=float, default=300.0,
                    help="HU threshold for CT bone (default: 300)")
    ap.add_argument("--n-elements", type=int, default=64,
                    help="Number of array elements (default: 64)")
    ap.add_argument("--fold", type=int, default=None,
                    help="nnU-Net fold to use for skull-only inference "
                         "(overrides auto-detection from splits_final.json)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Run on CPU instead of GPU")
    args = ap.parse_args()

    subj = args.subject
    synthrad_dir = Path(args.synthrad_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    use_gpu = not args.no_gpu
    n_elements = args.n_elements
    hu_threshold = args.hu_threshold

    t_total = time.time()

    # -----------------------------------------------------------------------
    # 1. Load SynthRAD paired data
    # -----------------------------------------------------------------------
    subj_dir = synthrad_dir / subj
    ct_path = subj_dir / "ct.nii.gz"
    mr_path = subj_dir / "mr.nii.gz"
    mask_path = subj_dir / "mask.nii.gz"

    print("=" * 72)
    print(f"MRI vs CT Skull Validation (skull-only model) | subject={subj}")
    print(f"  CT:   {ct_path}")
    print(f"  MR:   {mr_path}")
    print(f"  Mask: {mask_path}")
    print(f"  HU threshold: {hu_threshold}")
    print(f"  Elements: {n_elements}")
    print("=" * 72)

    for p, label in [(ct_path, "CT"), (mr_path, "MR"), (mask_path, "Mask")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Determine fold for skull-only model
    # -----------------------------------------------------------------------
    if args.fold is not None:
        fold = args.fold
        print(f"\n[1] Using user-specified fold: {fold}")
    else:
        print("\n[1] Looking up out-of-fold number for skull-only model...")
        fold = lookup_oof_fold(subj)
        if fold is None:
            print("  WARNING: Subject not found in splits_final.json, defaulting to fold 0")
            fold = 0
        else:
            print(f"  Auto-detected fold: {fold}")

    # -----------------------------------------------------------------------
    # 3. Get skull-only nnU-Net prediction (binary mask from MRI)
    # -----------------------------------------------------------------------
    print(f"\n[2] Getting skull-only nnU-Net prediction (Dataset001, fold {fold})...")
    skull_pred_path = run_or_load_skull_prediction(mr_path, output_dir, subj, fold)
    skull_pred_img = nib.load(str(skull_pred_path))
    skull_pred_data = np.asarray(skull_pred_img.dataobj).astype(np.int16)
    skull_pred_affine = skull_pred_img.affine
    skull_pred_shape = skull_pred_data.shape

    n_skull_pred = int((skull_pred_data == 1).sum())
    pct_skull_pred = 100.0 * n_skull_pred / skull_pred_data.size
    print(f"  Skull prediction shape: {skull_pred_shape}")
    print(f"  nnU-Net skull voxels (native): {n_skull_pred:,d} ({pct_skull_pred:.2f}%)")

    # -----------------------------------------------------------------------
    # 4. Build CT-derived skull mask
    # -----------------------------------------------------------------------
    print(f"\n[3] Building CT skull mask (HU > {hu_threshold})...")
    ct_img = nib.load(str(ct_path))
    ct_bone = ct_skull_mask(ct_path, mask_path, hu_threshold)
    ct_affine = ct_img.affine
    ct_shape = ct_img.shape
    n_ct_bone = int(ct_bone.sum())
    pct_ct_bone = 100.0 * n_ct_bone / ct_bone.size
    print(f"  CT bone voxels: {n_ct_bone:,d} ({pct_ct_bone:.2f}%)")

    # -----------------------------------------------------------------------
    # 5. Build simulation grid and resample MRI
    # -----------------------------------------------------------------------
    print("\n[4] Setting up simulation grid...")
    mri_volume = load_nifti_as_xarray(mr_path)

    # We need the target and array to define the grid, so run ThresholdMRI
    # on the native MRI first to find brain center for target placement.
    # Then build the sim grid, resample MRI onto it, and run ThresholdMRI
    # on the sim-grid volume for the actual label map.

    # Quick ThresholdMRI on native MRI just to find brain center
    print("  Running ThresholdMRI on native MRI to find brain target...")
    tmri_native = ThresholdMRI(classify_brain_tissues=True)
    native_labels = tmri_native._segment(mri_volume)
    native_mat_idx = tmri_native._material_indices()
    native_arr = native_labels.to_numpy()

    dim_names = list(mri_volume.dims)
    coord_arrays = {d: mri_volume.coords[d].to_numpy() for d in dim_names}

    brain_mask = np.zeros(native_arr.shape, dtype=bool)
    for k in ("csf", "gray_matter", "white_matter"):
        if k in native_mat_idx:
            brain_mask |= native_arr == native_mat_idx[k]
    if brain_mask.sum() == 0:
        if "tissue" in native_mat_idx:
            brain_mask = native_arr == native_mat_idx["tissue"]
    if brain_mask.sum() == 0:
        print("ERROR: no brain voxels found in segmentation")
        sys.exit(2)

    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"  Target (brain center): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    # Determine approach direction (axis with max skull extent from target)
    skull_mask_native = native_arr == native_mat_idx["skull"]
    skull_indices = np.argwhere(skull_mask_native)
    if skull_indices.size == 0:
        print("  WARNING: no skull voxels in ThresholdMRI segmentation, using nnU-Net skull mask for approach direction")
        skull_indices = np.argwhere(skull_pred_data == 1)
        if skull_indices.size == 0:
            print("ERROR: no skull voxels in nnU-Net prediction either")
            sys.exit(2)

    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])
    max_skull_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_per_axis))
    approach_dir = np.zeros(3)
    approach_dir[approach_axis] = 1.0
    print(f"  Approach direction: axis {dim_names[approach_axis]} (auto-detected)")

    # Create the transducer array
    print("\n[5] Creating hemispherical array...")
    arr_local = create_hemispherical_array(
        n_elements=n_elements, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )

    # Rotate array to approach direction
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, approach_dir)
    c = np.dot(z_axis, approach_dir)
    if np.linalg.norm(v) < 1e-10:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
    transform = np.eye(4)
    transform[:3, :3] = R
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
    aperture_center_mm = positions.mean(axis=0)
    print(f"  Aperture center: ({aperture_center_mm[0]:.1f}, {aperture_center_mm[1]:.1f}, {aperture_center_mm[2]:.1f}) mm")

    # Build simulation grid
    print("\n[6] Building simulation grid...")
    all_points = np.vstack([positions, target_mm[np.newaxis, :]])
    grid_min = np.floor((all_points.min(axis=0) - GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    grid_max = np.ceil((all_points.max(axis=0) + GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    sim_coords = {}
    for ax, dim in enumerate(["x", "y", "z"]):
        n_pts = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        sim_coords[dim] = xa.Variable(
            dim, np.linspace(grid_min[ax], grid_max[ax], n_pts),
            attrs={"units": "mm"},
        )
    grid_shape = tuple(len(sim_coords[d]) for d in ["x", "y", "z"])
    print(f"  Grid shape: {grid_shape}")

    # Resample MRI volume onto sim grid
    orig_coords_list = [mri_volume.coords[d].to_numpy() for d in mri_volume.dims]
    interp = RegularGridInterpolator(
        orig_coords_list, mri_volume.to_numpy(),
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

    # -----------------------------------------------------------------------
    # 6. Run ThresholdMRI on sim-grid volume for base labels
    # -----------------------------------------------------------------------
    print("\n[7] Running ThresholdMRI on sim-grid MRI (classify_brain_tissues=True)...")
    tmri = ThresholdMRI(classify_brain_tissues=True)
    base_labels_da = tmri._segment(sim_volume)
    base_labels = base_labels_da.to_numpy()
    material_idx = tmri._material_indices()

    print("  ThresholdMRI base label histogram:")
    for mat_name, idx in sorted(material_idx.items(), key=lambda x: x[1]):
        n = int((base_labels == idx).sum())
        pct = 100.0 * n / base_labels.size
        print(f"    {idx} ({mat_name}): {n:,d} ({pct:.2f}%)")

    # -----------------------------------------------------------------------
    # 7. Resample skull masks to sim grid
    # -----------------------------------------------------------------------
    print("\n[8] Resampling skull masks to simulation grid...")

    # nnU-Net skull mask -> sim grid
    nnunet_skull_bool = skull_pred_data == 1
    nnunet_skull_sim = resample_mask_to_sim_grid(
        nnunet_skull_bool, skull_pred_affine, skull_pred_shape, sim_coord_arrays,
    )
    n_nnunet_sim = int(nnunet_skull_sim.sum())
    print(f"  nnU-Net skull on sim grid: {n_nnunet_sim:,d} voxels")

    # CT bone mask -> sim grid
    ct_skull_sim = resample_mask_to_sim_grid(
        ct_bone, ct_affine, ct_shape, sim_coord_arrays,
    )
    n_ct_sim = int(ct_skull_sim.sum())
    print(f"  CT skull on sim grid: {n_ct_sim:,d} voxels")

    # -----------------------------------------------------------------------
    # 8. Build two label volumes
    # -----------------------------------------------------------------------
    print("\n[9] Building label volumes (shared soft tissue, different skull)...")

    # MRI-skull labels: ThresholdMRI base + nnU-Net skull overlay
    mri_labels = build_labels_with_skull_overlay(base_labels, nnunet_skull_sim, material_idx)

    # CT-skull labels: ThresholdMRI base + CT-thresholded skull overlay
    ct_labels = build_labels_with_skull_overlay(base_labels, ct_skull_sim, material_idx)

    # Save label NIfTIs for inspection
    sim_affine = np.diag([GRID_SPACING_MM, GRID_SPACING_MM, GRID_SPACING_MM, 1.0])
    sim_affine[0, 3] = float(sim_coord_arrays[0][0])
    sim_affine[1, 3] = float(sim_coord_arrays[1][0])
    sim_affine[2, 3] = float(sim_coord_arrays[2][0])

    mri_label_path = output_dir / f"{subj}_mri_skull_labels.nii.gz"
    save_labels_nifti(mri_labels, sim_affine, mri_label_path)
    ct_label_path = output_dir / f"{subj}_ct_skull_labels.nii.gz"
    save_labels_nifti(ct_labels, sim_affine, ct_label_path)

    # Compare skull voxel counts
    skull_idx = material_idx["skull"]
    n_mri_skull = int((mri_labels == skull_idx).sum())
    n_ct_skull_in_labels = int((ct_labels == skull_idx).sum())
    overlap = int(((mri_labels == skull_idx) & (ct_labels == skull_idx)).sum())
    union = int(((mri_labels == skull_idx) | (ct_labels == skull_idx)).sum())
    dice = 2.0 * overlap / (n_mri_skull + n_ct_skull_in_labels) if (n_mri_skull + n_ct_skull_in_labels) > 0 else 0.0
    iou = overlap / union if union > 0 else 0.0

    print(f"  MRI skull voxels (sim grid): {n_mri_skull:,d}")
    print(f"  CT skull voxels (sim grid): {n_ct_skull_in_labels:,d}")
    print(f"  Overlap: {overlap:,d}")
    print(f"  Dice coefficient: {dice:.4f}")
    print(f"  IoU: {iou:.4f}")

    # Print histograms for both label sets
    for lbl_name, lbl_arr in [("MRI-skull labels", mri_labels), ("CT-skull labels", ct_labels)]:
        print(f"  {lbl_name} histogram:")
        for mat_name, idx in sorted(material_idx.items(), key=lambda x: x[1]):
            n = int((lbl_arr == idx).sum())
            pct = 100.0 * n / lbl_arr.size
            print(f"    {idx} ({mat_name}): {n:,d} ({pct:.2f}%)")

    # -----------------------------------------------------------------------
    # 9. Build sim params for each label set
    # -----------------------------------------------------------------------
    print("\n[10] Generating simulation parameters for both label sets...")

    # Build xarray DataArrays for the label volumes so we can use _map_params
    mri_labels_da = xa.DataArray(
        mri_labels, dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )
    ct_labels_da = xa.DataArray(
        ct_labels, dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )

    sim_params_mri = tmri._map_params(mri_labels_da)
    sim_params_ct = tmri._map_params(ct_labels_da)

    # Fix air cavities (replace air acoustic props with water, same as other scripts)
    water_mat = tmri.materials["water"]
    for sim_params_set, label_name, lbl_arr in [
        (sim_params_mri, "MRI-skull", mri_labels),
        (sim_params_ct, "CT-skull", ct_labels),
    ]:
        air_mask = lbl_arr == material_idx["air"]
        skull_mask_sim = lbl_arr == skull_idx
        n_skull = int(skull_mask_sim.sum())
        n_air = int(air_mask.sum())
        total_vox = lbl_arr.size
        pct_skull = 100.0 * n_skull / total_vox
        print(f"  {label_name}: bone={pct_skull:.2f}% ({n_skull:,d} vox), air={n_air:,d} vox")
        if air_mask.any():
            sim_params_set["sound_speed"].data[air_mask] = water_mat.sound_speed
            sim_params_set["density"].data[air_mask] = water_mat.density
            sim_params_set["attenuation"].data[air_mask] = water_mat.attenuation

    # -----------------------------------------------------------------------
    # 10. Compute delays
    # -----------------------------------------------------------------------
    print("\n[11] Computing delays...")
    target = Point(position=target_mm.copy(), id="brain_center",
                   name="Brain Center Target", units="mm")
    direct = Direct(c0=C0)
    delays_geo = direct.calc_delays(arr, target, sim_params_mri)

    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=CFL,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    t_end = (max_dist_mm * 1e-3) / C0 * T_END_SAFETY
    c_max_mri = float(sim_params_mri["sound_speed"].to_numpy().max())
    c_max_ct = float(sim_params_ct["sound_speed"].to_numpy().max())
    c_max = max(c_max_mri, c_max_ct)
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max

    # Phase correction for MRI-skull
    print("\n  SimulationCorrected for MRI-skull...")
    sim_corrected_mri = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=use_gpu)
    t0 = time.time()
    with gpu_flock():
        delays_mri = sim_corrected_mri.calc_delays(arr, target, sim_params_mri)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Delay range: {delays_mri.min()*1e6:.2f} to {delays_mri.max()*1e6:.2f} us")

    # Phase correction for CT-skull
    print("\n  SimulationCorrected for CT-skull...")
    sim_corrected_ct = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=use_gpu)
    t0 = time.time()
    with gpu_flock():
        delays_ct = sim_corrected_ct.calc_delays(arr, target, sim_params_ct)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Delay range: {delays_ct.min()*1e6:.2f} to {delays_ct.max()*1e6:.2f} us")

    # Per-element delay differences
    delay_diff = delays_mri - delays_ct
    print(f"\n  Per-element delay difference (MRI - CT):")
    print(f"    Mean: {delay_diff.mean()*1e6:.3f} us")
    print(f"    Std:  {delay_diff.std()*1e6:.3f} us")
    print(f"    Min:  {delay_diff.min()*1e6:.3f} us")
    print(f"    Max:  {delay_diff.max()*1e6:.3f} us")
    print(f"    RMS:  {np.sqrt(np.mean(delay_diff**2))*1e6:.3f} us")

    apod = np.ones(arr.numelements())
    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=use_gpu, source_method="point_source",
    )

    cx, cy, cz = sim_coord_arrays

    # -----------------------------------------------------------------------
    # 11. Run simulations
    # -----------------------------------------------------------------------
    # SIM 1: Water baseline (geometric delays, homogeneous water)
    print("\n[SIM 1] Water baseline (geometric delays, homogeneous)...")
    t0 = time.time()
    with gpu_flock():
        result_water = run_simulation(params=sim_params_mri, delays=delays_geo,
                                       ref_values_only=True, **common_kwargs)
    print(f"  Done in {time.time()-t0:.1f}s")
    probe_water = _run_timegated_probe(
        arr=arr, params=sim_params_mri, delays=delays_geo, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=True,
        sim_label="water",
        target_gate_center_s=None,
    )

    # Extract water-target peak time for gate calibration
    t_water_target_peak = None
    if probe_water is not None:
        t_water_target_peak = probe_water.get("target_peak_time_s", None)
        if t_water_target_peak is None or not np.isfinite(t_water_target_peak):
            t_water_target_peak = None
        else:
            tgt_geom_tof = probe_water.get("target_tof_s", float("nan"))
            print(
                f"  [water-gate] target peak time = {t_water_target_peak*1e6:.2f} us, "
                f"geom tof = {tgt_geom_tof*1e6:.2f} us"
            )

    # SIM 2: MRI-skull (corrected delays)
    print("\n[SIM 2] MRI-skull (SimulationCorrected delays)...")
    t0 = time.time()
    with gpu_flock():
        result_mri = run_simulation(params=sim_params_mri, delays=delays_mri,
                                     ref_values_only=False, **common_kwargs)
    print(f"  Done in {time.time()-t0:.1f}s")
    probe_mri = _run_timegated_probe(
        arr=arr, params=sim_params_mri, delays=delays_mri, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=False,
        sim_label="MRI-skull",
        target_gate_center_s=t_water_target_peak,
    )

    # SIM 3: CT-skull (corrected delays)
    print("\n[SIM 3] CT-skull (SimulationCorrected delays)...")
    t0 = time.time()
    with gpu_flock():
        result_ct = run_simulation(params=sim_params_ct, delays=delays_ct,
                                    ref_values_only=False, **common_kwargs)
    print(f"  Done in {time.time()-t0:.1f}s")
    probe_ct = _run_timegated_probe(
        arr=arr, params=sim_params_ct, delays=delays_ct, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=False,
        sim_label="CT-skull",
        target_gate_center_s=t_water_target_peak,
    )

    # -----------------------------------------------------------------------
    # 12. Extract focal metrics from p_max fields
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    FOCAL_ROI_RADIUS_MM = 20.0

    def _focal_stats(result, label):
        pmax = result["p_max"].to_numpy()
        tidx = tuple(
            int(np.argmin(np.abs(sim_coord_arrays[ax] - target_mm[ax])))
            for ax in range(3)
        )
        p_at_target = float(pmax[tidx])

        mgx, mgy, mgz = np.meshgrid(cx, cy, cz, indexing="ij")
        dist_from_target = np.sqrt(
            (mgx - target_mm[0]) ** 2
            + (mgy - target_mm[1]) ** 2
            + (mgz - target_mm[2]) ** 2
        )
        roi_mask = dist_from_target <= FOCAL_ROI_RADIUS_MM

        pmax_roi = np.where(roi_mask, pmax, -np.inf)
        idx_roi = np.unravel_index(pmax_roi.argmax(), pmax_roi.shape)
        focal_pos = np.array([float(cx[idx_roi[0]]), float(cy[idx_roi[1]]), float(cz[idx_roi[2]])])
        focal_error = float(np.linalg.norm(focal_pos - target_mm))
        p_focal_peak = float(pmax[idx_roi])

        p_max_global = float(pmax.max())
        idx_global = np.unravel_index(pmax.argmax(), pmax.shape)
        global_peak_pos = np.array([float(cx[idx_global[0]]), float(cy[idx_global[1]]), float(cz[idx_global[2]])])

        threshold_6db = p_focal_peak / 2.0
        focal_region = roi_mask & (pmax >= threshold_6db)
        n_above = int(focal_region.sum())
        voxel_vol_mm3 = GRID_SPACING_MM ** 3
        focal_vol_mm3 = n_above * voxel_vol_mm3

        print(f"\n  {label}:")
        print(f"    p_focal_peak = {p_focal_peak:.4g} Pa at ({focal_pos[0]:.1f}, {focal_pos[1]:.1f}, {focal_pos[2]:.1f}) mm")
        print(f"    p_max_global = {p_max_global:.4g} Pa at ({global_peak_pos[0]:.1f}, {global_peak_pos[1]:.1f}, {global_peak_pos[2]:.1f}) mm")
        print(f"    p_at_target = {p_at_target:.4g} Pa")
        print(f"    Focal error = {focal_error:.1f} mm (ROI peak vs target, r={FOCAL_ROI_RADIUS_MM}mm)")
        print(f"    -6dB focal volume = {focal_vol_mm3:.1f} mm^3 ({n_above} voxels)")

        return {
            "p_max": p_max_global,
            "p_focal_peak": p_focal_peak,
            "p_at_target": p_at_target,
            "focal_pos_mm": focal_pos,
            "focal_error_mm": focal_error,
            "focal_vol_6db_mm3": focal_vol_mm3,
            "global_peak_pos_mm": global_peak_pos,
        }

    stats_water = _focal_stats(result_water, "Water baseline")
    stats_mri = _focal_stats(result_mri, "MRI-skull (corrected)")
    stats_ct = _focal_stats(result_ct, "CT-skull (corrected)")

    # Time-gated probe metrics
    def _db(num, den):
        if np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0:
            return 20.0 * np.log10(den / num)
        return float("nan")

    p_fw_water = probe_water.get("p_focal_window_at_target_Pa", float("nan")) if probe_water else float("nan")
    p_fw_mri = probe_mri.get("p_focal_window_at_target_Pa", float("nan")) if probe_mri else float("nan")
    p_fw_ct = probe_ct.get("p_focal_window_at_target_Pa", float("nan")) if probe_ct else float("nan")
    p_fw_water_mri = probe_mri.get("p_focal_window_water_gate_at_target_Pa", float("nan")) if probe_mri else float("nan")
    p_fw_water_ct = probe_ct.get("p_focal_window_water_gate_at_target_Pa", float("nan")) if probe_ct else float("nan")

    atten_mri_geom = _db(p_fw_mri, p_fw_water)
    atten_ct_geom = _db(p_fw_ct, p_fw_water)
    atten_mri_water = _db(p_fw_water_mri, p_fw_water)
    atten_ct_water = _db(p_fw_water_ct, p_fw_water)

    # -----------------------------------------------------------------------
    # 13. Comparison
    # -----------------------------------------------------------------------
    focal_pos_diff = float(np.linalg.norm(stats_mri["focal_pos_mm"] - stats_ct["focal_pos_mm"]))
    p_target_ratio = stats_mri["p_at_target"] / stats_ct["p_at_target"] if stats_ct["p_at_target"] > 0 else float("nan")
    atten_diff_geom = atten_mri_geom - atten_ct_geom
    atten_diff_water = atten_mri_water - atten_ct_water

    print("\n--- COMPARISON: MRI-skull vs CT-skull ---")
    print(f"  Focal position difference:     {focal_pos_diff:.2f} mm")
    print(f"    MRI focal: ({stats_mri['focal_pos_mm'][0]:.1f}, {stats_mri['focal_pos_mm'][1]:.1f}, {stats_mri['focal_pos_mm'][2]:.1f})")
    print(f"    CT  focal: ({stats_ct['focal_pos_mm'][0]:.1f}, {stats_ct['focal_pos_mm'][1]:.1f}, {stats_ct['focal_pos_mm'][2]:.1f})")
    print(f"  p_at_target ratio (MRI/CT):    {p_target_ratio:.4f}")
    print(f"  p_at_target MRI:               {stats_mri['p_at_target']:.4g} Pa")
    print(f"  p_at_target CT:                {stats_ct['p_at_target']:.4g} Pa")
    print(f"  Focal vol -6dB MRI:            {stats_mri['focal_vol_6db_mm3']:.1f} mm^3")
    print(f"  Focal vol -6dB CT:             {stats_ct['focal_vol_6db_mm3']:.1f} mm^3")

    print("\n--- TIME-GATED PROBE METRICS ---")
    print(f"  p_fw (geom gate) water:        {p_fw_water:.4g} Pa")
    print(f"  p_fw (geom gate) MRI-skull:    {p_fw_mri:.4g} Pa  (atten {atten_mri_geom:.2f} dB)")
    print(f"  p_fw (geom gate) CT-skull:     {p_fw_ct:.4g} Pa  (atten {atten_ct_geom:.2f} dB)")
    print(f"  p_fw (water gate) MRI-skull:   {p_fw_water_mri:.4g} Pa  (atten {atten_mri_water:.2f} dB)")
    print(f"  p_fw (water gate) CT-skull:    {p_fw_water_ct:.4g} Pa  (atten {atten_ct_water:.2f} dB)")
    print(f"  Atten diff (MRI - CT, geom):   {atten_diff_geom:.2f} dB")
    print(f"  Atten diff (MRI - CT, water):  {atten_diff_water:.2f} dB")

    print("\n--- SKULL SEGMENTATION OVERLAP ---")
    print(f"  Dice coefficient:              {dice:.4f}")
    print(f"  IoU:                           {iou:.4f}")
    print(f"  MRI skull voxels:              {n_mri_skull:,d}")
    print(f"  CT skull voxels (sim grid):    {n_ct_skull_in_labels:,d}")

    print("\n--- PER-ELEMENT DELAY COMPARISON ---")
    print(f"  Mean delay diff (MRI - CT):    {delay_diff.mean()*1e6:.3f} us")
    print(f"  Std delay diff:                {delay_diff.std()*1e6:.3f} us")
    print(f"  RMS delay diff:                {np.sqrt(np.mean(delay_diff**2))*1e6:.3f} us")
    print(f"  Max abs delay diff:            {np.abs(delay_diff).max()*1e6:.3f} us")

    # -----------------------------------------------------------------------
    # 14. Save p_max NIfTIs
    # -----------------------------------------------------------------------
    out_affine = np.diag([
        float(cx[1] - cx[0]) if len(cx) > 1 else 1.0,
        float(cy[1] - cy[0]) if len(cy) > 1 else 1.0,
        float(cz[1] - cz[0]) if len(cz) > 1 else 1.0,
        1.0,
    ])
    out_affine[0, 3] = float(cx[0])
    out_affine[1, 3] = float(cy[0])
    out_affine[2, 3] = float(cz[0])

    for sim_label, result_data in [
        ("water", result_water),
        ("mri_skull", result_mri),
        ("ct_skull", result_ct),
    ]:
        p_max_data = result_data["p_max"].to_numpy()
        out_path = output_dir / f"{subj}_{sim_label}_pmax.nii.gz"
        nib.save(nib.Nifti1Image(p_max_data.astype(np.float32), out_affine), str(out_path))
        print(f"  Saved: {out_path}")

    # -----------------------------------------------------------------------
    # 15. Save CSV results
    # -----------------------------------------------------------------------
    csv_path = output_dir / "mri_vs_ct_results.csv"
    write_header = not csv_path.exists()

    def _fmt(v):
        try:
            if not np.isfinite(v):
                return "nan"
            return f"{v:.6g}"
        except Exception:
            return "nan"

    row = {
        "subject": subj,
        "hu_threshold": hu_threshold,
        "n_elements": n_elements,
        "fold": fold,
        # Skull overlap
        "dice": f"{dice:.4f}",
        "iou": f"{iou:.4f}",
        "n_mri_skull_vox": n_mri_skull,
        "n_ct_skull_vox": n_ct_skull_in_labels,
        # Focal stats from p_max
        "p_at_target_water": _fmt(stats_water["p_at_target"]),
        "p_at_target_mri": _fmt(stats_mri["p_at_target"]),
        "p_at_target_ct": _fmt(stats_ct["p_at_target"]),
        "focal_error_mri_mm": _fmt(stats_mri["focal_error_mm"]),
        "focal_error_ct_mm": _fmt(stats_ct["focal_error_mm"]),
        "focal_pos_diff_mm": _fmt(focal_pos_diff),
        "p_target_ratio_mri_ct": _fmt(p_target_ratio),
        "focal_vol_6db_mri_mm3": _fmt(stats_mri["focal_vol_6db_mm3"]),
        "focal_vol_6db_ct_mm3": _fmt(stats_ct["focal_vol_6db_mm3"]),
        # Time-gated probe
        "p_fw_geom_water": _fmt(p_fw_water),
        "p_fw_geom_mri": _fmt(p_fw_mri),
        "p_fw_geom_ct": _fmt(p_fw_ct),
        "p_fw_water_gate_mri": _fmt(p_fw_water_mri),
        "p_fw_water_gate_ct": _fmt(p_fw_water_ct),
        "atten_mri_geom_dB": _fmt(atten_mri_geom),
        "atten_ct_geom_dB": _fmt(atten_ct_geom),
        "atten_mri_water_dB": _fmt(atten_mri_water),
        "atten_ct_water_dB": _fmt(atten_ct_water),
        "atten_diff_geom_dB": _fmt(atten_diff_geom),
        "atten_diff_water_dB": _fmt(atten_diff_water),
        # Delay comparison
        "delay_diff_mean_us": _fmt(delay_diff.mean() * 1e6),
        "delay_diff_std_us": _fmt(delay_diff.std() * 1e6),
        "delay_diff_rms_us": _fmt(np.sqrt(np.mean(delay_diff**2)) * 1e6),
        "delay_diff_max_abs_us": _fmt(np.abs(delay_diff).max() * 1e6),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  CSV row appended: {csv_path}")

    # -----------------------------------------------------------------------
    # 16. Save JSON sidecar with full details
    # -----------------------------------------------------------------------
    sidecar = {
        "subject": subj,
        "hu_threshold": hu_threshold,
        "n_elements": n_elements,
        "fold": fold,
        "skull_model": "Dataset001_SkullSeg",
        "target_mm": target_mm.tolist(),
        "aperture_center_mm": aperture_center_mm.tolist(),
        "approach_axis": dim_names[approach_axis],
        "grid_shape": list(grid_shape),
        "dice": dice,
        "iou": iou,
        "n_mri_skull_vox": n_mri_skull,
        "n_ct_skull_vox": n_ct_skull_in_labels,
        "delays_mri_us": (delays_mri * 1e6).tolist(),
        "delays_ct_us": (delays_ct * 1e6).tolist(),
        "delay_diff_us": (delay_diff * 1e6).tolist(),
        "stats_water": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats_water.items()},
        "stats_mri": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats_mri.items()},
        "stats_ct": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats_ct.items()},
        "focal_pos_diff_mm": focal_pos_diff,
        "p_target_ratio_mri_ct": p_target_ratio,
        "probe_water": {k: v for k, v in probe_water.items() if k != "sensors"} if probe_water else None,
        "probe_mri": {k: v for k, v in probe_mri.items() if k != "sensors"} if probe_mri else None,
        "probe_ct": {k: v for k, v in probe_ct.items() if k != "sensors"} if probe_ct else None,
    }
    sidecar_path = output_dir / f"{subj}_mri_vs_ct_sidecar.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  Saved sidecar: {sidecar_path}")

    t_elapsed = time.time() - t_total
    print(f"\n[{subj}] Total runtime: {t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")

    # One-line summary for batch parsing
    print(
        f"VALIDATION_SUMMARY subject={subj} "
        f"dice={dice:.4f} "
        f"iou={iou:.4f} "
        f"focal_pos_diff_mm={focal_pos_diff:.2f} "
        f"p_target_ratio={_fmt(p_target_ratio)} "
        f"atten_mri_geom={_fmt(atten_mri_geom)} "
        f"atten_ct_geom={_fmt(atten_ct_geom)} "
        f"atten_diff_geom={_fmt(atten_diff_geom)} "
        f"atten_mri_water={_fmt(atten_mri_water)} "
        f"atten_ct_water={_fmt(atten_ct_water)} "
        f"atten_diff_water={_fmt(atten_diff_water)} "
        f"delay_rms_us={_fmt(np.sqrt(np.mean(delay_diff**2))*1e6)}"
    )


if __name__ == "__main__":
    main()
