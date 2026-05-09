#!/usr/bin/env python3
"""Compare first_arrival vs argmax delay correction through real skull.

Runs THREE delay-correction strategies on a single subject:
  1. Direct geometric delays (no skull correction)
  2. SimulationCorrected with peak_method="first_arrival"
  3. SimulationCorrected with peak_method="argmax"

For each, runs a forward simulation through the nnU-Net-segmented skull and
a time-gated probe at the target. Reports p_focal_window for each method
and the ratio vs geometric baseline.

Total: 2 reciprocal sims + 3 forward sims + 3 probes = ~8 GPU calls.
Expected runtime: 5-10 minutes on RTX 4090.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pathlib
import sys
import time

# ---------------------------------------------------------------------------
# Workaround for kwave v3 logging bug
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load _gpu_flock (standalone script module, no package deps).
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
from openlifu.seg.seg_methods.nnunet_seg import LABEL_MAP_FULLHEAD
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER
from openlifu.sim.kwave_if import (
    get_kgrid,
    get_medium,
    get_point_source,
    run_simulation,
)
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
# Constants (same as run_gladys_nnunet_subject.py)
# ---------------------------------------------------------------------------
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
GRID_MARGIN_MM = 25.0


# ---------------------------------------------------------------------------
# PreSegmented (copied from run_gladys_nnunet_subject.py)
# ---------------------------------------------------------------------------
def _default_fullhead_materials() -> dict[str, Material]:
    m = MATERIALS.copy()
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


@dataclass
class PreSegmented(SegmentationMethod):
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
        assert set(tgt_dims) == set(src_dims)
        tgt_coord_arrays = [volume.coords[d].to_numpy() for d in src_dims]
        mg = np.meshgrid(*tgt_coord_arrays, indexing="ij")
        frac_idx = []
        for i, d in enumerate(src_dims):
            fi = (mg[i] - src_origin[d]) / src_spacing[d]
            frac_idx.append(fi)
        frac_stack = np.stack(frac_idx, axis=0)
        resampled = map_coordinates(
            src.to_numpy().astype(np.float32), frac_stack, order=0,
            mode="constant", cval=0.0,
        ).astype(np.int16)
        water_idx = mat_idx["water"]
        output = np.full(resampled.shape, water_idx, dtype=int)
        for nn_label, mat_key in self.nnunet_label_map.items():
            output[resampled == nn_label] = mat_idx[mat_key]
        labels_da = xa.DataArray(
            output, dims=src_dims,
            coords={d: volume.coords[d] for d in src_dims},
        )
        return labels_da.transpose(*tgt_dims)

    def to_table(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"Name": "Type", "Value": "PreSegmented (nnU-Net label NIfTI)", "Unit": ""},
            {"Name": "Label NIfTI", "Value": self.label_nifti_path, "Unit": ""},
            {"Name": "Reference Material", "Value": self.ref_material, "Unit": ""},
        ])


# ---------------------------------------------------------------------------
# Array construction (copied from run_gladys_nnunet_subject.py)
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
        id="hemi64", name=f"Hemispherical {n_elements}-element array",
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
# Sparse sensor sim + mask builder (copied from run_gladys_nnunet_subject.py)
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
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build a single-voxel sensor mask at the target location.

    Returns (mask_array, xyz_grid_index_tuple).
    """
    sim_dim_names = list(params.dims)
    sim_coord_params = {d: params.coords[d].to_numpy() for d in sim_dim_names}
    mask = np.zeros(tuple(len(sim_coord_params[d]) for d in sim_dim_names), dtype=np.int32)

    idx_in_params: list[int] = []
    idx_in_xyz = [0, 0, 0]
    for _params_ax, dim in enumerate(sim_dim_names):
        cv = sim_coord_params[dim]
        xyz_ax = {"x": 0, "y": 1, "z": 2}[dim]
        val = target_mm[xyz_ax]
        i = int(np.argmin(np.abs(cv - val)))
        idx_in_params.append(i)
        idx_in_xyz[xyz_ax] = i
    mask[tuple(idx_in_params)] = 1
    return mask, tuple(idx_in_xyz)


def _run_target_probe(
    *,
    arr: Transducer,
    params: xa.Dataset,
    delays: np.ndarray,
    apod: np.ndarray,
    target_mm: np.ndarray,
    aperture_center_mm: np.ndarray,
    common_kwargs: dict,
    ref_values_only: bool,
    label: str,
) -> dict:
    """Run a single-sensor probe at the target and return time-gated metrics."""
    mask, xyz_idx = _build_target_sensor_mask(params, target_mm)

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
    elapsed = time.time() - t0
    logger.info("  [%s] probe done in %.1fs", label, elapsed)

    pulse_dur = CYCLES / FREQ_HZ
    Nt = p_sensor.shape[0]
    t_axis = np.arange(Nt) * dt_probe

    # Target sensor is the only sensor (column 0)
    p_t = p_sensor[:, 0]
    p_abs = np.abs(p_t)
    p_allt = float(p_abs.max())
    peak_time_s = float(t_axis[int(np.argmax(p_abs))]) if Nt > 0 else float("nan")

    # Geometric gate: TOF from aperture center to target at C0
    d_m = float(np.linalg.norm(target_mm - aperture_center_mm)) * 1e-3
    tof_s = d_m / C0
    t_lo = tof_s - 2.0 * pulse_dur
    t_hi = tof_s + 2.0 * pulse_dur
    in_win = (t_axis >= t_lo) & (t_axis <= t_hi)
    p_focal_window = float(p_abs[in_win].max()) if in_win.any() else float("nan")

    return {
        "label": label,
        "p_focal_window_Pa": p_focal_window,
        "p_allt_Pa": p_allt,
        "peak_time_s": peak_time_s,
        "tof_s": tof_s,
        "probe_runtime_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare first_arrival vs argmax delay correction")
    ap.add_argument("--subject", default="GU002",
                    help="Subject ID (default: GU002)")
    ap.add_argument("--mri-path", default=None,
                    help="Override MRI path")
    ap.add_argument("--label-path", default=None,
                    help="Override label path")
    args = ap.parse_args()

    subj = args.subject
    mri_path = Path(args.mri_path) if args.mri_path else (
        Path.home() / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
        / "Anonymized_Subjects" / "T1-Weighted MRI" / f"{subj}_deface.nii"
    )
    label_path = Path(args.label_path) if args.label_path else (
        Path.home() / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"
    )

    t_total = time.time()
    print("=" * 72)
    print(f"first_arrival vs argmax comparison | subject={subj}")
    print(f"  MRI:    {mri_path}")
    print(f"  Labels: {label_path}")
    print("=" * 72)
    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}")
        sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load MRI + labels
    # -----------------------------------------------------------------------
    print("\n[1] Loading MRI and labels...")
    volume = load_nifti_as_xarray(mri_path)
    seg_method = PreSegmented(label_nifti_path=str(label_path))
    print(f"  MRI shape: {volume.shape}, label shape: {seg_method._labels.shape}")

    # -----------------------------------------------------------------------
    # Find brain center target
    # -----------------------------------------------------------------------
    print("\n[2] Finding brain center target...")
    seg_labels = seg_method._segment(volume)
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()
    coord_arrays = {d: volume.coords[d].to_numpy() for d in volume.dims}
    dim_names = list(volume.dims)

    brain_mask = np.zeros(seg_arr.shape, dtype=bool)
    for k in ("csf", "gray_matter", "white_matter"):
        if k in material_idx:
            brain_mask |= seg_arr == material_idx[k]
    if brain_mask.sum() == 0:
        brain_mask = seg_arr == material_idx["tissue"]
    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"  Target (brain center): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    # -----------------------------------------------------------------------
    # Build + position array
    # -----------------------------------------------------------------------
    print("\n[3] Building hemispherical array...")
    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    if skull_indices.size == 0:
        print("ERROR: no skull voxels; aborting.")
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
    print(f"  Approach axis: {dim_names[approach_axis]} (auto-detected)")

    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )

    # Rotation from local z-axis to approach_dir
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

    # -----------------------------------------------------------------------
    # Build sim grid
    # -----------------------------------------------------------------------
    print("\n[4] Building sim grid...")
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

    from scipy.interpolate import RegularGridInterpolator
    orig_coords_list = [volume.coords[d].to_numpy() for d in volume.dims]
    interp = RegularGridInterpolator(
        orig_coords_list, volume.to_numpy(),
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

    sim_params = seg_method.seg_params(sim_volume)

    # Replace air with water (same as run_gladys_nnunet_subject.py)
    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask = sim_seg_arr == material_idx["air"]
    skull_mask_sim = sim_seg_arr == material_idx["skull"]
    water_mat = seg_method.materials["water"]
    n_skull = int(skull_mask_sim.sum())
    total_vox = sim_seg_arr.size
    pct_skull = 100.0 * n_skull / total_vox
    print(f"  Bone fraction: {pct_skull:.2f}% ({n_skull:,d}/{total_vox:,d} voxels)")
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation

    # -----------------------------------------------------------------------
    # Compute timing parameters
    # -----------------------------------------------------------------------
    target = Point(position=target_mm.copy(), id="brain_center",
                   name="Brain Center Target", units="mm")

    from openlifu.sim.sim_setup import SimSetup
    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=CFL,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    t_end = (max_dist_mm * 1e-3) / C0 * T_END_SAFETY
    c_max = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max

    apod = np.ones(arr.numelements())
    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, source_method="point_source",
    )

    # -----------------------------------------------------------------------
    # Compute THREE delay sets
    # -----------------------------------------------------------------------
    print("\n[5] Computing geometric delays (Direct)...")
    direct = Direct(c0=C0)
    delays_geo = direct.calc_delays(arr, target, sim_params)
    print(f"  Geometric delay range: {delays_geo.min()*1e6:.2f} to {delays_geo.max()*1e6:.2f} us")

    print("\n[6] Computing SimulationCorrected delays (peak_method='first_arrival')...")
    sc_first = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True, peak_method="first_arrival")
    t0 = time.time()
    with gpu_flock():
        delays_first = sc_first.calc_delays(arr, target, sim_params)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  first_arrival delay range: {delays_first.min()*1e6:.2f} to {delays_first.max()*1e6:.2f} us")

    print("\n[7] Computing SimulationCorrected delays (peak_method='argmax')...")
    sc_argmax = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True, peak_method="argmax")
    t0 = time.time()
    with gpu_flock():
        delays_argmax = sc_argmax.calc_delays(arr, target, sim_params)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  argmax delay range: {delays_argmax.min()*1e6:.2f} to {delays_argmax.max()*1e6:.2f} us")

    # Compare delay differences
    delta_fa_geo = delays_first - delays_geo
    delta_am_geo = delays_argmax - delays_geo
    delta_fa_am = delays_first - delays_argmax
    print("\n  Delay differences (us):")
    print(f"    first_arrival - geometric: mean={delta_fa_geo.mean()*1e6:+.3f}, "
          f"std={delta_fa_geo.std()*1e6:.3f}, max|delta|={np.abs(delta_fa_geo).max()*1e6:.3f}")
    print(f"    argmax - geometric:        mean={delta_am_geo.mean()*1e6:+.3f}, "
          f"std={delta_am_geo.std()*1e6:.3f}, max|delta|={np.abs(delta_am_geo).max()*1e6:.3f}")
    print(f"    first_arrival - argmax:    mean={delta_fa_am.mean()*1e6:+.3f}, "
          f"std={delta_fa_am.std()*1e6:.3f}, max|delta|={np.abs(delta_fa_am).max()*1e6:.3f}")

    # -----------------------------------------------------------------------
    # Run THREE forward sims (all through skull)
    # -----------------------------------------------------------------------
    sims = [
        ("geometric",     delays_geo),
        ("first_arrival", delays_first),
        ("argmax",        delays_argmax),
    ]

    forward_results = {}
    for i, (label, delays) in enumerate(sims):
        print(f"\n[{8+i}] Forward sim: {label} delays + skull...")
        t0 = time.time()
        with gpu_flock():
            result = run_simulation(params=sim_params, delays=delays,
                                    ref_values_only=False, **common_kwargs)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        pmax = result["p_max"].to_numpy()
        cx, cy, cz = sim_coord_arrays
        tidx = tuple(
            int(np.argmin(np.abs(sim_coord_arrays[ax] - target_mm[ax])))
            for ax in range(3)
        )
        p_at_target = float(pmax[tidx])
        raw_max = float(pmax.max())
        idx_max = np.unravel_index(pmax.argmax(), pmax.shape)
        max_loc = np.array([float(cx[idx_max[0]]), float(cy[idx_max[1]]), float(cz[idx_max[2]])])
        err_mm = float(np.linalg.norm(max_loc - target_mm))
        print(f"  p@target={p_at_target:.4g} Pa, raw_max={raw_max:.4g} Pa (offset={err_mm:.1f} mm)")
        forward_results[label] = {
            "result": result,
            "p_at_target": p_at_target,
            "raw_max": raw_max,
            "max_offset_mm": err_mm,
        }

    # -----------------------------------------------------------------------
    # Run THREE time-gated probes
    # -----------------------------------------------------------------------
    probe_results = {}
    for i, (label, delays) in enumerate(sims):
        print(f"\n[{11+i}] Probe: {label}...")
        probe = _run_target_probe(
            arr=arr, params=sim_params, delays=delays, apod=apod,
            target_mm=target_mm, aperture_center_mm=aperture_center_mm,
            common_kwargs=common_kwargs, ref_values_only=False,
            label=label,
        )
        probe_results[label] = probe
        print(f"  p_focal_window={probe['p_focal_window_Pa']:.4g} Pa, "
              f"p_allt={probe['p_allt_Pa']:.4g} Pa, "
              f"peak_time={probe['peak_time_s']*1e6:.2f} us")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"RESULTS: first_arrival vs argmax comparison | subject={subj}")
    print("=" * 72)

    p_geo = probe_results["geometric"]["p_focal_window_Pa"]
    p_fa = probe_results["first_arrival"]["p_focal_window_Pa"]
    p_am = probe_results["argmax"]["p_focal_window_Pa"]

    def _ratio(val, base):
        if np.isfinite(val) and np.isfinite(base) and base > 0:
            return val / base
        return float("nan")

    def _db(val, base):
        r = _ratio(val, base)
        if np.isfinite(r) and r > 0:
            return 20.0 * np.log10(r)
        return float("nan")

    print(f"\n  {'Method':<20s} {'p_focal_window (Pa)':>20s} {'ratio vs geometric':>20s} {'dB vs geometric':>18s}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*18}")
    for label in ["geometric", "first_arrival", "argmax"]:
        p = probe_results[label]["p_focal_window_Pa"]
        r = _ratio(p, p_geo)
        db = _db(p, p_geo)
        print(f"  {label:<20s} {p:>20.4f} {r:>20.4f} {db:>+18.2f}")

    print(f"\n  first_arrival vs argmax:")
    r_fa_am = _ratio(p_fa, p_am)
    db_fa_am = _db(p_fa, p_am)
    print(f"    ratio = {r_fa_am:.4f}, delta = {db_fa_am:+.2f} dB")
    if r_fa_am > 1.0:
        print(f"    first_arrival WINS by {(r_fa_am - 1)*100:.1f}%")
    elif r_fa_am < 1.0:
        print(f"    argmax WINS by {(1/r_fa_am - 1)*100:.1f}%")
    else:
        print(f"    TIE")

    # Also report p_at_target from forward sims (pmax grid, not time-gated)
    print(f"\n  Forward sim p@target (pmax grid):")
    for label in ["geometric", "first_arrival", "argmax"]:
        fr = forward_results[label]
        r = _ratio(fr["p_at_target"], forward_results["geometric"]["p_at_target"])
        print(f"    {label:<20s} p@target={fr['p_at_target']:.4g} Pa, "
              f"raw_max={fr['raw_max']:.4g} Pa (offset={fr['max_offset_mm']:.1f} mm), "
              f"ratio_vs_geo={r:.4f}")

    t_elapsed = time.time() - t_total
    print(f"\nTotal runtime: {t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")

    # Machine-parseable summary
    print(
        f"\nDELAY_COMPARE subject={subj} "
        f"p_fw_geometric={p_geo:.6g} "
        f"p_fw_first_arrival={p_fa:.6g} "
        f"p_fw_argmax={p_am:.6g} "
        f"ratio_fa_vs_geo={_ratio(p_fa, p_geo):.6f} "
        f"ratio_am_vs_geo={_ratio(p_am, p_geo):.6f} "
        f"ratio_fa_vs_am={_ratio(p_fa, p_am):.6f} "
        f"db_fa_vs_geo={_db(p_fa, p_geo):+.3f} "
        f"db_am_vs_geo={_db(p_am, p_geo):+.3f} "
        f"db_fa_vs_am={_db(p_fa, p_am):+.3f} "
        f"delay_diff_fa_am_mean_us={delta_fa_am.mean()*1e6:+.4f} "
        f"delay_diff_fa_am_std_us={delta_fa_am.std()*1e6:.4f} "
        f"delay_diff_fa_am_maxabs_us={np.abs(delta_fa_am).max()*1e6:.4f} "
        f"bone_pct={pct_skull:.3f} "
        f"runtime_s={t_elapsed:.0f}"
    )


if __name__ == "__main__":
    main()
