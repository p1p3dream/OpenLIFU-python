#!/usr/bin/env python3
"""Parameterized GLADYS-nnU-Net sim for Birnbaum full-head subjects.

Forked from run_gladys_nnunet.py (which was hardcoded to GU008). Adds:
  - --subject <SubjectID> flag to derive MRI + label paths
  - Inline focal-gain metric (p@target vs aperture-band mean) at 10 mm radius
  - Time-gated probe (sparse sensor) with BOTH a geometric-gate and a
    water-calibrated-gate p_focal_window@target. The water-calibrated gate
    uses the peak time of the target sensor in the homogeneous-water sim as
    the gate center for the corresponding skull sims' target sensor.
  - OUTPUT_TAG env var to prefix output filenames (so batch reruns don't
    clobber prior results).
  - Runs SIM C (water) FIRST so its target-peak time is available for the
    skull sim probes.
  - A final one-line machine-parseable summary:
      SUBJECT_SUMMARY subject=<ID> bone_pct=<X> skull_path_near=<Y>
      p_water=<Z> p_skull=<W> p_geom_skull=<V>
      gain_vs_mean_water=<G> gain_vs_mean_skull=<H> atten_db=<A>
      p_fw_geom_water=<> p_fw_geom_skull_corr=<> p_fw_geom_skull_geom=<>
      p_fw_water_skull_corr=<> p_fw_water_skull_geom=<>
      t_water_peak_us=<> tof_geom_us=<>
      atten_db_fw_geom=<> atten_db_fw_water=<>

This script is intentionally NOT committed; it's a local batch-driver tool.
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
# Workaround for kwave v3 logging bug (same as parent script)
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

# Load the standalone per-voxel water-gate helpers by path. The helper module
# documents this importlib pattern; we use it to avoid requiring scripts/ on
# sys.path and to keep the helper a standalone file (no package deps).
import importlib.util as _ilu
_helper_spec = _ilu.spec_from_file_location(
    "_probe_helpers",
    os.path.join(os.path.dirname(__file__), "_probe_helpers.py"),
)
_probe_helpers = _ilu.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_probe_helpers)
per_voxel_water_calibrated_gate_center = _probe_helpers.per_voxel_water_calibrated_gate_center

# Load _gpu_flock the same way (standalone script module, no package deps).
_gpu_flock_spec = _ilu.spec_from_file_location(
    "_gpu_flock",
    os.path.join(os.path.dirname(__file__), "_gpu_flock.py"),
)
_gpu_flock_mod = _ilu.module_from_spec(_gpu_flock_spec)
_gpu_flock_spec.loader.exec_module(_gpu_flock_mod)
gpu_flock = _gpu_flock_mod.gpu_flock

from openlifu.bf.delay_methods.complex_weighted import ComplexWeighted
from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.seg.material import MATERIALS, MATERIALS_TWO_CLASS_BONE, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.seg_methods.nnunet_seg import LABEL_MAP_FULLHEAD, LABEL_MAP_FULLHEAD_TWO_CLASS_BONE
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
APERTURE_BAND_RADIUS_MM = 10.0

ENABLE_TIMEGATED_PROBE = os.environ.get("OPENLIFU_DISABLE_TIMEGATED_PROBE", "") != "1"
EXPANDED_TARGET_PROBE = os.environ.get("EXPANDED_TARGET_PROBE", "") == "1"
# Half-extent of the cube (mm) around target when EXPANDED_TARGET_PROBE=1.
# At 0.5 mm grid spacing, 5 mm half-extent -> 11x11x11 = 1331 voxels.
EXPANDED_TARGET_HALF_MM = float(os.environ.get("EXPANDED_TARGET_HALF_MM", "5.0"))
# Escape hatch: if set to "1", cube voxels reuse the SINGLE target-centered
# water gate (legacy behavior). When unset/0 (default), cube voxels use a
# per-voxel water-calibrated gate per Codex's recipe:
#     t_gate(v) = t_water_peak(target) + (|aperture - v| - |aperture - target|) / c0
# The legacy mode is retained for A/B comparison against the new metric.
USE_TARGET_GATE_FOR_CUBE = os.environ.get("USE_TARGET_GATE_FOR_CUBE", "") == "1"
# Optional cap on number of cube voxels whose per-voxel metadata is written
# into the sidecar JSON. When the cube voxel count exceeds this cap, the
# sidecar is trimmed to the named sensors plus the single spatial-max cube
# voxel. Cube aggregate stats (spatial_max_*, cube_n_sensors) are still
# computed in memory from the full cube before trimming.
_sidecar_cap_raw = os.environ.get("SIDECAR_MAX_VOXELS", "").strip()
SIDECAR_MAX_VOXELS: int | None = int(_sidecar_cap_raw) if _sidecar_cap_raw else None
# Stride for cube probe sampling. 1 = every voxel (default). 2 = every other
# voxel in each axis (1/8 the sensor count for 3D cube). Useful for large
# half-extents where the full-density probe sensor tensor would exceed
# available RAM. Center voxel (ix=iy=iz=0) is always included since 0 % s == 0.
_stride_raw = os.environ.get("CUBE_PROBE_STRIDE", "1").strip()
CUBE_PROBE_STRIDE: int = max(1, int(_stride_raw)) if _stride_raw else 1

# Delay method selector. Mirrors run_gladys_nnunet.py. "simulation_corrected"
# (default) preserves previous behavior. "complex_weighted" swaps in
# ComplexWeighted (narrowband complex weights) which returns (delays, apod)
# via calc_delays_and_apod; the returned apod is multiplied into the
# (uniform) apod base.
DELAY_METHOD = os.environ.get("DELAY_METHOD", "simulation_corrected").strip().lower()
if DELAY_METHOD not in ("simulation_corrected", "complex_weighted"):
    raise ValueError(
        f"DELAY_METHOD must be 'simulation_corrected' or 'complex_weighted', got '{DELAY_METHOD}'"
    )
CW_NORM = os.environ.get("CW_NORM", "max").strip().lower()
if CW_NORM not in ("max", "sum", "rms"):
    raise ValueError(f"CW_NORM must be max/sum/rms, got '{CW_NORM}'")

# --- Bounded-shell spatial probe ---------------------------------------------
# When PROBE_SHELL_INNER_MM and PROBE_SHELL_OUTER_MM are both set (and outer
# > inner), replace the cube probe with a spherical shell: include voxels
# satisfying `inner < ||v - target||_mm <= outer`. This removes cube-corner
# noise that causes the `max|p_fw|` metric to drift with cube size on
# reverberant skull sims (GU010). Default outer falls back to
# EXPANDED_TARGET_HALF_MM for backwards-compat cube behavior.
_shell_inner_raw = os.environ.get("PROBE_SHELL_INNER_MM", "").strip()
_shell_outer_raw = os.environ.get("PROBE_SHELL_OUTER_MM", "").strip()
PROBE_SHELL_INNER_MM: float | None = (
    float(_shell_inner_raw) if _shell_inner_raw else None
)
PROBE_SHELL_OUTER_MM: float | None = (
    float(_shell_outer_raw) if _shell_outer_raw else None
)
# Shell mode is active only if outer is explicitly provided AND outer > inner
# (inner defaults to 0.0 if outer alone is set).
if PROBE_SHELL_OUTER_MM is not None:
    _shell_inner_eff = PROBE_SHELL_INNER_MM if PROBE_SHELL_INNER_MM is not None else 0.0
    USE_SHELL_PROBE = PROBE_SHELL_OUTER_MM > _shell_inner_eff
else:
    USE_SHELL_PROBE = False


def _default_fullhead_materials() -> dict[str, Material]:
    m = MATERIALS.copy()
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


def _default_fullhead_materials_two_class_bone() -> dict[str, Material]:
    m = MATERIALS_TWO_CLASS_BONE.copy()
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


def compute_focal_gain(pmax, coord_arrays_xyz, target_mm, positions_world,
                        band_radius_mm=10.0):
    """Return dict with p_at_target, p_aperture_mean, gain_vs_mean."""
    cx, cy, cz = coord_arrays_xyz
    xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
    coord_stack = np.stack([xx, yy, zz], axis=-1)
    tidx = tuple(
        int(np.argmin(np.abs(coord_arrays_xyz[ax] - target_mm[ax])))
        for ax in range(3)
    )
    p_at_target = float(pmax[tidx])
    r2 = band_radius_mm ** 2
    band_mask = np.zeros(pmax.shape, dtype=bool)
    for pos in positions_world:
        d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
        band_mask |= d2 <= r2
    n_band = int(band_mask.sum())
    if n_band == 0:
        return {
            "p_at_target": p_at_target, "p_aperture_mean": float("nan"),
            "gain_vs_mean": float("nan"), "n_band": 0,
        }
    vals = pmax[band_mask]
    p_ap_mean = float(vals.mean())
    gain = p_at_target / p_ap_mean if p_ap_mean > 0 else float("nan")
    return {
        "p_at_target": p_at_target, "p_aperture_mean": p_ap_mean,
        "gain_vs_mean": gain, "n_band": n_band,
    }


# ---------------------------------------------------------------------------
# Time-gated probe (mirrored from run_gladys_nnunet.py, with the addition of
# a water-calibrated gate center for the target sensor).
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


def _build_sensor_mask(
    params: xa.Dataset,
    target_mm: np.ndarray,
    aperture_center_mm: np.ndarray,
    expanded_target_cube: bool = False,
    cube_half_mm: float = 5.0,
    cube_stride: int = 1,
    shell_inner_mm: float | None = None,
    shell_outer_mm: float | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[int, int, int]], list[np.ndarray]]:
    axis_vec = aperture_center_mm - target_mm
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 1e-6:
        raise ValueError("aperture center coincides with target; cannot build sensor mask")
    axis_unit = axis_vec / axis_len

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

    sensor_world_mm: list[tuple[str, np.ndarray]] = [
        ("target",        target_mm.copy()),
        ("aperture_ctr",  aperture_center_mm.copy()),
        ("midway",        target_mm + 0.5 * axis_vec),
        ("quarterway",    target_mm + 0.75 * axis_vec),
        ("off_axis_15mm", target_mm + 15.0 * perp),
    ]

    # Optionally add a cube (or shell) of sensors centered on target.
    use_shell = (
        expanded_target_cube
        and shell_outer_mm is not None
        and shell_outer_mm > (shell_inner_mm if shell_inner_mm is not None else 0.0)
    )
    if expanded_target_cube:
        # Figure out grid spacing along each axis from params
        sim_dim_names = list(params.dims)
        sim_coord_params = {d: params.coords[d].to_numpy() for d in sim_dim_names}
        # Use x,y,z direct (not sim-dim order) for cube construction
        cx = params.coords["x"].to_numpy()
        cy = params.coords["y"].to_numpy()
        cz = params.coords["z"].to_numpy()
        dx = float(cx[1] - cx[0]) if len(cx) > 1 else 0.5
        dy = float(cy[1] - cy[0]) if len(cy) > 1 else 0.5
        dz = float(cz[1] - cz[0]) if len(cz) > 1 else 0.5

        existing_positions = {tuple(np.round(pos, 6)) for _, pos in sensor_world_mm}
        stride = max(1, int(cube_stride))

        if use_shell:
            # Shell mode: enumerate a bounding cube of half-extent = outer,
            # then keep only voxels with inner < ||v - target|| <= outer.
            inner = float(shell_inner_mm) if shell_inner_mm is not None else 0.0
            outer = float(shell_outer_mm)
            nx = int(np.ceil(outer / abs(dx)))
            ny = int(np.ceil(outer / abs(dy)))
            nz_ = int(np.ceil(outer / abs(dz)))
            name_prefix = "shell"
            inner2 = inner * inner
            outer2 = outer * outer
            for ix in range(-nx, nx + 1):
                if ix % stride != 0:
                    continue
                for iy in range(-ny, ny + 1):
                    if iy % stride != 0:
                        continue
                    for iz in range(-nz_, nz_ + 1):
                        if iz % stride != 0:
                            continue
                        dvx = ix * dx
                        dvy = iy * dy
                        dvz = iz * dz
                        r2 = dvx * dvx + dvy * dvy + dvz * dvz
                        # inner < r <= outer ; always retain target (ix=iy=iz=0)
                        # when inner == 0 (so r2 == 0 passes through when inner2 == 0).
                        if not (r2 > inner2 and r2 <= outer2):
                            # include center voxel if inner=0 (target reuse)
                            if ix == 0 and iy == 0 and iz == 0 and inner <= 0.0:
                                continue  # target already present in sensor_world_mm
                            continue
                        if ix == 0 and iy == 0 and iz == 0:
                            continue  # target already present
                        pos = np.array([
                            target_mm[0] + dvx,
                            target_mm[1] + dvy,
                            target_mm[2] + dvz,
                        ])
                        key = tuple(np.round(pos, 6))
                        if key in existing_positions:
                            continue
                        existing_positions.add(key)
                        sensor_world_mm.append((f"{name_prefix}_{ix:+d}_{iy:+d}_{iz:+d}", pos))
        else:
            def _steps(d):
                return int(round(cube_half_mm / abs(d)))

            nx, ny, nz_ = _steps(dx), _steps(dy), _steps(dz)
            for ix in range(-nx, nx + 1):
                if ix % stride != 0:
                    continue
                for iy in range(-ny, ny + 1):
                    if iy % stride != 0:
                        continue
                    for iz in range(-nz_, nz_ + 1):
                        if iz % stride != 0:
                            continue
                        if ix == 0 and iy == 0 and iz == 0:
                            continue  # target already present
                        pos = np.array([
                            target_mm[0] + ix * dx,
                            target_mm[1] + iy * dy,
                            target_mm[2] + iz * dz,
                        ])
                        key = tuple(np.round(pos, 6))
                        if key in existing_positions:
                            continue
                        existing_positions.add(key)
                        sensor_world_mm.append((f"cube_{ix:+d}_{iy:+d}_{iz:+d}", pos))

    sim_dim_names = list(params.dims)
    sim_coord_params = {d: params.coords[d].to_numpy() for d in sim_dim_names}
    mask = np.zeros(tuple(len(sim_coord_params[d]) for d in sim_dim_names), dtype=np.int32)

    names: list[str] = []
    xyz_idx_list: list[tuple[int, int, int]] = []
    actual_list: list[np.ndarray] = []
    seen_xyz: set[tuple[int, int, int]] = set()
    for name, pos_mm in sensor_world_mm:
        idx_in_params: list[int] = []
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
            # Skip duplicates caused by grid snapping (e.g., two cube voxels colliding)
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
    """Run the sparse-sensor probe and compute time-gated metrics.

    Returns per-sensor metrics + summary. For the target sensor, always reports
    `p_focal_window_Pa` (geometric-gate using tof @ C0=1500), and additionally
    `p_focal_window_water_gate_Pa` when `target_gate_center_s` is supplied
    (this is the peak time extracted from the water sim's target time series,
    used as the gate center for skull sims). Other sensors always use their
    geometric TOF as the gate center.
    """
    try:
        mask, names, xyz_idx_list, actual_list = _build_sensor_mask(
            params, target_mm, aperture_center_mm,
            expanded_target_cube=EXPANDED_TARGET_PROBE,
            cube_half_mm=EXPANDED_TARGET_HALF_MM,
            cube_stride=CUBE_PROBE_STRIDE,
            shell_inner_mm=PROBE_SHELL_INNER_MM if USE_SHELL_PROBE else None,
            shell_outer_mm=PROBE_SHELL_OUTER_MM if USE_SHELL_PROBE else None,
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

        pulse_dur = CYCLES / FREQ_HZ  # seconds
        Nt = p_sensor.shape[0]
        t_axis = np.arange(Nt) * dt_probe

        # Build O(1) lookup from xyz_idx tuple -> column index in p_sensor.
        # This replaces an earlier O(N) numpy scan that caused O(N^2) total
        # post-processing time for large cubes (e.g. 25mm/30mm spatial sweeps).
        _col_index: dict[tuple[int, int, int], int] = {}
        for _ci in range(xyz_order.shape[0]):
            _key = (int(xyz_order[_ci, 0]), int(xyz_order[_ci, 1]), int(xyz_order[_ci, 2]))
            if _key not in _col_index:
                _col_index[_key] = _ci

        def _col_for(xyz_idx_tuple: tuple[int, int, int]) -> int | None:
            return _col_index.get(tuple(int(v) for v in xyz_idx_tuple))

        per_sensor = {}
        # Named sensors get full detail (including time series). Cube sensors
        # only get summary scalars, to keep sidecar size manageable.
        named_sensors = {
            "target", "aperture_ctr", "midway", "quarterway", "off_axis_15mm",
        }
        for name, xyz_idx, actual in zip(names, xyz_idx_list, actual_list):
            is_cube = name.startswith("cube_") or name.startswith("shell_")
            col = _col_for(xyz_idx)
            d_m = float(np.linalg.norm(actual - aperture_center_mm)) * 1e-3
            tof_s = d_m / C0  # geometric TOF @ C0 (water speed)
            rec: dict[str, Any] = {
                "sensor_world_mm": actual.tolist(),
                "xyz_idx": list(xyz_idx),
                "tof_s": tof_s,
                "tof_us": tof_s * 1e6,
                "dist_from_target_mm": float(np.linalg.norm(actual - target_mm)),
                "dist_from_aperture_ctr_mm": float(np.linalg.norm(actual - aperture_center_mm)),
            }
            if col is None:
                rec.update({
                    "column": None,
                    "p_focal_window_Pa": float("nan"),
                    "p_focal_window_water_gate_Pa": float("nan"),
                    "p_allt_Pa": float("nan"),
                    "ratio": float("nan"),
                    "peak_time_s": float("nan"),
                    "note": "voxel collision; no dedicated column",
                })
                if not is_cube:
                    rec["p_t"] = []
            else:
                p_t = p_sensor[:, col]
                p_abs = np.abs(p_t)
                p_allt = float(p_abs.max())
                peak_time_s = float(t_axis[int(np.argmax(p_abs))]) if Nt > 0 else float("nan")

                # Geometric gate (existing metric) - per-sensor TOF
                t_lo = tof_s - 2.0 * pulse_dur
                t_hi = tof_s + 2.0 * pulse_dur
                in_win = (t_axis >= t_lo) & (t_axis <= t_hi)
                p_fw_geom = float(p_abs[in_win].max()) if in_win.any() else float("nan")

                # Water-calibrated gate:
                #   - For the target sensor: use the target water-peak time directly.
                #   - For cube_* sensors (NEW, default): use a per-voxel
                #     water-calibrated gate center that accounts for the
                #     geometric TOF difference between the voxel and target:
                #         t_gate(v) = t_water_peak(target)
                #                   + (|aperture - v| - |aperture - target|) / c0
                #     Without this correction, voxels far from target are
                #     scored by random scatter arriving in the target's gate
                #     rather than by focal-arrival energy, which produced
                #     monotonic offset drift toward the cube face in the
                #     GU010 spatial sweep (5/10/15/20/25 mm half-extent).
                #   - Escape hatch USE_TARGET_GATE_FOR_CUBE=1 reverts to the
                #     old single-gate behavior for A/B comparison.
                p_fw_water = float("nan")
                if target_gate_center_s is not None and (name == "target" or is_cube):
                    if is_cube and not USE_TARGET_GATE_FOR_CUBE:
                        gate_center_s = per_voxel_water_calibrated_gate_center(
                            sensor_world_mm=actual,
                            target_world_mm=target_mm,
                            aperture_center_world_mm=aperture_center_mm,
                            t_water_peak_target=float(target_gate_center_s),
                            c0_mps=C0,
                        )
                    else:
                        gate_center_s = float(target_gate_center_s)
                    t_lo_w = gate_center_s - 2.0 * pulse_dur
                    t_hi_w = gate_center_s + 2.0 * pulse_dur
                    in_win_w = (t_axis >= t_lo_w) & (t_axis <= t_hi_w)
                    p_fw_water = float(p_abs[in_win_w].max()) if in_win_w.any() else float("nan")

                ratio = (p_fw_geom / p_allt) if (p_allt > 0 and np.isfinite(p_fw_geom)) else float("nan")
                rec.update({
                    "column": col,
                    "p_focal_window_Pa": p_fw_geom,
                    "p_focal_window_water_gate_Pa": p_fw_water,
                    "p_allt_Pa": p_allt,
                    "ratio": ratio,
                    "peak_time_s": peak_time_s,
                    "note": "",
                })
                if not is_cube:
                    rec["p_t"] = p_t.astype(np.float32).tolist()
            per_sensor[name] = rec

        result = {
            "sim_label": sim_label,
            "ref_values_only": bool(ref_values_only),
            "dt_s": dt_probe,
            "n_timesteps": int(Nt),
            "pulse_duration_s": pulse_dur,
            "focal_window_halfwidth_s": 2.0 * pulse_dur,
            "c0_m_s": C0,
            "target_mm": target_mm.tolist(),
            "aperture_center_mm": aperture_center_mm.tolist(),
            "sensors": per_sensor,
            "probe_runtime_s": probe_secs,
            "target_gate_center_s": target_gate_center_s,
            "expanded_target_probe": bool(EXPANDED_TARGET_PROBE),
            "cube_half_extent_mm": float(EXPANDED_TARGET_HALF_MM) if EXPANDED_TARGET_PROBE else None,
            "probe_stride": int(CUBE_PROBE_STRIDE) if EXPANDED_TARGET_PROBE else 1,
            "shell_probe": bool(USE_SHELL_PROBE),
            "shell_inner_mm": float(PROBE_SHELL_INNER_MM) if (USE_SHELL_PROBE and PROBE_SHELL_INNER_MM is not None) else None,
            "shell_outer_mm": float(PROBE_SHELL_OUTER_MM) if (USE_SHELL_PROBE and PROBE_SHELL_OUTER_MM is not None) else None,
            "cube_gate_mode": (
                "per_voxel_water_calibrated"
                if (target_gate_center_s is not None and not USE_TARGET_GATE_FOR_CUBE)
                else ("single_target_water_gate" if target_gate_center_s is not None else "geometric_only")
            ),
        }
        tgt = per_sensor.get("target", {})
        result["p_focal_window_at_target_Pa"] = tgt.get("p_focal_window_Pa", float("nan"))
        result["p_focal_window_water_gate_at_target_Pa"] = tgt.get("p_focal_window_water_gate_Pa", float("nan"))
        result["p_allt_at_target_Pa"] = tgt.get("p_allt_Pa", float("nan"))
        result["ratio_at_target"] = tgt.get("ratio", float("nan"))
        result["target_peak_time_s"] = tgt.get("peak_time_s", float("nan"))
        result["target_tof_s"] = tgt.get("tof_s", float("nan"))

        # --- Spatial search over target cube (+target itself) ----------------
        # For skull sims (target_gate_center_s provided), use water-gate values.
        # For the water sim (no gate center), use geometric-gate values.
        use_water_gate_for_cube = target_gate_center_s is not None
        cube_candidates = []
        for s_name, s_rec in per_sensor.items():
            if not (s_name == "target" or s_name.startswith("cube_") or s_name.startswith("shell_")):
                continue
            if use_water_gate_for_cube:
                val = s_rec.get("p_focal_window_water_gate_Pa", float("nan"))
            else:
                val = s_rec.get("p_focal_window_Pa", float("nan"))
            if val is None or not np.isfinite(val):
                continue
            cube_candidates.append((s_name, float(val), s_rec.get("sensor_world_mm")))
        result["cube_n_sensors"] = len(cube_candidates)
        result["cube_uses_water_gate"] = bool(use_water_gate_for_cube)
        if cube_candidates:
            best_name, best_val, best_pos = max(cube_candidates, key=lambda t: t[1])
            best_pos_arr = np.array(best_pos) if best_pos is not None else np.array([float("nan")] * 3)
            offset = float(np.linalg.norm(best_pos_arr - target_mm)) if np.all(np.isfinite(best_pos_arr)) else float("nan")
            result["spatial_max_sensor_name"] = best_name
            result["spatial_max_world_mm"] = best_pos_arr.tolist()
            result["spatial_max_offset_from_target_mm"] = offset
            if use_water_gate_for_cube:
                result["spatial_max_p_focal_window_water_gate_Pa"] = best_val
                result["spatial_max_p_focal_window_Pa"] = float("nan")
            else:
                result["spatial_max_p_focal_window_Pa"] = best_val
                result["spatial_max_p_focal_window_water_gate_Pa"] = float("nan")
        else:
            result["spatial_max_sensor_name"] = None
            result["spatial_max_world_mm"] = None
            result["spatial_max_offset_from_target_mm"] = float("nan")
            result["spatial_max_p_focal_window_Pa"] = float("nan")
            result["spatial_max_p_focal_window_water_gate_Pa"] = float("nan")

        # --- Optional sidecar trimming for very large cubes ------------------
        cube_voxel_count = sum(
            1 for k in per_sensor.keys()
            if k.startswith("cube_") or k.startswith("shell_")
        )
        result["sidecar_cube_voxels_original"] = int(cube_voxel_count)
        result["sidecar_trimmed"] = False
        if SIDECAR_MAX_VOXELS is not None and cube_voxel_count > SIDECAR_MAX_VOXELS:
            keep_name = result.get("spatial_max_sensor_name")
            trimmed: dict = {}
            for k, v in per_sensor.items():
                if not (k.startswith("cube_") or k.startswith("shell_")):
                    trimmed[k] = v
                elif keep_name is not None and k == keep_name:
                    trimmed[k] = v
            logger.info(
                "    [%s] sidecar trimmed: %d cube/shell voxels -> %d kept (threshold=%d); spatial max @ %s retained",
                sim_label, cube_voxel_count,
                sum(1 for k in trimmed if k.startswith("cube_") or k.startswith("shell_")),
                SIDECAR_MAX_VOXELS, keep_name,
            )
            result["sensors"] = trimmed
            result["sidecar_trimmed"] = True
        return result
    except Exception as exc:
        logger.warning(
            "Time-gated probe failed for %s (%s); continuing without it.",
            sim_label, exc, exc_info=True,
        )
        return None


def _save_timegated_json(result: dict | None, out_path: Path) -> None:
    if result is None:
        return
    try:
        with open(out_path, "w") as f:
            json.dump(result, f)
        logger.info("    Saved time-gated sidecar: %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save time-gated sidecar %s: %s", out_path, exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True,
                    help="Subject ID, e.g., GU002, GU010, NC004")
    ap.add_argument("--mri-path", default=None,
                    help="Override MRI path (default: birnbaum <subject>_deface.nii)")
    ap.add_argument("--label-path", default=None,
                    help="Override label path (default: ~/Data/openlifu-validation/results/<subject>_nnunet_labels.nii.gz)")
    ap.add_argument("--results-dir", default=None,
                    help="Results dir (default: ~/Data/openlifu-validation/results)")
    ap.add_argument("--output-tag", default=None,
                    help="Prefix tag for output filenames (overrides OUTPUT_TAG env var)")
    ap.add_argument("--orient-theta", type=float, default=None,
                    help="Polar angle (degrees) of array approach direction. "
                         "theta=0 is +z (superior), theta=90 is in the xy-plane. "
                         "Must be used together with --orient-phi.")
    ap.add_argument("--orient-phi", type=float, default=None,
                    help="Azimuthal angle (degrees) of array approach direction. "
                         "phi=0 is +x, phi=90 is +y. Direction is from brain center "
                         "to array aperture center. Must be used together with --orient-theta.")
    ap.add_argument("--bone-model", choices=["single", "two_class"], default="single",
                    help="Bone material model: 'single' (homogeneous skull) or "
                         "'two_class' (cortical + trabecular). Two-class requires "
                         "label NIfTIs with label 5=cortical, 7=trabecular.")
    args = ap.parse_args()

    if (args.orient_theta is None) != (args.orient_phi is None):
        ap.error("--orient-theta and --orient-phi must be provided together.")

    subj = args.subject
    mri_path = Path(args.mri_path) if args.mri_path else (
        Path.home() / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
        / "Anonymized_Subjects" / "T1-Weighted MRI" / f"{subj}_deface.nii"
    )
    label_path = Path(args.label_path) if args.label_path else (
        Path.home() / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"
    )
    results_dir = Path(args.results_dir) if args.results_dir else (
        Path.home() / "Data/openlifu-validation/results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    output_tag = args.output_tag if args.output_tag is not None else os.environ.get("OUTPUT_TAG", "")

    t_total = time.time()
    print("=" * 72)
    print(f"GLADYS nnU-Net sim | subject={subj} | output_tag='{output_tag}'")
    print(f"  MRI:    {mri_path}")
    print(f"  Labels: {label_path}")
    print("=" * 72)
    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}")
        sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}")
        sys.exit(1)

    volume = load_nifti_as_xarray(mri_path)
    if args.bone_model == "two_class":
        seg_method = PreSegmented(
            label_nifti_path=str(label_path),
            nnunet_label_map=dict(LABEL_MAP_FULLHEAD_TWO_CLASS_BONE),
            materials=_default_fullhead_materials_two_class_bone(),
        )
    else:
        seg_method = PreSegmented(label_nifti_path=str(label_path))
    bone_model = args.bone_model
    print(f"MRI shape: {volume.shape}, label shape: {seg_method._labels.shape}")
    print(f"Bone model: {bone_model}")

    lab_arr = seg_method._labels.to_numpy()
    print("nnU-Net label histogram (raw label file):")
    for nn_label, mat_key in sorted(seg_method.nnunet_label_map.items()):
        n = int((lab_arr == nn_label).sum())
        pct = 100.0 * n / lab_arr.size
        print(f"  {nn_label} ({mat_key}): {n:,d} ({pct:.2f}%)")

    # Segment MRI grid to find target
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
    print(f"Target (brain center): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    if bone_model == "two_class":
        skull_mask = (
            (seg_arr == material_idx["cortical_bone"])
            | (seg_arr == material_idx["trabecular_bone"])
        )
    else:
        skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    if skull_indices.size == 0:
        print("ERROR: no bone voxels in segmentation; aborting.")
        print(f"SUBJECT_SUMMARY subject={subj} STATUS=no_skull")
        sys.exit(2)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])
    max_skull_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])

    if args.orient_theta is not None and args.orient_phi is not None:
        # Manual override: build approach direction from spherical angles
        theta_rad = np.radians(args.orient_theta)
        phi_rad = np.radians(args.orient_phi)
        approach_dir = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad),
        ])
        orient_label = (
            f"theta={args.orient_theta:.1f} deg, phi={args.orient_phi:.1f} deg "
            f"(manual override)"
        )
    else:
        # Auto-detect: axis with maximum skull extent
        approach_axis = int(np.argmax(max_skull_per_axis))
        approach_dir = np.zeros(3)
        approach_dir[approach_axis] = 1.0
        # Compute equivalent spherical angles for logging
        auto_theta = np.degrees(np.arccos(np.clip(approach_dir[2], -1, 1)))
        auto_phi = np.degrees(np.arctan2(approach_dir[1], approach_dir[0]))
        orient_label = (
            f"theta={auto_theta:.1f} deg, phi={auto_phi:.1f} deg "
            f"(auto-detected axis {dim_names[approach_axis]})"
        )

    print(f"Array orientation: {orient_label}")

    # Array
    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )

    # Build rotation from local z-axis to approach_dir
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

    # Build sim grid
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
    print(f"Sim grid: {grid_shape}")

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

    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask = sim_seg_arr == material_idx["air"]
    if bone_model == "two_class":
        skull_mask_sim = (
            (sim_seg_arr == material_idx["cortical_bone"])
            | (sim_seg_arr == material_idx["trabecular_bone"])
        )
    else:
        skull_mask_sim = sim_seg_arr == material_idx["skull"]
    water_mat = seg_method.materials["water"]
    n_skull = int(skull_mask_sim.sum())
    total_vox = sim_seg_arr.size
    pct_skull = 100.0 * n_skull / total_vox
    print(f"SIM grid bone fraction: {pct_skull:.2f}% ({n_skull:,d}/{total_vox:,d} voxels)")
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation

    # Skull path along target->aperture ray
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

        if bone_model == "two_class":
            _bone_indices = {material_idx["cortical_bone"], material_idx["trabecular_bone"]}
        else:
            _bone_indices = {material_idx["skull"]}

        def _skull_path(direction):
            n = int(np.ceil(probe_len_mm / step_mm)) + 1
            ts = np.linspace(0.0, probe_len_mm, n)
            pts = target_mm[None, :] + ts[:, None] * direction[None, :]
            frac = ((pts - sim_origins[None, :]) / sim_specs[None, :]).T
            sampled = map_coordinates(
                sim_seg_arr.astype(np.float32), frac, order=0,
                mode="constant", cval=-1.0,
            ).astype(np.int16)
            return int(sum(int((sampled == bi).sum()) for bi in _bone_indices)) * step_mm

        skull_path_near_mm = _skull_path(ray_unit)
        skull_path_far_mm = _skull_path(-ray_unit)
    print(f"Skull path near-side: {skull_path_near_mm:.1f} mm, far-side: {skull_path_far_mm:.1f} mm")

    target = Point(position=target_mm.copy(), id="brain_center",
                   name="Brain Center Target", units="mm")
    direct = Direct(c0=C0)
    delays_geo = direct.calc_delays(arr, target, sim_params)

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

    apod_cw: np.ndarray | None = None
    cw_apod_info: dict | None = None
    if DELAY_METHOD == "complex_weighted":
        print("\n[8] ComplexWeighted (narrowband complex weights)...")
        delay_method = ComplexWeighted(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
        t0 = time.time()
        with gpu_flock():
            delays_corrected, apod_cw = delay_method.calc_delays_and_apod(
                arr, target, sim_params, transform=None,
            )
        print(f"    ComplexWeighted done in {time.time()-t0:.1f}s")
        delays_corrected = np.asarray(delays_corrected, dtype=float)
        apod_cw = np.asarray(apod_cw, dtype=float) if apod_cw is not None else None
        if apod_cw is not None:
            apod_cw_pre = apod_cw.copy()
            if CW_NORM == "max":
                pass  # already max-normalized so max(apod) == 1
            elif CW_NORM == "sum":
                s = float(apod_cw.sum())
                if s > 0:
                    apod_cw = apod_cw * (len(apod_cw) / s)
            elif CW_NORM == "rms":
                current_sumsq = float(np.sum(apod_cw ** 2))
                if current_sumsq > 0:
                    apod_cw = apod_cw * np.sqrt(len(apod_cw) / current_sumsq)
            print(
                f"    CW_NORM={CW_NORM}: pre-norm sum={apod_cw_pre.sum():.3f}, "
                f"max={apod_cw_pre.max():.4f}; "
                f"post-norm sum={apod_cw.sum():.3f}, max={apod_cw.max():.4f}"
            )
            n_hot = int((apod_cw > 2.0).sum())
            if n_hot > 0:
                print(
                    f"    WARNING: {n_hot} element(s) have apod > 2.0 after "
                    f"CW_NORM={CW_NORM} (max={apod_cw.max():.3f}); not clamping."
                )
            thr = 0.01
            n_total_cw = int(apod_cw.size)
            n_active_cw = int((apod_cw > thr).sum())
            frac_above = n_active_cw / n_total_cw if n_total_cw > 0 else 0.0
            print(
                f"    CW apod stats: min={apod_cw.min():.4f}, max={apod_cw.max():.4f}, "
                f"mean={apod_cw.mean():.4f}, std={apod_cw.std():.4f}"
            )
            print(
                f"    CW apod active (> {thr}): {n_active_cw}/{n_total_cw} "
                f"(fraction={frac_above:.3f})"
            )
            cw_apod_info = {
                "cw_norm": CW_NORM,
                "min": float(apod_cw.min()),
                "max": float(apod_cw.max()),
                "mean": float(apod_cw.mean()),
                "std": float(apod_cw.std()),
                "apod_max": float(apod_cw.max()),
                "apod_sum": float(apod_cw.sum()),
                "pre_norm_max": float(apod_cw_pre.max()),
                "pre_norm_sum": float(apod_cw_pre.sum()),
                "threshold": thr,
                "n_active": n_active_cw,
                "n_total": n_total_cw,
                "n_hot_above_2": n_hot,
                "fraction_above_threshold": frac_above,
                "weights": apod_cw.tolist(),
            }
        else:
            print("    CW returned apod=None; treating as uniform amplitude.")
        print(
            f"    Delay range (corrected): {delays_corrected.min()*1e6:.2f} to "
            f"{delays_corrected.max()*1e6:.2f} us  (min>=0 enforced)"
        )
    else:
        print("\n[8] SimulationCorrected (phase correction)...")
        sim_corrected = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
        t0 = time.time()
        with gpu_flock():
            delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
        print(f"    Phase correction done in {time.time()-t0:.1f}s")

    apod = np.ones(arr.numelements())
    if apod_cw is not None:
        if apod_cw.shape != apod.shape:
            raise ValueError(
                f"apod_cw shape {apod_cw.shape} does not match apod shape {apod.shape}"
            )
        apod = np.asarray(apod, dtype=float) * np.asarray(apod_cw, dtype=float)
        print(
            f"    Combined apod after CW: min={apod.min():.4f}, max={apod.max():.4f}, "
            f"mean={apod.mean():.4f}, sum={apod.sum():.2f}"
        )
    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, source_method="point_source",
    )

    # -------------------------------------------------------------------
    # Run SIM C (water) FIRST so we can extract the water-target peak time
    # and use it as the gate center for SIM A and SIM B probes.
    # -------------------------------------------------------------------
    print("\n[SIM C] geometric + water  (run FIRST for gate calibration)")
    t0 = time.time()
    with gpu_flock():
        result_c = run_simulation(params=sim_params, delays=delays_geo,
                                  ref_values_only=True, **common_kwargs)
    print(f"  done in {time.time()-t0:.1f}s")
    probe_c = _run_timegated_probe(
        arr=arr, params=sim_params, delays=delays_geo, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=True,
        sim_label="SIM C (geometric+water)",
        target_gate_center_s=None,  # water itself: only geom gate
    ) if ENABLE_TIMEGATED_PROBE else None

    t_water_target_peak = None
    if probe_c is not None:
        t_water_target_peak = probe_c.get("target_peak_time_s", None)
        if t_water_target_peak is None or not np.isfinite(t_water_target_peak):
            t_water_target_peak = None
        else:
            tgt_geom_tof = probe_c.get("target_tof_s", float("nan"))
            print(
                f"  [water-gate] target peak time = {t_water_target_peak*1e6:.2f} us, "
                f"geom tof = {tgt_geom_tof*1e6:.2f} us, "
                f"delta = {(t_water_target_peak - tgt_geom_tof)*1e6:+.2f} us"
            )

    print("\n[SIM A] corrected + skull (nnU-Net)")
    t0 = time.time()
    with gpu_flock():
        result_a = run_simulation(params=sim_params, delays=delays_corrected,
                                  ref_values_only=False, **common_kwargs)
    print(f"  done in {time.time()-t0:.1f}s")
    probe_a = _run_timegated_probe(
        arr=arr, params=sim_params, delays=delays_corrected, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=False,
        sim_label="SIM A (corrected+skull)",
        target_gate_center_s=t_water_target_peak,
    ) if ENABLE_TIMEGATED_PROBE else None

    print("\n[SIM B] geometric + skull")
    t0 = time.time()
    with gpu_flock():
        result_b = run_simulation(params=sim_params, delays=delays_geo,
                                  ref_values_only=False, **common_kwargs)
    print(f"  done in {time.time()-t0:.1f}s")
    probe_b = _run_timegated_probe(
        arr=arr, params=sim_params, delays=delays_geo, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=False,
        sim_label="SIM B (geometric+skull)",
        target_gate_center_s=t_water_target_peak,
    ) if ENABLE_TIMEGATED_PROBE else None

    cx = sim_coord_arrays[0]
    cy = sim_coord_arrays[1]
    cz = sim_coord_arrays[2]

    def _stats(result, label):
        pmax = result["p_max"].to_numpy()
        fg = compute_focal_gain(pmax, (cx, cy, cz), target_mm, positions,
                                 band_radius_mm=APERTURE_BAND_RADIUS_MM)
        raw_max = float(pmax.max())
        idx = np.unravel_index(pmax.argmax(), pmax.shape)
        raw_loc = np.array([float(cx[idx[0]]), float(cy[idx[1]]), float(cz[idx[2]])])
        err = float(np.linalg.norm(raw_loc - target_mm))
        print(f"  {label}: raw_max={raw_max:.4g} Pa (err={err:.1f}mm), "
              f"p@target={fg['p_at_target']:.4g} Pa, "
              f"p_ap_mean={fg['p_aperture_mean']:.4g} Pa, "
              f"gain_vs_mean={fg['gain_vs_mean']:.3f}")
        return fg, raw_max, err

    print("\n--- FOCAL STATS ---")
    stats_c, _, _ = _stats(result_c, "C geometric+water ")
    stats_a, _, _ = _stats(result_a, "A corrected+skull")
    stats_b, _, _ = _stats(result_b, "B geometric+skull")

    p_water = stats_c["p_at_target"]
    p_skull = stats_a["p_at_target"]
    p_geom_skull = stats_b["p_at_target"]
    if p_water > 0 and p_skull > 0:
        atten_db = 20.0 * np.log10(p_water / p_skull)
    else:
        atten_db = float("nan")

    # Probe-derived attenuation
    def _db(num, den):
        if np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0:
            return 20.0 * np.log10(den / num)
        return float("nan")

    p_fw_geom_water = probe_c.get("p_focal_window_at_target_Pa", float("nan")) if probe_c else float("nan")
    p_fw_geom_skull_corr = probe_a.get("p_focal_window_at_target_Pa", float("nan")) if probe_a else float("nan")
    p_fw_geom_skull_geom = probe_b.get("p_focal_window_at_target_Pa", float("nan")) if probe_b else float("nan")
    p_fw_water_skull_corr = probe_a.get("p_focal_window_water_gate_at_target_Pa", float("nan")) if probe_a else float("nan")
    p_fw_water_skull_geom = probe_b.get("p_focal_window_water_gate_at_target_Pa", float("nan")) if probe_b else float("nan")

    atten_db_fw_geom = _db(p_fw_geom_skull_corr, p_fw_geom_water)
    atten_db_fw_water = _db(p_fw_water_skull_corr, p_fw_geom_water)

    tof_geom_s = probe_c.get("target_tof_s", float("nan")) if probe_c else float("nan")
    t_water_peak_s = t_water_target_peak if t_water_target_peak is not None else float("nan")

    print("\n--- PROBE METRICS ---")
    print(f"  p_fw_geom  water:       {p_fw_geom_water:.4g} Pa")
    print(f"  p_fw_geom  skull_corr:  {p_fw_geom_skull_corr:.4g} Pa  (atten {atten_db_fw_geom:.2f} dB)")
    print(f"  p_fw_geom  skull_geom:  {p_fw_geom_skull_geom:.4g} Pa")
    print(f"  p_fw_water skull_corr:  {p_fw_water_skull_corr:.4g} Pa  (atten {atten_db_fw_water:.2f} dB)")
    print(f"  p_fw_water skull_geom:  {p_fw_water_skull_geom:.4g} Pa")
    print(f"  t_water_peak={t_water_peak_s*1e6 if np.isfinite(t_water_peak_s) else float('nan'):.2f} us, "
          f"tof_geom={tof_geom_s*1e6 if np.isfinite(tof_geom_s) else float('nan'):.2f} us")

    # Save pmax NIfTIs with subject prefix + optional tag
    for sim_label, result, probe in [
        ("corrected", result_a, probe_a),
        ("geometric", result_b, probe_b),
        ("water", result_c, probe_c),
    ]:
        p_max_data = result["p_max"].to_numpy()
        out_affine = np.diag([
            float(cx[1] - cx[0]) if len(cx) > 1 else 1.0,
            float(cy[1] - cy[0]) if len(cy) > 1 else 1.0,
            float(cz[1] - cz[0]) if len(cz) > 1 else 1.0,
            1.0,
        ])
        out_affine[0, 3] = float(cx[0])
        out_affine[1, 3] = float(cy[0])
        out_affine[2, 3] = float(cz[0])
        out_path = results_dir / f"{subj}_{output_tag}gladys_nnunet_{sim_label}_pmax.nii.gz"
        nib.save(nib.Nifti1Image(p_max_data.astype(np.float32), out_affine), str(out_path))
        print(f"    Saved: {out_path}")
        sidecar_path = results_dir / f"{subj}_{output_tag}gladys_nnunet_{sim_label}_timegated.json"
        _save_timegated_json(probe, sidecar_path)

    t_elapsed = time.time() - t_total
    print(f"\n[{subj}] Total: {t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")

    def _fmt(v):
        try:
            if not np.isfinite(v):
                return "nan"
            return f"{v:.6g}"
        except Exception:
            return "nan"

    # Spatial-search fields (NaN if not enabled)
    sp_water_target = p_fw_geom_water  # baseline: water sim target, geom gate
    sp_water_cube_max = (
        probe_c.get("spatial_max_p_focal_window_Pa", float("nan"))
        if probe_c else float("nan")
    )
    sp_corr_target = p_fw_water_skull_corr  # skull corrected, water gate, target
    sp_corr_cube_max = (
        probe_a.get("spatial_max_p_focal_window_water_gate_Pa", float("nan"))
        if probe_a else float("nan")
    )
    sp_geom_target = p_fw_water_skull_geom
    sp_geom_cube_max = (
        probe_b.get("spatial_max_p_focal_window_water_gate_Pa", float("nan"))
        if probe_b else float("nan")
    )
    sp_offset_corr = probe_a.get("spatial_max_offset_from_target_mm", float("nan")) if probe_a else float("nan")
    sp_offset_geom = probe_b.get("spatial_max_offset_from_target_mm", float("nan")) if probe_b else float("nan")
    sp_offset_water = probe_c.get("spatial_max_offset_from_target_mm", float("nan")) if probe_c else float("nan")

    # Spatial-search attenuation: water baseline (use spatial max of water cube
    # so we compare apples-to-apples: spatial-peak in water vs spatial-peak in skull)
    atten_db_fw_water_sp_waterbase_target = _db(sp_corr_cube_max, sp_water_target)
    atten_db_fw_water_sp_waterbase_cube = _db(sp_corr_cube_max, sp_water_cube_max)

    print("\n--- SPATIAL-SEARCH METRICS ---")
    print(f"  water    target  : {sp_water_target:.4g} Pa")
    print(f"  water    cube-max: {sp_water_cube_max:.4g} Pa  (offset={sp_offset_water:.2f} mm)")
    print(f"  skull C  target  : {sp_corr_target:.4g} Pa")
    print(f"  skull C  cube-max: {sp_corr_cube_max:.4g} Pa  (offset={sp_offset_corr:.2f} mm)")
    print(f"  atten vs water_target (cube-max skull C): {atten_db_fw_water_sp_waterbase_target:.2f} dB")
    print(f"  atten vs water_cube   (cube-max skull C): {atten_db_fw_water_sp_waterbase_cube:.2f} dB")

    print(
        f"SUBJECT_SUMMARY subject={subj} "
        f"bone_pct={pct_skull:.3f} "
        f"skull_path_near={skull_path_near_mm:.2f} "
        f"p_water={_fmt(p_water)} "
        f"p_skull={_fmt(p_skull)} "
        f"p_geom_skull={_fmt(p_geom_skull)} "
        f"gain_vs_mean_water={stats_c['gain_vs_mean']:.4f} "
        f"gain_vs_mean_skull={stats_a['gain_vs_mean']:.4f} "
        f"atten_db={_fmt(atten_db)} "
        f"p_fw_geom_water={_fmt(p_fw_geom_water)} "
        f"p_fw_geom_skull_corr={_fmt(p_fw_geom_skull_corr)} "
        f"p_fw_geom_skull_geom={_fmt(p_fw_geom_skull_geom)} "
        f"p_fw_water_skull_corr={_fmt(p_fw_water_skull_corr)} "
        f"p_fw_water_skull_geom={_fmt(p_fw_water_skull_geom)} "
        f"t_water_peak_us={_fmt(t_water_peak_s*1e6) if np.isfinite(t_water_peak_s) else 'nan'} "
        f"tof_geom_us={_fmt(tof_geom_s*1e6) if np.isfinite(tof_geom_s) else 'nan'} "
        f"atten_db_fw_geom={_fmt(atten_db_fw_geom)} "
        f"atten_db_fw_water={_fmt(atten_db_fw_water)} "
        f"sp_water_target={_fmt(sp_water_target)} "
        f"sp_water_cube_max={_fmt(sp_water_cube_max)} "
        f"sp_offset_water_mm={_fmt(sp_offset_water)} "
        f"sp_corr_target={_fmt(sp_corr_target)} "
        f"sp_corr_cube_max={_fmt(sp_corr_cube_max)} "
        f"sp_offset_corr_mm={_fmt(sp_offset_corr)} "
        f"sp_geom_target={_fmt(sp_geom_target)} "
        f"sp_geom_cube_max={_fmt(sp_geom_cube_max)} "
        f"sp_offset_geom_mm={_fmt(sp_offset_geom)} "
        f"atten_db_sp_target_base={_fmt(atten_db_fw_water_sp_waterbase_target)} "
        f"atten_db_sp_cube_base={_fmt(atten_db_fw_water_sp_waterbase_cube)}"
    )


if __name__ == "__main__":
    main()
