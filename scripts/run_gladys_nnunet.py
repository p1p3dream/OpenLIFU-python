#!/usr/bin/env python3
"""GLADYS Simulation with nnU-Net segmentation (fullhead, Dataset002).

Forked from run_gladys_simulation.py, with ThresholdMRI replaced by a
PreSegmented wrapper that loads a pre-computed label NIfTI produced by
nnUNetv2_predict.

The label NIfTI uses the Dataset002 fullhead mapping (see
openlifu.seg.seg_methods.nnunet_seg.LABEL_MAP_FULLHEAD):
  0=water (background), 1=air, 2=csf, 3=gray_matter, 4=white_matter,
  5=skull, 6=tissue (soft tissue / scalp).

Outputs:
  - pmax NIfTIs for sims A/B/C: ~/Data/openlifu-validation/results/gladys_nnunet_{corrected,geometric,water}_pmax.nii.gz
  - analysis PNG: ~/Data/openlifu-validation/results/gladys_nnunet_pmax_analysis_2026-04-18.png
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
import contextlib
import json
import pathlib
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
# Ensure sibling script module (_gpu_flock) is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _gpu_flock import gpu_flock
from openlifu.bf.apod_methods.skull_incidence import SkullIncidenceApodization
from openlifu.bf.delay_methods.complex_weighted import ComplexWeighted
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

# Enable the lightweight time-gated probe after each full run_simulation call.
# Set the env var OPENLIFU_DISABLE_TIMEGATED_PROBE=1 to turn it off at runtime
# without editing this file. If the extra k-wave call fails for any reason the
# probe is skipped with a warning and the main pipeline continues.
ENABLE_TIMEGATED_PROBE = os.environ.get("OPENLIFU_DISABLE_TIMEGATED_PROBE", "") != "1"

# Opt-in apodization mode: down-weight elements whose rays strike the skull
# near-perpendicular. Off by default (uniform apod).
SKULL_INCIDENCE_APOD = os.environ.get("SKULL_INCIDENCE_APOD", "") == "1"
SKULL_APOD_MIN_ANGLE_DEG = float(os.environ.get("SKULL_APOD_MIN_ANGLE_DEG", "20.0"))
SKULL_APOD_ROLLOFF_ANGLE_DEG = float(os.environ.get("SKULL_APOD_ROLLOFF_ANGLE_DEG", "45.0"))
APOD_INFIX = "_skullapod" if SKULL_INCIDENCE_APOD else ""

# Delay method selector. "simulation_corrected" (default) preserves current
# behavior. "complex_weighted" swaps in ComplexWeighted (narrowband complex
# weights), which returns (delays, apod) via calc_delays_and_apod; the
# returned apod is multiplied into the existing (usually uniform) apod.
DELAY_METHOD = os.environ.get("DELAY_METHOD", "simulation_corrected").strip().lower()
if DELAY_METHOD not in ("simulation_corrected", "complex_weighted"):
    raise ValueError(
        f"DELAY_METHOD must be 'simulation_corrected' or 'complex_weighted', got '{DELAY_METHOD}'"
    )
DELAY_INFIX = "_cw" if DELAY_METHOD == "complex_weighted" else ""


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
    """A SegmentationMethod that returns labels from a pre-computed NIfTI.

    The labels DataArray is loaded once at init. In `_segment`, nearest-neighbor
    resampling maps those labels onto the query volume's coordinate grid, then
    converts nnU-Net label ints to the material-index ordering used by
    `_map_params` / `seg_params`.
    """

    label_nifti_path: str = ""
    nnunet_label_map: dict[int, str] = field(default_factory=lambda: dict(LABEL_MAP_FULLHEAD))
    materials: dict[str, Material] = field(default_factory=_default_fullhead_materials)

    def __post_init__(self):
        super().__post_init__()
        # Cache the label volume in world-coordinate xarray form.
        if not self.label_nifti_path:
            raise ValueError("label_nifti_path is required")
        img = nib.load(self.label_nifti_path)
        # nibabel data comes out as (X, Y, Z). Use its affine diagonal for coords.
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

        # Validate every model label maps to a known material.
        mat_idx = self._material_indices()
        for label_int, mat_key in self.nnunet_label_map.items():
            if mat_key not in mat_idx:
                raise ValueError(
                    f"nnunet_label_map references material '{mat_key}' "
                    f"which is not in self.materials (for label {label_int})."
                )

    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        """Resample the pre-computed labels onto `volume.coords` (nearest neighbor)
        and remap nnU-Net label ints to material indices."""
        mat_idx = self._material_indices()
        # Build mapping: label_int (nnU-Net) -> material_index (float for map_coordinates)
        # We first resample the INT label array via order=0 interpolation, then remap.
        src = self._labels  # xarray of shape (Nx, Ny, Nz)
        src_dims = list(src.dims)

        # For each dim in the target volume, find fractional indices into src.
        # Source uses uniform spacing (we built coords from affine diag) so
        # frac_idx = (target_coord - origin) / spacing.
        src_origin = {d: float(src.coords[d].to_numpy()[0]) for d in src_dims}
        src_spacing = {}
        for d in src_dims:
            cs = src.coords[d].to_numpy()
            src_spacing[d] = float(cs[1] - cs[0]) if len(cs) > 1 else 1.0

        # Target grid must have the same dim names.
        tgt_dims = list(volume.dims)
        assert set(tgt_dims) == set(src_dims), (
            f"Dim mismatch: src={src_dims}, tgt={tgt_dims}"
        )

        # Build 3-D coordinate arrays of fractional source indices for each
        # target voxel.
        tgt_coord_arrays = [volume.coords[d].to_numpy() for d in src_dims]
        mg = np.meshgrid(*tgt_coord_arrays, indexing="ij")
        frac_idx = []
        for i, d in enumerate(src_dims):
            fi = (mg[i] - src_origin[d]) / src_spacing[d]
            frac_idx.append(fi)
        frac_stack = np.stack(frac_idx, axis=0)  # (3, Nx, Ny, Nz)

        # Use nearest-neighbor (order=0) so we stay on the integer label set.
        resampled = map_coordinates(
            src.to_numpy().astype(np.float32),
            frac_stack,
            order=0,
            mode="constant",
            cval=0.0,  # label 0 = water outside the original field
        ).astype(np.int16)

        # Remap nnU-Net label ints -> material indices.
        water_idx = mat_idx["water"]
        output = np.full(resampled.shape, water_idx, dtype=int)
        for nn_label, mat_key in self.nnunet_label_map.items():
            output[resampled == nn_label] = mat_idx[mat_key]

        # Re-dim to match target volume order.
        # Because we used src_dims order above, assemble with those dim names.
        labels_da = xa.DataArray(
            output,
            dims=src_dims,
            coords={d: volume.coords[d] for d in src_dims},
        )
        # Transpose to volume.dims order for the caller.
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


# ---------------------------------------------------------------------------
# Masked argmax with wider (20 mm) exclusion radius
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Time-gated probe: runs a second lightweight k-wave sim with a sparse sensor
# mask that records full p(t) at a handful of strategic voxels, then extracts
# peak pressure within the physical focal-arrival window (tof +/- 2*pulse_dur)
# vs peak over all time. Motivation: on real skull, p_max@target is often
# dominated by post-focal reverberation, so the all-time peak overstates the
# coherent focal pressure. The time-gated window gives a more honest number.
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
    """Run k-wave with a sparse sensor mask, returning raw p(t, sensor).

    Mirrors run_simulation(source_method='point_source') but replaces the full
    grid sensor with a binary mask covering only the selected sensor voxels.
    """
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end, cfl=cfl)
    # Extend time axis same way run_simulation does when t_end==0.
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

    # Reorder sensor mask from params order to xyz for k-wave.
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
) -> tuple[np.ndarray, list[str], list[tuple[int, int, int]], list[np.ndarray]]:
    """Build the 5-voxel sensor mask in params dim order.

    Returns (mask, names, xyz_indices_list, actual_world_mm_list).
    """
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

    sensor_world_mm = [
        ("target",        target_mm.copy()),
        ("aperture_ctr",  aperture_center_mm.copy()),
        ("midway",        target_mm + 0.5 * axis_vec),
        ("quarterway",    target_mm + 0.75 * axis_vec),
        ("off_axis_15mm", target_mm + 15.0 * perp),
    ]

    sim_dim_names = list(params.dims)
    sim_coord_params = {d: params.coords[d].to_numpy() for d in sim_dim_names}
    mask = np.zeros(tuple(len(sim_coord_params[d]) for d in sim_dim_names), dtype=np.int32)

    names: list[str] = []
    xyz_idx_list: list[tuple[int, int, int]] = []
    actual_list: list[np.ndarray] = []
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
        mask[tuple(idx_in_params)] = 1
        names.append(name)
        xyz_idx_list.append(tuple(idx_in_xyz))
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
) -> dict | None:
    """Run the sparse-sensor probe and compute time-gated metrics.

    Returns a dict with per-sensor metrics, raw time series, and a summary
    p_focal_window@target. Returns None (and logs a warning) on any failure.
    """
    try:
        mask, names, xyz_idx_list, actual_list = _build_sensor_mask(
            params, target_mm, aperture_center_mm,
        )
        probe_kwargs = dict(common_kwargs)
        # Probe manages its own sensor, not the full-grid one, so we don't
        # need ref_values_only passed via common_kwargs; we pass it directly.
        probe_kwargs.pop("arr", None)
        probe_kwargs.pop("apod", None)
        # common_kwargs already has freq/cycles/amplitude/dt/t_end/cfl/gpu and
        # source_method; we don't need source_method for the sparse sim.
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

        def _col_for(xyz_idx_tuple: tuple[int, int, int]) -> int | None:
            matches = np.where(
                (xyz_order[:, 0] == xyz_idx_tuple[0])
                & (xyz_order[:, 1] == xyz_idx_tuple[1])
                & (xyz_order[:, 2] == xyz_idx_tuple[2])
            )[0]
            return int(matches[0]) if len(matches) else None

        per_sensor = {}
        for name, xyz_idx, actual in zip(names, xyz_idx_list, actual_list):
            col = _col_for(xyz_idx)
            d_m = float(np.linalg.norm(actual - aperture_center_mm)) * 1e-3
            tof_s = d_m / C0
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
                    "p_allt_Pa": float("nan"),
                    "ratio": float("nan"),
                    "p_t": [],
                    "note": "voxel collision; no dedicated column",
                })
            else:
                p_t = p_sensor[:, col]
                p_abs = np.abs(p_t)
                p_allt = float(p_abs.max())
                t_lo = tof_s - 2.0 * pulse_dur
                t_hi = tof_s + 2.0 * pulse_dur
                in_win = (t_axis >= t_lo) & (t_axis <= t_hi)
                p_fw = float(p_abs[in_win].max()) if in_win.any() else float("nan")
                ratio = (p_fw / p_allt) if (p_allt > 0 and np.isfinite(p_fw)) else float("nan")
                rec.update({
                    "column": col,
                    "p_focal_window_Pa": p_fw,
                    "p_allt_Pa": p_allt,
                    "ratio": ratio,
                    "p_t": p_t.astype(np.float32).tolist(),
                    "note": "",
                })
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
        }
        # Convenience shortcut for the summary table.
        tgt = per_sensor.get("target", {})
        result["p_focal_window_at_target_Pa"] = tgt.get("p_focal_window_Pa", float("nan"))
        result["p_allt_at_target_Pa"] = tgt.get("p_allt_Pa", float("nan"))
        result["ratio_at_target"] = tgt.get("ratio", float("nan"))
        return result
    except Exception as exc:
        logger.warning(
            "Time-gated probe failed for %s (%s); continuing without it.",
            sim_label, exc, exc_info=True,
        )
        return None


def _save_timegated_json(result: dict | None, out_path: Path) -> None:
    """Write the probe result to disk as JSON. No-op if result is None."""
    if result is None:
        return
    try:
        with open(out_path, "w") as f:
            json.dump(result, f)
        logger.info("    Saved time-gated sidecar: %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save time-gated sidecar %s: %s", out_path, exc)


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_total = time.time()

    print("=" * 72)
    print("GLADYS Simulation: Transcranial FUS (nnU-Net fullhead segmentation)")
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
    # Histogram of labels
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

    # Brain centroid = union of csf, gray_matter, white_matter (nnU-Net labels 2/3/4).
    # Fall back to "tissue" if those are empty (shouldn't happen with fullhead model).
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

    # Resample MRI onto sim grid (for anatomical reference only; segmentation
    # comes from the pre-computed NIfTI via PreSegmented._segment).
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
    print(f"    SKULL voxels (bone fraction): {n_skull:,d} ({pct_skull:.2f}%)  [THRESHOLD_MRI GAVE ~55%]")

    if n_air > 0:
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation
        print("    Replaced air with water.")

    # Print full sim-grid histogram
    print("    Sim-grid material histogram:")
    for name, idx in material_idx.items():
        count = int((sim_seg_arr == idx).sum())
        pct = 100.0 * count / total_vox
        print(f"      {name:>14s}: {count:>10,d} voxels ({pct:.2f}%)")

    # -------------------------------------------------------------------
    # 5b. Skull path length along target-to-aperture axis
    # -------------------------------------------------------------------
    print("\n[5b] Measuring skull path length along target->aperture-center axis...")
    aperture_center_mm = positions.mean(axis=0)
    ray_dir = aperture_center_mm - target_mm
    ray_len_mm = float(np.linalg.norm(ray_dir))
    # Extend both directions: from target go +axis up to RADIUS_MM+extra to
    # catch near-side skull, and go -axis up to RADIUS_MM+extra to catch
    # far-side skull (the "through transmission" scenario). We report the
    # near-side (target->aperture) since that is what the beam crosses.
    skull_path_near_mm = float("nan")
    skull_path_far_mm = float("nan")
    if ray_len_mm > 1e-6:
        ray_unit = ray_dir / ray_len_mm
        # Go from target outward toward (and past) the aperture center.
        # Use max distance of 1.2 * RADIUS_MM to ensure we cross the full
        # skull shell plus some scalp.
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
        # Report: near-side = target -> aperture, far-side = target -> opposite
        print(f"    Probe length: {probe_len_mm:.1f} mm each side of target")
        print(f"    NEAR-side skull (target -> aperture direction): {skull_path_near_mm:.1f} mm")
        print(f"    FAR-side  skull (target -> opposite direction): {skull_path_far_mm:.1f} mm")
        # Find first skull segment on the near side for diagnostics
        if mask_near.any():
            idx = np.argwhere(mask_near).flatten()
            entry = float(ts_near[idx[0]])
            exit_ = float(ts_near[idx[-1]])
            print(f"    Near-side skull first encountered at t={entry:.1f} mm, last at t={exit_:.1f} mm")
        if mask_far.any():
            idx = np.argwhere(mask_far).flatten()
            entry = float(ts_far[idx[0]])
            exit_ = float(ts_far[idx[-1]])
            print(f"    Far-side  skull first encountered at t={entry:.1f} mm, last at t={exit_:.1f} mm")
    skull_path_mm = skull_path_near_mm  # primary metric for report

    # -------------------------------------------------------------------
    # 6. Target, delays, t_end, sim kwargs
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
            # CW_NORM selects how the per-element amplitude weights are
            # normalized. ComplexWeighted returns apod with max(apod)==1,
            # which discards aperture-sum power. "sum" preserves total
            # radiated power (sum(apod)==N); "rms" preserves radiated
            # intensity (sum(apod^2)==N). Default is "max" for backward
            # compatibility.
            cw_norm = os.environ.get("CW_NORM", "max").lower()
            apod_cw_pre = apod_cw.copy()
            if cw_norm == "max":
                pass  # already normalized so max(apod) == 1
            elif cw_norm == "sum":
                s = float(apod_cw.sum())
                if s > 0:
                    apod_cw = apod_cw * (len(apod_cw) / s)
            elif cw_norm == "rms":
                current_sumsq = float(np.sum(apod_cw ** 2))
                if current_sumsq > 0:
                    apod_cw = apod_cw * np.sqrt(len(apod_cw) / current_sumsq)
            else:
                raise ValueError(f"CW_NORM must be max/sum/rms, got {cw_norm}")
            print(
                f"    CW_NORM={cw_norm}: pre-norm sum={apod_cw_pre.sum():.3f}, "
                f"max={apod_cw_pre.max():.4f}; "
                f"post-norm sum={apod_cw.sum():.3f}, max={apod_cw.max():.4f}"
            )
            n_hot = int((apod_cw > 2.0).sum())
            if n_hot > 0:
                print(
                    f"    WARNING: {n_hot} element(s) have apod > 2.0 after "
                    f"CW_NORM={cw_norm} (max={apod_cw.max():.3f}); not clamping."
                )
            thr = 0.01
            n_total = int(apod_cw.size)
            n_active = int((apod_cw > thr).sum())
            frac_above = n_active / n_total if n_total > 0 else 0.0
            print(
                f"    CW apod stats: min={apod_cw.min():.4f}, max={apod_cw.max():.4f}, "
                f"mean={apod_cw.mean():.4f}, std={apod_cw.std():.4f}"
            )
            print(
                f"    CW apod active (> {thr}): {n_active}/{n_total} "
                f"(fraction={frac_above:.3f})"
            )
            cw_apod_info = {
                "cw_norm": cw_norm,
                "min": float(apod_cw.min()),
                "max": float(apod_cw.max()),
                "mean": float(apod_cw.mean()),
                "std": float(apod_cw.std()),
                "apod_max": float(apod_cw.max()),
                "apod_sum": float(apod_cw.sum()),
                "pre_norm_max": float(apod_cw_pre.max()),
                "pre_norm_sum": float(apod_cw_pre.sum()),
                "threshold": thr,
                "n_active": n_active,
                "n_total": n_total,
                "n_hot_above_2": n_hot,
                "fraction_above_threshold": frac_above,
                "weights": apod_cw.tolist(),
            }
        else:
            print("    CW returned apod=None; treating as uniform amplitude.")
        print(f"    Delay range (corrected): {delays_corrected.min()*1e6:.2f} to "
              f"{delays_corrected.max()*1e6:.2f} us  (min>=0 enforced)")
    else:
        print("\n[8] SimulationCorrected (phase correction)...")
        sim_corrected = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
        t0 = time.time()
        with gpu_flock():
            delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
        print(f"    Phase correction done in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------
    # 8b. Apodization: uniform (default) or SkullIncidenceApodization.
    # -------------------------------------------------------------------
    apod_info: dict = {"mode": "uniform"}
    if SKULL_INCIDENCE_APOD:
        print("\n[8b] Computing SkullIncidenceApodization weights "
              f"(min_angle={SKULL_APOD_MIN_ANGLE_DEG} deg, "
              f"rolloff={SKULL_APOD_ROLLOFF_ANGLE_DEG} deg)...")
        # Build a skull mask DataArray matching the sim grid. sim_seg is
        # already an xa.DataArray on the sim grid with mm coords; derive a
        # boolean/uint8 mask and preserve its dims+coords.
        skull_mask_xa = xa.DataArray(
            (sim_seg.to_numpy() == material_idx["skull"]).astype(np.uint8),
            dims=sim_seg.dims,
            coords={d: sim_seg.coords[d] for d in sim_seg.dims},
        )
        print(f"    Skull-mask voxels: {int(skull_mask_xa.sum())} "
              f"({100.0 * float(skull_mask_xa.mean()):.2f}% of sim grid)")
        apod_method = SkullIncidenceApodization(
            skull_mask=skull_mask_xa,
            min_angle_deg=SKULL_APOD_MIN_ANGLE_DEG,
            rolloff_angle_deg=SKULL_APOD_ROLLOFF_ANGLE_DEG,
        )
        t0 = time.time()
        # Elements have world positions pre-baked above; pass transform=None.
        apod = np.asarray(
            apod_method.calc_apodization(arr, target, sim_params, transform=None),
            dtype=float,
        )
        print(f"    apod computed in {time.time()-t0:.1f}s")
        n_total = int(apod.size)
        n_zero = int((apod <= 0.0).sum())
        n_full = int((apod >= 1.0).sum())
        n_partial = int(((apod > 0.0) & (apod < 1.0)).sum())
        n_active = n_total - n_zero
        print(f"    Elements: active={n_active}/{n_total}, "
              f"full={n_full}, partial={n_partial}, zero={n_zero}")
        print(f"    apod stats: min={apod.min():.3f}, max={apod.max():.3f}, "
              f"mean={apod.mean():.3f}, sum={apod.sum():.2f}")
        apod_info = {
            "mode": "skull_incidence",
            "min_angle_deg": SKULL_APOD_MIN_ANGLE_DEG,
            "rolloff_angle_deg": SKULL_APOD_ROLLOFF_ANGLE_DEG,
            "n_total": n_total,
            "n_active": n_active,
            "n_full": n_full,
            "n_partial": n_partial,
            "n_zero": n_zero,
            "weights": apod.tolist(),
            "apod_sum": float(apod.sum()),
        }
    else:
        apod = np.ones(arr.numelements())
        apod_info = {
            "mode": "uniform",
            "n_total": int(apod.size),
            "n_active": int(apod.size),
            "n_full": int(apod.size),
            "n_partial": 0,
            "n_zero": 0,
            "weights": apod.tolist(),
            "apod_sum": float(apod.sum()),
        }

    # Fold in ComplexWeighted per-element amplitude if applicable. The
    # existing apod above is either uniform or skull-incidence; CW amplitude
    # stacks on top as a simple element-wise multiplication.
    if apod_cw is not None:
        if apod_cw.shape != apod.shape:
            raise ValueError(
                f"apod_cw shape {apod_cw.shape} does not match apod shape {apod.shape}"
            )
        apod = np.asarray(apod, dtype=float) * np.asarray(apod_cw, dtype=float)
        apod_info["cw_apod"] = cw_apod_info
        apod_info["mode"] = (
            f"{apod_info.get('mode', 'uniform')}+complex_weighted"
        )
        apod_info["weights"] = apod.tolist()
        apod_info["apod_sum"] = float(apod.sum())
        print(
            f"    Combined apod after CW: min={apod.min():.4f}, max={apod.max():.4f}, "
            f"mean={apod.mean():.4f}, sum={apod.sum():.2f}"
        )

    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, source_method="point_source",
    )

    # -------------------------------------------------------------------
    # 9-11. Sims A, B, C
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[SIM A] Phase-corrected + heterogeneous skull (nnU-Net)")
    print("=" * 72)
    t0 = time.time()
    with gpu_flock():
        result_a = run_simulation(params=sim_params, delays=delays_corrected, ref_values_only=False, **common_kwargs)
    print(f"    Completed in {time.time()-t0:.1f}s")
    stats_a = extract_focal_stats(result_a, target_mm, "SIM A (corrected+hetero)", element_positions_mm=positions)
    masked_a_20 = extract_masked_argmax(result_a, target_mm, positions, 20.0)
    probe_a = _run_timegated_probe(
        arr=arr, params=sim_params, delays=delays_corrected, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=False,
        sim_label="SIM A (corrected+hetero)",
    ) if ENABLE_TIMEGATED_PROBE else None

    print("\n" + "=" * 72)
    print("[SIM B] Geometric + heterogeneous skull (nnU-Net)")
    print("=" * 72)
    t0 = time.time()
    with gpu_flock():
        result_b = run_simulation(params=sim_params, delays=delays_geo, ref_values_only=False, **common_kwargs)
    print(f"    Completed in {time.time()-t0:.1f}s")
    stats_b = extract_focal_stats(result_b, target_mm, "SIM B (geometric+hetero)", element_positions_mm=positions)
    masked_b_20 = extract_masked_argmax(result_b, target_mm, positions, 20.0)
    probe_b = _run_timegated_probe(
        arr=arr, params=sim_params, delays=delays_geo, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=False,
        sim_label="SIM B (geometric+hetero)",
    ) if ENABLE_TIMEGATED_PROBE else None

    print("\n" + "=" * 72)
    print("[SIM C] Geometric + homogeneous water (ref_values_only)")
    print("=" * 72)
    t0 = time.time()
    with gpu_flock():
        result_c = run_simulation(params=sim_params, delays=delays_geo, ref_values_only=True, **common_kwargs)
    print(f"    Completed in {time.time()-t0:.1f}s")
    stats_c = extract_focal_stats(result_c, target_mm, "SIM C (geometric+water)", element_positions_mm=positions)
    masked_c_20 = extract_masked_argmax(result_c, target_mm, positions, 20.0)
    probe_c = _run_timegated_probe(
        arr=arr, params=sim_params, delays=delays_geo, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=True,
        sim_label="SIM C (geometric+water)",
    ) if ENABLE_TIMEGATED_PROBE else None

    # -------------------------------------------------------------------
    # 12. Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY (nnU-Net segmentation)")
    print("=" * 72)
    print(f"Bone fraction in sim grid : {pct_skull:.2f}%  (ThresholdMRI: ~55.62%)")
    print(f"Skull path NEAR side      : {skull_path_near_mm:.1f} mm  (ThresholdMRI: ~42 mm)")
    print(f"Skull path FAR  side      : {skull_path_far_mm:.1f} mm")
    print()
    for label, stats, masked20, probe in [
        ("A: Corrected + skull", stats_a, masked_a_20, probe_a),
        ("B: Geometric + skull", stats_b, masked_b_20, probe_b),
        ("C: Geometric + water", stats_c, masked_c_20, probe_c),
    ]:
        print(f"{label}")
        print(f"  raw max_p     = {stats['max_pressure']:.4g} Pa")
        print(f"  p@target      = {stats['p_at_target']:.4g} Pa")
        print(f"  raw focal err = {stats['focal_error']:.2f} mm")
        if masked20["max_p"] is not None:
            print(f"  masked@20mm   = {masked20['max_p']:.4g} Pa @ err={masked20['error']:.2f} mm")
        if probe is not None:
            p_fw = probe.get("p_focal_window_at_target_Pa", float("nan"))
            p_al = probe.get("p_allt_at_target_Pa", float("nan"))
            ratio = probe.get("ratio_at_target", float("nan"))
            print(f"  p_focal_window@target = {p_fw:.4g} Pa  (p_allt@target = {p_al:.4g} Pa, ratio = {ratio:.3f})")
        elif ENABLE_TIMEGATED_PROBE:
            print(f"  p_focal_window@target = (probe failed; see warnings above)")
        print()

    # Compact table summary (matches task spec columns).
    print("-" * 92)
    hdr = f"{'Sim':<6}{'max_p(Pa)':>14}{'focal_err':>11}{'p@target':>14}{'p_focal_win@tgt':>18}{'ratio':>9}"
    print(hdr)
    print("-" * 92)
    for sim_letter, stats, probe in [
        ("A", stats_a, probe_a),
        ("B", stats_b, probe_b),
        ("C", stats_c, probe_c),
    ]:
        if probe is not None:
            pfw = probe.get("p_focal_window_at_target_Pa", float("nan"))
            ratio = probe.get("ratio_at_target", float("nan"))
            pfw_s = f"{pfw:.4g}" if np.isfinite(pfw) else "NaN"
            ratio_s = f"{ratio:.3f}" if np.isfinite(ratio) else "NaN"
        else:
            pfw_s = "-"
            ratio_s = "-"
        print(
            f"{sim_letter:<6}"
            f"{stats['max_pressure']:>14.4g}"
            f"{stats['focal_error']:>11.2f}"
            f"{stats['p_at_target']:>14.4g}"
            f"{pfw_s:>18}"
            f"{ratio_s:>9}"
        )
    print("-" * 92)
    print()

    # -------------------------------------------------------------------
    # 13. Save pmax NIfTIs
    # -------------------------------------------------------------------
    results_dir = Path.home() / "Data/openlifu-validation/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for sim_label, result, probe in [
        ("corrected", result_a, probe_a),
        ("geometric", result_b, probe_b),
        ("water", result_c, probe_c),
    ]:
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
        out_path = results_dir / f"gladys_nnunet{APOD_INFIX}{DELAY_INFIX}_{sim_label}_pmax.nii.gz"
        nib.save(nib.Nifti1Image(p_max_data.astype(np.float32), out_affine), str(out_path))
        print(f"    Saved: {out_path}")

        # Time-gated sidecar JSON with raw time series + metrics.
        # Always write a sidecar (even if the probe failed) so the apod
        # metadata and weights are captured for later analysis.
        sidecar_path = results_dir / f"gladys_nnunet{APOD_INFIX}{DELAY_INFIX}_{sim_label}_timegated.json"
        probe_out = dict(probe) if probe is not None else {
            "sim_label": sim_label,
            "probe_failed": True,
        }
        probe_out["apod_info"] = apod_info
        _save_timegated_json(probe_out, sidecar_path)

    # -------------------------------------------------------------------
    # 14. Plot: 2D slice + axial profile per sim
    # -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 2D slice: use approach axis for profile; show slice perpendicular to
    # the axis with smallest skull extent (typically y for an axial head scan).
    # For simplicity: take slice at target index along the axis with minimum
    # extent among the two non-approach axes.
    plot_path = results_dir / f"gladys_nnunet{APOD_INFIX}{DELAY_INFIX}_pmax_analysis_2026-04-18.png"

    sim_x = np.asarray(sim_coord_arrays[0])
    sim_y = np.asarray(sim_coord_arrays[1])
    sim_z = np.asarray(sim_coord_arrays[2])
    axes_coords = [sim_x, sim_y, sim_z]
    target_idx_sim = tuple(
        int(np.argmin(np.abs(axes_coords[ax] - target_mm[ax])))
        for ax in range(3)
    )
    # Choose slicing axis = non-approach axis whose coord at target gives the
    # best cross-section through focus & transducer. Use the axis orthogonal
    # to both approach_axis and the widest remaining axis.
    non_approach = [ax for ax in range(3) if ax != approach_axis]
    # Use the first non-approach axis as the slice axis.
    slice_axis = non_approach[0]

    def slice_2d(vol: np.ndarray) -> tuple[np.ndarray, tuple[str, str]]:
        other_axes = [ax for ax in range(3) if ax != slice_axis]
        idx = target_idx_sim[slice_axis]
        sl = [slice(None), slice(None), slice(None)]
        sl[slice_axis] = idx
        return vol[tuple(sl)], tuple("xyz"[ax] for ax in other_axes)

    # Axial profile: values along the approach axis at the target's other coords.
    def axial_profile(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx_list = list(target_idx_sim)
        idx_list[approach_axis] = slice(None)
        profile = vol[tuple(idx_list)]
        coord = axes_coords[approach_axis]
        return coord, np.asarray(profile)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    sims = [("A: corrected+skull", result_a), ("B: geometric+skull", result_b), ("C: geometric+water", result_c)]
    # Global vmax in log scale for consistent comparison
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
        # Mark target
        other_axes = [ax for ax in range(3) if ax != slice_axis]
        ax_top.plot(target_mm[other_axes[0]], target_mm[other_axes[1]], "wx", ms=10, mew=2)
        ax_top.set_title(f"{title}\nlog10(p_max) slice @{'xyz'[slice_axis]}={target_mm[slice_axis]:.1f}mm")
        ax_top.set_xlabel(f"{u_label} (mm)")
        ax_top.set_ylabel(f"{v_label} (mm)")
        plt.colorbar(im, ax=ax_top, label="log10(Pa)")

        # Axial profile
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
        f"GLADYS sim with nnU-Net segmentation (GU008)  |  "
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
