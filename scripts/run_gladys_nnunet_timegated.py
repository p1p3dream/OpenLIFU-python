#!/usr/bin/env python3
"""GLADYS transcranial sim with TIME-GATED target pressure recording.

Diagnostic alternative to p_max. Instead of recording peak pressure over all
time across the whole grid (dominated by skull-surface standing waves on real
heads), this script records the FULL pressure time series p(t) at a small
set of sensor voxels and extracts the peak amplitude in a physically-motivated
time window (t ~= time-of-flight +/- 2*pulse_duration). For a clean focus,
the time-gated peak at the target should approach the all-time peak there,
while near-aperture voxels see much bigger all-time peaks than focal-window
peaks (standing-wave dominated).

Runs SIM A (corrected + heterogeneous skull) and SIM C (water reference) only.
Skips SIM B to save time.

Reuses the pre-computed nnU-Net labels produced by run_gladys_nnunet.py.
Does not write any NIfTIs; prints a table and exits.
"""

import logging
import os
import sys
import time

# Same kwave logging workaround as run_gladys_nnunet.py.
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
import pathlib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import xarray as xa
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
# Configuration (identical to run_gladys_nnunet.py where possible)
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


# ---------------------------------------------------------------------------
# PreSegmented (copy of the version in run_gladys_nnunet.py).
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
            src.to_numpy().astype(np.float32), frac_stack,
            order=0, mode="constant", cval=0.0,
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
# Array
# ---------------------------------------------------------------------------
def create_hemispherical_array(
    n_elements=64, radius_mm=90.0, aperture_mm=80.0,
    freq_hz=500e3, element_size_mm=5.0,
) -> Transducer:
    half_aperture = aperture_mm / 2.0
    if half_aperture > radius_mm:
        raise ValueError("aperture > radius")
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
# Custom k-wave call that records p(t) at sparse sensor voxels only.
# ---------------------------------------------------------------------------
def run_sparse_sensor_sim(
    arr: Transducer,
    params: xa.Dataset,
    delays: np.ndarray,
    apod: np.ndarray,
    sensor_mask_params_order: np.ndarray,  # 3D binary, in params dim order
    freq: float = 500e3,
    cycles: int = 5,
    amplitude: float = 1.0,
    dt: float = 0.0,
    t_end: float = 0.0,
    cfl: float = 0.1,
    gpu: bool = True,
    ref_values_only: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Run k-wave with a sparse sensor mask, returning the raw p(t, sensor) array.

    This mirrors run_simulation(source_method='point_source') but replaces the
    full-grid sensor with a binary mask covering only the chosen sensor voxels.

    Returns:
        p_sensor: (n_timesteps, n_sensors) pressure time series, where the
            sensor axis is in Fortran (column-major) order of the nonzero
            voxels in the xyz-transposed sensor mask.
        dt: time step in seconds.
        xyz_sensor_indices: (n_sensors, 3) array of (x_idx, y_idx, z_idx) for
            each sensor, matching the k-wave xyz Fortran order used in the
            second axis of p_sensor. These indices are into the x,y,z-ordered
            grid, NOT params dim order.
    """
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end, cfl=cfl)
    # Extend time axis same way run_simulation does
    if t_end == 0:
        _coord_units = [params[dim].attrs['units'] for dim in params.dims]
        _scl_to_m = getunitconversion(_coord_units[0], 'm')
        _c_ref = float(params['sound_speed'].attrs.get('ref_value', 1500.0))
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

    # Source signal
    t = np.arange(
        0,
        np.min([cycles / freq, (kgrid.Nt - np.ceil(max(delays) / kgrid.dt)) * kgrid.dt]),
        kgrid.dt,
    )
    input_signal = amplitude * np.sin(2 * np.pi * freq * t)

    medium = get_medium(params, ref_values_only=ref_values_only)

    # Point source: elements mapped to nearest grid voxels.
    source_mat = arr.calc_output(input_signal, kgrid.dt, delays, apod)
    source = get_point_source(arr, params, source_mat)

    # Reorder sensor mask from params order to xyz for k-wave.
    dim_names = list(params.dims)
    _dim_order = {'x': 0, 'y': 1, 'z': 2}
    perm = [_dim_order[d] for d in dim_names]
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    sensor_mask_xyz = np.transpose(sensor_mask_params_order, inv_perm)

    # Compute xyz Fortran order of sensor voxels so the caller knows which
    # column corresponds to which voxel.
    nz = np.nonzero(sensor_mask_xyz)
    # k-wave uses Fortran (column-major) ordering over (x, y, z).
    lin = nz[0].astype(np.int64) \
          + nz[1].astype(np.int64) * sensor_mask_xyz.shape[0] \
          + nz[2].astype(np.int64) * sensor_mask_xyz.shape[0] * sensor_mask_xyz.shape[1]
    order = np.argsort(lin)
    xyz_sensor_indices = np.stack([nz[0][order], nz[1][order], nz[2][order]], axis=-1)

    sensor = kSensor(sensor_mask_xyz, record=['p'])

    simulation_options = SimulationOptions(
        pml_auto=True, pml_inside=False, save_to_disk=True, data_cast='single',
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=gpu)
    inputs = {
        'kgrid': kgrid, 'source': source, 'sensor': sensor, 'medium': medium,
        'simulation_options': simulation_options, 'execution_options': execution_options,
    }
    logger.info("Running sparse-sensor k-wave simulation (%d sensor voxels)...",
                int(xyz_sensor_indices.shape[0]))
    try:
        output = kspaceFirstOrder3D(**deepcopy(inputs))
    finally:
        for fpath in [simulation_options.input_filename, simulation_options.output_filename]:
            with contextlib.suppress(OSError):
                pathlib.Path(fpath).unlink(missing_ok=True)

    p_sensor = np.asarray(output['p'])
    if p_sensor.ndim == 1:
        p_sensor = p_sensor.reshape(-1, 1)
    # p_sensor shape expected: (Nt, n_sensors)
    return p_sensor, float(kgrid.dt), xyz_sensor_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_total = time.time()
    print("=" * 72)
    print("GLADYS TIME-GATED TARGET PRESSURE (sparse sensor diagnostic)")
    print("=" * 72)

    # 1. Load MRI + labels
    print(f"\n[1] Loading MRI: {MRI_PATH}")
    if not MRI_PATH.exists():
        print(f"ERROR: MRI not found at {MRI_PATH}")
        sys.exit(1)
    if not LABEL_NIFTI_PATH.exists():
        print(f"ERROR: nnU-Net labels not found at {LABEL_NIFTI_PATH}")
        sys.exit(1)
    volume = load_nifti_as_xarray(MRI_PATH)
    seg_method = PreSegmented(label_nifti_path=str(LABEL_NIFTI_PATH))

    # 2. Segment MRI to find brain centroid + approach axis
    print("\n[2] Segmenting MRI for target geometry...")
    seg_labels = seg_method._segment(volume)
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()
    brain_keys = [k for k in ("csf", "gray_matter", "white_matter") if k in material_idx]
    brain_mask = np.zeros(seg_arr.shape, dtype=bool)
    for k in brain_keys:
        brain_mask |= (seg_arr == material_idx[k])
    dim_names = list(volume.dims)
    coord_arrays = {d: volume.coords[d].to_numpy() for d in dim_names}
    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"    Target (brain centroid): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]] for ax in range(3)
    ])
    max_skull_dist_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_dist_per_axis))
    approach_dim = dim_names[approach_axis]
    print(f"    Approach axis: {approach_dim} (axis {approach_axis})")

    # 3. Build array + transform
    print(f"\n[3] Building {N_ELEMENTS}-element hemispherical array...")
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
    aperture_center_mm = positions.mean(axis=0)
    print(f"    Aperture center: ({aperture_center_mm[0]:.1f}, {aperture_center_mm[1]:.1f}, {aperture_center_mm[2]:.1f}) mm")

    # 4. Build sim grid (same as run_gladys_nnunet.py)
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
    print(f"\n[4] Sim grid: {grid_shape} ({int(np.prod(grid_shape)):,d} voxels)")

    # Resample MRI onto sim grid
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

    print("    Building sim_params (seg_params on sim grid)...")
    sim_params = seg_method.seg_params(sim_volume)

    # Replace air with water
    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask_sim = sim_seg_arr == material_idx["air"]
    water_mat = seg_method.materials["water"]
    if air_mask_sim.any():
        sim_params["sound_speed"].data[air_mask_sim] = water_mat.sound_speed
        sim_params["density"].data[air_mask_sim] = water_mat.density
        sim_params["attenuation"].data[air_mask_sim] = water_mat.attenuation

    # 5. Delays + timing
    target = Point(position=target_mm.copy(), id="brain_center", name="Brain Center", units="mm")
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
    max_dist_m = max_dist_mm * 1e-3
    t_end = max_dist_m / C0 * T_END_SAFETY
    c_max = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max
    print(f"\n[5] t_end={t_end*1e6:.1f} us | dt={dt*1e9:.2f} ns")

    print("\n[6] SimulationCorrected delays (phase correction)...")
    sim_corrected = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
    t0 = time.time()
    delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
    print(f"    Phase correction done in {time.time()-t0:.1f}s")

    # 6. Sensor voxel selection
    # Build 5 sensor voxels:
    #   s0: target
    #   s1: aperture center
    #   s2: midway (target + 0.5 * (aperture_ctr - target))
    #   s3: quarter-way toward aperture (target + 0.25 * ...) [nearer target]
    #   s4: off-axis control (target offset 15 mm perpendicular to approach axis)
    #
    # We snap each world position to nearest sim-grid voxel and build a binary
    # mask in params dim order.
    axis_vec = aperture_center_mm - target_mm
    axis_len = float(np.linalg.norm(axis_vec))
    axis_unit = axis_vec / axis_len

    # Perpendicular direction for off-axis control: use a unit vector
    # perpendicular to the approach axis (in one of the two non-approach dims).
    perp_candidates = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    # Choose one not parallel to axis_unit.
    perp = None
    for cand in perp_candidates:
        cross = np.cross(axis_unit, cand)
        if np.linalg.norm(cross) > 0.5:
            # project out axis component
            p = cand - axis_unit * np.dot(cand, axis_unit)
            p_norm = np.linalg.norm(p)
            if p_norm > 1e-6:
                perp = p / p_norm
                break
    assert perp is not None

    sensor_world_mm = [
        ("target",        target_mm.copy()),
        ("aperture_ctr",  aperture_center_mm.copy()),
        ("midway",        target_mm + 0.5 * axis_vec),
        ("quarterway",    target_mm + 0.75 * axis_vec),  # 75% toward aperture (closer to aperture)
        ("off_axis_15mm", target_mm + 15.0 * perp),
    ]

    sim_dim_names = list(sim_params.dims)
    sim_coord_params = {d: sim_params.coords[d].to_numpy() for d in sim_dim_names}
    sensor_mask_params = np.zeros(tuple(len(sim_coord_params[d]) for d in sim_dim_names), dtype=np.int32)
    # Map each sensor world position -> params-order voxel idx
    sensor_params_idx: list[tuple[int, int, int]] = []  # in params dim order
    sensor_xyz_idx: list[tuple[int, int, int]] = []     # in x,y,z order
    sensor_names = []
    sensor_actual_mm: list[np.ndarray] = []
    for name, pos_mm in sensor_world_mm:
        idx_in_params = []
        idx_in_xyz = [0, 0, 0]
        actual = np.zeros(3)
        for params_ax, dim in enumerate(sim_dim_names):
            cv = sim_coord_params[dim]
            xyz_ax = {'x': 0, 'y': 1, 'z': 2}[dim]
            val = pos_mm[xyz_ax]
            i = int(np.argmin(np.abs(cv - val)))
            idx_in_params.append(i)
            idx_in_xyz[xyz_ax] = i
            actual[xyz_ax] = float(cv[i])
        sensor_mask_params[tuple(idx_in_params)] = 1
        sensor_params_idx.append(tuple(idx_in_params))
        sensor_xyz_idx.append(tuple(idx_in_xyz))
        sensor_names.append(name)
        sensor_actual_mm.append(actual)

    print("\n[7] Sensor voxels (world mm):")
    for name, actual in zip(sensor_names, sensor_actual_mm):
        dist_from_target = float(np.linalg.norm(actual - target_mm))
        dist_from_apct = float(np.linalg.norm(actual - aperture_center_mm))
        print(f"    {name:>14s}: ({actual[0]:6.1f},{actual[1]:6.1f},{actual[2]:6.1f}) mm  "
              f"| d(target)={dist_from_target:5.1f} mm  d(ap_ctr)={dist_from_apct:5.1f} mm")

    n_unique_mask = int(sensor_mask_params.sum())
    print(f"    Unique voxels in mask: {n_unique_mask} (of {len(sensor_names)} requested; collisions likely on coarse grid)")

    apod = np.ones(arr.numelements())

    # ---- Run SIM A (corrected + skull) -----------------------------------
    print("\n" + "=" * 72)
    print("[SIM A] Corrected + heterogeneous skull (nnU-Net), sparse sensor")
    print("=" * 72)
    t0 = time.time()
    p_A, dt_A, xyz_order_A = run_sparse_sensor_sim(
        arr=arr, params=sim_params,
        delays=delays_corrected, apod=apod,
        sensor_mask_params_order=sensor_mask_params,
        freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, ref_values_only=False,
    )
    print(f"    SIM A complete in {time.time()-t0:.1f}s  (p shape={p_A.shape}, dt={dt_A*1e9:.2f} ns)")

    # ---- Run SIM C (water reference) -------------------------------------
    print("\n" + "=" * 72)
    print("[SIM C] Geometric + homogeneous water, sparse sensor")
    print("=" * 72)
    t0 = time.time()
    p_C, dt_C, xyz_order_C = run_sparse_sensor_sim(
        arr=arr, params=sim_params,
        delays=delays_geo, apod=apod,
        sensor_mask_params_order=sensor_mask_params,
        freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, ref_values_only=True,
    )
    print(f"    SIM C complete in {time.time()-t0:.1f}s  (p shape={p_C.shape}, dt={dt_C*1e9:.2f} ns)")

    # ---- Map sensor_names to output columns ------------------------------
    # Output column order: k-wave Fortran traversal of nonzero voxels in
    # xyz-transposed mask. run_sparse_sensor_sim returns xyz_sensor_indices
    # in that same order. We need to match each sensor_name to its column.
    def _col_for_sensor(xyz_idx_tuple, xyz_order_arr):
        matches = np.where(
            (xyz_order_arr[:, 0] == xyz_idx_tuple[0]) &
            (xyz_order_arr[:, 1] == xyz_idx_tuple[1]) &
            (xyz_order_arr[:, 2] == xyz_idx_tuple[2])
        )[0]
        if len(matches) == 0:
            return None
        return int(matches[0])

    # ---- Analysis --------------------------------------------------------
    pulse_dur = CYCLES / FREQ_HZ  # seconds
    print(f"\n[8] Analysis: pulse_dur = {pulse_dur*1e6:.1f} us")
    print(f"    Focal window = tof +/- 2*pulse_dur = tof +/- {2*pulse_dur*1e6:.1f} us")
    print(f"    tof reference: distance from aperture_ctr to sensor / c0 (c0={C0} m/s)")

    # Compute tof for each sensor from aperture center, in seconds
    tofs_s = {}
    for name, actual in zip(sensor_names, sensor_actual_mm):
        d_m = float(np.linalg.norm(actual - aperture_center_mm)) * 1e-3
        tofs_s[name] = d_m / C0

    rows = []
    for sim_label, p_arr, dt_sim, xyz_order in [
        ("A (skull)", p_A, dt_A, xyz_order_A),
        ("C (water)", p_C, dt_C, xyz_order_C),
    ]:
        Nt = p_arr.shape[0]
        t_axis = np.arange(Nt) * dt_sim  # seconds
        for name, xyz_idx in zip(sensor_names, sensor_xyz_idx):
            col = _col_for_sensor(xyz_idx, xyz_order)
            if col is None:
                rows.append({
                    "sim": sim_label, "sensor": name,
                    "p_focal_window_Pa": np.nan, "p_allt_Pa": np.nan, "ratio": np.nan,
                    "tof_us": tofs_s[name] * 1e6,
                    "note": "voxel collision; no dedicated column",
                })
                continue
            p_t = p_arr[:, col]
            p_abs = np.abs(p_t)
            p_allt = float(p_abs.max())

            tof = tofs_s[name]
            t_lo = tof - 2.0 * pulse_dur
            t_hi = tof + 2.0 * pulse_dur
            in_win = (t_axis >= t_lo) & (t_axis <= t_hi)
            if in_win.any():
                p_fw = float(p_abs[in_win].max())
            else:
                p_fw = np.nan
            ratio = (p_fw / p_allt) if (p_allt > 0 and np.isfinite(p_fw)) else np.nan
            rows.append({
                "sim": sim_label, "sensor": name,
                "p_focal_window_Pa": p_fw, "p_allt_Pa": p_allt, "ratio": ratio,
                "tof_us": tof * 1e6,
                "note": "",
            })

    # Print results table
    print("\n" + "=" * 92)
    print("TIME-GATED vs ALL-TIME PEAK PRESSURE")
    print("=" * 92)
    print(f"{'sim':<12}{'sensor':<16}{'tof(us)':>9}{'p_focal(Pa)':>15}{'p_allt(Pa)':>15}{'ratio':>9}  note")
    print("-" * 92)
    for r in rows:
        pfw = r["p_focal_window_Pa"]
        pal = r["p_allt_Pa"]
        ratio = r["ratio"]
        pfw_s = f"{pfw:.4g}" if np.isfinite(pfw) else "NaN"
        pal_s = f"{pal:.4g}" if np.isfinite(pal) else "NaN"
        ratio_s = f"{ratio:.3f}" if np.isfinite(ratio) else "NaN"
        print(f"{r['sim']:<12}{r['sensor']:<16}{r['tof_us']:>9.1f}{pfw_s:>15}{pal_s:>15}{ratio_s:>9}  {r['note']}")
    print("=" * 92)

    print("\nInterpretation hints:")
    print("  - target: ratio near 1.0 means all-time peak at target IS the focal arrival")
    print("    (focus is forming cleanly). Ratio << 1.0 means standing waves elsewhere")
    print("    in time dominate (bad: target's p_max is noise, not focus).")
    print("  - aperture_ctr / near-aperture voxels: expect ratio << 1.0 in SIM A because")
    print("    their all-time peak is the direct source pulse or skull reverberation,")
    print("    not a focal arrival. Their focal-window peak is what we actually want.")
    print("  - SIM C (water) is a sanity check: ratios at target should be ~1.0, and")
    print("    should be noticeably higher than SIM A at target if skull is adding")
    print("    spurious peaks outside the focal window.")

    elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
