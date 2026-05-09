#!/usr/bin/env python3
"""Per-element path-length-vs-amplitude analysis on GU008 using BOTH a raw
nnU-Net skull mask and a 1mm-dilated skull mask.

Context: the earlier ``incidence_angle_amplitude_gu008.py`` run found:
  - path-length AUC = 0.988 (strong predictor of dead elements)
  - entry-angle AUC = 0.123 (no predictive power)
  - 24 elements had "no skull intersection" along their rays.

A follow-up ray-sanity investigation found 8 of those 24 are "grazers"
(<1 mm from skull): their rays miss by sub-voxel geometry, and dilating
the skull mask by 1 mm rescues them. This script re-runs the path-length
predictor experiment with both the raw mask and the 1mm-dilated mask so
we can decide whether ``SkullPathApodization`` should default to using a
dilated skull mask.

This is a LOCAL diagnostic: it does not modify src/ and is not committed.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import xarray as xa
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation
from scipy.signal import hilbert


# -------------------------------------------------------------------
# Patch kwave's mis-formatted logging.log() calls.
# -------------------------------------------------------------------
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from openlifu.bf.apod_methods.skull_incidence import SkullIncidenceApodization  # noqa: E402
from openlifu.sim.kwave_if import run_point_source_simulation  # noqa: E402
from openlifu.util.units import getunitconversion  # noqa: E402
from run_gladys_nnunet import (  # noqa: E402
    PreSegmented,
    create_hemispherical_array,
    load_nifti_as_xarray,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dilated_path_gu008")

# ---- Must match incidence_angle_amplitude_gu008.py --------------------------
MRI_PATH = Path.home() / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)
LABEL_NIFTI_PATH = Path.home() / "Data/openlifu-validation/results/GU008_nnunet_labels.nii.gz"
PLOT_PATH = Path.home() / "Data/openlifu-validation/results/dilated_path_analysis_gu008.png"

N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
GRID_SPACING_MM = 0.5
RECIPROCAL_CFL = 0.3
RECIPROCAL_N_CYCLES = 3
C0 = 1500.0
GRID_MARGIN_MM = 25.0

RAY_STEP_MM = 0.25


def ray_skull_geometry(apod, el_pos_axes, target_pos_axes):
    """Return (entry_angle_deg, skull_path_length_mm, hit_found).

    Same as in ``incidence_angle_amplitude_gu008.py``: marches the SDF
    looking for outside->inside then inside->outside sign changes.
    """
    ray_vec = target_pos_axes - el_pos_axes
    total_dist = float(np.linalg.norm(ray_vec))
    if total_dist == 0.0:
        return float("nan"), 0.0, False
    ray_dir = ray_vec / total_dist

    step = RAY_STEP_MM
    n_steps = max(2, int(np.ceil(total_dist / step)) + 1)
    ts = np.linspace(0.0, total_dist, n_steps)

    entry_t = None
    entry_point = None
    prev_sdf = None
    prev_point = None
    prev_t = None
    for t in ts:
        p = el_pos_axes + ray_dir * t
        s = apod._sample_sdf(p)
        if prev_sdf is not None and prev_sdf > 0.0 and s <= 0.0:
            denom = prev_sdf - s
            alpha = 0.0 if denom == 0.0 else float(np.clip(prev_sdf / denom, 0.0, 1.0))
            entry_point = prev_point + alpha * (p - prev_point)
            entry_t = prev_t + alpha * (t - prev_t)
            break
        if prev_sdf is None and s <= 0.0:
            entry_point = p.copy()
            entry_t = float(t)
            break
        prev_sdf = s
        prev_point = p
        prev_t = float(t)

    if entry_point is None:
        return float("nan"), 0.0, False

    n = apod._sample_grad(entry_point)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        entry_angle_deg = float("nan")
    else:
        n = n / n_norm
        ray_dir_for_dot = ray_dir.copy()
        for axis_i, asc in enumerate(apod._coord_ascending):
            if not asc:
                ray_dir_for_dot[axis_i] = -ray_dir_for_dot[axis_i]
        cos_theta = float(np.clip(abs(np.dot(ray_dir_for_dot, n)), 0.0, 1.0))
        entry_angle_deg = float(np.degrees(np.arccos(cos_theta)))

    path_step = min(RAY_STEP_MM, 0.2)
    t_start = entry_t
    t_end = total_dist
    n_fwd = max(2, int(np.ceil((t_end - t_start) / path_step)) + 1)
    ts_fwd = np.linspace(t_start, t_end, n_fwd)
    prev_sdf_fwd = None
    prev_t_fwd = None
    exit_t = None
    for t in ts_fwd:
        p = el_pos_axes + ray_dir * t
        s = apod._sample_sdf(p)
        if prev_sdf_fwd is not None and prev_sdf_fwd <= 0.0 and s > 0.0:
            denom = s - prev_sdf_fwd
            alpha = 0.0 if denom == 0.0 else float(np.clip(-prev_sdf_fwd / denom, 0.0, 1.0))
            exit_t = prev_t_fwd + alpha * (t - prev_t_fwd)
            break
        prev_sdf_fwd = s
        prev_t_fwd = float(t)
    if exit_t is None:
        skull_path_length_mm = float(total_dist - entry_t)
    else:
        skull_path_length_mm = float(exit_t - entry_t)
    return entry_angle_deg, skull_path_length_mm, True


def rank_auc(pred, is_dead_arr):
    """AUC for "higher pred => more likely to be dead"."""
    dead_vals = pred[is_dead_arr]
    live_vals = pred[~is_dead_arr]
    if len(dead_vals) == 0 or len(live_vals) == 0:
        return 0.5
    n_pairs = len(dead_vals) * len(live_vals)
    wins = 0
    ties = 0
    for d in dead_vals:
        for l in live_vals:
            if d > l:
                wins += 1
            elif d == l:
                ties += 1
    return (wins + 0.5 * ties) / n_pairs


def print_histogram(paths, label, edges=None):
    if edges is None:
        edges = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 9999.0]
    print(f"\n  Histogram of skull path lengths ({label}, mm):")
    print(f"    {'bin':>16} {'count':>6}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        n = int(((paths >= lo) & (paths < hi)).sum())
        bin_label = f"[{lo:.1f}-{hi:.1f})"
        print(f"    {bin_label:>16s} {n:>6d}")


def main() -> int:
    t_total = time.time()
    print("=" * 78)
    print("Per-element path-length-vs-amplitude (raw vs 1mm-dilated skull), GU008")
    print("=" * 78)

    if not MRI_PATH.exists():
        print(f"ERROR: MRI not found: {MRI_PATH}")
        return 1
    if not LABEL_NIFTI_PATH.exists():
        print(f"ERROR: nnU-Net labels not found: {LABEL_NIFTI_PATH}")
        return 1

    print(f"\n[1] Loading MRI: {MRI_PATH}")
    volume = load_nifti_as_xarray(MRI_PATH)

    print(f"[2] Loading nnU-Net labels: {LABEL_NIFTI_PATH}")
    seg_method = PreSegmented(label_nifti_path=str(LABEL_NIFTI_PATH))
    seg_labels = seg_method._segment(volume)
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()

    dim_names = list(volume.dims)
    coord_arrays = {d: volume.coords[d].to_numpy() for d in dim_names}
    brain_keys = [k for k in ("csf", "gray_matter", "white_matter") if k in material_idx]
    brain_mask = np.zeros(seg_arr.shape, dtype=bool)
    for k in brain_keys:
        brain_mask |= (seg_arr == material_idx[k])
    if brain_mask.sum() == 0:
        brain_mask = seg_arr == material_idx["tissue"]
    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"    Brain center target: ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    skull_mask_full = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask_full)
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
    print(f"    Approach axis: {approach_dim} (axis {approach_axis})")

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
            el_angle = -np.arctan2(n[1], np.sqrt(n[0] ** 2 + n[2] ** 2))
            el.orientation = np.array([az, el_angle, 0.0])

    positions = arr.get_positions(units="mm")
    print(f"[3] Array: 64 elements baked to world frame; radius ~{RADIUS_MM:.0f} mm")

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
    grid_shape_xyz = tuple(len(sim_coords[d]) for d in ["x", "y", "z"])
    print(f"[4] Sim grid: {grid_shape_xyz} = {int(np.prod(grid_shape_xyz)):,d} voxels "
          f"@ {GRID_SPACING_MM} mm")

    orig_coords = [volume.coords[d].to_numpy() for d in volume.dims]
    interp = RegularGridInterpolator(
        orig_coords, volume.to_numpy(),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    sim_coord_arrays = [sim_coords[d].data for d in ["x", "y", "z"]]
    mg = np.meshgrid(*sim_coord_arrays, indexing="ij")
    query_pts = np.stack([m.ravel() for m in mg], axis=-1)
    resampled_data = interp(query_pts).reshape(grid_shape_xyz).astype(np.float32)
    sim_volume = xa.DataArray(
        resampled_data, dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )

    print("    Running PreSegmented.seg_params on sim grid ...")
    t0 = time.time()
    sim_params = seg_method.seg_params(sim_volume)
    print(f"    seg_params done in {time.time() - t0:.1f}s")

    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    water_mat = seg_method.materials["water"]
    air_mask = sim_seg_arr == material_idx["air"]
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation
    n_skull_sim = int((sim_seg_arr == material_idx["skull"]).sum())
    print(f"    Bone fraction in sim grid: {100.0 * n_skull_sim / sim_seg_arr.size:.2f}%")

    coord_dims = list(sim_params.coords.dims)
    coord_units = sim_params[coord_dims[0]].attrs.get("units", "mm")
    _DIM_IDX = {"x": 0, "y": 1, "z": 2}
    scl_m_to_coord = getunitconversion("m", coord_units)
    scl_to_m = getunitconversion(coord_units, "m")

    matrix = np.eye(4)
    element_positions_raw = np.array([
        el.get_position(units="m", matrix=matrix) * scl_m_to_coord
        for el in arr.elements
    ])
    target_pos_raw = np.array(target_mm, dtype=float)

    coord_axis_arrays = [sim_params.coords[dim].to_numpy() for dim in coord_dims]
    grid_shape_params = tuple(len(c) for c in coord_axis_arrays)

    sensor_indices: list[tuple[int, int, int]] = []
    for epos_xyz in element_positions_raw:
        idx = []
        for dim_i, dim_name in enumerate(coord_dims):
            coord_vals = coord_axis_arrays[dim_i]
            pos_component = epos_xyz[_DIM_IDX[dim_name]]
            nearest_idx = int(np.argmin(np.abs(coord_vals - pos_component)))
            idx.append(nearest_idx)
        sensor_indices.append(tuple(idx))

    sensor_mask = np.zeros(grid_shape_params, dtype=int)
    for idx in sensor_indices:
        sensor_mask[idx] = 1
    n_unique_sensor_vox = int(sensor_mask.sum())
    print(f"[5] Sensor mask: {n_unique_sensor_vox} unique voxels "
          f"(out of {N_ELEMENTS}); collisions: {N_ELEMENTS - n_unique_sensor_vox}")

    target_idx = tuple(
        int(np.argmin(np.abs(coord_axis_arrays[dim_i] - target_pos_raw[_DIM_IDX[dim_name]])))
        for dim_i, dim_name in enumerate(coord_dims)
    )
    source_mask = np.zeros(grid_shape_params, dtype=int)
    source_mask[target_idx] = 1

    dists_m = np.linalg.norm(element_positions_raw - target_pos_raw, axis=1) * scl_to_m
    max_dist_m = float(np.max(dists_m))
    t_end_needed = max_dist_m / C0 * 1.5 + RECIPROCAL_N_CYCLES / FREQ_HZ

    print(f"[6] Reciprocal sim: point source @ target, sensors @ elements ...")
    print(f"    freq={FREQ_HZ / 1e3:.0f} kHz, cfl={RECIPROCAL_CFL}, "
          f"t_end={t_end_needed * 1e6:.1f} us")
    t0 = time.time()
    sensor_data, dt = run_point_source_simulation(
        params=sim_params,
        source_mask=source_mask,
        sensor_mask=sensor_mask,
        freq=FREQ_HZ,
        n_cycles=RECIPROCAL_N_CYCLES,
        sound_speed_ref=C0,
        cfl=RECIPROCAL_CFL,
        gpu=True,
        t_end=t_end_needed,
    )
    t_sim = time.time() - t0
    Nt, n_cols = sensor_data.shape
    print(f"    Reciprocal sim done in {t_sim:.1f}s (sensor_data={sensor_data.shape}, "
          f"dt={dt * 1e9:.2f} ns)")

    perm_to_xyz = [coord_dims.index(d) for d in ["x", "y", "z"]]
    sensor_mask_xyz = np.transpose(sensor_mask, perm_to_xyz)
    grid_shape_xyz_mask = sensor_mask_xyz.shape
    nonzero_xyz = list(zip(*np.nonzero(sensor_mask_xyz)))

    def fortran_linear_index(idx, shape):
        lin = idx[0]
        stride = shape[0]
        for d in range(1, len(shape)):
            lin += idx[d] * stride
            stride *= shape[d]
        return lin

    nonzero_with_fortran = [
        (fortran_linear_index(idx, grid_shape_xyz_mask), idx) for idx in nonzero_xyz
    ]
    nonzero_with_fortran.sort(key=lambda x: x[0])
    sorted_nonzero = [item[1] for item in nonzero_with_fortran]
    voxel_to_col = {idx: col for col, idx in enumerate(sorted_nonzero)}

    print(f"[7] Extracting per-element amplitudes ...")
    amplitudes = np.zeros(N_ELEMENTS)
    c_max = float(np.max(sim_params["sound_speed"].to_numpy()))
    c_max = max(c_max, C0)

    for el_i, sensor_idx in enumerate(sensor_indices):
        sensor_idx_xyz = tuple(sensor_idx[i] for i in perm_to_xyz)
        col = voxel_to_col[sensor_idx_xyz]
        p_t = sensor_data[:, col]
        envelope = np.abs(hilbert(p_t))

        earliest = (
            np.linalg.norm(element_positions_raw[el_i] - target_pos_raw)
            * scl_to_m / c_max
        )
        gate_start = max(0, int((earliest - 2 * dt) / dt))
        if gate_start >= len(envelope):
            gate_start = 0
        peak_sample = gate_start + int(np.argmax(envelope[gate_start:]))
        amplitudes[el_i] = float(envelope[peak_sample])

    amp_mean = float(np.mean(amplitudes))
    amp_median = float(np.median(amplitudes))
    amp_min = float(np.min(amplitudes))
    amp_max = float(np.max(amplitudes))
    dead_threshold = amp_median / 10.0
    print(f"    amp: mean={amp_mean:.3g} median={amp_median:.3g} "
          f"min={amp_min:.3g} max={amp_max:.3g} dyn={amp_max/max(amp_min,1e-30):.1f}x")
    print(f"    dead threshold (median/10) = {dead_threshold:.3g}")
    is_dead = amplitudes < dead_threshold
    n_dead = int(is_dead.sum())
    print(f"    {n_dead}/{N_ELEMENTS} dead elements")

    # ------ Build raw + 1mm-dilated skull masks and ray-cast twice. ---------
    print(f"\n[8] Building raw + 1mm-dilated skull masks ...")
    skull_raw_sim = (sim_seg_arr == material_idx["skull"]).astype(np.uint8)
    # 3x3x3 structuring element = 1 voxel in each direction = 0.5 mm here.
    # Two iterations of that would be 1 mm; one iteration is 0.5 mm.
    # Ray-sanity agent recommended "dilate by 1 mm". At 0.5 mm grid, that
    # means 2 iterations of a 3x3x3 structuring element (or 1 iteration of
    # a 5x5x5). The task spec phrases it as "3x3x3 structuring element = 1
    # mm at 0.5mm grid" -- but mathematically a single 3x3x3 dilation
    # expands by 1 voxel = 0.5 mm. We'll do 2 iterations to honor the 1 mm
    # intent, and also print the 1-iteration (0.5 mm) case for reference.
    struct_3 = np.ones((3, 3, 3), dtype=bool)
    skull_dil_05mm = binary_dilation(
        skull_raw_sim.astype(bool), structure=struct_3, iterations=1
    ).astype(np.uint8)
    skull_dil_1mm = binary_dilation(
        skull_raw_sim.astype(bool), structure=struct_3, iterations=2
    ).astype(np.uint8)
    print(f"    raw skull voxels:       {int(skull_raw_sim.sum()):,d}")
    print(f"    dilated 0.5 mm voxels:  {int(skull_dil_05mm.sum()):,d}")
    print(f"    dilated 1.0 mm voxels:  {int(skull_dil_1mm.sum()):,d}")

    def build_apod(mask_np):
        mask_da = xa.DataArray(
            mask_np,
            dims=sim_seg.dims,
            coords={d: sim_seg.coords[d] for d in sim_seg.dims},
        )
        return SkullIncidenceApodization(
            skull_mask=mask_da,
            min_angle_deg=0.0,
            rolloff_angle_deg=90.0,
            step_mm=RAY_STEP_MM,
            coord_units=coord_units,
        )

    t_sdf0 = time.time()
    apod_raw = build_apod(skull_raw_sim)
    print(f"    raw SDF cached in {time.time() - t_sdf0:.1f}s")
    t_sdf0 = time.time()
    apod_dil = build_apod(skull_dil_1mm)
    print(f"    1mm-dilated SDF cached in {time.time() - t_sdf0:.1f}s")

    print(f"\n[9] Ray-casting per-element through raw + dilated SDFs ...")

    sdf_dims = list(sim_seg.dims)

    def xyz_to_dims(vec3_xyz, dims):
        out = np.empty(3, dtype=float)
        for j, d in enumerate(dims):
            out[j] = vec3_xyz[_DIM_IDX[d]]
        return out

    paths_raw = np.zeros(N_ELEMENTS)
    paths_dil = np.zeros(N_ELEMENTS)
    hit_raw = np.zeros(N_ELEMENTS, dtype=bool)
    hit_dil = np.zeros(N_ELEMENTS, dtype=bool)

    target_axes = xyz_to_dims(target_pos_raw, sdf_dims)
    for el_i in range(N_ELEMENTS):
        el_axes = xyz_to_dims(element_positions_raw[el_i], sdf_dims)
        _, p_r, h_r = ray_skull_geometry(apod_raw, el_axes, target_axes)
        _, p_d, h_d = ray_skull_geometry(apod_dil, el_axes, target_axes)
        paths_raw[el_i] = p_r
        paths_dil[el_i] = p_d
        hit_raw[el_i] = h_r
        hit_dil[el_i] = h_d

    n_zero_raw = int((paths_raw == 0.0).sum())
    n_zero_dil = int((paths_dil == 0.0).sum())
    n_miss_raw = int((~hit_raw).sum())
    n_miss_dil = int((~hit_dil).sum())
    print(f"    raw:     path=0 count = {n_zero_raw:3d},  no-hit count = {n_miss_raw}")
    print(f"    dilated: path=0 count = {n_zero_dil:3d},  no-hit count = {n_miss_dil}")
    print(f"    rescued by 1 mm dilation: {n_zero_raw - n_zero_dil} elements")

    print_histogram(paths_raw, "raw")
    print_histogram(paths_dil, "1mm-dilated")

    # ------ AUCs for "high path -> dead" predictor. -------------------------
    auc_raw = rank_auc(paths_raw, is_dead)
    auc_dil = rank_auc(paths_dil, is_dead)
    print("\n[10] Dead-vs-path-length rank-AUC (higher path => more dead):")
    print(f"    raw mask:          AUC = {auc_raw:.3f}  (|delta|={abs(auc_raw-0.5):.3f})")
    print(f"    1mm-dilated mask:  AUC = {auc_dil:.3f}  (|delta|={abs(auc_dil-0.5):.3f})")
    if auc_dil > auc_raw + 1e-4:
        verdict = "DILATED improves"
    elif auc_dil < auc_raw - 1e-4:
        verdict = "DILATED DEGRADES"
    else:
        verdict = "no change"
    print(f"    verdict: {verdict}")

    # ------ Plot -----------------------------------------------------------
    print(f"\n[11] Plotting to {PLOT_PATH} ...")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    amp_floor = max(amp_min, 1e-30)
    plot_amps = np.where(amplitudes > 0, amplitudes, amp_floor)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, paths, auc_val, title in [
        (ax1, paths_raw, auc_raw, "Raw skull mask"),
        (ax2, paths_dil, auc_dil, "1 mm-dilated skull mask"),
    ]:
        zero_sel = paths == 0.0
        nz_sel = ~zero_sel
        ax.scatter(
            paths[nz_sel], plot_amps[nz_sel],
            c=["tab:red" if d else "tab:blue" for d in is_dead[nz_sel]],
            s=70, edgecolors="k", linewidths=0.5,
            label="path > 0",
        )
        ax.scatter(
            np.full(zero_sel.sum(), -0.5), plot_amps[zero_sel],
            c=["tab:red" if d else "tab:blue" for d in is_dead[zero_sel]],
            marker="x", s=70,
            label=f"path = 0 (n={int(zero_sel.sum())})",
        )
        ax.axhline(dead_threshold, color="k", ls="--", lw=1,
                   label=f"dead thr = {dead_threshold:.2g}")
        ax.set_yscale("log")
        ax.set_xlabel("Skull path length (mm)  [x=-0.5: path=0]")
        ax.set_title(
            f"{title}\npath=0 count = {int(zero_sel.sum())},  "
            f"AUC(dead|path) = {auc_val:.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        for el_i in range(N_ELEMENTS):
            if is_dead[el_i]:
                x = -0.5 if paths[el_i] == 0.0 else paths[el_i]
                ax.annotate(str(el_i), (x, plot_amps[el_i]),
                            fontsize=7, alpha=0.7)

    ax1.set_ylabel("Per-element received amplitude (a_i, reciprocal sim)")
    fig.suptitle(
        "GU008: per-element amplitude vs skull path length "
        f"(raw vs 1 mm dilation; {n_dead}/{N_ELEMENTS} dead)"
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {PLOT_PATH}")

    # ------ Per-element table: raw vs dilated path. -------------------------
    print("\n" + "=" * 78)
    print(" Per-element table (sorted by amplitude ASC)")
    print("=" * 78)
    print(f"{'rank':>4} {'el':>3} {'dist_mm':>8} {'a_i':>10} "
          f"{'raw_mm':>8} {'dil_mm':>8} {'delta_mm':>9} {'dead?':>6}")
    sort_idx = np.argsort(amplitudes)
    for rank, el_i in enumerate(sort_idx):
        d_mm = float(np.linalg.norm(element_positions_raw[el_i] - target_pos_raw))
        dead = "YES" if is_dead[el_i] else ""
        delta = paths_dil[el_i] - paths_raw[el_i]
        print(f"{rank:>4d} {el_i:>3d} {d_mm:8.2f} {amplitudes[el_i]:10.3g} "
              f"{paths_raw[el_i]:8.2f} {paths_dil[el_i]:8.2f} {delta:9.2f} "
              f"{dead:>6s}")

    print(f"\nTotal elapsed: {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
