#!/usr/bin/env python3
"""SkullPathApodization alpha sweep on GU010.

Tests whether post-simulation element apodization with a HIGHER effective
alpha_bone (consistent with the per-element scatter finding on GU008 where
observed amplitude drops ~2.4x faster than Beer's-law with alpha=4.29 dB/cm,
implying an "effective" alpha closer to ~10 dB/cm) delivers focal-quality
gains on GU010.

Apodization is a POST-simulation per-element weight, so alpha_bone here is
independent of medium attenuation (which stays at its default 3.5 dB/cm/MHz).

For each alpha in [3.5, 8.4, 15.0] dB/cm:
  - Build SkullPathApodization(alpha_bone_db_per_cm=alpha, max_path_mm=None)
  - Compute per-element weights + log stats (active count, mean, sum, min/max)
  - Run SIM C (water reference, ref_values_only=True) with those weights
  - Run SIM A (phase-corrected + skull) with those weights (medium alpha unchanged)
  - Probe metrics: p@target, p_focal_window_water_gate@target, p_allt@target,
    spatial-max p_focal_window (water-gate, per-voxel gate) within a 20 mm cube
    around target with stride 2 (to keep sensor count tractable), offset mm.
  - Diagnostic: ratio = p_focal_window_water_gate / p_allt at target.

Gate center for skull sims is the alpha=3.5 water reference's target-peak
time (physical TOF is apod-invariant in homogeneous water), to ensure an
identical gate across all 3 alpha values.

Local / uncommitted.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

# --- Cube-probe configuration (must be set BEFORE importing subject script) ---
# 20 mm cube with stride 2 at 0.5 mm grid -> ~21^3 ~= 9.3k cube sensors.
# This matches the spatial-search baseline referenced in the comparison
# ("Spatial offset: 20.32 mm (20mm cube w/ per-voxel gate)").
os.environ["EXPANDED_TARGET_PROBE"] = "1"
os.environ["EXPANDED_TARGET_HALF_MM"] = "20.0"
os.environ["CUBE_PROBE_STRIDE"] = "2"
os.environ.setdefault("SIDECAR_MAX_VOXELS", "500")
os.environ.pop("OPENLIFU_DISABLE_TIMEGATED_PROBE", None)
os.environ.pop("USE_TARGET_GATE_FOR_CUBE", None)  # force per-voxel water-gate

# --- kwave logging workaround (same as alpha_bone_sweep_gu010.py) ---
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

import numpy as np
import xarray as xa
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.bf.apod_methods.skull_path import SkullPathApodization
from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.sim.kwave_if import run_simulation
from openlifu.sim.sim_setup import SimSetup

# Re-use geometry + probe helpers from the subject script
sys.path.insert(0, os.path.dirname(__file__))
from run_gladys_nnunet_subject import (  # type: ignore
    APERTURE_BAND_RADIUS_MM,
    APERTURE_MM,
    AMPLITUDE,
    C0,
    CFL,
    CYCLES,
    ELEMENT_SIZE_MM,
    FREQ_HZ,
    GRID_MARGIN_MM,
    GRID_SPACING_MM,
    N_ELEMENTS,
    RADIUS_MM,
    T_END_SAFETY,
    PreSegmented,
    _run_timegated_probe,
    _save_timegated_json,
    compute_focal_gain,
    create_hemispherical_array,
    load_nifti_as_xarray,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUBJECT = "GU010"
APOD_ALPHA_VALUES = [3.5, 8.4, 15.0]  # dB/cm (apodization bone attenuation)
BASELINE_ALPHA_FOR_WATER_GATE = 3.5    # alpha whose water sim defines the gate center


def _apod_stats(apod: np.ndarray) -> dict:
    active = apod > 1e-6
    return {
        "n_elements": int(apod.size),
        "active_count": int(active.sum()),
        "mean_weight": float(apod.mean()),
        "sum_weight": float(apod.sum()),
        "min_weight": float(apod.min()),
        "max_weight": float(apod.max()),
        "median_weight": float(np.median(apod)),
        "below_0p1": int((apod < 0.1).sum()),
        "below_0p5": int((apod < 0.5).sum()),
        "above_0p9": int((apod > 0.9).sum()),
    }


def _fnum(v):
    try:
        return "nan" if not np.isfinite(v) else f"{v:.6g}"
    except Exception:
        return "nan"


def _db(num, den):
    if np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0:
        return 20.0 * np.log10(den / num)
    return float("nan")


def main():
    subj = SUBJECT
    home = Path.home()
    mri_path = (
        home / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
        "Anonymized_Subjects/T1-Weighted MRI" / f"{subj}_deface.nii"
    )
    label_path = home / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"
    results_dir = home / "Data/openlifu-validation/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"SKULL_PATH_APOD alpha sweep | subject={subj} | "
          f"apod alpha values = {APOD_ALPHA_VALUES} dB/cm")
    print(f"  Cube half extent: {os.environ['EXPANDED_TARGET_HALF_MM']} mm "
          f"(stride {os.environ['CUBE_PROBE_STRIDE']})")
    print(f"  Medium attenuation left at default (NOT swept)")
    print(f"  MRI:    {mri_path}")
    print(f"  Labels: {label_path}")
    print("=" * 72)
    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}"); sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}"); sys.exit(1)

    t_total = time.time()

    # -------------------------------------------------------------
    # Build geometry + sim grid ONCE (all alphas share)
    # -------------------------------------------------------------
    volume = load_nifti_as_xarray(mri_path)
    seg_method = PreSegmented(label_nifti_path=str(label_path))
    print(f"MRI shape: {volume.shape}, label shape: {seg_method._labels.shape}")

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

    skull_mask_orig = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask_orig)
    if skull_indices.size == 0:
        print("ERROR: no skull voxels; aborting."); sys.exit(2)
    skull_mm_all = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]] for ax in range(3)
    ])
    max_skull_per_axis = np.array([
        float(skull_mm_all[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_per_axis))
    print(f"Approach axis: {dim_names[approach_axis]} (axis {approach_axis})")

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

    # Sim grid
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

    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask = sim_seg_arr == material_idx["air"]
    skull_mask_sim = sim_seg_arr == material_idx["skull"]
    n_skull = int(skull_mask_sim.sum())
    total_vox = sim_seg_arr.size
    pct_skull = 100.0 * n_skull / total_vox
    print(f"SIM grid bone fraction: {pct_skull:.2f}% "
          f"({n_skull:,d}/{total_vox:,d} voxels)")

    # Build the xarray skull mask for SkullPathApodization
    # (dims / coords in mm, matching sim_volume)
    skull_mask_xa = xa.DataArray(
        skull_mask_sim.astype(bool),
        dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )

    # Geometric skull path along target -> aperture_center (for logging only)
    aperture_center_mm = positions.mean(axis=0)
    ray_dir = aperture_center_mm - target_mm
    ray_len_mm = float(np.linalg.norm(ray_dir))
    ray_unit = ray_dir / ray_len_mm
    probe_len_mm = 1.3 * RADIUS_MM
    step_mm = GRID_SPACING_MM / 2.0
    sim_origins = np.array([sim_coord_arrays[i][0] for i in range(3)])
    sim_specs = np.array([sim_coord_arrays[i][1] - sim_coord_arrays[i][0] for i in range(3)])
    n_steps = int(np.ceil(probe_len_mm / step_mm)) + 1
    ts = np.linspace(0.0, probe_len_mm, n_steps)
    pts = target_mm[None, :] + ts[:, None] * ray_unit[None, :]
    frac = ((pts - sim_origins[None, :]) / sim_specs[None, :]).T
    sampled = map_coordinates(
        sim_seg_arr.astype(np.float32), frac, order=0,
        mode="constant", cval=-1.0,
    ).astype(np.int16)
    skull_path_near_mm = float(int((sampled == material_idx["skull"]).sum()) * step_mm)
    print(f"Skull path near-side (axial): {skull_path_near_mm:.2f} mm")

    target = Point(position=target_mm.copy(), id="brain_center",
                   name="Brain Center Target", units="mm")
    direct = Direct(c0=C0)

    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=CFL,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    t_end = (max_dist_mm * 1e-3) / C0 * T_END_SAFETY

    # -------------------------------------------------------------
    # Build sim_params ONCE (medium alpha at default, 3.5 dB/cm/MHz for skull)
    # -------------------------------------------------------------
    sim_params = seg_method.seg_params(sim_volume)
    water_mat = seg_method.materials["water"]
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation
    alpha_skull_observed = float(sim_params["attenuation"].to_numpy()[skull_mask_sim].mean())
    print(f"Medium skull mean attenuation (unchanged): "
          f"{alpha_skull_observed:.3f} dB/cm/MHz")

    c_max = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max

    # -------------------------------------------------------------
    # Pre-compute phase-corrected delays ONCE (medium independent of apod)
    # -------------------------------------------------------------
    sim_corrected = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
    t0 = time.time()
    delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
    print(f"SimulationCorrected back-prop in {time.time()-t0:.1f}s")
    delays_geo = direct.calc_delays(arr, target, sim_params)

    cx = sim_coord_arrays[0]; cy = sim_coord_arrays[1]; cz = sim_coord_arrays[2]

    def _stats(result):
        pmax = result["p_max"].to_numpy()
        return compute_focal_gain(pmax, (cx, cy, cz), target_mm, positions,
                                  band_radius_mm=APERTURE_BAND_RADIUS_MM)

    # -------------------------------------------------------------
    # Pre-compute SkullPathApodization weights for each alpha
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Pre-computing SkullPathApodization weights")
    print("=" * 72)
    apod_by_alpha: dict[float, np.ndarray] = {}
    for alpha in APOD_ALPHA_VALUES:
        t0 = time.time()
        apod_method = SkullPathApodization(
            skull_mask=skull_mask_xa,
            alpha_bone_db_per_cm=alpha,
            max_path_mm=None,
            step_mm=GRID_SPACING_MM / 2.0,
        )
        apod_vec = apod_method.calc_apodization(arr, target, sim_params, transform=None)
        apod_by_alpha[alpha] = apod_vec
        s = _apod_stats(apod_vec)
        print(
            f"  alpha={alpha:5.2f}: "
            f"active={s['active_count']}/{s['n_elements']}, "
            f"mean={s['mean_weight']:.4f}, sum={s['sum_weight']:.2f}, "
            f"median={s['median_weight']:.4f}, "
            f"min={s['min_weight']:.4g}, max={s['max_weight']:.4f}, "
            f"<0.1: {s['below_0p1']}, <0.5: {s['below_0p5']}, >0.9: {s['above_0p9']} "
            f"(comp in {time.time()-t0:.1f}s)"
        )

    # -------------------------------------------------------------
    # SIM C baseline (uniform apod) - used ONLY to establish the water-gate
    # center time. Apod doesn't change arrival time in homogeneous water, so
    # one water sim is sufficient to define t_water_target_peak, and we reuse
    # it as the gate for all skull sims regardless of apod alpha. This keeps
    # the gate identical across the sweep for a fair comparison.
    # -------------------------------------------------------------
    common_kwargs = dict(
        arr=arr, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, source_method="point_source",
    )
    uniform_apod = np.ones(arr.numelements())

    print("\n" + "=" * 72)
    print("SIM C (water, uniform apod) - baseline for gate center")
    print("=" * 72)
    t0 = time.time()
    result_c_uniform = run_simulation(
        params=sim_params, delays=delays_geo,
        apod=uniform_apod, ref_values_only=True, **common_kwargs,
    )
    print(f"  done in {time.time()-t0:.1f}s")
    probe_c_uniform = _run_timegated_probe(
        arr=arr, params=sim_params, delays=delays_geo, apod=uniform_apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=True,
        sim_label="SIM C (water, uniform apod baseline)",
        target_gate_center_s=None,
    )
    t_water_target_peak = probe_c_uniform.get("target_peak_time_s") if probe_c_uniform else None
    if t_water_target_peak is not None and not np.isfinite(t_water_target_peak):
        t_water_target_peak = None

    stats_c_uniform = _stats(result_c_uniform)
    p_water_uniform_target = stats_c_uniform["p_at_target"]
    p_fw_water_uniform_tgt = probe_c_uniform.get("p_focal_window_at_target_Pa", float("nan"))
    sp_water_uniform_cube = probe_c_uniform.get("spatial_max_p_focal_window_Pa", float("nan"))
    sp_water_uniform_offset = probe_c_uniform.get("spatial_max_offset_from_target_mm", float("nan"))
    print(
        f"  uniform water: p@target={p_water_uniform_target:.4g} Pa, "
        f"p_fw(target)={p_fw_water_uniform_tgt:.4g}, "
        f"sp_cube_max={sp_water_uniform_cube:.4g} (offset={sp_water_uniform_offset:.2f} mm), "
        f"t_water_peak={t_water_target_peak*1e6 if t_water_target_peak else float('nan'):.3f} us"
    )

    _save_timegated_json(
        probe_c_uniform,
        results_dir / f"{subj}_skullpathapod_water_uniform_timegated.json",
    )

    # -------------------------------------------------------------
    # Sweep: for each alpha, run SIM C (water, apod'd) + SIM A (skull corrected)
    # -------------------------------------------------------------
    sweep_rows = []
    for alpha in APOD_ALPHA_VALUES:
        print("\n" + "#" * 72)
        print(f"# apod alpha_bone = {alpha:.2f} dB/cm "
              f"(SkullPathApodization, max_path_mm=None)")
        print("#" * 72)
        apod_vec = apod_by_alpha[alpha]
        s = _apod_stats(apod_vec)
        print(
            f"  apod stats: active={s['active_count']}/{s['n_elements']}, "
            f"mean={s['mean_weight']:.4f}, sum={s['sum_weight']:.2f}"
        )

        # ---- SIM C (water ref with this apod) ----
        print(f"  [SIM C] water ref w/ skull-path apod (alpha={alpha})")
        t0 = time.time()
        result_c = run_simulation(
            params=sim_params, delays=delays_geo,
            apod=apod_vec, ref_values_only=True, **common_kwargs,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        probe_c = _run_timegated_probe(
            arr=arr, params=sim_params, delays=delays_geo, apod=apod_vec,
            target_mm=target_mm, aperture_center_mm=aperture_center_mm,
            common_kwargs=common_kwargs, ref_values_only=True,
            sim_label=f"SIM C apod_alpha={alpha}",
            target_gate_center_s=None,
        )
        stats_c = _stats(result_c)
        p_water_target = stats_c["p_at_target"]
        p_fw_water_target = probe_c.get("p_focal_window_at_target_Pa", float("nan"))
        sp_water_cube = probe_c.get("spatial_max_p_focal_window_Pa", float("nan"))
        sp_water_offset = probe_c.get("spatial_max_offset_from_target_mm", float("nan"))
        print(
            f"    water(apod={alpha}): p@target={p_water_target:.4g} Pa, "
            f"p_fw(target)={p_fw_water_target:.4g}, "
            f"sp_cube_max={sp_water_cube:.4g} (offset={sp_water_offset:.2f} mm)"
        )
        _save_timegated_json(
            probe_c,
            results_dir / f"{subj}_skullpathapod_water_alpha{alpha:04.1f}_timegated.json",
        )

        # ---- SIM A (corrected + skull with this apod) ----
        print(f"  [SIM A] corrected + skull w/ skull-path apod (alpha={alpha})")
        t0 = time.time()
        result_a = run_simulation(
            params=sim_params, delays=delays_corrected,
            apod=apod_vec, ref_values_only=False, **common_kwargs,
        )
        print(f"    full-grid sim done in {time.time()-t0:.1f}s")
        probe_a = _run_timegated_probe(
            arr=arr, params=sim_params, delays=delays_corrected, apod=apod_vec,
            target_mm=target_mm, aperture_center_mm=aperture_center_mm,
            common_kwargs=common_kwargs, ref_values_only=False,
            sim_label=f"SIM A apod_alpha={alpha}",
            target_gate_center_s=t_water_target_peak,
        )
        stats_a = _stats(result_a)
        p_skull_target = stats_a["p_at_target"]
        p_allt_target = probe_a.get("p_allt_at_target_Pa", float("nan")) if probe_a else float("nan")
        p_fw_skull_target = probe_a.get("p_focal_window_water_gate_at_target_Pa", float("nan")) if probe_a else float("nan")
        sp_skull_cube = probe_a.get("spatial_max_p_focal_window_water_gate_Pa", float("nan")) if probe_a else float("nan")
        sp_skull_offset = probe_a.get("spatial_max_offset_from_target_mm", float("nan")) if probe_a else float("nan")
        sp_skull_world = probe_a.get("spatial_max_world_mm") if probe_a else None

        # Coherence ratio diagnostic
        ratio_fw_over_allt = (
            p_fw_skull_target / p_allt_target
            if (np.isfinite(p_fw_skull_target) and np.isfinite(p_allt_target)
                and p_allt_target > 0)
            else float("nan")
        )

        atten_pmax_db = _db(p_skull_target, p_water_target)
        atten_fw_db = _db(p_fw_skull_target, p_fw_water_target)
        atten_sp_vs_water_cube_db = _db(sp_skull_cube, sp_water_cube)
        atten_sp_vs_water_tgt_db = _db(sp_skull_cube, p_fw_water_target)

        print(f"  RESULT apod_alpha={alpha}:")
        print(f"    water : p@tgt={p_water_target:.4g}, "
              f"p_fw(tgt)={p_fw_water_target:.4g}, sp_cube={sp_water_cube:.4g}")
        print(f"    skull : p@tgt={p_skull_target:.4g}, "
              f"p_fw(tgt)={p_fw_skull_target:.4g}, p_allt(tgt)={p_allt_target:.4g}, "
              f"sp_cube={sp_skull_cube:.4g} (offset={sp_skull_offset:.2f} mm)")
        print(f"    DIAGNOSTIC ratio p_fw/p_allt @ target (skull) = {ratio_fw_over_allt:.4f}")
        print(f"    atten(dB): pmax={atten_pmax_db:.2f}, fw={atten_fw_db:.2f}, "
              f"sp_vs_water_cube={atten_sp_vs_water_cube_db:.2f}, "
              f"sp_vs_water_tgt={atten_sp_vs_water_tgt_db:.2f}")

        _save_timegated_json(
            probe_a,
            results_dir / f"{subj}_skullpathapod_skull_alpha{alpha:04.1f}_timegated.json",
        )

        sweep_rows.append({
            "apod_alpha": alpha,
            "active_elts": s["active_count"],
            "mean_weight": s["mean_weight"],
            "sum_weight": s["sum_weight"],
            "p_water_target": p_water_target,
            "p_fw_water_target": p_fw_water_target,
            "sp_water_cube": sp_water_cube,
            "sp_water_offset": sp_water_offset,
            "p_skull_target": p_skull_target,
            "p_allt_skull_target": p_allt_target,
            "p_fw_skull_target": p_fw_skull_target,
            "sp_skull_cube": sp_skull_cube,
            "sp_skull_offset": sp_skull_offset,
            "sp_skull_world_mm": sp_skull_world,
            "ratio_fw_over_allt_skull": ratio_fw_over_allt,
            "atten_pmax_db": atten_pmax_db,
            "atten_fw_db": atten_fw_db,
            "atten_sp_vs_water_cube_db": atten_sp_vs_water_cube_db,
            "atten_sp_vs_water_tgt_db": atten_sp_vs_water_tgt_db,
        })

        print(
            f"SKULL_PATH_APOD_SWEEP subject={subj} apod_alpha={alpha:.2f} "
            f"active={s['active_count']} mean_w={_fnum(s['mean_weight'])} "
            f"sum_w={_fnum(s['sum_weight'])} "
            f"p_water_target={_fnum(p_water_target)} "
            f"p_fw_water_target={_fnum(p_fw_water_target)} "
            f"sp_water_cube={_fnum(sp_water_cube)} "
            f"p_skull_target={_fnum(p_skull_target)} "
            f"p_allt_skull_target={_fnum(p_allt_target)} "
            f"p_fw_skull_target={_fnum(p_fw_skull_target)} "
            f"sp_skull_cube={_fnum(sp_skull_cube)} "
            f"sp_skull_offset_mm={_fnum(sp_skull_offset)} "
            f"ratio_fw_over_allt={_fnum(ratio_fw_over_allt)} "
            f"atten_pmax_db={_fnum(atten_pmax_db)} "
            f"atten_fw_db={_fnum(atten_fw_db)} "
            f"atten_sp_vs_water_cube_db={_fnum(atten_sp_vs_water_cube_db)} "
            f"atten_sp_vs_water_tgt_db={_fnum(atten_sp_vs_water_tgt_db)}"
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SKULL_PATH_APOD SWEEP SUMMARY (GU010)")
    print("=" * 72)
    hdr = (
        f"{'apod_a':>7} {'active':>7} {'mean_w':>7} {'sum_w':>7} "
        f"{'p_w_tgt':>10} {'p_fw_w_tgt':>11} {'sp_w_cube':>10} "
        f"{'p_s_tgt':>10} {'p_fw_s_tgt':>11} {'p_allt_s_tgt':>12} "
        f"{'sp_s_cube':>10} {'off_mm':>7} "
        f"{'ratio':>7} {'attPmax':>8} {'attFW':>7} {'attSPvTgt':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sweep_rows:
        print(
            f"{r['apod_alpha']:>7.2f} {r['active_elts']:>7d} "
            f"{r['mean_weight']:>7.4f} {r['sum_weight']:>7.2f} "
            f"{r['p_water_target']:>10.4g} {r['p_fw_water_target']:>11.4g} "
            f"{r['sp_water_cube']:>10.4g} "
            f"{r['p_skull_target']:>10.4g} {r['p_fw_skull_target']:>11.4g} "
            f"{r['p_allt_skull_target']:>12.4g} "
            f"{r['sp_skull_cube']:>10.4g} {r['sp_skull_offset']:>7.2f} "
            f"{r['ratio_fw_over_allt_skull']:>7.4f} "
            f"{r['atten_pmax_db']:>8.2f} {r['atten_fw_db']:>7.2f} "
            f"{r['atten_sp_vs_water_tgt_db']:>10.2f}"
        )

    print("\nBaseline references (uniform apod):")
    print(
        f"  water uniform: p@tgt={p_water_uniform_target:.4g}, "
        f"p_fw(tgt)={p_fw_water_uniform_tgt:.4g}, "
        f"sp_cube={sp_water_uniform_cube:.4g} (offset={sp_water_uniform_offset:.2f} mm)"
    )

    t_elapsed = time.time() - t_total
    print(f"\n[{subj}] Total skull-path-apod alpha sweep: "
          f"{t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
