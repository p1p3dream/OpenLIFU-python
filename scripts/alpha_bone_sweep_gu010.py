#!/usr/bin/env python3
"""Skull absorption coefficient (alpha_bone) sensitivity sweep on GU010.

Tests whether the 15.6 dB unexplained attenuation residual is driven by wrong
bulk absorption. Sweeps alpha_bone in [4, 8, 12, 16] dB/cm/MHz by monkey-
patching the SKULL Material singleton BEFORE building sim_params.

For each alpha value, runs only SIM A (phase-corrected, skull) with the
expanded-target cube probe (5 mm half-extent). SIM C (water) is run exactly
ONCE because the homogeneous-water reference simulation uses water's
attenuation only (ref_values_only=True). SIM B is skipped.

Output: one ALPHA_SWEEP_RESULT line per alpha, plus a final summary and
linear fit of attenuation_db vs alpha_bone.

Local / uncommitted.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

# Force expanded-target probe (5 mm cube) before importing subject script
os.environ["EXPANDED_TARGET_PROBE"] = "1"
os.environ["EXPANDED_TARGET_HALF_MM"] = "5.0"
# Keep time-gated probe enabled
os.environ.pop("OPENLIFU_DISABLE_TIMEGATED_PROBE", None)

# -------------------------------------------------------------
# Same logging workaround as the parent script
# -------------------------------------------------------------
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

import nibabel as nib
import numpy as np
import xarray as xa
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.seg.material import MATERIALS
from openlifu.sim.kwave_if import run_simulation
from openlifu.sim.sim_setup import SimSetup

# Re-use the geometry + probe helpers from the subject script
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
ALPHA_VALUES = [4.0, 8.0, 12.0, 16.0]  # dB/cm/MHz
F_MHZ = 0.5
ALPHA_POWER = 0.9


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
    print(f"ALPHA_BONE sweep | subject={subj} | values={ALPHA_VALUES} dB/cm/MHz")
    print(f"  MRI:    {mri_path}")
    print(f"  Labels: {label_path}")
    print("=" * 72)
    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}"); sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}"); sys.exit(1)

    t_total = time.time()

    # ---------------------------------------------------------
    # Build geometry ONCE (subject-specific but alpha-independent)
    # ---------------------------------------------------------
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

    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
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
    print(f"SIM grid bone fraction: {pct_skull:.2f}% ({n_skull:,d}/{total_vox:,d} voxels)")

    # Skull path (geometric, along target -> aperture_center)
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
    skull_path_cm = skull_path_near_mm / 10.0
    print(f"Skull path near-side: {skull_path_near_mm:.2f} mm = {skull_path_cm:.3f} cm")

    target = Point(position=target_mm.copy(), id="brain_center",
                   name="Brain Center Target", units="mm")
    direct = Direct(c0=C0)

    # Common kwargs for full-grid sim and probe sub-sims
    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=CFL,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    t_end = (max_dist_mm * 1e-3) / C0 * T_END_SAFETY

    apod = np.ones(arr.numelements())

    # ------------------------------------------------------------
    # SIM C (water) - run ONCE. Water reference uses water's
    # attenuation (ref_value), independent of skull alpha.
    # ------------------------------------------------------------
    # Build sim_params with default (alpha=8) for water sim just to have dt etc.
    sim_params_water = seg_method.seg_params(sim_volume)
    water_mat = seg_method.materials["water"]
    if air_mask.any():
        sim_params_water["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params_water["density"].data[air_mask] = water_mat.density
        sim_params_water["attenuation"].data[air_mask] = water_mat.attenuation
    c_max_water = float(sim_params_water["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max_water

    delays_geo = direct.calc_delays(arr, target, sim_params_water)

    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, source_method="point_source",
    )

    print("\n[SIM C] water reference (run ONCE)")
    t0 = time.time()
    result_c = run_simulation(params=sim_params_water, delays=delays_geo,
                              ref_values_only=True, **common_kwargs)
    print(f"  done in {time.time()-t0:.1f}s")
    probe_c = _run_timegated_probe(
        arr=arr, params=sim_params_water, delays=delays_geo, apod=apod,
        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
        common_kwargs=common_kwargs, ref_values_only=True,
        sim_label="SIM C (water)",
        target_gate_center_s=None,
    )
    t_water_target_peak = probe_c.get("target_peak_time_s") if probe_c else None
    if t_water_target_peak is not None and not np.isfinite(t_water_target_peak):
        t_water_target_peak = None

    cx = sim_coord_arrays[0]; cy = sim_coord_arrays[1]; cz = sim_coord_arrays[2]

    def _stats(result):
        pmax = result["p_max"].to_numpy()
        return compute_focal_gain(pmax, (cx, cy, cz), target_mm, positions,
                                  band_radius_mm=APERTURE_BAND_RADIUS_MM)

    stats_c = _stats(result_c)
    p_water_target = stats_c["p_at_target"]
    sp_water_target = probe_c.get("p_focal_window_at_target_Pa", float("nan"))
    sp_water_cube_max = probe_c.get("spatial_max_p_focal_window_Pa", float("nan"))
    sp_offset_water = probe_c.get("spatial_max_offset_from_target_mm", float("nan"))
    print(f"  water: p@target={p_water_target:.4g} Pa, "
          f"p_fw(target)={sp_water_target:.4g}, "
          f"sp_cube_max={sp_water_cube_max:.4g} (offset={sp_offset_water:.2f} mm)")

    _save_timegated_json(
        probe_c, results_dir / f"{subj}_alphasweep_water_timegated.json"
    )

    # ------------------------------------------------------------
    # SIM A loop over alpha_bone values
    # ------------------------------------------------------------
    # Store results for regression / reporting
    sweep_rows = []  # list of dicts

    for alpha in ALPHA_VALUES:
        print("\n" + "#" * 72)
        print(f"# alpha_bone = {alpha:.2f} dB/cm/MHz  (alpha_eff@500kHz = "
              f"{alpha * F_MHZ**ALPHA_POWER:.3f} dB/cm)")
        print("#" * 72)

        # Monkey-patch SKULL material (singleton) AND seg_method's reference.
        MATERIALS["skull"].attenuation = alpha
        seg_method.materials["skull"].attenuation = alpha  # same object, but explicit

        # Rebuild sim_params from segmentation with new alpha
        sim_params = seg_method.seg_params(sim_volume)
        if air_mask.any():
            sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
            sim_params["density"].data[air_mask] = water_mat.density
            sim_params["attenuation"].data[air_mask] = water_mat.attenuation

        # Verify: skull voxels carry the new alpha
        alpha_skull_observed = float(sim_params["attenuation"].to_numpy()[skull_mask_sim].mean())
        print(f"  verify: sim_params skull mean attenuation = "
              f"{alpha_skull_observed:.3f} dB/cm/MHz (expected {alpha:.3f})")

        # Recompute delays with the new medium (sim-corrected back-prop senses alpha)
        sim_corrected = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
        t0 = time.time()
        delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
        print(f"  SimulationCorrected back-prop in {time.time()-t0:.1f}s")

        # SIM A: corrected + skull
        print(f"  [SIM A] corrected + skull (alpha={alpha})")
        t0 = time.time()
        result_a = run_simulation(params=sim_params, delays=delays_corrected,
                                  ref_values_only=False, **common_kwargs)
        print(f"    full-grid sim done in {time.time()-t0:.1f}s")
        probe_a = _run_timegated_probe(
            arr=arr, params=sim_params, delays=delays_corrected, apod=apod,
            target_mm=target_mm, aperture_center_mm=aperture_center_mm,
            common_kwargs=common_kwargs, ref_values_only=False,
            sim_label=f"SIM A alpha={alpha}",
            target_gate_center_s=t_water_target_peak,
        )

        stats_a = _stats(result_a)
        p_skull_target = stats_a["p_at_target"]
        p_fw_water_target = probe_a.get("p_focal_window_water_gate_at_target_Pa", float("nan")) if probe_a else float("nan")
        sp_cube_max = probe_a.get("spatial_max_p_focal_window_water_gate_Pa", float("nan")) if probe_a else float("nan")
        sp_offset = probe_a.get("spatial_max_offset_from_target_mm", float("nan")) if probe_a else float("nan")
        sp_world = probe_a.get("spatial_max_world_mm") if probe_a else None

        def _db(num, den):
            if np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0:
                return 20.0 * np.log10(den / num)
            return float("nan")

        atten_pmax = _db(p_skull_target, p_water_target)
        atten_fw_target = _db(p_fw_water_target, sp_water_target)
        atten_sp_cube = _db(sp_cube_max, sp_water_cube_max)
        atten_sp_cube_vs_water_target = _db(sp_cube_max, sp_water_target)

        print(f"  RESULT alpha={alpha}:")
        print(f"    p@target         = {p_skull_target:.4g} Pa  (atten pmax = {atten_pmax:.2f} dB)")
        print(f"    p_fw_water(target)= {p_fw_water_target:.4g} Pa  (atten = {atten_fw_target:.2f} dB)")
        print(f"    sp_cube_max      = {sp_cube_max:.4g} Pa  (offset={sp_offset:.2f} mm)  "
              f"(atten vs water_sp_cube = {atten_sp_cube:.2f} dB, "
              f"vs water_target = {atten_sp_cube_vs_water_target:.2f} dB)")

        _save_timegated_json(
            probe_a,
            results_dir / f"{subj}_alphasweep_alpha{alpha:04.1f}_corrected_timegated.json",
        )

        sweep_rows.append({
            "alpha": alpha,
            "alpha_eff_dbcm": alpha * F_MHZ**ALPHA_POWER,
            "p_target": p_skull_target,
            "p_fw_water_target": p_fw_water_target,
            "sp_cube_max": sp_cube_max,
            "sp_offset_mm": sp_offset,
            "sp_world_mm": sp_world,
            "atten_pmax_db": atten_pmax,
            "atten_fw_target_db": atten_fw_target,
            "atten_sp_cube_db": atten_sp_cube,
            "atten_sp_cube_vs_water_target_db": atten_sp_cube_vs_water_target,
        })

        # Machine-parseable line
        def _f(v):
            try:
                return "nan" if not np.isfinite(v) else f"{v:.6g}"
            except Exception:
                return "nan"
        print(
            f"ALPHA_SWEEP_RESULT subject={subj} alpha={alpha:.3f} "
            f"alpha_eff_dbcm={alpha * F_MHZ**ALPHA_POWER:.3f} "
            f"skull_path_mm={skull_path_near_mm:.3f} "
            f"p_water_target={_f(sp_water_target)} "
            f"p_water_sp_cube={_f(sp_water_cube_max)} "
            f"p_target={_f(p_skull_target)} "
            f"p_fw_water_target={_f(p_fw_water_target)} "
            f"sp_cube_max={_f(sp_cube_max)} "
            f"sp_offset_mm={_f(sp_offset)} "
            f"atten_pmax_db={_f(atten_pmax)} "
            f"atten_fw_target_db={_f(atten_fw_target)} "
            f"atten_sp_cube_db={_f(atten_sp_cube)} "
            f"atten_sp_cube_vs_water_target_db={_f(atten_sp_cube_vs_water_target)}"
        )

    # ------------------------------------------------------------
    # Summary: linear fit of attenuation (spatial best) vs alpha_bone
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SWEEP SUMMARY")
    print("=" * 72)
    print(
        f"{'alpha':>6} {'a_eff':>7} {'p_tgt':>10} {'p_fw_w(t)':>10} "
        f"{'sp_cube':>10} {'off_mm':>7} {'dB_pmax':>8} "
        f"{'dB_fw_t':>8} {'dB_sp_cube':>10} {'dB_sp_vs_wt':>11}"
    )
    for r in sweep_rows:
        print(
            f"{r['alpha']:>6.2f} {r['alpha_eff_dbcm']:>7.3f} "
            f"{r['p_target']:>10.4g} {r['p_fw_water_target']:>10.4g} "
            f"{r['sp_cube_max']:>10.4g} {r['sp_offset_mm']:>7.2f} "
            f"{r['atten_pmax_db']:>8.2f} {r['atten_fw_target_db']:>8.2f} "
            f"{r['atten_sp_cube_db']:>10.2f} "
            f"{r['atten_sp_cube_vs_water_target_db']:>11.2f}"
        )

    # Linear fit: atten_sp_cube_db = m * alpha + b
    alphas = np.array([r["alpha"] for r in sweep_rows], dtype=float)
    attens = np.array([r["atten_sp_cube_vs_water_target_db"] for r in sweep_rows], dtype=float)
    mask = np.isfinite(alphas) & np.isfinite(attens)
    if mask.sum() >= 2:
        m_fit, b_fit = np.polyfit(alphas[mask], attens[mask], 1)
    else:
        m_fit = float("nan"); b_fit = float("nan")
    # Theoretical slope: atten_per_alpha = skull_path_cm * f_MHz^alpha_power
    slope_theory = skull_path_cm * (F_MHZ ** ALPHA_POWER)
    print("\nLINEAR FIT: atten_sp_cube_vs_water_target_db = "
          f"{m_fit:.4f} * alpha + {b_fit:.4f}")
    print(f"  measured slope = {m_fit:.4f} dB per (dB/cm/MHz)")
    print(f"  theoretical   = path_cm * f_MHz^0.9 = {skull_path_cm:.3f} * "
          f"{F_MHZ**ALPHA_POWER:.4f} = {slope_theory:.4f}")
    if np.isfinite(m_fit) and slope_theory > 0:
        ratio = m_fit / slope_theory
        print(f"  measured/theoretical = {ratio:.3f}")

    # What alpha would be needed to reach target residual?
    # Residual ~ 15.6 dB -> what alpha brings SIM A atten close to 33.9 dB?
    # Using fit: alpha* = (target - b_fit) / m_fit
    for target_atten in (25.0, 30.0, 33.9):
        if np.isfinite(m_fit) and abs(m_fit) > 1e-9:
            alpha_needed = (target_atten - b_fit) / m_fit
            print(f"  alpha_bone needed for atten={target_atten:.1f} dB -> {alpha_needed:.2f} dB/cm/MHz")

    print(f"\nFINAL_FIT slope_meas={m_fit:.6f} slope_theory={slope_theory:.6f} "
          f"b_fit={b_fit:.6f} skull_path_cm={skull_path_cm:.4f} "
          f"p_water_target={sp_water_target:.6g} p_water_sp_cube={sp_water_cube_max:.6g}")

    t_elapsed = time.time() - t_total
    print(f"\n[{subj}] Total alpha sweep: {t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
