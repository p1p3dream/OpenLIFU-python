#!/usr/bin/env python3
"""CFL stability diagnostic at alpha_bone=12 dB/cm/MHz on GU010.

Tests whether reducing CFL from 0.1 to 0.05 or 0.03 stabilizes k-Wave at
alpha=12 dB/cm/MHz (which blew up in the alpha_bone sweep with CFL=0.1).

Runs SIM A (phase-corrected + heterogeneous skull) only, for three CFL
values: [0.1, 0.05, 0.03]. Reports max|p|, p@target, and
p_focal_window @ target for each run.

Skips SIM B and SIM C (water reference not needed for stability diagnosis).

Local / uncommitted.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

# Force expanded-target probe (5 mm cube) to match alpha sweep environment
os.environ["EXPANDED_TARGET_PROBE"] = "1"
os.environ["EXPANDED_TARGET_HALF_MM"] = "5.0"
os.environ.pop("OPENLIFU_DISABLE_TIMEGATED_PROBE", None)

# Same logging workaround as the parent script
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

from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.seg.material import MATERIALS
from openlifu.sim.kwave_if import run_simulation
from openlifu.sim.sim_setup import SimSetup

sys.path.insert(0, os.path.dirname(__file__))
from run_gladys_nnunet_subject import (  # type: ignore
    APERTURE_BAND_RADIUS_MM,
    APERTURE_MM,
    AMPLITUDE,
    C0,
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
ALPHA_TEST = 12.0  # dB/cm/MHz  -- the unstable value from the alpha sweep
CFL_VALUES = [0.1, 0.05, 0.03]
F_MHZ = 0.5
ALPHA_POWER = 0.9
# Per task: if CFL=0.03 exceeds this wall-clock per sim, abort and report.
PER_SIM_TIME_CAP_S = 15 * 60


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
    print(f"CFL diagnostic | subject={subj} | alpha_bone={ALPHA_TEST} dB/cm/MHz")
    print(f"  CFL values: {CFL_VALUES}")
    print(f"  MRI:    {mri_path}")
    print(f"  Labels: {label_path}")
    print("=" * 72)
    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}"); sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}"); sys.exit(1)

    t_total = time.time()

    # ---------------------------------------------------------
    # Geometry + grid (identical to alpha_bone_sweep_gu010.py)
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

    # Skull path (for context)
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

    # SimSetup only used for get_max_distance (CFL-independent geometry)
    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=0.1,  # placeholder; CFL varied per run below
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    t_end = (max_dist_mm * 1e-3) / C0 * T_END_SAFETY
    print(f"t_end = {t_end*1e6:.3f} us (CFL-independent)")

    apod = np.ones(arr.numelements())

    # ---------------------------------------------------------
    # Monkey-patch SKULL alpha to 12 ONCE
    # ---------------------------------------------------------
    MATERIALS["skull"].attenuation = ALPHA_TEST
    seg_method.materials["skull"].attenuation = ALPHA_TEST

    sim_params = seg_method.seg_params(sim_volume)
    water_mat = seg_method.materials["water"]
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation

    alpha_skull_observed = float(sim_params["attenuation"].to_numpy()[skull_mask_sim].mean())
    print(f"verify: sim_params skull mean attenuation = "
          f"{alpha_skull_observed:.3f} dB/cm/MHz (expected {ALPHA_TEST:.3f})")

    c_max_params = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    print(f"c_max in sim_params: {c_max_params:.1f} m/s; dx = {dx_m*1e3:.3f} mm")

    cx = sim_coord_arrays[0]; cy = sim_coord_arrays[1]; cz = sim_coord_arrays[2]

    def _stats(result):
        pmax = result["p_max"].to_numpy()
        return compute_focal_gain(pmax, (cx, cy, cz), target_mm, positions,
                                  band_radius_mm=APERTURE_BAND_RADIUS_MM)

    # ---------------------------------------------------------
    # Loop over CFL values at alpha=12
    # ---------------------------------------------------------
    rows = []

    for cfl in CFL_VALUES:
        print("\n" + "#" * 72)
        print(f"# CFL = {cfl}   alpha_bone = {ALPHA_TEST}")
        print("#" * 72)

        dt = cfl * dx_m / c_max_params
        n_t = int(np.ceil(t_end / dt))
        print(f"  dt = {dt*1e9:.4f} ns   t_end = {t_end*1e6:.3f} us   N_t = {n_t:,d}")

        sim_corrected = SimulationCorrected(c0=C0, cfl=cfl, n_cycles=3, gpu=True)
        t0 = time.time()
        delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
        t_backprop = time.time() - t0
        print(f"  SimulationCorrected back-prop: {t_backprop:.1f}s")

        common_kwargs = dict(
            arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
            dt=dt, t_end=t_end, cfl=cfl, gpu=True, source_method="point_source",
        )

        print(f"  [SIM A] corrected + skull  alpha={ALPHA_TEST}  CFL={cfl}")
        t0 = time.time()
        try:
            result_a = run_simulation(
                params=sim_params, delays=delays_corrected,
                ref_values_only=False, **common_kwargs,
            )
            sim_error = None
        except Exception as e:
            result_a = None
            sim_error = repr(e)
        t_sim = time.time() - t0
        print(f"    full-grid sim: {t_sim:.1f}s"
              + (f"  ERROR: {sim_error}" if sim_error else ""))

        pmax_global = float("nan")
        p_target = float("nan")
        p_fw_target = float("nan")
        sp_cube_max = float("nan")

        if result_a is not None:
            pmax_arr = result_a["p_max"].to_numpy()
            pmax_global = float(np.nanmax(np.abs(pmax_arr)))
            try:
                stats_a = _stats(result_a)
                p_target = float(stats_a["p_at_target"])
            except Exception as e:
                print(f"    WARN: _stats failed: {e!r}")

            # Only run probe if the sim looks stable (skip if blown up -- saves time)
            if np.isfinite(pmax_global) and pmax_global < 1e6:
                try:
                    probe_a = _run_timegated_probe(
                        arr=arr, params=sim_params, delays=delays_corrected, apod=apod,
                        target_mm=target_mm, aperture_center_mm=aperture_center_mm,
                        common_kwargs=common_kwargs, ref_values_only=False,
                        sim_label=f"CFL={cfl} alpha={ALPHA_TEST}",
                        target_gate_center_s=None,
                    )
                    if probe_a is not None:
                        p_fw_target = float(probe_a.get(
                            "p_focal_window_at_target_Pa", float("nan")))
                        sp_cube_max = float(probe_a.get(
                            "spatial_max_p_focal_window_Pa", float("nan")))
                except Exception as e:
                    print(f"    WARN: probe failed: {e!r}")
            else:
                print(f"    skipping probe: pmax_global={pmax_global:.4g} (blown up)")

        # Stability classification per task spec
        if np.isfinite(pmax_global) and pmax_global < 10.0:
            stable_str = "YES"
        elif np.isfinite(pmax_global) and pmax_global > 1e3:
            stable_str = "NO (blowup)"
        elif not np.isfinite(pmax_global):
            stable_str = "NO (NaN/inf)"
        else:
            stable_str = "AMBIGUOUS"

        print(f"\n  RESULT CFL={cfl}  alpha={ALPHA_TEST}:")
        print(f"    max|p|             = {pmax_global:.6g} Pa  -> stable={stable_str}")
        print(f"    p@target           = {p_target:.6g} Pa")
        print(f"    p_fw(target)       = {p_fw_target:.6g} Pa")
        print(f"    sp_cube_max        = {sp_cube_max:.6g} Pa")
        print(f"    wall clock (sim)   = {t_sim:.1f}s")
        print(
            f"CFL_DIAG_RESULT subject={subj} alpha={ALPHA_TEST} cfl={cfl} "
            f"dt_ns={dt*1e9:.4f} n_t={n_t} "
            f"pmax_global={pmax_global:.6g} p_target={p_target:.6g} "
            f"p_fw_target={p_fw_target:.6g} sp_cube_max={sp_cube_max:.6g} "
            f"stable={stable_str.split()[0]} sim_sec={t_sim:.1f}"
        )

        rows.append({
            "cfl": cfl, "dt_ns": dt * 1e9, "n_t": n_t,
            "pmax_global": pmax_global, "p_target": p_target,
            "p_fw_target": p_fw_target, "sp_cube_max": sp_cube_max,
            "stable": stable_str, "t_sim_s": t_sim, "error": sim_error,
        })

        if cfl == 0.03 and t_sim > PER_SIM_TIME_CAP_S:
            print(f"  CFL=0.03 exceeded {PER_SIM_TIME_CAP_S}s cap ({t_sim:.0f}s); stopping.")
            break

    # Summary
    print("\n" + "=" * 72)
    print(f"CFL SWEEP SUMMARY @ alpha_bone = {ALPHA_TEST} dB/cm/MHz")
    print("=" * 72)
    print(f"{'cfl':>6} {'dt_ns':>8} {'n_t':>8} "
          f"{'max|p|':>12} {'p@tgt':>12} {'p_fw(tgt)':>12} "
          f"{'sp_cube':>12} {'t_sim':>7} {'stable':>14}")
    for r in rows:
        print(f"{r['cfl']:>6.3f} {r['dt_ns']:>8.3f} {r['n_t']:>8d} "
              f"{r['pmax_global']:>12.4g} {r['p_target']:>12.4g} "
              f"{r['p_fw_target']:>12.4g} {r['sp_cube_max']:>12.4g} "
              f"{r['t_sim_s']:>7.1f} {r['stable']:>14}")

    stable_cfls = [r["cfl"] for r in rows if r["stable"] == "YES"]
    if stable_cfls:
        print(f"\nMinimum stable CFL at alpha={ALPHA_TEST}: {min(stable_cfls)}")
    else:
        print(f"\nNO stable CFL found at alpha={ALPHA_TEST}.")

    t_elapsed = time.time() - t_total
    print(f"\n[{subj}] Total CFL diagnostic: {t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
