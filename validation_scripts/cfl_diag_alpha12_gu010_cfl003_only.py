#!/usr/bin/env python3
"""Single-CFL variant of cfl_diagnostic_alpha12_gu010.py: runs CFL=0.03 only.

Rerun just the third CFL after the prior job died silently mid-sim. All
other logic identical.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ["EXPANDED_TARGET_PROBE"] = "1"
os.environ["EXPANDED_TARGET_HALF_MM"] = "5.0"
os.environ.pop("OPENLIFU_DISABLE_TIMEGATED_PROBE", None)

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
    APERTURE_BAND_RADIUS_MM, APERTURE_MM, AMPLITUDE, C0, CYCLES,
    ELEMENT_SIZE_MM, FREQ_HZ, GRID_MARGIN_MM, GRID_SPACING_MM, N_ELEMENTS,
    RADIUS_MM, T_END_SAFETY, PreSegmented, compute_focal_gain,
    create_hemispherical_array, load_nifti_as_xarray,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUBJECT = "GU010"
ALPHA_TEST = 12.0
CFL_ONLY = 0.03
F_MHZ = 0.5


def main():
    subj = SUBJECT
    home = Path.home()
    mri_path = (
        home / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
        "Anonymized_Subjects/T1-Weighted MRI" / f"{subj}_deface.nii"
    )
    label_path = home / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"

    print("=" * 72)
    print(f"CFL={CFL_ONLY} single-run diagnostic @ alpha={ALPHA_TEST} subject={subj}")
    print("=" * 72)

    volume = load_nifti_as_xarray(mri_path)
    seg_method = PreSegmented(label_nifti_path=str(label_path))

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
    print(f"Target: {target_mm}")

    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    skull_mm_all = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]] for ax in range(3)
    ])
    max_skull_per_axis = np.array([
        float(skull_mm_all[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_per_axis))

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

    target = Point(position=target_mm.copy(), id="brain_center",
                   name="Brain Center Target", units="mm")

    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=0.1,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    t_end = (max_dist_mm * 1e-3) / C0 * T_END_SAFETY
    apod = np.ones(arr.numelements())

    MATERIALS["skull"].attenuation = ALPHA_TEST
    seg_method.materials["skull"].attenuation = ALPHA_TEST

    sim_params = seg_method.seg_params(sim_volume)
    water_mat = seg_method.materials["water"]
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation

    c_max_params = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3

    cx = sim_coord_arrays[0]; cy = sim_coord_arrays[1]; cz = sim_coord_arrays[2]

    cfl = CFL_ONLY
    dt = cfl * dx_m / c_max_params
    n_t = int(np.ceil(t_end / dt))
    print(f"CFL={cfl} dt={dt*1e9:.4f}ns N_t={n_t} t_end={t_end*1e6:.2f}us")

    sim_corrected = SimulationCorrected(c0=C0, cfl=cfl, n_cycles=3, gpu=True)
    t0 = time.time()
    delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
    print(f"back-prop: {time.time()-t0:.1f}s")

    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=cfl, gpu=True, source_method="point_source",
    )

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
    print(f"sim: {t_sim:.1f}s" + (f" ERROR {sim_error}" if sim_error else ""))

    pmax_global = float("nan"); p_target = float("nan")
    if result_a is not None:
        pmax_arr = result_a["p_max"].to_numpy()
        pmax_global = float(np.nanmax(np.abs(pmax_arr)))
        try:
            stats = compute_focal_gain(pmax_arr, (cx, cy, cz), target_mm, positions,
                                       band_radius_mm=APERTURE_BAND_RADIUS_MM)
            p_target = float(stats["p_at_target"])
        except Exception as e:
            print(f"stats error: {e!r}")

    if np.isfinite(pmax_global) and pmax_global < 10.0:
        stable = "YES"
    elif np.isfinite(pmax_global) and pmax_global > 1e3:
        stable = "NO_blowup"
    elif not np.isfinite(pmax_global):
        stable = "NO_nan"
    else:
        stable = "AMBIG"

    print(f"\n  RESULT CFL={cfl} alpha={ALPHA_TEST}:")
    print(f"    max|p|   = {pmax_global:.6g} Pa")
    print(f"    p@target = {p_target:.6g} Pa")
    print(f"    stable   = {stable}")
    print(
        f"CFL_DIAG_RESULT subject={subj} alpha={ALPHA_TEST} cfl={cfl} "
        f"dt_ns={dt*1e9:.4f} n_t={n_t} "
        f"pmax_global={pmax_global:.6g} p_target={p_target:.6g} "
        f"stable={stable} sim_sec={t_sim:.1f}"
    )


if __name__ == "__main__":
    main()
