#!/usr/bin/env python3
"""Coherence factor analysis at target for all Birnbaum subjects using ComplexWeighted corrections.

For each subject, runs the CW reciprocal sim (virtual point source at target,
sensors at 64 element voxels), extracts per-element narrowband (amplitude, phase),
and computes CF three ways:

  CF_raw:        |sum(a_i * exp(j*phi_i))| / sum(a_i)
  CF_cw:         |sum(w_i * a_i * exp(j*phi_i))| / sum(w_i * a_i)
  CF_phase_only: |sum(exp(j*phi_i))| / N

where w_i are the CW apodization weights (max-normalized).
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

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

_gpu_flock_spec = _ilu.spec_from_file_location(
    "_gpu_flock",
    os.path.join(os.path.dirname(__file__), "_gpu_flock.py"),
)
_gpu_flock_mod = _ilu.module_from_spec(_gpu_flock_spec)
_gpu_flock_spec.loader.exec_module(_gpu_flock_mod)
gpu_flock = _gpu_flock_mod.gpu_flock

from openlifu.bf.delay_methods.complex_weighted import ComplexWeighted
from openlifu.geo import Point
from run_gladys_nnunet import (
    PreSegmented,
    create_hemispherical_array,
    load_nifti_as_xarray,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cf_cw_batch")

SUBJECTS = ["GU008", "GU002", "GU010", "NC004"]
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
GRID_SPACING_MM = 0.5
C0 = 1500.0
GRID_MARGIN_MM = 10.0

DATA_ROOT = Path.home() / "Data/openlifu-validation"
MRI_DIR = DATA_ROOT / "datasets/birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI"
RESULTS_DIR = DATA_ROOT / "results"


def _mri_path(subj: str) -> Path:
    return MRI_DIR / f"{subj}_deface.nii"


def _label_path(subj: str) -> Path:
    return RESULTS_DIR / f"{subj}_nnunet_labels.nii.gz"


def run_subject(subj: str) -> dict:
    """Run CW reciprocal sim + CF analysis for one subject."""
    import xarray as xa
    from scipy.interpolate import RegularGridInterpolator

    t0_subj = time.time()
    print(f"\n{'='*78}")
    print(f" Subject: {subj}")
    print(f"{'='*78}")

    mri_path = _mri_path(subj)
    label_path = _label_path(subj)
    for p, label in [(mri_path, "MRI"), (label_path, "Labels")]:
        if not p.exists():
            print(f"  ERROR: {label} not found: {p}")
            return {"subject": subj, "error": f"{label} not found"}

    volume = load_nifti_as_xarray(mri_path)
    seg_method = PreSegmented(label_nifti_path=str(label_path))
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
    print(f"  Target (brain center): {target_mm.tolist()}")

    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])
    max_skull_dist_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_dist_per_axis))
    print(f"  Approach axis: {dim_names[approach_axis]} (axis {approach_axis})")

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
    print(f"  Sim grid: {grid_shape}")

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
    water_mat = seg_method.materials["water"]
    air_mask = sim_seg_arr == material_idx["air"]
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation

    target_point = Point(
        position=target_mm.copy(), id="brain_center",
        name="Brain Center Target", units="mm",
    )

    # Run CW reciprocal sim (this calls run_point_source_simulation internally).
    cw = ComplexWeighted(c0=C0, cfl=0.3, n_cycles=3, gpu=True)
    print(f"  Running CW reciprocal sim...")
    t0_sim = time.time()
    with gpu_flock():
        amplitudes, phases, f0 = cw._run_reciprocal_simulation_complex(
            arr, target_point, sim_params, transform=None,
        )
    sim_secs = time.time() - t0_sim
    print(f"  Reciprocal sim done in {sim_secs:.1f}s")

    # CW apodization weights (max-normalized amplitudes).
    _, apod = ComplexWeighted._weights_from_coefficients(amplitudes, phases, f0)

    # --- CF_raw: raw reciprocal amplitudes, no CW weighting ---
    phasors_raw = amplitudes * np.exp(1j * phases)
    cf_raw = float(np.abs(np.sum(phasors_raw)) / np.sum(amplitudes))

    # --- CF_cw: weighted by CW apodization ---
    wa = apod * amplitudes
    phasors_cw = wa * np.exp(1j * phases)
    wa_sum = float(np.sum(wa))
    cf_cw = float(np.abs(np.sum(phasors_cw)) / wa_sum) if wa_sum > 0 else float("nan")

    # --- CF_phase_only: uniform amplitude, phases only ---
    phasors_phase = np.exp(1j * phases)
    cf_phase = float(np.abs(np.sum(phasors_phase)) / N_ELEMENTS)

    def _db(cf):
        return 20.0 * np.log10(cf) if cf > 0 else float("-inf")

    n_dead = int(np.sum(apod < 0.1))
    amp_std = float(np.std(amplitudes))
    amp_mean = float(np.mean(amplitudes))
    phase_std = float(np.std(phases))

    elapsed = time.time() - t0_subj
    print(f"\n  --- {subj} CF Results ---")
    print(f"  CF_raw         = {cf_raw:.4f}  ({_db(cf_raw):+.2f} dB)")
    print(f"  CF_cw          = {cf_cw:.4f}  ({_db(cf_cw):+.2f} dB)")
    print(f"  CF_phase_only  = {cf_phase:.4f}  ({_db(cf_phase):+.2f} dB)")
    print(f"  Dead elements (apod < 0.1): {n_dead}/{N_ELEMENTS}")
    print(f"  Amplitude: mean={amp_mean:.4g}, std={amp_std:.4g}")
    print(f"  Phase std (rad): {phase_std:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "subject": subj,
        "CF_raw": cf_raw,
        "CF_raw_dB": _db(cf_raw),
        "CF_cw": cf_cw,
        "CF_cw_dB": _db(cf_cw),
        "CF_phase_only": cf_phase,
        "CF_phase_only_dB": _db(cf_phase),
        "n_dead_elements": n_dead,
        "n_elements": N_ELEMENTS,
        "amplitude_mean": amp_mean,
        "amplitude_std": amp_std,
        "phase_std_rad": phase_std,
        "amplitudes": amplitudes.tolist(),
        "phases": phases.tolist(),
        "apod": apod.tolist(),
        "target_mm": target_mm.tolist(),
        "approach_axis": approach_axis,
        "sim_seconds": sim_secs,
        "total_seconds": elapsed,
    }


def main():
    t_total = time.time()
    print("=" * 78)
    print("Coherence Factor (CW) batch analysis")
    print(f"Subjects: {SUBJECTS}")
    print("=" * 78)

    results = []
    for subj in SUBJECTS:
        r = run_subject(subj)
        results.append(r)

    # Cross-subject comparison table.
    print("\n" + "=" * 78)
    print(" CROSS-SUBJECT COMPARISON")
    print("=" * 78)
    hdr = f"{'Subject':>8}  {'CF_raw':>8}  {'dB':>7}  {'CF_cw':>8}  {'dB':>7}  {'CF_ph':>8}  {'dB':>7}  {'dead':>4}  {'amp_std':>8}  {'ph_std':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['subject']:>8}  ERROR: {r['error']}")
            continue
        print(
            f"{r['subject']:>8}  "
            f"{r['CF_raw']:8.4f}  {r['CF_raw_dB']:+7.2f}  "
            f"{r['CF_cw']:8.4f}  {r['CF_cw_dB']:+7.2f}  "
            f"{r['CF_phase_only']:8.4f}  {r['CF_phase_only_dB']:+7.2f}  "
            f"{r['n_dead_elements']:4d}  "
            f"{r['amplitude_std']:8.4g}  "
            f"{r['phase_std_rad']:7.4f}"
        )
    print("=" * 78)

    out_path = RESULTS_DIR / "coherence_factor_cw_batch.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total elapsed: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
