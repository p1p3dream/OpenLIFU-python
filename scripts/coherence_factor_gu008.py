#!/usr/bin/env python3
"""Coherence Factor diagnostic for GU008.

Runs ONLY the reciprocal k-wave sim (virtual point source at target, sensors
at the 64 transducer element voxels) used by SimulationCorrected, then
computes the complex coherence factor

    CF = | sum_i a_i * exp(j * phi_i) | / sum_i a_i

at the target voxel. By time-reversal reciprocity, the pressure time series
recorded at element i in the reciprocal sim is (up to amplitude constants)
the signal that element i WOULD deliver to the target voxel if it transmitted
its (phase-corrected) pulse, so per-element amplitude and narrowband phase at
500 kHz at the envelope peak give the spatial coherence of the transmit focus
AFTER SimulationCorrected's delay correction has removed the bulk arrival time.

    a_i   = envelope peak magnitude of the recorded signal at element i
    phi_i = phase of the analytic signal at the center frequency at the
            envelope-peak sample, median-subtracted across elements so the
            bulk arrival-time offset (which the delay correction removes)
            drops out.

CF == 1  -> perfectly coherent (ideal water-like focus)
CF == 0.5 -> 6 dB loss from decoherence
20*log10(CF) is the pressure-amplitude loss, in dB, due to incoherent
summation at the target voxel. If it accounts for the ~20 dB residual gap
between the measured 34 dB focal attenuation and the 23 dB predicted from
bulk absorption + impedance reflection, the spatial-decoherence hypothesis
is confirmed.

This script does NOT modify src/ or touch other scripts; it reuses
PreSegmented / create_hemispherical_array / load_nifti_as_xarray from
run_gladys_nnunet.py by importing them.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xa
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import hilbert

# -------------------------------------------------------------------
# Patch kwave's mis-formatted logging.log() calls (same patch that
# run_gladys_nnunet.py uses) before any kwave import.
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
sys.path.insert(0, os.path.dirname(__file__))  # so we can import run_gladys_nnunet

from openlifu.geo import Point  # noqa: E402
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
logger = logging.getLogger("coherence_gu008")

# ---- Must match run_gladys_nnunet.py exactly -------------------------------
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
CFL = 0.1                      # forward-sim CFL used in run_gladys_nnunet.py
RECIPROCAL_CFL = 0.3           # SimulationCorrected default
RECIPROCAL_N_CYCLES = 3        # SimulationCorrected default
C0 = 1500.0
GRID_MARGIN_MM = 10.0


def main() -> int:
    t_total = time.time()
    print("=" * 78)
    print("Coherence Factor diagnostic on GU008 (reciprocal sim only)")
    print("=" * 78)

    if not MRI_PATH.exists():
        print(f"ERROR: MRI not found: {MRI_PATH}")
        return 1
    if not LABEL_NIFTI_PATH.exists():
        print(f"ERROR: nnU-Net labels not found: {LABEL_NIFTI_PATH}")
        return 1

    # -------------------------------------------------------------------
    # Load MRI + pre-computed segmentation.
    # -------------------------------------------------------------------
    print(f"\n[1] Loading MRI: {MRI_PATH}")
    volume = load_nifti_as_xarray(MRI_PATH)
    print(f"    MRI shape={volume.shape}, dims={volume.dims}")

    print(f"\n[2] Loading nnU-Net labels: {LABEL_NIFTI_PATH}")
    seg_method = PreSegmented(label_nifti_path=str(LABEL_NIFTI_PATH))
    seg_labels = seg_method._segment(volume)
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()

    # -------------------------------------------------------------------
    # Pick brain-centroid target and approach axis EXACTLY like run_gladys_nnunet.
    # -------------------------------------------------------------------
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
    print(f"    Approach axis: {approach_dim} (axis {approach_axis})")

    # -------------------------------------------------------------------
    # Build array & bake world positions (verbatim logic from run_gladys_nnunet).
    # -------------------------------------------------------------------
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
    print(f"\n[3] Array: 64 elements baked to world frame; radius ~{RADIUS_MM:.0f} mm")

    # -------------------------------------------------------------------
    # Build sim grid.
    # -------------------------------------------------------------------
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

    # Resample MRI onto sim grid (for PreSegmented.seg_params).
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

    # Replace air voxels with water (same as run_gladys_nnunet.py).
    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    water_mat = seg_method.materials["water"]
    air_mask = sim_seg_arr == material_idx["air"]
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation
    n_skull = int((sim_seg_arr == material_idx["skull"]).sum())
    print(f"    Bone fraction in sim grid: {100.0 * n_skull / sim_seg_arr.size:.2f}%")

    # -------------------------------------------------------------------
    # Build reciprocal-sim masks directly. Mirrors SimulationCorrected so the
    # resulting sensor_data layout matches what that method would have used.
    # -------------------------------------------------------------------
    coord_dims = list(sim_params.coords.dims)
    coord_units = sim_params[coord_dims[0]].attrs.get("units", "mm")
    _DIM_IDX = {"x": 0, "y": 1, "z": 2}
    scl_m_to_coord = getunitconversion("m", coord_units)
    scl_to_m = getunitconversion(coord_units, "m")

    # Element positions in coord_units (reuse what we just baked).
    matrix = np.eye(4)
    element_positions_raw = np.array([
        el.get_position(units="m", matrix=matrix) * scl_m_to_coord
        for el in arr.elements
    ])
    target_pos_raw = np.array(target_mm, dtype=float)  # already in mm, coord_units is mm

    coord_axis_arrays = [sim_params.coords[dim].to_numpy() for dim in coord_dims]
    grid_shape_params = tuple(len(c) for c in coord_axis_arrays)

    # Nearest sensor voxel per element (in coord_dims order).
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
    print(f"\n[5] Sensor mask: {n_unique_sensor_vox} unique voxels "
          f"(out of {N_ELEMENTS} elements) "
          f"-- voxel collisions: {N_ELEMENTS - n_unique_sensor_vox}")

    # Source = target voxel.
    target_idx = tuple(
        int(np.argmin(np.abs(coord_axis_arrays[dim_i] - target_pos_raw[_DIM_IDX[dim_name]])))
        for dim_i, dim_name in enumerate(coord_dims)
    )
    source_mask = np.zeros(grid_shape_params, dtype=int)
    source_mask[target_idx] = 1

    # t_end like SimulationCorrected.
    dists_m = np.linalg.norm(element_positions_raw - target_pos_raw, axis=1) * scl_to_m
    max_dist_m = float(np.max(dists_m))
    t_end_needed = max_dist_m / C0 * 1.5 + RECIPROCAL_N_CYCLES / FREQ_HZ

    # -------------------------------------------------------------------
    # Run the reciprocal sim (only this one k-wave run).
    # -------------------------------------------------------------------
    print(f"\n[6] Reciprocal sim: point source @ target, sensors @ elements ...")
    print(f"    freq={FREQ_HZ / 1e3:.0f} kHz, cycles={RECIPROCAL_N_CYCLES}, "
          f"cfl={RECIPROCAL_CFL}, t_end={t_end_needed * 1e6:.1f} us")
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
    print(f"    Reciprocal sim done in {t_sim:.1f}s "
          f"(sensor_data shape={sensor_data.shape}, dt={dt * 1e9:.2f} ns)")

    # -------------------------------------------------------------------
    # Map sensor_indices (coord_dims order) -> Fortran-order column in
    # xyz-transposed mask (exactly like SimulationCorrected does).
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # Per-element narrowband amplitude & phase at 500 kHz at envelope peak.
    # -------------------------------------------------------------------
    print(f"\n[7] Extracting per-element amplitudes and narrowband phases ...")
    t_axis = np.arange(Nt) * dt
    # Narrowband demodulator: carrier at FREQ_HZ. phase_i at peak = angle of
    # analytic_signal(t_peak) * exp(-j * 2*pi*FREQ*t_peak).
    carrier = np.exp(-1j * 2 * np.pi * FREQ_HZ * t_axis)

    amplitudes = np.zeros(N_ELEMENTS)
    phases_raw = np.zeros(N_ELEMENTS)
    peak_samples = np.zeros(N_ELEMENTS, dtype=int)
    tof_samples = np.zeros(N_ELEMENTS, dtype=int)

    # Gate bound uses max sound speed in grid (bone ~3000 m/s) for earliest
    # plausible arrival, same as SimulationCorrected.
    c_max = float(np.max(sim_params["sound_speed"].to_numpy()))
    c_max = max(c_max, C0)

    for el_i, sensor_idx in enumerate(sensor_indices):
        sensor_idx_xyz = tuple(sensor_idx[i] for i in perm_to_xyz)
        col = voxel_to_col[sensor_idx_xyz]
        p_t = sensor_data[:, col]
        analytic = hilbert(p_t)
        envelope = np.abs(analytic)

        earliest_arrival_s = (
            np.linalg.norm(element_positions_raw[el_i] - target_pos_raw)
            * scl_to_m
            / c_max
        )
        gate_start = max(0, int((earliest_arrival_s - 2 * dt) / dt))
        if gate_start >= len(envelope):
            gate_start = 0
        peak_sample = gate_start + int(np.argmax(envelope[gate_start:]))
        peak_samples[el_i] = peak_sample

        # Amplitude = envelope peak
        amplitudes[el_i] = float(envelope[peak_sample])

        # Narrowband phase at the envelope peak sample, relative to the
        # 500 kHz carrier (so the unwrapped phase is ~constant if the signal
        # is a clean tone burst).
        demod = analytic[peak_sample] * carrier[peak_sample]
        phases_raw[el_i] = float(np.angle(demod))

        # Geometric TOF (for diagnostics only).
        tof_samples[el_i] = int(round(
            (np.linalg.norm(element_positions_raw[el_i] - target_pos_raw) * scl_to_m / C0)
            / dt
        ))

    # -------------------------------------------------------------------
    # Normalize phases: subtract the circular median so median(phi_i) == 0.
    # The bulk arrival-time offset that the delay correction would cancel
    # is a constant phase across elements, so factoring out the median
    # (robust to a few outliers) leaves only the RESIDUAL per-element
    # incoherence.
    # -------------------------------------------------------------------
    # "Circular median" via the phase of the median of (cos, sin).
    cos_med = float(np.median(np.cos(phases_raw)))
    sin_med = float(np.median(np.sin(phases_raw)))
    median_phase = float(np.arctan2(sin_med, cos_med))
    phases = np.angle(np.exp(1j * (phases_raw - median_phase)))  # wrap to (-pi, pi]

    # -------------------------------------------------------------------
    # Complex coherence factor.
    # -------------------------------------------------------------------
    a_sum = float(np.sum(amplitudes))
    vec_sum = np.sum(amplitudes * np.exp(1j * phases))
    CF = float(np.abs(vec_sum) / a_sum) if a_sum > 0 else float("nan")
    CF_db = 20.0 * np.log10(CF) if CF > 0 else float("-inf")

    # Amplitude-weighted phase statistics (circular).
    w = amplitudes / a_sum if a_sum > 0 else np.ones_like(amplitudes) / len(amplitudes)
    mean_cos = float(np.sum(w * np.cos(phases)))
    mean_sin = float(np.sum(w * np.sin(phases)))
    R = float(np.sqrt(mean_cos ** 2 + mean_sin ** 2))  # circular mean resultant
    circ_std_weighted = float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))

    # Unweighted stats (for direct intuition).
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    R_unw = float(np.sqrt(cos_mean ** 2 + sin_mean ** 2))
    circ_std_unweighted = float(np.sqrt(-2.0 * np.log(max(R_unw, 1e-12))))

    # Per-element wrapped-phase std (linear stat - coarse; for intuition only).
    phase_std_rad = float(np.std(phases))
    phase_std_cycles = phase_std_rad / (2 * np.pi)
    amp_mean = float(np.mean(amplitudes))
    amp_std = float(np.std(amplitudes))
    amp_min = float(np.min(amplitudes))
    amp_max = float(np.max(amplitudes))

    # -------------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(" COHERENCE FACTOR RESULT")
    print("=" * 78)
    print(f"  CF                 = {CF:.4f}")
    print(f"  20*log10(CF)       = {CF_db:.2f} dB  (negative = pressure-amplitude loss)")
    print(f"  -> unexplained gap = 20 dB;   predicted from CF = {-CF_db:.2f} dB")
    if np.isfinite(CF_db):
        residual = 20.0 - (-CF_db)
        verdict = "CONFIRMED" if abs(residual) <= 3.0 else (
            "PARTIAL" if -CF_db > 10.0 else "NOT SUPPORTED"
        )
        print(f"  residual after CF  = {residual:+.2f} dB   verdict: {verdict}")

    print("\n  Amplitude distribution across 64 elements:")
    print(f"    mean={amp_mean:.3g}  std={amp_std:.3g}  min={amp_min:.3g}  max={amp_max:.3g}")
    print(f"    dynamic range (max/min) = {amp_max / max(amp_min, 1e-30):.2f}x")
    # 10-bin amplitude histogram
    amp_bins = np.linspace(amp_min, amp_max, 11) if amp_max > amp_min else np.linspace(0, 1, 11)
    amp_hist, _ = np.histogram(amplitudes, bins=amp_bins)
    print("    hist (edges -> count):")
    for lo, hi, c in zip(amp_bins[:-1], amp_bins[1:], amp_hist):
        bar = "#" * int(c)
        print(f"      [{lo:>10.3g} .. {hi:<10.3g}] {c:>3d} {bar}")

    print("\n  Phase (median-centered) distribution in radians:")
    print(f"    std (linear, rad)             = {phase_std_rad:.4f}  ({phase_std_cycles:.4f} cycles)")
    print(f"    circular std, unweighted      = {circ_std_unweighted:.4f} rad")
    print(f"    circular std, amplitude-weight = {circ_std_weighted:.4f} rad   "
          f"({circ_std_weighted / (2 * np.pi):.4f} cycles)")
    print(f"    circular mean resultant R      = {R:.4f}  (amplitude-weighted)")
    # 12-bin phase histogram over (-pi, pi]
    ph_bins = np.linspace(-np.pi, np.pi, 13)
    ph_hist, _ = np.histogram(phases, bins=ph_bins)
    print("    hist (rad edges -> count):")
    for lo, hi, c in zip(ph_bins[:-1], ph_bins[1:], ph_hist):
        bar = "#" * int(c)
        print(f"      [{lo:>+5.2f} .. {hi:<+5.2f}] {c:>3d} {bar}")

    # Per-element table
    print("\n  Per-element details (el, dist_mm, a_i, phi_rad, phi_cycles):")
    for el_i in range(N_ELEMENTS):
        d_mm = float(np.linalg.norm(element_positions_raw[el_i] - target_pos_raw))
        print(f"    el={el_i:02d}  d={d_mm:6.2f}mm  a={amplitudes[el_i]:.3g}  "
              f"phi={phases[el_i]:+.3f}rad  ({phases[el_i] / (2 * np.pi):+.3f} cyc)")

    # Summary line for grep-friendliness.
    print("\n" + "=" * 78)
    print(f" SUMMARY  CF={CF:.4f}  20log10(CF)={CF_db:+.2f}dB  "
          f"phase_std_weighted={circ_std_weighted:.3f}rad  "
          f"amp_dyn={amp_max / max(amp_min, 1e-30):.1f}x")
    print("=" * 78)

    print(f"\nTotal elapsed: {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
