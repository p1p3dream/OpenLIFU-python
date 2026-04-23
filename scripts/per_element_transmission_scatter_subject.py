#!/usr/bin/env python3
"""Per-element transmission scatter diagnostic (subject-parametric).

Generalized from scripts/per_element_transmission_scatter_gu008.py. Takes
--subject, --mri-path, --label-path so it can run on any subject. Saves a
per-subject NPZ of (amp_observed, T_pred, skull_paths, entry_angles,
hit_flags) alongside the PNG so that a cross-subject comparison plot can be
produced (GU008 vs GU010) automatically once both have run.

For each of the 64 elements we compute:

  observed :  |c_i|  from the common-focal-time narrowband projection of
              the reciprocal k-wave sim (same machinery as
              coherence_factor_gu008.py / incidence_angle_amplitude_gu008.py).

  predicted:  T_bulk * T_interface_slab * (1/r)  where

                alpha_skull_dB_per_cm = 4.29 dB/cm @ 500 kHz
                T_bulk = 10**(-alpha * (path_mm/10) / 20)    # amplitude
                Z_skull = 4080 * 1900 kg/m^2/s = 7.752 MRayl
                Z_water = 1.5 MRayl
                R_normal = |Z_skull - Z_water| / (Z_skull + Z_water)
                T_interface_single = sqrt(1 - R_normal^2)
                T_interface_slab   = T_interface_single ** 2  # enter + exit
                geom = 1 / r  (element-to-target distance)

Normalize observed -> predicted on the OPEN-PATH elements. Outputs a 3-panel
scatter + fit stats, plus an NPZ artifact for cross-subject comparison.

LOCAL diagnostic; not committed.
"""
from __future__ import annotations

import argparse
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
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import spearmanr


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
logger = logging.getLogger("per_element_transmission_subject")

RESULTS_DIR = Path.home() / "Data/openlifu-validation/results"

N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
GRID_SPACING_MM = 0.5
RECIPROCAL_CFL = 0.3
RECIPROCAL_N_CYCLES = 3
C0 = 1500.0
GRID_MARGIN_MM = 10.0
RAY_STEP_MM = 0.25

# --- Physics model parameters ---
ALPHA_SKULL_DB_PER_CM = 4.29       # bulk skull absorption @ 500 kHz (dB/cm)
Z_SKULL = 4080.0 * 1900.0          # 7.752e6 kg/m^2/s = 7.752 MRayl
Z_WATER = 1.5e6                    # 1.5 MRayl


# -------------------------------------------------------------------
# Ray-skull geometry.
# -------------------------------------------------------------------
def ray_skull_geometry(apod, el_pos_axes, target_pos_axes):
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


# -------------------------------------------------------------------
# Narrowband projection (|c_i|).
# -------------------------------------------------------------------
def common_focal_time_amplitudes(
    *,
    per_element_signals: np.ndarray,
    dt: float,
    element_positions_raw: np.ndarray,
    target_pos_raw: np.ndarray,
    freq_hz: float,
    n_cycles: int,
    scl_to_m: float,
    c_max: float,
) -> np.ndarray:
    n_el, Nt = per_element_signals.shape
    fs = 1.0 / dt

    low = 0.9 * freq_hz
    high = 1.1 * freq_hz
    nyq = 0.5 * fs
    if high >= nyq:
        high = 0.95 * nyq
        low = max(low, 0.5 * freq_hz)
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")

    pulse_dur = n_cycles / freq_hz
    t_axis = np.arange(Nt) * dt

    arrival_samples = np.zeros(n_el, dtype=int)
    filtered = np.zeros_like(per_element_signals, dtype=np.float64)
    for el_i in range(n_el):
        filt = sosfiltfilt(sos, per_element_signals[el_i].astype(np.float64))
        filtered[el_i] = filt
        env = np.abs(hilbert(filt))
        earliest_arrival_s = (
            float(np.linalg.norm(element_positions_raw[el_i] - target_pos_raw))
            * scl_to_m / c_max
        )
        gate_start = max(0, int((earliest_arrival_s - 2 * dt) / dt))
        if gate_start >= len(env):
            gate_start = 0
        arrival_samples[el_i] = gate_start + int(np.argmax(env[gate_start:]))
    arrival_times = arrival_samples * dt

    complex_coeffs = np.zeros(n_el, dtype=np.complex128)
    for el_i in range(n_el):
        t_arr = arrival_times[el_i]
        t_lo = t_arr - pulse_dur
        t_hi = t_arr + pulse_dur
        i_lo = max(0, int(np.floor(t_lo / dt)))
        i_hi = min(Nt, int(np.ceil(t_hi / dt)) + 1)
        if i_hi <= i_lo:
            continue
        seg = filtered[el_i, i_lo:i_hi]
        tt = t_axis[i_lo:i_hi]
        ref = np.exp(-1j * 2 * np.pi * freq_hz * (tt - t_arr))
        integrand = seg * ref
        coeff = np.trapezoid(integrand, tt) / (tt[-1] - tt[0] if len(tt) > 1 else dt)
        complex_coeffs[el_i] = coeff
    return np.abs(complex_coeffs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True, help="Subject ID, e.g. GU008, GU010")
    ap.add_argument("--mri-path", required=True, type=Path, help="Path to T1 NIfTI")
    ap.add_argument("--label-path", required=True, type=Path, help="Path to nnU-Net labels NIfTI")
    ap.add_argument(
        "--results-dir", default=RESULTS_DIR, type=Path,
        help="Directory to write artifacts",
    )
    return ap.parse_args()


def maybe_write_comparison_plot(results_dir: Path) -> None:
    """If both GU008 and GU010 NPZ artifacts exist, produce the overlay plot."""
    npz_a = results_dir / "per_element_transmission_scatter_GU008.npz"
    npz_b = results_dir / "per_element_transmission_scatter_GU010.npz"
    if not (npz_a.exists() and npz_b.exists()):
        print(f"    [comparison skipped: need both {npz_a.name} and {npz_b.name}]")
        return

    a = np.load(npz_a)
    b = np.load(npz_b)

    out = results_dir / "per_element_transmission_scatter_gu008_vs_gu010.png"
    print(f"    Writing comparison plot -> {out}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: observed vs predicted, both subjects
    ax = axes[0]
    subjects = [("GU008", a, "tab:blue"), ("GU010", b, "tab:orange")]
    all_vals = []
    for name, d, color in subjects:
        T_pred = d["T_pred"]
        amp = d["amp_observed"]
        hit = d["hit_flags"].astype(bool)
        nohit = ~hit
        slope = float(d["slope"])
        intercept = float(d["intercept"])
        r2 = float(d["r2"])
        # hit markers
        ax.scatter(T_pred[hit], amp[hit], s=40, color=color, alpha=0.65,
                   edgecolors="k", linewidths=0.4,
                   label=f"{name} skull-hit (slope={slope:.2f}, R^2={r2:.2f})")
        ax.scatter(T_pred[nohit], amp[nohit], s=70, facecolor="white",
                   edgecolors=color, linewidths=1.6, marker="o",
                   label=f"{name} open-path (n={int(nohit.sum())})")
        all_vals.extend([T_pred.min(), T_pred.max(), amp.min(), amp.max()])
        # fit line
        finite = np.isfinite(T_pred) & np.isfinite(amp) & (T_pred > 0) & (amp > 0)
        if finite.any():
            log_pred = np.log10(T_pred[finite])
            xs = np.linspace(log_pred.min(), log_pred.max(), 32)
            ys = slope * xs + intercept
            ax.plot(10 ** xs, 10 ** ys, color=color, lw=1.5, ls="--", alpha=0.8)

    vmin = max(min(v for v in all_vals if v > 0) * 0.5, 1e-12)
    vmax = max(all_vals) * 2
    xx = np.array([vmin, vmax])
    ax.plot(xx, xx, "k-", lw=1, alpha=0.5, label="y = x (slope 1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted transmission (T_bulk * T_int * 1/r), normalized")
    ax.set_ylabel("Observed amplitude |c_i|")
    ax.set_title("GU008 vs GU010: per-element transmission (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # Panel 2: residual (log10 ratio) vs skull path, both subjects
    ax2 = axes[1]
    ax2.axhline(0.0, color="k", ls="--", lw=1, alpha=0.5)
    for name, d, color in subjects:
        T_pred = d["T_pred"]
        amp = d["amp_observed"]
        paths = d["skull_paths"]
        hit = d["hit_flags"].astype(bool)
        ratio = amp / np.maximum(T_pred, 1e-30)
        log_ratio = np.log10(ratio)
        rho = float(d["spearman_rho_path"])
        ax2.scatter(paths[hit], log_ratio[hit], s=40, color=color, alpha=0.65,
                    edgecolors="k", linewidths=0.4,
                    label=f"{name} (rho={rho:.2f})")
        ax2.scatter(paths[~hit], log_ratio[~hit], s=70, facecolor="white",
                    edgecolors=color, linewidths=1.6, marker="o")
    ax2.set_xlabel("Skull path length along ray (mm)")
    ax2.set_ylabel("log10(observed / predicted)")
    ax2.set_title("Residual vs path: GU008 vs GU010")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "Subject comparison: per-element transmission (reciprocal sim) vs physics model "
        f"(alpha={ALPHA_SKULL_DB_PER_CM} dB/cm)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


def main() -> int:
    args = parse_args()
    subject = args.subject
    mri_path = Path(args.mri_path).expanduser()
    label_path = Path(args.label_path).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_path = results_dir / f"per_element_transmission_scatter_{subject}.png"
    npz_path = results_dir / f"per_element_transmission_scatter_{subject}.npz"

    t_total = time.time()
    print("=" * 78)
    print(f"Per-element transmission scatter (observed vs model-predicted) -- {subject}")
    print("=" * 78)

    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}")
        return 1
    if not label_path.exists():
        print(f"ERROR: nnU-Net labels not found: {label_path}")
        return 1

    # ---- Load MRI + labels ----
    print(f"\n[1] Loading MRI: {mri_path}")
    volume = load_nifti_as_xarray(mri_path)
    print(f"[2] Loading nnU-Net labels: {label_path}")
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
    print(f"    Brain center target: ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    skull_mask_full = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask_full)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])
    max_skull_dist_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_dist_per_axis))
    approach_dim = dim_names[approach_axis]
    print(f"    Approach axis: {approach_dim} (axis {approach_axis})")

    # ---- Build array ----
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

    # ---- Sim grid ----
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
    print(f"[4] Sim grid: {grid_shape_xyz} = {int(np.prod(grid_shape_xyz)):,d} voxels @ {GRID_SPACING_MM} mm")

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

    # ---- Reciprocal sim masks ----
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

    # ---- Reciprocal sim ----
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

    # Map sensor idx -> column (Fortran order in xyz-transposed mask).
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

    per_element_signals = np.zeros((N_ELEMENTS, Nt), dtype=sensor_data.dtype)
    for el_i, sensor_idx in enumerate(sensor_indices):
        sensor_idx_xyz = tuple(sensor_idx[i] for i in perm_to_xyz)
        col = voxel_to_col[sensor_idx_xyz]
        per_element_signals[el_i] = sensor_data[:, col]

    c_max = float(np.max(sim_params["sound_speed"].to_numpy()))
    c_max = max(c_max, C0)

    # ---- Observed amplitudes via common-focal-time narrowband projection ----
    print(f"[7] Extracting per-element observed amplitudes (bandpass + narrowband DFT) ...")
    amp_observed = common_focal_time_amplitudes(
        per_element_signals=per_element_signals,
        dt=dt,
        element_positions_raw=element_positions_raw,
        target_pos_raw=target_pos_raw,
        freq_hz=FREQ_HZ,
        n_cycles=RECIPROCAL_N_CYCLES,
        scl_to_m=scl_to_m,
        c_max=c_max,
    )
    print(f"    |c_i| mean={amp_observed.mean():.3g} median={np.median(amp_observed):.3g} "
          f"min={amp_observed.min():.3g} max={amp_observed.max():.3g}")

    # ---- Build skull SDF + ray-cast per element ----
    print(f"[8] Building skull SDF and ray-casting ...")
    skull_mask_sim = (sim_seg_arr == material_idx["skull"]).astype(np.uint8)
    skull_mask_da = xa.DataArray(
        skull_mask_sim,
        dims=sim_seg.dims,
        coords={d: sim_seg.coords[d] for d in sim_seg.dims},
    )
    t_sdf0 = time.time()
    apod = SkullIncidenceApodization(
        skull_mask=skull_mask_da,
        min_angle_deg=0.0,
        rolloff_angle_deg=90.0,
        step_mm=RAY_STEP_MM,
        coord_units=coord_units,
    )
    print(f"    SDF + gradient cached in {time.time() - t_sdf0:.1f}s")

    sdf_dims = list(skull_mask_da.dims)

    def xyz_to_dims(vec3_xyz, dims):
        out = np.empty(3, dtype=float)
        for j, d in enumerate(dims):
            out[j] = vec3_xyz[_DIM_IDX[d]]
        return out

    entry_angles = np.full(N_ELEMENTS, np.nan)
    skull_paths = np.zeros(N_ELEMENTS)
    hit_flags = np.zeros(N_ELEMENTS, dtype=bool)
    target_axes = xyz_to_dims(target_pos_raw, sdf_dims)
    for el_i in range(N_ELEMENTS):
        el_axes = xyz_to_dims(element_positions_raw[el_i], sdf_dims)
        ang, path, hit = ray_skull_geometry(apod, el_axes, target_axes)
        entry_angles[el_i] = ang
        skull_paths[el_i] = path
        hit_flags[el_i] = hit

    no_intersection = ~hit_flags
    n_open = int(no_intersection.sum())
    print(f"    Open-path (no_intersection) elements: {n_open} / {N_ELEMENTS}")
    if n_open < N_ELEMENTS:
        print(f"    Hit elements: path median={np.nanmedian(skull_paths[hit_flags]):.2f} mm, "
              f"max={np.nanmax(skull_paths[hit_flags]):.2f}, "
              f"angle median={np.nanmedian(entry_angles[hit_flags]):.1f} deg")

    # ---- Model prediction ----
    print(f"[9] Computing model-predicted transmission ...")
    T_bulk = np.power(10.0, -ALPHA_SKULL_DB_PER_CM * (skull_paths / 10.0) / 20.0)

    R_normal = abs(Z_SKULL - Z_WATER) / (Z_SKULL + Z_WATER)
    T_interface_single = float(np.sqrt(1.0 - R_normal ** 2))
    T_interface_slab = T_interface_single ** 2
    print(f"    Z_skull={Z_SKULL:.3e}  Z_water={Z_WATER:.3e}")
    print(f"    R_normal={R_normal:.4f}  T_interface_single={T_interface_single:.4f}  "
          f"T_interface_slab={T_interface_slab:.4f}")

    T_int = np.where(hit_flags, T_interface_slab, 1.0)

    dists_m = np.linalg.norm(element_positions_raw - target_pos_raw, axis=1) * scl_to_m
    geom = 1.0 / dists_m
    print(f"    element-to-target distance (mm): median={np.median(dists_m)*1000:.1f}  "
          f"min={dists_m.min()*1000:.1f}  max={dists_m.max()*1000:.1f}  "
          f"spread={(dists_m.max()-dists_m.min())*1000:.2f}")

    T_pred_raw = T_bulk * T_int * geom

    # ---- Normalize observed to predicted on open-path elements ----
    if n_open >= 2:
        scale = float(np.median(amp_observed[no_intersection] / T_pred_raw[no_intersection]))
    elif n_open == 1:
        idx = int(np.argmax(no_intersection))
        scale = float(amp_observed[idx] / T_pred_raw[idx])
    else:
        top = np.argsort(T_pred_raw)[-6:]
        scale = float(np.median(amp_observed[top] / T_pred_raw[top]))
        print("    WARNING: no open-path elements; normalizing on top-6 predicted instead.")
    T_pred = T_pred_raw * scale
    print(f"    Normalization scale (observed / predicted on open-path): {scale:.4g}")

    # ---- Fit / stats in log space ----
    finite = np.isfinite(amp_observed) & np.isfinite(T_pred) & (amp_observed > 0) & (T_pred > 0)
    log_obs = np.log10(amp_observed[finite])
    log_pred = np.log10(T_pred[finite])

    slope, intercept = np.polyfit(log_pred, log_obs, 1)
    pred_hat = slope * log_pred + intercept
    ss_res = float(np.sum((log_obs - pred_hat) ** 2))
    ss_tot = float(np.sum((log_obs - log_obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson_r = float(np.corrcoef(log_pred, log_obs)[0, 1])

    ratio = amp_observed / np.maximum(T_pred, 1e-30)
    log_ratio = np.log10(ratio)

    hit_finite = hit_flags & np.isfinite(entry_angles) & np.isfinite(log_ratio)
    corr_path_spear, p_path = (float("nan"), float("nan"))
    corr_theta_spear, p_theta = (float("nan"), float("nan"))
    if hit_finite.sum() >= 4:
        rho_p, p_p = spearmanr(skull_paths[hit_finite], log_ratio[hit_finite])
        rho_t, p_t = spearmanr(entry_angles[hit_finite], log_ratio[hit_finite])
        corr_path_spear, p_path = float(rho_p), float(p_p)
        corr_theta_spear, p_theta = float(rho_t), float(p_t)

    if n_open >= 3:
        log_obs_o = np.log10(amp_observed[no_intersection])
        log_pred_o = np.log10(T_pred[no_intersection])
        slope_o, intercept_o = np.polyfit(log_pred_o, log_obs_o, 1)
        yhat_o = slope_o * log_pred_o + intercept_o
        ss_res_o = float(np.sum((log_obs_o - yhat_o) ** 2))
        ss_tot_o = float(np.sum((log_obs_o - log_obs_o.mean()) ** 2))
        r2_open = 1.0 - ss_res_o / ss_tot_o if ss_tot_o > 0 else float("nan")
        ratio_o = amp_observed[no_intersection] / T_pred[no_intersection]
        ratio_o_std_log = float(np.std(np.log10(ratio_o)))
    else:
        r2_open = float("nan")
        slope_o = float("nan")
        ratio_o_std_log = float("nan")

    # Effective alpha implied by the fitted slope
    alpha_eff_db_per_cm = ALPHA_SKULL_DB_PER_CM * slope

    print("\n" + "=" * 78)
    print(f" Fit stats (log10 observed vs log10 predicted) -- {subject}")
    print("=" * 78)
    print(f"   N (finite):               {int(finite.sum())}")
    print(f"   slope (log-log):          {slope:.4f}")
    print(f"   intercept (log-log):      {intercept:.4f}")
    print(f"   R^2 (log-log):            {r2:.4f}")
    print(f"   Pearson r (log-log):      {pearson_r:.4f}")
    print(f"   Spearman rho(resid, path):  {corr_path_spear:.4f}  (p={p_path:.3g})")
    print(f"   Spearman rho(resid, theta): {corr_theta_spear:.4f}  (p={p_theta:.3g})")
    print(f"   Open-path N:              {n_open}")
    print(f"   Open-path R^2:            {r2_open:.4f}")
    print(f"   Open-path slope:          {slope_o:.4f}")
    print(f"   Open-path log10(ratio) std: {ratio_o_std_log:.4f}")
    print(f"   alpha_nominal (dB/cm):    {ALPHA_SKULL_DB_PER_CM:.3f}")
    print(f"   alpha_effective = slope * alpha_nominal: {alpha_eff_db_per_cm:.3f} dB/cm")

    # ---- Save NPZ artifact ----
    np.savez(
        npz_path,
        subject=subject,
        amp_observed=amp_observed,
        T_pred=T_pred,
        T_pred_raw=T_pred_raw,
        skull_paths=skull_paths,
        entry_angles=entry_angles,
        hit_flags=hit_flags,
        slope=slope,
        intercept=intercept,
        r2=r2,
        r2_open=r2_open,
        slope_open=slope_o,
        spearman_rho_path=corr_path_spear,
        spearman_rho_theta=corr_theta_spear,
        alpha_nominal_db_per_cm=ALPHA_SKULL_DB_PER_CM,
        alpha_effective_db_per_cm=alpha_eff_db_per_cm,
        n_open=n_open,
        normalization_scale=scale,
    )
    print(f"\n    Wrote NPZ artifact: {npz_path}")

    # ---- Plot: 3 panels ----
    print(f"\n[10] Plotting to {plot_path} ...")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axA = axes[0]
    mask_open = no_intersection
    mask_hit = hit_flags
    vmin = min(T_pred[finite].min(), amp_observed[finite].min())
    vmax = max(T_pred[finite].max(), amp_observed[finite].max())
    xx = np.array([vmin, vmax])
    axA.plot(xx, xx, "k--", lw=1, alpha=0.6, label="y = x")
    x_fit_log = np.linspace(log_pred.min(), log_pred.max(), 32)
    y_fit_log = slope * x_fit_log + intercept
    axA.plot(10 ** x_fit_log, 10 ** y_fit_log, "r-", lw=1.5,
             label=f"fit: slope={slope:.2f}, R^2={r2:.2f}")

    sc = axA.scatter(
        T_pred[mask_hit], amp_observed[mask_hit],
        c=skull_paths[mask_hit], cmap="viridis",
        s=70, edgecolors="k", linewidths=0.5, label=f"skull-hit (n={int(mask_hit.sum())})",
    )
    axA.scatter(
        T_pred[mask_open], amp_observed[mask_open],
        c="white", edgecolors="red", linewidths=1.5, s=90, marker="o",
        label=f"open-path (n={n_open})",
    )
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("Predicted transmission (T_bulk * T_interface * 1/r), normalized")
    axA.set_ylabel("Observed amplitude |c_i| (reciprocal sim, narrowband)")
    axA.set_title("Panel A: observed vs predicted (log-log)")
    axA.grid(True, which="both", alpha=0.3)
    axA.legend(loc="lower right", fontsize=9)
    cb = fig.colorbar(sc, ax=axA)
    cb.set_label("skull path length (mm)")

    axB = axes[1]
    axB.axhline(0.0, color="k", ls="--", lw=1, alpha=0.6)
    sc_b = axB.scatter(
        skull_paths[mask_hit], log_ratio[mask_hit],
        c=entry_angles[mask_hit], cmap="plasma",
        s=70, edgecolors="k", linewidths=0.5,
    )
    axB.scatter(
        skull_paths[mask_open], log_ratio[mask_open],
        c="white", edgecolors="red", linewidths=1.5, s=90, marker="o",
        label=f"open-path",
    )
    axB.set_xlabel("Skull path length along ray (mm)")
    axB.set_ylabel("log10(observed / predicted)")
    axB.set_title(f"Panel B: residual vs path\nSpearman rho={corr_path_spear:.2f} (p={p_path:.2g})")
    axB.grid(True, alpha=0.3)
    cb_b = fig.colorbar(sc_b, ax=axB)
    cb_b.set_label("entry angle (deg)")
    axB.legend(loc="lower left", fontsize=9)

    axC = axes[2]
    axC.axhline(0.0, color="k", ls="--", lw=1, alpha=0.6)
    sc_c = axC.scatter(
        entry_angles[mask_hit], log_ratio[mask_hit],
        c=skull_paths[mask_hit], cmap="viridis",
        s=70, edgecolors="k", linewidths=0.5,
    )
    axC.set_xlabel("Skull entry angle (deg, 0 = normal)")
    axC.set_ylabel("log10(observed / predicted)")
    axC.set_title(f"Panel C: residual vs theta\nSpearman rho={corr_theta_spear:.2f} (p={p_theta:.2g})")
    axC.grid(True, alpha=0.3)
    cb_c = fig.colorbar(sc_c, ax=axC)
    cb_c.set_label("skull path length (mm)")

    fig.suptitle(
        f"{subject} per-element transmission: observed |c_i| vs physics model "
        f"(alpha={ALPHA_SKULL_DB_PER_CM} dB/cm, Z_skull={Z_SKULL/1e6:.2f} MRayl)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {plot_path}")

    # ---- Per-element table ----
    print("\n" + "=" * 78)
    print(f" Per-element table (sorted by predicted ascending) -- {subject}")
    print("=" * 78)
    print(f"{'rank':>4} {'el':>3} {'path_mm':>8} {'theta':>7} "
          f"{'T_pred':>12} {'a_obs':>12} {'ratio':>10} {'open?':>6}")
    sort_idx = np.argsort(T_pred)
    for rank, el_i in enumerate(sort_idx):
        op = "YES" if no_intersection[el_i] else ""
        ratio_v = amp_observed[el_i] / max(T_pred[el_i], 1e-30)
        print(f"{rank:>4d} {el_i:>3d} {skull_paths[el_i]:8.2f} "
              f"{entry_angles[el_i]:7.2f} {T_pred[el_i]:12.4g} {amp_observed[el_i]:12.4g} "
              f"{ratio_v:10.3f} {op:>6s}")

    # ---- Try to produce the comparison plot ----
    print("\n[11] Attempting subject-vs-subject comparison plot ...")
    maybe_write_comparison_plot(results_dir)

    print("\n" + "=" * 78)
    print(f" VERDICT -- {subject}")
    print("=" * 78)
    print(f"  R^2 (log-log):              {r2:.3f}")
    print(f"  fit slope:                  {slope:.3f}")
    print(f"  open-path R^2:              {r2_open:.3f}")
    print(f"  Spearman rho(resid, path):  {corr_path_spear:.3f}")
    print(f"  Spearman rho(resid, theta): {corr_theta_spear:.3f}")
    print(f"  alpha_effective:            {alpha_eff_db_per_cm:.3f} dB/cm  "
          f"(nominal {ALPHA_SKULL_DB_PER_CM:.3f}, ratio {slope:.3f})")

    print(f"\nTotal elapsed: {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
