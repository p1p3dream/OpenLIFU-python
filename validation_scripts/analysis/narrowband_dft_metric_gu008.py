#!/usr/bin/env python3
"""Narrowband DFT vs time-domain max-peak focal-quality metric probe on GU008.

Purpose
-------
Test whether a narrowband DFT at the source frequency (f0 = 500 kHz) applied
over the same per-voxel water-calibrated focal window produces a more stable
spatial-search result than the current `max |p(t)|` time-domain metric.

Approach
--------
1. Monkey-patch ``run_gladys_nnunet_subject._run_sparse_sensor_sim`` and
   ``_build_sensor_mask`` to capture the probe's per-sensor time series,
   dt, sensor names, and world positions from each sim (water/geometric/
   corrected).
2. Invoke ``run_gladys_nnunet_subject.main()`` via argv override on
   subject GU008 with ``EXPANDED_TARGET_PROBE=1``,
   ``EXPANDED_TARGET_HALF_MM=15``, ``SIDECAR_MAX_VOXELS=500``,
   ``OUTPUT_TAG=nb_``.
3. After the sim completes, for each sim (water, corrected+skull), compute
   per-voxel:
       time-domain metric    = max |p(t)| over per-voxel window
       narrowband DFT metric = |sum_t p(t) * exp(-j 2 pi f0 t)| / sqrt(N_win)
   with identical per-voxel windows (centered on water-calibrated gate,
   half-width 2 * pulse_dur). Report spatial argmax, amplitude at target,
   amplitude at argmax, offset from target, and peak/target ratio for both
   metrics.

Not committed. Run on stonkbot under flock via the invocation in the task
description.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent

# --- Load _probe_helpers by path (same trick as other local scripts) -------
_ph_spec = importlib.util.spec_from_file_location(
    "_probe_helpers", HERE / "_probe_helpers.py"
)
_probe_helpers = importlib.util.module_from_spec(_ph_spec)
_ph_spec.loader.exec_module(_probe_helpers)  # type: ignore[attr-defined]
per_voxel_gate = _probe_helpers.per_voxel_water_calibrated_gate_center


# --- Output locations ------------------------------------------------------
OUT_DIR = pathlib.Path(
    os.environ.get("NB_OUT_DIR") or "/mnt/data/tmp/brandon/narrowband_gu008"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Capture state (populated by monkey-patched simulation hooks) ----------
@dataclass
class CaptureSlot:
    """Holds one sim's probe capture (mask + time series + labels).

    There are three sims per subject (water/corrected/geometric). The
    patched ``_build_sensor_mask`` fills ``names/xyz_idx/actuals``; the
    patched ``_run_sparse_sensor_sim`` then fills ``p_sensor/dt/xyz_order``.
    """
    names: list[str] = field(default_factory=list)
    xyz_idx: list[tuple[int, int, int]] = field(default_factory=list)
    actuals: list[np.ndarray] = field(default_factory=list)
    p_sensor: np.ndarray | None = None
    dt: float | None = None
    xyz_order: np.ndarray | None = None


_captures: list[CaptureSlot] = []
_current_slot: CaptureSlot | None = None


def _install_monkey_patches(rgns_module):
    """Install wrappers on _build_sensor_mask and _run_sparse_sensor_sim.

    The wrappers call the originals and also push captured data into
    ``_captures`` (a new CaptureSlot is opened on each _build_sensor_mask
    call since that is always the first call per sim).
    """
    orig_build_mask = rgns_module._build_sensor_mask
    orig_run_sparse = rgns_module._run_sparse_sensor_sim

    def patched_build_mask(*args, **kwargs):
        global _current_slot
        out = orig_build_mask(*args, **kwargs)
        mask, names, xyz_idx_list, actual_list = out
        slot = CaptureSlot(
            names=list(names),
            xyz_idx=[tuple(int(v) for v in t) for t in xyz_idx_list],
            actuals=[np.array(a, dtype=float).copy() for a in actual_list],
        )
        _captures.append(slot)
        _current_slot = slot
        return out

    def patched_run_sparse(*args, **kwargs):
        global _current_slot
        p_sensor, dt_probe, xyz_order = orig_run_sparse(*args, **kwargs)
        if _current_slot is not None:
            _current_slot.p_sensor = np.asarray(p_sensor, dtype=np.float32)
            _current_slot.dt = float(dt_probe)
            _current_slot.xyz_order = np.asarray(xyz_order, dtype=np.int64).copy()
            _current_slot = None  # consumed
        return p_sensor, dt_probe, xyz_order

    rgns_module._build_sensor_mask = patched_build_mask
    rgns_module._run_sparse_sensor_sim = patched_run_sparse


# --- Metric helpers --------------------------------------------------------
def _window_indices(n_samples: int, dt: float, t_center: float, t_half: float):
    t_lo = t_center - t_half
    t_hi = t_center + t_half
    i_lo = max(0, int(np.floor(t_lo / dt)))
    i_hi = min(n_samples, int(np.ceil(t_hi / dt)) + 1)
    return i_lo, i_hi


def compute_per_voxel_metrics(
    *,
    p_sensor: np.ndarray,      # (Nt, Ns)
    dt: float,
    t_axis: np.ndarray,        # (Nt,)
    names: list[str],
    actuals: list[np.ndarray],
    target_mm: np.ndarray,
    aperture_center_mm: np.ndarray,
    target_gate_center_s: float,
    pulse_dur_s: float,
    f0_hz: float,
    c0_mps: float,
):
    """Return a dict of per-voxel arrays: both metrics, positions, is_cube,
    per-voxel gate centers, window sample counts.
    """
    n_s = len(names)
    assert p_sensor.shape[1] == n_s, (p_sensor.shape, n_s)
    half_w = 2.0 * pulse_dur_s

    pos = np.stack(actuals, axis=0)  # (Ns, 3)
    td_metric = np.zeros(n_s, dtype=float)
    nb_metric = np.zeros(n_s, dtype=float)
    gate_center_s = np.zeros(n_s, dtype=float)
    n_win_samples = np.zeros(n_s, dtype=np.int64)
    is_cube = np.zeros(n_s, dtype=bool)

    for i, name in enumerate(names):
        cube_like = name.startswith("cube_") or name.startswith("shell_") or name == "target"
        is_cube[i] = cube_like
        if not cube_like:
            # Not part of spatial search (midway, quarterway, etc.). Skip
            # per-voxel gate; leave metrics at 0 for these; we won't use
            # them in the spatial-argmax analysis.
            continue
        # Per-voxel water-calibrated gate center (same as _probe_helpers).
        if name == "target":
            gc = float(target_gate_center_s)
        else:
            gc = per_voxel_gate(
                sensor_world_mm=pos[i],
                target_world_mm=target_mm,
                aperture_center_world_mm=aperture_center_mm,
                t_water_peak_target=float(target_gate_center_s),
                c0_mps=c0_mps,
            )
        gate_center_s[i] = gc

        i_lo, i_hi = _window_indices(p_sensor.shape[0], dt, gc, half_w)
        if i_hi <= i_lo:
            continue
        seg = p_sensor[i_lo:i_hi, i]
        n = int(seg.shape[0])
        n_win_samples[i] = n

        # Time-domain metric: max |p(t)| in window
        td_metric[i] = float(np.max(np.abs(seg)))

        # Narrowband DFT metric: |sum_t p(t) * exp(-j 2 pi f0 t)| over window
        # normalize by 2/N so that for a pure sin(2 pi f0 t) of amplitude A
        # the metric equals A (rectangular window, integer cycles).
        t_seg = t_axis[i_lo:i_hi]
        phasor = np.exp(-1j * 2.0 * np.pi * f0_hz * t_seg)
        Z = complex(np.sum(seg.astype(np.float64) * phasor))
        nb_metric[i] = float((2.0 / n) * abs(Z))

    return {
        "names": list(names),
        "positions_mm": pos,
        "is_cube": is_cube,
        "gate_center_s": gate_center_s,
        "n_win_samples": n_win_samples,
        "td_metric": td_metric,
        "nb_metric": nb_metric,
    }


def _argmax_summary(metric: np.ndarray, names: list[str], pos: np.ndarray,
                    is_cube: np.ndarray, target_mm: np.ndarray) -> dict:
    """Report argmax (restricted to cube voxels) and target-nearest amplitude."""
    idxs = np.flatnonzero(is_cube)
    if idxs.size == 0:
        return {"argmax_name": None, "argmax_world_mm": None, "argmax_amp": float("nan"),
                "target_nearest_amp": float("nan"), "offset_mm": float("nan"),
                "ratio_peak_over_target": float("nan")}
    sub_vals = metric[idxs]
    sub_pos = pos[idxs]
    k = int(np.argmax(sub_vals))
    gi = int(idxs[k])
    argmax_name = names[gi]
    argmax_world = pos[gi]
    argmax_amp = float(sub_vals[k])

    d_to_target = np.linalg.norm(sub_pos - target_mm[None, :], axis=1)
    ki = int(np.argmin(d_to_target))
    tgt_gi = int(idxs[ki])
    tgt_amp = float(sub_vals[ki])
    offset_mm = float(np.linalg.norm(argmax_world - target_mm))
    ratio = argmax_amp / tgt_amp if tgt_amp > 0 else float("nan")
    return {
        "argmax_name": argmax_name,
        "argmax_world_mm": [float(x) for x in argmax_world],
        "argmax_amp": argmax_amp,
        "target_nearest_name": names[tgt_gi],
        "target_nearest_world_mm": [float(x) for x in pos[tgt_gi]],
        "target_nearest_amp": tgt_amp,
        "offset_mm": offset_mm,
        "ratio_peak_over_target": ratio,
    }


# --- Main flow -------------------------------------------------------------
def main():
    # Force env flags for a clean probe.
    # Cube half = 20 mm, stride = 2 -> 21^3 = 9261 voxels at 2 mm spacing.
    # Diagonal reaches sqrt(3)*20 ~ 34.6 mm, so we have corner voxels well past
    # 30 mm from target for a "far-voxel" noise-floor estimate per metric.
    # ~10k voxels keeps captured p_sensor RAM modest
    # (9261 * 15240 * 4 bytes ~ 565 MB per sim, 3 sims -> ~1.7 GB).
    os.environ["EXPANDED_TARGET_PROBE"] = "1"
    os.environ.setdefault("EXPANDED_TARGET_HALF_MM", "20.0")
    os.environ.setdefault("CUBE_PROBE_STRIDE", "2")
    os.environ.setdefault("SIDECAR_MAX_VOXELS", "500")
    os.environ.setdefault("OUTPUT_TAG", "nb_")

    # Import the subject driver module fresh so env flags are honored.
    if "run_gladys_nnunet_subject" in sys.modules:
        del sys.modules["run_gladys_nnunet_subject"]
    sys.path.insert(0, str(HERE))
    import run_gladys_nnunet_subject as rgns  # noqa: E402

    _install_monkey_patches(rgns)

    # Override argv for rgns.main()
    saved_argv = sys.argv
    sys.argv = ["run_gladys_nnunet_subject.py", "--subject", "GU008"]
    print("[narrowband_dft_metric_gu008] launching rgns.main() for GU008...", flush=True)
    t0 = time.time()
    try:
        rgns.main()
    except SystemExit as e:
        print(f"[narrowband_dft_metric_gu008] rgns.main() SystemExit={e}", flush=True)
    finally:
        sys.argv = saved_argv
    dt_sim = time.time() - t0
    print(f"[narrowband_dft_metric_gu008] rgns.main() done in {dt_sim:.1f}s", flush=True)
    print(f"[narrowband_dft_metric_gu008] captured {len(_captures)} slot(s)", flush=True)

    # Sim order inside rgns.main() for subject flow (confirmed by reading source):
    #   1) SIM C water  (run first; target_gate_center_s=None)
    #   2) SIM A corrected+skull  (target_gate_center_s = water target peak)
    #   3) SIM B geometric+skull  (same water target peak)
    sim_labels = ["water", "corrected_skull", "geometric_skull"]
    if len(_captures) != 3:
        print(f"WARNING: expected 3 capture slots, got {len(_captures)}. "
              f"Results may be partial.", flush=True)

    # Pull params we need from the sidecars (for target_gate_center_s + target_mm).
    results_dir = pathlib.Path.home() / "Data/openlifu-validation/results"
    tag = os.environ["OUTPUT_TAG"]
    side_water = results_dir / f"GU008_{tag}gladys_nnunet_water_timegated.json"
    side_corr = results_dir / f"GU008_{tag}gladys_nnunet_corrected_timegated.json"
    side_geom = results_dir / f"GU008_{tag}gladys_nnunet_geometric_timegated.json"

    if not side_corr.exists():
        print(f"ERROR: corrected sidecar missing: {side_corr}", flush=True)
        return

    with open(side_corr) as f:
        d_corr = json.load(f)
    if side_water.exists():
        with open(side_water) as f:
            d_water = json.load(f)
    else:
        d_water = {}

    target_mm = np.array(d_corr["target_mm"], dtype=float)
    aperture_mm = np.array(d_corr["aperture_center_mm"], dtype=float)
    pulse_dur = float(d_corr["pulse_duration_s"])
    c0 = float(d_corr.get("c0_m_s", 1500.0))
    f0 = 500e3  # FREQ_HZ

    # target_gate_center_s in skull sims is the water target peak time.
    water_gate = d_corr.get("target_gate_center_s")
    if water_gate is None:
        water_gate = d_water.get("target_peak_time_s")
    if water_gate is None:
        raise RuntimeError("Could not resolve water target peak time")
    water_gate = float(water_gate)
    water_own_gate = d_water.get("target_peak_time_s", water_gate)

    print(f"target_mm={target_mm}  aperture={aperture_mm}", flush=True)
    print(f"pulse_dur={pulse_dur*1e6:.2f} us  f0={f0*1e-3:.0f} kHz  "
          f"c0={c0}  water_peak={water_gate*1e6:.3f} us", flush=True)

    # Analyze each captured slot.
    all_results = {}
    for i, slot in enumerate(_captures):
        label = sim_labels[i] if i < len(sim_labels) else f"slot_{i}"
        if slot.p_sensor is None or slot.dt is None:
            print(f"  slot {i} ({label}): no p_sensor captured; skipping", flush=True)
            continue

        # Reorder capture arrays to match per-sensor column order.
        # _build_sensor_mask returns (names, xyz_idx_list, actual_list) in
        # enumeration order. _run_sparse_sensor_sim returns xyz_order in
        # lexicographic (lin-index) order. We must reorder so that column j
        # of p_sensor corresponds to names[j'] where xyz_idx[j'] matches
        # xyz_order[j].
        col_index = {}
        for ci in range(slot.xyz_order.shape[0]):
            key = (int(slot.xyz_order[ci, 0]),
                   int(slot.xyz_order[ci, 1]),
                   int(slot.xyz_order[ci, 2]))
            col_index.setdefault(key, ci)

        Nt_ = int(slot.p_sensor.shape[0])
        dt_ = float(slot.dt)
        t_axis = np.arange(Nt_) * dt_
        Ns = len(slot.names)
        # Build column-aligned p view: p_aligned[:, i] is time series for names[i]
        p_aligned = np.zeros((Nt_, Ns), dtype=np.float32)
        missing = 0
        for i_name, xyz in enumerate(slot.xyz_idx):
            col = col_index.get(tuple(int(v) for v in xyz))
            if col is None:
                missing += 1
                continue
            p_aligned[:, i_name] = slot.p_sensor[:, col]

        # Use water_gate for skull sims; for water sim, use water's own
        # target peak time (since water IS the reference).
        gate_center_s = water_gate if label != "water" else float(water_own_gate)

        metrics = compute_per_voxel_metrics(
            p_sensor=p_aligned,
            dt=dt_,
            t_axis=t_axis,
            names=slot.names,
            actuals=slot.actuals,
            target_mm=target_mm,
            aperture_center_mm=aperture_mm,
            target_gate_center_s=gate_center_s,
            pulse_dur_s=pulse_dur,
            f0_hz=f0,
            c0_mps=c0,
        )
        td_summary = _argmax_summary(metrics["td_metric"], metrics["names"],
                                     metrics["positions_mm"], metrics["is_cube"],
                                     target_mm)
        nb_summary = _argmax_summary(metrics["nb_metric"], metrics["names"],
                                     metrics["positions_mm"], metrics["is_cube"],
                                     target_mm)

        # Noise-floor stats from voxels >30 mm from target (can't physically
        # contain direct focal arrival) and local-focal peak stats from
        # voxels within 5 mm of target.
        pos_all = metrics["positions_mm"]
        is_cube = metrics["is_cube"]
        d_to_target = np.linalg.norm(pos_all - target_mm[None, :], axis=1)
        far_mask = is_cube & (d_to_target > 30.0)
        near_mask = is_cube & (d_to_target <= 5.0)

        def _stats(mask, vals):
            if mask.sum() == 0:
                return {"n": 0}
            v = vals[mask]
            return {
                "n": int(mask.sum()),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "median": float(np.median(v)),
                "max": float(np.max(v)),
            }

        td_far = _stats(far_mask, metrics["td_metric"])
        nb_far = _stats(far_mask, metrics["nb_metric"])
        td_near = _stats(near_mask, metrics["td_metric"])
        nb_near = _stats(near_mask, metrics["nb_metric"])

        # SNR proxy: spatial-argmax amp / far-voxel mean amp.
        def _snr(peak, floor):
            if floor.get("n", 0) == 0 or floor["mean"] == 0:
                return float("nan")
            return peak / floor["mean"]

        td_snr = _snr(td_summary["argmax_amp"], td_far)
        nb_snr = _snr(nb_summary["argmax_amp"], nb_far)

        print(f"\n=== SIM {label} (slot {i}) ===", flush=True)
        print(f"  num cube-eligible voxels: {int(metrics['is_cube'].sum())}", flush=True)
        print(f"  gate_center_s = {gate_center_s*1e6:.3f} us   "
              f"window = +/- {2.0*pulse_dur*1e6:.2f} us   "
              f"missing cols: {missing}", flush=True)
        print(f"  TIME-DOMAIN max|p(t)| :")
        print(f"    argmax        = {td_summary['argmax_name']}  "
              f"@ {td_summary['argmax_world_mm']}  "
              f"amp={td_summary['argmax_amp']:.4g}")
        print(f"    target-nearest= {td_summary['target_nearest_name']}  "
              f"@ {td_summary['target_nearest_world_mm']}  "
              f"amp={td_summary['target_nearest_amp']:.4g}")
        print(f"    offset={td_summary['offset_mm']:.2f} mm   "
              f"peak/target={td_summary['ratio_peak_over_target']:.3f}")
        print(f"  NARROWBAND |DFT(p, f0)| :")
        print(f"    argmax        = {nb_summary['argmax_name']}  "
              f"@ {nb_summary['argmax_world_mm']}  "
              f"amp={nb_summary['argmax_amp']:.4g}")
        print(f"    target-nearest= {nb_summary['target_nearest_name']}  "
              f"@ {nb_summary['target_nearest_world_mm']}  "
              f"amp={nb_summary['target_nearest_amp']:.4g}")
        print(f"    offset={nb_summary['offset_mm']:.2f} mm   "
              f"peak/target={nb_summary['ratio_peak_over_target']:.3f}")

        print(f"  FAR-VOXEL NOISE FLOOR (dist > 30 mm, n={td_far['n']}):")
        if td_far["n"] > 0:
            print(f"    td_metric mean={td_far['mean']:.4g}  std={td_far['std']:.4g}  "
                  f"median={td_far['median']:.4g}  max={td_far['max']:.4g}")
            print(f"    nb_metric mean={nb_far['mean']:.4g}  std={nb_far['std']:.4g}  "
                  f"median={nb_far['median']:.4g}  max={nb_far['max']:.4g}")
            print(f"    SNR (argmax / far_mean):  td={td_snr:.2f}  nb={nb_snr:.2f}")
        else:
            print("    (no cube voxels beyond 30 mm from target)")
        print(f"  NEAR-FOCAL STATS (dist <= 5 mm, n={td_near['n']}):")
        if td_near["n"] > 0:
            print(f"    td_metric mean={td_near['mean']:.4g}  max={td_near['max']:.4g}")
            print(f"    nb_metric mean={nb_near['mean']:.4g}  max={nb_near['max']:.4g}")

        all_results[label] = {
            "td": td_summary,
            "nb": nb_summary,
            "td_far_stats": td_far,
            "nb_far_stats": nb_far,
            "td_near_stats": td_near,
            "nb_near_stats": nb_near,
            "td_snr_vs_far": td_snr,
            "nb_snr_vs_far": nb_snr,
            "gate_center_s": gate_center_s,
            "num_cube_voxels": int(metrics["is_cube"].sum()),
            "missing_cols": int(missing),
        }

        # Save compressed per-voxel arrays for this slot for possible reuse.
        npz_path = OUT_DIR / f"gu008_narrowband_{label}.npz"
        try:
            np.savez_compressed(
                npz_path,
                names=np.array(metrics["names"]),
                positions_mm=metrics["positions_mm"],
                is_cube=metrics["is_cube"],
                gate_center_s=metrics["gate_center_s"],
                n_win_samples=metrics["n_win_samples"],
                td_metric=metrics["td_metric"],
                nb_metric=metrics["nb_metric"],
                d_to_target_mm=d_to_target,
                target_mm=target_mm,
                aperture_mm=aperture_mm,
                pulse_dur=pulse_dur,
                f0_hz=f0,
                dt=dt_,
            )
            print(f"  saved per-voxel arrays: {npz_path}")
        except Exception as e:
            print(f"  WARN: could not save npz: {e}")

    # Write final JSON summary
    summary_path = OUT_DIR / "gu008_narrowband_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[narrowband_dft_metric_gu008] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
