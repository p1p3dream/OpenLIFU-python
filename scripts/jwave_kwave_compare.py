#!/usr/bin/env python3
"""Compare j-Wave (Helmholtz) vs k-Wave (FDTD) for two-class bone skulls.

Runs the full 64-element GLADYS hemispherical array focused at the brain
target, three ways:
  1. Water baseline (no skull, geometric delays)
  2. Skull with geometric delays only
  3. Skull with Helmholtz phase correction (reciprocity)

The phase correction works by reciprocity: place a point source at the
target, solve Helmholtz through the skull medium, read the complex pressure
at each element position, then conjugate those phases for the corrected
64-element transmit solve. This is the frequency-domain equivalent of
k-Wave's SimulationCorrected delay method.

Usage (in jwave-absorption venv on stonkbot):
    source ~/envs/jwave-absorption/bin/activate
    python3 -u scripts/jwave_kwave_compare.py --subject NC024
    python3 -u scripts/jwave_kwave_compare.py --subject NC024 --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import nibabel as nib
import numpy as np


FREQ_HZ = 500_000.0
OMEGA = 2 * np.pi * FREQ_HZ
C_WATER = 1500.0

N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0

PROPS = {
    0: {"name": "background",     "c": 1500.0, "rho": 1000.0, "alpha": 0.0},
    1: {"name": "skin",           "c": 1610.0, "rho": 1090.0, "alpha": 2.5},
    2: {"name": "brain",          "c": 1560.0, "rho": 1040.0, "alpha": 1.2},
    3: {"name": "csf",            "c": 1500.0, "rho": 1007.0, "alpha": 0.002},
    4: {"name": "air",            "c": 1500.0, "rho": 1000.0, "alpha": 0.0},
    5: {"name": "cortical_bone",  "c": 2800.0, "rho": 1850.0, "alpha": 4.0},
    6: {"name": "eye",            "c": 1534.0, "rho": 1006.0, "alpha": 0.6},
    7: {"name": "trabecular_bone","c": 2300.0, "rho": 1700.0, "alpha": 8.0},
}

ALPHA_POWER = 0.9

KWAVE_REF = {
    # Outlier subjects (two-class k-Wave reference)
    "GU006":  {"geo_dB": 18.24, "corr_dB": 13.01},
    "GU035":  {"geo_dB": 24.66, "corr_dB": 18.27},
    "NC011":  {"geo_dB": 15.83, "corr_dB": 11.29},
    "NC012":  {"geo_dB": 19.00, "corr_dB": 17.92},
    "NC015":  {"geo_dB": 16.01, "corr_dB": 10.03},
    "NC017":  {"geo_dB": 18.21, "corr_dB": 13.14},
    "NC024":  {"geo_dB": 13.55, "corr_dB": 9.94},
    "NC029":  {"geo_dB": 20.71, "corr_dB": 16.13},
    "NC033":  {"geo_dB": 20.55, "corr_dB": 17.85},
    "NYU005": {"geo_dB": 17.52, "corr_dB": 13.09},
    "NYU008": {"geo_dB": 18.06, "corr_dB": 13.67},
    # Non-outlier subjects (single-class k-Wave reference)
    "NC021":  {"geo_dB": 1.00, "corr_dB": 5.43},
    "GU027":  {"geo_dB": 2.40, "corr_dB": 6.15},
    "NC008":  {"geo_dB": 3.51, "corr_dB": 6.94},
    "NC019":  {"geo_dB": 4.11, "corr_dB": 7.11},
    "GU025":  {"geo_dB": 4.67, "corr_dB": 8.95},
    "NC020":  {"geo_dB": 5.67, "corr_dB": 13.82},
    "NYU004": {"geo_dB": 6.40, "corr_dB": 12.83},
    "NC027":  {"geo_dB": 7.09, "corr_dB": 11.46},
    "GU020":  {"geo_dB": 7.84, "corr_dB": 12.27},
    "NYU002": {"geo_dB": 8.14, "corr_dB": 11.36},
}


def create_element_positions(target_mm, approach_dir=None):
    """Create 64-element hemispherical array positions focused at target.

    Uses the same golden-angle spiral placement as run_gladys_nnunet_subject.py.
    Returns positions in mm (world coordinates).
    """
    half_aperture = APERTURE_MM / 2.0
    theta_max = np.arcsin(half_aperture / RADIUS_MM)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    positions_local = []
    for i in range(N_ELEMENTS):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / N_ELEMENTS
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        x = RADIUS_MM * np.sin(theta) * np.cos(phi)
        y = RADIUS_MM * np.sin(theta) * np.sin(phi)
        z = RADIUS_MM * np.cos(theta)
        positions_local.append([x, y, z])
    positions_local = np.array(positions_local)

    if approach_dir is None:
        approach_dir = np.array([0.0, 0.0, -1.0])
    approach_dir = approach_dir / np.linalg.norm(approach_dir)

    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, approach_dir)
    c = np.dot(z_axis, approach_dir)
    if np.linalg.norm(v) < 1e-10:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)

    positions_world = (R @ positions_local.T).T + target_mm[np.newaxis, :]
    return positions_world


def load_and_downsample(label_path: str, factor: int = 1):
    """Load label NIfTI, optionally downsample by majority vote."""
    img = nib.load(label_path)
    labels = np.asarray(img.dataobj).astype(np.int16)
    spacing = tuple(abs(float(x)) for x in img.header.get_zooms()[:3])
    print(f"  Original shape: {labels.shape}, spacing: {spacing} mm")

    if factor <= 1:
        return labels, spacing

    new_shape = tuple(s // factor for s in labels.shape)
    new_spacing = tuple(s * factor for s in spacing)
    downsampled = np.zeros(new_shape, dtype=np.int16)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                block = labels[
                    i*factor:(i+1)*factor,
                    j*factor:(j+1)*factor,
                    k*factor:(k+1)*factor,
                ]
                vals, counts = np.unique(block, return_counts=True)
                downsampled[i, j, k] = vals[counts.argmax()]

    print(f"  Downsampled: {downsampled.shape}, spacing: {new_spacing} mm")
    return downsampled, new_spacing


def build_medium_arrays(labels, spacing):
    """Build c, rho, alpha arrays from label volume."""
    c = np.full(labels.shape, C_WATER, dtype=np.float32)
    rho = np.full(labels.shape, 1000.0, dtype=np.float32)
    alpha = np.zeros(labels.shape, dtype=np.float32)

    for label_id, props in PROPS.items():
        mask = labels == label_id
        n = int(mask.sum())
        if n > 0:
            c[mask] = props["c"]
            rho[mask] = props["rho"]
            alpha[mask] = props["alpha"]
            print(f"    {label_id} {props['name']:20s}: {n:>10,} vox, "
                  f"c={props['c']:.0f}, alpha={props['alpha']}")

    return c, rho, alpha


def find_brain_target(labels, spacing):
    """Brain centroid as target, approach from superior (negative z in voxel space)."""
    brain_mask = labels == 2
    if brain_mask.sum() == 0:
        brain_mask = labels == 3
    coords = np.argwhere(brain_mask)
    centroid_idx = coords.mean(axis=0).astype(int)
    target_mm = centroid_idx * np.array(spacing)
    print(f"  Target: idx={tuple(centroid_idx)}, mm={target_mm}")
    return target_mm


def _build_medium(c, rho, alpha_arr, spacing):
    """Build j-Wave Medium from numpy arrays."""
    import jax.numpy as jnp
    from jaxdf.discretization import FourierSeries
    from jwave.geometry import Domain, Medium

    N = c.shape
    dx = tuple(s * 1e-3 for s in spacing)
    domain = Domain(N, dx)

    c_field = FourierSeries(jnp.array(c[..., None], dtype=jnp.float32), domain)
    rho_field = FourierSeries(jnp.array(rho[..., None], dtype=jnp.float32), domain)
    alpha_field = FourierSeries(jnp.array(alpha_arr[..., None], dtype=jnp.float32), domain)

    medium = Medium(
        domain=domain,
        sound_speed=c_field,
        density=rho_field,
        attenuation=alpha_field,
        pml_size=10,
        alpha_power=ALPHA_POWER,
    )
    return medium, domain, N


def _solve_helmholtz(medium, domain, source_field_np, label=""):
    """Run a single Helmholtz solve and return the complex pressure field."""
    import jax
    import jax.numpy as jnp
    from jaxdf.discretization import FourierSeries
    from jwave.acoustics.time_harmonic import helmholtz_solver

    source = FourierSeries(jnp.array(source_field_np), domain)

    print(f"    Solving Helmholtz ({label})...")
    t0 = time.time()
    result = helmholtz_solver(medium, OMEGA, source, tol=1e-5, maxiter=500)
    jax.block_until_ready(result.params)
    elapsed = time.time() - t0
    print(f"    Solve time: {elapsed:.1f} s")

    p_complex = np.asarray(result.params).squeeze(-1)
    return p_complex, elapsed


def compute_phase_correction(c, rho, alpha_arr, spacing, target_mm, positions_mm):
    """Helmholtz phase correction via reciprocity.

    Places a point source at the target, solves through the skull medium,
    and reads the complex pressure at each element position. The conjugate
    of those phases corrects for skull-induced aberration.

    Returns complex amplitudes (one per element) to use as source weights.
    """
    N = c.shape
    medium, domain, _ = _build_medium(c, rho, alpha_arr, spacing)

    target_idx = tuple(np.rint(target_mm / np.array(spacing)).astype(int))
    src_field = np.zeros(N + (1,), dtype=np.complex64)
    src_field[target_idx + (0,)] = 1.0 + 0j

    print(f"    Point source at target idx={target_idx}")
    mem_gb = 20 * np.prod(N) * 16 / 1e9
    print(f"    Grid: {N}, est. memory: {mem_gb:.1f} GB")

    p_complex, elapsed = _solve_helmholtz(medium, domain, src_field, label="reciprocity")

    corrections = np.zeros(N_ELEMENTS, dtype=np.complex64)
    valid_mask = np.zeros(N_ELEMENTS, dtype=bool)
    for i in range(N_ELEMENTS):
        idx = tuple(np.rint(positions_mm[i] / np.array(spacing)).astype(int))
        if all(0 <= idx[d] < N[d] for d in range(3)):
            p_at_elem = p_complex[idx]
            if abs(p_at_elem) > 0:
                corrections[i] = np.conj(p_at_elem) / abs(p_at_elem)
                valid_mask[i] = True

    n_valid = int(valid_mask.sum())
    print(f"    Valid corrections: {n_valid}/{N_ELEMENTS}")
    print(f"    Phase correction time: {elapsed:.1f} s")

    if n_valid > 0:
        phases_deg = np.angle(corrections[valid_mask]) * 180 / np.pi
        print(f"    Phase correction range: {phases_deg.min():.1f} to {phases_deg.max():.1f} deg")
        print(f"    Phase correction std: {phases_deg.std():.1f} deg")

    return corrections, elapsed


def run_jwave_helmholtz_64(c, rho, alpha_arr, spacing, target_mm, positions_mm,
                            source_weights=None, label="skull"):
    """Run j-Wave Helmholtz with 64-element focused array.

    source_weights: complex array of per-element weights. If None, all
    elements get amplitude 1.0 (geometric focusing only, which is trivial
    for equidistant hemispheres).
    """
    N = c.shape
    medium, domain, _ = _build_medium(c, rho, alpha_arr, spacing)

    src_field = np.zeros(N + (1,), dtype=np.complex64)
    target_idx = tuple(np.rint(target_mm / np.array(spacing)).astype(int))

    if source_weights is None:
        source_weights = np.ones(N_ELEMENTS, dtype=np.complex64)

    n_placed = 0
    for i in range(N_ELEMENTS):
        idx = tuple(np.rint(positions_mm[i] / np.array(spacing)).astype(int))
        if all(0 <= idx[d] < N[d] for d in range(3)):
            src_field[idx + (0,)] += source_weights[i]
            n_placed += 1

    print(f"    Elements placed on grid: {n_placed}/{N_ELEMENTS}")
    mem_gb = 20 * np.prod(N) * 16 / 1e9
    print(f"    Grid: {N}, est. memory: {mem_gb:.1f} GB")

    p_complex, elapsed = _solve_helmholtz(medium, domain, src_field, label=label)
    p_amp = np.abs(p_complex)

    p_at_target = float(p_amp[target_idx])
    p_max = float(p_amp.max())
    p_max_idx = np.unravel_index(np.argmax(p_amp), p_amp.shape)
    p_max_mm = np.array(p_max_idx) * np.array(spacing)
    focal_err = np.linalg.norm(p_max_mm - target_mm)

    print(f"    p at target: {p_at_target:.6g} Pa")
    print(f"    p_max: {p_max:.6g} Pa at idx={p_max_idx} ({p_max_mm} mm)")
    print(f"    Focal error: {focal_err:.1f} mm")

    return {
        "label": label,
        "time_s": elapsed,
        "p_at_target": p_at_target,
        "p_max": p_max,
        "focal_err_mm": focal_err,
        "p_field": p_amp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-path", default=None)
    ap.add_argument("--subject", default="NC024")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--csv", default=None,
                    help="Path to CSV file; appends one row per subject. "
                         "Creates the file with headers if it does not exist.")
    args = ap.parse_args()

    if args.label_path is None:
        from pathlib import Path
        results_dir = Path.home() / "Data" / "openlifu-validation" / "results"
        args.label_path = str(results_dir / f"{args.subject}_two_class_labels.nii.gz")

    print("=" * 60)
    print("j-Wave 64-Element Array Comparison")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Frequency: {FREQ_HZ/1e3:.0f} kHz, alpha_power: {ALPHA_POWER}")
    print()

    print("[1] Loading labels...")
    labels, spacing = load_and_downsample(args.label_path, factor=args.downsample)

    print("\n[2] Building medium...")
    c_skull, rho_skull, alpha_skull = build_medium_arrays(labels, spacing)

    c_water = np.full_like(c_skull, C_WATER)
    rho_water = np.full_like(rho_skull, 1000.0)
    alpha_water = np.zeros_like(alpha_skull)

    print("\n[3] Setting up 64-element array...")
    target_mm = find_brain_target(labels, spacing)
    approach_dir = np.array([0.0, 0.0, -1.0])
    positions_mm = create_element_positions(target_mm, approach_dir)
    print(f"  Array center: {positions_mm.mean(axis=0)} mm")
    print(f"  Mean distance to target: {np.linalg.norm(positions_mm - target_mm, axis=1).mean():.1f} mm")

    print("\n[4] j-Wave Helmholtz: WATER (no skull, geometric)...")
    water_result = run_jwave_helmholtz_64(
        c_water, rho_water, alpha_water, spacing,
        target_mm, positions_mm, label="water",
    )

    print("\n[5] j-Wave Helmholtz: SKULL (geometric delays)...")
    skull_geo_result = run_jwave_helmholtz_64(
        c_skull, rho_skull, alpha_skull, spacing,
        target_mm, positions_mm, label="skull_geometric",
    )

    print("\n[6] Helmholtz phase correction (reciprocity through skull)...")
    corrections, corr_time = compute_phase_correction(
        c_skull, rho_skull, alpha_skull, spacing, target_mm, positions_mm,
    )

    print("\n[7] j-Wave Helmholtz: SKULL (phase-corrected)...")
    skull_corr_result = run_jwave_helmholtz_64(
        c_skull, rho_skull, alpha_skull, spacing,
        target_mm, positions_mm, source_weights=corrections, label="skull_corrected",
    )

    p_w = water_result["p_at_target"]
    p_geo = skull_geo_result["p_at_target"]
    p_corr = skull_corr_result["p_at_target"]

    atten_geo = 20 * np.log10(p_w / p_geo) if p_w > 0 and p_geo > 0 else float("nan")
    atten_corr = 20 * np.log10(p_w / p_corr) if p_w > 0 and p_corr > 0 else float("nan")
    improvement = atten_geo - atten_corr

    total_time = water_result["time_s"] + skull_geo_result["time_s"] + corr_time + skull_corr_result["time_s"]

    kw = KWAVE_REF.get(args.subject, {})
    kw_geo = kw.get("geo_dB", float("nan"))
    kw_corr = kw.get("corr_dB", float("nan"))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Water:          p_target={p_w:.6g} Pa, time={water_result['time_s']:.1f}s")
    print(f"  Skull (geo):    p_target={p_geo:.6g} Pa, atten={atten_geo:.2f} dB, time={skull_geo_result['time_s']:.1f}s")
    print(f"  Skull (corr):   p_target={p_corr:.6g} Pa, atten={atten_corr:.2f} dB, time={skull_corr_result['time_s']:.1f}s")
    print(f"  Improvement:    {improvement:.2f} dB")
    print(f"  k-Wave ref:     geometric={kw_geo:.2f} dB, corrected={kw_corr:.2f} dB")
    print(f"  Geo delta:      {atten_geo - kw_geo:.2f} dB (jWave - kWave)")
    print(f"  Corr delta:     {atten_corr - kw_corr:.2f} dB (jWave - kWave)")
    print(f"  Total time:     {total_time:.1f} s (incl {corr_time:.1f}s phase correction)")

    if args.csv:
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "subject", "p_water", "p_geo", "p_corr",
                    "jw_geo_dB", "jw_corr_dB", "jw_improvement_dB",
                    "kw_geo_dB", "kw_corr_dB",
                    "delta_geo_dB", "delta_corr_dB",
                    "total_time_s",
                ])
            writer.writerow([
                args.subject,
                f"{p_w:.6g}", f"{p_geo:.6g}", f"{p_corr:.6g}",
                f"{atten_geo:.2f}", f"{atten_corr:.2f}", f"{improvement:.2f}",
                f"{kw_geo:.2f}", f"{kw_corr:.2f}",
                f"{atten_geo - kw_geo:.2f}", f"{atten_corr - kw_corr:.2f}",
                f"{total_time:.1f}",
            ])
        print(f"\n  CSV row appended to: {args.csv}")


if __name__ == "__main__":
    main()
