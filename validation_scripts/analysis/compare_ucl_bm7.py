#!/usr/bin/env python3
"""Load UCL Benchmark 7 data and compare k-Wave reference against other solvers.

Also loads the skull mask and prepares it for running our pipeline's k-Wave
simulation against the same geometry for direct comparison.

UCL BM7: Truncated skull, V1 target, single-element 64mm focused bowl at 500 kHz.
Comparison grid: 0.5mm spacing, 241x141x141 (x=[0,120], y=[-35,35], z=[-35,35]).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


DATA_DIR = Path.home() / "Data/openlifu-validation/ucl-benchmarks/data"

SOLVERS = [
    "KWAVE", "STRIDE", "JWAVE", "FULLWAVE", "SIM4LIFE", "FOCUS",
    "BABELVISCOFDTD", "GMFDTD", "HAS", "MSOUND", "OPTIMUS", "SALVUS",
]

BM7_GRID = {
    "nx": 241, "ny": 141, "nz": 141,
    "dx_mm": 0.5,
    "x_range": (0, 120),
    "y_range": (-35, 35),
    "z_range": (-35, 35),
}


def load_mat_pressure(path: Path) -> np.ndarray | None:
    """Load p_amp from .mat file (v5 or HDF5). Returns array in (x, y, z) order."""
    if not path.exists():
        return None
    try:
        with h5py.File(str(path), "r") as f:
            if "p_amp" not in f:
                return None
            p = np.array(f["p_amp"], dtype=np.float32)
        # HDF5/MATLAB axis transposition: stored as (z, y, x), need (x, y, z)
        return np.transpose(p, (2, 1, 0))
    except OSError:
        # MATLAB v5 format
        mat = sio.loadmat(str(path))
        if "p_amp" not in mat:
            return None
        return np.array(mat["p_amp"], dtype=np.float32)


def load_skull_mask(bm: int, dx_mm: float = 1.0) -> dict:
    """Load pre-rasterized skull mask for a benchmark."""
    dx_str = f"{dx_mm:g}"
    fname = f"skull_mask_bm{bm}_dx_{dx_str}mm.mat"
    path = DATA_DIR / "SKULL-MAPS" / fname
    if not path.exists():
        raise FileNotFoundError(f"Skull mask not found: {path}")

    with h5py.File(str(path), "r") as f:
        keys = list(f.keys())
        result = {}
        for key in ["skull_mask", "brain_mask", "xi", "yi", "zi", "dx"]:
            if key in f:
                arr = np.array(f[key])
                if key in ("skull_mask", "brain_mask"):
                    arr = np.transpose(arr, (2, 1, 0))  # (z,y,x) -> (x,y,z)
                result[key] = arr
        result["all_keys"] = keys
    return result


def compute_metrics(p_ref: np.ndarray, p_test: np.ndarray,
                    brain_mask: np.ndarray | None = None) -> dict:
    """Compute comparison metrics between reference and test pressure fields."""
    if brain_mask is not None:
        mask = brain_mask > 0
    else:
        mask = p_ref > 0.01 * p_ref.max()

    p_r = p_ref[mask]
    p_t = p_test[mask]

    l2_err = np.linalg.norm(p_r - p_t) / np.linalg.norm(p_r) * 100
    linf_err = np.max(np.abs(p_r - p_t)) / np.max(np.abs(p_r)) * 100

    ref_peak = float(p_ref.max())
    test_peak = float(p_test.max())
    peak_diff_pct = (test_peak - ref_peak) / ref_peak * 100

    ref_peak_idx = np.unravel_index(np.argmax(p_ref), p_ref.shape)
    test_peak_idx = np.unravel_index(np.argmax(p_test), p_test.shape)
    peak_dist_mm = np.linalg.norm(
        np.array(test_peak_idx, dtype=float) - np.array(ref_peak_idx, dtype=float)
    ) * BM7_GRID["dx_mm"]

    return {
        "l2_error_pct": l2_err,
        "linf_error_pct": linf_err,
        "ref_peak_Pa": ref_peak,
        "test_peak_Pa": test_peak,
        "peak_diff_pct": peak_diff_pct,
        "peak_distance_mm": peak_dist_mm,
        "ref_peak_idx": ref_peak_idx,
        "test_peak_idx": test_peak_idx,
    }


def main():
    ap = argparse.ArgumentParser(description="UCL Benchmark 7 comparison")
    ap.add_argument("--benchmark", type=int, default=7, choices=[7, 8, 9])
    ap.add_argument("--source", type=int, default=1, choices=[1, 2],
                    help="1=focused bowl, 2=plane piston")
    ap.add_argument("--reference", default="KWAVE", help="Reference solver")
    ap.add_argument("--skull-dx", type=float, default=1.0,
                    help="Skull mask resolution in mm")
    args = ap.parse_args()

    bm = args.benchmark
    sc = args.source
    ref_name = args.reference

    print("=" * 72)
    print(f"  UCL Benchmark {bm} | Source {sc} | Reference: {ref_name}")
    print("=" * 72)

    # Load reference
    ref_file = f"PH1-BM{bm}-SC{sc}_{ref_name}.mat"
    ref_path = DATA_DIR / ref_name / ref_file
    print(f"\nLoading reference: {ref_path.name}")
    p_ref = load_mat_pressure(ref_path)
    if p_ref is None:
        print(f"  ERROR: Could not load {ref_path}")
        return
    print(f"  Shape: {p_ref.shape}, peak: {p_ref.max():.1f} Pa")

    # Load skull mask
    print(f"\nLoading skull mask (BM{bm}, dx={args.skull_dx}mm)")
    try:
        skull_data = load_skull_mask(bm, args.skull_dx)
        skull_mask = skull_data.get("skull_mask")
        brain_mask = skull_data.get("brain_mask")
        if skull_mask is not None:
            print(f"  Skull mask shape: {skull_mask.shape}, "
                  f"skull voxels: {(skull_mask > 0).sum():,}")
        if brain_mask is not None:
            print(f"  Brain mask shape: {brain_mask.shape}, "
                  f"brain voxels: {(brain_mask > 0).sum():,}")
        print(f"  Available keys: {skull_data['all_keys']}")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        brain_mask = None

    # Load k-Wave simulation settings
    print(f"\nExtracting simulation parameters from {ref_name}...")
    with h5py.File(str(ref_path), "r") as f:
        if "general_settings" in f:
            gs = f["general_settings"]
            for key in sorted(gs.keys()):
                val = np.array(gs[key]).flat[0]
                print(f"  {key}: {val}")
        if "simulation_settings" in f:
            ss = f["simulation_settings"]
            print("\n  Simulation settings:")
            for key in sorted(ss.keys()):
                val = np.array(ss[key]).flat[0]
                print(f"  {key}: {val}")

    # Compare all solvers against reference
    print(f"\n{'=' * 72}")
    print(f"  SOLVER COMPARISON (vs {ref_name})")
    print(f"{'=' * 72}")

    header = (f"{'Solver':<20} {'Peak (Pa)':>10} {'Peak %':>8} "
              f"{'L2 %':>8} {'Linf %':>8} {'Dist (mm)':>10}")
    print(header)
    print("-" * len(header))

    # Reference self-comparison
    print(f"{ref_name:<20} {p_ref.max():>10.1f} {'0.0':>8} "
          f"{'0.0':>8} {'0.0':>8} {'0.000':>10}")

    for solver in SOLVERS:
        if solver == ref_name:
            continue
        test_file = f"PH1-BM{bm}-SC{sc}_{solver}.mat"
        test_path = DATA_DIR / solver / test_file
        p_test = load_mat_pressure(test_path)
        if p_test is None:
            print(f"{solver:<20} {'N/A':>10}")
            continue

        if p_test.shape != p_ref.shape:
            print(f"{solver:<20} shape mismatch: {p_test.shape} vs {p_ref.shape}")
            continue

        m = compute_metrics(p_ref, p_test, None)
        print(f"{solver:<20} {m['test_peak_Pa']:>10.1f} "
              f"{m['peak_diff_pct']:>+8.1f} "
              f"{m['l2_error_pct']:>8.1f} "
              f"{m['linf_error_pct']:>8.1f} "
              f"{m['peak_distance_mm']:>10.3f}")

    # Summary for our pipeline
    print(f"\n{'=' * 72}")
    print(f"  SKULL GEOMETRY SUMMARY (for pipeline integration)")
    print(f"{'=' * 72}")
    if skull_mask is not None:
        print(f"  Skull mask resolution: {args.skull_dx} mm")
        print(f"  Skull mask shape: {skull_mask.shape}")
        skull_extent = np.argwhere(skull_mask > 0)
        print(f"  Skull extent (voxels): "
              f"x=[{skull_extent[:,0].min()}, {skull_extent[:,0].max()}], "
              f"y=[{skull_extent[:,1].min()}, {skull_extent[:,1].max()}], "
              f"z=[{skull_extent[:,2].min()}, {skull_extent[:,2].max()}]")
        print(f"  Skull extent (mm): "
              f"x=[{skull_extent[:,0].min()*args.skull_dx:.1f}, {skull_extent[:,0].max()*args.skull_dx:.1f}], "
              f"y=[{(skull_extent[:,1].min()-skull_mask.shape[1]//2)*args.skull_dx:.1f}, "
              f"{(skull_extent[:,1].max()-skull_mask.shape[1]//2)*args.skull_dx:.1f}], "
              f"z=[{(skull_extent[:,2].min()-skull_mask.shape[2]//2)*args.skull_dx:.1f}, "
              f"{(skull_extent[:,2].max()-skull_mask.shape[2]//2)*args.skull_dx:.1f}]")

    # Focal analysis
    peak_idx = np.unravel_index(np.argmax(p_ref), p_ref.shape)
    peak_mm = np.array(peak_idx) * BM7_GRID["dx_mm"]
    peak_mm[1] += BM7_GRID["y_range"][0]
    peak_mm[2] += BM7_GRID["z_range"][0]
    print(f"\n  k-Wave focal peak: {p_ref.max():.1f} Pa at voxel {peak_idx}")
    print(f"  k-Wave focal position: ({peak_mm[0]:.1f}, {peak_mm[1]:.1f}, {peak_mm[2]:.1f}) mm")

    # Axial profile through focus
    ax_profile = p_ref[:, peak_idx[1], peak_idx[2]]
    fwhm_mask = ax_profile >= 0.5 * ax_profile.max()
    fwhm_indices = np.where(fwhm_mask)[0]
    if len(fwhm_indices) > 1:
        fwhm_mm = (fwhm_indices[-1] - fwhm_indices[0]) * BM7_GRID["dx_mm"]
        print(f"  Axial FWHM: {fwhm_mm:.1f} mm")

    # Lateral profile through focus
    lat_y = p_ref[peak_idx[0], :, peak_idx[2]]
    fwhm_mask_y = lat_y >= 0.5 * lat_y.max()
    fwhm_y = np.where(fwhm_mask_y)[0]
    if len(fwhm_y) > 1:
        print(f"  Lateral FWHM (y): {(fwhm_y[-1] - fwhm_y[0]) * BM7_GRID['dx_mm']:.1f} mm")

    lat_z = p_ref[peak_idx[0], peak_idx[1], :]
    fwhm_mask_z = lat_z >= 0.5 * lat_z.max()
    fwhm_z = np.where(fwhm_mask_z)[0]
    if len(fwhm_z) > 1:
        print(f"  Lateral FWHM (z): {(fwhm_z[-1] - fwhm_z[0]) * BM7_GRID['dx_mm']:.1f} mm")


if __name__ == "__main__":
    main()
