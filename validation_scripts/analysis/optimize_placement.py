#!/usr/bin/env python3
"""Placement optimization: find the best array orientation for a given subject.

Dense spherical search over orientations, ray-casting all 64 elements at each
candidate to score skull avoidance. Optionally prints the stonkbot command to
run the CW simulation at the best orientation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
RAY_STEP_MM = 0.25
SKULL_LABEL = 5
BRAIN_LABELS = {2, 3, 4}


def create_hemispherical_array(
    n_elements=64, radius_mm=90.0, aperture_mm=80.0,
    freq_hz=500e3, element_size_mm=5.0,
) -> Transducer:
    half_aperture = aperture_mm / 2.0
    theta_max = np.arcsin(half_aperture / radius_mm)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    elements = []
    for i in range(n_elements):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        x = radius_mm * np.sin(theta) * np.cos(phi)
        y = radius_mm * np.sin(theta) * np.sin(phi)
        z = radius_mm * np.cos(theta)
        nx, ny, nz = -x, -y, -z
        az = np.arctan2(nx, nz)
        el = -np.arctan2(ny, np.sqrt(nx**2 + nz**2))
        elements.append(Element(
            index=i + 1, pin=i + 1,
            position=np.array([x, y, z]),
            orientation=np.array([az, el, 0.0]),
            size=np.array([element_size_mm, element_size_mm]),
            units="mm",
        ))
    return Transducer(
        id="hemi64", name=f"Hemispherical {n_elements}-element array",
        elements=elements, frequency=freq_hz, units="mm",
    )


def load_label_nifti(nifti_path: Path):
    """Load label NIfTI. Returns (data_f32, origins, spacings, coord_arrays)."""
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj).astype(np.float32)
    affine = img.affine
    origins = np.array([float(affine[ax, 3]) for ax in range(3)])
    spacings = np.array([float(affine[ax, ax]) for ax in range(3)])
    coord_arrays = [origins[ax] + np.arange(data.shape[ax]) * spacings[ax] for ax in range(3)]
    return data, origins, spacings, coord_arrays


def find_brain_center(lab_data: np.ndarray, coord_arrays: list[np.ndarray]) -> np.ndarray:
    lab_int = np.round(lab_data).astype(np.int16)
    brain_mask = np.zeros(lab_int.shape, dtype=bool)
    for bl in BRAIN_LABELS:
        brain_mask |= (lab_int == bl)
    if brain_mask.sum() == 0:
        raise ValueError("No brain voxels found in label volume")
    brain_indices = np.argwhere(brain_mask)
    return np.array([
        float(np.mean(coord_arrays[ax][brain_indices[:, ax]]))
        for ax in range(3)
    ])


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return (n, 2) array of (theta, phi) uniformly on the sphere. Phi wrapped to [0, 2pi)."""
    golden = np.pi * (3 - np.sqrt(5))
    i = np.arange(n, dtype=np.float64)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = (golden * i) % (2 * np.pi)
    return np.column_stack([theta, phi])


def rotation_from_zaxis_to_direction(d: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping [0,0,1] to unit vector d (Rodrigues)."""
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z, d)
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    c = dot
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def get_local_positions(arr: Transducer) -> np.ndarray:
    positions = np.zeros((len(arr.elements), 3))
    for i, el in enumerate(arr.elements):
        positions[i] = el.position
    return positions


def evaluate_orientation_batch(
    theta: float,
    phi: float,
    local_positions: np.ndarray,
    brain_center: np.ndarray,
    lab_data_f32: np.ndarray,
    origins: np.ndarray,
    spacings: np.ndarray,
    step_mm: float = 0.25,
) -> dict:
    """Score one (theta, phi) orientation. All 64 rays are batched into one map_coordinates call."""
    d = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    R = rotation_from_zaxis_to_direction(d)
    world_positions = (R @ local_positions.T).T + brain_center

    n_elem = local_positions.shape[0]
    directions = brain_center - world_positions  # (n_elem, 3)
    lengths = np.linalg.norm(directions, axis=1)  # (n_elem,)

    # Build sample points for all rays at once
    all_voxel_coords = []
    ray_boundaries = []  # (start_idx, end_idx) for each element in the flat array
    offset = 0
    for i in range(n_elem):
        length = lengths[i]
        if length < 1e-6:
            ray_boundaries.append((offset, offset))
            continue
        n_steps = int(np.ceil(length / step_mm)) + 1
        t_values = np.linspace(0, 1, n_steps)
        points_mm = world_positions[i] + t_values[:, np.newaxis] * directions[i]
        voxel_coords = ((points_mm - origins) / spacings).T  # (3, n_steps)
        all_voxel_coords.append(voxel_coords)
        ray_boundaries.append((offset, offset + n_steps))
        offset += n_steps

    if not all_voxel_coords:
        return {
            "theta": theta, "phi": phi,
            "theta_deg": float(np.degrees(theta)),
            "phi_deg": float(np.degrees(phi)),
            "direction": d.tolist(),
            "n_clear": 0, "n_through_skull": n_elem,
            "mean_path_mm": 0.0, "max_path_mm": 0.0, "total_path_mm": 0.0,
        }

    # Single map_coordinates call for all rays
    all_coords = np.concatenate(all_voxel_coords, axis=1)  # (3, total_samples)
    all_labels = map_coordinates(lab_data_f32, all_coords, order=0, mode="constant", cval=0.0)
    all_labels = np.round(all_labels).astype(np.int16)

    n_clear = 0
    n_through_skull = 0
    paths = []
    for i in range(n_elem):
        start, end = ray_boundaries[i]
        if start == end:
            n_through_skull += 1
            paths.append(0.0)
            continue
        segment = all_labels[start:end]
        skull_count = int((segment == SKULL_LABEL).sum())
        if skull_count > 0:
            n_through_skull += 1
            paths.append(skull_count * step_mm)
        else:
            n_clear += 1

    mean_path = float(np.mean(paths)) if paths else 0.0
    max_path = float(np.max(paths)) if paths else 0.0
    total_path = float(np.sum(paths)) if paths else 0.0

    return {
        "theta": theta,
        "phi": phi,
        "theta_deg": float(np.degrees(theta)),
        "phi_deg": float(np.degrees(phi)) % 360.0,
        "direction": d.tolist(),
        "n_clear": n_clear,
        "n_through_skull": n_through_skull,
        "mean_path_mm": mean_path,
        "max_path_mm": max_path,
        "total_path_mm": total_path,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Find optimal array orientation for a subject via ray-cast search."
    )
    ap.add_argument("--subject", required=True, help="Subject ID (e.g. GU008)")
    ap.add_argument("--n-search", type=int, default=200,
                    help="Number of search orientations (default: 200)")
    ap.add_argument("--run-sim", action="store_true",
                    help="Print the stonkbot sim command for the best orientation")
    ap.add_argument("--results-dir", default=None,
                    help="Output directory (default: ~/Data/openlifu-validation/results)")
    ap.add_argument("--label-path", default=None, help="Override label path")
    args = ap.parse_args()

    subj = args.subject
    results_dir = Path(args.results_dir) if args.results_dir else (
        Path.home() / "Data/openlifu-validation/results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    label_path = Path(args.label_path) if args.label_path else (
        Path.home() / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"
    )

    print("=" * 72)
    print(f"  Placement Optimization | subject={subj} | n_search={args.n_search}")
    print("=" * 72)

    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}")
        sys.exit(1)

    t0 = time.time()
    print(f"Loading labels: {label_path}")
    lab_data_f32, origins, spacings, coord_arrays = load_label_nifti(label_path)
    print(f"  Label shape: {tuple(int(s) for s in lab_data_f32.shape)}, spacing: {spacings}")

    brain_center = find_brain_center(lab_data_f32, coord_arrays)
    print(f"  Brain center: ({brain_center[0]:.1f}, {brain_center[1]:.1f}, {brain_center[2]:.1f}) mm")

    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )
    local_positions = get_local_positions(arr_local)
    print(f"  Array: {N_ELEMENTS} elements, radius={RADIUS_MM}mm, aperture={APERTURE_MM}mm")

    orientations = fibonacci_sphere(args.n_search)
    print(f"\nSearching {args.n_search} orientations...")

    results = []
    for idx in range(len(orientations)):
        theta, phi = orientations[idx]
        r = evaluate_orientation_batch(
            theta, phi, local_positions, brain_center,
            lab_data_f32, origins, spacings, step_mm=RAY_STEP_MM,
        )
        r["rank_idx"] = idx
        results.append(r)
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{args.n_search}] elapsed={elapsed:.1f}s "
                  f"  current: n_clear={r['n_clear']}, mean_path={r['mean_path_mm']:.1f}mm")

    elapsed = time.time() - t0
    print(f"\nSearch complete in {elapsed:.1f}s")

    # Rank: primary = max n_clear, secondary = min mean_path_mm
    results.sort(key=lambda r: (-r["n_clear"], r["mean_path_mm"]))

    print(f"\n{'='*72}")
    print(f"  TOP 10 ORIENTATIONS")
    print(f"{'='*72}")
    header = (
        f"{'Rank':>4}  {'theta':>7} {'phi':>7}  "
        f"{'n_clear':>7} {'n_skull':>7}  "
        f"{'mean_path':>9} {'max_path':>9} {'total_path':>10}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results[:10]):
        print(
            f"{i+1:>4}  "
            f"{r['theta_deg']:>7.1f} {r['phi_deg']:>7.1f}  "
            f"{r['n_clear']:>7} {r['n_through_skull']:>7}  "
            f"{r['mean_path_mm']:>9.2f} {r['max_path_mm']:>9.2f} {r['total_path_mm']:>10.1f}"
        )

    best = results[0]
    print(f"\nBest orientation:")
    print(f"  theta={best['theta_deg']:.2f} deg, phi={best['phi_deg']:.2f} deg")
    print(f"  direction=({best['direction'][0]:.3f}, {best['direction'][1]:.3f}, {best['direction'][2]:.3f})")
    print(f"  n_clear={best['n_clear']}/{N_ELEMENTS}, "
          f"mean_skull_path={best['mean_path_mm']:.2f}mm, "
          f"max_skull_path={best['max_path_mm']:.2f}mm")
    print(f"BEST: theta={best['theta_deg']:.2f} phi={best['phi_deg']:.2f} n_clear={best['n_clear']}")

    output = {
        "subject": subj,
        "n_search": args.n_search,
        "brain_center_mm": brain_center.tolist(),
        "n_elements": N_ELEMENTS,
        "radius_mm": RADIUS_MM,
        "aperture_mm": APERTURE_MM,
        "ray_step_mm": RAY_STEP_MM,
        "elapsed_s": elapsed,
        "best": best,
        "all_results": results,
    }
    json_path = results_dir / f"{subj}_placement_optimization.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {json_path}")

    if args.run_sim:
        print(f"\n{'='*72}")
        print(f"  SIMULATION COMMAND")
        print(f"{'='*72}")
        cmd = (
            f"python scripts/run_gladys_nnunet_subject.py "
            f"--subject {subj} "
            f"--orient-theta {best['theta_deg']:.2f} "
            f"--orient-phi {best['phi_deg']:.2f} "
            f"--output-tag optimized_"
        )
        print(f"\n  {cmd}\n")


if __name__ == "__main__":
    main()
