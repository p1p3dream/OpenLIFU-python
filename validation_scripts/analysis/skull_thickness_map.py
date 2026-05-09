#!/usr/bin/env python3
"""Map skull thickness over the full sphere around brain center for Birnbaum subjects.

For each subject, casts rays in ~650 directions from brain center, measures skull
thickness, then finds the optimal temporal/parietal window for array placement.
Compares naive (max-skull-axis) placement vs optimized placement through thinnest skull.
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xa
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

SUBJECTS = ["GU008", "GU002", "GU010", "NC004"]
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
RAY_STEP_MM = 0.25
SKULL_LABEL = 5
BRAIN_LABELS = {2, 3, 4}
RAY_LENGTH_MM = 150.0
THETA_STEP_DEG = 10
PHI_STEP_DEG = 10
ARRAY_SEARCH_STEP_DEG = 20  # coarser grid for full-sphere array evaluation


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


def load_label_nifti(nifti_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load nifti labels, return (data, origins, spacings)."""
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj).astype(np.int16)
    affine = img.affine
    origins = np.array([float(affine[ax, 3]) for ax in range(3)])
    spacings = np.array([float(affine[ax, ax]) for ax in range(3)])
    coord_arrays = []
    for ax in range(3):
        coord_arrays.append(origins[ax] + np.arange(data.shape[ax]) * spacings[ax])
    return data, origins, spacings, coord_arrays


def direction_from_spherical(theta, phi):
    """Convert spherical (theta=polar from +z, phi=azimuthal from +x) to unit vector."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def skull_thickness_along_ray(origin_mm, direction, lab_data, origins, spacings,
                              ray_length=RAY_LENGTH_MM, step_mm=RAY_STEP_MM):
    """Cast a ray from origin_mm in given direction, count skull voxels.

    Returns (skull_thickness_mm, distance_to_first_skull_mm, n_skull_steps).
    """
    n_steps = int(np.ceil(ray_length / step_mm))
    t_values = np.linspace(0, ray_length, n_steps + 1)

    points_mm = origin_mm[np.newaxis, :] + t_values[:, np.newaxis] * direction[np.newaxis, :]
    voxel_coords = np.zeros((3, len(t_values)))
    for ax in range(3):
        voxel_coords[ax] = (points_mm[:, ax] - origins[ax]) / spacings[ax]

    label_samples = map_coordinates(
        lab_data.astype(np.float32), voxel_coords, order=0, mode="constant", cval=0.0
    )
    label_samples = np.round(label_samples).astype(int)

    skull_mask = label_samples == SKULL_LABEL
    n_skull = int(skull_mask.sum())
    skull_thickness_mm = n_skull * step_mm

    dist_to_first = float("inf")
    if n_skull > 0:
        first_idx = np.argmax(skull_mask)
        dist_to_first = t_values[first_idx]

    return skull_thickness_mm, dist_to_first, n_skull


def rotation_matrix_from_direction(direction):
    """Build a 3x3 rotation matrix that maps +z to the given direction.

    The array's default axis is +z. This rotation aligns it with the approach direction.
    Uses Rodrigues' formula when direction != +z or -z.
    """
    d = direction / np.linalg.norm(direction)
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

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def pose_array_in_direction(target_mm, approach_direction):
    """Create and pose the 64-element array aimed from approach_direction toward target.

    The array center sits at target_mm + approach_direction * RADIUS_MM (so the focus
    is at brain center). Each element is rotated into world coordinates.

    Returns array of element positions (N, 3).
    """
    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )

    R = rotation_matrix_from_direction(approach_direction)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = target_mm

    arr = deepcopy(arr_local)
    positions = np.zeros((N_ELEMENTS, 3))
    for i, el in enumerate(arr.elements):
        world_pos = el.get_position(units="mm", matrix=transform)
        positions[i] = world_pos

    return positions


def evaluate_array_placement(target_mm, approach_direction, lab_data, origins, spacings):
    """Place array in given direction, ray-cast each element to brain center.

    Returns dict with per-element skull paths and summary stats.
    """
    positions = pose_array_in_direction(target_mm, approach_direction)

    n_clear = 0
    n_through_skull = 0
    skull_paths = []
    element_details = []

    for i in range(N_ELEMENTS):
        elem_pos = positions[i]
        direction = target_mm - elem_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        d = direction / dist

        n_steps = int(np.ceil(dist / RAY_STEP_MM))
        t_values = np.linspace(0, 1, n_steps + 1)
        points_mm = elem_pos[np.newaxis, :] + t_values[:, np.newaxis] * direction[np.newaxis, :]

        voxel_coords = np.zeros((3, len(t_values)))
        for ax in range(3):
            voxel_coords[ax] = (points_mm[:, ax] - origins[ax]) / spacings[ax]

        label_samples = map_coordinates(
            lab_data.astype(np.float32), voxel_coords, order=0, mode="constant", cval=0.0
        )
        label_samples = np.round(label_samples).astype(int)
        skull_mask = label_samples == SKULL_LABEL
        skull_path_mm = float(skull_mask.sum()) * RAY_STEP_MM

        if skull_path_mm < 0.01:
            n_clear += 1
        else:
            n_through_skull += 1

        skull_paths.append(skull_path_mm)
        element_details.append({
            "element": i,
            "skull_path_mm": skull_path_mm,
            "ray_length_mm": float(dist),
        })

    skull_paths = np.array(skull_paths)
    through_skull_paths = skull_paths[skull_paths >= 0.01]

    return {
        "n_clear_path": n_clear,
        "n_through_skull": n_through_skull,
        "mean_skull_path_mm": float(np.mean(skull_paths)),
        "total_skull_path_mm": float(np.sum(skull_paths)),
        "mean_skull_path_through_skull_mm": float(np.mean(through_skull_paths)) if len(through_skull_paths) > 0 else 0.0,
        "max_skull_path_mm": float(np.max(skull_paths)),
        "element_details": element_details,
    }


def process_subject(subj: str):
    print(f"\n{'='*80}")
    print(f"  {subj}: Skull Thickness Mapping")
    print(f"{'='*80}")

    mri_path = (
        Path.home() / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data"
        / "Anonymized_Subjects" / "T1-Weighted MRI" / f"{subj}_deface.nii"
    )
    label_path = (
        Path.home() / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"
    )

    if not mri_path.exists() or not label_path.exists():
        print(f"  ERROR: missing data files")
        return None

    lab_data, origins, spacings, coord_arrays = load_label_nifti(label_path)
    print(f"  Label shape: {lab_data.shape}, spacing: {spacings} mm")

    # Brain center
    brain_mask = np.zeros(lab_data.shape, dtype=bool)
    for bl in BRAIN_LABELS:
        brain_mask |= (lab_data == bl)
    if brain_mask.sum() == 0:
        print("  ERROR: no brain voxels")
        return None
    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[ax][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"  Brain center: ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    # Naive approach axis (max skull distance from brain center)
    skull_mask_3d = lab_data == SKULL_LABEL
    skull_indices = np.argwhere(skull_mask_3d)
    skull_mm_coords = np.array([
        coord_arrays[ax][skull_indices[:, ax]] for ax in range(3)
    ])
    max_skull_per_axis = np.array([
        float(skull_mm_coords[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_per_axis))
    naive_direction = np.zeros(3)
    naive_direction[approach_axis] = 1.0
    print(f"  Naive approach axis: {approach_axis} (direction: {naive_direction})")

    # Build spherical direction grid
    thetas = np.arange(0, 181, THETA_STEP_DEG) * np.pi / 180
    phis = np.arange(0, 360, PHI_STEP_DEG) * np.pi / 180
    n_dirs = len(thetas) * len(phis)
    print(f"  Scanning {n_dirs} directions ({len(thetas)} theta x {len(phis)} phi)...")

    t0 = time.time()
    thickness_map = []
    lab_float = lab_data.astype(np.float32)

    for ti, theta in enumerate(thetas):
        for phi in phis:
            d = direction_from_spherical(theta, phi)
            skull_thick, dist_first, n_skull = skull_thickness_along_ray(
                target_mm, d, lab_float, origins, spacings,
            )
            thickness_map.append({
                "theta_deg": float(np.degrees(theta)),
                "phi_deg": float(np.degrees(phi)),
                "theta_rad": float(theta),
                "phi_rad": float(phi),
                "skull_thickness_mm": skull_thick,
                "dist_to_first_skull_mm": dist_first if dist_first != float("inf") else -1,
                "direction": d.tolist(),
            })

    elapsed = time.time() - t0
    print(f"  Ray-casting done in {elapsed:.1f}s")

    # Sort by skull thickness
    thickness_map.sort(key=lambda r: r["skull_thickness_mm"])

    # Top-10 thinnest
    print(f"\n  Top 10 thinnest skull directions:")
    print(f"    {'Rank':>4} {'Theta':>7} {'Phi':>7} {'Thickness':>11} {'Dist1st':>9}")
    for i, entry in enumerate(thickness_map[:10]):
        print(f"    {i+1:>4} {entry['theta_deg']:>6.1f}° {entry['phi_deg']:>6.1f}° "
              f"{entry['skull_thickness_mm']:>10.2f}mm {entry['dist_to_first_skull_mm']:>8.1f}mm")

    # Top-10 thickest (for reference)
    print(f"\n  Top 10 thickest skull directions:")
    print(f"    {'Rank':>4} {'Theta':>7} {'Phi':>7} {'Thickness':>11}")
    for i, entry in enumerate(thickness_map[-10:][::-1]):
        print(f"    {i+1:>4} {entry['theta_deg']:>6.1f}° {entry['phi_deg']:>6.1f}° "
              f"{entry['skull_thickness_mm']:>10.2f}mm")

    # Stats
    all_thick = np.array([r["skull_thickness_mm"] for r in thickness_map])
    print(f"\n  Skull thickness stats across all directions:")
    print(f"    Mean: {np.mean(all_thick):.2f} mm, Std: {np.std(all_thick):.2f} mm")
    print(f"    Min: {np.min(all_thick):.2f} mm, Max: {np.max(all_thick):.2f} mm")
    print(f"    Median: {np.median(all_thick):.2f} mm")

    # Evaluate array placements: full-sphere search
    search_thetas = np.arange(10, 171, ARRAY_SEARCH_STEP_DEG) * np.pi / 180
    search_phis = np.arange(0, 360, ARRAY_SEARCH_STEP_DEG) * np.pi / 180
    n_search = len(search_thetas) * len(search_phis)
    print(f"\n  Evaluating array placement at {n_search} directions "
          f"({len(search_thetas)} theta x {len(search_phis)} phi, {ARRAY_SEARCH_STEP_DEG}° steps)...")

    # Naive placement first
    print(f"  [naive] direction = axis {approach_axis}")
    naive_result = evaluate_array_placement(target_mm, naive_direction, lab_float, origins, spacings)
    print(f"    Clear path: {naive_result['n_clear_path']}/64, "
          f"Through skull: {naive_result['n_through_skull']}/64, "
          f"Mean skull path: {naive_result['mean_skull_path_mm']:.2f} mm, "
          f"Total: {naive_result['total_skull_path_mm']:.1f} mm")

    # Full-sphere array search
    t1 = time.time()
    all_placements = []
    for theta in search_thetas:
        for phi in search_phis:
            d = direction_from_spherical(theta, phi)
            result = evaluate_array_placement(target_mm, d, lab_float, origins, spacings)
            result["theta_deg"] = float(np.degrees(theta))
            result["phi_deg"] = float(np.degrees(phi))
            result["direction"] = d.tolist()
            # Find corresponding single-ray thickness
            theta_deg = float(np.degrees(theta))
            phi_deg = float(np.degrees(phi))
            matching = [e for e in thickness_map
                        if abs(e["theta_deg"] - theta_deg) < 1 and abs(e["phi_deg"] - phi_deg) < 1]
            result["ray_skull_thickness_mm"] = matching[0]["skull_thickness_mm"] if matching else -1
            all_placements.append(result)

    elapsed_search = time.time() - t1
    print(f"  Array search done in {elapsed_search:.1f}s")

    # Sort by (most clear elements, then lowest total skull path)
    all_placements.sort(key=lambda r: (-r["n_clear_path"], r["total_skull_path_mm"]))

    # Show top 10
    print(f"\n  Top 10 array placements (by clear elements, then total skull path):")
    print(f"    {'Rank':>4} {'Theta':>7} {'Phi':>7} {'Clear':>5} {'MeanPath':>9} "
          f"{'TotalPath':>10} {'MaxPath':>8} {'RaySkull':>9}")
    for i, p in enumerate(all_placements[:10]):
        print(f"    {i+1:>4} {p['theta_deg']:>6.0f}° {p['phi_deg']:>6.0f}° "
              f"{p['n_clear_path']:>5d} {p['mean_skull_path_mm']:>8.2f}mm "
              f"{p['total_skull_path_mm']:>9.1f}mm {p['max_skull_path_mm']:>7.2f}mm "
              f"{p['ray_skull_thickness_mm']:>8.2f}mm")

    # Also show bottom 5 (worst)
    print(f"\n  Bottom 5 array placements (worst):")
    for i, p in enumerate(all_placements[-5:][::-1]):
        print(f"    {i+1:>4} {p['theta_deg']:>6.0f}° {p['phi_deg']:>6.0f}° "
              f"{p['n_clear_path']:>5d} {p['mean_skull_path_mm']:>8.2f}mm "
              f"{p['total_skull_path_mm']:>9.1f}mm")

    # Best by max clear-path elements (tiebreak: lowest total skull)
    best_clear = all_placements[0]

    # Best by lowest total skull path (regardless of clear count)
    best_total = min(all_placements, key=lambda r: r["total_skull_path_mm"])

    # Also keep top 5 as "optimal_results" for JSON
    optimal_results = all_placements[:5]

    # Print both optima
    for label, best in [("MAX-CLEAR", best_clear), ("MIN-TOTAL-SKULL", best_total)]:
        print(f"\n  BEST ({label}): theta={best['theta_deg']:.0f}° phi={best['phi_deg']:.0f}°")
        print(f"    Clear path: {best['n_clear_path']}/64 (naive: {naive_result['n_clear_path']}/64)")
        print(f"    Mean skull path: {best['mean_skull_path_mm']:.2f} mm "
              f"(naive: {naive_result['mean_skull_path_mm']:.2f} mm)")
        print(f"    Total skull path: {best['total_skull_path_mm']:.1f} mm "
              f"(naive: {naive_result['total_skull_path_mm']:.1f} mm)")
        dc = best["n_clear_path"] - naive_result["n_clear_path"]
        dt = naive_result["total_skull_path_mm"] - best["total_skull_path_mm"]
        print(f"    vs naive: {dc:+d} clear elements, {dt:+.1f} mm total skull path")

    improvement_clear = best_clear["n_clear_path"] - naive_result["n_clear_path"]
    improvement_total = naive_result["total_skull_path_mm"] - best_clear["total_skull_path_mm"]

    return {
        "subject": subj,
        "brain_center_mm": target_mm.tolist(),
        "naive_approach_axis": approach_axis,
        "naive_direction": naive_direction.tolist(),
        "naive_placement": {
            k: v for k, v in naive_result.items() if k != "element_details"
        },
        "thickness_map_summary": {
            "n_directions": len(thickness_map),
            "mean_thickness_mm": float(np.mean(all_thick)),
            "std_thickness_mm": float(np.std(all_thick)),
            "min_thickness_mm": float(np.min(all_thick)),
            "max_thickness_mm": float(np.max(all_thick)),
            "median_thickness_mm": float(np.median(all_thick)),
        },
        "top10_thinnest": [
            {k: v for k, v in e.items()} for e in thickness_map[:10]
        ],
        "optimal_placements": [
            {k: v for k, v in r.items() if k != "element_details"} for r in optimal_results
        ],
        "best_max_clear": {k: v for k, v in best_clear.items() if k != "element_details"},
        "best_min_total": {k: v for k, v in best_total.items() if k != "element_details"},
        "improvement_max_clear": {
            "clear_elements_gained": improvement_clear,
            "total_skull_path_reduction_mm": float(improvement_total),
        },
        "improvement_min_total": {
            "clear_elements_gained": best_total["n_clear_path"] - naive_result["n_clear_path"],
            "total_skull_path_reduction_mm": float(naive_result["total_skull_path_mm"] - best_total["total_skull_path_mm"]),
        },
    }


def print_comparison_table(all_results):
    print(f"\n{'='*110}")
    print(f"  CROSS-SUBJECT COMPARISON: Naive vs Optimal Array Placement")
    print(f"{'='*110}")

    header = (
        f"{'Subject':>8} | {'Placement':>14} | {'Theta':>6} {'Phi':>6} | "
        f"{'Clear':>5} {'Skull':>5} | {'MeanPath':>9} {'TotalPath':>10} | {'MaxPath':>8}"
    )
    print(header)
    print("-" * len(header))

    for subj in SUBJECTS:
        if subj not in all_results:
            continue
        r = all_results[subj]
        naive = r["naive_placement"]
        best_clear = r["best_max_clear"]
        best_total = r["best_min_total"]

        naive_theta = {0: 90, 1: 90, 2: 0}[r["naive_approach_axis"]]
        naive_phi = {0: 0, 1: 90, 2: 0}[r["naive_approach_axis"]]

        def print_row(label, p, theta=None, phi=None):
            t = theta if theta is not None else p.get("theta_deg", 0)
            ph = phi if phi is not None else p.get("phi_deg", 0)
            print(f"{subj if label == 'NAIVE' else '':>8} | {label:>14} | {t:>5.0f}° {ph:>5.0f}° | "
                  f"{p['n_clear_path']:>5d} {p['n_through_skull']:>5d} | "
                  f"{p['mean_skull_path_mm']:>8.2f}mm {p['total_skull_path_mm']:>9.1f}mm | "
                  f"{p['max_skull_path_mm']:>7.2f}mm")

        print_row("NAIVE", naive, naive_theta, naive_phi)
        print_row("MAX-CLEAR", best_clear)

        # Only show MIN-TOTAL if it differs from MAX-CLEAR
        if (best_total["theta_deg"] != best_clear["theta_deg"] or
                best_total["phi_deg"] != best_clear["phi_deg"]):
            print_row("MIN-TOTAL", best_total)

        print("-" * len(header))

    # Interpretation
    print(f"\n  INTERPRETATION")
    print(f"  {'='*70}")
    for subj in SUBJECTS:
        if subj not in all_results:
            continue
        r = all_results[subj]
        naive = r["naive_placement"]
        bc = r["best_max_clear"]
        bt = r["best_min_total"]

        print(f"\n  {subj}:")
        print(f"    Naive (axis {r['naive_approach_axis']}): "
              f"{naive['n_clear_path']}/64 clear, "
              f"{naive['total_skull_path_mm']:.0f}mm total skull, "
              f"{naive['mean_skull_path_mm']:.1f}mm mean/element")

        dc = bc["n_clear_path"] - naive["n_clear_path"]
        dt = naive["total_skull_path_mm"] - bc["total_skull_path_mm"]
        print(f"    Max-clear (theta={bc['theta_deg']:.0f}, phi={bc['phi_deg']:.0f}): "
              f"{bc['n_clear_path']}/64 clear ({dc:+d}), "
              f"{bc['total_skull_path_mm']:.0f}mm total ({dt:+.0f}mm)")

        if (bt["theta_deg"] != bc["theta_deg"] or bt["phi_deg"] != bc["phi_deg"]):
            dc2 = bt["n_clear_path"] - naive["n_clear_path"]
            dt2 = naive["total_skull_path_mm"] - bt["total_skull_path_mm"]
            print(f"    Min-total (theta={bt['theta_deg']:.0f}, phi={bt['phi_deg']:.0f}): "
                  f"{bt['n_clear_path']}/64 clear ({dc2:+d}), "
                  f"{bt['total_skull_path_mm']:.0f}mm total ({dt2:+.0f}mm)")

        # Key finding
        if dc > 0 and dt > 0:
            ratio = bt["total_skull_path_mm"] / naive["total_skull_path_mm"]
            print(f"    ==> Optimization wins on both axes: more clear elements AND less skull.")
        elif dc > 0 and dt <= 0:
            print(f"    ==> Tradeoff: gains {dc} clear elements but increases total skull path by {-dt:.0f}mm.")
            print(f"        Clear elements dominate the coherent sum, so this is likely still a net win.")
        elif dc <= 0 and dt > 0:
            print(f"    ==> Tradeoff: reduces total skull but loses clear elements.")
        else:
            print(f"    ==> Naive placement is near-optimal for this subject.")


def main():
    all_results = {}
    t_start = time.time()

    for subj in SUBJECTS:
        result = process_subject(subj)
        if result is not None:
            all_results[subj] = result

    print_comparison_table(all_results)

    # Save results
    output_dir = Path.home() / "Data/openlifu-validation/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "skull_thickness_maps.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to: {output_path}")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
