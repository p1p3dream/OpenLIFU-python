#!/usr/bin/env python3
"""Sanity-plot the 24 'no skull intersection' rays from the GU008 incidence
diagnostic to decide whether they are real brain-only corridors or artifacts
of segmentation holes / ray-caster sampling.

Procedure (all on the sim-grid segmentation, same as
``incidence_angle_amplitude_gu008.py``):

 1. For each of 64 elements, cast a straight ray to the brain-centroid
    target. Sample the skull mask along the ray at 0.25 mm increments with
    ``map_coordinates`` at order=0 (nearest) AND order=1 (linear). Record
    the disagreement rate between samplers and the skull path length.
 2. Render a 3D figure with the skull mask as a translucent point cloud,
    the 64 rays (red = no intersection, blue = intersection), element
    positions as dots, target as a star. Also provide three orthogonal
    2D projections in the same figure.
 3. For the no-intersection rays, compute the minimum distance from the
    ray to the nearest skull voxel using a Euclidean distance transform.
    Report a histogram of those min-distances.
 4. Render the mask value and distance-to-skull along the first
    no-intersection ray; print its first 20 sampled positions.
 5. Repeat the intersection count with the skull mask dilated by 2
    voxels. If the no-intersection count drops significantly under
    dilation, the skull label has small holes.

LOCAL / UNCOMMITTED diagnostic. Does NOT modify src/.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
import numpy as np
import xarray as xa
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates


# -------------------------------------------------------------------
# Same logging patch as the sibling scripts.
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
logger = logging.getLogger("ray_sanity_gu008")

# ---- Must match incidence_angle_amplitude_gu008.py / coherence_factor_gu008.py ----
MRI_PATH = Path.home() / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)
LABEL_NIFTI_PATH = Path.home() / "Data/openlifu-validation/results/GU008_nnunet_labels.nii.gz"
PLOT_PATH = Path.home() / "Data/openlifu-validation/results/ray_sanity_gu008.png"
RAY_PLOT_PATH = Path.home() / "Data/openlifu-validation/results/ray_sanity_gu008_representative.png"

N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
GRID_SPACING_MM = 0.5
GRID_MARGIN_MM = 10.0
C0 = 1500.0

RAY_STEP_MM = 0.25


def world_mm_to_vox(pos_xyz_mm: np.ndarray, axis_origins: np.ndarray,
                    spacing_mm: float) -> np.ndarray:
    """Convert a world-mm (x,y,z) point to voxel indices (float)."""
    return (pos_xyz_mm - axis_origins) / spacing_mm


def ray_sample_mask(mask: np.ndarray, p0_mm: np.ndarray, p1_mm: np.ndarray,
                    axis_origins: np.ndarray, spacing_mm: float,
                    step_mm: float = RAY_STEP_MM, order: int = 0):
    """Sample ``mask`` along the straight segment p0 -> p1 in ``step_mm`` steps.

    Returns:
      ts      : array of distances from p0 (mm) at each sample.
      samples : sampled mask values (shape (N,)).
      pts_mm  : world-mm sample coordinates (shape (N, 3)).
      pts_vox : voxel-index sample coordinates (shape (N, 3)).
    """
    vec = p1_mm - p0_mm
    total = float(np.linalg.norm(vec))
    n = max(2, int(np.ceil(total / step_mm)) + 1)
    ts = np.linspace(0.0, total, n)
    pts_mm = p0_mm[None, :] + ts[:, None] * (vec / max(total, 1e-12))[None, :]
    pts_vox = (pts_mm - axis_origins[None, :]) / spacing_mm
    coords = np.stack([pts_vox[:, 0], pts_vox[:, 1], pts_vox[:, 2]], axis=0)
    samples = map_coordinates(mask, coords, order=order, mode="constant", cval=0.0)
    return ts, samples, pts_mm, pts_vox


def main() -> int:
    t_total = time.time()
    print("=" * 78)
    print("Ray-sanity of GU008 no-intersection elements")
    print("=" * 78)

    if not MRI_PATH.exists():
        print(f"ERROR: MRI not found: {MRI_PATH}")
        return 1
    if not LABEL_NIFTI_PATH.exists():
        print(f"ERROR: nnU-Net labels not found: {LABEL_NIFTI_PATH}")
        return 1

    # ---- Load MRI + nnU-Net labels ----
    print(f"[1] Loading MRI: {MRI_PATH}")
    volume = load_nifti_as_xarray(MRI_PATH)

    print(f"[2] Loading nnU-Net labels: {LABEL_NIFTI_PATH}")
    seg_method = PreSegmented(label_nifti_path=str(LABEL_NIFTI_PATH))
    seg_labels = seg_method._segment(volume)
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()

    # ---- Brain-centroid target + approach axis ----
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
    print(f"    Brain centroid target: ({target_mm[0]:.1f}, {target_mm[1]:.1f}, "
          f"{target_mm[2]:.1f}) mm")

    skull_mask_full = seg_arr == material_idx["skull"]
    skull_indices_full = np.argwhere(skull_mask_full)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices_full[:, ax]]
        for ax in range(3)
    ])
    max_skull_dist_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_dist_per_axis))
    print(f"    Approach axis: {dim_names[approach_axis]} (axis {approach_axis})")

    # ---- Build array, bake world positions ----
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

    # ---- Build sim grid (same recipe as reference script) ----
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

    # Resample MRI onto sim grid just so we can segment it.
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
    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    sim_dims = list(sim_seg.dims)  # should be ['x','y','z']
    print(f"    sim_seg dims: {sim_dims}")
    assert sim_dims == ["x", "y", "z"], f"unexpected dim order {sim_dims}"

    skull_mask = (sim_seg_arr == material_idx["skull"]).astype(np.uint8)
    n_skull_sim = int(skull_mask.sum())
    print(f"    skull voxels (sim grid): {n_skull_sim:,d} "
          f"({100.0 * n_skull_sim / skull_mask.size:.2f}%)")

    axis_origins = np.array([sim_coord_arrays[a][0] for a in range(3)])

    # ---- Precompute Euclidean distance transform (mm) to skull ----
    print(f"[5] Distance transform (mm) -> nearest skull voxel ...")
    t0 = time.time()
    dt_vox = distance_transform_edt(
        skull_mask == 0, sampling=[GRID_SPACING_MM, GRID_SPACING_MM, GRID_SPACING_MM]
    ).astype(np.float32)
    print(f"    done in {time.time() - t0:.1f}s; max DT = {dt_vox.max():.2f} mm")

    # ---- Dilated skull mask for the segmentation-hole experiment ----
    print(f"[6] Dilated skull mask (2 voxels = 1 mm) ...")
    skull_mask_dil = binary_dilation(skull_mask.astype(bool), iterations=2).astype(np.uint8)
    print(f"    dilated skull voxels: {int(skull_mask_dil.sum()):,d} "
          f"(delta = {int(skull_mask_dil.sum()) - n_skull_sim:+,d})")

    # ---- Element positions (world mm, x,y,z order) ----
    el_positions_mm = np.array([
        el.get_position(units="mm") for el in arr.elements
    ])

    # ---- Per-element ray-cast ----
    print(f"[7] Ray-casting 64 elements through skull mask ...")
    any_hit_nearest = np.zeros(N_ELEMENTS, dtype=bool)
    any_hit_linear = np.zeros(N_ELEMENTS, dtype=bool)
    any_hit_dilated = np.zeros(N_ELEMENTS, dtype=bool)
    skull_path_mm = np.zeros(N_ELEMENTS)
    first_entry_t = np.full(N_ELEMENTS, np.nan)
    last_exit_t = np.full(N_ELEMENTS, np.nan)
    min_dist_to_skull = np.full(N_ELEMENTS, np.nan)
    disagreement_counts = np.zeros(N_ELEMENTS, dtype=int)
    disagreement_totals = np.zeros(N_ELEMENTS, dtype=int)

    per_ray_samples = []  # keep for later (list of dicts)

    for el_i in range(N_ELEMENTS):
        p0 = el_positions_mm[el_i]
        p1 = target_mm
        ts, s_near, pts_mm, pts_vox = ray_sample_mask(
            skull_mask, p0, p1, axis_origins, GRID_SPACING_MM,
            step_mm=RAY_STEP_MM, order=0,
        )
        _, s_lin, _, _ = ray_sample_mask(
            skull_mask, p0, p1, axis_origins, GRID_SPACING_MM,
            step_mm=RAY_STEP_MM, order=1,
        )
        _, s_dil, _, _ = ray_sample_mask(
            skull_mask_dil, p0, p1, axis_origins, GRID_SPACING_MM,
            step_mm=RAY_STEP_MM, order=0,
        )
        # Distance transform along the ray (mm to nearest skull voxel).
        _, s_dt, _, _ = ray_sample_mask(
            dt_vox, p0, p1, axis_origins, GRID_SPACING_MM,
            step_mm=RAY_STEP_MM, order=1,
        )

        hit_near_mask = s_near >= 1
        hit_lin_mask = s_lin > 0.5
        hit_dil_mask = s_dil >= 1

        any_hit_nearest[el_i] = bool(hit_near_mask.any())
        any_hit_linear[el_i] = bool(hit_lin_mask.any())
        any_hit_dilated[el_i] = bool(hit_dil_mask.any())

        n_hits_near = int(hit_near_mask.sum())
        skull_path_mm[el_i] = n_hits_near * RAY_STEP_MM

        if any_hit_nearest[el_i]:
            first_entry_t[el_i] = float(ts[np.argmax(hit_near_mask)])
            # last index where nearest mask is True.
            last_idx = int(len(hit_near_mask) - 1 - np.argmax(hit_near_mask[::-1]))
            last_exit_t[el_i] = float(ts[last_idx])

        min_dist_to_skull[el_i] = float(s_dt.min())
        disag = (hit_near_mask != hit_lin_mask)
        disagreement_counts[el_i] = int(disag.sum())
        disagreement_totals[el_i] = int(len(hit_near_mask))

        per_ray_samples.append({
            "el": el_i, "ts": ts, "near": s_near, "lin": s_lin,
            "dt": s_dt, "pts_mm": pts_mm,
        })

    n_no_int_near = int((~any_hit_nearest).sum())
    n_no_int_lin = int((~any_hit_linear).sum())
    n_no_int_dil = int((~any_hit_dilated).sum())
    total_disag = int(disagreement_counts.sum())
    total_samples = int(disagreement_totals.sum())
    print(f"    no-intersection (nearest / order=0): {n_no_int_near} / {N_ELEMENTS}")
    print(f"    no-intersection (linear  / order=1): {n_no_int_lin} / {N_ELEMENTS}")
    print(f"    no-intersection (dilated 2 vox):     {n_no_int_dil} / {N_ELEMENTS}")
    print(f"    sampler disagreement rate: "
          f"{total_disag}/{total_samples} = "
          f"{100.0 * total_disag / max(total_samples, 1):.3f}%")

    no_int_ids = np.where(~any_hit_nearest)[0]
    print(f"    no-intersection element ids: {no_int_ids.tolist()}")

    # ---- Min-distance histogram over no-intersection rays ----
    min_dists_no_int = min_dist_to_skull[no_int_ids]
    print("\n[8] Min distance from ray to nearest skull voxel (no-intersection rays):")
    if len(min_dists_no_int) > 0:
        print(f"    n = {len(min_dists_no_int)}")
        print(f"    min   = {min_dists_no_int.min():.3f} mm")
        print(f"    p25   = {np.percentile(min_dists_no_int, 25):.3f} mm")
        print(f"    med   = {np.median(min_dists_no_int):.3f} mm")
        print(f"    p75   = {np.percentile(min_dists_no_int, 75):.3f} mm")
        print(f"    max   = {min_dists_no_int.max():.3f} mm")
        n_lt_1 = int((min_dists_no_int < 1.0).sum())
        n_1_2 = int(((min_dists_no_int >= 1.0) & (min_dists_no_int < 2.0)).sum())
        n_ge_2 = int((min_dists_no_int >= 2.0).sum())
        print(f"    <1 mm (grazing):      {n_lt_1}")
        print(f"    1-2 mm (borderline):  {n_1_2}")
        print(f"    >=2 mm (real gap):    {n_ge_2}")
    else:
        print("    (no no-intersection elements; nothing to analyze)")

    # ---- First representative no-intersection ray ----
    if len(no_int_ids) > 0:
        rep_el = int(no_int_ids[0])
        rep = per_ray_samples[rep_el]
        print(f"\n[9] Representative no-intersection ray: element {rep_el}")
        print(f"    element pos (mm): {el_positions_mm[rep_el]}")
        print(f"    target      (mm): {target_mm}")
        print(f"    ray length (mm):  {rep['ts'][-1]:.2f}")
        print(f"    {'i':>3} {'t_mm':>8} {'x':>8} {'y':>8} {'z':>8} "
              f"{'near':>5} {'lin':>7} {'dt_mm':>7}")
        for i in range(min(20, len(rep['ts']))):
            x, y, z = rep['pts_mm'][i]
            print(f"    {i:>3d} {rep['ts'][i]:8.3f} {x:8.2f} {y:8.2f} {z:8.2f} "
                  f"{int(rep['near'][i]):>5d} {rep['lin'][i]:7.4f} "
                  f"{rep['dt'][i]:7.3f}")
    else:
        rep_el = None

    # ---- Main 3D + 2D projection figure ----
    print(f"\n[10] Rendering {PLOT_PATH} ...")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Subsample skull voxels for plotting (point cloud).
    sk_idx = np.argwhere(skull_mask == 1)
    if len(sk_idx) > 5000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(sk_idx), size=5000, replace=False)
        sk_idx = sk_idx[sel]
    sk_mm = np.stack([
        sim_coord_arrays[0][sk_idx[:, 0]],
        sim_coord_arrays[1][sk_idx[:, 1]],
        sim_coord_arrays[2][sk_idx[:, 2]],
    ], axis=-1)

    fig = plt.figure(figsize=(20, 14))

    # 3D plot top-left
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.scatter(sk_mm[:, 0], sk_mm[:, 1], sk_mm[:, 2],
                 c="0.6", s=1, alpha=0.15, label="skull (subsampled)")
    for el_i in range(N_ELEMENTS):
        p0 = el_positions_mm[el_i]
        p1 = target_mm
        color = "red" if not any_hit_nearest[el_i] else "blue"
        alpha = 0.9 if not any_hit_nearest[el_i] else 0.25
        lw = 1.6 if not any_hit_nearest[el_i] else 0.7
        ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                  color=color, alpha=alpha, lw=lw)
    ax3d.scatter(el_positions_mm[:, 0], el_positions_mm[:, 1],
                 el_positions_mm[:, 2], c="k", s=14, label="elements")
    ax3d.scatter([target_mm[0]], [target_mm[1]], [target_mm[2]],
                 c="gold", s=250, marker="*", edgecolors="k",
                 linewidths=0.8, label="target")
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.set_title(f"3D: {n_no_int_near} red = no skull intersection, "
                   f"{N_ELEMENTS - n_no_int_near} blue = intersection")
    ax3d.legend(loc="upper left", fontsize=8)

    # Three orthogonal 2D projections.
    proj_axes = [
        ("XY (project Z)", 0, 1),
        ("XZ (project Y)", 0, 2),
        ("YZ (project X)", 1, 2),
    ]
    for i, (title, ax_a, ax_b) in enumerate(proj_axes):
        axp = fig.add_subplot(2, 3, 2 + i)
        axp.scatter(sk_mm[:, ax_a], sk_mm[:, ax_b],
                    c="0.6", s=1, alpha=0.2)
        for el_i in range(N_ELEMENTS):
            p0 = el_positions_mm[el_i]
            p1 = target_mm
            color = "red" if not any_hit_nearest[el_i] else "blue"
            alpha = 0.9 if not any_hit_nearest[el_i] else 0.25
            lw = 1.3 if not any_hit_nearest[el_i] else 0.5
            axp.plot([p0[ax_a], p1[ax_a]], [p0[ax_b], p1[ax_b]],
                     color=color, alpha=alpha, lw=lw)
        axp.scatter(el_positions_mm[:, ax_a], el_positions_mm[:, ax_b],
                    c="k", s=10)
        axp.scatter([target_mm[ax_a]], [target_mm[ax_b]],
                    c="gold", s=160, marker="*", edgecolors="k",
                    linewidths=0.6)
        axp.set_aspect("equal")
        axp.set_title(title)
        axp.grid(True, alpha=0.3)

    # Min-distance histogram (bottom-left).
    ax_hist = fig.add_subplot(2, 3, 5)
    if len(min_dists_no_int) > 0:
        bins = np.linspace(0, max(4.0, float(min_dists_no_int.max() * 1.05)), 25)
        ax_hist.hist(min_dists_no_int, bins=bins, color="red",
                     alpha=0.75, edgecolor="k")
        ax_hist.axvline(1.0, color="orange", ls="--", lw=1,
                        label="1 mm (grazing)")
        ax_hist.axvline(2.0, color="green", ls="--", lw=1,
                        label="2 mm (real gap)")
        ax_hist.set_xlabel("min distance from ray to nearest skull voxel (mm)")
        ax_hist.set_ylabel("count")
        ax_hist.set_title(f"Min-distance histogram "
                          f"({len(min_dists_no_int)} no-int. rays)")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
    else:
        ax_hist.text(0.5, 0.5, "(none)", ha="center", va="center",
                     transform=ax_hist.transAxes)
        ax_hist.set_title("Min-distance histogram (no rays)")

    # Dilation summary (bottom-right).
    ax_txt = fig.add_subplot(2, 3, 6)
    ax_txt.axis("off")
    txt_lines = [
        "Dilation experiment (skull mask dilated by 2 voxels = 1 mm):",
        f"  no-intersection count without dilation: {n_no_int_near} / {N_ELEMENTS}",
        f"  no-intersection count with    dilation: {n_no_int_dil} / {N_ELEMENTS}",
        f"  rays 'rescued' by dilation:             {n_no_int_near - n_no_int_dil}",
        "",
        "Sampler comparison:",
        f"  no-intersection (order=0 nearest): {n_no_int_near}",
        f"  no-intersection (order=1 linear):  {n_no_int_lin}",
        f"  total sampler disagreements: "
        f"{total_disag}/{total_samples} = "
        f"{100.0 * total_disag / max(total_samples, 1):.3f}%",
        "",
    ]
    if len(min_dists_no_int) > 0:
        n_lt_1 = int((min_dists_no_int < 1.0).sum())
        n_1_2 = int(((min_dists_no_int >= 1.0) & (min_dists_no_int < 2.0)).sum())
        n_ge_2 = int((min_dists_no_int >= 2.0).sum())
        txt_lines += [
            "Min-distance classification of no-intersection rays:",
            f"  <1 mm (grazing real skull):   {n_lt_1}",
            f"  1-2 mm (borderline):          {n_1_2}",
            f"  >=2 mm (real gap in mask):    {n_ge_2}",
        ]
    ax_txt.text(0.0, 1.0, "\n".join(txt_lines), va="top", ha="left",
                family="monospace", fontsize=9, transform=ax_txt.transAxes)

    fig.suptitle("GU008 ray-sanity: no-intersection elements vs skull mask",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {PLOT_PATH}")

    # ---- Representative-ray plot (mask value + DT vs t) ----
    if rep_el is not None:
        rep = per_ray_samples[rep_el]
        fig2, (axm, axd) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axm.step(rep["ts"], rep["near"], where="post",
                 label="nearest (order=0)", color="k")
        axm.plot(rep["ts"], rep["lin"], label="linear (order=1)",
                 color="tab:red", lw=1)
        axm.set_ylabel("skull mask value")
        axm.set_title(f"Element {rep_el} ray: mask along ray")
        axm.legend()
        axm.grid(True, alpha=0.3)

        axd.plot(rep["ts"], rep["dt"], color="tab:blue")
        axd.axhline(1.0, color="orange", ls="--", lw=1, label="1 mm")
        axd.axhline(2.0, color="green", ls="--", lw=1, label="2 mm")
        axd.set_xlabel("distance from element along ray (mm)")
        axd.set_ylabel("distance to nearest skull voxel (mm)")
        axd.legend()
        axd.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(RAY_PLOT_PATH, dpi=140, bbox_inches="tight")
        plt.close(fig2)
        print(f"    wrote {RAY_PLOT_PATH}")

    # ---- Final verdict ----
    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    if len(min_dists_no_int) > 0:
        n_lt_1 = int((min_dists_no_int < 1.0).sum())
        n_1_2 = int(((min_dists_no_int >= 1.0) & (min_dists_no_int < 2.0)).sum())
        n_ge_2 = int((min_dists_no_int >= 2.0).sum())
        rescued = n_no_int_near - n_no_int_dil
        print(f"  {len(min_dists_no_int)} no-intersection rays total.")
        print(f"  min-dist <1mm (grazing real skull): {n_lt_1}")
        print(f"  min-dist 1-2mm (borderline):        {n_1_2}")
        print(f"  min-dist >=2mm (real mask gap):     {n_ge_2}")
        print(f"  dilation (1 mm) rescued:            {rescued}")
        if n_ge_2 >= 0.5 * len(min_dists_no_int):
            print("  -> Most no-intersection rays pass through real gaps in the ")
            print("     skull label. The segmentation is fragmented; path-length")
            print("     predictor is likely under-counting skull for those rays.")
        elif n_lt_1 >= 0.5 * len(min_dists_no_int):
            print("  -> Most no-intersection rays graze real skull edges.")
            print("     The ray-caster is legit; these are narrow brain-only")
            print("     corridors through thin bone / foramina.")
        else:
            print("  -> Borderline distribution. Consider the dilation run.")

    print(f"\nTotal elapsed: {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
