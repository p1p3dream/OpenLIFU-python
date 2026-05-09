#!/usr/bin/env python3
"""Analyze skull morphology for outlier subjects that didn't improve with two-class bone.

Compares 4 "stubborn" subjects (still >15 dB after two-class) vs 7 "improved" subjects
to identify morphological differences that explain why two-class splitting didn't help.

Metrics computed:
  - Total skull voxel count and volume (mm^3)
  - Cortical vs trabecular voxel counts and ratio
  - Skull thickness distribution (via EDT from exterior)
  - Mean/median/max skull thickness
  - Percentage of skull thinner than cortical threshold (2.5mm)
  - Connected component analysis
  - Thickness percentile distribution
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndlabel

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RESULTS_DIR = os.path.expanduser("~/Data/openlifu-validation/results")
CORTICAL_THICKNESS_MM = 2.5
SKULL_LABEL = 5
TRABECULAR_LABEL = 7

# All 11 outlier subjects
SUBJECTS = [
    "NC024", "NC011", "GU006", "NC015", "GU035",
    "NYU005", "NC017", "NC033", "NC029", "NC012", "NYU008",
]

# Categorization
STUBBORN = {"GU035", "NC033", "NC029", "NC012"}  # still >15 dB
IMPROVED = {"NC024", "NC011", "GU006", "NC015", "NYU005", "NC017", "NYU008"}

# Known attenuation values
ATTEN = {
    "NC024": (16.28, 9.94),
    "NC011": (15.41, 10.37),
    "GU006": (16.00, 12.83),
    "NC015": (16.69, 9.94),
    "GU035": (17.42, 17.41),
    "NYU005": (15.93, 12.77),
    "NC017": (15.06, 12.39),
    "NC033": (18.19, 17.19),
    "NC029": (18.96, 15.62),
    "NC012": (20.60, 17.82),
    "NYU008": (20.77, 13.52),
}

SKULL_PATH_NEAR = {
    "GU035": 19.0,
    "NC033": 11.5,
    "NC029": 8.0,
    "NC012": 14.0,
    "NC024": 8.0,
    "NYU008": 9.5,
    "NC015": 14.0,
}


def compute_skull_thickness_via_rays(skull_mask: np.ndarray, spacing: tuple) -> np.ndarray:
    """Compute local skull thickness by measuring distance from each boundary
    voxel to the opposite boundary.

    For each skull voxel, the local thickness is approximated as:
        thickness = 2 * distance_from_boundary
    where distance_from_boundary is the EDT of the skull mask (distance to
    nearest non-skull voxel). The maximum distance in a region gives half
    the local thickness.

    This is equivalent to the "inscribed sphere" method for thickness measurement.

    Returns the full EDT array (only meaningful where skull_mask is True).
    """
    return distance_transform_edt(skull_mask, sampling=spacing)


def analyze_subject(subj: str) -> dict:
    """Analyze skull morphology for a single subject."""
    nnunet_path = os.path.join(RESULTS_DIR, f"{subj}_nnunet_labels.nii.gz")
    twoclass_path = os.path.join(RESULTS_DIR, f"{subj}_two_class_labels.nii.gz")

    img_nn = nib.load(nnunet_path)
    labels_nn = np.asarray(img_nn.dataobj).astype(np.int16)
    spacing = tuple(abs(float(x)) for x in img_nn.header.get_zooms()[:3])
    voxel_vol = float(np.prod(spacing))

    img_tc = nib.load(twoclass_path)
    labels_tc = np.asarray(img_tc.dataobj).astype(np.int16)

    # Original skull mask (from nnunet, label 5)
    skull_mask = labels_nn == SKULL_LABEL
    n_skull = int(skull_mask.sum())

    # Two-class counts
    n_cortical = int((labels_tc == SKULL_LABEL).sum())
    n_trabecular = int((labels_tc == TRABECULAR_LABEL).sum())

    # EDT from skull boundary (distance into the skull interior)
    dist = compute_skull_thickness_via_rays(skull_mask, spacing)
    skull_dists = dist[skull_mask]

    # The local "half-thickness" at each voxel is its EDT value.
    # The max EDT value gives the radius of the largest inscribed sphere,
    # which is half the max skull thickness at that point.
    max_half_thickness = float(skull_dists.max())
    mean_half_thickness = float(skull_dists.mean())
    median_half_thickness = float(np.median(skull_dists))

    # Percentiles of distance-from-boundary
    pcts = np.percentile(skull_dists, [10, 25, 50, 75, 90, 95, 99])

    # What fraction of skull voxels are within the cortical threshold?
    # These voxels are PURELY cortical (no trabecular relabeling)
    frac_within_threshold = float((skull_dists <= CORTICAL_THICKNESS_MM).sum()) / n_skull

    # What fraction of skull has essentially zero trabecular (i.e., skull is
    # thinner than 2*cortical_threshold everywhere in local region)?
    # A voxel with dist > threshold gets relabeled trabecular.
    # If frac_within_threshold is very high, the skull is mostly thin and
    # there's almost no trabecular to relabel.
    frac_trabecular = n_trabecular / n_skull if n_skull > 0 else 0

    # Connected components of skull
    labeled_skull, n_components = ndlabel(skull_mask)

    # Component sizes
    comp_sizes = []
    for i in range(1, n_components + 1):
        comp_sizes.append(int((labeled_skull == i).sum()))
    comp_sizes.sort(reverse=True)
    largest_component_frac = comp_sizes[0] / n_skull if comp_sizes else 0

    # Trabecular voxel distance stats (from two-class labels)
    trab_mask = labels_tc == TRABECULAR_LABEL
    if trab_mask.any():
        trab_dists = dist[trab_mask]
        trab_mean_dist = float(trab_dists.mean())
        trab_max_dist = float(trab_dists.max())
        trab_median_dist = float(np.median(trab_dists))
    else:
        trab_mean_dist = 0.0
        trab_max_dist = 0.0
        trab_median_dist = 0.0

    # Thickness distribution: for each "column" through the skull,
    # the local thickness ~ 2 * max(EDT) along that column.
    # We approximate by looking at the distribution of 2*EDT for all skull voxels.
    # The histogram of max EDT per connected region gives thickness variation.

    # Compute actual raycast thickness along each axis (z-axis = axial)
    # For each (x,y) column, count consecutive skull voxels along z
    z_thickness_mm = []
    for i in range(skull_mask.shape[0]):
        for j in range(skull_mask.shape[1]):
            col = skull_mask[i, j, :]
            if not col.any():
                continue
            # Find runs of True
            diffs = np.diff(col.astype(int))
            starts = np.where(diffs == 1)[0] + 1
            ends = np.where(diffs == -1)[0] + 1
            # Handle edge cases
            if col[0]:
                starts = np.concatenate([[0], starts])
            if col[-1]:
                ends = np.concatenate([ends, [len(col)]])
            for s, e in zip(starts, ends):
                z_thickness_mm.append((e - s) * spacing[2])

    z_thickness_mm = np.array(z_thickness_mm) if z_thickness_mm else np.array([0.0])

    return {
        "subj": subj,
        "group": "STUBBORN" if subj in STUBBORN else "IMPROVED",
        "spacing": spacing,
        "voxel_vol_mm3": voxel_vol,
        "n_skull": n_skull,
        "skull_vol_mm3": n_skull * voxel_vol,
        "n_cortical": n_cortical,
        "n_trabecular": n_trabecular,
        "pct_cortical": 100.0 * n_cortical / n_skull if n_skull > 0 else 0,
        "pct_trabecular": 100.0 * n_trabecular / n_skull if n_skull > 0 else 0,
        "frac_within_threshold": frac_within_threshold,
        "max_half_thickness_mm": max_half_thickness,
        "mean_half_thickness_mm": mean_half_thickness,
        "median_half_thickness_mm": median_half_thickness,
        "edt_pct": pcts,  # 10, 25, 50, 75, 90, 95, 99
        "n_components": n_components,
        "largest_component_frac": largest_component_frac,
        "top3_components": comp_sizes[:3],
        "trab_mean_dist_mm": trab_mean_dist,
        "trab_max_dist_mm": trab_max_dist,
        "trab_median_dist_mm": trab_median_dist,
        "z_thickness_mean_mm": float(z_thickness_mm.mean()),
        "z_thickness_median_mm": float(np.median(z_thickness_mm)),
        "z_thickness_max_mm": float(z_thickness_mm.max()),
        "z_thickness_p90_mm": float(np.percentile(z_thickness_mm, 90)),
        "z_thickness_p95_mm": float(np.percentile(z_thickness_mm, 95)),
        "z_thickness_p99_mm": float(np.percentile(z_thickness_mm, 99)),
    }


def print_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    # Sort: stubborn first, then improved, each alphabetically
    stubborn = sorted([r for r in results if r["group"] == "STUBBORN"], key=lambda r: r["subj"])
    improved = sorted([r for r in results if r["group"] == "IMPROVED"], key=lambda r: r["subj"])

    print("\n" + "=" * 130)
    print("SKULL MORPHOLOGY ANALYSIS: STUBBORN vs IMPROVED SUBJECTS")
    print("=" * 130)

    # Core metrics table
    header = (
        f"{'Subj':<8} {'Group':<10} {'Atten 1cl':>9} {'Atten 2cl':>9} {'Delta':>6} "
        f"{'SkullVox':>10} {'Vol(cc)':>8} {'%Cort':>6} {'%Trab':>6} "
        f"{'MaxHT':>6} {'MeanHT':>7} {'MedHT':>6} "
        f"{'%<2.5':>6} {'Comps':>5}"
    )
    print("\n" + header)
    print("-" * len(header))

    for group_label, group_data in [("STUBBORN", stubborn), ("IMPROVED", improved)]:
        for r in group_data:
            s = r["subj"]
            a1, a2 = ATTEN.get(s, (0, 0))
            delta = a2 - a1
            print(
                f"{s:<8} {r['group']:<10} {a1:>9.2f} {a2:>9.2f} {delta:>+6.2f} "
                f"{r['n_skull']:>10,d} {r['skull_vol_mm3']/1000:>8.1f} "
                f"{r['pct_cortical']:>6.1f} {r['pct_trabecular']:>6.1f} "
                f"{r['max_half_thickness_mm']:>6.2f} {r['mean_half_thickness_mm']:>7.2f} "
                f"{r['median_half_thickness_mm']:>6.2f} "
                f"{r['frac_within_threshold']*100:>6.1f} {r['n_components']:>5d}"
            )
        if group_label == "STUBBORN":
            print("-" * len(header))

    # EDT percentile table
    print("\n\nEDT PERCENTILE DISTRIBUTION (mm from skull boundary)")
    print("  (Higher values = thicker skull regions)")
    pct_header = (
        f"{'Subj':<8} {'Group':<10} "
        f"{'P10':>6} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6} {'P95':>6} {'P99':>6}"
    )
    print(pct_header)
    print("-" * len(pct_header))
    for group_data in [stubborn, improved]:
        for r in group_data:
            p = r["edt_pct"]
            print(
                f"{r['subj']:<8} {r['group']:<10} "
                f"{p[0]:>6.2f} {p[1]:>6.2f} {p[2]:>6.2f} {p[3]:>6.2f} "
                f"{p[4]:>6.2f} {p[5]:>6.2f} {p[6]:>6.2f}"
            )
        print("-" * len(pct_header))

    # Trabecular depth stats
    print("\n\nTRABECULAR VOXEL DEPTH (distance from skull boundary, mm)")
    trab_header = (
        f"{'Subj':<8} {'Group':<10} "
        f"{'MeanDist':>8} {'MedianD':>8} {'MaxDist':>8} "
        f"{'N_trab':>10} {'TrabVol(cc)':>12}"
    )
    print(trab_header)
    print("-" * len(trab_header))
    for group_data in [stubborn, improved]:
        for r in group_data:
            print(
                f"{r['subj']:<8} {r['group']:<10} "
                f"{r['trab_mean_dist_mm']:>8.2f} {r['trab_median_dist_mm']:>8.2f} "
                f"{r['trab_max_dist_mm']:>8.2f} "
                f"{r['n_trabecular']:>10,d} {r['n_trabecular']*r['voxel_vol_mm3']/1000:>12.2f}"
            )
        print("-" * len(trab_header))

    # Z-axis raycast thickness
    print("\n\nZ-AXIS RAYCAST SKULL THICKNESS (mm, per-column measurement)")
    z_header = (
        f"{'Subj':<8} {'Group':<10} "
        f"{'Mean':>7} {'Median':>7} {'P90':>7} {'P95':>7} {'P99':>7} {'Max':>7}"
    )
    print(z_header)
    print("-" * len(z_header))
    for group_data in [stubborn, improved]:
        for r in group_data:
            print(
                f"{r['subj']:<8} {r['group']:<10} "
                f"{r['z_thickness_mean_mm']:>7.2f} {r['z_thickness_median_mm']:>7.2f} "
                f"{r['z_thickness_p90_mm']:>7.2f} {r['z_thickness_p95_mm']:>7.2f} "
                f"{r['z_thickness_p99_mm']:>7.2f} {r['z_thickness_max_mm']:>7.2f}"
            )
        print("-" * len(z_header))

    # Connected component details
    print("\n\nCONNECTED COMPONENT ANALYSIS")
    cc_header = f"{'Subj':<8} {'Group':<10} {'N_comp':>6} {'LargestFrac':>12} {'Top3 sizes':>30}"
    print(cc_header)
    print("-" * len(cc_header))
    for group_data in [stubborn, improved]:
        for r in group_data:
            top3_str = ", ".join(f"{s:,d}" for s in r["top3_components"])
            print(
                f"{r['subj']:<8} {r['group']:<10} "
                f"{r['n_components']:>6d} {r['largest_component_frac']:>12.4f} "
                f"{top3_str:>30}"
            )
        print("-" * len(cc_header))

    # Summary statistics
    print("\n\n" + "=" * 80)
    print("GROUP SUMMARY STATISTICS")
    print("=" * 80)

    for group_label, group_data in [("STUBBORN", stubborn), ("IMPROVED", improved)]:
        print(f"\n--- {group_label} (n={len(group_data)}) ---")
        metrics = {
            "Skull volume (cc)": [r["skull_vol_mm3"] / 1000 for r in group_data],
            "% Cortical": [r["pct_cortical"] for r in group_data],
            "% Trabecular": [r["pct_trabecular"] for r in group_data],
            "Max half-thickness (mm)": [r["max_half_thickness_mm"] for r in group_data],
            "Mean half-thickness (mm)": [r["mean_half_thickness_mm"] for r in group_data],
            "% within 2.5mm threshold": [r["frac_within_threshold"] * 100 for r in group_data],
            "Z-axis mean thickness (mm)": [r["z_thickness_mean_mm"] for r in group_data],
            "Z-axis P95 thickness (mm)": [r["z_thickness_p95_mm"] for r in group_data],
            "Trabecular mean depth (mm)": [r["trab_mean_dist_mm"] for r in group_data],
            "N components": [r["n_components"] for r in group_data],
        }
        for name, vals in metrics.items():
            arr = np.array(vals)
            print(f"  {name:<32s}: mean={arr.mean():.2f}, std={arr.std():.2f}, "
                  f"min={arr.min():.2f}, max={arr.max():.2f}")

    # GU035 deep dive
    print("\n\n" + "=" * 80)
    print("GU035 DEEP DIVE (skull_path_near=19mm, basically no improvement)")
    print("=" * 80)
    gu035 = [r for r in results if r["subj"] == "GU035"][0]
    print(f"  Voxel spacing: {gu035['spacing']}")
    print(f"  Total skull voxels: {gu035['n_skull']:,d}")
    print(f"  Skull volume: {gu035['skull_vol_mm3']/1000:.1f} cc")
    print(f"  Cortical voxels: {gu035['n_cortical']:,d} ({gu035['pct_cortical']:.1f}%)")
    print(f"  Trabecular voxels: {gu035['n_trabecular']:,d} ({gu035['pct_trabecular']:.1f}%)")
    print(f"  Max distance from boundary (half-thickness): {gu035['max_half_thickness_mm']:.2f} mm")
    print(f"  Mean distance from boundary: {gu035['mean_half_thickness_mm']:.2f} mm")
    print(f"  Fraction of skull within cortical threshold: {gu035['frac_within_threshold']*100:.1f}%")
    print(f"  Connected components: {gu035['n_components']}")
    print(f"  Largest component fraction: {gu035['largest_component_frac']:.4f}")
    print(f"  Z-axis thickness: mean={gu035['z_thickness_mean_mm']:.2f}, "
          f"P95={gu035['z_thickness_p95_mm']:.2f}, max={gu035['z_thickness_max_mm']:.2f} mm")

    # Check if GU035 has abnormally few trabecular voxels
    all_trab_pcts = [r["pct_trabecular"] for r in results]
    gu035_rank = sorted(all_trab_pcts).index(gu035["pct_trabecular"]) + 1
    print(f"\n  GU035 trabecular % rank: {gu035_rank}/{len(results)} "
          f"(1=least trabecular)")

    # Interpretation
    print("\n\nINTERPRETATION")
    print("-" * 80)
    if gu035["frac_within_threshold"] > 0.90:
        print("  GU035: >90% of skull is within 2.5mm of boundary.")
        print("  -> Skull is predominantly THIN. Very little gets relabeled as trabecular.")
        print("  -> Two-class model has almost nothing to change.")
    if gu035["pct_trabecular"] < 10:
        print(f"  GU035: Only {gu035['pct_trabecular']:.1f}% trabecular.")
        print("  -> The 2.5mm cortical threshold captures nearly the entire skull.")
        print("  -> Attenuation is dominated by cortical bone regardless of model.")

    # Check correlation between trabecular fraction and improvement
    print("\n\nCORRELATION: Trabecular fraction vs Improvement")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["pct_trabecular"]):
        s = r["subj"]
        a1, a2 = ATTEN.get(s, (0, 0))
        delta = a2 - a1
        skull_path = SKULL_PATH_NEAR.get(s, "N/A")
        print(
            f"  {s:<8} {r['group']:<10} %trab={r['pct_trabecular']:>5.1f}  "
            f"delta={delta:>+6.2f} dB  skull_path={skull_path}"
        )


def main():
    print("Loading and analyzing skull morphology for 11 outlier subjects...")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Cortical thickness threshold: {CORTICAL_THICKNESS_MM} mm")
    print()

    results = []
    for subj in SUBJECTS:
        print(f"  Processing {subj}...", flush=True)
        try:
            r = analyze_subject(subj)
            results.append(r)
            print(f"    skull={r['n_skull']:,d} vox, "
                  f"cort={r['pct_cortical']:.1f}%, trab={r['pct_trabecular']:.1f}%, "
                  f"maxHT={r['max_half_thickness_mm']:.2f}mm, "
                  f"comps={r['n_components']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_table(results)


if __name__ == "__main__":
    main()
