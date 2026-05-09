"""Test potential fixes for the refinement regression.

Fix candidates:
1. Reduce closing iterations (2 -> 1 or 0)
2. Use a percentile-based threshold instead of Otsu
3. Add a "scalp dominance" guard that skips refinement when the shell
   is mostly scalp
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.ndimage import label as ndlabel
from skimage.filters import threshold_otsu

from openlifu.seg.seg_methods.threshold_mri import compute_foreground_mask


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def run_refinement_variant(t1_data, zooms, skull_mm, closing_iters, use_percentile_threshold=False, percentile=None, skip_lcc=False):
    """Run refinement with customizable parameters and return the bone mask."""
    spacing = np.array(zooms, dtype=float)
    foreground = compute_foreground_mask(t1_data)
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)
    brain_mask = foreground_dist > skull_mm
    shell_mask = foreground & ~brain_mask

    shell_intensities = t1_data[shell_mask]
    nonzero_shell = shell_intensities[shell_intensities > 0]

    if use_percentile_threshold and percentile is not None:
        thresh = np.percentile(nonzero_shell, percentile)
    else:
        thresh = threshold_otsu(nonzero_shell)

    bone_mask = shell_mask & (t1_data < thresh)

    if closing_iters > 0:
        closing_voxels = max(1, int(2.0 / np.min(spacing)))
        bone_mask = binary_closing(bone_mask, iterations=closing_iters)
        bone_mask = bone_mask & shell_mask

    if not skip_lcc:
        labeled_arr, n_features = ndlabel(bone_mask)
        if n_features > 0:
            sizes = np.bincount(labeled_arr.ravel())[1:]
            bone_mask = labeled_arr == (np.argmax(sizes) + 1)

    return bone_mask, shell_mask, thresh


def load_and_analyze(name, t1_data, gt_data, zooms, gt_bone_label, gt_scalp_label, skull_mm=12.0):
    gt_bone = gt_data == gt_bone_label
    gt_shell = (gt_data == gt_bone_label) | (gt_data == gt_scalp_label)

    print(f"\n{'='*70}")
    print(f"  {name} at {skull_mm}mm -- Variant Comparison")
    print(f"{'='*70}")
    print(f"  {'Variant':<50} {'Bone Dice':>10} {'Shell Dice':>10}")
    print(f"  {'-'*70}")

    # Baseline: no refinement (= the unrefined shell)
    spacing = np.array(zooms, dtype=float)
    foreground = compute_foreground_mask(t1_data)
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)
    brain_mask = foreground_dist > skull_mm
    shell_mask = foreground & ~brain_mask
    print(f"  {'No refinement (shell only)':<50} {dice(shell_mask, gt_bone):>10.4f} {dice(shell_mask, gt_shell):>10.4f}")

    # Current implementation: Otsu + 2-iter closing + LCC
    bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=2)
    print(f"  {'Current: Otsu + 2-closing + LCC':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")

    # Variant 1: Otsu + 1-iter closing + LCC
    bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=1)
    print(f"  {'Otsu + 1-closing + LCC':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")

    # Variant 2: Otsu + 0 closing + LCC
    bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=0)
    print(f"  {'Otsu + no closing + LCC':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")

    # Variant 3: Otsu + 0 closing + no LCC
    bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=0, skip_lcc=True)
    print(f"  {'Otsu + no closing + no LCC':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")

    # Variant 4: P30 percentile threshold + 2-closing + LCC
    bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=2, use_percentile_threshold=True, percentile=30)
    print(f"  {'P30 threshold + 2-closing + LCC':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")

    # Variant 5: P40 percentile threshold + 2-closing + LCC
    bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=2, use_percentile_threshold=True, percentile=40)
    print(f"  {'P40 threshold + 2-closing + LCC':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")

    # Variant 6: P25 percentile (conservative: only keep darkest 25%)
    bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=1, use_percentile_threshold=True, percentile=25)
    print(f"  {'P25 threshold + 1-closing + LCC':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")

    # Variant 7: Otsu + 2-closing + LCC but skip if Otsu splits shell ~50/50
    nonzero_shell_vals = t1_data[shell_mask]
    nonzero_shell_vals = nonzero_shell_vals[nonzero_shell_vals > 0]
    otsu_t = threshold_otsu(nonzero_shell_vals)
    below_frac = np.sum(nonzero_shell_vals < otsu_t) / nonzero_shell_vals.size
    print(f"\n  Otsu threshold: {otsu_t:.1f}, below fraction: {below_frac:.3f}")
    if abs(below_frac - 0.5) < 0.15:
        print(f"  -> SKIP guard would trigger (fraction {below_frac:.3f} is near 50%)")
        print(f"  {'SKIP guard: keep unrefined shell':<50} {dice(shell_mask, gt_bone):>10.4f} {dice(shell_mask, gt_shell):>10.4f}")
    else:
        print(f"  -> SKIP guard would NOT trigger (fraction {below_frac:.3f} is far from 50%)")
        bone, shell, thresh = run_refinement_variant(t1_data, zooms, skull_mm, closing_iters=2)
        print(f"  {'SKIP guard: do refinement':<50} {dice(bone, gt_bone):>10.4f} {dice(bone, gt_shell):>10.4f}")


def main():
    # Ernie
    base_e = os.path.expanduser("~/Data/openlifu-validation/datasets/ernie/m2m_ernie")
    t1_e = nib.load(os.path.join(base_e, "T1.nii.gz")).get_fdata().astype(np.float32)
    gt_e = nib.load(os.path.join(base_e, "final_tissues.nii.gz")).get_fdata().astype(int)
    if gt_e.ndim == 4:
        gt_e = gt_e[:, :, :, 0]
    zooms_e = nib.load(os.path.join(base_e, "T1.nii.gz")).header.get_zooms()[:3]

    load_and_analyze("Ernie", t1_e, gt_e, zooms_e, gt_bone_label=4, gt_scalp_label=5, skull_mm=12.0)

    # Birnbaum GU002
    birn_base = os.path.expanduser("~/Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects")
    t1_b = nib.load(os.path.join(birn_base, "T1-Weighted MRI", "GU002_deface.nii")).get_fdata().astype(np.float32)
    gt_b = nib.load(os.path.join(birn_base, "Full-Head Segmentation", "GU002_label_deface.nii")).get_fdata().astype(int)
    if gt_b.ndim == 4:
        gt_b = gt_b[:, :, :, 0]
    zooms_b = nib.load(os.path.join(birn_base, "T1-Weighted MRI", "GU002_deface.nii")).header.get_zooms()[:3]

    load_and_analyze("Birnbaum GU002", t1_b, gt_b, zooms_b, gt_bone_label=5, gt_scalp_label=6, skull_mm=12.0)


if __name__ == "__main__":
    main()
