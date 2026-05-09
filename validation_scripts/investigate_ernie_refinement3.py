"""Third investigation: LCC and reclassification effects on Ernie.

Check if the largest-connected-component step is discarding valid bone
fragments. Also check what the "30% depth" reclassification does.
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


def main():
    base = os.path.expanduser("~/Data/openlifu-validation/datasets/ernie/m2m_ernie")
    t1 = nib.load(os.path.join(base, "T1.nii.gz"))
    gt = nib.load(os.path.join(base, "final_tissues.nii.gz"))
    t1_data = t1.get_fdata().astype(np.float32)
    gt_data = gt.get_fdata().astype(int)
    if gt_data.ndim == 4:
        gt_data = gt_data[:, :, :, 0]
    zooms = t1.header.get_zooms()[:3]
    spacing = np.array(zooms, dtype=float)

    gt_bone = gt_data == 4
    gt_shell = np.isin(gt_data, [4, 5])

    foreground = compute_foreground_mask(t1_data)
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)

    skull_mm = 12.0
    brain_mask = foreground_dist > skull_mm
    shell_mask = foreground & ~brain_mask

    shell_intensities = t1_data[shell_mask]
    nonzero_shell = shell_intensities[shell_intensities > 0]
    otsu = threshold_otsu(nonzero_shell)

    # Step-by-step refinement with metrics at each stage
    print("Step-by-step refinement with Dice at each stage:")
    print(f"  {'Step':<40} {'Voxels':>10} {'Bone Dice':>10} {'Shell Dice':>10}")
    print(f"  {'-'*70}")

    # Stage 0: unrefined shell
    print(f"  {'Unrefined shell':<40} {int(shell_mask.sum()):>10,} {dice(shell_mask, gt_bone):>10.4f} {dice(shell_mask, gt_shell):>10.4f}")

    # Stage 1: After Otsu
    bone_after_otsu = shell_mask & (t1_data < otsu)
    print(f"  {'After Otsu (data < {:.0f})':<40} {int(bone_after_otsu.sum()):>10,} {dice(bone_after_otsu, gt_bone):>10.4f} {dice(bone_after_otsu, gt_shell):>10.4f}".format(otsu))

    # Stage 2: After closing
    closing_voxels = max(1, int(2.0 / np.min(spacing)))
    bone_after_closing = binary_closing(bone_after_otsu, iterations=closing_voxels)
    bone_after_closing = bone_after_closing & shell_mask
    print(f"  {'After closing ({} iters)':<40} {int(bone_after_closing.sum()):>10,} {dice(bone_after_closing, gt_bone):>10.4f} {dice(bone_after_closing, gt_shell):>10.4f}".format(closing_voxels))

    # Stage 3: After LCC
    labeled_arr, n_features = ndlabel(bone_after_closing)
    sizes = np.bincount(labeled_arr.ravel())[1:]
    bone_after_lcc = labeled_arr == (np.argmax(sizes) + 1)
    print(f"  {'After LCC ({} -> 1 component)':<40} {int(bone_after_lcc.sum()):>10,} {dice(bone_after_lcc, gt_bone):>10.4f} {dice(bone_after_lcc, gt_shell):>10.4f}".format(n_features))

    # How many components were discarded?
    discarded_voxels = int(bone_after_closing.sum()) - int(bone_after_lcc.sum())
    print(f"\n  LCC discarded {discarded_voxels:,} voxels ({discarded_voxels/bone_after_closing.sum():.1%}) in {n_features - 1} smaller components")

    # What GT labels do the discarded voxels overlap with?
    discarded_mask = bone_after_closing & ~bone_after_lcc
    disc_bone = np.sum(discarded_mask & gt_bone)
    disc_scalp = np.sum(discarded_mask & (gt_data == 5))
    disc_brain = np.sum(discarded_mask & np.isin(gt_data, [1,2,3]))
    disc_bg = np.sum(discarded_mask & (gt_data == 0))
    print(f"  Discarded voxels by GT label:")
    print(f"    Bone:  {int(disc_bone):,}")
    print(f"    Scalp: {int(disc_scalp):,}")
    print(f"    Brain: {int(disc_brain):,}")
    print(f"    BG:    {int(disc_bg):,}")

    # Stage 4: After reclassification (what the code does)
    # Non-bone shell voxels: those far enough from surface (>30% of skull_mm)
    # are reclassified as brain
    shell_not_bone = shell_mask & ~bone_after_lcc
    reclassified_to_brain = shell_not_bone & (foreground_dist > skull_mm * 0.3)
    remaining_as_water = shell_not_bone & ~reclassified_to_brain
    print(f"\n  Reclassification of non-bone shell voxels:")
    print(f"    Total non-bone shell: {int(shell_not_bone.sum()):,}")
    print(f"    Reclassified to brain (dist > {skull_mm * 0.3:.1f}mm): {int(reclassified_to_brain.sum()):,}")
    print(f"    Left as water/scalp (dist <= {skull_mm * 0.3:.1f}mm):  {int(remaining_as_water.sum()):,}")

    # Final skull mask = bone_after_lcc
    # Final brain mask = original brain_mask | reclassified_to_brain
    final_brain = brain_mask | reclassified_to_brain
    print(f"\n  Brain mask: {int(brain_mask.sum()):,} -> {int(final_brain.sum()):,} (+{int(reclassified_to_brain.sum()):,})")

    # Now the key insight: on Birnbaum, what does the same analysis look like?
    print(f"\n{'='*70}")
    print("Same analysis on Birnbaum GU002:")
    print(f"{'='*70}")

    birn_base = os.path.expanduser("~/Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects")
    birn_t1 = nib.load(os.path.join(birn_base, "T1-Weighted MRI", "GU002_deface.nii")).get_fdata().astype(np.float32)
    birn_gt = nib.load(os.path.join(birn_base, "Full-Head Segmentation", "GU002_label_deface.nii")).get_fdata().astype(int)
    if birn_gt.ndim == 4:
        birn_gt = birn_gt[:, :, :, 0]
    birn_zooms = nib.load(os.path.join(birn_base, "T1-Weighted MRI", "GU002_deface.nii")).header.get_zooms()[:3]
    birn_spacing = np.array(birn_zooms, dtype=float)

    birn_gt_bone = birn_gt == 5
    birn_gt_shell = np.isin(birn_gt, [5, 6])

    birn_fg = compute_foreground_mask(birn_t1)
    birn_fdist = distance_transform_edt(birn_fg, sampling=birn_spacing)
    birn_brain = birn_fdist > skull_mm
    birn_shell = birn_fg & ~birn_brain

    birn_shell_int = birn_t1[birn_shell]
    birn_nz = birn_shell_int[birn_shell_int > 0]
    birn_otsu = threshold_otsu(birn_nz)

    print(f"  {'Step':<40} {'Voxels':>10} {'Bone Dice':>10} {'Shell Dice':>10}")
    print(f"  {'-'*70}")

    print(f"  {'Unrefined shell':<40} {int(birn_shell.sum()):>10,} {dice(birn_shell, birn_gt_bone):>10.4f} {dice(birn_shell, birn_gt_shell):>10.4f}")

    birn_bone_otsu = birn_shell & (birn_t1 < birn_otsu)
    print(f"  {'After Otsu (data < {:.0f})':<40} {int(birn_bone_otsu.sum()):>10,} {dice(birn_bone_otsu, birn_gt_bone):>10.4f} {dice(birn_bone_otsu, birn_gt_shell):>10.4f}".format(birn_otsu))

    birn_closing = max(1, int(2.0 / np.min(birn_spacing)))
    birn_bone_closed = binary_closing(birn_bone_otsu, iterations=birn_closing)
    birn_bone_closed = birn_bone_closed & birn_shell
    print(f"  {'After closing ({} iters)':<40} {int(birn_bone_closed.sum()):>10,} {dice(birn_bone_closed, birn_gt_bone):>10.4f} {dice(birn_bone_closed, birn_gt_shell):>10.4f}".format(birn_closing))

    birn_labeled, birn_nf = ndlabel(birn_bone_closed)
    birn_sizes = np.bincount(birn_labeled.ravel())[1:]
    birn_bone_lcc = birn_labeled == (np.argmax(birn_sizes) + 1)
    print(f"  {'After LCC ({} -> 1 component)':<40} {int(birn_bone_lcc.sum()):>10,} {dice(birn_bone_lcc, birn_gt_bone):>10.4f} {dice(birn_bone_lcc, birn_gt_shell):>10.4f}".format(birn_nf))

    birn_discarded = int(birn_bone_closed.sum()) - int(birn_bone_lcc.sum())
    print(f"\n  LCC discarded {birn_discarded:,} voxels ({birn_discarded/birn_bone_closed.sum():.1%}) in {birn_nf - 1} components")


if __name__ == "__main__":
    main()
