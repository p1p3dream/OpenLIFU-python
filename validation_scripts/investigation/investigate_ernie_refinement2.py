"""Deeper investigation: Why does refinement hurt Ernie's shell Dice so much?

The key insight from the first investigation:
- The unrefined shell (1.69M voxels) captures 893K GT scalp voxels (53%)
- Refinement removes 567K voxels (34% of shell), keeping 1.12M
- But the Otsu threshold (7451) cuts roughly at the boundary between bone and scalp
- 30.8% of GT scalp in shell is BELOW Otsu (classified as bone) -- these are misclassified
- But the real problem: removing scalp HURTS shell Dice because shell = bone+scalp

The fundamental issue: on Ernie, the EDT shell (foreground minus brain) IS the bone+scalp
region. Refinement removes the scalp portion, which is correct anatomically, but the
Dice metric we're tracking is bone+scalp, so removal hurts.

On Birnbaum, the improvement might be because: the shell is less aligned with
bone+scalp, and refinement removes non-skull tissue that was being misclassified.

Let's also check: what fraction of the GT bone is NOT in the EDT shell at all?
That's the fundamental coverage gap.
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
import xarray as xa

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

from openlifu.seg.seg_methods.threshold_mri import compute_foreground_mask


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def load_ernie():
    base = os.path.expanduser("~/Data/openlifu-validation/datasets/ernie/m2m_ernie")
    t1 = nib.load(os.path.join(base, "T1.nii.gz"))
    gt = nib.load(os.path.join(base, "final_tissues.nii.gz"))
    t1_data = t1.get_fdata().astype(np.float32)
    gt_data = gt.get_fdata().astype(int)
    if gt_data.ndim == 4:
        gt_data = gt_data[:, :, :, 0]
    zooms = t1.header.get_zooms()[:3]
    return t1_data, gt_data, zooms


def load_birnbaum_first():
    """Load first Birnbaum subject for comparison."""
    base = os.path.expanduser("~/Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects")
    t1_path = os.path.join(base, "T1-Weighted MRI", "GU002_deface.nii")
    gt_path = os.path.join(base, "Full-Head Segmentation", "GU002_label_deface.nii")
    t1 = nib.load(t1_path)
    gt = nib.load(gt_path)
    t1_data = t1.get_fdata().astype(np.float32)
    gt_data = gt.get_fdata().astype(int)
    if gt_data.ndim == 4:
        gt_data = gt_data[:, :, :, 0]
    zooms = t1.header.get_zooms()[:3]
    return t1_data, gt_data, zooms


def analyze_dataset(name, t1_data, gt_data, zooms, gt_bone_label, gt_scalp_label, gt_brain_labels, skull_mm=12.0):
    print(f"\n{'='*70}")
    print(f"Dataset: {name} at {skull_mm}mm")
    print(f"{'='*70}")

    spacing = np.array(zooms, dtype=float)
    foreground = compute_foreground_mask(t1_data)
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)
    brain_mask = foreground_dist > skull_mm
    shell_mask = foreground & ~brain_mask

    gt_bone = gt_data == gt_bone_label
    gt_scalp = gt_data == gt_scalp_label
    gt_shell = gt_bone | gt_scalp
    gt_brain = np.isin(gt_data, gt_brain_labels)

    # Coverage analysis
    gt_bone_in_shell = gt_bone & shell_mask
    gt_bone_not_in_shell = gt_bone & ~shell_mask
    gt_bone_outside_fg = gt_bone & ~foreground
    gt_bone_in_brain = gt_bone & brain_mask

    print(f"  GT bone total:           {int(gt_bone.sum()):,}")
    print(f"  GT bone in shell:        {int(gt_bone_in_shell.sum()):,} ({gt_bone_in_shell.sum()/gt_bone.sum():.1%})")
    print(f"  GT bone in brain mask:   {int(gt_bone_in_brain.sum()):,} ({gt_bone_in_brain.sum()/gt_bone.sum():.1%})")
    print(f"  GT bone outside fg:      {int(gt_bone_outside_fg.sum()):,} ({gt_bone_outside_fg.sum()/gt_bone.sum():.1%})")

    gt_scalp_in_shell = gt_scalp & shell_mask
    print(f"\n  GT scalp total:          {int(gt_scalp.sum()):,}")
    print(f"  GT scalp in shell:       {int(gt_scalp_in_shell.sum()):,} ({gt_scalp_in_shell.sum()/gt_scalp.sum():.1%})")

    print(f"\n  Shell total:             {int(shell_mask.sum()):,}")
    print(f"  Shell that is GT bone:   {int(gt_bone_in_shell.sum()):,} ({gt_bone_in_shell.sum()/shell_mask.sum():.1%})")
    print(f"  Shell that is GT scalp:  {int(gt_scalp_in_shell.sum()):,} ({gt_scalp_in_shell.sum()/shell_mask.sum():.1%})")
    shell_other = shell_mask & ~gt_bone & ~gt_scalp
    print(f"  Shell that is other:     {int(shell_other.sum()):,} ({shell_other.sum()/shell_mask.sum():.1%})")

    # Intensity analysis of the shell
    shell_intensities = t1_data[shell_mask]
    nonzero_shell = shell_intensities[shell_intensities > 0]

    # Intensity analysis split by GT label within the shell
    bone_in_shell_vals = t1_data[gt_bone_in_shell]
    scalp_in_shell_vals = t1_data[gt_scalp_in_shell]

    bone_nz = bone_in_shell_vals[bone_in_shell_vals > 0]
    scalp_nz = scalp_in_shell_vals[scalp_in_shell_vals > 0]

    print(f"\n  Intensity in shell by GT label:")
    if bone_nz.size > 0:
        print(f"    Bone:  mean={np.mean(bone_nz):.1f}, median={np.median(bone_nz):.1f}, P25={np.percentile(bone_nz,25):.1f}, P75={np.percentile(bone_nz,75):.1f}")
    if scalp_nz.size > 0:
        print(f"    Scalp: mean={np.mean(scalp_nz):.1f}, median={np.median(scalp_nz):.1f}, P25={np.percentile(scalp_nz,25):.1f}, P75={np.percentile(scalp_nz,75):.1f}")

    # Otsu threshold
    if nonzero_shell.size > 100:
        otsu = threshold_otsu(nonzero_shell)
        print(f"\n  Otsu threshold: {otsu:.1f}")

        # What Otsu does to GT bone in shell
        if bone_nz.size > 0:
            bone_kept = np.sum(bone_nz < otsu)
            bone_removed = np.sum(bone_nz >= otsu)
            print(f"  GT bone in shell: {int(bone_kept):,} kept ({bone_kept/bone_nz.size:.1%}), {int(bone_removed):,} removed ({bone_removed/bone_nz.size:.1%})")

        if scalp_nz.size > 0:
            scalp_kept = np.sum(scalp_nz < otsu)
            scalp_removed = np.sum(scalp_nz >= otsu)
            print(f"  GT scalp in shell: {int(scalp_kept):,} mis-kept ({scalp_kept/scalp_nz.size:.1%}), {int(scalp_removed):,} correctly removed ({scalp_removed/scalp_nz.size:.1%})")

    # Intensity overlap: what fraction of bone intensities overlap with scalp intensities?
    if bone_nz.size > 100 and scalp_nz.size > 100:
        bone_p75 = np.percentile(bone_nz, 75)
        scalp_p25 = np.percentile(scalp_nz, 25)
        print(f"\n  Bone P75: {bone_p75:.1f}, Scalp P25: {scalp_p25:.1f}")
        if bone_p75 > scalp_p25:
            overlap_zone = f"[{scalp_p25:.0f}, {bone_p75:.0f}]"
            bone_in_overlap = np.sum((bone_nz >= scalp_p25) & (bone_nz <= bone_p75))
            scalp_in_overlap = np.sum((scalp_nz >= scalp_p25) & (scalp_nz <= bone_p75))
            print(f"  Overlap zone: {overlap_zone}")
            print(f"    Bone voxels in overlap:  {int(bone_in_overlap):,} ({bone_in_overlap/bone_nz.size:.1%})")
            print(f"    Scalp voxels in overlap: {int(scalp_in_overlap):,} ({scalp_in_overlap/scalp_nz.size:.1%})")
        else:
            print(f"  Good separation! No overlap between bone P75 and scalp P25.")


def main():
    print("Loading datasets...")
    ernie_t1, ernie_gt, ernie_zooms = load_ernie()
    birn_t1, birn_gt, birn_zooms = load_birnbaum_first()

    # Ernie: bone=4, scalp=5, brain=1,2,3
    analyze_dataset("Ernie", ernie_t1, ernie_gt, ernie_zooms,
                    gt_bone_label=4, gt_scalp_label=5, gt_brain_labels=[1, 2, 3],
                    skull_mm=12.0)

    # Birnbaum: bone=5, scalp/soft=6, brain=2,3,4
    analyze_dataset("Birnbaum GU002", birn_t1, birn_gt, birn_zooms,
                    gt_bone_label=5, gt_scalp_label=6, gt_brain_labels=[2, 3, 4],
                    skull_mm=12.0)

    print(f"\n{'='*70}")
    print("KEY QUESTION: Where does the Ernie GT bone live relative to the EDT shell?")
    print("The bone that's NOT in the shell is the fundamental coverage limit.")
    print(f"{'='*70}")

    # For Ernie, check the distance transform values for GT bone
    spacing = np.array(ernie_zooms, dtype=float)
    foreground = compute_foreground_mask(ernie_t1)
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)

    gt_bone = ernie_gt == 4
    bone_distances = foreground_dist[gt_bone]
    print(f"\n  GT bone distance-from-surface distribution:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p:02d}: {np.percentile(bone_distances, p):.1f} mm")
    print(f"    Mean: {np.mean(bone_distances):.1f} mm")

    # At 12mm threshold: bone with dist > 12 is in the brain mask (lost)
    lost_at_12 = np.sum(bone_distances > 12.0)
    print(f"\n  GT bone with distance > 12mm (lost to brain mask): {int(lost_at_12):,} ({lost_at_12/gt_bone.sum():.1%})")
    lost_at_15 = np.sum(bone_distances > 15.0)
    print(f"  GT bone with distance > 15mm (lost at 15mm):        {int(lost_at_15):,} ({lost_at_15/gt_bone.sum():.1%})")
    in_fg = np.sum(gt_bone & foreground)
    print(f"  GT bone in foreground mask:                          {int(in_fg):,} ({in_fg/gt_bone.sum():.1%})")

    # For Birnbaum, same analysis
    spacing_b = np.array(birn_zooms, dtype=float)
    foreground_b = compute_foreground_mask(birn_t1)
    foreground_dist_b = distance_transform_edt(foreground_b, sampling=spacing_b)

    gt_bone_b = birn_gt == 5
    bone_distances_b = foreground_dist_b[gt_bone_b]
    print(f"\n  Birnbaum GT bone distance-from-surface distribution:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p:02d}: {np.percentile(bone_distances_b, p):.1f} mm")
    lost_at_12_b = np.sum(bone_distances_b > 12.0)
    print(f"\n  Birnbaum GT bone with distance > 12mm: {int(lost_at_12_b):,} ({lost_at_12_b/gt_bone_b.sum():.1%})")


if __name__ == "__main__":
    main()
