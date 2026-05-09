"""Investigate why refine_skull_intensity hurts Ernie bone-only Dice.

Loads Ernie T1 + CHARM ground truth, runs ThresholdMRI with and without
skull intensity refinement at 12mm, and prints detailed diagnostics about
the Otsu threshold behavior and intensity distribution in the shell.
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
import xarray as xa

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.ndimage import label as ndlabel
from skimage.filters import threshold_otsu

from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI, compute_foreground_mask


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


def make_volume(t1_data, zooms):
    nz, ny, nx = t1_data.shape
    return xa.DataArray(
        t1_data, dims=["z", "y", "x"],
        coords={
            "z": np.arange(nz) * zooms[0],
            "y": np.arange(ny) * zooms[1],
            "x": np.arange(nx) * zooms[2],
        },
    )


def run_segmentation(vol, skull_mm, refine):
    seg = ThresholdMRI(skull_thickness_mm=skull_mm, refine_skull_intensity=refine)
    result = seg._segment(vol)
    idx = seg._material_indices()
    return result.to_numpy(), idx


def analyze_shell_internals(t1_data, zooms, skull_mm):
    """Replicate the refinement logic step-by-step and print diagnostics."""
    data = t1_data.copy()
    spacing = np.array(zooms, dtype=float)

    # Step 1: Foreground mask
    foreground = compute_foreground_mask(data)

    # Step 2: EDT-based brain mask
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)
    brain_mask = foreground_dist > skull_mm

    # Step 3: Shell
    shell_mask = foreground & ~brain_mask
    shell_intensities = data[shell_mask]

    print(f"\n{'='*70}")
    print(f"Shell analysis for skull_thickness_mm = {skull_mm}")
    print(f"{'='*70}")
    print(f"  Foreground voxels: {int(np.sum(foreground)):,}")
    print(f"  Brain mask voxels: {int(np.sum(brain_mask)):,}")
    print(f"  Shell voxels:      {int(np.sum(shell_mask)):,}")

    # Intensity distribution in the shell
    nonzero_shell = shell_intensities[shell_intensities > 0]
    print(f"\n  Shell intensity stats (non-zero):")
    print(f"    Count:  {nonzero_shell.size:,}")
    print(f"    Mean:   {np.mean(nonzero_shell):.1f}")
    print(f"    Std:    {np.std(nonzero_shell):.1f}")
    shell_mean = float(np.mean(nonzero_shell))
    shell_cv = float(np.std(nonzero_shell)) / shell_mean if shell_mean > 0 else 0
    print(f"    CV:     {shell_cv:.4f}")
    print(f"    Min:    {np.min(nonzero_shell):.1f}")
    print(f"    Max:    {np.max(nonzero_shell):.1f}")

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\n  Percentiles of non-zero shell intensities:")
    for p in percentiles:
        val = np.percentile(nonzero_shell, p)
        print(f"    P{p:02d}: {val:.1f}")

    # Otsu threshold
    otsu_thresh = threshold_otsu(nonzero_shell)
    print(f"\n  Otsu threshold: {otsu_thresh:.1f}")

    below = np.sum(nonzero_shell < otsu_thresh)
    above = np.sum(nonzero_shell >= otsu_thresh)
    print(f"  Voxels below Otsu (bone candidates): {int(below):,} ({below/nonzero_shell.size:.1%})")
    print(f"  Voxels above Otsu (non-bone):        {int(above):,} ({above/nonzero_shell.size:.1%})")

    # Apply Otsu to shell
    bone_mask = shell_mask & (data < otsu_thresh)
    print(f"\n  Bone mask after Otsu (in shell):  {int(np.sum(bone_mask)):,}")

    # Morphological closing
    closing_voxels = max(1, int(2.0 / np.min(spacing)))
    bone_closed = binary_closing(bone_mask, iterations=closing_voxels)
    bone_closed = bone_closed & shell_mask
    print(f"  After closing ({closing_voxels} iters):      {int(np.sum(bone_closed)):,}")

    # Largest connected component
    labeled_arr, n_features = ndlabel(bone_closed)
    if n_features > 0:
        sizes = np.bincount(labeled_arr.ravel())[1:]
        bone_final = labeled_arr == (np.argmax(sizes) + 1)
        print(f"  After LCC (of {n_features} components): {int(np.sum(bone_final)):,}")
    else:
        bone_final = bone_closed
        print(f"  No connected components found!")

    removed = int(np.sum(shell_mask)) - int(np.sum(bone_final))
    removed_frac = removed / int(np.sum(shell_mask))
    print(f"\n  Voxels removed from shell:   {removed:,} ({removed_frac:.1%})")
    print(f"  Voxels kept as bone:         {int(np.sum(bone_final)):,}")

    return bone_final, shell_mask, foreground_dist, otsu_thresh, brain_mask, foreground


def main():
    print("Loading Ernie dataset...")
    t1_data, gt_data, zooms = load_ernie()
    vol = make_volume(t1_data, zooms)

    print(f"T1 shape: {t1_data.shape}, zooms: {zooms}")
    print(f"T1 range: [{t1_data.min():.1f}, {t1_data.max():.1f}]")
    print(f"GT labels present: {np.unique(gt_data)}")

    gt_bone = gt_data == 4
    gt_scalp = gt_data == 5
    gt_shell = np.isin(gt_data, [4, 5])
    gt_brain = np.isin(gt_data, [1, 2, 3])

    print(f"\nGround truth:")
    print(f"  Bone (label 4):   {int(gt_bone.sum()):,} voxels")
    print(f"  Scalp (label 5):  {int(gt_scalp.sum()):,} voxels")
    print(f"  Shell (4+5):      {int(gt_shell.sum()):,} voxels")
    print(f"  Brain (1+2+3):    {int(gt_brain.sum()):,} voxels")

    # ---- What does bone look like intensity-wise? ----
    bone_intensities = t1_data[gt_bone]
    scalp_intensities = t1_data[gt_scalp]
    brain_intensities = t1_data[gt_brain]

    print(f"\nIntensity stats by GT label:")
    for name, intensities in [("Bone (4)", bone_intensities), ("Scalp (5)", scalp_intensities), ("Brain (1+2+3)", brain_intensities)]:
        nz = intensities[intensities > 0]
        print(f"  {name}:")
        print(f"    Mean: {np.mean(nz):.1f}, Std: {np.std(nz):.1f}, CV: {np.std(nz)/np.mean(nz):.3f}")
        for p in [5, 25, 50, 75, 95]:
            print(f"    P{p:02d}: {np.percentile(nz, p):.1f}")

    # ---- Run without refinement ----
    print("\n" + "="*70)
    print("Running ThresholdMRI at 12mm WITHOUT refinement...")
    print("="*70)
    seg_no_refine, idx = run_segmentation(vol, 12.0, refine=False)
    skull_no = seg_no_refine == idx["skull"]

    print(f"  Skull voxels: {int(skull_no.sum()):,}")
    print(f"  Dice vs gt_bone:  {dice(skull_no, gt_bone):.4f}")
    print(f"  Dice vs gt_shell: {dice(skull_no, gt_shell):.4f}")

    # ---- Run with refinement ----
    print("\n" + "="*70)
    print("Running ThresholdMRI at 12mm WITH refinement...")
    print("="*70)
    seg_refine, idx2 = run_segmentation(vol, 12.0, refine=True)
    skull_yes = seg_refine == idx2["skull"]

    print(f"  Skull voxels: {int(skull_yes.sum()):,}")
    print(f"  Dice vs gt_bone:  {dice(skull_yes, gt_bone):.4f}")
    print(f"  Dice vs gt_shell: {dice(skull_yes, gt_shell):.4f}")

    # ---- Detailed shell analysis at 12mm ----
    bone_final_12, shell_12, fdist_12, otsu_12, brain_12, fg_12 = analyze_shell_internals(t1_data, zooms, 12.0)

    # ---- Compare refined bone with GT ----
    print(f"\n{'='*70}")
    print("Overlap analysis: refined bone vs GT regions")
    print(f"{'='*70}")

    refined_bone = bone_final_12
    overlap_with_gt_bone = np.sum(refined_bone & gt_bone)
    overlap_with_gt_scalp = np.sum(refined_bone & gt_scalp)
    overlap_with_gt_brain = np.sum(refined_bone & gt_brain)
    overlap_with_gt_bg = np.sum(refined_bone & (gt_data == 0))
    total_refined = int(np.sum(refined_bone))

    print(f"  Refined bone total: {total_refined:,}")
    print(f"  Overlap with GT bone (4):   {int(overlap_with_gt_bone):,} ({overlap_with_gt_bone/total_refined:.1%})")
    print(f"  Overlap with GT scalp (5):  {int(overlap_with_gt_scalp):,} ({overlap_with_gt_scalp/total_refined:.1%})")
    print(f"  Overlap with GT brain (1-3): {int(overlap_with_gt_brain):,} ({overlap_with_gt_brain/total_refined:.1%})")
    print(f"  Overlap with GT bg (0):      {int(overlap_with_gt_bg):,} ({overlap_with_gt_bg/total_refined:.1%})")

    # What did the unrefined shell overlap with?
    print(f"\n  Unrefined shell total: {int(np.sum(shell_12)):,}")
    shell_overlap_bone = np.sum(shell_12 & gt_bone)
    shell_overlap_scalp = np.sum(shell_12 & gt_scalp)
    shell_overlap_brain = np.sum(shell_12 & gt_brain)
    shell_total = int(np.sum(shell_12))
    print(f"  Overlap with GT bone (4):   {int(shell_overlap_bone):,} ({shell_overlap_bone/shell_total:.1%})")
    print(f"  Overlap with GT scalp (5):  {int(shell_overlap_scalp):,} ({shell_overlap_scalp/shell_total:.1%})")
    print(f"  Overlap with GT brain (1-3): {int(shell_overlap_brain):,} ({shell_overlap_brain/shell_total:.1%})")

    # What are the intensities of the GT bone voxels that are in the shell?
    gt_bone_in_shell = gt_bone & shell_12
    gt_scalp_in_shell = gt_scalp & shell_12
    if np.sum(gt_bone_in_shell) > 0:
        bone_in_shell_vals = t1_data[gt_bone_in_shell]
        below_otsu = np.sum(bone_in_shell_vals < otsu_12)
        above_otsu = np.sum(bone_in_shell_vals >= otsu_12)
        print(f"\n  GT bone voxels within the EDT shell: {int(np.sum(gt_bone_in_shell)):,}")
        print(f"    Below Otsu ({otsu_12:.1f}): {int(below_otsu):,} ({below_otsu/np.sum(gt_bone_in_shell):.1%}) -> classified as bone")
        print(f"    Above Otsu:           {int(above_otsu):,} ({above_otsu/np.sum(gt_bone_in_shell):.1%}) -> REMOVED by refinement")
        print(f"    Intensity: mean={np.mean(bone_in_shell_vals):.1f}, P25={np.percentile(bone_in_shell_vals, 25):.1f}, P50={np.percentile(bone_in_shell_vals, 50):.1f}, P75={np.percentile(bone_in_shell_vals, 75):.1f}")

    if np.sum(gt_scalp_in_shell) > 0:
        scalp_in_shell_vals = t1_data[gt_scalp_in_shell]
        below_otsu = np.sum(scalp_in_shell_vals < otsu_12)
        above_otsu = np.sum(scalp_in_shell_vals >= otsu_12)
        print(f"\n  GT scalp voxels within the EDT shell: {int(np.sum(gt_scalp_in_shell)):,}")
        print(f"    Below Otsu ({otsu_12:.1f}): {int(below_otsu):,} ({below_otsu/np.sum(gt_scalp_in_shell):.1%}) -> classified as bone")
        print(f"    Above Otsu:           {int(above_otsu):,} ({above_otsu/np.sum(gt_scalp_in_shell):.1%}) -> correctly removed")
        print(f"    Intensity: mean={np.mean(scalp_in_shell_vals):.1f}, P25={np.percentile(scalp_in_shell_vals, 25):.1f}, P50={np.percentile(scalp_in_shell_vals, 50):.1f}, P75={np.percentile(scalp_in_shell_vals, 75):.1f}")

    # ---- Test with 15mm shell for more context ----
    print("\n" + "="*70)
    print("Testing with 15mm skull thickness (wider shell for more Otsu context)...")
    print("="*70)
    bone_final_15, shell_15, fdist_15, otsu_15, brain_15, fg_15 = analyze_shell_internals(t1_data, zooms, 15.0)

    seg_15_refine, idx15 = run_segmentation(vol, 15.0, refine=True)
    skull_15 = seg_15_refine == idx15["skull"]
    seg_15_no, idx15n = run_segmentation(vol, 15.0, refine=False)
    skull_15_no = seg_15_no == idx15n["skull"]

    print(f"\n  15mm WITHOUT refinement:")
    print(f"    Skull voxels: {int(skull_15_no.sum()):,}")
    print(f"    Dice vs gt_bone:  {dice(skull_15_no, gt_bone):.4f}")
    print(f"    Dice vs gt_shell: {dice(skull_15_no, gt_shell):.4f}")
    print(f"  15mm WITH refinement:")
    print(f"    Skull voxels: {int(skull_15.sum()):,}")
    print(f"    Dice vs gt_bone:  {dice(skull_15, gt_bone):.4f}")
    print(f"    Dice vs gt_shell: {dice(skull_15, gt_shell):.4f}")

    # ---- Ernie-specific analysis: what is the diploe like? ----
    print(f"\n{'='*70}")
    print("Diploe analysis: CHARM labels 7 (compact bone) and 8 (spongy bone)")
    print(f"{'='*70}")
    compact_bone = gt_data == 7
    spongy_bone = gt_data == 8
    print(f"  Compact bone (label 7): {int(compact_bone.sum()):,} voxels")
    print(f"  Spongy bone (label 8):  {int(spongy_bone.sum()):,} voxels")
    if np.sum(compact_bone) > 0:
        cb_vals = t1_data[compact_bone]
        print(f"  Compact bone intensity: mean={np.mean(cb_vals):.1f}, P50={np.percentile(cb_vals, 50):.1f}")
    if np.sum(spongy_bone) > 0:
        sb_vals = t1_data[spongy_bone]
        print(f"  Spongy bone intensity: mean={np.mean(sb_vals):.1f}, P50={np.percentile(sb_vals, 50):.1f}")

    # How much of the bone label is actually spongy (bright)?
    # In Ernie's CHARM, label 4 is the combined bone mask. Labels 7 and 8
    # are sub-classifications. Check if spongy bone is bright enough to be
    # above the Otsu threshold.
    all_bone_gt = gt_data == 4
    all_bone_vals = t1_data[all_bone_gt]
    above_12 = np.sum(all_bone_vals >= otsu_12)
    print(f"\n  Of all GT bone (label 4) voxels ({int(all_bone_gt.sum()):,}):")
    print(f"    Above 12mm Otsu ({otsu_12:.1f}): {int(above_12):,} ({above_12/all_bone_gt.sum():.1%})")
    print(f"    Below 12mm Otsu:            {int(all_bone_gt.sum() - above_12):,} ({1 - above_12/all_bone_gt.sum():.1%})")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  12mm, no refinement:  bone Dice = {dice(skull_no, gt_bone):.4f}, shell Dice = {dice(skull_no, gt_shell):.4f}")
    print(f"  12mm, with refinement: bone Dice = {dice(skull_yes, gt_bone):.4f}, shell Dice = {dice(skull_yes, gt_shell):.4f}")
    print(f"  15mm, no refinement:  bone Dice = {dice(skull_15_no, gt_bone):.4f}, shell Dice = {dice(skull_15_no, gt_shell):.4f}")
    print(f"  15mm, with refinement: bone Dice = {dice(skull_15, gt_bone):.4f}, shell Dice = {dice(skull_15, gt_shell):.4f}")
    print(f"\n  Otsu threshold at 12mm: {otsu_12:.1f}")
    print(f"  Otsu threshold at 15mm: {otsu_15:.1f}")


if __name__ == "__main__":
    main()
