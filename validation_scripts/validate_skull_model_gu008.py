"""Validate skull_seg.onnx on GU008 and compare to fullhead bone Dice."""

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xa


def load_nifti_as_xarray(nifti_path: str | Path) -> xa.DataArray:
    """Load a NIfTI file and return an xarray DataArray with mm coordinates."""
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine

    coords = {}
    dim_names = ("x", "y", "z")
    for axis, dim in enumerate(dim_names):
        n_voxels = data.shape[axis]
        origin = affine[axis, 3]
        spacing = affine[axis, axis]
        coord_values = origin + np.arange(n_voxels) * spacing
        coords[dim] = xa.Variable(dim, coord_values, attrs={"units": "mm"})

    return xa.DataArray(data, dims=dim_names, coords=coords)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between two boolean masks."""
    intersection = np.sum(pred & gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0  # both empty = perfect agreement
    return 2.0 * intersection / total


def main():
    # Paths
    mri_path = Path.home() / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
    gt_path = Path.home() / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects/Full-Head Segmentation/GU008_label_deface.nii"
    model_path = Path.home() / ".openlifu/models/skull_seg.onnx"

    print("=" * 60)
    print("Skull Model Validation: GU008")
    print("=" * 60)

    # Step 1: Load MRI
    print("\nLoading MRI volume...")
    volume = load_nifti_as_xarray(mri_path)
    print(f"  Shape: {volume.shape}")
    print(f"  Dims: {volume.dims}")
    spacing = []
    for dim in volume.dims:
        c = volume.coords[dim].to_numpy()
        spacing.append(abs(c[1] - c[0]))
    print(f"  Spacing: {spacing}")

    # Step 2: Run skull model segmentation
    print("\nInitializing NNUNetSegmentation (skull model, no mirroring)...")
    from openlifu.seg.seg_methods.nnunet_seg import NNUNetSegmentation

    seg = NNUNetSegmentation(
        model_path=str(model_path),
        model_type="skull",
        use_mirroring=False,
    )
    material_idx = seg._material_indices()
    print(f"  Material indices: {material_idx}")

    print("\nRunning skull model inference...")
    t0 = time.time()
    result = seg._segment(volume)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.1f}s")

    result_np = result.to_numpy()
    unique_labels, counts = np.unique(result_np, return_counts=True)
    print(f"  Unique output labels: {dict(zip(unique_labels, counts))}")

    # Step 3: Load ground truth
    print("\nLoading ground truth segmentation...")
    gt_img = nib.load(str(gt_path))
    gt_data = np.asarray(gt_img.dataobj, dtype=np.int16)
    gt_unique, gt_counts = np.unique(gt_data, return_counts=True)
    print(f"  GT shape: {gt_data.shape}")
    print(f"  GT unique labels: {dict(zip(gt_unique, gt_counts))}")
    print("  GT label meanings: 0=bg, 1=skin, 2=CSF, 3=gray, 4=white, 5=bone, 6=air")

    # Step 4: Compute Dice for bone/skull
    # Skull model: material_idx["skull"] is the skull class
    pred_skull_mask = result_np == material_idx["skull"]
    gt_bone_mask = gt_data == 5

    bone_dice = dice_coefficient(pred_skull_mask, gt_bone_mask)

    print(f"\n{'=' * 60}")
    print("BONE / SKULL DICE")
    print(f"{'=' * 60}")
    print(f"  Predicted skull voxels: {np.sum(pred_skull_mask):,}")
    print(f"  GT bone voxels (label 5): {np.sum(gt_bone_mask):,}")
    print(f"  Dice coefficient: {bone_dice:.4f}")

    # Step 5: Compute brain mask Dice
    # Skull model assigns "tissue" to brain interior via foreground detection
    pred_brain_mask = result_np == material_idx["tissue"]
    # GT brain = CSF (2) + gray matter (3) + white matter (4)
    gt_brain_mask = (gt_data == 2) | (gt_data == 3) | (gt_data == 4)

    brain_dice = dice_coefficient(pred_brain_mask, gt_brain_mask)

    print(f"\n{'=' * 60}")
    print("BRAIN MASK DICE (tissue vs CSF+GM+WM)")
    print(f"{'=' * 60}")
    print(f"  Predicted tissue voxels: {np.sum(pred_brain_mask):,}")
    print(f"  GT brain voxels (2+3+4): {np.sum(gt_brain_mask):,}")
    print(f"  Dice coefficient: {brain_dice:.4f}")

    # Step 6: Timing comparison
    fullhead_time = 103.0  # known from previous runs

    print(f"\n{'=' * 60}")
    print("TIMING COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Skull model:    {elapsed:.1f}s (no mirroring)")
    print(f"  Fullhead model: {fullhead_time:.1f}s (no mirroring)")
    print(f"  Speedup:        {fullhead_time / elapsed:.2f}x")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Bone Dice:     {bone_dice:.4f}")
    print(f"  Brain Dice:    {brain_dice:.4f}")
    print(f"  Time:          {elapsed:.1f}s (vs {fullhead_time:.1f}s fullhead)")


if __name__ == "__main__":
    main()
