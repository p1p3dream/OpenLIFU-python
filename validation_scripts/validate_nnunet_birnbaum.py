"""Validate NNUNetSegmentation against Birnbaum full-head ground truth.

Runs inference on 5 subjects and computes per-class Dice scores.
"""

import sys
import time

import nibabel as nib
import numpy as np
import xarray as xa

# Ensure the local openlifu package is importable
sys.path.insert(0, "/Users/brandon/code/openwater/OpenLIFU-python/src")

from openlifu.seg.seg_methods.nnunet_seg import NNUNetSegmentation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = "/Users/brandon/Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects"
T1_DIR = f"{DATA_ROOT}/T1-Weighted MRI"
GT_DIR = f"{DATA_ROOT}/Full-Head Segmentation"

# NC001 and NYU001 don't exist in the dataset; using NC004 and NYU002 instead.
SUBJECTS = ["GU002", "GU008", "GU010", "NC004", "NYU002"]

MODEL_PATH = "/Users/brandon/.openlifu/models/fullhead_seg.onnx"

# Birnbaum ground truth labels -> our material keys
BIRNBAUM_TO_MATERIAL = {
    1: "air",
    2: "csf",
    3: "gray_matter",
    4: "white_matter",
    5: "skull",
    6: "tissue",
}

CLASS_NAMES = {
    "air": "Air",
    "csf": "CSF",
    "gray_matter": "Gray Matter",
    "white_matter": "White Matter",
    "skull": "Bone/Skull",
    "tissue": "Soft Tissue",
}


def load_nifti_as_xarray(nifti_path: str) -> xa.DataArray:
    """Load a NIfTI file and return an xarray DataArray with dims=['z','y','x']."""
    img = nib.load(nifti_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine

    spacing = np.abs(np.diag(affine)[:3])

    # Build coordinates: origin from affine, then step by spacing
    coords = {}
    dim_names = ["z", "y", "x"]
    for i, dim in enumerate(dim_names):
        origin = affine[i, 3]
        n = data.shape[i]
        coords[dim] = np.arange(n) * spacing[i] + origin

    return xa.DataArray(data, dims=dim_names, coords=coords)


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between two boolean masks."""
    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)
    if total == 0:
        return 1.0  # Both empty = perfect agreement
    return float(2.0 * intersection / total)


def main():
    print("=" * 80)
    print("NNUNet Segmentation Validation against Birnbaum Ground Truth")
    print("=" * 80)
    print()

    # Initialize the segmentation model (session is lazy-loaded)
    print("Initializing NNUNetSegmentation...")
    seg = NNUNetSegmentation(
        model_path=MODEL_PATH,
        model_type="fullhead",
        use_mirroring=False,
    )

    # Get our material index mapping
    mat_indices = seg._material_indices()
    print(f"Material indices: {mat_indices}")
    print()

    # Classes to evaluate
    eval_classes = list(BIRNBAUM_TO_MATERIAL.keys())

    # Store results: subject -> {material_key: dice}
    all_results = {}

    for subj_idx, subj in enumerate(SUBJECTS):
        print("-" * 70)
        print(f"[{subj_idx + 1}/{len(SUBJECTS)}] Processing subject: {subj}")
        print("-" * 70)

        t1_path = f"{T1_DIR}/{subj}_deface.nii"
        gt_path = f"{GT_DIR}/{subj}_label_deface.nii"

        # Load T1
        print(f"  Loading T1 from: {t1_path}")
        t1_xa = load_nifti_as_xarray(t1_path)
        print(f"  T1 shape: {t1_xa.shape}, dims: {t1_xa.dims}")

        # Load ground truth
        print(f"  Loading ground truth from: {gt_path}")
        gt_img = nib.load(gt_path)
        gt_data = np.asarray(gt_img.dataobj).astype(int)
        print(f"  GT shape: {gt_data.shape}, unique labels: {np.unique(gt_data)}")

        # Run segmentation
        print(f"  Running NNUNet inference (no mirroring)...")
        t0 = time.time()
        pred_xa = seg._segment(t1_xa)
        elapsed = time.time() - t0
        print(f"  Inference completed in {elapsed:.1f}s")

        pred_data = pred_xa.to_numpy()
        print(f"  Prediction shape: {pred_data.shape}, unique labels: {np.unique(pred_data)}")

        # Compute per-class Dice
        subj_dice = {}
        for birnbaum_label, mat_key in BIRNBAUM_TO_MATERIAL.items():
            our_index = mat_indices[mat_key]
            gt_mask = gt_data == birnbaum_label
            pred_mask = pred_data == our_index
            d = dice_score(pred_mask, gt_mask)
            subj_dice[mat_key] = d

        all_results[subj] = subj_dice

        # Print this subject's results
        print(f"  Per-class Dice for {subj}:")
        for mat_key in BIRNBAUM_TO_MATERIAL.values():
            print(f"    {CLASS_NAMES[mat_key]:>15s}: {subj_dice[mat_key]:.4f}")
        print()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY: Per-Subject, Per-Class Dice Scores")
    print("=" * 80)

    # Header
    class_keys = list(BIRNBAUM_TO_MATERIAL.values())
    header = f"{'Subject':>10s}"
    for ck in class_keys:
        header += f"  {CLASS_NAMES[ck]:>12s}"
    header += f"  {'Mean':>8s}"
    print(header)
    print("-" * len(header))

    # Per-subject rows
    for subj in SUBJECTS:
        row = f"{subj:>10s}"
        scores = []
        for ck in class_keys:
            d = all_results[subj][ck]
            row += f"  {d:>12.4f}"
            scores.append(d)
        row += f"  {np.mean(scores):>8.4f}"
        print(row)

    # Mean row
    print("-" * len(header))
    mean_row = f"{'MEAN':>10s}"
    overall_scores = []
    for ck in class_keys:
        class_mean = np.mean([all_results[s][ck] for s in SUBJECTS])
        mean_row += f"  {class_mean:>12.4f}"
        overall_scores.append(class_mean)
    mean_row += f"  {np.mean(overall_scores):>8.4f}"
    print(mean_row)
    print()


if __name__ == "__main__":
    main()
