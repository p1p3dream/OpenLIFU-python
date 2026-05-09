"""Validate ThresholdMRI against MNI152 FreeSurfer aseg and tissue probability maps.

Uses the TemplateFlow MNI152NLin2009cAsym template with:
- FreeSurfer aseg segmentation (44 labels) as anatomical ground truth
- Tissue probability maps (CSF/GM/WM) thresholded to create tissue ground truth

Requirements:
    Downloaded files in ~/Downloads/:
    - mni_T1w.nii.gz (TemplateFlow T1w template)
    - mni_aseg_dseg.nii.gz (FreeSurfer aseg)
    - mni_tissue_dseg.nii.gz (argmax of probability maps)
"""
from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import xarray as xa

from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = int(np.sum(pred & gt))
    denom = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * intersection / denom) if denom > 0 else 1.0


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    d = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    vr = (tp + fp) / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"dice": d, "sensitivity": sens, "specificity": spec, "volume_ratio": vr}


def main() -> None:
    t1_path = os.path.expanduser("~/Downloads/mni_T1w.nii.gz")
    aseg_path = os.path.expanduser("~/Downloads/mni_aseg_dseg.nii.gz")
    tissue_path = os.path.expanduser("~/Downloads/mni_tissue_dseg.nii.gz")

    for p in [t1_path, aseg_path, tissue_path]:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            return

    t1 = nib.load(t1_path)
    t1_data = t1.get_fdata()
    zooms = t1.header.get_zooms()[:3]
    nz, ny, nx = t1_data.shape
    print(f"MNI152 T1w: {t1_data.shape}, spacing: {zooms}")

    volume = xa.DataArray(
        t1_data, dims=["z", "y", "x"],
        coords={
            "z": np.arange(nz) * zooms[0],
            "y": np.arange(ny) * zooms[1],
            "x": np.arange(nx) * zooms[2],
        },
    )

    # FreeSurfer aseg ground truth
    aseg = nib.load(aseg_path).get_fdata().astype(int)
    wm_labels = {2, 41, 7, 46, 251, 252, 253, 254, 255}
    gm_labels = {3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60, 16}
    csf_labels = {4, 43, 5, 44, 14, 15, 24}
    aseg_wm = np.isin(aseg, list(wm_labels))
    aseg_gm = np.isin(aseg, list(gm_labels))
    aseg_csf = np.isin(aseg, list(csf_labels))
    aseg_brain = aseg_wm | aseg_gm | aseg_csf

    # Tissue probability ground truth
    tissue_gt = nib.load(tissue_path).get_fdata().astype(int)
    prob_wm = tissue_gt == 3
    prob_gm = tissue_gt == 2
    prob_csf = tissue_gt == 1

    # Run segmentation
    seg = ThresholdMRI(
        classify_brain_tissues=True,
        air_threshold_quantile=0.05,
        bias_correction_sigma_mm=30.0,
    )
    result = seg._segment(volume)
    idx = seg._material_indices()

    our_wm = result.to_numpy() == idx["white_matter"]
    our_gm = result.to_numpy() == idx["gray_matter"]
    our_csf = result.to_numpy() == idx["csf"]

    # Dice vs FreeSurfer aseg
    header = f"{'Tissue':<15s} {'Dice':>8s} {'Sensitivity':>12s} {'Specificity':>12s} {'Vol Ratio':>10s}"
    sep = "-" * 57

    print(f"\n{'='*60}")
    print(f"DICE SCORES vs FreeSurfer aseg ground truth")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for name, our, gt in [("White Matter", our_wm, aseg_wm), ("Gray Matter", our_gm, aseg_gm), ("CSF", our_csf, aseg_csf)]:
        m = metrics(our, gt)
        print(f"{name:<15s} {m['dice']:>8.3f} {m['sensitivity']:>12.3f} {m['specificity']:>12.3f} {m['volume_ratio']:>10.2f}")

    brain_correct = ((our_wm & aseg_wm) | (our_gm & aseg_gm) | (our_csf & aseg_csf)).sum()
    print(f"\nWithin-brain accuracy: {brain_correct / aseg_brain.sum():.3f}")

    # Dice vs probability maps
    print(f"\n{'='*60}")
    print(f"DICE SCORES vs probability-map ground truth")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for name, our, gt in [("White Matter", our_wm, prob_wm), ("Gray Matter", our_gm, prob_gm), ("CSF", our_csf, prob_csf)]:
        m = metrics(our, gt)
        print(f"{name:<15s} {m['dice']:>8.3f} {m['sensitivity']:>12.3f} {m['specificity']:>12.3f} {m['volume_ratio']:>10.2f}")


if __name__ == "__main__":
    main()
