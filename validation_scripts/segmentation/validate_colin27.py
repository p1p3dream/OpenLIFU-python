"""Validate ThresholdMRI against Colin27 Average Brain with 12-class tissue labels.

Colin27 provides a high-resolution (0.5mm) averaged brain with manual tissue
segmentation including skull, CSF, GM, WM, fat, muscle, dura, marrow, vessels.

Download:
    curl -O https://packages.bic.mni.mcgill.ca/mni-models/colin27/mni_colin27_2008_nifti.zip
    unzip mni_colin27_2008_nifti.zip -d ~/Downloads/colin27_2008/

Labels: 0=BG, 1=CSF, 2=GM, 3=WM, 4=Fat, 5=Muscles, 6=Skin/Muscles,
        7=Skull, 9=Fat2, 10=Dura, 11=Marrow, 12=Vessels
"""
from __future__ import annotations

import os
import time

import nibabel as nib
import numpy as np
import xarray as xa

from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = int(np.sum(pred & gt))
    denom = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * intersection / denom) if denom > 0 else 1.0


def main() -> None:
    base = os.path.expanduser("~/Downloads/colin27_2008")
    t1_path = os.path.join(base, "colin27_t1_tal_hires.nii")
    cls_path = os.path.join(base, "colin27_cls_tal_hires.nii")

    for p in [t1_path, cls_path]:
        if not os.path.exists(p):
            print(f"Missing: {p}. Download Colin27 first (see docstring).")
            return

    t1 = nib.load(t1_path)
    t1_data = t1.get_fdata()
    cls = nib.load(cls_path).get_fdata().astype(int)
    zooms = t1.header.get_zooms()[:3]
    print(f"Colin27: {t1_data.shape} at {zooms[0]}mm isotropic ({t1_data.size:,} voxels)")

    # Downsample to 1mm for practical runtime
    t1_ds = t1_data[::2, ::2, ::2]
    cls_ds = cls[::2, ::2, ::2]
    spacing = zooms[0] * 2

    nz, ny, nx = t1_ds.shape
    volume = xa.DataArray(
        t1_ds, dims=["z", "y", "x"],
        coords={
            "z": np.arange(nz) * spacing,
            "y": np.arange(ny) * spacing,
            "x": np.arange(nx) * spacing,
        },
    )

    gt_csf = cls_ds == 1
    gt_gm = cls_ds == 2
    gt_wm = cls_ds == 3
    gt_skull = cls_ds == 7
    gt_brain = gt_csf | gt_gm | gt_wm

    print(f"\nGround truth (downsampled to {spacing}mm):")
    for name, mask in [("CSF", gt_csf), ("GM", gt_gm), ("WM", gt_wm), ("Skull", gt_skull)]:
        print(f"  {name}: {mask.sum():,} voxels")

    # Run segmentation
    t0 = time.time()
    seg = ThresholdMRI(
        classify_brain_tissues=True,
        air_threshold_quantile=0.05,
        bias_correction_sigma_mm=30.0,
    )
    result = seg._segment(volume)
    elapsed = time.time() - t0
    idx = seg._material_indices()
    print(f"\nSegmentation completed in {elapsed:.1f}s")

    our_csf = result.to_numpy() == idx["csf"]
    our_gm = result.to_numpy() == idx["gray_matter"]
    our_wm = result.to_numpy() == idx["white_matter"]
    our_skull = result.to_numpy() == idx["skull"]

    print(f"\n{'='*65}")
    print(f"DICE SCORES: ThresholdMRI vs Colin27 ground truth")
    print(f"{'='*65}")
    print(f"{'Tissue':<15s} {'Dice':>8s} {'Ours':>12s} {'GT':>12s} {'Vol Ratio':>10s}")
    print(f"{'-'*57}")

    for name, ours, gt in [
        ("Skull", our_skull, gt_skull),
        ("White Matter", our_wm, gt_wm),
        ("Gray Matter", our_gm, gt_gm),
        ("CSF", our_csf, gt_csf),
    ]:
        d = dice(ours, gt)
        our_count = int(ours.sum())
        gt_count = int(gt.sum())
        vr = our_count / gt_count if gt_count > 0 else float("nan")
        print(f"{name:<15s} {d:>8.3f} {our_count:>12,} {gt_count:>12,} {vr:>10.2f}")

    brain_correct = ((our_wm & gt_wm) | (our_gm & gt_gm) | (our_csf & gt_csf)).sum()
    brain_total = gt_brain.sum()
    if brain_total > 0:
        print(f"\nWithin-brain accuracy: {brain_correct / brain_total:.3f}")


if __name__ == "__main__":
    main()
