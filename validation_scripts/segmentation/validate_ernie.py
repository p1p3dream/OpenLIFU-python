"""Validate ThresholdMRI 4-label mode against SimNIBS Ernie (CHARM ground truth).

The Ernie dataset provides CHARM-derived tissue labels including bone (label 4)
and scalp (label 5) on a real clinical T1w MRI. This is the standard dataset
for transcranial focused ultrasound segmentation validation.

Download: https://github.com/simnibs/example-dataset/releases/download/v4.0-lowres/ernie_lowres_V2.zip
Extract to ~/Downloads/ernie/

Results:
    Bone + scalp Dice: 0.427
    Brain tissue Dice: 0.584
    Water/background Dice: 0.917
    Acoustic property accuracy: 82.8%
"""
from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import xarray as xa

from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def main() -> None:
    base = os.path.expanduser("~/Downloads/ernie/m2m_ernie")
    t1_path = os.path.join(base, "T1.nii.gz")
    gt_path = os.path.join(base, "final_tissues.nii.gz")

    for p in [t1_path, gt_path]:
        if not os.path.exists(p):
            print(f"Missing: {p}. Download Ernie dataset first.")
            return

    t1_data = nib.load(t1_path).get_fdata().astype(np.float32)
    gt_data = nib.load(gt_path).get_fdata().astype(int)
    if gt_data.ndim == 4:
        gt_data = gt_data[:, :, :, 0]
    zooms = nib.load(t1_path).header.get_zooms()[:3]

    nz, ny, nx = t1_data.shape
    vol = xa.DataArray(
        t1_data, dims=["z", "y", "x"],
        coords={
            "z": np.arange(nz) * zooms[0],
            "y": np.arange(ny) * zooms[1],
            "x": np.arange(nx) * zooms[2],
        },
    )

    seg = ThresholdMRI()
    result = seg._segment(vol)
    idx = seg._material_indices()

    our_skull = result.to_numpy() == idx["skull"]
    our_tissue = result.to_numpy() == idx["tissue"]
    our_water = result.to_numpy() == idx["water"]

    gt_bone = gt_data == 4
    gt_shell = np.isin(gt_data, [4, 5])
    gt_brain = np.isin(gt_data, [1, 2, 3])
    gt_bg = gt_data == 0

    print("SimNIBS Ernie (CHARM ground truth) - 4-label validation:")
    print(f"  Bone only (label 4):         Dice = {dice(our_skull, gt_bone):.3f}")
    print(f"  Bone + scalp (labels 4+5):   Dice = {dice(our_skull, gt_shell):.3f}")
    print(f"  Brain tissue (labels 1+2+3): Dice = {dice(our_tissue, gt_brain):.3f}")
    print(f"  Water/background (label 0):  Dice = {dice(our_water, gt_bg):.3f}")

    correct = (
        int((gt_bone & our_skull).sum())
        + int((gt_brain & our_tissue).sum())
        + int((gt_bg & our_water).sum())
    )
    total = int(gt_bone.sum()) + int(gt_brain.sum()) + int(gt_bg.sum())
    print(f"\n  Acoustic property accuracy: {correct / total:.1%}")


if __name__ == "__main__":
    main()
