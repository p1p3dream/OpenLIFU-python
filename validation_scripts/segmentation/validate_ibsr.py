"""Validate ThresholdMRI against IBSR v2.0 manual expert ground truth.

IBSR provides 18 T1w MRIs with voxel-level tissue labels drawn by expert
neuroanatomists. This is the gold standard for brain tissue segmentation
validation on real (not simulated) data.

Download: https://www.nitrc.org/projects/ibsr (free NITRC account required)
Extract to ~/Downloads/ibsr/IBSR_nifti_stripped/

IBSR segTRI_fill labels: 0=background, 1=CSF, 2=GM, 3=WM

Results (18 subjects, EM-GMM + auto skull-strip + bias correction):
    WM=0.824  GM=0.850  CSF=0.492
    ANTs Atropos on same data (5 subjects): WM=0.874 GM=0.805 CSF=0.524
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
    base = os.path.expanduser("~/Downloads/ibsr/IBSR_nifti_stripped")
    subjects = sorted([d for d in os.listdir(base) if d.startswith("IBSR_")])

    if not subjects:
        print(f"No IBSR subjects found in {base}. Download from NITRC first.")
        return

    seg = ThresholdMRI(
        classify_brain_tissues=True,
        air_threshold_quantile=0.0,
        bias_correction_sigma_mm=30.0,
    )

    results: dict[str, list[float]] = {"wm": [], "gm": [], "csf": []}

    for subj in subjects:
        t1_path = os.path.join(base, subj, f"{subj}_ana_strip.nii.gz")
        gt_path = os.path.join(base, subj, f"{subj}_segTRI_fill_ana.nii.gz")
        if not os.path.exists(t1_path) or not os.path.exists(gt_path):
            continue

        data = nib.load(t1_path).get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[:, :, :, 0]
        gt = nib.load(gt_path).get_fdata().astype(int)
        if gt.ndim == 4:
            gt = gt[:, :, :, 0]
        zooms = nib.load(t1_path).header.get_zooms()[:3]

        nz, ny, nx = data.shape
        volume = xa.DataArray(
            data, dims=["z", "y", "x"],
            coords={
                "z": np.arange(nz) * zooms[0],
                "y": np.arange(ny) * zooms[1],
                "x": np.arange(nx) * zooms[2],
            },
        )

        result = seg._segment(volume)
        idx = seg._material_indices()

        d_wm = dice(result.to_numpy() == idx["white_matter"], gt == 3)
        d_gm = dice(result.to_numpy() == idx["gray_matter"], gt == 2)
        d_csf = dice(result.to_numpy() == idx["csf"], gt == 1)

        results["wm"].append(d_wm)
        results["gm"].append(d_gm)
        results["csf"].append(d_csf)
        print(f"  {subj}: WM={d_wm:.3f} GM={d_gm:.3f} CSF={d_csf:.3f}")

    print(f"\nMean ({len(results['wm'])} subjects):")
    print(f"  WM={np.mean(results['wm']):.3f} GM={np.mean(results['gm']):.3f} CSF={np.mean(results['csf']):.3f}")
    print(f"\nANTs Atropos on IBSR (5 subjects): WM=0.874 GM=0.805 CSF=0.524")


if __name__ == "__main__":
    main()
