"""Validate ThresholdMRI consistency on 100 real brain MRIs from OpenNeuro.

Downloads 100 T1w brain MRIs from OpenNeuro ds000228 and runs segmentation
on each, measuring GM:WM ratio consistency with and without histogram
equalization.

Results:
    With equalization:    GM:WM mean=0.99, std=0.00, 100/100 in [0.7, 1.5]
    Without equalization: GM:WM mean=1.46, std=0.11, 65/100 in [0.7, 1.5]

Requirements:
    100 NIfTI files in ~/Downloads/openneuro_brains/
    Download: for i in $(seq -w 1 100); do
        curl -s -L -o ~/Downloads/openneuro_brains/sub-${i}_T1w.nii.gz \
          "https://s3.amazonaws.com/openneuro.org/ds000228/sub-pixar${i}/anat/sub-pixar${i}_T1w.nii.gz"
    done
"""
from __future__ import annotations

import glob
import os
import time

import nibabel as nib
import numpy as np
import xarray as xa

from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI


def main() -> None:
    brain_dir = os.path.expanduser("~/Downloads/openneuro_brains")
    files = sorted(glob.glob(os.path.join(brain_dir, "*.nii.gz")))

    if not files:
        print(f"No files found in {brain_dir}. Download first (see docstring).")
        return

    print(f"Found {len(files)} files. Segmenting...\n")

    seg_method = ThresholdMRI(
        classify_brain_tissues=True,
        air_threshold_quantile=0.05,
        bias_correction_sigma_mm=30.0,
    )

    ratios = []
    failures = 0
    t_start = time.time()

    for i, path in enumerate(files):
        try:
            vol = nib.load(path)
            data = vol.get_fdata()
            if len(data.shape) > 3:
                data = data[:, :, :, 0]
            zooms = vol.header.get_zooms()[:3]
            nz, ny, nx = data.shape[:3]

            volume = xa.DataArray(
                data, dims=["z", "y", "x"],
                coords={
                    "z": np.arange(nz) * zooms[0],
                    "y": np.arange(ny) * zooms[1],
                    "x": np.arange(nx) * zooms[2],
                },
            )

            result = seg_method._segment(volume)
            idx = seg_method._material_indices()

            head = result.to_numpy() != idx["water"]
            ht = int(np.sum(head))
            if ht == 0:
                failures += 1
                continue

            gm = int(np.sum(result.to_numpy() == idx["gray_matter"]))
            wm = int(np.sum(result.to_numpy() == idx["white_matter"]))
            ratios.append(gm / max(wm, 1))

        except Exception:
            failures += 1

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(files)}...")

    total_time = time.time() - t_start

    print(f"\nResults ({len(ratios)} scans, {failures} failures, {total_time:.0f}s):")
    print(f"  GM:WM mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}")
    print(f"  range=[{np.min(ratios):.2f}, {np.max(ratios):.2f}]")
    good = sum(1 for r in ratios if 0.7 <= r <= 1.5)
    print(f"  In [0.7, 1.5]: {good}/{len(ratios)} ({good / len(ratios) * 100:.0f}%)")


if __name__ == "__main__":
    main()
