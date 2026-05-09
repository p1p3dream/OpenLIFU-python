"""Check the Otsu below-fraction across all Birnbaum subjects.

Verify that the 50/50 skip guard would NOT trigger on any Birnbaum scan.
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

from openlifu.seg.seg_methods.threshold_mri import compute_foreground_mask


def check_otsu_fraction(t1_data, zooms, skull_mm=12.0):
    spacing = np.array(zooms, dtype=float)
    foreground = compute_foreground_mask(t1_data)
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)
    brain_mask = foreground_dist > skull_mm
    shell_mask = foreground & ~brain_mask
    shell_int = t1_data[shell_mask]
    nz = shell_int[shell_int > 0]
    otsu = threshold_otsu(nz)
    below = np.sum(nz < otsu) / nz.size
    return otsu, below, int(shell_mask.sum())


def main():
    skull_mm = 12.0
    birn_base = os.path.expanduser("~/Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects")
    t1_dir = os.path.join(birn_base, "T1-Weighted MRI")
    seg_dir = os.path.join(birn_base, "Full-Head Segmentation")

    subjects = []
    for f in sorted(os.listdir(t1_dir)):
        if f.endswith(".nii"):
            sid = f.replace("_deface.nii", "")
            subjects.append((sid, os.path.join(t1_dir, f)))

    print(f"{'Subject':<10} {'Otsu':>10} {'Below%':>10} {'Shell vox':>12} {'Guard?':>10}")
    print(f"{'-'*55}")

    for sid, t1_path in subjects:
        t1 = nib.load(t1_path)
        t1_data = t1.get_fdata().astype(np.float32)
        zooms = t1.header.get_zooms()[:3]
        otsu, below_frac, shell_count = check_otsu_fraction(t1_data, zooms, skull_mm)
        guard = abs(below_frac - 0.5) < 0.15
        print(f"{sid:<10} {otsu:>10.1f} {below_frac:>10.3f} {shell_count:>12,} {'SKIP' if guard else 'REFINE':>10}")

    # Also check Ernie
    ernie_base = os.path.expanduser("~/Data/openlifu-validation/datasets/ernie/m2m_ernie")
    t1_e = nib.load(os.path.join(ernie_base, "T1.nii.gz")).get_fdata().astype(np.float32)
    zooms_e = nib.load(os.path.join(ernie_base, "T1.nii.gz")).header.get_zooms()[:3]
    otsu_e, below_e, shell_e = check_otsu_fraction(t1_e, zooms_e, skull_mm)
    guard_e = abs(below_e - 0.5) < 0.15
    print(f"{'ernie':<10} {otsu_e:>10.1f} {below_e:>10.3f} {shell_e:>12,} {'SKIP' if guard_e else 'REFINE':>10}")


if __name__ == "__main__":
    main()
