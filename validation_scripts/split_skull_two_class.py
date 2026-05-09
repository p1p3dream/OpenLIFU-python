#!/usr/bin/env python3
"""Split single-class skull labels into cortical + trabecular bone.

Takes a 7-class label NIfTI (label 5 = skull) and produces an 8-class version
where label 5 = cortical_bone (outer shell) and label 7 = trabecular_bone
(interior diploe), matching LABEL_MAP_FULLHEAD_TWO_CLASS_BONE.

The split uses a distance-from-boundary heuristic: voxels within
`--cortical-thickness` mm of the skull surface are labeled cortical, and the
interior remainder is labeled trabecular. This is an approximation; real
diploe distribution is anatomically irregular (calibrated against SimNIBS
Ernie Extended ground truth, Dice ~56% at 2.5mm threshold, but volumetric
ratio is approximately correct).

Usage:
    python scripts/split_skull_two_class.py \
        --input NC024_nnunet_labels.nii.gz \
        --output NC024_two_class_labels.nii.gz \
        --cortical-thickness 2.5

Batch mode (all subjects in a directory):
    python scripts/split_skull_two_class.py \
        --batch-dir ~/Data/openlifu-validation/results \
        --pattern '*_nnunet_labels.nii.gz' \
        --cortical-thickness 2.5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


def split_skull_labels(
    labels: np.ndarray,
    voxel_spacing: tuple[float, ...],
    cortical_thickness_mm: float = 2.5,
    skull_label: int = 5,
    cortical_label: int = 5,
    trabecular_label: int = 7,
) -> np.ndarray:
    """Split skull voxels into cortical (surface) and trabecular (interior).

    Parameters
    ----------
    labels : np.ndarray
        Integer label volume (7-class, skull = skull_label).
    voxel_spacing : tuple of float
        Voxel dimensions in mm (from NIfTI header).
    cortical_thickness_mm : float
        Thickness of cortical shell in mm. Voxels within this distance of the
        skull boundary are labeled cortical; deeper voxels become trabecular.
    skull_label : int
        Input label for skull voxels (default 5).
    cortical_label : int
        Output label for cortical bone (default 5, same position).
    trabecular_label : int
        Output label for trabecular bone (default 7, new slot).

    Returns
    -------
    np.ndarray
        8-class label volume with skull split into cortical + trabecular.
    """
    skull_mask = labels == skull_label
    n_skull = int(skull_mask.sum())
    if n_skull == 0:
        return labels.copy()

    non_skull = ~skull_mask
    dist_from_boundary = distance_transform_edt(skull_mask, sampling=voxel_spacing)

    output = labels.copy()
    cortical_mask = skull_mask & (dist_from_boundary <= cortical_thickness_mm)
    trabecular_mask = skull_mask & (dist_from_boundary > cortical_thickness_mm)

    output[cortical_mask] = cortical_label
    output[trabecular_mask] = trabecular_label

    n_cortical = int(cortical_mask.sum())
    n_trabecular = int(trabecular_mask.sum())
    pct_cortical = 100.0 * n_cortical / n_skull if n_skull > 0 else 0
    pct_trabecular = 100.0 * n_trabecular / n_skull if n_skull > 0 else 0

    print(f"  Skull voxels: {n_skull:,d}")
    print(f"  Cortical (label {cortical_label}): {n_cortical:,d} ({pct_cortical:.1f}%)")
    print(f"  Trabecular (label {trabecular_label}): {n_trabecular:,d} ({pct_trabecular:.1f}%)")
    print(f"  Max distance from boundary: {dist_from_boundary[skull_mask].max():.2f} mm")

    return output


def process_single(input_path: str, output_path: str, cortical_thickness_mm: float) -> None:
    """Process a single label NIfTI."""
    print(f"Processing: {input_path}")
    img = nib.load(input_path)
    labels = np.asarray(img.dataobj).astype(np.int16)

    voxel_spacing = tuple(abs(float(x)) for x in img.header.get_zooms()[:3])
    print(f"  Shape: {labels.shape}, spacing: {voxel_spacing} mm")
    print(f"  Unique labels: {sorted(np.unique(labels))}")

    result = split_skull_labels(labels, voxel_spacing, cortical_thickness_mm)

    out_img = nib.Nifti1Image(result, img.affine, img.header)
    nib.save(out_img, output_path)
    print(f"  Saved: {output_path}")
    print(f"  Unique labels: {sorted(np.unique(result))}")


def main():
    ap = argparse.ArgumentParser(
        description="Split single-class skull into cortical + trabecular bone labels."
    )
    ap.add_argument("--input", help="Single input label NIfTI path")
    ap.add_argument("--output", help="Single output label NIfTI path")
    ap.add_argument("--batch-dir", help="Directory for batch processing")
    ap.add_argument("--pattern", default="*_nnunet_labels.nii.gz",
                    help="Glob pattern for batch mode (default: *_nnunet_labels.nii.gz)")
    ap.add_argument("--cortical-thickness", type=float, default=2.5,
                    help="Cortical shell thickness in mm (default: 2.5)")
    ap.add_argument("--suffix", default="_two_class_labels.nii.gz",
                    help="Output suffix for batch mode (default: _two_class_labels.nii.gz)")
    args = ap.parse_args()

    if args.input and args.output:
        process_single(args.input, args.output, args.cortical_thickness)
    elif args.batch_dir:
        search = os.path.join(args.batch_dir, args.pattern)
        files = sorted(glob.glob(search))
        if not files:
            print(f"No files found matching: {search}")
            sys.exit(1)
        print(f"Found {len(files)} files matching {args.pattern}")
        print(f"Cortical thickness: {args.cortical_thickness} mm\n")
        for f in files:
            base = os.path.basename(f)
            subj = base.replace("_nnunet_labels.nii.gz", "")
            out = os.path.join(args.batch_dir, f"{subj}{args.suffix}")
            process_single(f, out, args.cortical_thickness)
            print()
    else:
        ap.error("Provide either --input/--output or --batch-dir")


if __name__ == "__main__":
    main()
