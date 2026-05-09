#!/usr/bin/env python3
"""Remap SimNIBS Ernie Extended tissue labels to OpenLIFU fullhead format.

SimNIBS labels:  0=BG, 1=WM, 2=GM, 3=CSF, 4=Bone, 5=Scalp, 6=Eyes,
                 7=Compact bone, 8=Spongy bone, 9=Blood, 10=Muscle,
                 11=Cartilage, 12=Fat

OpenLIFU labels (single-class bone, default):
    0=water, 1=air, 2=CSF, 3=GM, 4=WM, 5=skull, 6=tissue

OpenLIFU labels (two-class bone, --two-class-bone):
    0=water, 1=air, 2=CSF, 3=GM, 4=WM, 5=cortical_bone, 6=tissue,
    7=trabecular_bone

The two-class bone mode preserves the distinction between compact bone
(cortical, outer/inner table) and spongy bone (trabecular, diploe),
allowing different acoustic properties to be assigned to each layer.
Facial bone (SimNIBS label 4) is mapped to cortical_bone in two-class
mode since it is predominantly compact bone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


# Single-class bone remap: all bone types merge into label 5 (skull).
REMAP_SINGLE = np.array([
    0,   # 0  Background -> water
    4,   # 1  White Matter -> white_matter
    3,   # 2  Gray Matter -> gray_matter
    2,   # 3  CSF -> csf
    5,   # 4  Bone (facial) -> skull
    6,   # 5  Scalp -> tissue
    6,   # 6  Eye balls -> tissue
    5,   # 7  Compact bone -> skull
    5,   # 8  Spongy bone -> skull
    6,   # 9  Blood -> tissue
    6,   # 10 Muscle -> tissue
    6,   # 11 Cartilage -> tissue
    6,   # 12 Fat -> tissue
], dtype=np.int16)

# Two-class bone remap: cortical bone = label 5, trabecular bone = label 7.
# Facial bone (label 4) is mapped to cortical_bone (label 5) since facial
# bone is predominantly compact/cortical.
REMAP_TWO_CLASS = np.array([
    0,   # 0  Background -> water
    4,   # 1  White Matter -> white_matter
    3,   # 2  Gray Matter -> gray_matter
    2,   # 3  CSF -> csf
    5,   # 4  Bone (facial) -> cortical_bone
    6,   # 5  Scalp -> tissue
    6,   # 6  Eye balls -> tissue
    5,   # 7  Compact bone -> cortical_bone
    7,   # 8  Spongy bone -> trabecular_bone
    6,   # 9  Blood -> tissue
    6,   # 10 Muscle -> tissue
    6,   # 11 Cartilage -> tissue
    6,   # 12 Fat -> tissue
], dtype=np.int16)

# Label name lookups for reporting.
LABEL_NAMES_SINGLE = {
    0: "water", 1: "air", 2: "csf", 3: "gray_matter",
    4: "white_matter", 5: "skull", 6: "tissue",
}
LABEL_NAMES_TWO_CLASS = {
    0: "water", 1: "air", 2: "csf", 3: "gray_matter",
    4: "white_matter", 5: "cortical_bone", 6: "tissue",
    7: "trabecular_bone",
}


def main():
    ap = argparse.ArgumentParser(description="Remap SimNIBS Ernie labels to OpenLIFU format")
    ap.add_argument("--input", default=str(
        Path.home() / "Data/openlifu-validation/simnibs-ernie/ErnieExtended"
        "/m2m_ernie_extended/final_tissues.nii.gz"))
    ap.add_argument("--output", default=str(
        Path.home() / "Data/openlifu-validation/results/ernie_nnunet_labels.nii.gz"))
    ap.add_argument("--reorient", action="store_true", default=True,
                    help="Reorient to RAS (closest canonical)")
    ap.add_argument("--two-class-bone", action="store_true", default=False,
                    help="Preserve cortical/trabecular bone distinction "
                    "(label 5=cortical, 7=trabecular) instead of merging "
                    "all bone into a single skull label")
    args = ap.parse_args()

    two_class = args.two_class_bone
    remap_table = REMAP_TWO_CLASS if two_class else REMAP_SINGLE
    label_names = LABEL_NAMES_TWO_CLASS if two_class else LABEL_NAMES_SINGLE
    mode_str = "two-class bone" if two_class else "single-class bone"

    # If --two-class-bone and no explicit --output, use a distinct filename.
    if two_class and args.output == str(
            Path.home() / "Data/openlifu-validation/results/ernie_nnunet_labels.nii.gz"):
        args.output = str(
            Path.home() / "Data/openlifu-validation/results/ernie_two_class_bone_labels.nii.gz")

    print(f"Mode: {mode_str}")
    print(f"Loading: {args.input}")
    img = nib.load(args.input)
    data = np.asarray(img.dataobj).astype(np.int16)
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")

    if data.ndim == 4 and data.shape[3] == 1:
        data = data[:, :, :, 0]
        print(f"  Squeezed singleton dim -> {data.shape}")

    labels_before = np.unique(data)
    print(f"  Input labels: {labels_before.tolist()}")

    max_label = int(data.max())
    if max_label >= len(remap_table):
        raise ValueError(f"Unexpected label {max_label} > {len(remap_table)-1}")

    remapped = remap_table[data]
    labels_after = np.unique(remapped)
    print(f"  Output labels: {labels_after.tolist()}")
    for lbl in labels_after:
        count = int((remapped == lbl).sum())
        name = label_names.get(int(lbl), f"unknown_{lbl}")
        print(f"    Label {lbl} ({name}): {count:,} voxels")

    out_img = nib.Nifti1Image(remapped, img.affine, img.header)

    if args.reorient:
        out_img = nib.as_closest_canonical(out_img)
        print(f"  Reoriented to RAS: shape={out_img.shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, args.output)
    print(f"\nSaved: {args.output}")

    out_data = np.asarray(out_img.dataobj)

    brain_labels = {2, 3, 4}
    brain_mask = np.isin(out_data, list(brain_labels))
    affine = out_img.affine
    brain_idx = np.argwhere(brain_mask)
    center_vox = brain_idx.mean(axis=0)
    center_mm = affine[:3, :3] @ center_vox + affine[:3, 3]
    print(f"  Brain center: ({center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f}) mm")
    print(f"  Brain voxels: {brain_mask.sum():,}")

    if two_class:
        cortical_mask = out_data == 5
        trabecular_mask = out_data == 7
        total_bone = int(cortical_mask.sum()) + int(trabecular_mask.sum())
        total_fg = int((out_data > 0).sum())
        print(f"  Cortical bone voxels: {cortical_mask.sum():,}")
        print(f"  Trabecular bone voxels: {trabecular_mask.sum():,}")
        print(f"  Total bone voxels: {total_bone:,}")
        print(f"  Bone % of foreground: {total_bone / total_fg * 100:.1f}%")
        if total_bone > 0:
            cortical_pct = cortical_mask.sum() / total_bone * 100
            print(f"  Cortical/trabecular ratio: {cortical_pct:.1f}% / {100-cortical_pct:.1f}%")
    else:
        skull_mask = out_data == 5
        print(f"  Skull voxels: {skull_mask.sum():,}")
        skull_pct = skull_mask.sum() / (out_data > 0).sum() * 100
        print(f"  Skull %: {skull_pct:.1f}%")


if __name__ == "__main__":
    main()
