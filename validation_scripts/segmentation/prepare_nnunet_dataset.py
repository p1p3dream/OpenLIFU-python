"""Prepare all skull segmentation data into nnU-Net v2 dataset format.

nnU-Net expects:
  nnUNet_raw/Dataset001_SkullSeg/
    dataset.json
    imagesTr/
      CASE_ID_0000.nii.gz   (T1 MRI, modality 0000)
    labelsTr/
      CASE_ID.nii.gz        (binary skull label: 0=background, 1=skull)

Sources:
  1. Birnbaum (64 subjects): T1 + label 5 = bone
  2. SynthRAD2023 (180 subjects): T1 MRI + CT-derived skull labels
  3. SimNIBS Group (5 subjects): T1 + label 4 = bone
  4. Ernie (1 subject): T1 + label 4 = bone
  5. Colin27 (1 subject): T1 + labels 7+11 = skull+marrow (needs resampling from 0.5mm to 1mm)
  6. IXI025 (1 subject): T1 + labels 2+15 = cortical+cancellous bone
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as scipy_zoom


BASE = Path(os.path.expanduser("~/Data/openlifu-validation/datasets"))
OUT = Path(os.path.expanduser("~/Data/openlifu-validation/nnunet_raw/Dataset001_SkullSeg"))
IMAGES = OUT / "imagesTr"
LABELS = OUT / "labelsTr"


def save_binary_label(label_data, affine, header, out_path):
    """Save a binary label as uint8 NIfTI."""
    img = nib.Nifti1Image(label_data.astype(np.uint8), affine, header)
    nib.save(img, str(out_path))


def copy_as_gz(src, dst):
    """Copy a NIfTI file, compressing if needed."""
    if str(src).endswith(".nii.gz"):
        shutil.copy2(str(src), str(dst))
    else:
        # Load and save as .nii.gz
        img = nib.load(str(src))
        nib.save(img, str(dst))


def process_birnbaum():
    """64 subjects. Label 5 = bone."""
    count = 0
    birn = BASE / "birnbaum-fullhead" / "Data"
    for group, prefix in [("Anonymized_Subjects", "_deface"), ("Control_Subjects", "")]:
        t1_dir = birn / group / "T1-Weighted MRI"
        seg_dir = birn / group / "Full-Head Segmentation"
        if not t1_dir.exists():
            continue
        for f in sorted(t1_dir.glob("*.nii")):
            sid = f.stem.replace("_deface", "")
            label_name = f"{sid}_label{prefix}.nii"
            label_path = seg_dir / label_name
            if not label_path.exists():
                continue

            case_id = f"birn_{sid}"
            # Copy T1
            copy_as_gz(f, IMAGES / f"{case_id}_0000.nii.gz")
            # Extract bone label (label 5)
            labels = nib.load(str(label_path))
            bone = (labels.get_fdata().astype(int) == 5).astype(np.uint8)
            save_binary_label(bone, labels.affine, labels.header, LABELS / f"{case_id}.nii.gz")
            count += 1
    return count


def process_synthrad():
    """180 subjects. CT-derived skull labels already generated."""
    count = 0
    skull_dir = BASE / "synthrad2023" / "skull_labels"
    brain_dir = BASE / "synthrad2023" / "Task1" / "brain"
    for skull_file in sorted(skull_dir.glob("*_skull.nii.gz")):
        sid = skull_file.stem.replace("_skull", "")
        mr_path = brain_dir / sid / "mr.nii.gz"
        if not mr_path.exists():
            continue

        case_id = f"srad_{sid}"
        shutil.copy2(str(mr_path), str(IMAGES / f"{case_id}_0000.nii.gz"))
        shutil.copy2(str(skull_file), str(LABELS / f"{case_id}.nii.gz"))
        count += 1
    return count


def process_simnibs_group():
    """5 subjects. Label 4 = bone (also 7=compact, 8=spongy, but use 4 as union)."""
    count = 0
    group_dir = BASE / "simnibs-group"
    for subj_dir in sorted(group_dir.glob("m2m_sub*")):
        sid = subj_dir.name.replace("m2m_", "")
        t1 = subj_dir / "T1.nii.gz"
        seg = subj_dir / "final_tissues.nii.gz"
        if not t1.exists() or not seg.exists():
            continue

        case_id = f"snim_{sid}"
        shutil.copy2(str(t1), str(IMAGES / f"{case_id}_0000.nii.gz"))

        labels = nib.load(str(seg))
        label_data = labels.get_fdata().astype(int)
        if label_data.ndim == 4:
            label_data = label_data[:, :, :, 0]
        # Labels 4, 7, 8 are all bone variants; union them
        bone = np.isin(label_data, [4, 7, 8]).astype(np.uint8)
        save_binary_label(bone, labels.affine, labels.header, LABELS / f"{case_id}.nii.gz")
        count += 1
    return count


def process_ernie():
    """1 subject. Label 4 = bone."""
    t1 = BASE / "ernie" / "m2m_ernie" / "T1.nii.gz"
    seg = BASE / "ernie" / "m2m_ernie" / "final_tissues.nii.gz"
    if not t1.exists() or not seg.exists():
        return 0

    case_id = "ernie_001"
    shutil.copy2(str(t1), str(IMAGES / f"{case_id}_0000.nii.gz"))

    labels = nib.load(str(seg))
    label_data = labels.get_fdata().astype(int)
    if label_data.ndim == 4:
        label_data = label_data[:, :, :, 0]
    bone = (label_data == 4).astype(np.uint8)
    save_binary_label(bone, labels.affine, labels.header, LABELS / f"{case_id}.nii.gz")
    return 1


def process_colin27():
    """1 subject. Labels 7 (skull) + 11 (marrow) = bone. Needs resampling from 0.5mm to 1mm."""
    t1_path = BASE / "colin27" / "colin27_t1_tal_hires.nii"
    cls_path = BASE / "colin27" / "colin27_cls_tal_hires.nii"
    if not t1_path.exists() or not cls_path.exists():
        return 0

    case_id = "colin_001"

    # Load at 0.5mm
    t1 = nib.load(str(t1_path))
    cls = nib.load(str(cls_path))
    t1_data = t1.get_fdata()
    cls_data = cls.get_fdata().astype(int)

    # Downsample to 1mm (factor 0.5 in each dim)
    t1_ds = scipy_zoom(t1_data, 0.5, order=3)
    cls_ds = scipy_zoom(cls_data, 0.5, order=0)  # nearest neighbor for labels

    # Update affine for 1mm spacing
    new_affine = t1.affine.copy()
    new_affine[:3, :3] *= 2  # double voxel size
    new_affine[:3, 3] = t1.affine[:3, 3]  # keep origin

    # Save T1
    t1_img = nib.Nifti1Image(t1_ds.astype(np.float32), new_affine)
    nib.save(t1_img, str(IMAGES / f"{case_id}_0000.nii.gz"))

    # Labels 7 + 11 = bone
    bone = np.isin(cls_ds, [7, 11]).astype(np.uint8)
    save_binary_label(bone, new_affine, None, LABELS / f"{case_id}.nii.gz")
    return 1


def process_ixi025():
    """1 subject. Labels 2 (cortical bone) + 15 (cancellous bone)."""
    seg_dir = BASE / "ixi025" / "IXI025-Model_v1.0.0"
    t1 = seg_dir / "anat" / "IXI025-Guys-0852-T1.nii.gz"
    seg = seg_dir / "seg" / "IXI025-Guys-0852-SEG.nii.gz"
    if not t1.exists() or not seg.exists():
        return 0

    case_id = "ixi025_001"
    shutil.copy2(str(t1), str(IMAGES / f"{case_id}_0000.nii.gz"))

    labels = nib.load(str(seg))
    label_data = labels.get_fdata().astype(int)
    bone = np.isin(label_data, [2, 15]).astype(np.uint8)
    save_binary_label(bone, labels.affine, labels.header, LABELS / f"{case_id}.nii.gz")
    return 1


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    total = 0
    print("Preparing nnU-Net dataset...")

    n = process_birnbaum()
    print(f"  Birnbaum: {n} subjects")
    total += n

    n = process_synthrad()
    print(f"  SynthRAD2023: {n} subjects")
    total += n

    n = process_simnibs_group()
    print(f"  SimNIBS Group: {n} subjects")
    total += n

    n = process_ernie()
    print(f"  Ernie: {n} subjects")
    total += n

    n = process_colin27()
    print(f"  Colin27: {n} subjects")
    total += n

    n = process_ixi025()
    print(f"  IXI025: {n} subjects")
    total += n

    print(f"\nTotal: {total} subjects")

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "T1"},
        "labels": {"background": 0, "skull": 1},
        "numTraining": total,
        "file_ending": ".nii.gz",
        "name": "Dataset001_SkullSeg",
        "description": "Binary skull segmentation from T1 MRI. Mixed sources: expert-corrected (Birnbaum), CT-derived (SynthRAD), CHARM (SimNIBS/Ernie), semi-auto (Colin27), expert (IXI025).",
        "reference": "OpenLIFU GLADYS project",
        "licence": "Mixed (CC BY-NC-SA, CC BY-NC, CC BY-SA, CC BY)",
        "release": "1.0",
    }
    with open(OUT / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"dataset.json written to {OUT / 'dataset.json'}")

    # Verify
    images = sorted(IMAGES.glob("*_0000.nii.gz"))
    labels = sorted(LABELS.glob("*.nii.gz"))
    print(f"\nVerification: {len(images)} images, {len(labels)} labels")
    assert len(images) == len(labels) == total, f"Mismatch: {len(images)} images, {len(labels)} labels, expected {total}"
    print("All good.")


if __name__ == "__main__":
    main()
