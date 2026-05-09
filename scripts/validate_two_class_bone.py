#!/usr/bin/env python3
"""Validate two-class bone modeling on the SimNIBS Ernie Extended dataset.

Loads the Ernie Extended tissue labels, applies the extended remapping that
preserves the cortical/trabecular bone distinction, assigns acoustic
properties using the two-class bone model, and reports tissue composition
and property distributions.

Compares single-class (homogeneous skull) vs. two-class (cortical + trabecular)
models to quantify the acoustic property differences, particularly the
attenuation contrast that drives simulation accuracy for thick skulls.

Usage:
    python scripts/validate_two_class_bone.py

    # With custom input path:
    python scripts/validate_two_class_bone.py \
        --input ~/Data/openlifu-validation/simnibs-ernie/ErnieExtended/m2m_ernie_extended/final_tissues.nii.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

# Import material definitions from the library.
from openlifu.seg.material import (
    CORTICAL_BONE,
    MATERIALS,
    SKULL,
    TRABECULAR_BONE,
)

# Reuse remap tables from the remap script.
from remap_ernie_labels import (
    LABEL_NAMES_SINGLE,
    LABEL_NAMES_TWO_CLASS,
    REMAP_SINGLE,
    REMAP_TWO_CLASS,
)


def _load_and_prep(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the tissue label volume, squeeze if needed, return data."""
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.int16)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[:, :, :, 0]
    # Reorient to RAS.
    img_ras = nib.as_closest_canonical(nib.Nifti1Image(data, img.affine, img.header))
    data_ras = np.asarray(img_ras.dataobj).astype(np.int16)
    affine = img_ras.affine
    return data_ras, affine


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    """Compute voxel volume in mm^3 from the affine."""
    return float(np.abs(np.linalg.det(affine[:3, :3])))


def _report_composition(data: np.ndarray, label_names: dict, title: str) -> None:
    """Print tissue composition summary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    total_fg = int((data > 0).sum())
    total_all = int(data.size)
    for lbl in sorted(label_names.keys()):
        count = int((data == lbl).sum())
        name = label_names[lbl]
        if count > 0:
            if lbl == 0:
                pct = count / total_all * 100
                print(f"  Label {lbl:2d} ({name:16s}): {count:>10,} voxels  (background)")
            else:
                pct = count / total_fg * 100 if total_fg > 0 else 0
                print(f"  Label {lbl:2d} ({name:16s}): {count:>10,} voxels  ({pct:5.1f}%)")
    print(f"  {'Total foreground':28s}: {total_fg:>10,} voxels")


def _report_acoustic_properties(title: str, materials_map: dict, volumes_mm3: dict) -> None:
    """Print acoustic property summary for each tissue type."""
    print(f"\n{'-'*60}")
    print(f"  Acoustic Properties: {title}")
    print(f"{'-'*60}")
    print(f"  {'Material':18s}  {'c (m/s)':>8s}  {'rho (kg/m3)':>11s}  "
          f"{'alpha (dB/cm/MHz)':>17s}  {'Volume (cm3)':>12s}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*11}  {'-'*17}  {'-'*12}")
    for name, mat in materials_map.items():
        vol_cm3 = volumes_mm3.get(name, 0) / 1000.0
        if vol_cm3 > 0:
            print(f"  {name:18s}  {mat.sound_speed:8.0f}  {mat.density:11.0f}  "
                  f"{mat.attenuation:17.3f}  {vol_cm3:12.2f}")


def _compute_effective_properties(
    cortical_vol: float, trabecular_vol: float
) -> dict:
    """Compute volume-weighted effective bone properties for comparison."""
    total = cortical_vol + trabecular_vol
    if total == 0:
        return {}
    w_c = cortical_vol / total
    w_t = trabecular_vol / total
    return {
        "effective_c": w_c * CORTICAL_BONE.sound_speed + w_t * TRABECULAR_BONE.sound_speed,
        "effective_rho": w_c * CORTICAL_BONE.density + w_t * TRABECULAR_BONE.density,
        "effective_alpha": w_c * CORTICAL_BONE.attenuation + w_t * TRABECULAR_BONE.attenuation,
        "cortical_fraction": w_c,
        "trabecular_fraction": w_t,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Validate two-class bone model on Ernie Extended data")
    ap.add_argument("--input", default=str(
        Path.home() / "Data/openlifu-validation/simnibs-ernie/ErnieExtended"
        "/m2m_ernie_extended/final_tissues.nii.gz"))
    args = ap.parse_args()

    print(f"Loading Ernie Extended labels: {args.input}")
    raw_data, affine = _load_and_prep(args.input)
    voxel_vol = _voxel_volume_mm3(affine)
    print(f"  Shape: {raw_data.shape}")
    print(f"  Voxel volume: {voxel_vol:.4f} mm^3")
    print(f"  Input labels present: {np.unique(raw_data).tolist()}")

    # ----- Single-class bone model -----
    single_data = REMAP_SINGLE[raw_data]
    _report_composition(single_data, LABEL_NAMES_SINGLE, "Single-Class Bone Model (standard)")

    skull_voxels = int((single_data == 5).sum())
    skull_vol = skull_voxels * voxel_vol

    single_materials = {
        "skull": SKULL,
    }
    single_volumes = {
        "skull": skull_vol,
    }
    _report_acoustic_properties("Single-Class Bone", single_materials, single_volumes)

    # ----- Two-class bone model -----
    two_class_data = REMAP_TWO_CLASS[raw_data]
    _report_composition(two_class_data, LABEL_NAMES_TWO_CLASS, "Two-Class Bone Model (extended)")

    cortical_voxels = int((two_class_data == 5).sum())
    trabecular_voxels = int((two_class_data == 7).sum())
    cortical_vol = cortical_voxels * voxel_vol
    trabecular_vol = trabecular_voxels * voxel_vol

    two_class_materials = {
        "cortical_bone": CORTICAL_BONE,
        "trabecular_bone": TRABECULAR_BONE,
    }
    two_class_volumes = {
        "cortical_bone": cortical_vol,
        "trabecular_bone": trabecular_vol,
    }
    _report_acoustic_properties("Two-Class Bone", two_class_materials, two_class_volumes)

    # ----- Comparison -----
    eff = _compute_effective_properties(cortical_vol, trabecular_vol)
    print(f"\n{'='*60}")
    print(f"  Model Comparison")
    print(f"{'='*60}")
    print(f"  Single-class skull properties:")
    print(f"    c = {SKULL.sound_speed:.0f} m/s,  rho = {SKULL.density:.0f} kg/m3,  "
          f"alpha = {SKULL.attenuation:.2f} dB/cm/MHz")
    print(f"\n  Two-class volume-weighted effective properties:")
    print(f"    c = {eff['effective_c']:.0f} m/s,  rho = {eff['effective_rho']:.0f} kg/m3,  "
          f"alpha = {eff['effective_alpha']:.2f} dB/cm/MHz")
    print(f"    Cortical fraction: {eff['cortical_fraction']*100:.1f}%")
    print(f"    Trabecular fraction: {eff['trabecular_fraction']*100:.1f}%")

    delta_c = eff["effective_c"] - SKULL.sound_speed
    delta_alpha = eff["effective_alpha"] - SKULL.attenuation
    print(f"\n  Delta (two-class effective vs single-class):")
    print(f"    Speed of sound: {delta_c:+.0f} m/s")
    print(f"    Attenuation:    {delta_alpha:+.2f} dB/cm/MHz")

    # ----- Layer thickness analysis -----
    # Estimate average cortical and trabecular thickness along the beam path.
    # Use the fraction of bone voxels to estimate relative thickness.
    total_bone = cortical_voxels + trabecular_voxels
    if total_bone > 0:
        spacing = np.sqrt(voxel_vol)  # approximate isotropic spacing
        print(f"\n  Bone composition:")
        print(f"    Total bone voxels: {total_bone:,}")
        print(f"    Cortical: {cortical_voxels:,} ({cortical_voxels/total_bone*100:.1f}%)")
        print(f"    Trabecular: {trabecular_voxels:,} ({trabecular_voxels/total_bone*100:.1f}%)")

        # Attenuation impact estimate: for a typical 7mm skull at 500 kHz,
        # compare total attenuation through homogeneous vs layered model.
        skull_thickness_mm = 7.0  # typical average
        freq_mhz = 0.5
        cortical_frac = eff["cortical_fraction"]
        trabecular_frac = eff["trabecular_fraction"]
        cortical_thick = skull_thickness_mm * cortical_frac
        trabecular_thick = skull_thickness_mm * trabecular_frac

        atten_single = SKULL.attenuation * freq_mhz * (skull_thickness_mm / 10.0)
        atten_two_class = (
            CORTICAL_BONE.attenuation * freq_mhz * (cortical_thick / 10.0)
            + TRABECULAR_BONE.attenuation * freq_mhz * (trabecular_thick / 10.0)
        )
        print(f"\n  Estimated attenuation through {skull_thickness_mm:.0f}mm skull at {freq_mhz} MHz:")
        print(f"    Single-class:  {atten_single:.2f} dB")
        print(f"    Two-class:     {atten_two_class:.2f} dB")
        print(f"    Difference:    {atten_two_class - atten_single:+.2f} dB")

    print()


if __name__ == "__main__":
    main()
