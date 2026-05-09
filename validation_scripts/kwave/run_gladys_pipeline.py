#!/usr/bin/env python3
"""Run the full GLADYS pipeline on GU008 MRI: segmentation -> material assignment -> acoustic params."""

import os
import sys
import time

import nibabel as nib
import numpy as np
import xarray as xa

# Ensure the local openlifu package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.seg.seg_methods.nnunet_seg import NNUNetSegmentation, LABEL_MAP_FULLHEAD

# ---------------------------------------------------------------
# 1. Load GU008 MRI
# ---------------------------------------------------------------
nii_path = os.path.expanduser(
    "~/Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)
print(f"Loading MRI: {nii_path}")
img = nib.load(nii_path)
data = img.get_fdata().astype(np.float32)
affine = img.affine
print(f"  Shape: {data.shape}")
print(f"  Affine diagonal (spacing): {np.abs(np.diag(affine)[:3])}")

# ---------------------------------------------------------------
# 2. Build xarray DataArray with proper coordinates
# ---------------------------------------------------------------
spacing = np.abs(np.diag(affine)[:3])  # [sz, sy, sx]
origin = affine[:3, 3]  # [oz, oy, ox]

dims = ["z", "y", "x"]
coords = {}
for i, dim in enumerate(dims):
    coords[dim] = np.arange(data.shape[i]) * spacing[i] + origin[i]

volume = xa.DataArray(data, dims=dims, coords=coords)
print(f"  xarray dims: {volume.dims}, shape: {volume.shape}")
print(f"  Coord ranges: z=[{coords['z'][0]:.2f}, {coords['z'][-1]:.2f}], "
      f"y=[{coords['y'][0]:.2f}, {coords['y'][-1]:.2f}], "
      f"x=[{coords['x'][0]:.2f}, {coords['x'][-1]:.2f}]")

# ---------------------------------------------------------------
# 3. Create NNUNetSegmentation and run full pipeline
# ---------------------------------------------------------------
model_path = os.path.expanduser("~/.openlifu/models/fullhead_seg.onnx")
print(f"\nCreating NNUNetSegmentation (model_type='fullhead', use_mirroring=False)")
print(f"  Model: {model_path}")

seg = NNUNetSegmentation(
    model_path=model_path,
    model_type="fullhead",
    use_mirroring=False,
)

# Print the material table for reference
print("\nMaterial properties:")
material_idx = seg._material_indices()
for mat_key, mat in seg.materials.items():
    idx = material_idx[mat_key]
    print(f"  [{idx}] {mat_key:15s}: c={mat.sound_speed:.0f} m/s, "
          f"rho={mat.density:.0f} kg/m3, alpha={mat.attenuation:.3f} dB/cm/MHz")

print("\nRunning seg_params (segmentation + material assignment)...")
t0 = time.time()
params = seg.seg_params(volume)
elapsed = time.time() - t0
print(f"  Completed in {elapsed:.1f}s")

# ---------------------------------------------------------------
# 4. Print summary stats
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY STATS: Acoustic parameters across the full volume")
print("=" * 70)
for param_name in ["sound_speed", "density", "attenuation"]:
    arr = params[param_name].values
    print(f"  {param_name:20s}: min={arr.min():.2f}, max={arr.max():.2f}, "
          f"mean={arr.mean():.2f} {params[param_name].attrs.get('units', '')}")

# ---------------------------------------------------------------
# 5. Per-tissue stats (mean sound_speed for each tissue region)
# ---------------------------------------------------------------
# To get the segmentation label map, we run _segment separately
# (it was already run internally by seg_params, but we need the labels)
print("\nRunning _segment to get the label map...")
t0 = time.time()
seg_labels = seg._segment(volume)
elapsed = time.time() - t0
print(f"  Completed in {elapsed:.1f}s")

label_data = seg_labels.values
sound_speed_data = params["sound_speed"].values

print("\n" + "=" * 70)
print("PER-TISSUE STATS: Mean sound speed by tissue region")
print("=" * 70)
for mat_key, mat in seg.materials.items():
    idx = material_idx[mat_key]
    mask = label_data == idx
    voxel_count = int(mask.sum())
    if voxel_count > 0:
        mean_ss = float(sound_speed_data[mask].mean())
        pct = 100.0 * voxel_count / label_data.size
        print(f"  [{idx}] {mat_key:15s}: mean_c={mean_ss:.1f} m/s, "
              f"expected={mat.sound_speed:.1f} m/s, "
              f"voxels={voxel_count:>10,d} ({pct:.1f}%)")
    else:
        print(f"  [{idx}] {mat_key:15s}: NO VOXELS")

# Verify material assignment correctness
print("\nMATERIAL ASSIGNMENT VERIFICATION:")
all_correct = True
for mat_key, mat in seg.materials.items():
    idx = material_idx[mat_key]
    mask = label_data == idx
    if mask.sum() > 0:
        actual_ss = float(sound_speed_data[mask].mean())
        if abs(actual_ss - mat.sound_speed) > 0.01:
            print(f"  MISMATCH: {mat_key} expected {mat.sound_speed}, got {actual_ss}")
            all_correct = False
if all_correct:
    print("  All tissue regions have correct acoustic properties assigned.")

# ---------------------------------------------------------------
# 6. Save outputs as NIfTI
# ---------------------------------------------------------------
results_dir = os.path.expanduser("~/Data/openlifu-validation/results")
os.makedirs(results_dir, exist_ok=True)

# Save segmentation label map
seg_out_path = os.path.join(results_dir, "gu008_nnunet_seg.nii.gz")
seg_nii = nib.Nifti1Image(label_data.astype(np.int16), affine)
nib.save(seg_nii, seg_out_path)
print(f"\nSaved segmentation label map: {seg_out_path}")

# Save sound_speed parameter map
ss_out_path = os.path.join(results_dir, "gu008_sound_speed.nii.gz")
ss_nii = nib.Nifti1Image(sound_speed_data.astype(np.float32), affine)
nib.save(ss_nii, ss_out_path)
print(f"Saved sound speed map: {ss_out_path}")

print("\nDone.")
