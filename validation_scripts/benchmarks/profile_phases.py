"""Profile time spent in each phase of a single ThresholdMRI scan.

Tests one Birnbaum scan and one IXI scan, timing:
  1. NIfTI loading (nib.load + get_fdata)
  2. xarray DataArray creation
  3. ThresholdMRI._segment() total
  4. Result extraction and Dice computation
"""
from __future__ import annotations

import os
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

DATA_ROOT = os.path.expanduser("~/Data/openlifu-validation/datasets")


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def profile_scan(name, t1_path, label_path=None, mmap_mode=False):
    print(f"\n{'='*60}")
    print(f"Profiling: {name}")
    print(f"  mmap={mmap_mode}")
    print(f"{'='*60}")

    # Phase 1: NIfTI loading
    t0 = time.perf_counter()
    if mmap_mode:
        t1_img = nib.load(t1_path, mmap='r')
    else:
        t1_img = nib.load(t1_path)
    t_load_header = time.perf_counter() - t0

    t0 = time.perf_counter()
    t1_data_raw = t1_img.get_fdata()
    t_get_fdata = time.perf_counter() - t0

    print(f"  Raw dtype from get_fdata: {t1_data_raw.dtype}")
    print(f"  Shape: {t1_data_raw.shape}")
    print(f"  Memory (raw): {t1_data_raw.nbytes / 1e6:.1f} MB")

    t0 = time.perf_counter()
    t1_data = t1_data_raw.astype(np.float32)
    t_cast32 = time.perf_counter() - t0

    t0 = time.perf_counter()
    t1_data_f64 = t1_data_raw.astype(np.float64)
    t_cast64 = time.perf_counter() - t0

    print(f"  Memory (float32): {t1_data.nbytes / 1e6:.1f} MB")
    print(f"  Memory (float64): {t1_data_f64.nbytes / 1e6:.1f} MB")

    zooms = t1_img.header.get_zooms()[:3]

    if label_path:
        t0 = time.perf_counter()
        labels = nib.load(label_path).get_fdata().astype(int)
        if labels.ndim == 4:
            labels = labels[:, :, :, 0]
        t_load_labels = time.perf_counter() - t0
    else:
        t_load_labels = 0.0

    # Phase 2: xarray DataArray creation
    import xarray as xa
    nz, ny, nx = t1_data.shape

    t0 = time.perf_counter()
    vol = xa.DataArray(
        t1_data, dims=["z", "y", "x"],
        coords={
            "z": np.arange(nz) * zooms[0],
            "y": np.arange(ny) * zooms[1],
            "x": np.arange(nx) * zooms[2],
        },
    )
    t_xarray_create = time.perf_counter() - t0

    # Phase 3: ThresholdMRI._segment() total
    from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI

    seg_obj = ThresholdMRI(
        classify_brain_tissues=False,
        skull_thickness_mm=7.0,
    )

    t0 = time.perf_counter()
    result = seg_obj._segment(vol)
    t_segment = time.perf_counter() - t0

    # Phase 3b: Time internal .to_numpy() at the start of _segment
    t0 = time.perf_counter()
    _ = vol.to_numpy()
    t_to_numpy = time.perf_counter() - t0

    # Phase 3c: Time wrapping output back to xarray
    seg_np = result.to_numpy()
    t0 = time.perf_counter()
    _ = xa.DataArray(seg_np, coords=vol.coords, dims=vol.dims)
    t_xarray_wrap_output = time.perf_counter() - t0

    # Phase 4: Result extraction and Dice
    t0 = time.perf_counter()
    seg_arr = result.to_numpy()
    idx = seg_obj._material_indices()
    our_skull = seg_arr == idx["skull"]
    our_brain = seg_arr == idx["tissue"]
    if label_path:
        labels_int = nib.load(label_path).get_fdata().astype(int)
        if labels_int.ndim == 4:
            labels_int = labels_int[:, :, :, 0]
        gt_bone = labels_int == 5
        gt_brain = np.isin(labels_int, [2, 3, 4])
        skull_d = dice(our_skull, gt_bone)
        brain_d = dice(our_brain, gt_brain)
    t_dice = time.perf_counter() - t0

    # Print results
    print(f"\n  TIMING BREAKDOWN:")
    print(f"    nib.load (header):      {t_load_header*1000:8.1f} ms")
    print(f"    get_fdata:              {t_get_fdata*1000:8.1f} ms")
    print(f"    .astype(float32):       {t_cast32*1000:8.1f} ms")
    print(f"    .astype(float64):       {t_cast64*1000:8.1f} ms")
    if label_path:
        print(f"    load labels:            {t_load_labels*1000:8.1f} ms")
    print(f"    xarray create:          {t_xarray_create*1000:8.1f} ms")
    print(f"    vol.to_numpy():         {t_to_numpy*1000:8.1f} ms")
    print(f"    _segment() TOTAL:       {t_segment*1000:8.1f} ms")
    print(f"    xarray wrap output:     {t_xarray_wrap_output*1000:8.1f} ms")
    print(f"    result extract + Dice:  {t_dice*1000:8.1f} ms")
    total_overhead = t_load_header + t_get_fdata + t_cast32 + t_xarray_create + t_to_numpy + t_xarray_wrap_output
    print(f"    ---")
    print(f"    I/O + overhead total:   {total_overhead*1000:8.1f} ms")
    print(f"    _segment() alone:       {t_segment*1000:8.1f} ms")
    print(f"    segment % of total:     {t_segment/(total_overhead+t_segment)*100:8.1f}%")

    # Phase 5: Test ThresholdMRI object reuse
    print(f"\n  OBJECT REUSE TEST:")
    seg_obj2 = ThresholdMRI(classify_brain_tissues=False, skull_thickness_mm=7.0)
    t0 = time.perf_counter()
    _ = seg_obj2._segment(vol)
    t_seg_fresh = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = seg_obj._segment(vol)  # reuse the first object
    t_seg_reuse = time.perf_counter() - t0
    print(f"    Fresh ThresholdMRI:     {t_seg_fresh*1000:8.1f} ms")
    print(f"    Reused ThresholdMRI:    {t_seg_reuse*1000:8.1f} ms")

    return {
        "name": name,
        "load_header": t_load_header,
        "get_fdata": t_get_fdata,
        "cast32": t_cast32,
        "load_labels": t_load_labels,
        "xarray_create": t_xarray_create,
        "to_numpy": t_to_numpy,
        "segment": t_segment,
        "xarray_wrap_output": t_xarray_wrap_output,
        "dice": t_dice,
    }


def main():
    # Birnbaum scan
    birnbaum_t1 = os.path.join(DATA_ROOT, "birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI/GU002_deface.nii")
    birnbaum_label = os.path.join(DATA_ROOT, "birnbaum-fullhead/Data/Anonymized_Subjects/Full-Head Segmentation/GU002_label_deface.nii")

    # IXI scan
    ixi_t1 = os.path.join(DATA_ROOT, "ixi-t1/IXI002-Guys-0828-T1.nii.gz")

    # Profile without mmap
    profile_scan("Birnbaum GU002", birnbaum_t1, birnbaum_label, mmap_mode=False)
    profile_scan("IXI-002", ixi_t1, mmap_mode=False)

    # Profile with mmap
    profile_scan("Birnbaum GU002 (mmap)", birnbaum_t1, birnbaum_label, mmap_mode=True)
    profile_scan("IXI-002 (mmap)", ixi_t1, mmap_mode=True)


if __name__ == "__main__":
    main()
