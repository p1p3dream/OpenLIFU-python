"""Profile the internal phases of ThresholdMRI._segment().

Breaks down time within _segment:
  1. NaN check
  2. Spacing extraction from xarray coords
  3. compute_foreground_mask (Otsu + connected component + morphological closing)
  4. distance_transform_edt (foreground distance)
  5. Skull/brain mask creation
  6. Air detection
  7. Label assembly
  8. xarray output wrapping
"""
from __future__ import annotations

import os
import sys
import time

import nibabel as nib
import numpy as np
import xarray as xa
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

from openlifu.seg.skinseg import compute_foreground_mask

DATA_ROOT = os.path.expanduser("~/Data/openlifu-validation/datasets")


def profile_segment_internals(name, t1_path):
    print(f"\n{'='*60}")
    print(f"Internal _segment() breakdown: {name}")
    print(f"{'='*60}")

    t1_img = nib.load(t1_path)
    t1_data = t1_img.get_fdata().astype(np.float32)
    zooms = t1_img.header.get_zooms()[:3]
    spacing = np.array(zooms, dtype=np.float64)
    skull_thickness_mm = 7.0
    air_threshold_quantile = 0.05

    print(f"  Shape: {t1_data.shape}, size: {t1_data.size:,} voxels")

    # Phase 1: NaN check
    t0 = time.perf_counter()
    has_nan = np.isnan(t1_data).any()
    t_nan = time.perf_counter() - t0

    # Phase 2: compute_foreground_mask
    t0 = time.perf_counter()
    foreground = compute_foreground_mask(t1_data)
    t_foreground = time.perf_counter() - t0
    print(f"  Foreground voxels: {foreground.sum():,}")

    # Phase 3: Skull-stripped detection check
    t0 = time.perf_counter()
    nonzero_count = int(np.sum(t1_data > 0))
    foreground_count = int(np.sum(foreground))
    _ = foreground_count / nonzero_count if nonzero_count > 0 else 0
    t_strip_check = time.perf_counter() - t0

    # Phase 4: distance_transform_edt
    t0 = time.perf_counter()
    foreground_dist = distance_transform_edt(foreground, sampling=spacing)
    t_edt = time.perf_counter() - t0

    # Phase 5: brain mask + skull mask
    t0 = time.perf_counter()
    brain_mask = foreground_dist > skull_thickness_mm
    skull_mask = foreground & ~brain_mask
    t_masks = time.perf_counter() - t0
    print(f"  Brain voxels: {brain_mask.sum():,}")
    print(f"  Skull voxels: {skull_mask.sum():,}")

    # Phase 6: Air detection
    t0 = time.perf_counter()
    brain_intensities = t1_data[brain_mask]
    air_threshold = float(np.quantile(brain_intensities, air_threshold_quantile))
    air_mask = brain_mask & (t1_data < air_threshold)
    parenchyma_mask = brain_mask & ~air_mask
    t_air = time.perf_counter() - t0

    # Phase 7: Label assembly
    t0 = time.perf_counter()
    seg = np.full(t1_data.shape, 0, dtype=int)
    seg[skull_mask] = 1
    seg[air_mask] = 2
    seg[parenchyma_mask] = 3
    t_labels = time.perf_counter() - t0

    total = t_nan + t_foreground + t_strip_check + t_edt + t_masks + t_air + t_labels

    print(f"\n  TIMING BREAKDOWN:")
    print(f"    NaN check:              {t_nan*1000:8.1f} ms  ({t_nan/total*100:5.1f}%)")
    print(f"    compute_foreground_mask:{t_foreground*1000:8.1f} ms  ({t_foreground/total*100:5.1f}%)")
    print(f"    skull-strip detect:     {t_strip_check*1000:8.1f} ms  ({t_strip_check/total*100:5.1f}%)")
    print(f"    distance_transform_edt: {t_edt*1000:8.1f} ms  ({t_edt/total*100:5.1f}%)")
    print(f"    brain/skull masks:      {t_masks*1000:8.1f} ms  ({t_masks/total*100:5.1f}%)")
    print(f"    air detection:          {t_air*1000:8.1f} ms  ({t_air/total*100:5.1f}%)")
    print(f"    label assembly:         {t_labels*1000:8.1f} ms  ({t_labels/total*100:5.1f}%)")
    print(f"    ---")
    print(f"    TOTAL:                  {total*1000:8.1f} ms")

    # Now profile compute_foreground_mask breakdown
    print(f"\n  compute_foreground_mask breakdown:")
    import skimage.filters
    import skimage.measure

    t0 = time.perf_counter()
    threshold_lower, threshold_upper = np.quantile(t1_data, [0.02, 0.99])
    t_quantile = time.perf_counter() - t0

    t0 = time.perf_counter()
    threshold_foreground = skimage.filters.threshold_otsu(
        t1_data[(t1_data >= threshold_lower) & (t1_data <= threshold_upper)]
    )
    t_otsu = time.perf_counter() - t0

    t0 = time.perf_counter()
    fg_mask = t1_data >= threshold_foreground
    t_threshold_apply = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask_labeled = skimage.measure.label(fg_mask)
    t_label_cc = time.perf_counter() - t0

    t0 = time.perf_counter()
    cc_info = skimage.measure.regionprops(mask_labeled)
    largest = cc_info[np.argmax([rp.area for rp in cc_info])].label
    fg_mask = mask_labeled == largest
    t_largest_cc = time.perf_counter() - t0

    closing_radius = 9.0
    pad_width = int(closing_radius + 2)

    t0 = time.perf_counter()
    fg_padded = np.pad(fg_mask, pad_width, mode='constant')
    t_pad = time.perf_counter() - t0

    t0 = time.perf_counter()
    bg_edt = distance_transform_edt(~fg_padded)
    t_bg_edt = time.perf_counter() - t0

    t0 = time.perf_counter()
    fg_dilated = bg_edt <= closing_radius
    fg_dilated_edt = distance_transform_edt(fg_dilated)
    fg_closed = fg_dilated_edt >= closing_radius
    t_closing = time.perf_counter() - t0

    t0 = time.perf_counter()
    h, w, d = fg_mask.shape
    p = pad_width
    fg_cropped = fg_closed[p:p+h, p:p+w, p:p+d]
    t_crop = time.perf_counter() - t0

    t0 = time.perf_counter()
    final = ~skimage.measure.label(~fg_cropped).astype(bool)
    # Actually do it properly
    mask_bg_labeled = skimage.measure.label(~fg_cropped)
    cc_bg = skimage.measure.regionprops(mask_bg_labeled)
    largest_bg = cc_bg[np.argmax([rp.area for rp in cc_bg])].label
    final = ~(mask_bg_labeled == largest_bg)
    t_hole_fill = time.perf_counter() - t0

    total_fg = t_quantile + t_otsu + t_threshold_apply + t_label_cc + t_largest_cc + t_pad + t_bg_edt + t_closing + t_crop + t_hole_fill
    print(f"    quantile:               {t_quantile*1000:8.1f} ms")
    print(f"    otsu threshold:         {t_otsu*1000:8.1f} ms")
    print(f"    apply threshold:        {t_threshold_apply*1000:8.1f} ms")
    print(f"    label connected comp:   {t_label_cc*1000:8.1f} ms")
    print(f"    find largest CC:        {t_largest_cc*1000:8.1f} ms")
    print(f"    pad:                    {t_pad*1000:8.1f} ms")
    print(f"    background EDT:         {t_bg_edt*1000:8.1f} ms")
    print(f"    morphological closing:  {t_closing*1000:8.1f} ms")
    print(f"    crop:                   {t_crop*1000:8.1f} ms")
    print(f"    hole fill:              {t_hole_fill*1000:8.1f} ms")
    print(f"    ---")
    print(f"    TOTAL:                  {total_fg*1000:8.1f} ms")


def main():
    birnbaum_t1 = os.path.join(DATA_ROOT, "birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI/GU002_deface.nii")
    ixi_t1 = os.path.join(DATA_ROOT, "ixi-t1/IXI002-Guys-0828-T1.nii.gz")

    profile_segment_internals("Birnbaum GU002", birnbaum_t1)
    profile_segment_internals("IXI-002", ixi_t1)


if __name__ == "__main__":
    main()
