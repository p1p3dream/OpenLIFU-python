"""Test potential optimizations to the EDT bottleneck.

The morphological closing in compute_foreground_mask uses EDT twice on a
padded volume. We test:
1. Whether using uint8 vs bool input matters for EDT speed
2. Whether scipy.ndimage.binary_dilation/erosion would be faster
3. Whether reducing closing_radius helps (e.g. from 9 to 5)
4. Whether the padded volume size matters significantly
"""
from __future__ import annotations

import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

DATA_ROOT = os.path.expanduser("~/Data/openlifu-validation/datasets")


def main():
    t1_path = os.path.join(DATA_ROOT, "birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI/GU002_deface.nii")
    t1 = nib.load(t1_path)
    t1_data = t1.get_fdata().astype(np.float32)

    # Create a foreground mask via simple thresholding (skip the full pipeline)
    import skimage.filters, skimage.measure
    tl, tu = np.quantile(t1_data, [0.02, 0.99])
    thresh = skimage.filters.threshold_otsu(t1_data[(t1_data >= tl) & (t1_data <= tu)])
    fg = t1_data >= thresh
    mask_labeled = skimage.measure.label(fg)
    cc = skimage.measure.regionprops(mask_labeled)
    largest = cc[np.argmax([rp.area for rp in cc])].label
    fg = (mask_labeled == largest)

    print(f"Volume shape: {t1_data.shape}")
    print(f"Foreground voxels: {fg.sum():,}")

    # Test 1: EDT on bool vs uint8
    fg_bool = fg.astype(bool)
    fg_uint8 = fg.astype(np.uint8)

    print("\n--- EDT input dtype comparison ---")
    t0 = time.perf_counter()
    edt_bool = distance_transform_edt(~fg_bool)
    t_bool = time.perf_counter() - t0
    print(f"  bool input:  {t_bool*1000:.1f} ms")

    t0 = time.perf_counter()
    edt_uint8 = distance_transform_edt(~fg_uint8)
    t_uint8 = time.perf_counter() - t0
    print(f"  uint8 input: {t_uint8*1000:.1f} ms")

    # Test 2: Effect of padding on EDT speed
    print("\n--- Padding overhead for morphological closing ---")
    closing_radius = 9.0
    pad_width = int(closing_radius + 2)

    fg_padded = np.pad(fg_bool, pad_width, mode='constant')
    print(f"  Original shape: {fg_bool.shape}, padded: {fg_padded.shape}")
    print(f"  Size ratio: {fg_padded.size / fg_bool.size:.2f}x")

    t0 = time.perf_counter()
    _ = distance_transform_edt(~fg_padded)
    t_padded = time.perf_counter() - t0
    print(f"  EDT on padded (~): {t_padded*1000:.1f} ms")

    t0 = time.perf_counter()
    _ = distance_transform_edt(~fg_bool)
    t_unpadded = time.perf_counter() - t0
    print(f"  EDT on unpadded: {t_unpadded*1000:.1f} ms")

    # Test 3: Full morphological closing with different radii
    print("\n--- Closing radius comparison ---")
    for radius in [5.0, 7.0, 9.0]:
        pw = int(radius + 2)
        padded = np.pad(fg_bool, pw, mode='constant')
        t0 = time.perf_counter()
        bg_edt = distance_transform_edt(~padded)
        dilated = bg_edt <= radius
        dilated_edt = distance_transform_edt(dilated)
        closed = dilated_edt >= radius
        t_close = time.perf_counter() - t0
        print(f"  radius={radius}: {t_close*1000:.1f} ms")

    # Test 4: scipy binary morphology as alternative to EDT-based closing
    print("\n--- Binary morphology vs EDT closing ---")
    from scipy.ndimage import generate_binary_structure, iterate_structure

    # Binary closing with iterations = radius (in voxels)
    struct = generate_binary_structure(3, 1)  # 6-connectivity
    iterations = 9

    t0 = time.perf_counter()
    closed_binary = binary_dilation(fg_bool, structure=struct, iterations=iterations)
    closed_binary = binary_erosion(closed_binary, structure=struct, iterations=iterations)
    t_binary = time.perf_counter() - t0
    print(f"  Binary closing (iters={iterations}): {t_binary*1000:.1f} ms")

    # Compare: EDT-based closing with radius=9
    t0 = time.perf_counter()
    pw = int(9 + 2)
    padded = np.pad(fg_bool, pw, mode='constant')
    bg_edt = distance_transform_edt(~padded)
    dilated = bg_edt <= 9
    dilated_edt = distance_transform_edt(dilated)
    closed_edt = dilated_edt >= 9
    h, w, d = fg_bool.shape
    closed_edt = closed_edt[pw:pw+h, pw:pw+w, pw:pw+d]
    t_edt_close = time.perf_counter() - t0
    print(f"  EDT closing (radius=9):      {t_edt_close*1000:.1f} ms")

    # Test 5: Whether distance_transform_edt with sampling= is slower
    print("\n--- EDT with vs without sampling ---")
    spacing = np.array([1.0, 1.0, 1.0])
    t0 = time.perf_counter()
    _ = distance_transform_edt(fg_bool, sampling=spacing)
    t_sampled = time.perf_counter() - t0
    print(f"  With sampling=[1,1,1]: {t_sampled*1000:.1f} ms")

    t0 = time.perf_counter()
    _ = distance_transform_edt(fg_bool)
    t_no_sample = time.perf_counter() - t0
    print(f"  Without sampling:      {t_no_sample*1000:.1f} ms")

    aniso = np.array([0.97, 1.0, 1.0])
    t0 = time.perf_counter()
    _ = distance_transform_edt(fg_bool, sampling=aniso)
    t_aniso = time.perf_counter() - t0
    print(f"  With sampling=aniso:   {t_aniso*1000:.1f} ms")


if __name__ == "__main__":
    main()
