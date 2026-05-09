"""Benchmark: compare different worker counts on 5 Birnbaum scans.

Measures throughput at 3 workers (baseline) vs 10 workers (optimized).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

DATA_ROOT = os.path.expanduser("~/Data/openlifu-validation/datasets")


def dice(pred, gt):
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def process_birnbaum_original(args):
    """Original approach: uses xarray, creates fresh ThresholdMRI each time."""
    import xarray as xa
    from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI

    subject_id, t1_path, label_path = args
    t0 = time.time()

    t1 = nib.load(t1_path)
    t1_data = t1.get_fdata().astype(np.float32)
    zooms = t1.header.get_zooms()[:3]
    labels = nib.load(label_path).get_fdata().astype(int)
    if labels.ndim == 4:
        labels = labels[:, :, :, 0]

    nz, ny, nx = t1_data.shape
    vol = xa.DataArray(
        t1_data, dims=["z", "y", "x"],
        coords={
            "z": np.arange(nz) * zooms[0],
            "y": np.arange(ny) * zooms[1],
            "x": np.arange(nx) * zooms[2],
        },
    )
    seg = ThresholdMRI(classify_brain_tissues=False, skull_thickness_mm=7.0)
    result = seg._segment(vol)
    seg_arr = result.to_numpy()
    idx = seg._material_indices()

    gt_bone = labels == 5
    gt_brain = np.isin(labels, [2, 3, 4])
    our_skull = seg_arr == idx["skull"]
    our_brain = seg_arr == idx["tissue"]

    elapsed = time.time() - t0
    return subject_id, elapsed, dice(our_skull, gt_bone), dice(our_brain, gt_brain)


def discover_birnbaum_5():
    """Get first 5 Birnbaum scans."""
    tasks = []
    base = os.path.join(DATA_ROOT, "birnbaum-fullhead/Data")
    for group, prefix in [("Anonymized_Subjects", "_deface"), ("Control_Subjects", "")]:
        t1_dir = os.path.join(base, group, "T1-Weighted MRI")
        seg_dir = os.path.join(base, group, "Full-Head Segmentation")
        if not os.path.isdir(t1_dir):
            continue
        for f in sorted(os.listdir(t1_dir)):
            if not f.endswith(".nii"):
                continue
            sid = f.replace("_deface.nii", "").replace(".nii", "")
            label_name = f"{sid}_label{prefix}.nii"
            label_path = os.path.join(seg_dir, label_name)
            if os.path.exists(label_path):
                tasks.append((sid, os.path.join(t1_dir, f), label_path))
            if len(tasks) >= 5:
                return tasks
    return tasks


def run_benchmark(worker_count, tasks, process_fn, label):
    print(f"\n--- {label}: {len(tasks)} scans, {worker_count} workers ---")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(process_fn, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sid, elapsed, skull_dice, brain_dice = future.result()
            results.append((sid, elapsed))
            print(f"  {sid}: {elapsed:.2f}s (skull_dice={skull_dice:.3f}, brain_dice={brain_dice:.3f})")

    wall_time = time.time() - t0
    total_cpu = sum(e for _, e in results)
    rate = len(tasks) / wall_time

    print(f"  Wall time:  {wall_time:.2f}s")
    print(f"  Total CPU:  {total_cpu:.2f}s")
    print(f"  Throughput: {rate:.2f} scans/sec")
    return wall_time, rate


def main():
    tasks = discover_birnbaum_5()
    print(f"Found {len(tasks)} Birnbaum scans for benchmarking:")
    for sid, t1, lbl in tasks:
        print(f"  {sid}")

    # Warmup: run one scan to prime imports and caches
    print("\nWarmup run (1 scan, 1 worker)...")
    warmup_t0 = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        f = executor.submit(process_birnbaum_original, tasks[0])
        sid, elapsed, _, _ = f.result()
    print(f"  Warmup done in {time.time()-warmup_t0:.2f}s")

    # Baseline: 3 workers
    wall_3, rate_3 = run_benchmark(3, tasks, process_birnbaum_original, "BASELINE (3 workers)")

    # Optimized: 10 workers
    wall_10, rate_10 = run_benchmark(10, tasks, process_birnbaum_original, "OPTIMIZED (10 workers)")

    # Also test 6 and 14 for comparison
    wall_6, rate_6 = run_benchmark(6, tasks, process_birnbaum_original, "6 workers")
    wall_14, rate_14 = run_benchmark(14, tasks, process_birnbaum_original, "14 workers")

    print(f"\n{'='*60}")
    print(f"SUMMARY (5 Birnbaum scans)")
    print(f"{'='*60}")
    print(f"   3 workers: {wall_3:.2f}s wall, {rate_3:.2f} scans/sec")
    print(f"   6 workers: {wall_6:.2f}s wall, {rate_6:.2f} scans/sec")
    print(f"  10 workers: {wall_10:.2f}s wall, {rate_10:.2f} scans/sec")
    print(f"  14 workers: {wall_14:.2f}s wall, {rate_14:.2f} scans/sec")
    print(f"  Speedup (3 -> 10): {wall_3/wall_10:.2f}x")
    print(f"  Speedup (3 -> 14): {wall_3/wall_14:.2f}x")


if __name__ == "__main__":
    main()
