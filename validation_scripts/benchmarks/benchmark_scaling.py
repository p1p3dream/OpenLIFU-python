"""Benchmark: test worker scaling with 20 Birnbaum scans.

With 20 scans we can see true parallel scaling at different worker counts.
Also tests whether internal _segment operations can benefit from float32 or
skipping the xarray round-trip.
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
    """Original approach."""
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
    return subject_id, elapsed


def discover_birnbaum_n(n):
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
            if len(tasks) >= n:
                return tasks
    return tasks


def run_benchmark(worker_count, tasks, label):
    print(f"\n--- {label}: {len(tasks)} scans, {worker_count} workers ---")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(process_birnbaum_original, t): t[0] for t in tasks}
        completed = 0
        for future in as_completed(futures):
            sid, elapsed = future.result()
            results.append((sid, elapsed))
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                print(f"  [{completed}/{len(tasks)}] last: {sid} in {elapsed:.1f}s")

    wall_time = time.time() - t0
    total_cpu = sum(e for _, e in results)
    rate = len(tasks) / wall_time
    avg_per_scan = total_cpu / len(tasks)

    print(f"  Wall time:      {wall_time:.2f}s")
    print(f"  Total CPU:      {total_cpu:.2f}s")
    print(f"  Avg per scan:   {avg_per_scan:.2f}s")
    print(f"  Throughput:     {rate:.2f} scans/sec")
    print(f"  Efficiency:     {total_cpu/(wall_time*worker_count)*100:.0f}%")
    return wall_time, rate, avg_per_scan


def main():
    tasks = discover_birnbaum_n(20)
    print(f"Found {len(tasks)} Birnbaum scans for scaling benchmark")

    # Warmup
    print("\nWarmup run...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        f = executor.submit(process_birnbaum_original, tasks[0])
        _, elapsed = f.result()
    print(f"  Warmup done in {elapsed:.2f}s")

    results = {}
    for workers in [3, 5, 6, 8, 10, 12]:
        wall, rate, avg = run_benchmark(workers, tasks, f"{workers} workers")
        results[workers] = (wall, rate, avg)

    print(f"\n{'='*60}")
    print(f"SCALING SUMMARY ({len(tasks)} Birnbaum scans)")
    print(f"{'='*60}")
    print(f"  {'Workers':>8s}  {'Wall(s)':>8s}  {'Rate':>8s}  {'Avg/scan':>10s}  {'Speedup':>8s}")
    base_wall = results[3][0]
    for w in sorted(results):
        wall, rate, avg = results[w]
        print(f"  {w:>8d}  {wall:>8.1f}  {rate:>7.2f}/s  {avg:>9.1f}s  {base_wall/wall:>7.2f}x")


if __name__ == "__main__":
    main()
