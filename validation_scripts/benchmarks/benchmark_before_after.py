"""Before/after benchmark: 5 Birnbaum scans comparing old vs new defaults.

Old default: 12 workers (min(12, cpu_count))
New default: 6 workers  (min(6, cpu_count))

Runs each configuration twice and takes the best time.
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


def process_birnbaum(args):
    """Standard Birnbaum processing (same as parallel_runner.py)."""
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


def run_trial(worker_count, tasks, label):
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(process_birnbaum, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sid, elapsed, skull_d, brain_d = future.result()
            results.append((sid, elapsed, skull_d, brain_d))
    wall_time = time.time() - t0
    total_cpu = sum(e for _, e, _, _ in results)
    rate = len(tasks) / wall_time
    avg = total_cpu / len(tasks)
    return wall_time, rate, avg, results


def main():
    tasks = discover_birnbaum_5()
    print(f"Before/After Benchmark: {len(tasks)} Birnbaum scans")
    print(f"Scans: {[t[0] for t in tasks]}")

    # Warmup
    print("\nWarmup...")
    with ProcessPoolExecutor(max_workers=1) as ex:
        f = ex.submit(process_birnbaum, tasks[0])
        f.result()
    print("  done")

    # Run each config twice, take the best wall time
    configs = [
        (12, "BEFORE (12 workers, old default)"),
        (6, "AFTER (6 workers, new default)"),
    ]

    best = {}
    for workers, label in configs:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        best_wall = float('inf')
        for trial in range(2):
            wall, rate, avg, results = run_trial(workers, tasks, label)
            print(f"  Trial {trial+1}: {wall:.2f}s wall, {rate:.2f}/s, {avg:.1f}s avg/scan")
            for sid, elapsed, skull_d, brain_d in sorted(results):
                print(f"    {sid}: {elapsed:.1f}s")
            if wall < best_wall:
                best_wall = wall
                best_rate = rate
                best_avg = avg
        best[workers] = (best_wall, best_rate, best_avg)
        print(f"  Best: {best_wall:.2f}s wall, {best_rate:.2f} scans/sec")

    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON (5 Birnbaum scans, best of 2 trials)")
    print(f"{'='*60}")
    w12, r12, a12 = best[12]
    w6, r6, a6 = best[6]
    print(f"  BEFORE (12 workers): {w12:.2f}s wall, {r12:.2f} scans/sec, {a12:.1f}s avg/scan")
    print(f"  AFTER  (6 workers):  {w6:.2f}s wall, {r6:.2f} scans/sec, {a6:.1f}s avg/scan")
    print(f"  Speedup: {w12/w6:.2f}x wall time, {r6/r12:.2f}x throughput")


if __name__ == "__main__":
    main()
