"""Definitive benchmark: 15 Birnbaum scans, 12 vs 6 workers.

With 15 scans, 12 workers processes them in ~2 batches while 6 workers
processes in ~3 batches, exposing the memory bandwidth contention difference.
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
    elapsed = time.time() - t0
    return subject_id, elapsed


def discover_birnbaum(n):
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


def run_trial(worker_count, tasks):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(process_birnbaum, t): t[0] for t in tasks}
        per_scan = []
        for future in as_completed(futures):
            sid, elapsed = future.result()
            per_scan.append(elapsed)
    wall_time = time.time() - t0
    return wall_time, np.mean(per_scan), np.sum(per_scan)


def main():
    tasks = discover_birnbaum(15)
    print(f"Benchmark: {len(tasks)} Birnbaum scans")

    # Warmup
    print("Warmup...")
    with ProcessPoolExecutor(max_workers=1) as ex:
        ex.submit(process_birnbaum, tasks[0]).result()

    for workers in [12, 6, 3]:
        print(f"\n--- {workers} workers ---")
        wall, avg, total_cpu = run_trial(workers, tasks)
        rate = len(tasks) / wall
        print(f"  Wall: {wall:.1f}s, Rate: {rate:.2f}/s, Avg/scan: {avg:.1f}s, CPU total: {total_cpu:.0f}s")


if __name__ == "__main__":
    main()
