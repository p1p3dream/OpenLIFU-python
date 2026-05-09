"""Parallel validation runner for ThresholdMRI across all datasets.

Uses ProcessPoolExecutor to run segmentation + Dice computation in parallel.

Performance notes (14-core Apple Silicon, 48GB RAM):
  - The bottleneck is scipy.ndimage.distance_transform_edt inside
    compute_foreground_mask(), which is memory-bandwidth-bound.
  - Optimal worker count is ~6 (not 12-14). Beyond 6, per-scan time
    degrades due to memory bandwidth contention, and wall-clock time
    does not improve. Empirical scaling on 20 Birnbaum scans:
      3 workers: 0.14 scans/sec (19s avg/scan)
      6 workers: 0.20 scans/sec (24s avg/scan)  [sweet spot]
      10 workers: 0.20 scans/sec (37s avg/scan)
      12 workers: 0.18 scans/sec (47s avg/scan)  [worse than 6]
  - Within _segment(), 97-99% of time is in the segmentation itself:
      compute_foreground_mask: 73-76% (EDT-based morphological closing)
      distance_transform_edt (skull boundary): ~20%
      everything else (xarray, I/O, Dice): <3%

Usage:
    python parallel_runner.py                    # Run all datasets
    python parallel_runner.py birnbaum           # Run one dataset
    python parallel_runner.py birnbaum ernie     # Run specific datasets
    python parallel_runner.py --workers 8        # Override parallelism
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

# Add OpenLIFU to path
sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))

DATA_ROOT = Path(os.path.expanduser("~/Data/openlifu-validation/datasets"))
RESULTS_DIR = Path(os.path.expanduser("~/Data/openlifu-validation/results"))


@dataclass
class ScanResult:
    dataset: str
    subject_id: str
    shape: tuple
    voxel_size: tuple
    skull_dice: float | None = None
    brain_dice: float | None = None
    gm_dice: float | None = None
    wm_dice: float | None = None
    csf_dice: float | None = None
    skull_voxels: int = 0
    brain_voxels: int = 0
    time_seconds: float = 0.0
    error: str | None = None
    extra: dict = field(default_factory=dict)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def run_threshold_mri(t1_data, zooms, classify_brain=False, skull_thickness_mm=7.0):
    """Run ThresholdMRI on a volume. Returns (result_array, material_indices)."""
    import xarray as xa
    from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI

    nz, ny, nx = t1_data.shape
    vol = xa.DataArray(
        t1_data, dims=["z", "y", "x"],
        coords={
            "z": np.arange(nz) * zooms[0],
            "y": np.arange(ny) * zooms[1],
            "x": np.arange(nx) * zooms[2],
        },
    )
    seg = ThresholdMRI(
        classify_brain_tissues=classify_brain,
        skull_thickness_mm=skull_thickness_mm,
    )
    result = seg._segment(vol)
    idx = seg._material_indices()
    return result.to_numpy(), idx


# ---- Dataset-specific processing functions ----
# Each returns a list of (subject_id, t1_path, gt_path_or_none, thickness, classify_brain) tuples


def process_birnbaum_scan(args) -> ScanResult:
    subject_id, t1_path, label_path, thickness, classify_brain = args
    try:
        t0 = time.time()
        t1 = nib.load(t1_path)
        t1_data = t1.get_fdata().astype(np.float32)
        zooms = t1.header.get_zooms()[:3]
        labels = nib.load(label_path).get_fdata().astype(int)
        if labels.ndim == 4:
            labels = labels[:, :, :, 0]

        seg_arr, idx = run_threshold_mri(t1_data, zooms, classify_brain=classify_brain, skull_thickness_mm=thickness)

        gt_bone = labels == 5
        gt_brain = np.isin(labels, [2, 3, 4])
        gt_gm = labels == 3
        gt_wm = labels == 4
        gt_csf = labels == 2
        gt_soft = labels == 6
        gt_skull_shell = np.isin(labels, [5, 6])  # bone + soft tissue

        our_skull = seg_arr == idx["skull"]

        # In 6-label mode, compute per-tissue Dice against individual labels
        if classify_brain:
            our_gm = seg_arr == idx["gray_matter"]
            our_wm = seg_arr == idx["white_matter"]
            our_csf = seg_arr == idx["csf"]
            our_brain = our_gm | our_wm | our_csf
            gm_dice_val = dice(our_gm, gt_gm)
            wm_dice_val = dice(our_wm, gt_wm)
            csf_dice_val = dice(our_csf, gt_csf)
        else:
            our_brain = seg_arr == idx["tissue"]
            gm_dice_val = dice(our_brain, gt_gm)
            wm_dice_val = dice(our_brain, gt_wm)
            csf_dice_val = dice(our_brain, gt_csf)

        elapsed = time.time() - t0
        return ScanResult(
            dataset="birnbaum",
            subject_id=subject_id,
            shape=t1_data.shape,
            voxel_size=tuple(float(z) for z in zooms),
            skull_dice=dice(our_skull, gt_bone),
            brain_dice=dice(our_brain, gt_brain),
            gm_dice=gm_dice_val,
            wm_dice=wm_dice_val,
            csf_dice=csf_dice_val,
            skull_voxels=int(our_skull.sum()),
            brain_voxels=int(our_brain.sum()),
            time_seconds=elapsed,
            extra={
                "skull_vs_shell_dice": dice(our_skull, gt_skull_shell),
                "gt_bone_voxels": int(gt_bone.sum()),
                "gt_soft_voxels": int(gt_soft.sum()),
                "classify_brain": classify_brain,
            },
        )
    except Exception as e:
        return ScanResult(
            dataset="birnbaum", subject_id=subject_id,
            shape=(0, 0, 0), voxel_size=(0, 0, 0), error=str(e),
        )


def process_ernie_scan(args) -> ScanResult:
    _, _, _, thickness, classify_brain = args
    try:
        t0 = time.time()
        base = DATA_ROOT / "ernie" / "m2m_ernie"
        t1 = nib.load(str(base / "T1.nii.gz"))
        gt = nib.load(str(base / "final_tissues.nii.gz"))
        t1_data = t1.get_fdata().astype(np.float32)
        gt_data = gt.get_fdata().astype(int)
        if gt_data.ndim == 4:
            gt_data = gt_data[:, :, :, 0]
        zooms = t1.header.get_zooms()[:3]

        seg_arr, idx = run_threshold_mri(t1_data, zooms, classify_brain=classify_brain, skull_thickness_mm=thickness)

        our_skull = seg_arr == idx["skull"]

        gt_bone = gt_data == 4
        gt_shell = np.isin(gt_data, [4, 5])
        gt_brain = np.isin(gt_data, [1, 2, 3])
        gt_wm = gt_data == 1
        gt_gm = gt_data == 2
        gt_csf = gt_data == 3

        if classify_brain:
            our_gm = seg_arr == idx["gray_matter"]
            our_wm = seg_arr == idx["white_matter"]
            our_csf = seg_arr == idx["csf"]
            our_brain = our_gm | our_wm | our_csf
        else:
            our_brain = seg_arr == idx["tissue"]

        elapsed = time.time() - t0
        extra = {
            "bone_only_dice": dice(our_skull, gt_bone),
            "gt_bone_voxels": int(gt_bone.sum()),
            "gt_scalp_voxels": int((gt_data == 5).sum()),
            "classify_brain": classify_brain,
        }
        if classify_brain:
            extra["wm_dice"] = dice(our_wm, gt_wm)
            extra["gm_dice"] = dice(our_gm, gt_gm)
            extra["csf_dice"] = dice(our_csf, gt_csf)

        return ScanResult(
            dataset="ernie", subject_id="ernie",
            shape=t1_data.shape, voxel_size=tuple(float(z) for z in zooms),
            skull_dice=dice(our_skull, gt_shell),
            brain_dice=dice(our_brain, gt_brain),
            gm_dice=extra.get("gm_dice"),
            wm_dice=extra.get("wm_dice"),
            csf_dice=extra.get("csf_dice"),
            skull_voxels=int(our_skull.sum()),
            brain_voxels=int(our_brain.sum()),
            time_seconds=elapsed,
            extra=extra,
        )
    except Exception as e:
        return ScanResult(
            dataset="ernie", subject_id="ernie",
            shape=(0, 0, 0), voxel_size=(0, 0, 0), error=str(e),
        )


def process_ixi_scan(args) -> ScanResult:
    subject_id, t1_path, _, thickness, classify_brain = args
    try:
        t0 = time.time()
        t1 = nib.load(t1_path)
        t1_data = t1.get_fdata().astype(np.float32)
        zooms = t1.header.get_zooms()[:3]

        seg_arr, idx = run_threshold_mri(t1_data, zooms, classify_brain=classify_brain, skull_thickness_mm=thickness)

        our_skull = seg_arr == idx["skull"]
        if classify_brain:
            our_brain = (seg_arr == idx["gray_matter"]) | (seg_arr == idx["white_matter"]) | (seg_arr == idx["csf"])
        else:
            our_brain = seg_arr == idx["tissue"]
        our_water = seg_arr == idx["water"]
        our_air = seg_arr == idx["air"]

        elapsed = time.time() - t0
        total = t1_data.size
        return ScanResult(
            dataset="ixi", subject_id=subject_id,
            shape=t1_data.shape, voxel_size=tuple(float(z) for z in zooms),
            skull_voxels=int(our_skull.sum()),
            brain_voxels=int(our_brain.sum()),
            time_seconds=elapsed,
            extra={
                "skull_pct": float(our_skull.sum()) / total * 100,
                "brain_pct": float(our_brain.sum()) / total * 100,
                "water_pct": float(our_water.sum()) / total * 100,
                "air_pct": float(our_air.sum()) / total * 100,
            },
        )
    except Exception as e:
        return ScanResult(
            dataset="ixi", subject_id=subject_id,
            shape=(0, 0, 0), voxel_size=(0, 0, 0), error=str(e),
        )


def process_neurite_scan(args) -> ScanResult:
    subject_id, orig_path, seg4_path, thickness, classify_brain = args
    try:
        t0 = time.time()
        orig = nib.load(orig_path)
        t1_data = orig.get_fdata().astype(np.float32)
        zooms = orig.header.get_zooms()[:3]
        seg4 = nib.load(seg4_path).get_fdata().astype(int)
        if seg4.ndim == 4:
            seg4 = seg4[:, :, :, 0]

        seg_arr, idx = run_threshold_mri(t1_data, zooms, classify_brain=classify_brain, skull_thickness_mm=thickness)

        # seg4: 1=cortex(GM), 2=subcortical GM, 3=WM, 4=CSF/ventricles
        gt_brain = seg4 > 0
        gt_gm = np.isin(seg4, [1, 2])
        gt_wm = seg4 == 3
        gt_csf = seg4 == 4

        if classify_brain:
            our_gm = seg_arr == idx["gray_matter"]
            our_wm = seg_arr == idx["white_matter"]
            our_csf = seg_arr == idx["csf"]
            our_brain = our_gm | our_wm | our_csf
            gm_dice_val = dice(our_gm, gt_gm)
            wm_dice_val = dice(our_wm, gt_wm)
            csf_dice_val = dice(our_csf, gt_csf)
        else:
            our_brain = seg_arr == idx["tissue"]
            gm_dice_val = dice(our_brain & gt_gm, gt_gm)
            wm_dice_val = dice(our_brain & gt_wm, gt_wm)
            csf_dice_val = dice(our_brain & gt_csf, gt_csf)

        elapsed = time.time() - t0
        return ScanResult(
            dataset="neurite-oasis", subject_id=subject_id,
            shape=t1_data.shape, voxel_size=tuple(float(z) for z in zooms),
            brain_dice=dice(our_brain, gt_brain),
            gm_dice=gm_dice_val,
            wm_dice=wm_dice_val,
            csf_dice=csf_dice_val,
            brain_voxels=int(our_brain.sum()),
            time_seconds=elapsed,
        )
    except Exception as e:
        return ScanResult(
            dataset="neurite-oasis", subject_id=subject_id,
            shape=(0, 0, 0), voxel_size=(0, 0, 0), error=str(e),
        )


def process_synthstrip_scan(args) -> ScanResult:
    subject_id, img_path, mask_path, thickness, classify_brain = args
    try:
        t0 = time.time()
        img = nib.load(img_path)
        t1_data = img.get_fdata().astype(np.float32)
        zooms = img.header.get_zooms()[:3]
        gt_mask = nib.load(mask_path).get_fdata().astype(bool)
        if gt_mask.ndim == 4:
            gt_mask = gt_mask[:, :, :, 0]

        seg_arr, idx = run_threshold_mri(t1_data, zooms, classify_brain=classify_brain, skull_thickness_mm=thickness)

        if classify_brain:
            our_brain = (seg_arr == idx["gray_matter"]) | (seg_arr == idx["white_matter"]) | (seg_arr == idx["csf"])
        else:
            our_brain = seg_arr == idx["tissue"]

        elapsed = time.time() - t0
        return ScanResult(
            dataset="synthstrip", subject_id=subject_id,
            shape=t1_data.shape, voxel_size=tuple(float(z) for z in zooms),
            brain_dice=dice(our_brain, gt_mask),
            brain_voxels=int(our_brain.sum()),
            time_seconds=elapsed,
            extra={"gt_brain_voxels": int(gt_mask.sum())},
        )
    except Exception as e:
        return ScanResult(
            dataset="synthstrip", subject_id=subject_id,
            shape=(0, 0, 0), voxel_size=(0, 0, 0), error=str(e),
        )


# ---- Dataset discovery ----

def discover_birnbaum(thickness, classify_brain):
    tasks = []
    base = DATA_ROOT / "birnbaum-fullhead" / "Data"
    for group, prefix in [("Anonymized_Subjects", "_deface"), ("Control_Subjects", "")]:
        t1_dir = base / group / "T1-Weighted MRI"
        seg_dir = base / group / "Full-Head Segmentation"
        if not t1_dir.exists():
            continue
        for f in sorted(t1_dir.glob("*.nii")):
            sid = f.stem.replace("_deface", "")
            label_name = f"{sid}_label{prefix}.nii"
            label_path = seg_dir / label_name
            if label_path.exists():
                tasks.append((sid, str(f), str(label_path), thickness, classify_brain))
    return tasks, process_birnbaum_scan


def discover_ernie(thickness, classify_brain):
    return [("ernie", "", "", thickness, classify_brain)], process_ernie_scan


def discover_ixi(thickness, classify_brain):
    tasks = []
    ixi_dir = DATA_ROOT / "ixi-t1"
    for f in sorted(ixi_dir.glob("*.nii.gz")):
        sid = f.stem.replace(".nii", "")
        tasks.append((sid, str(f), None, thickness, classify_brain))
    return tasks, process_ixi_scan


def discover_neurite(thickness, classify_brain):
    tasks = []
    base = DATA_ROOT / "neurite-oasis"
    for d in sorted(base.glob("OASIS_OAS1_*")):
        orig = d / "orig.nii.gz"
        seg4 = d / "seg4.nii.gz"
        if orig.exists() and seg4.exists():
            tasks.append((d.name, str(orig), str(seg4), thickness, classify_brain))
    return tasks, process_neurite_scan


def discover_synthstrip(thickness, classify_brain):
    tasks = []
    base = DATA_ROOT / "synthstrip"
    for d in sorted(base.iterdir()):
        img = d / "image.nii.gz"
        mask = d / "mask.nii.gz"
        if img.exists() and mask.exists():
            tasks.append((d.name, str(img), str(mask), thickness, classify_brain))
    return tasks, process_synthstrip_scan


DATASETS = {
    "birnbaum": discover_birnbaum,
    "ernie": discover_ernie,
    "ixi": discover_ixi,
    "neurite": discover_neurite,
    "synthstrip": discover_synthstrip,
}


def run_dataset(name, workers, thickness, classify_brain=False):
    discover_fn = DATASETS[name]
    tasks, process_fn = discover_fn(thickness, classify_brain)
    if not tasks:
        print(f"  No scans found for {name}")
        return []

    mode = "6-label" if classify_brain else "4-label"
    print(f"  {name}: {len(tasks)} scans, {workers} workers, thickness={thickness}mm, {mode}")
    results = []
    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_fn, t): t[0] for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / elapsed
                print(f"    [{completed}/{len(tasks)}] {rate:.1f} scans/sec"
                      + (f", last error: {r.error}" if r.error else ""))

    elapsed = time.time() - t0
    errors = sum(1 for r in results if r.error)
    print(f"  {name} done: {len(results)} scans in {elapsed:.1f}s "
          f"({len(results)/elapsed:.1f}/sec), {errors} errors")
    return results


def summarize(results: list[ScanResult], dataset_name: str):
    valid = [r for r in results if not r.error]
    if not valid:
        print(f"\n{dataset_name}: No valid results")
        return

    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()}: {len(valid)} scans ({len(results)-len(valid)} errors)")
    print(f"{'='*60}")

    for metric in ["skull_dice", "brain_dice", "gm_dice", "wm_dice", "csf_dice"]:
        vals = [getattr(r, metric) for r in valid if getattr(r, metric) is not None]
        if vals:
            arr = np.array(vals)
            print(f"  {metric:>12s}: mean={arr.mean():.3f} std={arr.std():.3f} "
                  f"min={arr.min():.3f} max={arr.max():.3f}")

    times = [r.time_seconds for r in valid]
    print(f"  {'time_sec':>12s}: mean={np.mean(times):.1f} total={np.sum(times):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Parallel ThresholdMRI validation")
    parser.add_argument("datasets", nargs="*", default=list(DATASETS.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 4),
                        help="Number of parallel workers (default: 6, empirically optimal for EDT-bound workloads)")
    parser.add_argument("--thickness", type=float, default=7.0,
                        help="skull_thickness_mm parameter (default: 7.0)")
    parser.add_argument("--classify-brain", action="store_true", default=False,
                        help="Use 6-label brain tissue classification (default: 4-label)")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output file (default: results/parallel_results.json)")
    args = parser.parse_args()

    if args.output is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        args.output = str(RESULTS_DIR / "parallel_results.json")

    mode = "6-label" if args.classify_brain else "4-label"
    print(f"ThresholdMRI Parallel Validation Runner")
    print(f"  Workers: {args.workers}")
    print(f"  Thickness: {args.thickness}mm")
    print(f"  Mode: {mode}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print()

    all_results = []
    t_total = time.time()

    for name in args.datasets:
        if name not in DATASETS:
            print(f"  Unknown dataset: {name}. Available: {', '.join(DATASETS.keys())}")
            continue
        results = run_dataset(name, args.workers, args.thickness, args.classify_brain)
        all_results.extend(results)
        summarize(results, name)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_results)} scans in {total_time:.1f}s")
    print(f"{'='*60}")

    # Save results
    out = [asdict(r) for r in all_results]
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
