"""Run PseudoCT on all Birnbaum scans sequentially."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xa

sys.path.insert(0, os.path.expanduser("~/code/openwater/OpenLIFU-python/src"))
from openlifu.seg.seg_methods import PseudoCTSegmentation


def dice(a, b):
    i = int(np.sum(a & b))
    d = int(np.sum(a)) + int(np.sum(b))
    return (2.0 * i / d) if d > 0 else 1.0


def main():
    birn = Path(os.path.expanduser("~/Data/openlifu-validation/datasets/birnbaum-fullhead/Data"))
    scans = []
    for group, prefix in [("Anonymized_Subjects", "_deface"), ("Control_Subjects", "")]:
        t1_dir = birn / group / "T1-Weighted MRI"
        seg_dir = birn / group / "Full-Head Segmentation"
        if not t1_dir.exists():
            continue
        for f in sorted(t1_dir.glob("*.nii")):
            sid = f.stem.replace("_deface", "")
            lp = seg_dir / f"{sid}_label{prefix}.nii"
            if lp.exists():
                scans.append((sid, str(f), str(lp)))

    print(f"Running PseudoCT on {len(scans)} Birnbaum scans sequentially...")
    results = []
    t_start = time.time()

    for i, (sid, tp, lp) in enumerate(scans):
        try:
            t0 = time.time()
            t1 = nib.load(tp)
            d = t1.get_fdata().astype(np.float32)
            labels = nib.load(lp).get_fdata().astype(int)
            z = t1.header.get_zooms()[:3]
            nz, ny, nx = d.shape
            v = xa.DataArray(
                d, dims=["z", "y", "x"],
                coords={"z": np.arange(nz) * z[0], "y": np.arange(ny) * z[1], "x": np.arange(nx) * z[2]},
            )

            seg = PseudoCTSegmentation(hu_bone_threshold=300.0, use_gpu=True, preprocess=True)
            result = seg._segment(v)
            idx = seg._material_indices()
            elapsed = time.time() - t0

            gt_bone = labels == 5
            gt_brain = np.isin(labels, [2, 3, 4])
            skull_d = dice(result.to_numpy() == idx["skull"], gt_bone)
            brain_d = dice(result.to_numpy() == idx["tissue"], gt_brain)

            results.append({"subject": sid, "skull_dice": skull_d, "brain_dice": brain_d, "time": elapsed, "error": None})
            print(f"  [{i+1}/{len(scans)}] {sid}: skull={skull_d:.3f} brain={brain_d:.3f} ({elapsed:.0f}s)")

        except Exception as e:
            results.append({"subject": sid, "skull_dice": 0, "brain_dice": 0, "time": 0, "error": str(e)})
            print(f"  [{i+1}/{len(scans)}] {sid}: ERROR: {e}")

    total = time.time() - t_start
    valid = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]
    skulls = [r["skull_dice"] for r in valid]
    brains = [r["brain_dice"] for r in valid]

    print(f"\nDone: {len(valid)} ok, {len(errors)} errors, {total:.0f}s")
    if skulls:
        print(f"Skull Dice: mean={np.mean(skulls):.3f} std={np.std(skulls):.3f} min={np.min(skulls):.3f} max={np.max(skulls):.3f}")
        print(f"Brain Dice: mean={np.mean(brains):.3f} std={np.std(brains):.3f} min={np.min(brains):.3f} max={np.max(brains):.3f}")

    out_path = os.path.expanduser("~/Data/openlifu-validation/results/birnbaum_pseudoct_final.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
