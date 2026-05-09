"""Compute Dice scores between nnU-Net predictions and SHARM ground truth skull masks."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def main():
    pred_dir = Path("/home/brandon/Data/openlifu-validation/results/nnunet_sharm_quick/predictions")
    gt_dir = Path("/home/brandon/Data/openlifu-validation/results/nnunet_sharm_quick/ground_truth")
    out_dir = Path("/home/brandon/Data/openlifu-validation/results/nnunet_sharm_quick")

    preds = sorted(pred_dir.glob("*.nii.gz"))
    print(f"Found {len(preds)} predictions")
    print(f"{'Subject':<15} {'Dice':>8} {'Pred voxels':>12} {'GT voxels':>12} {'Pred %':>8} {'GT %':>8}")
    print("-" * 70)

    results = []
    for pred_path in preds:
        sid = pred_path.stem.replace(".nii", "")
        gt_path = gt_dir / f"{sid}.nii.gz"
        if not gt_path.exists():
            print(f"  {sid}: no ground truth found")
            continue

        pred = nib.load(str(pred_path)).get_fdata().astype(bool)
        gt = nib.load(str(gt_path)).get_fdata().astype(bool)
        d = dice(pred, gt)
        pred_pct = pred.sum() / pred.size * 100
        gt_pct = gt.sum() / gt.size * 100

        results.append({
            "subject": sid,
            "dice": round(d, 4),
            "pred_voxels": int(pred.sum()),
            "gt_voxels": int(gt.sum()),
            "pred_pct": round(pred_pct, 2),
            "gt_pct": round(gt_pct, 2),
        })
        print(f"{sid:<15} {d:>8.4f} {int(pred.sum()):>12,} {int(gt.sum()):>12,} {pred_pct:>7.2f}% {gt_pct:>7.2f}%")

    dices = [r["dice"] for r in results]
    print("-" * 70)
    print(f"N = {len(results)}")
    print(f"Mean Dice:   {np.mean(dices):.4f}")
    print(f"Median Dice: {np.median(dices):.4f}")
    print(f"Std:         {np.std(dices):.4f}")
    print(f"Min:         {np.min(dices):.4f}")
    print(f"Max:         {np.max(dices):.4f}")

    out_path = out_dir / "sharm_dice_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": {"n": len(results), "mean": round(np.mean(dices), 4), "median": round(np.median(dices), 4), "std": round(np.std(dices), 4), "min": round(np.min(dices), 4), "max": round(np.max(dices), 4)}, "per_subject": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
