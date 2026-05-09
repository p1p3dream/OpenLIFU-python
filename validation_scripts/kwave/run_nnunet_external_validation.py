"""Run trained nnU-Net skull model on external datasets for validation.

Tests the model on data it was never trained on:
1. IXI (581 scans, 3 hospitals, robustness check, no ground truth)
2. SHARM (196 IXI scans with ForkNet-generated skull labels, cross-model comparison)

Usage:
    # On stonkbot after training completes:
    export nnUNet_raw=~/nnUNet_raw
    export nnUNet_preprocessed=~/nnUNet_preprocessed
    export nnUNet_results=~/nnUNet_results

    # IXI robustness check (no ground truth, just check output is plausible)
    python3 -u run_nnunet_external_validation.py ixi --input-dir ~/Data/openlifu-validation/datasets/ixi-t1/ --output-dir ~/Data/openlifu-validation/results/nnunet_ixi/

    # SHARM comparison (has skull labels for Dice computation)
    python3 -u run_nnunet_external_validation.py sharm --input-dir ~/Data/openlifu-validation/datasets/sharm/ --output-dir ~/Data/openlifu-validation/results/nnunet_sharm/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    i = int(np.sum(pred & gt))
    d = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * i / d) if d > 0 else 1.0


def prepare_nnunet_input(nifti_path: str, out_dir: Path, case_id: str) -> None:
    """Copy a NIfTI to nnU-Net input format (case_id_0000.nii.gz)."""
    import shutil
    dst = out_dir / f"{case_id}_0000.nii.gz"
    if str(nifti_path).endswith(".nii.gz"):
        shutil.copy2(nifti_path, str(dst))
    else:
        img = nib.load(nifti_path)
        nib.save(img, str(dst))


def run_nnunet_predict(input_dir: Path, output_dir: Path, fold: int = 0) -> None:
    """Run nnU-Net prediction."""
    import subprocess
    env = os.environ.copy()
    env["nnUNet_raw"] = os.path.expanduser("~/nnUNet_raw")
    env["nnUNet_preprocessed"] = os.path.expanduser("~/nnUNet_preprocessed")
    env["nnUNet_results"] = os.path.expanduser("~/nnUNet_results")

    cmd = [
        "nnUNetv2_predict",
        "-d", "1",
        "-c", "3d_fullres",
        "-f", str(fold),
        "-i", str(input_dir),
        "-o", str(output_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def validate_ixi(input_dir: str, output_dir: str) -> None:
    """Run nnU-Net on IXI scans and check output plausibility."""
    ixi_dir = Path(input_dir)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Prepare input in nnU-Net format
    nnunet_input = out_base / "input"
    nnunet_output = out_base / "predictions"
    nnunet_input.mkdir(exist_ok=True)
    nnunet_output.mkdir(exist_ok=True)

    scans = sorted(ixi_dir.glob("*.nii.gz"))
    print(f"Found {len(scans)} IXI scans")

    # Process in batches to avoid filling disk
    batch_size = 50
    results = []

    for batch_start in range(0, len(scans), batch_size):
        batch = scans[batch_start:batch_start + batch_size]
        print(f"\nBatch {batch_start // batch_size + 1}: scans {batch_start + 1}-{batch_start + len(batch)}")

        # Clear input dir
        for f in nnunet_input.glob("*"):
            f.unlink()

        # Prepare batch
        case_ids = []
        for scan in batch:
            case_id = scan.stem.replace(".nii", "")
            prepare_nnunet_input(str(scan), nnunet_input, case_id)
            case_ids.append(case_id)

        # Run prediction
        t0 = time.time()
        run_nnunet_predict(nnunet_input, nnunet_output)
        elapsed = time.time() - t0
        print(f"  Prediction: {elapsed:.1f}s ({elapsed / len(batch):.1f}s/scan)")

        # Analyze predictions (plausibility check)
        for case_id in case_ids:
            pred_path = nnunet_output / f"{case_id}.nii.gz"
            if not pred_path.exists():
                results.append({"subject": case_id, "error": "no prediction file"})
                continue

            pred = nib.load(str(pred_path)).get_fdata().astype(int)
            total_voxels = pred.size
            skull_voxels = int(np.sum(pred == 1))
            skull_pct = skull_voxels / total_voxels * 100

            results.append({
                "subject": case_id,
                "shape": list(pred.shape),
                "skull_voxels": skull_voxels,
                "skull_pct": round(skull_pct, 2),
                "error": None,
            })

        # Progress
        done = min(batch_start + len(batch), len(scans))
        valid = [r for r in results if not r.get("error")]
        if valid:
            pcts = [r["skull_pct"] for r in valid]
            print(f"  [{done}/{len(scans)}] skull%: mean={np.mean(pcts):.1f}, std={np.std(pcts):.1f}, range=[{np.min(pcts):.1f}, {np.max(pcts):.1f}]")

    # Summary
    valid = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    pcts = [r["skull_pct"] for r in valid]

    print(f"\n{'=' * 60}")
    print(f"IXI External Validation: {len(valid)} scans, {len(errors)} errors")
    print(f"Skull %: mean={np.mean(pcts):.2f}, std={np.std(pcts):.2f}, min={np.min(pcts):.2f}, max={np.max(pcts):.2f}")
    print(f"Expected range for plausible skull: 3-15% of head volume")
    anomalous = [r for r in valid if r["skull_pct"] < 1 or r["skull_pct"] > 20]
    print(f"Anomalous scans (skull% < 1 or > 20): {len(anomalous)}")
    for r in anomalous:
        print(f"  {r['subject']}: {r['skull_pct']}%")

    # Save
    out_path = out_base / "ixi_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


def validate_sharm(input_dir: str, output_dir: str) -> None:
    """Run nnU-Net on SHARM subjects and compare to ForkNet skull labels."""
    sharm_dir = Path(input_dir)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # SHARM structure: need to discover T1 and label files
    # This will depend on how SHARM is organized after download
    # Expected: each subject has a T1 and a segmentation with cortical (label X) + cancellous (label Y) bone

    print("SHARM validation")
    print(f"Looking in: {sharm_dir}")
    print("Directory contents:")
    for item in sorted(sharm_dir.iterdir())[:20]:
        print(f"  {item.name}")

    # TODO: Implement once SHARM is downloaded and structure is known
    # The key comparison: our prediction (binary skull) vs SHARM labels (cortical + cancellous bone combined)
    print("\nSHARM directory structure needs to be mapped after manual download.")
    print("Download from: https://figshare.com/s/a4d9ba6f18a6b7f7ba2c")
    print("Then update this script with the correct file paths and label mappings.")


def main():
    parser = argparse.ArgumentParser(description="External validation of nnU-Net skull model")
    parser.add_argument("dataset", choices=["ixi", "sharm"], help="Dataset to validate on")
    parser.add_argument("--input-dir", required=True, help="Path to input dataset")
    parser.add_argument("--output-dir", required=True, help="Path for output predictions and results")
    parser.add_argument("--fold", type=int, default=0, help="Which fold's model to use")
    args = parser.parse_args()

    if args.dataset == "ixi":
        validate_ixi(args.input_dir, args.output_dir)
    elif args.dataset == "sharm":
        validate_sharm(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
