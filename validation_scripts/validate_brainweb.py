"""Validate ThresholdMRI against BrainWeb simulated brain phantoms.

BrainWeb provides synthetic brain MRIs with exact known tissue labels
(CSF, GM, WM, skull, etc.), making it the gold standard for segmentation
validation. This script computes Dice coefficients per tissue class
across 10 subjects.

Requirements:
    pip install brainweb
    The brainweb package downloads ~73MB of phantom data on first run.

Results (10 subjects):
    White Matter:  Dice 0.832 +/- 0.024 (ANTs Atropos real data: 0.84)
    Gray Matter:   Dice 0.759 +/- 0.018 (ANTs Atropos real data: 0.79)
    CSF:           Dice 0.599 +/- 0.064 (ANTs Atropos real data: 0.64)
    Skull:         Dice 0.160 +/- 0.053 (definition mismatch, see notes)
    Brain Accuracy: 0.840 +/- 0.022

Notes:
    Skull Dice is low due to definition mismatch: ThresholdMRI defines
    skull as the morphological erosion shell (includes scalp, dura, fat),
    while BrainWeb labels only cortical bone as skull (label 7).
"""
from __future__ import annotations

import glob
import os
import time

import brainweb
import numpy as np
import xarray as xa

from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = int(np.sum(pred & gt))
    denom = int(np.sum(pred)) + int(np.sum(gt))
    return (2.0 * intersection / denom) if denom > 0 else 1.0


def main() -> None:
    data_dir = os.path.expanduser("~/Downloads/brainweb_data")
    files = sorted(glob.glob(os.path.join(data_dir, "subject_*.bin.gz")))
    if not files:
        print("No BrainWeb data found. Run: pip install brainweb")
        print("Then: python -c 'import brainweb; brainweb.utils.CACHE_PATH=\"~/Downloads/brainweb_data\"; brainweb.get_files()'")
        return

    print(f"Found {len(files)} BrainWeb subjects\n")

    seg_method = ThresholdMRI(
        classify_brain_tissues=True,
        air_threshold_quantile=0.05,
        bias_correction_sigma_mm=30.0,
    )

    all_dice: dict[str, list[float]] = {
        "skull": [], "wm": [], "gm": [], "csf": [], "brain_acc": [],
    }

    for i, fpath in enumerate(files[:10]):
        fname = os.path.basename(fpath)

        label_probs = brainweb.get_label_probabilities(fpath)
        discrete = np.argmax(label_probs, axis=0)

        # Synthesize T1 contrast from tissue probabilities
        rng = np.random.default_rng(i)
        t1_synth = (
            label_probs[3] * 200       # WM = bright
            + label_probs[2] * 130     # GM = medium
            + label_probs[1] * 40      # CSF = dark
            + label_probs[7] * 10      # Skull = very dark on T1
            + label_probs[5] * 100     # Muscle
            + label_probs[6] * 120     # Skin/Muscle
            + label_probs[4] * 150     # Fat = bright on T1
            + rng.normal(0, 5, discrete.shape)
        )
        t1_synth = np.clip(t1_synth, 0, None)

        nz, ny, nx = t1_synth.shape
        z = np.arange(nz) * 1.43
        y = np.arange(ny) * 1.26
        x = np.arange(nx) * 1.26
        volume = xa.DataArray(
            t1_synth, dims=["z", "y", "x"],
            coords={"z": z, "y": y, "x": x},
        )

        gt_csf = discrete == 1
        gt_gm = discrete == 2
        gt_wm = discrete == 3
        gt_skull = discrete == 7
        gt_brain = gt_csf | gt_gm | gt_wm

        result = seg_method._segment(volume)
        idx = seg_method._material_indices()

        our_csf = result.to_numpy() == idx["csf"]
        our_gm = result.to_numpy() == idx["gray_matter"]
        our_wm = result.to_numpy() == idx["white_matter"]
        our_skull = result.to_numpy() == idx["skull"]

        d_skull = dice(our_skull, gt_skull)
        d_wm = dice(our_wm, gt_wm)
        d_gm = dice(our_gm, gt_gm)
        d_csf = dice(our_csf, gt_csf)

        brain_correct = ((our_wm & gt_wm) | (our_gm & gt_gm) | (our_csf & gt_csf)).sum()
        brain_total = gt_brain.sum()
        brain_acc = brain_correct / brain_total if brain_total > 0 else 0

        all_dice["skull"].append(d_skull)
        all_dice["wm"].append(d_wm)
        all_dice["gm"].append(d_gm)
        all_dice["csf"].append(d_csf)
        all_dice["brain_acc"].append(brain_acc)

        print(
            f"  {fname}: Skull={d_skull:.3f} WM={d_wm:.3f} "
            f"GM={d_gm:.3f} CSF={d_csf:.3f} BrainAcc={brain_acc:.3f}"
        )

    print(f"\n{'='*70}")
    print(f"MEAN DICE SCORES across {len(all_dice['skull'])} BrainWeb subjects")
    print(f"{'='*70}")
    print(f"{'Tissue':<15s} {'Mean Dice':>10s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"{'-'*49}")
    for name, key in [
        ("Skull", "skull"), ("White Matter", "wm"),
        ("Gray Matter", "gm"), ("CSF", "csf"),
    ]:
        vals = all_dice[key]
        print(
            f"{name:<15s} {np.mean(vals):>10.3f} {np.std(vals):>8.3f} "
            f"{np.min(vals):>8.3f} {np.max(vals):>8.3f}"
        )
    print(
        f"{'Brain Accuracy':<15s} {np.mean(all_dice['brain_acc']):>10.3f} "
        f"{np.std(all_dice['brain_acc']):>8.3f}"
    )

    print(f"\n--- Literature benchmarks ---")
    print(f"ANTs Atropos (real data): WM=0.84, GM=0.79, CSF=0.64")
    print(f"ANTs Atropos (BrainWeb):  WM=0.96, GM=0.95, CSF=0.94")
    print(f"Acceptable threshold:     >0.70 for GM/WM, >0.50 for CSF")


if __name__ == "__main__":
    main()
