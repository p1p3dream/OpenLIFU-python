"""Head-to-head comparison: ThresholdMRI vs ANTs Atropos on identical BrainWeb data.

Both methods run on the same synthetic T1 volumes with the same ground truth
tissue labels. This is the only honest comparison since both use identical
input and reference data.

Requirements:
    pip install antspyx brainweb

Results (5 BrainWeb subjects):
    Tissue         ThresholdMRI   ANTs Atropos   Gap
    WM                    0.838          0.990   -0.152
    GM                    0.764          0.873   -0.109
    CSF                   0.585          0.861   -0.276

Conclusion:
    ANTs Atropos is substantially better at brain tissue classification.
    ThresholdMRI achieves acceptable Dice (>0.70) for WM and GM but is not
    competitive with ANTs. The primary value of ThresholdMRI is skull
    extraction (which ANTs cannot do), not brain tissue classification.
"""
from __future__ import annotations

import glob
import os
import time

import ants
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
    n_subjects = min(5, len(files))

    print(f"Head-to-head: ThresholdMRI vs ANTs Atropos on {n_subjects} BrainWeb subjects\n")

    results: dict[str, list[float]] = {
        "ours_wm": [], "ours_gm": [], "ours_csf": [],
        "ants_wm": [], "ants_gm": [], "ants_csf": [],
        "ours_time": [], "ants_time": [],
    }

    for i, fpath in enumerate(files[:n_subjects]):
        fname = os.path.basename(fpath)
        print(f"Subject {i + 1}: {fname}")

        label_probs = brainweb.get_label_probabilities(fpath)
        discrete = np.argmax(label_probs, axis=0)

        rng = np.random.default_rng(i)
        t1_synth = (
            label_probs[3] * 200 + label_probs[2] * 130 + label_probs[1] * 40
            + label_probs[7] * 10 + label_probs[5] * 100 + label_probs[6] * 120
            + label_probs[4] * 150 + rng.normal(0, 5, discrete.shape)
        )
        t1_synth = np.clip(t1_synth, 0, None).astype(np.float32)

        gt_csf = discrete == 1
        gt_gm = discrete == 2
        gt_wm = discrete == 3

        nz, ny, nx = t1_synth.shape
        spacing = (1.43, 1.26, 1.26)

        # ThresholdMRI
        t0 = time.time()
        volume = xa.DataArray(
            t1_synth, dims=["z", "y", "x"],
            coords={"z": np.arange(nz) * spacing[0], "y": np.arange(ny) * spacing[1], "x": np.arange(nx) * spacing[2]},
        )
        seg = ThresholdMRI(classify_brain_tissues=True, air_threshold_quantile=0.05, bias_correction_sigma_mm=30.0)
        result = seg._segment(volume)
        idx = seg._material_indices()
        ours_time = time.time() - t0

        our_wm = result.to_numpy() == idx["white_matter"]
        our_gm = result.to_numpy() == idx["gray_matter"]
        our_csf = result.to_numpy() == idx["csf"]

        # ANTs Atropos
        t0 = time.time()
        ants_img = ants.from_numpy(t1_synth, spacing=list(spacing))
        mask = ants.get_mask(ants_img)
        seg_ants = ants.atropos(ants_img, x=mask, m="[0.2, 1x1x1]")
        ants_time = time.time() - t0

        ants_labels = seg_ants["segmentation"].numpy().astype(int)
        ants_csf_mask = ants_labels == 1
        ants_gm_mask = ants_labels == 2
        ants_wm_mask = ants_labels == 3

        for key, our, ant, gt in [
            ("wm", our_wm, ants_wm_mask, gt_wm),
            ("gm", our_gm, ants_gm_mask, gt_gm),
            ("csf", our_csf, ants_csf_mask, gt_csf),
        ]:
            results[f"ours_{key}"].append(dice(our, gt))
            results[f"ants_{key}"].append(dice(ant, gt))
        results["ours_time"].append(ours_time)
        results["ants_time"].append(ants_time)

        print(f"  Ours: WM={dice(our_wm, gt_wm):.3f} GM={dice(our_gm, gt_gm):.3f} CSF={dice(our_csf, gt_csf):.3f} ({ours_time:.1f}s)")
        print(f"  ANTs: WM={dice(ants_wm_mask, gt_wm):.3f} GM={dice(ants_gm_mask, gt_gm):.3f} CSF={dice(ants_csf_mask, gt_csf):.3f} ({ants_time:.1f}s)")
        print()

    print(f"{'='*70}")
    print(f"HEAD-TO-HEAD COMPARISON (same data, same ground truth)")
    print(f"{'='*70}")
    print(f"{'Tissue':<12s} {'ThresholdMRI':>14s} {'ANTs Atropos':>14s} {'Difference':>12s}")
    print(f"{'-'*52}")
    for tissue, ours_key, ants_key in [("WM", "ours_wm", "ants_wm"), ("GM", "ours_gm", "ants_gm"), ("CSF", "ours_csf", "ants_csf")]:
        o = np.mean(results[ours_key])
        a = np.mean(results[ants_key])
        print(f"{tissue:<12s} {o:>14.3f} {a:>14.3f} {o - a:>+12.3f}")

    print(f"\nAvg time: ThresholdMRI {np.mean(results['ours_time']):.1f}s, ANTs {np.mean(results['ants_time']):.1f}s")
    print(f"Dependencies: ThresholdMRI=0, ANTs=antspyx (~46MB)")


if __name__ == "__main__":
    main()
