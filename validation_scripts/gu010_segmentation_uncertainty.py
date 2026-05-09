#!/usr/bin/env python3
"""GU010 segmentation-uncertainty analysis.

Hypothesis: the 15.6 dB unexplained residual in the GU010 GLADYS result
is driven by nnU-Net segmentation uncertainty rather than physics.

This is a LOCAL / uncommitted diagnostic.

Environment on stonkbot (confirmed 2026-04-16):
  * Dataset002_FullHeadSeg (7-class fullhead, used in GLADYS pipeline):
      only fold_0 exists.
  * Dataset001_SkullSeg (binary skull / bg): folds 0, 1, 2 exist.

Therefore we use Dataset001 folds 0/1/2 to get THREE independent binary
skull masks (a true multi-fold ensemble), compare them, and build
ensembles (majority / union / intersection). For the full GLADYS sim we
construct a HYBRID label NIfTI:
  * Start from Dataset002 fold_0 labels (all 7 classes).
  * Replace the skull label (5) with the Dataset001-majority-ensemble
    skull mask (voxels that are skull in >=2 of 3 Dataset001 folds).
  * Everything else (water, air, csf, gm, wm, soft_tissue) is preserved.
This keeps the sim's material assignment valid while changing only the
skull voxels to the ensemble estimate.

This script has two modes:
  --stage predict    Run the Dataset001 nnUNetv2_predict for folds 1,2
                     (fold_0 will be run too unless the output already exists).
  --stage analyze    Load existing predictions, compute Dice / volume /
                     path-length / build ensembles / save hybrid label
                     NIfTI for GLADYS. (default)

Outputs (under ~/Data/openlifu-validation/results/):
  GU010_skullseg_fold0.nii.gz           (Dataset001 fold 0, binary)
  GU010_skullseg_fold1.nii.gz           (Dataset001 fold 1, binary)
  GU010_skullseg_fold2.nii.gz           (Dataset001 fold 2, binary)
  GU010_skullseg_majority.nii.gz        (>=2 of 3, binary)
  GU010_skullseg_union.nii.gz           (>=1 of 3, binary)
  GU010_skullseg_intersection.nii.gz    (3 of 3, binary)
  GU010_nnunet_labels_ensemble_majority.nii.gz
      (Dataset002-fold0 labels with skull label replaced by majority skull)
  GU010_segmentation_uncertainty_report.txt
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xa
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, binary_erosion, map_coordinates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gu010_segunc")

# ---------------------------------------------------------------------------
# Paths (stonkbot layout).
# ---------------------------------------------------------------------------
HOME = Path.home()
RESULTS_DIR = HOME / "Data/openlifu-validation/results"
MRI_PATH = HOME / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU010_deface.nii"
)
DATASET002_LABELS = RESULTS_DIR / "GU010_nnunet_labels.nii.gz"  # Dataset002 fold_0

NNUNET_ENV = {
    "nnUNet_raw": str(HOME / "nnUNet_raw"),
    "nnUNet_preprocessed": str(HOME / "nnUNet_preprocessed"),
    "nnUNet_results": str(HOME / "nnUNet_results"),
}
PREDICT_INPUT_DIR = HOME / "nnUNet_predict_inputs/GU010_skullseg"
PREDICT_OUTPUT_DIR_TMPL = HOME / "nnUNet_predict_outputs/GU010_skullseg_fold{fold}"
PYTHON_BIN = HOME / "openlifu-env/bin/python"

# Array geometry (must match run_gladys_nnunet_subject.py / analyze_skull_path.py)
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_MHZ = 0.5
GRID_SPACING_MM = 0.5
GRID_MARGIN_MM = 25.0
RAY_STEP_MM = 0.25

SKULL_ATTEN_COEFF = 8.0  # dB/cm/MHz from material.py
POWER_LAW_Y = 0.9


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def run_predict(fold: int, device: str = "cuda") -> Path:
    """Run nnUNetv2_predict on Dataset001 for a given fold, return label NIfTI path."""
    out_dir = Path(str(PREDICT_OUTPUT_DIR_TMPL).format(fold=fold))
    out_nifti = out_dir / "GU010.nii.gz"
    final_path = RESULTS_DIR / f"GU010_skullseg_fold{fold}.nii.gz"

    if final_path.exists():
        logger.info("fold %d: %s already exists, skipping predict.", fold, final_path)
        return final_path

    PREDICT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_nifti = PREDICT_INPUT_DIR / "GU010_0000.nii.gz"
    if not input_nifti.exists():
        logger.info("Copying MRI to nnUNet input dir (gzipping).")
        img = nib.load(str(MRI_PATH))
        nib.save(img, str(input_nifti))

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(NNUNET_ENV)
    cmd = [
        str(HOME / "openlifu-env/bin/nnUNetv2_predict"),
        "-i", str(PREDICT_INPUT_DIR),
        "-o", str(out_dir),
        "-d", "001",
        "-c", "3d_fullres",
        "-f", str(fold),
        "-device", device,
        "--disable_tta",
    ]
    logger.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    logger.info("fold %d predict done in %.1fs", fold, time.time() - t0)

    if not out_nifti.exists():
        raise RuntimeError(f"Expected nnUNet output not found: {out_nifti}")

    # Copy to canonical results location
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(out_nifti.read_bytes())
    logger.info("Copied to %s", final_path)
    return final_path


def load_binary_skull(path: Path, reference_img: nib.Nifti1Image | None = None) -> np.ndarray:
    """Load a NIfTI and return a binary mask (Dataset001: label==1 -> True)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    mask = (data == 1)
    if reference_img is not None:
        # Resample if affine/shape differs. Dataset001 and Dataset002 both
        # predict on the same input MRI so shapes / affines should match.
        if (not np.allclose(img.affine, reference_img.affine) or
                data.shape != reference_img.shape):
            logger.warning("Resampling %s to reference grid (shape/affine differ).", path.name)
            mask = _resample_binary_to_reference(img, mask, reference_img)
    return mask.astype(bool)


def _resample_binary_to_reference(
    src_img: nib.Nifti1Image,
    src_mask: np.ndarray,
    ref_img: nib.Nifti1Image,
) -> np.ndarray:
    # Use nearest-neighbor on binary mask via map_coordinates.
    # Build the mapping from ref voxel -> src voxel.
    src_inv = np.linalg.inv(src_img.affine)
    # Ref voxel indices -> ref world -> src voxel indices.
    ref_shape = ref_img.shape
    ii, jj, kk = np.meshgrid(
        np.arange(ref_shape[0]), np.arange(ref_shape[1]), np.arange(ref_shape[2]),
        indexing="ij",
    )
    ones = np.ones_like(ii)
    ref_vox = np.stack([ii, jj, kk, ones], axis=-1).astype(np.float32)
    # world = ref_affine @ ref_vox; src_vox = src_inv @ world
    world = ref_vox @ ref_img.affine.T
    src_vox_idx = world @ src_inv.T
    coords = np.stack([src_vox_idx[..., 0], src_vox_idx[..., 1], src_vox_idx[..., 2]], axis=0)
    resampled = map_coordinates(
        src_mask.astype(np.float32), coords, order=0, mode="constant", cval=0.0,
    )
    return resampled > 0.5


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom > 0 else float("nan")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    u = np.logical_or(a, b).sum()
    return (np.logical_and(a, b).sum() / u) if u > 0 else float("nan")


def save_binary_nifti(mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(img, str(out_path))
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Array + target geometry helpers (match run_gladys_nnunet_subject.py).
# ---------------------------------------------------------------------------
def hemispherical_positions(target_mm: np.ndarray, approach_axis: int) -> np.ndarray:
    half_aperture = APERTURE_MM / 2.0
    theta_max = np.arcsin(half_aperture / RADIUS_MM)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    local = np.zeros((N_ELEMENTS, 3))
    for i in range(N_ELEMENTS):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / N_ELEMENTS
        theta = np.arccos(cos_theta)
        phi = golden * i
        local[i, 0] = RADIUS_MM * np.sin(theta) * np.cos(phi)
        local[i, 1] = RADIUS_MM * np.sin(theta) * np.sin(phi)
        local[i, 2] = RADIUS_MM * np.cos(theta)
    R = np.eye(3)
    if approach_axis == 0:
        a = np.pi / 2
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    elif approach_axis == 1:
        a = -np.pi / 2
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    world = local @ R.T + target_mm
    return world


def find_target_and_axis(
    fullhead_labels: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, np.ndarray]]:
    """Find target (brain center) and approach axis from Dataset002 labels."""
    # LABEL_MAP_FULLHEAD: 2=csf, 3=gm, 4=wm, 5=skull
    brain_mask = np.isin(fullhead_labels, [2, 3, 4])
    if brain_mask.sum() == 0:
        brain_mask = (fullhead_labels == 6)  # fallback to soft_tissue
    brain_idx = np.argwhere(brain_mask)
    # Build coord arrays (assumes diagonal affine scaled properly).
    dims = ("x", "y", "z")
    coord_arrays = {}
    for ax, d in enumerate(dims):
        origin = float(affine[ax, 3])
        spacing = float(affine[ax, ax])
        coord_arrays[d] = origin + np.arange(fullhead_labels.shape[ax]) * spacing
    target_mm = np.array([
        float(np.mean(coord_arrays[dims[ax]][brain_idx[:, ax]]))
        for ax in range(3)
    ])
    skull_mask = (fullhead_labels == 5)
    skull_idx = np.argwhere(skull_mask)
    skull_mm = np.array([coord_arrays[dims[ax]][skull_idx[:, ax]] for ax in range(3)])
    dists = np.array([float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)])
    approach_axis = int(np.argmax(dists))
    return target_mm, approach_axis, coord_arrays


def resample_to_sim_grid(
    mask: np.ndarray,
    src_coord_arrays: dict[str, np.ndarray],
    sim_coords: dict[str, np.ndarray],
) -> np.ndarray:
    """Resample a binary mask from native MRI grid to sim grid (nearest)."""
    dims = ("x", "y", "z")
    src_origin = np.array([src_coord_arrays[d][0] for d in dims])
    src_spacing = np.array([src_coord_arrays[d][1] - src_coord_arrays[d][0] for d in dims])
    mg = np.meshgrid(*[sim_coords[d] for d in dims], indexing="ij")
    pts = np.stack(mg, axis=-1)  # (Nx,Ny,Nz,3) world mm
    frac = (pts - src_origin) / src_spacing  # (Nx,Ny,Nz,3)
    coords = np.stack(
        [frac[..., 0], frac[..., 1], frac[..., 2]], axis=0,
    )
    resampled = map_coordinates(
        mask.astype(np.float32), coords, order=0, mode="constant", cval=0.0,
    )
    return resampled > 0.5


# ---------------------------------------------------------------------------
# Per-element path ray-cast on a binary skull mask.
# ---------------------------------------------------------------------------
def per_element_path_lengths(
    skull_sim: np.ndarray,
    sim_coord_arrays: list[np.ndarray],
    target_mm: np.ndarray,
    element_positions_mm: np.ndarray,
) -> np.ndarray:
    """For each of the 64 elements, cast a ray from element -> target and
    return the number of mm intersecting skull=True voxels.

    Uses 0.25 mm steps and nearest-voxel sampling (same as analyze_skull_path).
    """
    origins = np.array([c[0] for c in sim_coord_arrays])
    spacings = np.array([c[1] - c[0] for c in sim_coord_arrays])
    paths = np.zeros(N_ELEMENTS)
    for i in range(N_ELEMENTS):
        ray_vec = target_mm - element_positions_mm[i]
        total = float(np.linalg.norm(ray_vec))
        if total < 1e-6:
            continue
        ray_dir = ray_vec / total
        n_steps = int(np.ceil(total / RAY_STEP_MM)) + 1
        ts = np.linspace(0.0, total, n_steps)
        pts = element_positions_mm[i][None, :] + ts[:, None] * ray_dir[None, :]
        frac = ((pts - origins[None, :]) / spacings[None, :]).T
        # Clip to grid
        for ax in range(3):
            frac[ax] = np.clip(frac[ax], 0, skull_sim.shape[ax] - 1)
        sampled = map_coordinates(
            skull_sim.astype(np.float32), frac, order=0,
            mode="constant", cval=0.0,
        )
        paths[i] = float((sampled > 0.5).sum()) * RAY_STEP_MM
    return paths


def path_stats(paths: np.ndarray) -> dict:
    return {
        "n": int(len(paths)),
        "mean": float(np.mean(paths)),
        "std": float(np.std(paths)),
        "min": float(np.min(paths)),
        "max": float(np.max(paths)),
        "p50": float(np.percentile(paths, 50)),
        "p90": float(np.percentile(paths, 90)),
        "p99": float(np.percentile(paths, 99)),
        "n_zero": int((paths == 0.0).sum()),
        "n_zero_pct": 100.0 * int((paths == 0.0).sum()) / len(paths),
    }


def predicted_attenuation_db(mean_path_mm: float) -> float:
    # Double-pass not included; this is just a scale reference.
    alpha = SKULL_ATTEN_COEFF * (FREQ_MHZ ** POWER_LAW_Y)
    return alpha * (mean_path_mm / 10.0)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full",
                    choices=["predict", "analyze", "full"])
    ap.add_argument("--skip-fold2", action="store_true",
                    help="Skip fold 2 (budget mode).")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    t_total = time.time()
    report_lines: list[str] = []

    def p(line: str = ""):
        print(line, flush=True)
        report_lines.append(line)

    p("=" * 78)
    p("GU010 SEGMENTATION UNCERTAINTY ANALYSIS")
    p(f"  stage = {args.stage}, skip_fold2 = {args.skip_fold2}")
    p(f"  MRI   = {MRI_PATH}")
    p(f"  D002 fold_0 labels (reference) = {DATASET002_LABELS}")
    p("=" * 78)

    folds = [0, 1] if args.skip_fold2 else [0, 1, 2]

    if args.stage in ("predict", "full"):
        p("\n[1] Running Dataset001 (binary skull) predictions ...")
        for f in folds:
            t0 = time.time()
            run_predict(f, device=args.device)
            p(f"    fold {f}: {time.time() - t0:.1f}s")

    if args.stage == "predict":
        p("\n(stage=predict done; rerun with --stage analyze)")
        return 0

    # ------------------------------------------------------------------
    # Load Dataset002 fold_0 labels (reference grid, all 7 classes).
    # ------------------------------------------------------------------
    if not DATASET002_LABELS.exists():
        p(f"ERROR: Dataset002 fold_0 labels missing: {DATASET002_LABELS}")
        return 2
    ref_img = nib.load(str(DATASET002_LABELS))
    ref_labels = np.asarray(ref_img.dataobj).astype(np.int16)
    ref_affine = ref_img.affine
    p(f"\n[2] Reference Dataset002-fold0 shape={ref_labels.shape}, "
      f"spacing=({ref_affine[0,0]:.2f},{ref_affine[1,1]:.2f},{ref_affine[2,2]:.2f}) mm")
    # Label histogram
    for lbl, name in [(0, "water"), (1, "air"), (2, "csf"), (3, "gm"),
                      (4, "wm"), (5, "skull"), (6, "soft_tissue")]:
        n = int((ref_labels == lbl).sum())
        p(f"    label {lbl} ({name}): {n:>9,d} vox ({100.0*n/ref_labels.size:.2f}%)")

    # Dataset002-fold0 skull as a separate reference mask
    d002_skull = (ref_labels == 5)

    # ------------------------------------------------------------------
    # Load Dataset001 folds 0/1/(2) binary skull masks.
    # ------------------------------------------------------------------
    p("\n[3] Loading Dataset001 (binary skull) per-fold masks ...")
    fold_masks: dict[int, np.ndarray] = {}
    for f in folds:
        path = RESULTS_DIR / f"GU010_skullseg_fold{f}.nii.gz"
        if not path.exists():
            p(f"ERROR: missing {path}. Re-run with --stage predict first.")
            return 3
        fold_masks[f] = load_binary_skull(path, reference_img=ref_img)
        n = int(fold_masks[f].sum())
        p(f"    fold {f}: {n:>9,d} skull voxels ({100.0*n/ref_labels.size:.3f}%)")

    # Compare to Dataset002 fold_0 skull
    n_d002 = int(d002_skull.sum())
    p(f"    Dataset002-fold0 skull: {n_d002:>9,d} vox ({100.0*n_d002/ref_labels.size:.3f}%)")

    # ------------------------------------------------------------------
    # Pairwise Dice, per-fold volume, disagreement.
    # ------------------------------------------------------------------
    p("\n[4] Pairwise Dice (Dataset001 folds + Dataset002 fold_0):")
    all_named = {f"D001_f{f}": fold_masks[f] for f in folds}
    all_named["D002_f0"] = d002_skull
    names = list(all_named.keys())
    p(f"    {'':>10s} " + " ".join(f"{n:>10s}" for n in names))
    for ni in names:
        row = [f"{ni:>10s}"]
        for nj in names:
            row.append(f"{dice(all_named[ni], all_named[nj]):>10.4f}")
        p(" ".join(row))

    # Stack folds for ensemble
    stack = np.stack([fold_masks[f] for f in folds], axis=0)  # (nF, ...)
    vote = stack.sum(axis=0)
    if len(folds) == 3:
        majority = (vote >= 2)
        union = (vote >= 1)
        intersection = (vote >= 3)
    else:
        # 2 folds: majority = both; union = either.
        majority = (vote >= 2)
        union = (vote >= 1)
        intersection = (vote >= 2)

    ensembles = {
        "majority": majority,
        "union": union,
        "intersection": intersection,
    }
    p("\n[5] Ensemble skull volume fractions:")
    for k, m in ensembles.items():
        n = int(m.sum())
        p(f"    {k:>13s}: {n:>9,d} vox ({100.0*n/ref_labels.size:.3f}%)")

    # Disagreement map stats
    disagreement_vox = int(((vote > 0) & (vote < len(folds))).sum())
    p(f"\n[6] Disagreement (any fold disagrees): {disagreement_vox:,d} vox "
      f"({100.0*disagreement_vox/ref_labels.size:.3f}% of volume, "
      f"{100.0*disagreement_vox/max(union.sum(),1):.1f}% of union-skull).")

    # ------------------------------------------------------------------
    # Morphological perturbations on Dataset001 fold_0 skull.
    # ------------------------------------------------------------------
    p("\n[7] Morphological perturbations on Dataset001 fold_0 skull ...")
    # Voxel spacing from ref affine
    sp = np.array([abs(ref_affine[i, i]) for i in range(3)])
    p(f"    native voxel spacing = {sp} mm")
    # Build structuring element that approximates 1mm expansion
    # (3x3x3 expands by 1 voxel; at 0.8-1mm MRI spacing, 1 iter ~ 1mm.)
    iters_1mm = max(1, int(round(1.0 / float(np.mean(sp)))))
    p(f"    1mm dilation/erosion via {iters_1mm} iter(s) of 3x3x3 structure.")
    dil = binary_dilation(fold_masks[0], iterations=iters_1mm)
    ero = binary_erosion(fold_masks[0], iterations=iters_1mm)
    p(f"    fold_0:             {fold_masks[0].sum():>9,d} vox")
    p(f"    fold_0 +1mm dilate: {dil.sum():>9,d} vox  (+{(dil.sum()-fold_masks[0].sum()):,d})")
    p(f"    fold_0 -1mm erode:  {ero.sum():>9,d} vox  (-{(fold_masks[0].sum()-ero.sum()):,d})")

    # ------------------------------------------------------------------
    # Save binary skull ensembles to NIfTI.
    # ------------------------------------------------------------------
    p("\n[8] Saving ensemble NIfTI files ...")
    save_binary_nifti(majority, ref_img, RESULTS_DIR / "GU010_skullseg_majority.nii.gz")
    save_binary_nifti(union, ref_img, RESULTS_DIR / "GU010_skullseg_union.nii.gz")
    save_binary_nifti(intersection, ref_img, RESULTS_DIR / "GU010_skullseg_intersection.nii.gz")

    # Save hybrid full-head labels: take Dataset002-fold0 and replace skull-5 with majority
    hybrid = ref_labels.copy()
    # Clear existing skull voxels to water (0)
    hybrid[hybrid == 5] = 0
    # Set ensemble-majority skull voxels to label 5
    hybrid[majority] = 5
    hybrid_path = RESULTS_DIR / "GU010_nnunet_labels_ensemble_majority.nii.gz"
    nib.save(nib.Nifti1Image(hybrid.astype(np.int16), ref_affine, ref_img.header), str(hybrid_path))
    p(f"    Saved hybrid full-head labels (D002 + D001-majority skull): {hybrid_path}")
    n_hybrid_skull = int((hybrid == 5).sum())
    p(f"    hybrid skull vox: {n_hybrid_skull:,d} ({100.0*n_hybrid_skull/hybrid.size:.3f}%)")

    # ------------------------------------------------------------------
    # Build sim grid, target, element positions (for per-element path).
    # ------------------------------------------------------------------
    p("\n[9] Building sim grid and element geometry ...")
    target_mm, approach_axis, src_coord_arrays = find_target_and_axis(ref_labels, ref_affine)
    p(f"    target (brain center) = ({target_mm[0]:.1f},{target_mm[1]:.1f},{target_mm[2]:.1f}) mm")
    p(f"    approach axis = {('x','y','z')[approach_axis]}")
    positions = hemispherical_positions(target_mm, approach_axis)
    p(f"    64 element positions built; range = "
      f"[{positions.min():.1f}, {positions.max():.1f}] mm")

    # sim grid at 0.5mm iso encompassing target+elements
    all_pts = np.vstack([positions, target_mm[None, :]])
    grid_min = np.floor((all_pts.min(axis=0) - GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    grid_max = np.ceil((all_pts.max(axis=0) + GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    sim_coords = {}
    for ax, d in enumerate(("x", "y", "z")):
        n = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        sim_coords[d] = np.linspace(grid_min[ax], grid_max[ax], n)
    p(f"    sim grid: {[len(sim_coords[d]) for d in ('x','y','z')]} @ {GRID_SPACING_MM} mm")

    # Resample each mask variant to sim grid (binary)
    variants: dict[str, np.ndarray] = {}
    for f, m in fold_masks.items():
        variants[f"D001_fold{f}"] = m
    variants["majority"] = majority
    variants["union"] = union
    variants["intersection"] = intersection
    variants["fold0_dilated_1mm"] = dil
    variants["fold0_eroded_1mm"] = ero
    variants["D002_fold0"] = d002_skull

    p("\n[10] Resampling all skull-mask variants to sim grid (nearest) ...")
    sim_coord_arrays = [sim_coords[d] for d in ("x", "y", "z")]
    variants_sim: dict[str, np.ndarray] = {}
    for name, m in variants.items():
        t0 = time.time()
        variants_sim[name] = resample_to_sim_grid(m, src_coord_arrays, sim_coords)
        p(f"    {name:>22s}: native={m.sum():>9,d}, "
          f"sim={int(variants_sim[name].sum()):>9,d}, "
          f"dt={time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 64-ray per-element path-length analysis on each variant.
    # ------------------------------------------------------------------
    p("\n[11] Per-element path-length distributions (64 rays) ...")
    p(f"    {'variant':>22s} {'mean':>6s} {'std':>6s} {'max':>6s} {'P50':>6s} "
      f"{'P90':>6s} {'P99':>6s} {'zero':>5s} {'zero%':>6s} {'pred_dB*':>9s}")
    stats_by_variant: dict[str, dict] = {}
    paths_by_variant: dict[str, np.ndarray] = {}
    for name, m_sim in variants_sim.items():
        paths = per_element_path_lengths(m_sim, sim_coord_arrays, target_mm, positions)
        paths_by_variant[name] = paths
        s = path_stats(paths)
        stats_by_variant[name] = s
        pred_db = predicted_attenuation_db(s["mean"])
        p(f"    {name:>22s} {s['mean']:6.2f} {s['std']:6.2f} {s['max']:6.2f} "
          f"{s['p50']:6.2f} {s['p90']:6.2f} {s['p99']:6.2f} "
          f"{s['n_zero']:>5d} {s['n_zero_pct']:>6.1f} {pred_db:>9.2f}")
    p("    (*pred_dB = SKULL_ATTEN_COEFF * freq^y * mean_path_cm; ignores reflection/double-pass.)")

    # Compare vs Dataset002 fold_0 baseline
    base = stats_by_variant["D002_fold0"]
    p(f"\n[12] Delta vs Dataset002-fold0 baseline (mean path = {base['mean']:.2f} mm):")
    for name, s in stats_by_variant.items():
        if name == "D002_fold0":
            continue
        dmean = s["mean"] - base["mean"]
        dzero = s["n_zero"] - base["n_zero"]
        p(f"    {name:>22s}: dmean={dmean:+6.2f} mm "
          f"({100.0*dmean/max(base['mean'],1e-6):+6.1f}%)   dzero={dzero:+d}")

    # Spread across the three folds only (true segmentation uncertainty)
    fold_means = [stats_by_variant[f"D001_fold{f}"]["mean"] for f in folds]
    fold_spread = max(fold_means) - min(fold_means)
    fold_rel = 100.0 * fold_spread / max(np.mean(fold_means), 1e-6)
    p(f"\n[13] Dataset001 inter-fold spread of mean path length:")
    p(f"    means = {fold_means}")
    p(f"    max - min = {fold_spread:.2f} mm ({fold_rel:.1f}% of mean)")

    # Save report
    report_path = RESULTS_DIR / "GU010_segmentation_uncertainty_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    p(f"\nReport saved: {report_path}")
    p(f"Total elapsed: {time.time() - t_total:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
