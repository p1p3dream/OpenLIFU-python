"""GU010 investigation: why low bone fraction but stubborn residual attenuation.

Compares GU010 against GU008 on:
  1. Segmentation composition (sanity).
  2. Per-element (64 rays) skull path length distribution through the nnU-Net
     label volume.
  3. Visual overlay of skull mask on orthogonal MRI slices through target, to
     eyeball segmentation quality.
  4. Anomaly detection on GU010's skull mask: connected components, skull
     voxels in brain/CSF, L/R vault symmetry.

Local uncommitted script. Reads the already-saved nnU-Net label NIfTIs
(GU{008,010}_nnunet_labels.nii.gz) and MRI files. No k-wave / GPU used.

Outputs:
  ~/Data/openlifu-validation/results/gu010_seg_check.png       (slice overlay)
  ~/Data/openlifu-validation/results/gu010_path_histograms.png (per-ray histos)
Printed: path-length stats, composition, anomaly findings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import xarray as xa
from scipy.ndimage import label as ndi_label

sys.path.insert(0, str(Path.home() / "OpenLIFU-python/src"))

# Reuse array builder & NIfTI loader from the main pipeline to guarantee pose
# parity with the actual simulations. (No src/ changes; just imports.)
sys.path.insert(0, str(Path(__file__).parent))
from run_gladys_nnunet import (  # noqa: E402
    create_hemispherical_array,
    load_nifti_as_xarray,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path.home() / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects"
)
T1_DIR = DATA_ROOT / "T1-Weighted MRI"
RESULTS_DIR = Path.home() / "Data/openlifu-validation/results"

SUBJECTS = ["GU008", "GU010"]

# Array params (verbatim from run_gladys_nnunet.py)
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0

# Same label-to-material map that the pipeline uses. We only need semantics
# for composition + skull mask here.
# nnU-Net fullhead labels (LABEL_MAP_FULLHEAD in run_gladys_nnunet.py):
#   0 = background -> water
#   1 = air
#   2 = csf
#   3 = gray_matter
#   4 = white_matter
#   5 = skull
#   6 = tissue
NN_SKULL = 5
NN_AIR = 1
NN_CSF = 2
NN_GM = 3
NN_WM = 4
NN_TISSUE = 6
NN_BG = 0

LABEL_NAMES = {
    0: "water/bg", 1: "air", 2: "csf", 3: "gray", 4: "white",
    5: "skull", 6: "tissue",
}


# ---------------------------------------------------------------------------
# Pose (brain-centroid target + y-axis approach) - matches run_gladys_nnunet.py
# for the Birnbaum subjects (all 4 logs show "Approach axis: y (axis 1)").
# ---------------------------------------------------------------------------
def compute_target_and_approach(labels_arr: np.ndarray,
                                coord_arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    brain = (labels_arr == NN_CSF) | (labels_arr == NN_GM) | (labels_arr == NN_WM)
    if not brain.any():
        brain = labels_arr == NN_TISSUE
    idx = np.argwhere(brain)
    dim_names = ("x", "y", "z")
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][idx[:, ax]]))
        for ax in range(3)
    ])
    skull_idx = np.argwhere(labels_arr == NN_SKULL)
    max_dist = np.array([
        float(coord_arrays[dim_names[ax]][skull_idx[:, ax]].max() - target_mm[ax])
        for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_dist))
    return target_mm, approach_axis


def bake_array_positions(target_mm: np.ndarray, approach_axis: int) -> np.ndarray:
    """Returns (64, 3) world-frame element positions, matching run_gladys_nnunet.py."""
    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )
    transform = np.eye(4)
    if approach_axis == 0:
        angle = np.pi / 2
        transform[:3, :3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
    elif approach_axis == 1:
        angle = -np.pi / 2
        transform[:3, :3] = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
    transform[:3, 3] = target_mm
    positions = np.array([
        el.get_position(units="mm", matrix=transform) for el in arr_local.elements
    ])
    return positions


# ---------------------------------------------------------------------------
# Per-ray path length: march from element -> target, sample labels NN.
# ---------------------------------------------------------------------------
def per_ray_path_lengths(
    positions_mm: np.ndarray,
    target_mm: np.ndarray,
    labels_arr: np.ndarray,
    coord_arrays: dict[str, np.ndarray],
    step_mm: float = 0.25,
    target_keepout_mm: float = 2.0,
) -> dict:
    """For each element, integrate path-length (mm) of each label class along the
    straight line from element to target. Skips the last `target_keepout_mm` to
    avoid counting the target voxel itself. Returns per-element arrays."""
    n_el = positions_mm.shape[0]
    cx = coord_arrays["x"]
    cy = coord_arrays["y"]
    cz = coord_arrays["z"]

    # Precompute helpers for nearest-neighbor lookup on a uniform grid.
    x0, dx = float(cx[0]), float(cx[1] - cx[0])
    y0, dy = float(cy[0]), float(cy[1] - cy[0])
    z0, dz = float(cz[0]), float(cz[1] - cz[0])
    nx, ny, nz = len(cx), len(cy), len(cz)

    def nearest_label(pts_mm: np.ndarray) -> np.ndarray:
        ix = np.clip(np.rint((pts_mm[:, 0] - x0) / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((pts_mm[:, 1] - y0) / dy).astype(int), 0, ny - 1)
        iz = np.clip(np.rint((pts_mm[:, 2] - z0) / dz).astype(int), 0, nz - 1)
        return labels_arr[ix, iy, iz]

    per_ray_skull_mm = np.zeros(n_el)
    per_ray_tissue_mm = np.zeros(n_el)
    per_ray_air_mm = np.zeros(n_el)
    per_ray_total_mm = np.zeros(n_el)

    for i in range(n_el):
        p_el = positions_mm[i]
        vec = target_mm - p_el
        L = float(np.linalg.norm(vec))
        if L <= target_keepout_mm:
            continue
        L_eff = L - target_keepout_mm
        u = vec / L
        t_vals = np.arange(0.0, L_eff, step_mm)
        pts = p_el + np.outer(t_vals, u)
        lbls = nearest_label(pts)
        per_ray_skull_mm[i] = float(np.sum(lbls == NN_SKULL)) * step_mm
        per_ray_tissue_mm[i] = float(np.sum(lbls == NN_TISSUE)) * step_mm
        per_ray_air_mm[i] = float(np.sum(lbls == NN_AIR)) * step_mm
        per_ray_total_mm[i] = L_eff

    return dict(
        skull=per_ray_skull_mm,
        tissue=per_ray_tissue_mm,
        air=per_ray_air_mm,
        total=per_ray_total_mm,
    )


# ---------------------------------------------------------------------------
# Segmentation anomaly checks
# ---------------------------------------------------------------------------
def segmentation_anomalies(labels_arr: np.ndarray) -> dict:
    skull = labels_arr == NN_SKULL
    brain = (labels_arr == NN_CSF) | (labels_arr == NN_GM) | (labels_arr == NN_WM)

    # Connected components on skull (6-connectivity)
    cc_labels, n_cc = ndi_label(skull)
    cc_sizes = np.bincount(cc_labels.ravel())[1:] if n_cc > 0 else np.array([])
    cc_sizes_sorted = np.sort(cc_sizes)[::-1]
    top5 = cc_sizes_sorted[:5].tolist() if len(cc_sizes_sorted) else []
    largest = int(cc_sizes_sorted[0]) if len(cc_sizes_sorted) else 0
    total_skull = int(skull.sum())
    frac_in_largest = largest / total_skull if total_skull else 0.0

    # Skull voxels nominally inside brain tissue (should be 0 with proper seg).
    # A skull voxel that is itself labeled skull cannot also be CSF/GM/WM - they
    # are exclusive. What we really want: any skull voxel fully surrounded by
    # brain in a 5x5x5 neighborhood? That's expensive. Instead count skull
    # voxels that sit >5mm inside the brain-mask convex hull is overkill;
    # simple proxy: fraction of skull voxels for which all 6 face-neighbors are
    # brain labels (CSF/GM/WM).
    pad = np.pad(labels_arr, 1, constant_values=NN_BG)
    # 6 face-neighbors of each voxel in original.
    # We only care where center == skull.
    c = pad[1:-1, 1:-1, 1:-1]
    neighbors = [
        pad[0:-2, 1:-1, 1:-1], pad[2:, 1:-1, 1:-1],
        pad[1:-1, 0:-2, 1:-1], pad[1:-1, 2:, 1:-1],
        pad[1:-1, 1:-1, 0:-2], pad[1:-1, 1:-1, 2:],
    ]
    brain_neigh = np.zeros_like(c, dtype=np.uint8)
    for nb in neighbors:
        brain_neigh += (
            (nb == NN_CSF) | (nb == NN_GM) | (nb == NN_WM)
        ).astype(np.uint8)
    # Skull voxels entirely surrounded by brain (all 6 face neighbors brain).
    n_skull_in_brain = int(np.sum((c == NN_SKULL) & (brain_neigh == 6)))
    # Skull voxels with >=5 brain neighbors (loose).
    n_skull_mostly_in_brain = int(np.sum((c == NN_SKULL) & (brain_neigh >= 5)))

    # L/R symmetry: split along x (LEFT half vs RIGHT half of the volume).
    mid_x = skull.shape[0] // 2
    n_skull_L = int(skull[:mid_x].sum())
    n_skull_R = int(skull[mid_x:].sum())
    asym = abs(n_skull_L - n_skull_R) / max(n_skull_L + n_skull_R, 1)

    # Hole test: Is brain surrounded by skull+tissue+air without direct water
    # exposure? If any brain voxel is adjacent (face) to water/background, note
    # that (large holes in vault -> direct brain-water contact).
    water_neigh = np.zeros_like(c, dtype=np.uint8)
    for nb in neighbors:
        water_neigh += (nb == NN_BG).astype(np.uint8)
    brain_c = (c == NN_CSF) | (c == NN_GM) | (c == NN_WM)
    n_brain_touching_water = int(np.sum(brain_c & (water_neigh >= 1)))

    return dict(
        total_skull=total_skull,
        n_components=int(n_cc),
        top5_cc_sizes=top5,
        largest_cc_frac=frac_in_largest,
        n_skull_in_brain_strict=n_skull_in_brain,
        n_skull_in_brain_loose=n_skull_mostly_in_brain,
        n_skull_L=n_skull_L,
        n_skull_R=n_skull_R,
        asym_frac=asym,
        n_brain_touching_water=n_brain_touching_water,
    )


# ---------------------------------------------------------------------------
# Orthogonal-slice PNG with skull overlay + target marker
# ---------------------------------------------------------------------------
def save_orthogonal_overlay(
    mri: np.ndarray, labels_arr: np.ndarray,
    coord_arrays: dict[str, np.ndarray],
    target_mm: np.ndarray, subject: str, out_png: Path,
):
    cx, cy, cz = coord_arrays["x"], coord_arrays["y"], coord_arrays["z"]
    ix = int(np.argmin(np.abs(cx - target_mm[0])))
    iy = int(np.argmin(np.abs(cy - target_mm[1])))
    iz = int(np.argmin(np.abs(cz - target_mm[2])))

    skull = labels_arr == NN_SKULL
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # axial: fix z, show x-y
    mri_ax = mri[:, :, iz].T
    sk_ax = skull[:, :, iz].T
    axes[0].imshow(mri_ax, cmap="gray", origin="lower",
                   extent=[cx[0], cx[-1], cy[0], cy[-1]])
    axes[0].contour(sk_ax, levels=[0.5], colors="red", linewidths=0.8,
                    extent=[cx[0], cx[-1], cy[0], cy[-1]], origin="lower")
    axes[0].plot(target_mm[0], target_mm[1], "y+", ms=18, mew=2)
    axes[0].set_title(f"{subject} axial (z={cz[iz]:.1f}mm)")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")

    # coronal: fix y, show x-z
    mri_co = mri[:, iy, :].T
    sk_co = skull[:, iy, :].T
    axes[1].imshow(mri_co, cmap="gray", origin="lower",
                   extent=[cx[0], cx[-1], cz[0], cz[-1]])
    axes[1].contour(sk_co, levels=[0.5], colors="red", linewidths=0.8,
                    extent=[cx[0], cx[-1], cz[0], cz[-1]], origin="lower")
    axes[1].plot(target_mm[0], target_mm[2], "y+", ms=18, mew=2)
    axes[1].set_title(f"{subject} coronal (y={cy[iy]:.1f}mm)")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("z [mm]")

    # sagittal: fix x, show y-z
    mri_sa = mri[ix, :, :].T
    sk_sa = skull[ix, :, :].T
    axes[2].imshow(mri_sa, cmap="gray", origin="lower",
                   extent=[cy[0], cy[-1], cz[0], cz[-1]])
    axes[2].contour(sk_sa, levels=[0.5], colors="red", linewidths=0.8,
                    extent=[cy[0], cy[-1], cz[0], cz[-1]], origin="lower")
    axes[2].plot(target_mm[1], target_mm[2], "y+", ms=18, mew=2)
    axes[2].set_title(f"{subject} sagittal (x={cx[ix]:.1f}mm)")
    axes[2].set_xlabel("y [mm]"); axes[2].set_ylabel("z [mm]")

    fig.suptitle(f"{subject}: MRI + nnU-Net skull overlay, yellow + = target")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-ray path length histogram figure (both subjects overlaid).
# ---------------------------------------------------------------------------
def save_path_histograms(per_ray: dict[str, dict], out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # skull
    bins = np.linspace(0, max(
        per_ray[s]["skull"].max() for s in per_ray
    ) * 1.05, 21)
    for subj, data in per_ray.items():
        axes[0].hist(data["skull"], bins=bins, alpha=0.5, label=subj)
    axes[0].set_xlabel("per-ray skull path length (mm)")
    axes[0].set_ylabel("# elements")
    axes[0].set_title("64-element skull path-length distribution (nnU-Net labels)")
    axes[0].legend()

    # total ray length
    bins2 = np.linspace(0, max(
        per_ray[s]["total"].max() for s in per_ray
    ) * 1.05, 21)
    for subj, data in per_ray.items():
        axes[1].hist(data["total"], bins=bins2, alpha=0.5, label=subj)
    axes[1].set_xlabel("per-ray total length (mm) [element -> target, minus keepout]")
    axes[1].set_ylabel("# elements")
    axes[1].set_title("total ray length distribution")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def summarize(name: str, arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"  {name:<8s}: <empty>"
    return (f"  {name:<8s}: n={arr.size}  mean={arr.mean():.2f}  std={arr.std():.2f}  "
            f"min={arr.min():.2f}  p50={np.percentile(arr, 50):.2f}  "
            f"p90={np.percentile(arr, 90):.2f}  p99={np.percentile(arr, 99):.2f}  "
            f"max={arr.max():.2f}")


def main():
    print("=" * 78)
    print("GU010 investigation: segmentation + per-ray path-length + anomalies")
    print("=" * 78)

    per_ray_by_subj: dict[str, dict] = {}

    for subj in SUBJECTS:
        mri_path = T1_DIR / f"{subj}_deface.nii"
        lbl_path = RESULTS_DIR / f"{subj}_nnunet_labels.nii.gz"
        print(f"\n----- {subj} -----")
        print(f"  MRI:    {mri_path}")
        print(f"  labels: {lbl_path}")

        mri_xa = load_nifti_as_xarray(mri_path)
        lbl_img = nib.load(str(lbl_path))
        lbl_arr = np.asarray(lbl_img.dataobj).astype(np.int16)
        lbl_affine = lbl_img.affine

        # Build label coord arrays from affine diagonal (labels grid = MRI grid,
        # as generated by the nnU-Net pipeline).
        dim_names = ("x", "y", "z")
        lbl_coords = {}
        for ax, d in enumerate(dim_names):
            origin = float(lbl_affine[ax, 3])
            spacing = float(lbl_affine[ax, ax])
            lbl_coords[d] = origin + np.arange(lbl_arr.shape[ax]) * spacing

        # Composition
        total = lbl_arr.size
        comp = {k: int(np.sum(lbl_arr == k)) for k in range(7)}
        print("  composition (nnU-Net labels):")
        for k, n in comp.items():
            pct = 100.0 * n / total
            print(f"    {k} {LABEL_NAMES[k]:<10s}: {n:>10,d} ({pct:5.2f}%)")
        bone_frac = 100.0 * comp[NN_SKULL] / total
        print(f"  bone fraction: {bone_frac:.2f}%")

        # Target + approach
        target_mm, approach_axis = compute_target_and_approach(lbl_arr, lbl_coords)
        print(f"  target (world): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")
        print(f"  approach axis: {dim_names[approach_axis]} (axis {approach_axis})")

        # 64-element positions
        positions = bake_array_positions(target_mm, approach_axis)
        ap_center = positions.mean(axis=0)
        ap_axis_len = float(np.linalg.norm(ap_center - target_mm))
        print(f"  aperture center: ({ap_center[0]:.1f}, {ap_center[1]:.1f}, {ap_center[2]:.1f}) mm, "
              f"|target-apcenter|={ap_axis_len:.1f} mm")
        el_dists = np.linalg.norm(positions - target_mm, axis=1)
        print(f"  per-element target distance: mean={el_dists.mean():.1f}, "
              f"std={el_dists.std():.2f}, min={el_dists.min():.1f}, max={el_dists.max():.1f}")

        # Per-ray path lengths
        per_ray = per_ray_path_lengths(
            positions_mm=positions, target_mm=target_mm,
            labels_arr=lbl_arr, coord_arrays=lbl_coords,
        )
        per_ray_by_subj[subj] = per_ray
        print(f"  per-ray SKULL path (mm):")
        print(summarize("skull", per_ray["skull"]))
        print(summarize("tissue", per_ray["tissue"]))
        print(summarize("air", per_ray["air"]))
        print(summarize("total", per_ray["total"]))
        # Tail (long-path elements)
        n_ge_15 = int(np.sum(per_ray["skull"] >= 15.0))
        n_ge_20 = int(np.sum(per_ray["skull"] >= 20.0))
        n_ge_25 = int(np.sum(per_ray["skull"] >= 25.0))
        print(f"  rays with skull-path >=15/>=20/>=25 mm: "
              f"{n_ge_15}/{n_ge_20}/{n_ge_25}  (of {N_ELEMENTS})")

        # Integrated skull path across aperture (proxy for aggregate attenuation).
        SKULL_ALPHA_DB_PER_CM = 8.0 * (0.5 ** 0.9)  # ~4.29 dB/cm @ 500 kHz
        per_ray_skull_cm = per_ray["skull"] / 10.0
        per_ray_bulk_loss_db = per_ray_skull_cm * SKULL_ALPHA_DB_PER_CM
        print(f"  per-ray bulk bone attenuation at 500 kHz: "
              f"mean={per_ray_bulk_loss_db.mean():.2f} dB, "
              f"std={per_ray_bulk_loss_db.std():.2f}, "
              f"max={per_ray_bulk_loss_db.max():.2f}")

        # Anomaly checks
        an = segmentation_anomalies(lbl_arr)
        print(f"  skull connected components: n={an['n_components']}, "
              f"top5 sizes={an['top5_cc_sizes']}, "
              f"largest-cc fraction of total skull={an['largest_cc_frac']:.4f}")
        print(f"  skull voxels strictly inside brain (6/6 face-neighbors are brain): "
              f"{an['n_skull_in_brain_strict']}")
        print(f"  skull voxels mostly inside brain (>=5/6 face-neighbors are brain): "
              f"{an['n_skull_in_brain_loose']}")
        print(f"  L/R asymmetry (skull): L={an['n_skull_L']:,d}, R={an['n_skull_R']:,d}, "
              f"asym_frac={an['asym_frac']:.3f}")
        print(f"  brain voxels with water/bg face-neighbor (potential vault hole): "
              f"{an['n_brain_touching_water']}")

        # Save overlay PNG. For GU010 use the canonical name requested;
        # for others, save a sibling for comparison.
        mri_arr = np.asarray(mri_xa.to_numpy())
        if subj == "GU010":
            out_png = RESULTS_DIR / "gu010_seg_check.png"
        else:
            out_png = RESULTS_DIR / f"{subj.lower()}_seg_check.png"
        save_orthogonal_overlay(
            mri_arr, lbl_arr, lbl_coords, target_mm, subj, out_png,
        )
        print(f"  saved overlay -> {out_png}")

    # Comparison histogram figure
    hist_png = RESULTS_DIR / "gu010_path_histograms.png"
    save_path_histograms(per_ray_by_subj, hist_png)
    print(f"\nSaved per-ray path histograms -> {hist_png}")

    # Direct comparison summary
    print("\n" + "=" * 78)
    print("DIRECT COMPARISON: per-ray skull path length (mm), nnU-Net labels")
    print("=" * 78)
    header = f"{'subj':<8s} {'mean':>7s} {'std':>7s} {'min':>7s} {'p50':>7s} {'p90':>7s} {'p99':>7s} {'max':>7s} {'n>=15':>6s} {'n>=20':>6s}"
    print(header)
    print("-" * len(header))
    for subj in SUBJECTS:
        a = per_ray_by_subj[subj]["skull"]
        print(f"{subj:<8s} {a.mean():>7.2f} {a.std():>7.2f} {a.min():>7.2f} "
              f"{np.percentile(a, 50):>7.2f} {np.percentile(a, 90):>7.2f} "
              f"{np.percentile(a, 99):>7.2f} {a.max():>7.2f} "
              f"{int(np.sum(a >= 15)):>6d} {int(np.sum(a >= 20)):>6d}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
