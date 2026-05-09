"""Re-analyze the saved gladys_v7 p_max NIfTIs without re-running k-wave.

Sweeps the source-exclusion mask radius and reports, for each sim:
  - location and value of p_max at each radius
  - focal error (distance from target)
  - a 1D profile along the transducer-to-target axis
  - p@target for comparison

Also saves a 2D slice plot through the target-aperture axis as PNG.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Config must match scripts/run_gladys_simulation.py
RESULTS_DIR = Path.home() / "Data/openlifu-validation/results"
MRI_PATH = Path.home() / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)
GRID_SPACING_MM = 0.5
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0

sys.path.insert(0, str(Path.home() / "OpenLIFU-python/src"))
from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI

def load_nifti_as_xarray_coords(nifti_path):
    import xarray as xa
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    coords = {}
    for axis, dim in enumerate(("x", "y", "z")):
        origin = float(affine[axis, 3])
        spacing = float(affine[axis, axis])
        coords[dim] = origin + np.arange(data.shape[axis]) * spacing
    return data, coords


def hemispherical_positions(target_mm, approach_axis):
    """Re-create the element world positions used by run_gladys_simulation.py."""
    half_aperture = APERTURE_MM / 2.0
    theta_max = np.arcsin(half_aperture / RADIUS_MM)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    local = np.zeros((N_ELEMENTS, 3))
    for i in range(N_ELEMENTS):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / N_ELEMENTS
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        local[i, 0] = RADIUS_MM * np.sin(theta) * np.cos(phi)
        local[i, 1] = RADIUS_MM * np.sin(theta) * np.sin(phi)
        local[i, 2] = RADIUS_MM * np.cos(theta)
    if approach_axis == 1:
        # Rx(-90): local (x,y,z) -> world (x, -z, y) then + target
        angle = -np.pi / 2
        R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        world = local @ R.T + target_mm
    elif approach_axis == 0:
        angle = np.pi / 2
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        world = local @ R.T + target_mm
    else:
        world = local + target_mm
    return world


def find_target(mri_path):
    """Return target_mm and approach_axis the same way the main script does."""
    import xarray as xa
    img = nib.load(str(mri_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    dim_names = ("x", "y", "z")
    coords = {}
    for axis, dim in enumerate(dim_names):
        origin = float(affine[axis, 3])
        spacing = float(affine[axis, axis])
        coords[dim] = xa.Variable(dim, origin + np.arange(data.shape[axis]) * spacing, attrs={"units": "mm"})
    volume = xa.DataArray(data, dims=dim_names, coords=coords)
    seg = ThresholdMRI(skull_thickness_mm=12.0, air_threshold_quantile=0.05,
                       classify_brain_tissues=False, refine_skull_intensity=True)
    seg_labels = seg._segment(volume).to_numpy()
    mi = seg._material_indices()
    tissue_mask = seg_labels == mi["tissue"]
    tissue_idx = np.argwhere(tissue_mask)
    coord_arrays = {d: volume.coords[d].to_numpy() for d in dim_names}
    target_mm = np.array([float(np.mean(coord_arrays[dim_names[ax]][tissue_idx[:, ax]])) for ax in range(3)])
    skull_mask = seg_labels == mi["skull"]
    skull_idx = np.argwhere(skull_mask)
    skull_mm = np.array([coord_arrays[dim_names[ax]][skull_idx[:, ax]] for ax in range(3)])
    dists = np.array([float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)])
    return target_mm, int(np.argmax(dists))


def compute_focal_gain(pmax, coords, target_mm, element_positions_mm,
                       aperture_band_radius_mm=10.0):
    """Focal gain = p@target divided by near-aperture reference pressure.

    The "aperture band" is the union of spherical shells of radius
    aperture_band_radius_mm around every element position. We report gain
    both versus the mean and versus the max p_max inside that band, since:
      - gain_vs_mean > 1 means the target beats the average near-aperture
        pressure (a reasonable focus-quality sanity check).
      - gain_vs_max > 1 means the target beats the brightest Fresnel/near-
        field lobe near the aperture. For a well-focused skull sim this is
        the harder test; raw argmax often sits in those near-aperture lobes
        which is why the naive argmax metric can be misleading.

    For a correctly-focused 64-element array through skull, gain_vs_mean is
    typically ~2-5x; through water it can be 10-50x. gain < 1 indicates the
    focus does not dominate its local aperture neighborhood.

    Returns a dict with p_at_target, p_aperture_mean, p_aperture_max,
    n_aperture_voxels, gain_vs_mean, gain_vs_max.
    """
    dims = ("x", "y", "z")
    cx, cy, cz = coords["x"], coords["y"], coords["z"]
    xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
    coord_stack = np.stack([xx, yy, zz], axis=-1)

    # p@target: nearest voxel
    tidx = tuple(int(np.argmin(np.abs(coords[dims[ax]] - target_mm[ax]))) for ax in range(3))
    p_at_target = float(pmax[tidx])

    # Aperture band mask: voxels within aperture_band_radius_mm of any element
    r2 = aperture_band_radius_mm ** 2
    band_mask = np.zeros(pmax.shape, dtype=bool)
    for pos in element_positions_mm:
        d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
        band_mask |= d2 <= r2

    n_band = int(band_mask.sum())
    if n_band == 0:
        return {
            "p_at_target": p_at_target,
            "p_aperture_mean": float("nan"),
            "p_aperture_max": float("nan"),
            "n_aperture_voxels": 0,
            "gain_vs_mean": float("nan"),
            "gain_vs_max": float("nan"),
            "aperture_band_radius_mm": aperture_band_radius_mm,
        }

    band_vals = pmax[band_mask]
    p_ap_mean = float(band_vals.mean())
    p_ap_max = float(band_vals.max())
    gain_mean = p_at_target / p_ap_mean if p_ap_mean > 0 else float("nan")
    gain_max = p_at_target / p_ap_max if p_ap_max > 0 else float("nan")
    return {
        "p_at_target": p_at_target,
        "p_aperture_mean": p_ap_mean,
        "p_aperture_max": p_ap_max,
        "n_aperture_voxels": n_band,
        "gain_vs_mean": gain_mean,
        "gain_vs_max": gain_max,
        "aperture_band_radius_mm": aperture_band_radius_mm,
    }


def analyze(sim_label, pmax_path, target_mm, positions_world):
    pmax, coords = load_nifti_as_xarray_coords(pmax_path)
    dims = ("x", "y", "z")
    cx, cy, cz = coords["x"], coords["y"], coords["z"]

    # Coordinate meshgrid for mask construction and slicing
    xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
    coord_stack = np.stack([xx, yy, zz], axis=-1)

    print(f"\n{'=' * 72}")
    print(f"{sim_label}: file={pmax_path.name}, shape={pmax.shape}")
    print(f"{'=' * 72}")

    # --- p@target ---
    tidx = tuple(int(np.argmin(np.abs(coords[dims[ax]] - target_mm[ax]))) for ax in range(3))
    p_at_target = float(pmax[tidx])
    print(f"target      = ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")
    print(f"p@target    = {p_at_target:.4f} Pa")
    print(f"raw max_p   = {float(pmax.max()):.4f} Pa @ "
          f"({cx[np.unravel_index(pmax.argmax(), pmax.shape)[0]]:.1f}, "
          f"{cy[np.unravel_index(pmax.argmax(), pmax.shape)[1]]:.1f}, "
          f"{cz[np.unravel_index(pmax.argmax(), pmax.shape)[2]]:.1f}) mm")

    # --- Focal gain (p@target vs near-aperture reference) ---
    fg = compute_focal_gain(pmax, coords, target_mm, positions_world,
                            aperture_band_radius_mm=10.0)
    print(f"aperture band r={fg['aperture_band_radius_mm']:.1f} mm, "
          f"n_voxels={fg['n_aperture_voxels']}")
    print(f"p_aperture_mean = {fg['p_aperture_mean']:.4f} Pa")
    print(f"p_aperture_max  = {fg['p_aperture_max']:.4f} Pa")
    print(f"focal gain (vs mean) = {fg['gain_vs_mean']:.3f}")
    print(f"focal gain (vs max)  = {fg['gain_vs_max']:.3f}")

    # --- Mask sweep ---
    print(f"{'radius_mm':>10} {'max_p':>10} {'x':>7} {'y':>7} {'z':>7} {'err_mm':>8}")
    for r_mm in [3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        mask = np.ones(pmax.shape, dtype=bool)
        for pos in positions_world:
            d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
            mask &= d2 > r_mm ** 2
        if not mask.any():
            print(f"{r_mm:>10.1f} {'MASKED-OUT':>10}")
            continue
        masked = np.where(mask, pmax, -np.inf)
        idx = np.unravel_index(masked.argmax(), masked.shape)
        mp = float(pmax[idx])
        loc = np.array([float(cx[idx[0]]), float(cy[idx[1]]), float(cz[idx[2]])])
        err = float(np.linalg.norm(loc - target_mm))
        print(f"{r_mm:>10.1f} {mp:>10.4f} {loc[0]:>7.1f} {loc[1]:>7.1f} {loc[2]:>7.1f} {err:>8.2f}")

    # --- 1D profile along target-to-aperture-center axis ---
    ap_center = positions_world.mean(axis=0)
    axis_vec = ap_center - target_mm
    axis_len = np.linalg.norm(axis_vec)
    axis_unit = axis_vec / axis_len
    # Sample from 20mm past target toward aperture and 20mm past aperture
    t_vals = np.linspace(-20, axis_len + 20, 200)
    sample_pts = target_mm + np.outer(t_vals, axis_unit)
    # trilinear sample via nearest neighbor for speed (grid is 0.5mm iso)
    profile = []
    for p in sample_pts:
        ix = int(np.argmin(np.abs(cx - p[0])))
        iy = int(np.argmin(np.abs(cy - p[1])))
        iz = int(np.argmin(np.abs(cz - p[2])))
        if 0 <= ix < pmax.shape[0] and 0 <= iy < pmax.shape[1] and 0 <= iz < pmax.shape[2]:
            profile.append(float(pmax[ix, iy, iz]))
        else:
            profile.append(np.nan)
    profile = np.array(profile)

    return {
        "label": sim_label,
        "pmax": pmax,
        "coords": coords,
        "target_mm": target_mm,
        "ap_center": ap_center,
        "t_vals": t_vals,
        "profile": profile,
    }


def make_plots(results, positions_world, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, r in enumerate(results):
        pmax = r["pmax"]
        cx = r["coords"]["x"]
        cy = r["coords"]["y"]
        cz = r["coords"]["z"]
        target_mm = r["target_mm"]
        ap_center = r["ap_center"]

        # --- Top row: slice at z nearest target z (xy plane) ---
        iz = int(np.argmin(np.abs(cz - target_mm[2])))
        slc = pmax[:, :, iz]
        ax = axes[0, col]
        im = ax.imshow(
            slc.T, origin="lower",
            extent=[cx.min(), cx.max(), cy.min(), cy.max()],
            aspect="equal", cmap="magma",
            norm=matplotlib.colors.LogNorm(
                vmin=max(1e-6, float(pmax.min() + 1e-9)),
                vmax=float(pmax.max()),
            ),
        )
        ax.plot(target_mm[0], target_mm[1], "c*", markersize=18, label="target")
        ax.plot(positions_world[:, 0], positions_world[:, 1], "w.", markersize=3, label="elements")
        ax.plot(ap_center[0], ap_center[1], "g+", markersize=18, mew=2, label="aperture center")
        ax.set_title(f"{r['label']}\nxy slice at z={cz[iz]:.1f}mm (log p_max)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04, label="p_max (Pa)")

        # --- Bottom row: 1D profile along target->aperture axis ---
        ax = axes[1, col]
        ax.plot(r["t_vals"], r["profile"], "b-", label="p_max along axis")
        ax.axvline(0, color="c", linestyle="--", label="target")
        ax.axvline(np.linalg.norm(ap_center - target_mm), color="g", linestyle="--", label="aperture center")
        ax.set_xlabel("distance from target along axis (mm)")
        ax.set_ylabel("p_max (Pa)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"{r['label']}: axial profile")

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    print(f"\nSaved plot: {out_path}")


def main():
    print("Finding target from MRI segmentation (may take 20s)...")
    target_mm, approach_axis = find_target(MRI_PATH)
    print(f"target: ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm, approach_axis={approach_axis}")

    positions = hemispherical_positions(target_mm, approach_axis)
    print(f"Reconstructed {len(positions)} element positions")
    print(f"  x: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]")
    print(f"  y: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
    print(f"  z: [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}]")
    dists = np.linalg.norm(positions - target_mm, axis=1)
    print(f"  distance to target: {dists.mean():.2f} +/- {dists.std():.2f} mm "
          f"(range [{dists.min():.1f}, {dists.max():.1f}])")

    results = []
    for label, fname in [
        ("A: corrected+skull", "gladys_v7_corrected_pmax.nii.gz"),
        ("B: geometric+skull", "gladys_v7_geometric_pmax.nii.gz"),
        ("C: geometric+water", "gladys_v7_water_pmax.nii.gz"),
    ]:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        r = analyze(label, path, target_mm, positions)
        results.append(r)

    if results:
        out_png = RESULTS_DIR / "gladys_v7_pmax_analysis_2026-04-18.png"
        make_plots(results, positions, out_png)


if __name__ == "__main__":
    main()
