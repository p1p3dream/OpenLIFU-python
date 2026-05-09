"""Quantify skull path length and expected loss along the target-to-aperture axis.

Reuses the same MRI segmentation + material assignment as run_gladys_simulation.py
without running k-wave. Reports:
  - Skull path length (mm) along the axis
  - Expected attenuation from bulk absorption (dB)
  - Expected impedance reflection loss (dB)
  - Total expected loss vs observed ~55 dB
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import xarray as xa

sys.path.insert(0, str(Path.home() / "OpenLIFU-python/src"))
from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI

MRI_PATH = Path.home() / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)
RESULTS_DIR = Path.home() / "Data/openlifu-validation/results"
GRID_SPACING_MM = 0.5
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_MHZ = 0.5
SKULL_ATTEN_COEFF = 8.0  # dB/cm/MHz from material.py:109
POWER_LAW_Y = 0.9


def load_nifti_xarray(path):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    coords = {}
    for axis, dim in enumerate(("x", "y", "z")):
        origin = float(affine[axis, 3])
        spacing = float(affine[axis, axis])
        coords[dim] = xa.Variable(dim, origin + np.arange(data.shape[axis]) * spacing, attrs={"units": "mm"})
    return xa.DataArray(data, dims=("x", "y", "z"), coords=coords)


def hemispherical_positions(target_mm, approach_axis):
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
        angle = -np.pi / 2
        R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        world = local @ R.T + target_mm
    else:
        world = local + target_mm
    return world


def find_target_and_axis(volume, seg_method):
    seg_labels = seg_method._segment(volume).to_numpy()
    mi = seg_method._material_indices()
    tissue_mask = seg_labels == mi["tissue"]
    tissue_idx = np.argwhere(tissue_mask)
    dim_names = ("x", "y", "z")
    coord_arrays = {d: volume.coords[d].to_numpy() for d in dim_names}
    target_mm = np.array([float(np.mean(coord_arrays[dim_names[ax]][tissue_idx[:, ax]])) for ax in range(3)])
    skull_mask = seg_labels == mi["skull"]
    skull_idx = np.argwhere(skull_mask)
    skull_mm = np.array([coord_arrays[dim_names[ax]][skull_idx[:, ax]] for ax in range(3)])
    dists = np.array([float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)])
    return target_mm, int(np.argmax(dists))


def main():
    print("Loading MRI...")
    volume = load_nifti_xarray(MRI_PATH)
    seg = ThresholdMRI(
        skull_thickness_mm=12.0, air_threshold_quantile=0.05,
        classify_brain_tissues=False, refine_skull_intensity=True,
    )
    print("Finding target from native MRI segmentation...")
    target_mm, approach_axis = find_target_and_axis(volume, seg)
    print(f"  target: ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm, approach_axis={approach_axis}")

    positions = hemispherical_positions(target_mm, approach_axis)
    ap_center = positions.mean(axis=0)
    print(f"  aperture center: ({ap_center[0]:.1f}, {ap_center[1]:.1f}, {ap_center[2]:.1f}) mm")

    # Resample volume to the SAME grid the sim used (0.5mm iso, snapped around elements+target)
    print("\nResampling to sim grid (0.5mm iso)...")
    all_pts = np.vstack([positions, target_mm[np.newaxis, :]])
    grid_min = np.floor((all_pts.min(axis=0) - 10.0) / GRID_SPACING_MM) * GRID_SPACING_MM
    grid_max = np.ceil((all_pts.max(axis=0) + 10.0) / GRID_SPACING_MM) * GRID_SPACING_MM
    sim_coords = {}
    for ax, dim in enumerate(("x", "y", "z")):
        n_pts = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        sim_coords[dim] = xa.Variable(dim, np.linspace(grid_min[ax], grid_max[ax], n_pts), attrs={"units": "mm"})
    grid_shape = tuple(len(sim_coords[d]) for d in ("x", "y", "z"))
    print(f"  sim grid shape: {grid_shape}")

    from scipy.interpolate import RegularGridInterpolator
    orig_coords = [volume.coords[d].to_numpy() for d in volume.dims]
    interp = RegularGridInterpolator(orig_coords, volume.to_numpy(), method="linear", bounds_error=False, fill_value=0.0)
    sim_coord_arrays = [sim_coords[d].data for d in ("x", "y", "z")]
    mg = np.meshgrid(*sim_coord_arrays, indexing="ij")
    query = np.stack([m.ravel() for m in mg], axis=-1)
    resampled = interp(query).reshape(grid_shape).astype(np.float32)
    sim_volume = xa.DataArray(resampled, dims=("x", "y", "z"), coords={d: sim_coords[d] for d in ("x", "y", "z")})

    print("\nSegmenting on sim grid...")
    sim_seg = seg._segment(sim_volume).to_numpy()
    mi = seg._material_indices()

    total_vox = sim_seg.size
    print(f"\nSim-grid segmentation composition:")
    for name, idx in mi.items():
        n = int(np.sum(sim_seg == idx))
        pct = 100.0 * n / total_vox
        print(f"  {name:>10s} (idx={idx}): {n:>10,d} voxels ({pct:.2f}%)")

    # ===================================================================
    # Axial ray: target -> aperture center, count labels
    # ===================================================================
    axis_vec = ap_center - target_mm
    axis_len_mm = float(np.linalg.norm(axis_vec))
    axis_unit = axis_vec / axis_len_mm

    # Sample at 0.25mm resolution (finer than grid)
    step_mm = 0.25
    t_vals = np.arange(0, axis_len_mm, step_mm)
    sample_pts = target_mm + np.outer(t_vals, axis_unit)

    cx = sim_coords["x"].data
    cy = sim_coords["y"].data
    cz = sim_coords["z"].data
    ray_labels = []
    for p in sample_pts:
        ix = int(np.argmin(np.abs(cx - p[0])))
        iy = int(np.argmin(np.abs(cy - p[1])))
        iz = int(np.argmin(np.abs(cz - p[2])))
        ray_labels.append(sim_seg[ix, iy, iz])
    ray_labels = np.array(ray_labels)

    print(f"\nAxial ray: target -> aperture center, length {axis_len_mm:.1f} mm, step {step_mm}mm")
    for name, idx in mi.items():
        n = int(np.sum(ray_labels == idx))
        mm = n * step_mm
        print(f"  {name:>10s} path length along ray: {mm:.2f} mm ({n} samples)")

    skull_path_mm = float(np.sum(ray_labels == mi["skull"]) * step_mm)
    skull_path_cm = skull_path_mm / 10.0

    # ===================================================================
    # Expected loss calculations
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPECTED LOSS vs OBSERVED")
    print("=" * 70)

    # Bulk absorption: alpha = alpha0 * f^y dB/cm
    alpha_at_freq = SKULL_ATTEN_COEFF * (FREQ_MHZ ** POWER_LAW_Y)
    bulk_loss_db = alpha_at_freq * skull_path_cm
    print(f"Skull attenuation coefficient (alpha_0): {SKULL_ATTEN_COEFF} dB/cm/MHz^{POWER_LAW_Y}")
    print(f"Effective alpha at {FREQ_MHZ} MHz: {alpha_at_freq:.2f} dB/cm")
    print(f"Skull path along ray: {skull_path_mm:.2f} mm = {skull_path_cm:.2f} cm")
    print(f"Bulk absorption loss: {bulk_loss_db:.2f} dB")

    # Impedance mismatch: Z_skull / Z_water
    Z_water = 1500.0 * 1000.0  # c * rho
    Z_skull = 4080.0 * 1900.0
    T_amp = 2 * Z_skull / (Z_skull + Z_water)  # water -> skull
    T_amp_rev = 2 * Z_water / (Z_skull + Z_water)  # skull -> water
    # For a slab: amplitude transmission water -> skull -> water (neglecting internal reflections):
    T_slab = T_amp * T_amp_rev
    # Actually use R (reflection coeff) for single interface:
    R_amp = abs(Z_skull - Z_water) / (Z_skull + Z_water)
    T_single_interface_energy = 1 - R_amp ** 2
    # Double interface (enter + exit skull): amplitude factor = (1 - R_amp^2) ≈ for energy
    T_slab_energy = T_single_interface_energy ** 2
    impedance_loss_db = -10 * np.log10(T_slab_energy)
    print(f"\nZ_water = {Z_water/1e6:.2f} MRayl, Z_skull = {Z_skull/1e6:.2f} MRayl")
    print(f"Single-interface amplitude reflection: R = {R_amp:.3f}")
    print(f"Double-interface (slab) energy transmission: {T_slab_energy:.3f}")
    print(f"Impedance reflection loss: {impedance_loss_db:.2f} dB")

    # Total
    total_expected = bulk_loss_db + impedance_loss_db
    print(f"\nTotal expected loss (bulk + impedance): {total_expected:.2f} dB")
    print(f"Observed loss: ~55 dB (p@target dropped from 0.056 Pa water to 0.0001 Pa skull)")
    print(f"Unaccounted gap: {55.0 - total_expected:.2f} dB")

    # ===================================================================
    # Plot the label map along the ray and the sound_speed along it
    # ===================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Label codes as ints
    color_map = {mi["water"]: 0, mi["tissue"]: 1, mi["skull"]: 2, mi["air"]: 3}
    ray_colored = np.array([color_map.get(int(l), -1) for l in ray_labels])

    ax = axes[0]
    ax.imshow(ray_colored[np.newaxis, :], aspect="auto",
              extent=[t_vals[0], t_vals[-1], 0, 1],
              cmap=plt.cm.get_cmap("tab10", 4), vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_xlabel("distance from target along axis (mm)")
    ax.set_title(f"Segmentation labels along target->aperture axis\n"
                 f"(skull path = {skull_path_mm:.1f} mm, observed 55 dB loss, predicted {total_expected:.1f} dB)")

    # Second row: histogram of skull thicknesses perpendicular to axis
    # For each slice perpendicular to axis, count skull voxels in a 40mm-radius disk
    ax = axes[1]
    ax.set_title("skull voxel count along ray (per 0.25mm step)")
    is_skull = (ray_labels == mi["skull"]).astype(int)
    is_water = (ray_labels == mi["water"]).astype(int)
    is_tissue = (ray_labels == mi["tissue"]).astype(int)
    ax.fill_between(t_vals, 0, is_skull * 3, alpha=0.6, label="skull", color="red")
    ax.fill_between(t_vals, 0, is_tissue * 1, alpha=0.6, label="tissue", color="orange")
    ax.fill_between(t_vals, 0, is_water * 2, alpha=0.6, label="water", color="blue")
    ax.set_xlabel("distance from target along axis (mm)")
    ax.set_ylabel("label class")
    ax.legend()

    plt.tight_layout()
    out = RESULTS_DIR / "gladys_v7_skull_path_2026-04-18.png"
    plt.savefig(out, dpi=100)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
