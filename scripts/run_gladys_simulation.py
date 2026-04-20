#!/usr/bin/env python3
"""GLADYS Simulation v7: Full transcranial FUS simulation with timing fixes.

Runs three simulations on GU008 MRI with a 64-element hemispherical array:
  1) Phase-corrected heterogeneous (SimulationCorrected delays + skull medium)
  2) Uncorrected heterogeneous (geometric delays + skull medium)
  3) Homogeneous water reference (geometric delays + ref_values_only)

Key fixes in v7:
  - Explicit t_end = max_distance / c0 * 2.0 (2x safety margin for skull)
  - CFL = 0.1 (stable for skull bone at 4080 m/s)
  - 0.5mm grid spacing (6 ppw at 500kHz in water)
  - Air voxels replaced with water in the medium
  - source_method='point_source' (works with curved arrays)
  - Pre-transformed (baked) element positions

Usage on stonkbot:
  ssh brandon@192.168.68.71 "source ~/openlifu-env/bin/activate && \\
    export LD_LIBRARY_PATH=~/openlifu-env/lib:$LD_LIBRARY_PATH && \\
    PYTHONPATH=~/OpenLIFU-python/src:$PYTHONPATH \\
    python3 -u ~/OpenLIFU-python/scripts/run_gladys_simulation.py"
"""

import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xa

# Ensure local openlifu is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI
from openlifu.sim.kwave_if import run_simulation
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MRI_PATH = Path.home() / (
    "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)

# Array parameters
N_ELEMENTS = 64
RADIUS_MM = 90.0        # radius of curvature
APERTURE_MM = 80.0      # aperture diameter
FREQ_HZ = 500e3         # 500 kHz
ELEMENT_SIZE_MM = 5.0   # element width/length

# Simulation parameters
GRID_SPACING_MM = 0.5   # 6 ppw at 500kHz in water (lambda = 3mm)
CFL = 0.1               # stable for skull (c_skull = 4080 m/s)
CYCLES = 5              # source cycles
AMPLITUDE = 1.0
C0 = 1500.0             # reference sound speed (water, m/s)
T_END_SAFETY = 2.0      # safety factor for t_end calculation

# Grid margin around transducer and target
GRID_MARGIN_MM = 10.0


# ---------------------------------------------------------------------------
# Hemispherical array generator
# ---------------------------------------------------------------------------
def create_hemispherical_array(
    n_elements: int = 64,
    radius_mm: float = 90.0,
    aperture_mm: float = 80.0,
    freq_hz: float = 500e3,
    element_size_mm: float = 5.0,
) -> Transducer:
    """Create a hemispherical transducer array centered at the origin.

    Elements are distributed on a spherical cap using a Fibonacci spiral
    pattern. The cap is defined by the aperture diameter. Each element
    faces inward toward the geometric focus at the origin (0, 0, 0).

    The array is constructed in its local coordinate frame:
      - Geometric focus (center of curvature) at the origin
      - Array aperture faces -z (elements have positive z, looking toward -z)

    Args:
        n_elements: Number of transducer elements.
        radius_mm: Radius of curvature (distance from focus to element surface).
        aperture_mm: Diameter of the array aperture opening.
        freq_hz: Operating frequency in Hz.
        element_size_mm: Width and length of each rectangular element.

    Returns:
        A Transducer object with elements positioned on the spherical cap.
    """
    # Half-angle of the spherical cap from the aperture diameter
    # sin(theta_max) = (aperture/2) / radius
    half_aperture = aperture_mm / 2.0
    if half_aperture > radius_mm:
        raise ValueError(
            f"Aperture radius ({half_aperture}mm) exceeds sphere radius ({radius_mm}mm)"
        )
    theta_max = np.arcsin(half_aperture / radius_mm)

    # Fibonacci spiral distribution on the spherical cap.
    # The cap extends from z = R*cos(0) = R (pole) to z = R*cos(theta_max).
    # We distribute n points with uniform area weighting using the golden angle.
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    elements = []
    for i in range(n_elements):
        # Uniform area distribution: cos(theta) linearly spaced from 1 to cos(theta_max)
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = golden_angle * i

        # Spherical to Cartesian (element position on the cap)
        x = radius_mm * np.sin(theta) * np.cos(phi)
        y = radius_mm * np.sin(theta) * np.sin(phi)
        z = radius_mm * np.cos(theta)

        # Element normal points inward toward the origin (geometric focus).
        # The normal direction is -r_hat = (-x, -y, -z) / R.
        # Orientation angles (az, el, roll) for the Element:
        #   az = atan2(-x, -z)  (azimuth about y-axis)
        #   el = -atan2(-y, sqrt((-x)^2 + (-z)^2))  (elevation about x'-axis)
        nx, ny, nz = -x, -y, -z
        az = np.arctan2(nx, nz)
        el = -np.arctan2(ny, np.sqrt(nx**2 + nz**2))

        elements.append(Element(
            index=i + 1,
            pin=i + 1,
            position=np.array([x, y, z]),
            orientation=np.array([az, el, 0.0]),
            size=np.array([element_size_mm, element_size_mm]),
            units="mm",
        ))

    return Transducer(
        id="hemi64",
        name=f"Hemispherical {n_elements}-element array",
        elements=elements,
        frequency=freq_hz,
        units="mm",
    )


# ---------------------------------------------------------------------------
# NIfTI loading helper
# ---------------------------------------------------------------------------
def load_nifti_as_xarray(nifti_path: Path) -> xa.DataArray:
    """Load a NIfTI file and return an xarray DataArray with mm coordinates.

    Coordinates are built from the affine diagonal (assumes axis-aligned voxels).
    Dimension order follows the data shape: (dim0, dim1, dim2) mapped to (x, y, z)
    using the affine matrix diagonal signs and magnitudes.
    """
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine

    dim_names = ("x", "y", "z")
    coords = {}
    for axis, dim in enumerate(dim_names):
        origin = float(affine[axis, 3])
        spacing = float(affine[axis, axis])
        coord_values = origin + np.arange(data.shape[axis]) * spacing
        coords[dim] = xa.Variable(dim, coord_values, attrs={"units": "mm"})

    return xa.DataArray(data, dims=dim_names, coords=coords)


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------
def extract_focal_stats(
    result: xa.Dataset,
    target_mm: np.ndarray,
    label: str,
    element_positions_mm: np.ndarray | None = None,
    exclusion_radius_mm: float = 3.0,
) -> dict:
    """Extract focal spot stats from a simulation result.

    Reports three views of the peak pressure:
      1) Raw argmax over the whole volume (may be dominated by source voxels
         in point-source simulations since each source voxel receives the
         full input signal directly).
      2) Pressure at the target voxel.
      3) Argmax after masking out a sphere of `exclusion_radius_mm` around
         each element position (the "true" far-field focus, not the source).
    """
    p_max = result["p_max"].to_numpy()
    dims = list(result["p_max"].dims)
    coord_arrays = {d: result.coords[d].to_numpy() for d in dims}

    # --- View 1: raw argmax (legacy behavior) ---
    raw_idx = np.unravel_index(p_max.argmax(), p_max.shape)
    raw_focal_mm = np.array([
        float(coord_arrays[d][raw_idx[i]]) for i, d in enumerate(dims)
    ])
    raw_error = float(np.linalg.norm(raw_focal_mm - target_mm))
    raw_max_p = float(p_max.max())

    # --- View 2: pressure at target voxel ---
    target_idx = tuple(
        int(np.argmin(np.abs(coord_arrays[dims[ax]] - target_mm[ax])))
        for ax in range(3)
    )
    p_at_target = float(p_max[target_idx])

    # --- View 3: argmax with source voxels masked out ---
    masked_focal_mm = None
    masked_error = None
    masked_max_p = None
    if element_positions_mm is not None and exclusion_radius_mm > 0:
        mg = np.meshgrid(
            *[coord_arrays[d] for d in dims], indexing="ij",
        )
        coord_stack = np.stack([m for m in mg], axis=-1)  # (Nx, Ny, Nz, 3)
        mask = np.ones(p_max.shape, dtype=bool)
        for pos in element_positions_mm:
            d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
            mask &= d2 > exclusion_radius_mm ** 2
        if mask.any():
            masked_p = np.where(mask, p_max, -np.inf)
            m_idx = np.unravel_index(masked_p.argmax(), masked_p.shape)
            masked_focal_mm = np.array([
                float(coord_arrays[d][m_idx[i]]) for i, d in enumerate(dims)
            ])
            masked_error = float(np.linalg.norm(masked_focal_mm - target_mm))
            masked_max_p = float(p_max[m_idx])

    logger.info(
        "%s: raw max_p=%.4f Pa @ (%s) mm, raw_err=%.2f mm; "
        "p@target=%.4f Pa",
        label, raw_max_p,
        ", ".join(f"{v:.1f}" for v in raw_focal_mm),
        raw_error, p_at_target,
    )
    if masked_focal_mm is not None:
        logger.info(
            "  masked max_p=%.4f Pa @ (%s) mm, masked_err=%.2f mm "
            "(excl radius %.1f mm around %d sources)",
            masked_max_p,
            ", ".join(f"{v:.1f}" for v in masked_focal_mm),
            masked_error, exclusion_radius_mm, len(element_positions_mm),
        )

    return {
        "label": label,
        "max_pressure": raw_max_p,
        "focal_mm": raw_focal_mm,
        "focal_error": raw_error,
        "p_at_target": p_at_target,
        "masked_max_pressure": masked_max_p,
        "masked_focal_mm": masked_focal_mm,
        "masked_focal_error": masked_error,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_total = time.time()

    print("=" * 72)
    print("GLADYS Simulation v7: Transcranial FUS with Timing Fixes")
    print("=" * 72)

    # -------------------------------------------------------------------
    # 1. Load and segment GU008 MRI
    # -------------------------------------------------------------------
    print(f"\n[1] Loading MRI: {MRI_PATH}")
    if not MRI_PATH.exists():
        print(f"ERROR: MRI file not found at {MRI_PATH}")
        sys.exit(1)

    volume = load_nifti_as_xarray(MRI_PATH)
    print(f"    Shape: {volume.shape}")
    print(f"    Dims:  {volume.dims}")
    for dim in volume.dims:
        c = volume.coords[dim].to_numpy()
        print(f"    {dim}: [{c[0]:.1f}, {c[-1]:.1f}] mm  "
              f"(spacing={abs(c[1]-c[0]):.2f} mm, N={len(c)})")

    print("\n    Segmenting with ThresholdMRI...")
    seg_method = ThresholdMRI(
        skull_thickness_mm=12.0,
        air_threshold_quantile=0.05,
        classify_brain_tissues=False,
        refine_skull_intensity=True,
    )
    t0 = time.time()
    seg_labels = seg_method._segment(volume)
    print(f"    Segmentation complete in {time.time()-t0:.1f}s")

    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()
    for name, idx in material_idx.items():
        count = int(np.sum(seg_arr == idx))
        pct = 100.0 * count / seg_arr.size
        print(f"    {name:>10s} (idx={idx}): {count:>10,d} voxels ({pct:.1f}%)")

    # -------------------------------------------------------------------
    # 2. Determine brain center (target) and skull top (transducer placement)
    # -------------------------------------------------------------------
    print("\n[2] Computing target and transducer placement...")

    # Brain center = centroid of tissue voxels
    tissue_mask = seg_arr == material_idx["tissue"]
    tissue_indices = np.argwhere(tissue_mask)
    if tissue_indices.size == 0:
        print("ERROR: No tissue voxels found in segmentation")
        sys.exit(1)

    # Convert voxel indices to mm coordinates
    dim_names = list(volume.dims)
    coord_arrays = {d: volume.coords[d].to_numpy() for d in dim_names}
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][tissue_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"    Brain center (target): ({target_mm[0]:.1f}, "
          f"{target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    # Skull top: find the maximum coordinate in z where skull exists.
    # The transducer will be placed above (or outside) the skull.
    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)

    # Determine which axis is "up" (the axis with the largest range of skull voxels
    # relative to the brain center). For a head scan, this is typically x or z
    # depending on the orientation. We will place the transducer along the axis
    # where skull voxels extend furthest from the brain center.
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])

    # For simplicity, use the axis with the highest skull coordinate relative
    # to the brain center. We pick the z-axis (axis index 2) as the typical
    # superior direction, but verify.
    # Actually: find which axis has skull voxels furthest from the brain center
    # on the positive side.
    max_skull_dist_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax])
        for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_dist_per_axis))
    approach_dim = dim_names[approach_axis]
    skull_top_val = float(skull_mm[approach_axis].max())
    print(f"    Approach axis: {approach_dim} (axis {approach_axis})")
    print(f"    Skull top along {approach_dim}: {skull_top_val:.1f} mm")

    # Transducer center (geometric focus at target, array surface near skull)
    # The geometric focus of the hemispherical array is at the center of curvature.
    # Place it so the focus coincides with the brain center target.
    # The array surface (at radius R from focus) should be near the skull surface.
    # Offset from target toward skull by RADIUS_MM along the approach axis.
    xdc_center_mm = target_mm.copy()
    xdc_center_mm[approach_axis] = target_mm[approach_axis] + RADIUS_MM

    print(f"    Transducer center: ({xdc_center_mm[0]:.1f}, "
          f"{xdc_center_mm[1]:.1f}, {xdc_center_mm[2]:.1f}) mm")
    print(f"    Distance from focus to xdc center: {RADIUS_MM:.1f} mm")

    # -------------------------------------------------------------------
    # 3. Create hemispherical array and bake positions into world coords
    # -------------------------------------------------------------------
    print(f"\n[3] Creating {N_ELEMENTS}-element hemispherical array "
          f"(R={RADIUS_MM}mm, aperture={APERTURE_MM}mm, f={FREQ_HZ/1e3:.0f}kHz)")

    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS,
        radius_mm=RADIUS_MM,
        aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ,
        element_size_mm=ELEMENT_SIZE_MM,
    )

    # Build a transform that:
    #   1) Rotates the array so the aperture faces the approach direction
    #   2) Translates so the geometric focus lands on the target
    #
    # In the local frame, focus is at origin and the array cap is at +z.
    # We need the cap to face the approach axis direction from the target.
    # If approach_axis=2 (z), the cap is at +z from target which is correct.
    # If approach_axis=0 (x) or 1 (y), we need to rotate.
    transform = np.eye(4)
    transform[:3, 3] = target_mm  # translate focus to target

    if approach_axis == 0:
        # Rotate so local +z maps to world +x: Ry(+90 deg)
        angle = np.pi / 2
        transform[:3, :3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        transform[:3, 3] = target_mm
    elif approach_axis == 1:
        # Rotate so local +z maps to world +y: Rx(-90 deg)
        angle = -np.pi / 2
        transform[:3, :3] = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
        transform[:3, 3] = target_mm
    # axis == 2: identity rotation, local +z = world +z, no rotation needed

    # Bake the transform into element positions (pre-transform the array).
    # This means the array is in world coordinates and no transform is needed later.
    arr = deepcopy(arr_local)
    for el in arr.elements:
        world_pos = el.get_position(units="mm", matrix=transform)
        el.position = world_pos
        # Recompute orientation: element should face from its position toward the target
        direction = target_mm - world_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            n = direction / dist
            az = np.arctan2(n[0], n[2])
            el_angle = -np.arctan2(n[1], np.sqrt(n[0]**2 + n[2]**2))
            el.orientation = np.array([az, el_angle, 0.0])

    positions = arr.get_positions(units="mm")
    print(f"    Element positions range:")
    for ax, dim in enumerate(["x", "y", "z"]):
        print(f"      {dim}: [{positions[:, ax].min():.1f}, {positions[:, ax].max():.1f}] mm")
    print(f"    Mean element position: ({positions[:, 0].mean():.1f}, "
          f"{positions[:, 1].mean():.1f}, {positions[:, 2].mean():.1f}) mm")

    # Verify geometric focus
    dists_to_target = np.linalg.norm(positions - target_mm[np.newaxis, :], axis=1)
    print(f"    Distance from elements to target: "
          f"mean={dists_to_target.mean():.1f}mm, "
          f"std={dists_to_target.std():.2f}mm, "
          f"range=[{dists_to_target.min():.1f}, {dists_to_target.max():.1f}]mm")

    # -------------------------------------------------------------------
    # 4. Build 0.5mm simulation grid encompassing transducer + target
    # -------------------------------------------------------------------
    print(f"\n[4] Building simulation grid (spacing={GRID_SPACING_MM}mm)...")

    # Grid extents: include all element positions and target with margin
    all_points = np.vstack([positions, target_mm[np.newaxis, :]])
    grid_min = all_points.min(axis=0) - GRID_MARGIN_MM
    grid_max = all_points.max(axis=0) + GRID_MARGIN_MM

    # Snap to grid spacing
    grid_min = np.floor(grid_min / GRID_SPACING_MM) * GRID_SPACING_MM
    grid_max = np.ceil(grid_max / GRID_SPACING_MM) * GRID_SPACING_MM

    print(f"    Grid extents:")
    for ax, dim in enumerate(["x", "y", "z"]):
        n_pts = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        print(f"      {dim}: [{grid_min[ax]:.1f}, {grid_max[ax]:.1f}] mm ({n_pts} points)")

    # Build simulation coordinates
    sim_coords = {}
    for ax, dim in enumerate(["x", "y", "z"]):
        n_pts = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        sim_coords[dim] = xa.Variable(
            dim,
            np.linspace(grid_min[ax], grid_max[ax], n_pts),
            attrs={"units": "mm"},
        )

    grid_shape = tuple(len(sim_coords[d]) for d in ["x", "y", "z"])
    grid_voxels = int(np.prod(grid_shape))
    print(f"    Grid shape: {grid_shape} = {grid_voxels:,d} voxels")
    print(f"    Grid memory (float32): {grid_voxels * 4 / 1e9:.2f} GB per field")

    # Resample volume to sim grid and segment to get acoustic params
    print("\n    Resampling MRI to simulation grid and computing acoustic params...")
    from scipy.interpolate import RegularGridInterpolator

    # Build interpolator from original volume
    orig_coords = [volume.coords[d].to_numpy() for d in volume.dims]
    interp = RegularGridInterpolator(
        orig_coords, volume.to_numpy(),
        method="linear", bounds_error=False, fill_value=0.0,
    )

    # Evaluate on sim grid
    sim_coord_arrays = [sim_coords[d].data for d in ["x", "y", "z"]]
    mg = np.meshgrid(*sim_coord_arrays, indexing="ij")
    query_pts = np.stack([m.ravel() for m in mg], axis=-1)
    resampled_data = interp(query_pts).reshape(grid_shape).astype(np.float32)

    sim_volume = xa.DataArray(
        resampled_data,
        dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )
    print(f"    Resampled volume shape: {sim_volume.shape}")

    # Segment the resampled volume
    t0 = time.time()
    sim_params = seg_method.seg_params(sim_volume)
    print(f"    Segmentation + param mapping done in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------
    # 5. Replace air voxels with water in the medium
    # -------------------------------------------------------------------
    print("\n[5] Replacing air voxels with water in medium...")

    # Get the segmentation labels on the sim grid to find air voxels
    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask = sim_seg_arr == material_idx["air"]
    water_mat = seg_method.materials["water"]

    n_air = int(air_mask.sum())
    pct_air = 100.0 * n_air / sim_seg_arr.size
    print(f"    Air voxels: {n_air:,d} ({pct_air:.2f}%)")

    if n_air > 0:
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation
        print(f"    Replaced with water: c={water_mat.sound_speed} m/s, "
              f"rho={water_mat.density} kg/m3, alpha={water_mat.attenuation} dB/cm/MHz")
    else:
        print("    No air voxels found (nothing to replace)")

    # Print medium diagnostics
    print("\n    Medium diagnostics:")
    for var in ["sound_speed", "density", "attenuation"]:
        arr_data = sim_params[var].to_numpy()
        ref_val = sim_params[var].attrs["ref_value"]
        print(f"      {var}: ref={ref_val}, min={arr_data.min():.2f}, "
              f"max={arr_data.max():.2f}, mean={arr_data.mean():.2f}")

    # -------------------------------------------------------------------
    # 6. Create target point
    # -------------------------------------------------------------------
    target = Point(
        position=target_mm.copy(),
        id="brain_center",
        name="Brain Center Target",
        units="mm",
    )

    # -------------------------------------------------------------------
    # 7. Compute geometric delays
    # -------------------------------------------------------------------
    print("\n[6] Computing geometric delays...")
    direct = Direct(c0=C0)
    delays_geo = direct.calc_delays(arr, target, sim_params)
    print(f"    Delay range: {delays_geo.min()*1e6:.1f} to {delays_geo.max()*1e6:.1f} us")
    print(f"    Delay spread: {(delays_geo.max()-delays_geo.min())*1e6:.1f} us")

    # -------------------------------------------------------------------
    # 8. Calculate explicit t_end
    # -------------------------------------------------------------------
    print("\n[7] Calculating explicit t_end...")

    # Max distance from any element to any corner of the sim grid
    from openlifu.sim.sim_setup import SimSetup
    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM,
        units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0,
        cfl=CFL,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    max_dist_m = max_dist_mm * 1e-3

    # t_end = max_distance / c0 * safety_factor
    t_end = max_dist_m / C0 * T_END_SAFETY
    print(f"    Max element-to-corner distance: {max_dist_mm:.1f} mm")
    print(f"    t_end = {max_dist_m:.4f}m / {C0:.0f} m/s * {T_END_SAFETY} = {t_end*1e6:.1f} us")

    # Compute dt from CFL condition: dt = CFL * dx / c_max
    # c_max in skull is 4080 m/s
    c_max = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max
    n_timesteps = int(np.ceil(t_end / dt))
    print(f"    c_max in medium: {c_max:.0f} m/s")
    print(f"    dt = CFL * dx / c_max = {CFL} * {dx_m*1e6:.1f}um / {c_max:.0f} = {dt*1e9:.2f} ns")
    print(f"    N timesteps: {n_timesteps:,d}")

    # -------------------------------------------------------------------
    # 9. Compute phase-corrected delays via SimulationCorrected
    # -------------------------------------------------------------------
    print("\n[8] Computing phase-corrected delays (SimulationCorrected)...")
    print(f"    This runs a reciprocal k-wave simulation (point source at target).")
    print(f"    CFL={CFL}, n_cycles=3, gpu=True")

    sim_corrected = SimulationCorrected(
        c0=C0,
        cfl=CFL,
        n_cycles=3,
        gpu=True,
    )

    t0 = time.time()
    delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)
    t_correction = time.time() - t0
    print(f"    Phase correction complete in {t_correction:.1f}s")
    print(f"    Corrected delay range: {delays_corrected.min()*1e6:.1f} to "
          f"{delays_corrected.max()*1e6:.1f} us")
    print(f"    Corrected delay spread: "
          f"{(delays_corrected.max()-delays_corrected.min())*1e6:.1f} us")

    # Compare geometric vs corrected delays
    delay_diff = delays_corrected - delays_geo
    print(f"    Delay difference (corrected - geometric):")
    print(f"      mean={delay_diff.mean()*1e6:.2f} us, "
          f"std={delay_diff.std()*1e6:.2f} us, "
          f"max_abs={np.abs(delay_diff).max()*1e6:.2f} us")

    apod = np.ones(arr.numelements())

    # -------------------------------------------------------------------
    # 10. Shared simulation kwargs
    # -------------------------------------------------------------------
    common_kwargs = dict(
        arr=arr,
        apod=apod,
        freq=FREQ_HZ,
        cycles=CYCLES,
        amplitude=AMPLITUDE,
        dt=dt,
        t_end=t_end,
        cfl=CFL,
        gpu=True,
        source_method="point_source",
    )

    # -------------------------------------------------------------------
    # 11. Simulation A: Phase-corrected + heterogeneous skull
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[SIM A] Phase-corrected delays + heterogeneous skull medium")
    print("=" * 72)
    t0 = time.time()
    result_a = run_simulation(
        params=sim_params,
        delays=delays_corrected,
        ref_values_only=False,
        **common_kwargs,
    )
    t_sim_a = time.time() - t0
    print(f"    Completed in {t_sim_a:.1f}s")
    stats_a = extract_focal_stats(result_a, target_mm, "SIM A (corrected+hetero)", element_positions_mm=positions)

    # -------------------------------------------------------------------
    # 12. Simulation B: Geometric delays + heterogeneous skull (no correction)
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[SIM B] Geometric delays + heterogeneous skull medium (no correction)")
    print("=" * 72)
    t0 = time.time()
    result_b = run_simulation(
        params=sim_params,
        delays=delays_geo,
        ref_values_only=False,
        **common_kwargs,
    )
    t_sim_b = time.time() - t0
    print(f"    Completed in {t_sim_b:.1f}s")
    stats_b = extract_focal_stats(result_b, target_mm, "SIM B (geometric+hetero)", element_positions_mm=positions)

    # -------------------------------------------------------------------
    # 13. Simulation C: Homogeneous water reference
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[SIM C] Geometric delays + homogeneous water (ref_values_only=True)")
    print("=" * 72)
    t0 = time.time()
    result_c = run_simulation(
        params=sim_params,
        delays=delays_geo,
        ref_values_only=True,
        **common_kwargs,
    )
    t_sim_c = time.time() - t0
    print(f"    Completed in {t_sim_c:.1f}s")
    stats_c = extract_focal_stats(result_c, target_mm, "SIM C (geometric+water)", element_positions_mm=positions)

    # -------------------------------------------------------------------
    # 14. Comparison table
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COMPARISON RESULTS")
    print("=" * 72)

    header = (
        f"{'Simulation':<48} {'Max P (Pa)':>12} "
        f"{'Focal Err (mm)':>16} {'P Recovery':>12}"
    )
    print(header)
    print("-" * len(header))

    p_water = stats_c["max_pressure"]
    for label, stats in [
        ("A: Corrected delays + skull", stats_a),
        ("B: Geometric delays + skull", stats_b),
        ("C: Geometric delays + water (reference)", stats_c),
    ]:
        recovery = stats["max_pressure"] / p_water if p_water > 0 else 0.0
        print(
            f"{label:<48} {stats['max_pressure']:>12.4f} "
            f"{stats['focal_error']:>16.2f} {recovery:>12.1%}"
        )

    print()
    print("Pressure recovery = max_pressure / water_reference_max_pressure")
    print("  100% = no skull effect (ideal)")
    print("  >100% = phase correction partially recovered skull losses")
    print()

    # Correction benefit
    if stats_b["max_pressure"] > 0:
        correction_gain = stats_a["max_pressure"] / stats_b["max_pressure"]
        print(f"Phase correction benefit:")
        print(f"  Pressure gain (A/B): {correction_gain:.2f}x "
              f"({(correction_gain-1)*100:.1f}% improvement)")
        print(f"  Focal error reduction: "
              f"{stats_b['focal_error']:.2f}mm -> {stats_a['focal_error']:.2f}mm")
    else:
        print("WARNING: Uncorrected simulation produced zero pressure")

    # -------------------------------------------------------------------
    # 15. Save results
    # -------------------------------------------------------------------
    results_dir = Path.home() / "Data/openlifu-validation/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for sim_label, result in [("corrected", result_a), ("geometric", result_b), ("water", result_c)]:
        p_max_data = result["p_max"].to_numpy()
        # Build a NIfTI affine from the simulation grid coordinates
        x_coords = result.coords["x"].to_numpy()
        y_coords = result.coords["y"].to_numpy()
        z_coords = result.coords["z"].to_numpy()
        out_affine = np.diag([
            float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0,
            float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0,
            float(z_coords[1] - z_coords[0]) if len(z_coords) > 1 else 1.0,
            1.0,
        ])
        out_affine[0, 3] = float(x_coords[0])
        out_affine[1, 3] = float(y_coords[0])
        out_affine[2, 3] = float(z_coords[0])

        out_path = results_dir / f"gladys_v7_{sim_label}_pmax.nii.gz"
        nib.save(nib.Nifti1Image(p_max_data.astype(np.float32), out_affine), str(out_path))
        print(f"    Saved: {out_path}")

    t_elapsed = time.time() - t_total
    print(f"\nTotal elapsed time: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
