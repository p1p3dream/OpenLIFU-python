#!/usr/bin/env python3
"""Diagnose reverberation in SimulationCorrected reciprocal simulation.

Runs a reciprocal sim (point source at target, sensors at element positions)
through the skull for GU002, then analyzes the Hilbert envelope of each
element's time series to quantify how much stronger late reverberations are
compared to the direct arrival.

Usage (on stonkbot):
    PYTHONPATH=~/OpenLIFU-python/src ~/openlifu-env/bin/python3 diagnose_reverberation.py
"""
from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import sys
import time

# kwave v3 logging bug workaround
_orig_log = logging.log
def _patched_log(level, msg, *args, **kwargs):
    try:
        if args and isinstance(msg, str) and "%" not in msg:
            msg = msg + " " + " ".join(str(a) for a in args)
            args = ()
    except Exception:
        pass
    return _orig_log(level, msg, *args, **kwargs)
logging.log = _patched_log

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xa
from scipy.ndimage import map_coordinates
from scipy.signal import hilbert, find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.geo import Point
from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.seg_methods.nnunet_seg import LABEL_MAP_FULLHEAD
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER
from openlifu.sim.kwave_if import run_point_source_simulation
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants (matching run_gladys_nnunet_subject.py)
N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
GRID_SPACING_MM = 0.5
CFL = 0.3
N_CYCLES = 3
C0 = 1500.0
GRID_MARGIN_MM = 10.0


def _default_fullhead_materials() -> dict[str, Material]:
    m = MATERIALS.copy()
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


@dataclass
class PreSegmented(SegmentationMethod):
    label_nifti_path: str = ""
    nnunet_label_map: dict[int, str] = field(default_factory=lambda: dict(LABEL_MAP_FULLHEAD))
    materials: dict[str, Material] = field(default_factory=_default_fullhead_materials)

    def __post_init__(self):
        super().__post_init__()
        if not self.label_nifti_path:
            raise ValueError("label_nifti_path is required")
        img = nib.load(self.label_nifti_path)
        data = np.asarray(img.dataobj).astype(np.int16)
        affine = img.affine
        dim_names = ("x", "y", "z")
        coords = {}
        for axis, dim in enumerate(dim_names):
            origin = float(affine[axis, 3])
            spacing = float(affine[axis, axis])
            coord_values = origin + np.arange(data.shape[axis]) * spacing
            coords[dim] = xa.Variable(dim, coord_values, attrs={"units": "mm"})
        self._labels = xa.DataArray(data, dims=dim_names, coords=coords)

    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        mat_idx = self._material_indices()
        src = self._labels
        src_dims = list(src.dims)
        src_origin = {d: float(src.coords[d].to_numpy()[0]) for d in src_dims}
        src_spacing = {}
        for d in src_dims:
            cs = src.coords[d].to_numpy()
            src_spacing[d] = float(cs[1] - cs[0]) if len(cs) > 1 else 1.0
        tgt_dims = list(volume.dims)
        tgt_coord_arrays = [volume.coords[d].to_numpy() for d in src_dims]
        mg = np.meshgrid(*tgt_coord_arrays, indexing="ij")
        frac_idx = []
        for i, d in enumerate(src_dims):
            fi = (mg[i] - src_origin[d]) / src_spacing[d]
            frac_idx.append(fi)
        frac_stack = np.stack(frac_idx, axis=0)
        resampled = map_coordinates(
            src.to_numpy().astype(np.float32), frac_stack, order=0,
            mode="constant", cval=0.0,
        ).astype(np.int16)
        water_idx = mat_idx["water"]
        output = np.full(resampled.shape, water_idx, dtype=int)
        for nn_label, mat_key in self.nnunet_label_map.items():
            output[resampled == nn_label] = mat_idx[mat_key]
        labels_da = xa.DataArray(
            output, dims=src_dims,
            coords={d: volume.coords[d] for d in src_dims},
        )
        return labels_da.transpose(*tgt_dims)

    def to_table(self):
        import pandas as pd
        return pd.DataFrame()


def create_hemispherical_array(
    n_elements=64, radius_mm=90.0, aperture_mm=80.0,
    freq_hz=500e3, element_size_mm=5.0,
) -> Transducer:
    half_aperture = aperture_mm / 2.0
    theta_max = np.arcsin(half_aperture / radius_mm)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    elements = []
    for i in range(n_elements):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        x = radius_mm * np.sin(theta) * np.cos(phi)
        y = radius_mm * np.sin(theta) * np.sin(phi)
        z = radius_mm * np.cos(theta)
        nx, ny, nz = -x, -y, -z
        az = np.arctan2(nx, nz)
        el = -np.arctan2(ny, np.sqrt(nx**2 + nz**2))
        elements.append(Element(
            index=i + 1, pin=i + 1,
            position=np.array([x, y, z]),
            orientation=np.array([az, el, 0.0]),
            size=np.array([element_size_mm, element_size_mm]),
            units="mm",
        ))
    return Transducer(
        id="hemi64", name=f"Hemispherical {n_elements}-element array",
        elements=elements, frequency=freq_hz, units="mm",
    )


def load_nifti_as_xarray(nifti_path: Path) -> xa.DataArray:
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


def main():
    subj = "GU002"
    mri_path = Path.home() / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/Anonymized_Subjects/T1-Weighted MRI" / f"{subj}_deface.nii"
    label_path = Path.home() / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"

    print("=" * 72)
    print(f"REVERBERATION DIAGNOSTIC | subject={subj}")
    print(f"  MRI:    {mri_path}")
    print(f"  Labels: {label_path}")
    print("=" * 72)

    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}")
        sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}")
        sys.exit(1)

    # --- Load volume and segment ---
    volume = load_nifti_as_xarray(mri_path)
    seg_method = PreSegmented(label_nifti_path=str(label_path))
    print(f"MRI shape: {volume.shape}, label shape: {seg_method._labels.shape}")

    # Find brain center target
    seg_labels = seg_method._segment(volume)
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()
    coord_arrays = {d: volume.coords[d].to_numpy() for d in volume.dims}
    dim_names = list(volume.dims)

    brain_mask = np.zeros(seg_arr.shape, dtype=bool)
    for k in ("csf", "gray_matter", "white_matter"):
        if k in material_idx:
            brain_mask |= seg_arr == material_idx[k]
    if brain_mask.sum() == 0:
        brain_mask = seg_arr == material_idx["tissue"]
    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"Target (brain center): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    # Approach axis (auto-detect from skull extent)
    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])
    max_skull_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_per_axis))
    approach_dir = np.zeros(3)
    approach_dir[approach_axis] = 1.0
    print(f"Approach axis: {dim_names[approach_axis]} (axis {approach_axis})")

    # --- Create and position transducer ---
    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )

    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, approach_dir)
    c = np.dot(z_axis, approach_dir)
    if np.linalg.norm(v) < 1e-10:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = target_mm
    arr = deepcopy(arr_local)
    for el in arr.elements:
        world_pos = el.get_position(units="mm", matrix=transform)
        el.position = world_pos
        direction = target_mm - world_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            n = direction / dist
            az = np.arctan2(n[0], n[2])
            el_angle = -np.arctan2(n[1], np.sqrt(n[0]**2 + n[2]**2))
            el.orientation = np.array([az, el_angle, 0.0])
    positions = arr.get_positions(units="mm")

    # --- Build sim grid ---
    all_points = np.vstack([positions, target_mm[np.newaxis, :]])
    grid_min = np.floor((all_points.min(axis=0) - GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    grid_max = np.ceil((all_points.max(axis=0) + GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    sim_coords = {}
    for ax, dim in enumerate(["x", "y", "z"]):
        n_pts = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        sim_coords[dim] = xa.Variable(
            dim, np.linspace(grid_min[ax], grid_max[ax], n_pts),
            attrs={"units": "mm"},
        )
    grid_shape = tuple(len(sim_coords[d]) for d in ["x", "y", "z"])
    print(f"Sim grid: {grid_shape}")

    from scipy.interpolate import RegularGridInterpolator
    orig_coords_list = [volume.coords[d].to_numpy() for d in volume.dims]
    interp = RegularGridInterpolator(
        orig_coords_list, volume.to_numpy(),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    sim_coord_arrays = [sim_coords[d].data for d in ["x", "y", "z"]]
    mg = np.meshgrid(*sim_coord_arrays, indexing="ij")
    query_pts = np.stack([m.ravel() for m in mg], axis=-1)
    resampled_data = interp(query_pts).reshape(grid_shape).astype(np.float32)
    sim_volume = xa.DataArray(
        resampled_data, dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )

    sim_params = seg_method.seg_params(sim_volume)

    # Replace air with water (same as production script)
    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask = sim_seg_arr == material_idx["air"]
    water_mat = seg_method.materials["water"]
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation

    # --- Build reciprocal simulation masks ---
    coord_dims = list(sim_params.coords.dims)
    coord_units = sim_params[coord_dims[0]].attrs.get("units", "mm")
    _DIM_IDX = {"x": 0, "y": 1, "z": 2}

    scl_m_to_coord = getunitconversion("m", coord_units)
    scl_to_m = getunitconversion(coord_units, "m")

    # Element positions in grid units (mm)
    element_positions = positions.copy()  # already in mm

    # Target position
    target_pos = target_mm.copy()

    coord_arrays_grid = [sim_params.coords[dim].to_numpy() for dim in coord_dims]
    grid_shape_sim = tuple(len(c) for c in coord_arrays_grid)

    # Build sensor mask (element positions)
    sensor_indices = []
    for el_i, epos_xyz in enumerate(element_positions):
        idx = []
        for dim_i, dim_name in enumerate(coord_dims):
            coord_vals = coord_arrays_grid[dim_i]
            pos_component = epos_xyz[_DIM_IDX[dim_name]]
            nearest_idx = int(np.argmin(np.abs(coord_vals - pos_component)))
            idx.append(nearest_idx)
        sensor_indices.append(tuple(idx))

    sensor_mask = np.zeros(grid_shape_sim, dtype=int)
    for idx in sensor_indices:
        sensor_mask[idx] = 1

    # Build source mask (target)
    target_idx = []
    for dim_i, dim_name in enumerate(coord_dims):
        coord_vals = coord_arrays_grid[dim_i]
        pos_component = target_pos[_DIM_IDX[dim_name]]
        nearest_idx = int(np.argmin(np.abs(coord_vals - pos_component)))
        target_idx.append(nearest_idx)
    target_idx = tuple(target_idx)

    source_mask = np.zeros(grid_shape_sim, dtype=int)
    source_mask[target_idx] = 1

    # Compute distances and t_end
    dists_m = np.linalg.norm(element_positions - target_pos, axis=1) * scl_to_m
    max_dist_m = float(np.max(dists_m))
    sound_speed_ref = C0
    sound_speed_max = float(np.max(sim_params["sound_speed"].to_numpy()))
    sound_speed_max = max(sound_speed_max, sound_speed_ref)

    t_end_needed = max_dist_m / sound_speed_ref * 1.5 + N_CYCLES / FREQ_HZ

    print(f"\nMax distance: {max_dist_m*1e3:.1f} mm")
    print(f"Sound speed ref: {sound_speed_ref} m/s, max: {sound_speed_max:.0f} m/s")
    print(f"t_end: {t_end_needed*1e6:.1f} us")
    print(f"N sensor voxels: {int(sensor_mask.sum())}")
    print(f"N source voxels: {int(source_mask.sum())}")

    # --- Run reciprocal simulation ---
    print("\nRunning reciprocal simulation (GPU)...")
    t_start = time.time()
    sensor_data, dt = run_point_source_simulation(
        params=sim_params,
        source_mask=source_mask,
        sensor_mask=sensor_mask,
        freq=FREQ_HZ,
        n_cycles=N_CYCLES,
        sound_speed_ref=sound_speed_ref,
        cfl=CFL,
        gpu=True,
        t_end=t_end_needed,
    )
    t_sim = time.time() - t_start
    print(f"Reciprocal sim complete in {t_sim:.1f}s")
    print(f"sensor_data shape: {sensor_data.shape}, dt: {dt*1e9:.1f} ns")

    # --- Map sensor data columns back to elements ---
    perm_to_xyz = [coord_dims.index(d) for d in ["x", "y", "z"]]
    sensor_mask_xyz = np.transpose(sensor_mask, perm_to_xyz)
    grid_shape_xyz = sensor_mask_xyz.shape

    nonzero_xyz = list(zip(*np.nonzero(sensor_mask_xyz)))

    def fortran_linear_index(idx, shape):
        lin = idx[0]
        stride = shape[0]
        for d in range(1, len(shape)):
            lin += idx[d] * stride
            stride *= shape[d]
        return lin

    nonzero_with_fortran = [(fortran_linear_index(idx, grid_shape_xyz), idx) for idx in nonzero_xyz]
    nonzero_with_fortran.sort(key=lambda x: x[0])
    sorted_nonzero = [item[1] for item in nonzero_with_fortran]
    voxel_to_col = {idx: col for col, idx in enumerate(sorted_nonzero)}

    # --- Analyze each element ---
    print("\n" + "=" * 100)
    print(f"{'El':>3} {'dist_mm':>8} {'geo_tof':>10} {'earliest':>10} "
          f"{'1st_peak':>10} {'max_peak':>10} {'delay_us':>10} "
          f"{'amp_ratio':>10} {'n_peaks':>8} {'match':>6}")
    print("-" * 100)

    results = []
    for el_i, sensor_idx in enumerate(sensor_indices):
        sensor_idx_xyz = tuple(sensor_idx[i] for i in perm_to_xyz)
        col = voxel_to_col[sensor_idx_xyz]
        time_series = sensor_data[:, col]

        # Hilbert envelope
        analytic = hilbert(time_series)
        envelope = np.abs(analytic)

        # Distances and times
        dist_mm = float(np.linalg.norm(element_positions[el_i] - target_pos))
        dist_m = dist_mm * scl_to_m
        geo_tof_s = dist_m / sound_speed_ref
        earliest_arrival_s = dist_m / sound_speed_max

        # Gate: earliest plausible arrival
        gate_start = max(0, int((earliest_arrival_s - 2 * dt) / dt))
        if gate_start >= len(envelope):
            gate_start = 0

        gated_envelope = envelope[gate_start:]
        time_axis = np.arange(len(envelope)) * dt

        # Find ALL peaks in gated envelope
        # Use prominence threshold = 5% of max gated envelope
        max_gated = float(np.max(gated_envelope))
        prominence_thresh = 0.05 * max_gated if max_gated > 0 else 0

        peak_indices_gated, peak_props = find_peaks(
            gated_envelope,
            prominence=prominence_thresh,
            distance=int(0.5 / (FREQ_HZ * dt)),  # at least half a period apart
        )

        if len(peak_indices_gated) == 0:
            # Fallback: just use argmax
            peak_indices_gated = np.array([int(np.argmax(gated_envelope))])
            peak_props = {"prominences": np.array([max_gated])}

        # Convert gated indices to absolute
        peak_indices_abs = peak_indices_gated + gate_start
        peak_times_s = peak_indices_abs * dt
        peak_amplitudes = envelope[peak_indices_abs]

        # First significant peak (earliest in time)
        first_peak_time = float(peak_times_s[0])
        first_peak_amp = float(peak_amplitudes[0])

        # Strongest peak (highest amplitude)
        strongest_idx = int(np.argmax(peak_amplitudes))
        strongest_peak_time = float(peak_times_s[strongest_idx])
        strongest_peak_amp = float(peak_amplitudes[strongest_idx])

        # Argmax of gated envelope (what SimulationCorrected currently does)
        argmax_peak_sample = gate_start + int(np.argmax(gated_envelope))
        argmax_peak_time = argmax_peak_sample * dt

        amp_ratio = strongest_peak_amp / first_peak_amp if first_peak_amp > 0 else float("inf")
        time_diff_us = (strongest_peak_time - first_peak_time) * 1e6
        match = "YES" if strongest_idx == 0 else "NO"

        results.append({
            "element": el_i,
            "dist_mm": dist_mm,
            "geo_tof_us": geo_tof_s * 1e6,
            "earliest_us": earliest_arrival_s * 1e6,
            "first_peak_us": first_peak_time * 1e6,
            "strongest_peak_us": strongest_peak_time * 1e6,
            "argmax_peak_us": argmax_peak_time * 1e6,
            "time_diff_us": time_diff_us,
            "amp_ratio": amp_ratio,
            "first_peak_amp": first_peak_amp,
            "strongest_peak_amp": strongest_peak_amp,
            "n_peaks": len(peak_indices_gated),
            "match": strongest_idx == 0,
        })

        print(f"{el_i:>3} {dist_mm:>8.1f} {geo_tof_s*1e6:>10.2f} {earliest_arrival_s*1e6:>10.2f} "
              f"{first_peak_time*1e6:>10.2f} {strongest_peak_time*1e6:>10.2f} {time_diff_us:>10.2f} "
              f"{amp_ratio:>10.2f} {len(peak_indices_gated):>8} {match:>6}")

    # --- Summary statistics ---
    print("\n" + "=" * 72)
    print("SUMMARY STATISTICS")
    print("=" * 72)

    n_mismatch = sum(1 for r in results if not r["match"])
    n_total = len(results)
    print(f"\nElements where strongest peak != first peak: {n_mismatch}/{n_total} ({100*n_mismatch/n_total:.1f}%)")

    ratios = [r["amp_ratio"] for r in results]
    diffs = [r["time_diff_us"] for r in results]
    first_delays = [r["first_peak_us"] - r["geo_tof_us"] for r in results]
    argmax_delays = [r["argmax_peak_us"] - r["geo_tof_us"] for r in results]

    print(f"\nAmplitude ratio (strongest/first peak):")
    print(f"  Mean:   {np.mean(ratios):.2f}")
    print(f"  Median: {np.median(ratios):.2f}")
    print(f"  Max:    {np.max(ratios):.2f}")
    print(f"  Min:    {np.min(ratios):.2f}")

    print(f"\nTime difference (strongest - first peak), us:")
    print(f"  Mean:   {np.mean(diffs):.2f}")
    print(f"  Median: {np.median(diffs):.2f}")
    print(f"  Max:    {np.max(diffs):.2f}")

    print(f"\nFirst-peak delay vs geometric TOF, us:")
    print(f"  Mean:   {np.mean(first_delays):.2f}")
    print(f"  Median: {np.median(first_delays):.2f}")
    print(f"  Std:    {np.std(first_delays):.2f}")

    print(f"\nArgmax-peak delay vs geometric TOF, us (what SimulationCorrected picks):")
    print(f"  Mean:   {np.mean(argmax_delays):.2f}")
    print(f"  Median: {np.median(argmax_delays):.2f}")
    print(f"  Std:    {np.std(argmax_delays):.2f}")

    # Spread analysis (key metric for focusing quality)
    first_peak_times = [r["first_peak_us"] for r in results]
    argmax_peak_times = [r["argmax_peak_us"] for r in results]
    geo_tofs = [r["geo_tof_us"] for r in results]

    print(f"\nArrival time spread (max - min across elements):")
    print(f"  Using first peak:  {np.max(first_peak_times) - np.min(first_peak_times):.2f} us")
    print(f"  Using argmax peak: {np.max(argmax_peak_times) - np.min(argmax_peak_times):.2f} us")
    print(f"  Geometric TOF:     {np.max(geo_tofs) - np.min(geo_tofs):.2f} us")

    # Number of peaks distribution
    n_peaks_list = [r["n_peaks"] for r in results]
    print(f"\nPeaks per element:")
    print(f"  Mean:   {np.mean(n_peaks_list):.1f}")
    print(f"  Median: {np.median(n_peaks_list):.0f}")
    print(f"  Range:  {np.min(n_peaks_list)} - {np.max(n_peaks_list)}")

    print(f"\nDone. Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
