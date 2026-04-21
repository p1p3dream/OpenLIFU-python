"""Unit tests for scripts/_probe_helpers.py.

The helpers live under scripts/ (not an importable package), so we
load the module by absolute path. This mirrors how production scripts
are expected to pick it up.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_HELPERS_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts"
    / "_probe_helpers.py"
)

_spec = importlib.util.spec_from_file_location("_probe_helpers", _HELPERS_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"Could not build import spec for {_HELPERS_PATH}"
)
_probe_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe_helpers)

per_voxel_water_calibrated_gate_center = (
    _probe_helpers.per_voxel_water_calibrated_gate_center
)
extract_focal_window_peak = _probe_helpers.extract_focal_window_peak
compute_spatial_search_peaks = _probe_helpers.compute_spatial_search_peaks


# -------------------------------------------------------------------
# per_voxel_water_calibrated_gate_center
# -------------------------------------------------------------------

def test_target_voxel_gate_is_water_peak():
    """When sensor == target, the gate center equals the water-peak
    time exactly (no TOF correction to apply)."""
    target = np.array([10.0, -5.0, 120.0])
    aperture = np.array([0.0, 0.0, 0.0])
    t_water = 8.3456e-5  # seconds

    gate = per_voxel_water_calibrated_gate_center(
        sensor_world_mm=target.copy(),
        target_world_mm=target,
        aperture_center_world_mm=aperture,
        t_water_peak_target=t_water,
    )
    assert gate == pytest.approx(t_water, rel=0.0, abs=0.0)


def test_off_axis_voxel_gets_tof_correction():
    """A sensor farther from the aperture than the target should have
    a gate center strictly greater than t_water_peak_target, with the
    extra delay equal to (d_sensor - d_target) / c0."""
    aperture = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 100.0])            # 100 mm from aperture
    sensor = np.array([0.0, 0.0, 115.0])            # 115 mm from aperture
    t_water = 6.6667e-5  # ~target / c0, but doesn't have to be exact
    c0 = 1500.0

    gate = per_voxel_water_calibrated_gate_center(
        sensor_world_mm=sensor,
        target_world_mm=target,
        aperture_center_world_mm=aperture,
        t_water_peak_target=t_water,
        c0_mps=c0,
    )

    expected_delta = (0.115 - 0.100) / c0  # seconds
    assert gate > t_water
    assert gate - t_water == pytest.approx(expected_delta, rel=1e-12)

    # And symmetry: a sensor closer to the aperture gets a smaller gate.
    sensor_near = np.array([0.0, 0.0, 85.0])
    gate_near = per_voxel_water_calibrated_gate_center(
        sensor_world_mm=sensor_near,
        target_world_mm=target,
        aperture_center_world_mm=aperture,
        t_water_peak_target=t_water,
        c0_mps=c0,
    )
    assert gate_near < t_water
    assert t_water - gate_near == pytest.approx(
        (0.100 - 0.085) / c0, rel=1e-12
    )


# -------------------------------------------------------------------
# Helpers for the spatial-search tests
# -------------------------------------------------------------------

def _make_cube(center_mm: np.ndarray, step_mm: float, n_per_side: int = 3):
    """Return (Nv, 3) world positions of an n^3 cube centered on
    ``center_mm`` with voxel spacing ``step_mm``."""
    half = (n_per_side - 1) / 2.0
    offsets = (np.arange(n_per_side) - half) * step_mm
    xs, ys, zs = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    cube = np.stack(
        [
            xs.reshape(-1) + center_mm[0],
            ys.reshape(-1) + center_mm[1],
            zs.reshape(-1) + center_mm[2],
        ],
        axis=1,
    )
    return cube


def _pulse_at_sample(n_samples: int, sample_idx: int, amplitude: float = 1.0):
    """1-D time series with a single nonzero sample at ``sample_idx``."""
    ts = np.zeros(n_samples, dtype=float)
    if 0 <= sample_idx < n_samples:
        ts[sample_idx] = amplitude
    return ts


# -------------------------------------------------------------------
# compute_spatial_search_peaks
# -------------------------------------------------------------------

def test_compute_spatial_search_peaks_finds_target_when_centered():
    """Place a pulse only at the target voxel, timed to coincide with
    that voxel's own gate center. argmax must land at the target and
    spatial_offset_mm must be ~0."""
    aperture = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 100.0])
    cube = _make_cube(target, step_mm=0.5, n_per_side=3)  # 27 voxels
    n_v = cube.shape[0]

    # Target is the geometric center of the cube -> index 13 for 3^3
    # indexing="ij", but we compute it rather than hardcode.
    dists = np.linalg.norm(cube - target[None, :], axis=1)
    target_idx = int(np.argmin(dists))
    assert dists[target_idx] == pytest.approx(0.0)

    dt = 1e-7           # 10 MHz sampling
    n_t = 2000
    t_water = 5.0e-5    # 50 us
    gate_half_width = 5 * dt

    # Build time series: every voxel is zero except the target, which
    # has a unit pulse at the sample nearest its own gate center.
    ts = np.zeros((n_v, n_t), dtype=float)
    target_gate = per_voxel_water_calibrated_gate_center(
        sensor_world_mm=cube[target_idx],
        target_world_mm=target,
        aperture_center_world_mm=aperture,
        t_water_peak_target=t_water,
    )
    target_sample = int(round(target_gate / dt))
    ts[target_idx] = _pulse_at_sample(n_t, target_sample, amplitude=1.0)

    result = compute_spatial_search_peaks(
        target_cube_time_series=ts,
        cube_world_mm=cube,
        target_world_mm=target,
        aperture_center_world_mm=aperture,
        t_water_peak_target=t_water,
        dt=dt,
        gate_half_width_s=gate_half_width,
    )

    assert result['argmax_voxel_index'] == target_idx
    assert result['spatial_offset_mm'] == pytest.approx(0.0, abs=1e-9)
    assert result['argmax_peak'] == pytest.approx(1.0)
    assert result['at_target_peak'] == pytest.approx(1.0)
    assert result['per_voxel_peak'].shape == (n_v,)


def test_compute_spatial_search_peaks_finds_offset_when_peak_is_spatial():
    """Same cube, but now a different voxel (not the target) carries
    the pulse, timed to coincide with *its own* gate center. argmax
    must land at that voxel, and spatial_offset_mm must be > 0 and
    equal to the known displacement."""
    aperture = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 100.0])
    step_mm = 0.5
    cube = _make_cube(target, step_mm=step_mm, n_per_side=3)
    n_v = cube.shape[0]

    dists_to_target = np.linalg.norm(cube - target[None, :], axis=1)
    target_idx = int(np.argmin(dists_to_target))

    # Pick a corner voxel as the "true" spatial peak location.
    corner_idx = int(np.argmax(dists_to_target))
    assert corner_idx != target_idx
    expected_offset = float(dists_to_target[corner_idx])
    assert expected_offset > 0.0

    dt = 1e-7
    n_t = 2000
    t_water = 5.0e-5
    gate_half_width = 5 * dt

    # Pulse only at the corner voxel, at its per-voxel gate center.
    ts = np.zeros((n_v, n_t), dtype=float)
    corner_gate = per_voxel_water_calibrated_gate_center(
        sensor_world_mm=cube[corner_idx],
        target_world_mm=target,
        aperture_center_world_mm=aperture,
        t_water_peak_target=t_water,
    )
    corner_sample = int(round(corner_gate / dt))
    ts[corner_idx] = _pulse_at_sample(n_t, corner_sample, amplitude=2.5)

    result = compute_spatial_search_peaks(
        target_cube_time_series=ts,
        cube_world_mm=cube,
        target_world_mm=target,
        aperture_center_world_mm=aperture,
        t_water_peak_target=t_water,
        dt=dt,
        gate_half_width_s=gate_half_width,
    )

    assert result['argmax_voxel_index'] == corner_idx
    assert result['spatial_offset_mm'] == pytest.approx(expected_offset, rel=1e-9)
    assert result['spatial_offset_mm'] > 0.0
    assert result['argmax_peak'] == pytest.approx(2.5)
    # The target voxel saw nothing, so at_target_peak is 0.
    assert result['at_target_peak'] == pytest.approx(0.0)
    np.testing.assert_allclose(result['argmax_world_mm'], cube[corner_idx])
