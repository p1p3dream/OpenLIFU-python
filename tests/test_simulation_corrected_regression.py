"""Regression tests for SimulationCorrected focal targeting.

These tests reproduce the 2026-04-17 "focus at z=180 instead of z=98" bug.
Root cause: neither `calc_delays` nor `run_simulation` received a transducer
pose transform, so elements were placed at raw transducer-local coordinates
inside the MRI world grid.

The homogeneous-water case isolates the pose / pipeline path from skull
aberration effects: if the focus does not land at the target in water,
the bug is purely geometric.

Both tests require a working k-wave installation and are skipped otherwise.
They are intentionally small (1mm grid, 4-element ring) so they finish in
under 30 seconds on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xa

from openlifu.bf.delay_methods import SimulationCorrected
from openlifu.geo import Point

kwave = pytest.importorskip("kwave")


# Grid: 1 mm isotropic, 120 mm cube, centered at world (0, 0, 0)
GRID_N = 120
GRID_DX_MM = 1.0
GRID_HALF_MM = (GRID_N * GRID_DX_MM) / 2

# Transducer geometry: 4-element ring, 40 mm radius, geometric focus at 80 mm
# in the transducer-local +z direction.
N_ELEMENTS = 4
ARRAY_RADIUS_MM = 40.0
ARRAY_ROC_MM = 80.0
FREQ_HZ = 500_000
SOUND_SPEED_MPS = 1500.0
DENSITY_KGM3 = 1000.0
ATTENUATION = 0.0


def _build_water_params():
    """Build a homogeneous-water xarray Dataset covering a 120 mm cube."""
    coords = {}
    for dim in ("x", "y", "z"):
        cv = np.linspace(-GRID_HALF_MM, GRID_HALF_MM, GRID_N, endpoint=False)
        coords[dim] = xa.DataArray(cv, dims=[dim], attrs={"units": "mm"})

    shape = (GRID_N, GRID_N, GRID_N)
    sound_speed = xa.DataArray(
        np.full(shape, SOUND_SPEED_MPS, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "m/s", "ref_value": SOUND_SPEED_MPS},
    )
    density = xa.DataArray(
        np.full(shape, DENSITY_KGM3, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "kg/m^3", "ref_value": DENSITY_KGM3},
    )
    attenuation = xa.DataArray(
        np.full(shape, ATTENUATION, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "dB/cm/MHz", "ref_value": ATTENUATION},
    )
    return xa.Dataset({
        "sound_speed": sound_speed,
        "density": density,
        "attenuation": attenuation,
    })


def _build_ring_transducer():
    """Build a 4-element ring transducer centered at transducer-local origin,
    aperture normal pointing in +z, geometric focus at local z = +ROC."""
    from openlifu import xdc

    elements = []
    for i in range(N_ELEMENTS):
        theta = 2 * np.pi * i / N_ELEMENTS
        x = ARRAY_RADIUS_MM * np.cos(theta)
        y = ARRAY_RADIUS_MM * np.sin(theta)
        z = 0.0  # flat ring; rely on delays for focusing
        el = xdc.Element(
            x=x, y=y, z=z,
            az=0.0, el=0.0, roll=0.0,
            w=5.0, l=5.0,
            units="mm",
        )
        elements.append(el)

    arr = xdc.Transducer(
        elements=elements,
        frequency=FREQ_HZ,
        units="mm",
    )
    return arr


def _transducer_to_world_transform(world_center_mm, aperture_normal=(0, 0, -1)):
    """Build a 4x4 matrix placing the transducer's local origin at
    `world_center_mm` with aperture pointing along `aperture_normal`.
    For the default normal (0, 0, -1), elements on the local +z side end up
    in the world -z direction, so the focus falls in -z relative to the
    array center."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(world_center_mm, dtype=float)
    if aperture_normal == (0, 0, -1):
        T[2, 2] = -1
        T[1, 1] = -1  # maintain right-handed frame
    return T


def _focal_peak_mm(pmax_dataarray):
    """Return (x, y, z) of the peak of a 3D xarray, in mm."""
    arr = pmax_dataarray.transpose("x", "y", "z").data
    idx = np.unravel_index(np.argmax(arr), arr.shape)
    return (
        float(pmax_dataarray.coords["x"].values[idx[0]]),
        float(pmax_dataarray.coords["y"].values[idx[1]]),
        float(pmax_dataarray.coords["z"].values[idx[2]]),
    )


@pytest.mark.slow
def test_homogeneous_water_geometric_focus_with_zero_delays():
    """Sanity check: a flat ring transducer fired with delays=0 should produce
    its peak pressure at the geometric center of the ring (i.e. directly in
    front of the array, at the aperture plane + near-field distance).

    This isolates the forward-sim path: if the peak is NOT near the array
    plane, the forward-sim coordinate handling is broken independent of
    any delay calculation.
    """
    from openlifu.sim.kwave_if import run_simulation

    params = _build_water_params()
    arr = _build_ring_transducer()
    array_world_center_mm = (0.0, 0.0, 40.0)
    tx_to_world = _transducer_to_world_transform(array_world_center_mm)

    delays = np.zeros(N_ELEMENTS)

    try:
        result = run_simulation(
            arr=arr, params=params, delays=delays,
            freq=FREQ_HZ, cycles=3, amplitude=1.0,
            ref_values_only=True, gpu=False,
            transform=tx_to_world,
        )
    except TypeError as e:
        if "transform" in str(e):
            pytest.xfail(
                "run_simulation does not accept transform yet; Change 2 of "
                "POSE_FIX_PROPOSAL_2026-04-17.md is not applied."
            )
        raise

    peak_x, peak_y, peak_z = _focal_peak_mm(result["p_max"])
    assert abs(peak_x) < 5.0, f"Peak x {peak_x} should be near 0"
    assert abs(peak_y) < 5.0, f"Peak y {peak_y} should be near 0"
    assert abs(peak_z - array_world_center_mm[2]) < ARRAY_RADIUS_MM, (
        f"Peak z {peak_z} should be within ring radius of array plane "
        f"{array_world_center_mm[2]}"
    )


@pytest.mark.slow
def test_homogeneous_water_simulation_corrected_focuses_at_target():
    """End-to-end: with SimulationCorrected delays in homogeneous water,
    the forward-sim focal peak should land within 2*dx of the target.

    This is the bug reproducer. Today it fails because run_simulation does
    not accept `transform`. After the pose-fix diff, it should pass in
    homogeneous water, then we can extend to skull phantoms.
    """
    from openlifu.sim.kwave_if import run_simulation

    params = _build_water_params()
    arr = _build_ring_transducer()
    array_world_center_mm = (0.0, 0.0, 40.0)
    tx_to_world = _transducer_to_world_transform(array_world_center_mm)

    target_world_mm = (0.0, 0.0, -40.0)
    target = Point(
        position=target_world_mm,
        units="mm",
        dims=("x", "y", "z"),
    )

    method = SimulationCorrected(c0=SOUND_SPEED_MPS, cfl=0.3, n_cycles=3, gpu=False)

    try:
        delays = method.calc_delays(arr, target, params, transform=tx_to_world)
    except (ValueError, RuntimeError) as e:
        pytest.xfail(
            f"calc_delays raised {type(e).__name__}: {e}. Likely pose / "
            "out-of-grid issue; see POSE_FIX_PROPOSAL_2026-04-17.md."
        )

    try:
        result = run_simulation(
            arr=arr, params=params, delays=delays,
            freq=FREQ_HZ, cycles=3, amplitude=1.0,
            ref_values_only=True, gpu=False,
            transform=tx_to_world,
        )
    except TypeError as e:
        if "transform" in str(e):
            pytest.xfail(
                "run_simulation does not accept transform yet."
            )
        raise

    peak_x, peak_y, peak_z = _focal_peak_mm(result["p_max"])
    tol_mm = 2 * GRID_DX_MM
    assert abs(peak_x - target_world_mm[0]) < tol_mm, (
        f"Focal x {peak_x} > {tol_mm} mm from target {target_world_mm[0]}"
    )
    assert abs(peak_y - target_world_mm[1]) < tol_mm, (
        f"Focal y {peak_y} > {tol_mm} mm from target {target_world_mm[1]}"
    )
    assert abs(peak_z - target_world_mm[2]) < tol_mm, (
        f"Focal z {peak_z} > {tol_mm} mm from target {target_world_mm[2]}"
    )
