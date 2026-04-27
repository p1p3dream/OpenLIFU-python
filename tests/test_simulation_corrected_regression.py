"""Regression tests for SimulationCorrected pose/transform plumbing.

Root cause of the 2026-04-17 bug: neither `calc_delays` nor `run_simulation`
received a transducer pose transform, so elements were placed at raw local
coordinates inside the MRI world grid.

Test strategy:
  1. Delay-equality: in homogeneous water, SimulationCorrected delays should
     match Direct (geometric) delays within k-Wave numerical noise (~few dt).
     This validates the full reciprocal-sim pipeline without needing enough
     elements to produce a clean focal peak.
  2. Transform-invariance: delays should not depend on where the array is
     placed in the grid, only on the array-to-target geometry.
  3. Fallback detection: calc_delays must NOT silently fall back to Direct
     when a ValueError (out-of-grid) is raised.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xa

from openlifu.bf.delay_methods import SimulationCorrected
from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import OutOfGridError
from openlifu.geo import Point

try:
    import kwave  # noqa: F401
    _has_kwave = True
except ImportError:
    _has_kwave = False

requires_kwave = pytest.mark.skipif(not _has_kwave, reason="kwave not installed")


GRID_N = 120
GRID_DX_MM = 1.0
GRID_HALF_MM = (GRID_N * GRID_DX_MM) / 2

N_ELEMENTS = 4
ARRAY_RADIUS_MM = 40.0
FREQ_HZ = 500_000
SOUND_SPEED_MPS = 1500.0
DENSITY_KGM3 = 1000.0


def _build_water_params():
    coords = {}
    for dim in ("x", "y", "z"):
        cv = np.linspace(-GRID_HALF_MM, GRID_HALF_MM, GRID_N, endpoint=False)
        coords[dim] = xa.DataArray(cv, dims=[dim], attrs={"units": "mm"})
    shape = (GRID_N, GRID_N, GRID_N)
    return xa.Dataset({
        "sound_speed": xa.DataArray(
            np.full(shape, SOUND_SPEED_MPS, dtype=np.float32),
            dims=("x", "y", "z"), coords=coords,
            attrs={"units": "m/s", "ref_value": SOUND_SPEED_MPS},
        ),
        "density": xa.DataArray(
            np.full(shape, DENSITY_KGM3, dtype=np.float32),
            dims=("x", "y", "z"), coords=coords,
            attrs={"units": "kg/m^3", "ref_value": DENSITY_KGM3},
        ),
        "attenuation": xa.DataArray(
            np.full(shape, 0.0, dtype=np.float32),
            dims=("x", "y", "z"), coords=coords,
            attrs={"units": "dB/cm/MHz", "ref_value": 0.0},
        ),
    })


def _build_ring_transducer():
    from openlifu import xdc
    elements = []
    for i in range(N_ELEMENTS):
        theta = 2 * np.pi * i / N_ELEMENTS
        x = ARRAY_RADIUS_MM * np.cos(theta)
        y = ARRAY_RADIUS_MM * np.sin(theta)
        elements.append(xdc.Element(
            index=i,
            position=np.array([x, y, 0.0], dtype=float),
            orientation=np.array([0.0, 0.0, 0.0], dtype=float),
            size=np.array([5.0, 5.0], dtype=float),
            units="mm",
        ))
    return xdc.Transducer(elements=elements, frequency=FREQ_HZ, units="mm")


def _make_transform(world_center_mm, flip_z=True):
    T = np.eye(4)
    T[:3, 3] = np.asarray(world_center_mm, dtype=float) * 1e-3
    if flip_z:
        T[2, 2] = -1
        T[1, 1] = -1
    return T


@requires_kwave
@pytest.mark.slow
def test_delay_equality_simcorrected_vs_direct_in_water():
    """In homogeneous water, SimulationCorrected delays should match Direct
    delays within a few time steps of k-Wave numerical noise.

    This validates that the reciprocal simulation, Hilbert peak-picking, and
    transform plumbing all produce geometrically correct arrival times.
    """
    params = _build_water_params()
    arr = _build_ring_transducer()
    tx_to_world = _make_transform((0.0, 0.0, 30.0))
    target = Point(position=(0.0, 0.0, -20.0), units="mm", dims=("x", "y", "z"))

    direct = Direct(c0=SOUND_SPEED_MPS)
    delays_direct = direct.calc_delays(arr, target, params, transform=tx_to_world)

    sim_corr = SimulationCorrected(c0=SOUND_SPEED_MPS, cfl=0.3, n_cycles=3, gpu=False)
    delays_sim = sim_corr.calc_delays(arr, target, params, transform=tx_to_world)

    dt_s = GRID_DX_MM * 1e-3 / SOUND_SPEED_MPS
    tol_s = 5 * dt_s

    diff = np.abs(delays_sim - delays_direct)
    print(f"\n[delay-equality] Direct delays (us): {delays_direct * 1e6}")
    print(f"[delay-equality] SimCorr delays (us): {delays_sim * 1e6}")
    print(f"[delay-equality] Max diff: {diff.max() * 1e6:.3f} us, tol: {tol_s * 1e6:.3f} us")

    assert diff.max() < tol_s, (
        f"SimulationCorrected delays differ from Direct by {diff.max()*1e6:.3f} us "
        f"(tol={tol_s*1e6:.3f} us). Delays are not geometrically correct."
    )

    assert np.allclose(delays_direct, delays_direct[0]), (
        "4-fold symmetric ring should have equal Direct delays, but got: "
        f"{delays_direct * 1e6}"
    )


@requires_kwave
@pytest.mark.slow
def test_delay_invariance_under_translation():
    """Delays should depend only on array-to-target geometry, not on absolute
    grid position. Translating both the array and target by the same offset
    should produce identical delays.
    """
    params = _build_water_params()
    arr = _build_ring_transducer()

    target_a = Point(position=(0.0, 0.0, -20.0), units="mm", dims=("x", "y", "z"))
    tx_a = _make_transform((0.0, 0.0, 30.0))

    target_b = Point(position=(10.0, 5.0, -10.0), units="mm", dims=("x", "y", "z"))
    tx_b = _make_transform((10.0, 5.0, 40.0))

    sim_corr = SimulationCorrected(c0=SOUND_SPEED_MPS, cfl=0.3, n_cycles=3, gpu=False)

    delays_a = sim_corr.calc_delays(arr, target_a, params, transform=tx_a)
    delays_b = sim_corr.calc_delays(arr, target_b, params, transform=tx_b)

    dt_s = GRID_DX_MM * 1e-3 / SOUND_SPEED_MPS
    tol_s = 5 * dt_s

    diff = np.abs(delays_a - delays_b)
    print(f"\n[translation-invariance] Delays A (us): {delays_a * 1e6}")
    print(f"[translation-invariance] Delays B (us): {delays_b * 1e6}")
    print(f"[translation-invariance] Max diff: {diff.max() * 1e6:.3f} us")

    assert diff.max() < tol_s, (
        f"Delays changed by {diff.max()*1e6:.3f} us under rigid translation "
        f"(tol={tol_s*1e6:.3f} us). Transform plumbing is broken."
    )


def test_out_of_grid_error_propagates_when_fallback_disabled():
    """OutOfGridError must propagate through calc_delays, not be caught by the
    generic (RuntimeError, ValueError, ...) handler.

    This validates that allow_out_of_grid_fallback=False actually prevents
    silent fallback to Direct delays when elements are outside the grid.
    """
    from unittest.mock import patch

    params = _build_water_params()
    arr = _build_ring_transducer()
    target = Point(position=(0.0, 0.0, 0.0), units="mm", dims=("x", "y", "z"))
    tx = _make_transform((0.0, 0.0, 0.0), flip_z=False)

    sim_corr = SimulationCorrected(
        c0=SOUND_SPEED_MPS, cfl=0.3, n_cycles=3, gpu=False,
        allow_out_of_grid_fallback=False,
    )

    def _raise_out_of_grid(*args, **kwargs):
        raise OutOfGridError("Element outside simulation grid")

    with patch.object(sim_corr, "_run_reciprocal_simulation", _raise_out_of_grid):
        with patch.object(sim_corr, "_fallback_delays") as mock_fallback:
            with pytest.raises(OutOfGridError):
                sim_corr.calc_delays(arr, target, params, transform=tx)
            mock_fallback.assert_not_called()
