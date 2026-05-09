"""Validate that SimulationCorrected delays match Direct delays in homogeneous water.

In a uniform-speed medium (no skull), the reciprocal simulation pipeline
(sensor mapping, Hilbert extraction, peak-picking, delay conversion) should
reproduce the geometric time-of-flight delays to within a few time steps of
k-Wave numerical noise. Because all 16 ring elements are equidistant from
the target, both Direct and SimulationCorrected should produce all-zero
delays (after the max-minus-tof normalization).

Usage:
    PYTHONPATH=src python3 scripts/validate_delay_equality.py
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import xarray as xa

from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def build_ring_transducer(
    n_elements: int = 16,
    radius_mm: float = 30.0,
    z_mm: float = 30.0,
    frequency_hz: float = 500_000.0,
    element_size_mm: float = 3.0,
) -> Transducer:
    """Build a flat ring transducer in the XY plane at the given z offset.

    All elements sit on a circle of radius ``radius_mm`` at z = ``z_mm``,
    equally spaced in azimuth. Orientation is zero (flat, no tilt).
    """
    elements = []
    for i in range(n_elements):
        theta = 2 * np.pi * i / n_elements
        x = radius_mm * np.cos(theta)
        y = radius_mm * np.sin(theta)
        elements.append(
            Element(
                index=i,
                position=np.array([x, y, z_mm], dtype=float),
                orientation=np.array([0.0, 0.0, 0.0]),
                size=np.array([element_size_mm, element_size_mm]),
                units="mm",
            )
        )
    return Transducer(
        id="ring16",
        name="16-element ring",
        elements=elements,
        frequency=frequency_hz,
        units="mm",
    )


def build_water_params(
    n: int = 80,
    dx_mm: float = 1.0,
    c0: float = 1500.0,
    rho: float = 1000.0,
) -> xa.Dataset:
    """Homogeneous-water xarray Dataset on an n^3 cube centered at origin."""
    half = (n * dx_mm) / 2.0
    coords = {}
    for dim in ("x", "y", "z"):
        cv = np.linspace(-half, half, n, endpoint=False)
        coords[dim] = xa.DataArray(cv, dims=[dim], attrs={"units": "mm"})
    shape = (n, n, n)
    sound_speed = xa.DataArray(
        np.full(shape, c0, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "m/s", "ref_value": c0},
    )
    density = xa.DataArray(
        np.full(shape, rho, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "kg/m^3", "ref_value": rho},
    )
    attenuation = xa.DataArray(
        np.zeros(shape, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "dB/cm/MHz", "ref_value": 0.0},
    )
    return xa.Dataset({
        "sound_speed": sound_speed,
        "density": density,
        "attenuation": attenuation,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Check for k-wave availability early --
    try:
        import importlib
        if importlib.util.find_spec("kwave") is None:
            raise ImportError
    except ImportError:
        print("k-wave is not installed. SimulationCorrected requires k-wave.")
        print("Install it with: pip install k-wave-python")
        print("Skipping validation.")
        return 0

    # -- Constants --
    c0 = 1500.0          # m/s
    f0 = 500_000.0       # Hz
    n_elements = 16
    ring_radius_mm = 30.0
    ring_z_mm = 30.0
    grid_n = 80
    dx_mm = 1.0

    # -- Build geometry --
    arr = build_ring_transducer(
        n_elements=n_elements,
        radius_mm=ring_radius_mm,
        z_mm=ring_z_mm,
        frequency_hz=f0,
    )
    target = Point(
        position=np.array([0.0, 0.0, 0.0]),
        units="mm",
        dims=("x", "y", "z"),
    )
    params = build_water_params(n=grid_n, dx_mm=dx_mm, c0=c0)

    # -- Report geometry --
    print()
    print("=" * 72)
    print("SETUP")
    print("=" * 72)
    elem_pos = np.array([el.get_position(units="mm") for el in arr.elements])
    dists_mm = np.linalg.norm(elem_pos, axis=1)
    print(f"Transducer: {n_elements} elements, f0 = {f0/1e3:.0f} kHz, c0 = {c0:.0f} m/s")
    print(f"Ring radius = {ring_radius_mm:.1f} mm, ring z = {ring_z_mm:.1f} mm")
    print(f"Grid: {grid_n}^3 voxels at {dx_mm} mm spacing, homogeneous water")
    print(f"Target: origin (0, 0, 0)")
    print()
    print(f"Element-to-target distances (mm):")
    print(f"  mean = {dists_mm.mean():.4f}")
    print(f"  std  = {dists_mm.std():.6f}")
    print(f"  max-min = {dists_mm.max() - dists_mm.min():.6f}")
    expected_dist = np.sqrt(ring_radius_mm**2 + ring_z_mm**2)
    print(f"  expected = sqrt({ring_radius_mm}^2 + {ring_z_mm}^2) = {expected_dist:.4f} mm")
    print()

    # -- Direct delays --
    print("=" * 72)
    print("DIRECT DELAYS")
    print("=" * 72)
    direct = Direct(c0=c0)
    t0 = time.time()
    delays_direct = direct.calc_delays(arr, target, params=params)
    elapsed_direct = time.time() - t0
    print(f"Computed in {elapsed_direct:.3f} s")
    print(f"Delays (us): {(delays_direct * 1e6).round(4)}")
    print(f"Max delay = {delays_direct.max() * 1e6:.4f} us")
    print()

    # -- SimulationCorrected (first_arrival) --
    print("=" * 72)
    print("SIMULATION-CORRECTED (first_arrival)")
    print("=" * 72)
    sim_fa = SimulationCorrected(
        c0=c0, cfl=0.3, n_cycles=3, gpu=False,
        peak_method="first_arrival",
    )
    t0 = time.time()
    delays_sim_fa = sim_fa.calc_delays(arr, target, params=params)
    elapsed_sim_fa = time.time() - t0
    print(f"Computed in {elapsed_sim_fa:.3f} s")
    print(f"Delays (us): {(delays_sim_fa * 1e6).round(4)}")
    print(f"Max delay = {delays_sim_fa.max() * 1e6:.4f} us")
    print()

    # -- SimulationCorrected (argmax) --
    print("=" * 72)
    print("SIMULATION-CORRECTED (argmax)")
    print("=" * 72)
    sim_am = SimulationCorrected(
        c0=c0, cfl=0.3, n_cycles=3, gpu=False,
        peak_method="argmax",
    )
    t0 = time.time()
    delays_sim_am = sim_am.calc_delays(arr, target, params=params)
    elapsed_sim_am = time.time() - t0
    print(f"Computed in {elapsed_sim_am:.3f} s")
    print(f"Delays (us): {(delays_sim_am * 1e6).round(4)}")
    print(f"Max delay = {delays_sim_am.max() * 1e6:.4f} us")
    print()

    # -- Compute dt for the tolerance check --
    # dt = CFL * dx / c0
    dx_m = dx_mm * 1e-3
    dt = 0.3 * dx_m / c0
    print("=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)
    print(f"dt = CFL * dx / c0 = 0.3 * {dx_m*1e3:.1f}mm / {c0:.0f} = {dt*1e9:.2f} ns")
    print(f"Tolerance: 5 * dt = {5 * dt * 1e6:.4f} us = {5 * dt * 1e9:.2f} ns")
    print()

    header = f"{'Elem':>4s}  {'Direct':>10s}  {'Sim(FA)':>10s}  {'Sim(AM)':>10s}  {'|D-FA|':>10s}  {'|D-AM|':>10s}"
    print(header)
    print("-" * len(header))
    for i in range(n_elements):
        d = delays_direct[i] * 1e6
        fa = delays_sim_fa[i] * 1e6
        am = delays_sim_am[i] * 1e6
        diff_fa = abs(d - fa)
        diff_am = abs(d - am)
        print(f"{i:4d}  {d:10.4f}  {fa:10.4f}  {am:10.4f}  {diff_fa:10.4f}  {diff_am:10.4f}")
    print()

    # -- Differences --
    diff_fa = np.abs(delays_direct - delays_sim_fa)
    diff_am = np.abs(delays_direct - delays_sim_am)

    max_diff_fa_us = diff_fa.max() * 1e6
    max_diff_am_us = diff_am.max() * 1e6
    tolerance_us = 5 * dt * 1e6

    print(f"Max |Direct - Sim(first_arrival)| = {max_diff_fa_us:.4f} us ({diff_fa.max() / dt:.1f} dt)")
    print(f"Max |Direct - Sim(argmax)|        = {max_diff_am_us:.4f} us ({diff_am.max() / dt:.1f} dt)")
    print(f"Tolerance (5*dt)                  = {tolerance_us:.4f} us")
    print()

    # -- Assertions --
    ok = True
    if max_diff_fa_us > tolerance_us:
        print(f"FAIL: first_arrival difference ({max_diff_fa_us:.4f} us) exceeds 5*dt ({tolerance_us:.4f} us)")
        ok = False
    else:
        print(f"PASS: first_arrival difference ({max_diff_fa_us:.4f} us) within 5*dt ({tolerance_us:.4f} us)")

    if max_diff_am_us > tolerance_us:
        print(f"FAIL: argmax difference ({max_diff_am_us:.4f} us) exceeds 5*dt ({tolerance_us:.4f} us)")
        ok = False
    else:
        print(f"PASS: argmax difference ({max_diff_am_us:.4f} us) within 5*dt ({tolerance_us:.4f} us)")

    print()
    if ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
