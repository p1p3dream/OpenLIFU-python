"""Validate ComplexWeighted on a homogeneous-water case.

Purpose
-------
Sanity-check ``ComplexWeighted.calc_complex_weights`` against a known-correct
reference: a bowl transducer aimed at its geometric focus, in homogeneous water.
Because every element is equidistant from the focus, the true behavior is:

  * all per-element amplitudes equal (no skull => no element-level blocking);
  * all per-element phases equal (equal TOF, equal narrowband phase);
  * coherence factor CF ~= 1;
  * delays agree with Direct and SimulationCorrected within sample resolution.

If any of those fail substantially, ComplexWeighted has a bug we need to find
before any real-subject (GU008) integration.

Notes on the geometry
---------------------
The task spec suggested a 64 mm^3 water cube. With a bowl of radius-of-curvature
80 mm and a 40 mm aperture, the elements and their focus cannot both fit inside
a 64 mm cube. The elements sit at z ~= -radius * cos(asin(aperture/2/radius))
~= -77.5 mm from the focus, so the grid must span more than 80 mm on each side
of the focus. We use a 192 mm cube at 1 mm spacing (still very cheap). The bowl
is placed with its geometric focus at the grid origin; the focus is the
``target`` point for all three delay methods.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import xarray as xa

from openlifu import xdc
from openlifu.bf.delay_methods import ComplexWeighted, Direct, SimulationCorrected
from openlifu.geo import Point


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_bowl_transducer(
    n_elements: int = 8,
    radius_mm: float = 80.0,
    aperture_mm: float = 40.0,
    frequency_hz: float = 500_000,
    apex_below_focus: bool = True,
) -> xdc.Transducer:
    """Build an n-element bowl transducer on a spherical cap.

    The transducer's local origin is the bowl's geometric focus (i.e. the
    radius-of-curvature point). Elements are distributed evenly in azimuth on
    a spherical cap of radius ``radius_mm`` and aperture half-angle set so the
    lateral extent equals ``aperture_mm``/2.

    With ``apex_below_focus=True`` the bowl opens toward +z (focus at origin,
    elements at negative z, aimed up the z axis).
    """
    aperture_half = aperture_mm / 2.0
    # half-angle of the cap: sin(alpha) = aperture_half / radius
    alpha = np.arcsin(aperture_half / radius_mm)
    positions_mm = []
    z_sign = -1.0 if apex_below_focus else +1.0
    for i in range(n_elements):
        theta = 2 * np.pi * i / n_elements
        x = aperture_half * np.cos(theta)
        y = aperture_half * np.sin(theta)
        # z coordinate on the spherical cap, measured from the focus
        z = z_sign * radius_mm * np.cos(alpha)
        positions_mm.append((x, y, z))

    elements = [
        xdc.Element(
            index=i,
            position=np.array(p, dtype=float),
            orientation=np.array([0.0, 0.0, 0.0]),
            size=np.array([5.0, 5.0]),
            units="mm",
        )
        for i, p in enumerate(positions_mm)
    ]
    return xdc.Transducer(
        elements=elements, frequency=frequency_hz, units="mm",
    )


def build_water_params(n: int = 128, dx_mm: float = 1.0, c0: float = 1500.0, rho: float = 1000.0) -> xa.Dataset:
    """Homogeneous-water xarray Dataset on an n^3 cube centered at origin."""
    half = (n * dx_mm) / 2.0
    coords = {}
    for dim in ("x", "y", "z"):
        cv = np.linspace(-half, half, n, endpoint=False)
        coords[dim] = xa.DataArray(cv, dims=[dim], attrs={"units": "mm"})
    shape = (n, n, n)
    sound_speed = xa.DataArray(
        np.full(shape, c0, dtype=np.float32),
        dims=("x", "y", "z"), coords=coords,
        attrs={"units": "m/s", "ref_value": c0},
    )
    density = xa.DataArray(
        np.full(shape, rho, dtype=np.float32),
        dims=("x", "y", "z"), coords=coords,
        attrs={"units": "kg/m^3", "ref_value": rho},
    )
    attenuation = xa.DataArray(
        np.zeros(shape, dtype=np.float32),
        dims=("x", "y", "z"), coords=coords,
        attrs={"units": "dB/cm/MHz", "ref_value": 0.0},
    )
    return xa.Dataset({
        "sound_speed": sound_speed,
        "density": density,
        "attenuation": attenuation,
    })


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    mean: float
    std: float
    pmin: float
    pmax: float

    @classmethod
    def of(cls, x: np.ndarray) -> "Stats":
        x = np.asarray(x, dtype=float)
        return cls(
            mean=float(np.mean(x)),
            std=float(np.std(x)),
            pmin=float(np.min(x)),
            pmax=float(np.max(x)),
        )


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---------------- geometry ----------------
    c0 = 1500.0
    f0 = 500_000.0
    radius_mm = 80.0
    aperture_mm = 40.0
    n_elements = 8
    # Elements sit at z ~= -radius*cos(alpha) ~= -77.5 mm from the focus, so
    # the half-grid along z must exceed 78 mm. np.linspace(-half, half, n,
    # endpoint=False) gives bounds [-half, half - dx], i.e. [-96, +95] for
    # n=192 dx=1 mm: plenty of margin for a focus at origin + elements at
    # -77.5 mm.
    grid_n = 192

    arr = build_bowl_transducer(
        n_elements=n_elements,
        radius_mm=radius_mm,
        aperture_mm=aperture_mm,
        frequency_hz=f0,
        apex_below_focus=True,
    )
    # Target = bowl geometric focus at origin
    target = Point(
        position=np.array([0.0, 0.0, 0.0]), units="mm", dims=("x", "y", "z"),
    )
    params = build_water_params(n=grid_n, dx_mm=1.0, c0=c0)

    banner("Setup")
    # Report element geometry
    elem_pos = np.array([el.get_position(units="mm") for el in arr.elements])
    dists_mm = np.linalg.norm(elem_pos - np.array([0.0, 0.0, 0.0]), axis=1)
    print(f"Transducer: {n_elements} elements, f0 = {f0/1e3:.0f} kHz, c0 = {c0:.0f} m/s")
    print(f"Bowl radius = {radius_mm:.1f} mm, aperture = {aperture_mm:.1f} mm")
    print(f"Grid: {grid_n}^3 mm at 1 mm, water (uniform).")
    print("Element positions (mm, transducer-local):")
    for i, (p, d) in enumerate(zip(elem_pos, dists_mm)):
        print(f"  [{i}] xyz = ({p[0]:+7.3f}, {p[1]:+7.3f}, {p[2]:+7.3f})  "
              f"|dist-to-focus| = {d:.3f} mm")
    # In water, TOF = d/c
    tofs_s = dists_mm * 1e-3 / c0
    print(f"Geometric TOFs (us): "
          f"mean {tofs_s.mean()*1e6:.3f}, "
          f"std {tofs_s.std()*1e6:.6f}, "
          f"max-min {(tofs_s.max()-tofs_s.min())*1e6:.6f}")
    if dists_mm.std() / dists_mm.mean() > 1e-6:
        print("WARNING: bowl elements are NOT equidistant from the focus; "
              "uniformity expectations below assume equidistance.")

    # ---------------- Direct ----------------
    banner("Direct.calc_delays")
    direct = Direct(c0=c0)
    t0 = time.time()
    delays_direct = direct.calc_delays(arr, target, params=params, transform=None)
    print(f"Ran in {time.time()-t0:.3f} s")
    print(f"delays (ns): {(delays_direct*1e9).round(3)}")
    print(f"delay max-min = {(delays_direct.max()-delays_direct.min())*1e9:.3f} ns")

    # ---------------- SimulationCorrected ----------------
    banner("SimulationCorrected.calc_delays")
    sim = SimulationCorrected(c0=c0, cfl=0.3, n_cycles=3, gpu=True)
    t0 = time.time()
    try:
        delays_sim = sim.calc_delays(arr, target, params=params, transform=None)
    except Exception as exc:
        print(f"SimulationCorrected raised: {exc!r}")
        delays_sim = None
    if delays_sim is not None:
        print(f"Ran in {time.time()-t0:.3f} s")
        print(f"delays (ns): {(delays_sim*1e9).round(3)}")
        print(f"delay max-min = {(delays_sim.max()-delays_sim.min())*1e9:.3f} ns")

    # ---------------- ComplexWeighted ----------------
    banner("ComplexWeighted.calc_complex_weights")
    cw = ComplexWeighted(
        c0=c0, cfl=0.3, n_cycles=3, gpu=True,
        bandwidth_frac=0.1, window_cycles=6.0,
    )
    t0 = time.time()
    # Get the raw narrowband coefficients AND the (delays, apod) split so we can
    # reason about amplitudes, phases, and CF without re-running the sim.
    try:
        amplitudes, phases, f0_returned = cw._run_reciprocal_simulation_complex(
            arr, target, params, transform=None,
        )
    except Exception as exc:
        print(f"ComplexWeighted reciprocal sim raised: {exc!r}")
        return 1
    delays_cw, apod_cw = ComplexWeighted._weights_from_coefficients(
        amplitudes, phases, f0_returned,
    )
    dt_sim = 0.3 * 0.001 / c0  # approximate; used for reporting tol in samples
    print(f"Ran in {time.time()-t0:.3f} s, f0 returned = {f0_returned/1e3:.3f} kHz")

    # --- Amplitudes ---
    a_stats = Stats.of(amplitudes)
    apod_stats = Stats.of(apod_cw)
    amp_ratio = a_stats.pmax / a_stats.pmin if a_stats.pmin > 0 else np.inf
    amp_cv = a_stats.std / a_stats.mean if a_stats.mean > 0 else np.inf
    print("Per-element RAW amplitudes |coef|:")
    print(f"  values = {amplitudes.round(6)}")
    print(f"  mean   = {a_stats.mean:.6g}")
    print(f"  std    = {a_stats.std:.6g}")
    print(f"  max/min ratio = {amp_ratio:.4f}")
    print(f"  std/mean      = {amp_cv:.4f}  (expected < 0.05)")
    print("Per-element NORMALIZED apodization (max-normalized):")
    print(f"  values = {apod_cw.round(6)}")
    print(f"  mean   = {apod_stats.mean:.6g}")
    print(f"  std    = {apod_stats.std:.6g}")

    # --- Phases / delays ---
    wrapped_phases = np.angle(np.exp(1j * phases))
    p_stats = Stats.of(wrapped_phases)
    cw_delays_ns = delays_cw * 1e9
    d_stats = Stats.of(delays_cw)
    print("Per-element wrapped phases (rad):")
    print(f"  values = {wrapped_phases.round(6)}")
    print(f"  mean   = {p_stats.mean:.6g}, std = {p_stats.std:.6g}, "
          f"max-min = {p_stats.pmax - p_stats.pmin:.6g}")
    print("Per-element CW delays:")
    print(f"  values (ns) = {cw_delays_ns.round(3)}")
    print(f"  mean   = {d_stats.mean*1e9:.3f} ns")
    print(f"  std    = {d_stats.std*1e9:.3f} ns")
    print(f"  max-min= {(d_stats.pmax - d_stats.pmin)*1e9:.3f} ns")

    # --- Coherence factor from (a_i, phi_i) ---
    coef_complex = amplitudes * np.exp(1j * phases)
    cf_num = abs(np.sum(coef_complex))
    cf_den = float(np.sum(amplitudes))
    cf = cf_num / cf_den if cf_den > 0 else float("nan")
    print(f"Complex CF = |sum a_i exp(j phi_i)| / sum a_i = {cf:.6f}  (expected ~1.0)")

    # --- Cross-method delay agreement ---
    banner("Cross-method delay agreement")
    # Compare RELATIVE delays (subtract per-method mean): CW returns phase-only
    # delays with no absolute TOF origin, Direct returns max(tof)-tof (also
    # referenced to the peak), and SimulationCorrected references to the
    # envelope peak. The absolute offsets differ; what matters is the
    # *pattern* across elements.
    def _rel(x):
        return x - np.mean(x)

    rel_direct = _rel(delays_direct)
    rel_cw = _rel(delays_cw)
    diff_cw_direct_ns = (rel_cw - rel_direct) * 1e9
    print("Direct vs ComplexWeighted (after per-method mean subtraction):")
    print(f"  |max diff| = {np.max(np.abs(diff_cw_direct_ns)):.3f} ns")
    print(f"  |max diff| / dt (~{dt_sim*1e9:.1f} ns) "
          f"= {np.max(np.abs(diff_cw_direct_ns))/(dt_sim*1e9):.3f} samples")
    print(f"  diffs (ns) = {diff_cw_direct_ns.round(3)}")

    if delays_sim is not None:
        rel_sim = _rel(delays_sim)
        diff_cw_sim_ns = (rel_cw - rel_sim) * 1e9
        diff_sim_direct_ns = (rel_sim - rel_direct) * 1e9
        print("SimulationCorrected vs ComplexWeighted (mean-subtracted):")
        print(f"  |max diff| = {np.max(np.abs(diff_cw_sim_ns)):.3f} ns "
              f"(~{np.max(np.abs(diff_cw_sim_ns))/(dt_sim*1e9):.3f} samples)")
        print(f"  diffs (ns) = {diff_cw_sim_ns.round(3)}")
        print("Direct vs SimulationCorrected (mean-subtracted):")
        print(f"  |max diff| = {np.max(np.abs(diff_sim_direct_ns)):.3f} ns "
              f"(~{np.max(np.abs(diff_sim_direct_ns))/(dt_sim*1e9):.3f} samples)")
        print(f"  diffs (ns) = {diff_sim_direct_ns.round(3)}")

    # ---------------- Verdict ----------------
    banner("Verdict")
    amp_ok = amp_cv < 0.05
    delay_span_samples = (d_stats.pmax - d_stats.pmin) / dt_sim
    delay_ok = delay_span_samples < 2.0  # few-sample tolerance
    cf_ok = abs(cf - 1.0) < 0.05
    cw_vs_direct_samples = np.max(np.abs(diff_cw_direct_ns)) / (dt_sim * 1e9)
    cw_vs_direct_ok = cw_vs_direct_samples < 2.0
    if delays_sim is not None:
        cw_vs_sim_samples = np.max(np.abs(diff_cw_sim_ns)) / (dt_sim * 1e9)
        cw_vs_sim_ok = cw_vs_sim_samples < 2.0
    else:
        cw_vs_sim_samples = float("nan")
        cw_vs_sim_ok = False

    def tag(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"Amplitude uniformity (std/mean < 0.05): {tag(amp_ok)}  "
          f"(std/mean = {amp_cv:.4f})")
    print(f"Delay uniformity (max-min < 2 samples): {tag(delay_ok)}  "
          f"(span = {delay_span_samples:.3f} samples)")
    print(f"Coherence factor ~1.0 (|CF-1| < 0.05):  {tag(cf_ok)}  "
          f"(CF = {cf:.4f})")
    print(f"CW vs Direct relative delays (< 2 samples): {tag(cw_vs_direct_ok)}  "
          f"({cw_vs_direct_samples:.3f} samples)")
    if delays_sim is not None:
        print(f"CW vs SimCorrected relative delays (< 2 samples): {tag(cw_vs_sim_ok)}  "
              f"({cw_vs_sim_samples:.3f} samples)")

    all_ok = amp_ok and delay_ok and cf_ok and cw_vs_direct_ok and cw_vs_sim_ok
    if all_ok:
        print("\nRESULT: ComplexWeighted behaves correctly on the water reference. "
              "Recommend proceeding to GU008 integration.")
        return 0
    print("\nRESULT: ComplexWeighted FAILED at least one water-reference check.")
    failures = []
    if not amp_ok:
        failures.append("non-uniform amplitudes in water (possible envelope-peak "
                        "window mis-centering or normalization bug)")
    if not delay_ok:
        failures.append("non-uniform delays across equidistant elements (possible "
                        "phase-unwrap or window-centering inconsistency)")
    if not cf_ok:
        failures.append(f"coherence factor {cf:.3f} is not ~1 (phases disagree)")
    if not cw_vs_direct_ok:
        failures.append("CW delays do not match geometric Direct delays within sample tol")
    if delays_sim is not None and not cw_vs_sim_ok:
        failures.append("CW delays disagree with SimulationCorrected envelope-peak delays")
    for f in failures:
        print(f"  - {f}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
