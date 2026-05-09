#!/usr/bin/env python3
"""
UCL Transcranial Benchmark BM7 Analysis
========================================
Extract k-Wave reference results and compare all 11 solvers for BM7-SC1 and BM7-SC2.

BM7: Bowl transducer through skull bone (CTX bone model)
- SC1: Focused bowl source (64mm ROC, 64mm diameter)
- SC2: Plane piston source (20mm disc diameter)

Coordinate convention (from skull mask file):
  - xi (241 pts): axial direction, 0 to 120 mm
  - yi (141 pts): lateral direction, -35 to 35 mm
  - zi (141 pts): lateral direction, -35 to 35 mm
  - Arrays stored as (yi, zi, xi) = (141, 141, 241) for most solvers
  - Some solvers transpose to (xi, yi, zi) = (241, 141, 141)
"""

import numpy as np
import h5py
import scipy.io
from pathlib import Path

DATA_DIR = Path("/Users/brandon/Data/openlifu-validation/ucl-benchmarks/data")

# ============================================================
# Utility functions
# ============================================================

def load_p_amp(solver_name, scenario="SC1"):
    """Load p_amp array from any solver, return in canonical (yi, zi, xi) = (141, 141, 241) order."""

    # File naming conventions
    file_map = {
        "BABELVISCOFDTD": f"BABELVISCOFDTD/PH1-BM7-{scenario}_BabelViscoFDTD.h5",
        "FULLWAVE": f"FULLWAVE/PH1-BM7-{scenario}_FULLWAVE.mat",
        "GMFDTD": f"GMFDTD/PH1-BM7-{scenario}_GMFDTD.mat",
        "HAS": f"HAS/PH1-BM7-{scenario}_HAS.mat",
        "JWAVE": f"JWAVE/PH1-BM7-{scenario}_JWAVE.mat",
        "KWAVE": f"KWAVE/PH1-BM7-{scenario}_KWAVE.mat",
        "MSOUND": f"MSOUND/PH1-BM7-{scenario}_MSOUND.mat",
        "OPTIMUS": f"OPTIMUS/PH1-BM7-{scenario}_OPTIMUS.mat",
        "SALVUS": f"SALVUS/PH1-BM7-{scenario}_SALVUS.mat",
        "SIM4LIFE": f"SIM4LIFE/PH1-BM7-{scenario}_SIM4LIFE.mat",
        "STRIDE": f"STRIDE/PH1-BM7-{scenario}_STRIDE.mat",
    }

    fpath = DATA_DIR / file_map[solver_name]
    if not fpath.exists():
        return None

    # Solvers whose arrays are stored as (xi, yi, zi) = (241, 141, 141)
    # and need transposing to (yi, zi, xi) = (141, 141, 241)
    transposed_solvers = {"BABELVISCOFDTD", "MSOUND", "JWAVE", "OPTIMUS", "SALVUS", "SIM4LIFE"}

    p_amp = None
    p_phase = None

    # Try h5py first, fall back to scipy.io
    try:
        with h5py.File(fpath, 'r') as f:
            p_amp = f['p_amp'][()].astype(np.float64)
            if 'p_phase' in f:
                ph = f['p_phase'][()]
                if ph.size > 1:
                    p_phase = ph.astype(np.float64)
    except Exception:
        try:
            d = scipy.io.loadmat(str(fpath))
            p_amp = d['p_amp'].astype(np.float64)
            if 'p_phase' in d and d['p_phase'].size > 1:
                p_phase = d['p_phase'].astype(np.float64)
        except Exception as e:
            print(f"  ERROR loading {solver_name}: {e}")
            return None

    # Handle axis transposition
    if solver_name in transposed_solvers:
        # (241, 141, 141) -> (141, 141, 241)
        if p_amp.shape == (241, 141, 141):
            p_amp = np.transpose(p_amp, (1, 2, 0))
            if p_phase is not None:
                p_phase = np.transpose(p_phase, (1, 2, 0))
        elif p_amp.shape == (141, 141, 241):
            pass  # Already in canonical order
        else:
            print(f"  WARNING: {solver_name} unexpected shape {p_amp.shape}")
    elif solver_name == "GMFDTD":
        # GMFDTD: x_vec=141 (lateral), y_vec=141 (lateral), z_vec=241 (axial)
        # stored as (x, y, z) = (141, 141, 241), need to check
        # Actually the array is already (141, 141, 241) where last axis is axial
        # But GMFDTD's x,y are lateral and z is axial, matching our canonical form
        pass

    if p_amp.shape != (141, 141, 241):
        print(f"  WARNING: {solver_name} final shape {p_amp.shape} != (141, 141, 241)")

    return {"p_amp": p_amp, "p_phase": p_phase}


def load_skull_mask():
    """Load skull and brain masks + coordinate vectors."""
    fpath = DATA_DIR / "SKULL-MAPS" / "skull_mask_bm7_dx_0.5mm.mat"
    with h5py.File(fpath, 'r') as f:
        brain_mask = f['brain_mask'][()].astype(bool)  # (141, 141, 241)
        skull_mask = f['skull_mask'][()].astype(bool)
        xi = f['xi'][()].flatten()  # axial, 241 pts, [0, 120] mm
        yi = f['yi'][()].flatten()  # lateral, 141 pts, [-35, 35] mm
        zi = f['zi'][()].flatten()  # lateral, 141 pts, [-35, 35] mm
        dx = f['dx'][()].flatten()[0]  # 0.5 mm
    return {
        "brain_mask": brain_mask,
        "skull_mask": skull_mask,
        "xi": xi,  # axial coords in mm
        "yi": yi,  # lateral y coords in mm
        "zi": zi,  # lateral z coords in mm
        "dx": dx,
    }


def compute_fwhm_1d(profile, coords_mm):
    """Compute FWHM from a 1D pressure profile."""
    half_max = np.max(profile) / 2.0
    above = profile >= half_max
    if not np.any(above):
        return np.nan
    indices = np.where(above)[0]
    # Use first and last crossing
    return coords_mm[indices[-1]] - coords_mm[indices[0]]


def compute_metrics(p_amp, ref_amp, brain_mask, xi, yi, zi):
    """Compute all comparison metrics between a solver and reference."""
    dx = xi[1] - xi[0] if len(xi) > 1 else 0.5  # mm

    # Mask to brain region
    brain_p = p_amp[brain_mask]
    brain_ref = ref_amp[brain_mask]

    # Normalize both to reference peak in brain
    ref_peak_brain = np.max(brain_ref)

    # Peak pressure and location (in full volume)
    peak_val = np.max(p_amp)
    peak_idx = np.unravel_index(np.argmax(p_amp), p_amp.shape)
    peak_loc_mm = np.array([yi[peak_idx[0]], zi[peak_idx[1]], xi[peak_idx[2]]])

    # Peak in brain only
    brain_peak_val = np.max(brain_p)
    brain_peak_idx_flat = np.argmax(p_amp * brain_mask)
    brain_peak_idx = np.unravel_index(brain_peak_idx_flat, p_amp.shape)
    brain_peak_loc_mm = np.array([yi[brain_peak_idx[0]], zi[brain_peak_idx[1]], xi[brain_peak_idx[2]]])

    # Reference peak in brain
    ref_brain_peak_val = np.max(brain_ref)
    ref_brain_peak_idx_flat = np.argmax(ref_amp * brain_mask)
    ref_brain_peak_idx = np.unravel_index(ref_brain_peak_idx_flat, ref_amp.shape)
    ref_brain_peak_loc_mm = np.array([yi[ref_brain_peak_idx[0]], zi[ref_brain_peak_idx[1]], xi[ref_brain_peak_idx[2]]])

    # Focal pressure difference (%) relative to ref, brain only
    focal_pressure_diff_pct = 100.0 * (brain_peak_val - ref_brain_peak_val) / ref_brain_peak_val

    # Focal position offset (mm), brain only
    focal_position_offset_mm = np.linalg.norm(brain_peak_loc_mm - ref_brain_peak_loc_mm)

    # L2 and L-inf errors in brain, normalized by ref peak
    diff = brain_p - brain_ref
    l2_error = np.sqrt(np.mean(diff**2)) / ref_peak_brain
    linf_error = np.max(np.abs(diff)) / ref_peak_brain

    # Axial profile through brain peak (of the solver)
    axial_profile = p_amp[brain_peak_idx[0], brain_peak_idx[1], :]
    fwhm_axial = compute_fwhm_1d(axial_profile, xi)

    # Lateral profiles through brain peak
    lat_y_profile = p_amp[:, brain_peak_idx[1], brain_peak_idx[2]]
    fwhm_lat_y = compute_fwhm_1d(lat_y_profile, yi)

    lat_z_profile = p_amp[brain_peak_idx[0], :, brain_peak_idx[2]]
    fwhm_lat_z = compute_fwhm_1d(lat_z_profile, zi)

    return {
        "peak_pressure_global": peak_val,
        "peak_loc_global_mm": peak_loc_mm,
        "peak_pressure_brain": brain_peak_val,
        "peak_loc_brain_mm": brain_peak_loc_mm,
        "focal_pressure_diff_pct": focal_pressure_diff_pct,
        "focal_position_offset_mm": focal_position_offset_mm,
        "l2_error": l2_error,
        "linf_error": linf_error,
        "fwhm_axial_mm": fwhm_axial,
        "fwhm_lat_y_mm": fwhm_lat_y,
        "fwhm_lat_z_mm": fwhm_lat_z,
    }


# ============================================================
# Main analysis
# ============================================================

def analyze_scenario(scenario="SC1"):
    print(f"\n{'='*80}")
    print(f"  BM7-{scenario} ANALYSIS")
    print(f"{'='*80}")

    # Load skull/brain masks
    masks = load_skull_mask()
    brain_mask = masks["brain_mask"]
    skull_mask = masks["skull_mask"]
    xi = masks["xi"]
    yi = masks["yi"]
    zi = masks["zi"]
    dx = masks["dx"]

    print(f"\nSkull mask: {skull_mask.shape}, {skull_mask.sum()} voxels")
    print(f"Brain mask: {brain_mask.shape}, {brain_mask.sum()} voxels")
    print(f"Grid: yi={len(yi)} [{yi.min():.1f}, {yi.max():.1f}] mm, "
          f"zi={len(zi)} [{zi.min():.1f}, {zi.max():.1f}] mm, "
          f"xi={len(xi)} [{xi.min():.1f}, {xi.max():.1f}] mm")
    print(f"Spacing: {dx} mm")

    # Load k-Wave reference first
    print(f"\n--- Loading k-Wave reference (BM7-{scenario}) ---")
    ref_data = load_p_amp("KWAVE", scenario)
    if ref_data is None:
        print("ERROR: Could not load k-Wave reference!")
        return
    ref_amp = ref_data["p_amp"]
    ref_phase = ref_data["p_phase"]

    print(f"  p_amp shape: {ref_amp.shape}, dtype: {ref_amp.dtype}")
    print(f"  p_amp range: [{ref_amp.min():.2f}, {ref_amp.max():.2f}] Pa")
    if ref_phase is not None:
        print(f"  p_phase shape: {ref_phase.shape}, range: [{ref_phase.min():.4f}, {ref_phase.max():.4f}] rad")

    # Global peak
    peak_idx = np.unravel_index(np.argmax(ref_amp), ref_amp.shape)
    peak_val = ref_amp[peak_idx]
    peak_loc = (yi[peak_idx[0]], zi[peak_idx[1]], xi[peak_idx[2]])
    print(f"\n  Global peak: {peak_val:.2f} Pa at (y={peak_loc[0]:.1f}, z={peak_loc[1]:.1f}, x={peak_loc[2]:.1f}) mm")

    # Brain-only peak
    brain_ref = ref_amp * brain_mask
    brain_peak_idx = np.unravel_index(np.argmax(brain_ref), brain_ref.shape)
    brain_peak_val = ref_amp[brain_peak_idx]
    brain_peak_loc = (yi[brain_peak_idx[0]], zi[brain_peak_idx[1]], xi[brain_peak_idx[2]])
    print(f"  Brain peak:  {brain_peak_val:.2f} Pa at (y={brain_peak_loc[0]:.1f}, z={brain_peak_loc[1]:.1f}, x={brain_peak_loc[2]:.1f}) mm")

    # FWHM of reference
    axial_profile = ref_amp[brain_peak_idx[0], brain_peak_idx[1], :]
    fwhm_ax = compute_fwhm_1d(axial_profile, xi)
    lat_y_profile = ref_amp[:, brain_peak_idx[1], brain_peak_idx[2]]
    fwhm_ly = compute_fwhm_1d(lat_y_profile, yi)
    lat_z_profile = ref_amp[brain_peak_idx[0], :, brain_peak_idx[2]]
    fwhm_lz = compute_fwhm_1d(lat_z_profile, zi)
    print(f"  FWHM (axial):     {fwhm_ax:.2f} mm")
    print(f"  FWHM (lateral y): {fwhm_ly:.2f} mm")
    print(f"  FWHM (lateral z): {fwhm_lz:.2f} mm")

    # Mean pressure in brain
    brain_mean = np.mean(ref_amp[brain_mask])
    brain_std = np.std(ref_amp[brain_mask])
    print(f"  Brain mean pressure: {brain_mean:.2f} +/- {brain_std:.2f} Pa")

    # Load all solvers and compute metrics
    solvers = ["BABELVISCOFDTD", "FULLWAVE", "GMFDTD", "HAS", "JWAVE",
               "KWAVE", "MSOUND", "OPTIMUS", "SALVUS", "SIM4LIFE", "STRIDE"]

    results = {}
    for solver in solvers:
        print(f"\n--- Loading {solver} ---")
        data = load_p_amp(solver, scenario)
        if data is None:
            print(f"  SKIPPED (file not found)")
            continue

        p_amp = data["p_amp"]
        print(f"  p_amp shape: {p_amp.shape}, range: [{p_amp.min():.2f}, {p_amp.max():.2f}] Pa")

        if solver == "KWAVE":
            # Self-comparison (should be zero error)
            metrics = compute_metrics(p_amp, ref_amp, brain_mask, xi, yi, zi)
            print(f"  Brain peak: {metrics['peak_pressure_brain']:.2f} Pa at {metrics['peak_loc_brain_mm']}")
            print(f"  (Self-comparison, errors should be ~0)")
        else:
            metrics = compute_metrics(p_amp, ref_amp, brain_mask, xi, yi, zi)
            print(f"  Brain peak: {metrics['peak_pressure_brain']:.2f} Pa at {metrics['peak_loc_brain_mm']}")
            print(f"  Focal pressure diff: {metrics['focal_pressure_diff_pct']:+.2f}%")
            print(f"  Focal position offset: {metrics['focal_position_offset_mm']:.2f} mm")
            print(f"  L2 error (brain): {metrics['l2_error']:.4f}")
            print(f"  L-inf error (brain): {metrics['linf_error']:.4f}")

        results[solver] = metrics

    # Print comparison table
    print(f"\n\n{'='*120}")
    print(f"  COMPARISON TABLE: BM7-{scenario}")
    print(f"{'='*120}")
    print(f"{'Solver':<18} {'Brain Peak':>11} {'Peak Y':>7} {'Peak Z':>7} {'Peak X':>7} "
          f"{'dP(%)':>8} {'dPos(mm)':>9} {'L2':>8} {'L-inf':>8} "
          f"{'FWHM-ax':>8} {'FWHM-ly':>8} {'FWHM-lz':>8}")
    print(f"{'':.<18} {'(Pa)':>11} {'(mm)':>7} {'(mm)':>7} {'(mm)':>7} "
          f"{'':>8} {'':>9} {'':>8} {'':>8} "
          f"{'(mm)':>8} {'(mm)':>8} {'(mm)':>8}")
    print("-" * 120)

    for solver in solvers:
        if solver not in results:
            print(f"{solver:<18} {'N/A':>11}")
            continue
        m = results[solver]
        is_ref = " *" if solver == "KWAVE" else ""
        print(f"{solver + is_ref:<18} "
              f"{m['peak_pressure_brain']:>11.2f} "
              f"{m['peak_loc_brain_mm'][0]:>7.1f} "
              f"{m['peak_loc_brain_mm'][1]:>7.1f} "
              f"{m['peak_loc_brain_mm'][2]:>7.1f} "
              f"{m['focal_pressure_diff_pct']:>+8.2f} "
              f"{m['focal_position_offset_mm']:>9.2f} "
              f"{m['l2_error']:>8.4f} "
              f"{m['linf_error']:>8.4f} "
              f"{m['fwhm_axial_mm']:>8.2f} "
              f"{m['fwhm_lat_y_mm']:>8.2f} "
              f"{m['fwhm_lat_z_mm']:>8.2f}")

    print(f"\n  * = k-Wave reference (self-comparison)")
    print(f"  dP(%) = focal pressure difference vs k-Wave reference")
    print(f"  dPos(mm) = focal position offset vs k-Wave reference")
    print(f"  L2, L-inf = normalized errors in brain region")

    return results


def print_kwave_metadata():
    """Print detailed k-Wave metadata."""
    print("\n" + "=" * 80)
    print("  k-WAVE BM7 SIMULATION METADATA")
    print("=" * 80)

    fpath = DATA_DIR / "KWAVE" / "PH1-BM7-SC1_KWAVE.mat"
    with h5py.File(fpath, 'r') as f:
        gs = f['general_settings']
        ss = f['simulation_settings']

        print("\n--- Source Configuration ---")
        print(f"  Source frequency (f0): {gs['source_f0'][()].flatten()[0]/1e3:.0f} kHz")
        print(f"  Source magnitude: {gs['source_mag'][()].flatten()[0]:.0f} Pa")
        print(f"  Bowl ROC: {gs['bowl_roc'][()].flatten()[0]*1e3:.1f} mm")
        print(f"  Bowl diameter: {gs['bowl_diameter'][()].flatten()[0]*1e3:.1f} mm")
        print(f"  Disc diameter: {gs['disc_diameter'][()].flatten()[0]*1e3:.1f} mm")

        f0 = gs['source_f0'][()].flatten()[0]
        wavelength_water = gs['water_cp'][()].flatten()[0] / f0 * 1000  # mm
        print(f"  Wavelength in water: {wavelength_water:.2f} mm")

        print("\n--- Medium Properties ---")
        for medium in ['water', 'skin', 'cortical', 'trabecular', 'brain']:
            cp = gs[f'{medium}_cp'][()].flatten()[0]
            cs = gs[f'{medium}_cs'][()].flatten()[0]
            rho = gs[f'{medium}_rho'][()].flatten()[0]
            ap = gs[f'{medium}_ap'][()].flatten()[0]
            a_s = gs[f'{medium}_as'][()].flatten()[0]
            print(f"  {medium:>12s}: cp={cp:.0f} m/s, cs={cs:.0f} m/s, rho={rho:.0f} kg/m3, "
                  f"ap={ap:.1f} Np/m/MHz, as={a_s:.1f} Np/m/MHz")

        print("\n--- Simulation Grid ---")
        grid_size = ss['grid_size'][()].flatten()
        dx_sim = ss['grid_spacing'][()].flatten()[0]
        ppw = ss['ppw'][()].flatten()[0]
        cfl = ss['cfl'][()].flatten()[0]
        dt = ss['dt'][()].flatten()[0]
        Nt = int(ss['Nt'][()].flatten()[0])
        t_end = ss['t_end'][()].flatten()[0]
        axial = ss['axial_size'][()].flatten()[0]
        lat_y = ss['lateral_size_y'][()].flatten()[0]
        lat_z = ss['lateral_size_z'][()].flatten()[0]
        pml = ss['pml_size'][()].flatten()

        print(f"  Grid size: {int(grid_size[0])} x {int(grid_size[1])} x {int(grid_size[2])}")
        print(f"  Grid spacing: {dx_sim*1e3:.3f} mm ({dx_sim*1e6:.1f} um)")
        print(f"  PPW (points per wavelength): {ppw:.0f}")
        print(f"  CFL: {cfl:.4f}")
        print(f"  dt: {dt:.4e} s")
        print(f"  Nt (time steps): {Nt}")
        print(f"  t_end: {t_end*1e6:.1f} us")
        print(f"  Domain: axial={axial*1e3:.1f} mm, lateral_y={lat_y*1e3:.1f} mm, lateral_z={lat_z*1e3:.1f} mm")
        print(f"  PML size: {int(pml[0])}, {int(pml[1])}, {int(pml[2])}")
        print(f"  Output grid (0.5mm): {141} x {141} x {241}")
        print(f"  Record periods: {ss['record_periods'][()].flatten()[0]:.0f}")

        # Water absorbing layer info
        print(f"\n--- Water Absorbing Layer ---")
        print(f"  water_absorbing_ap: {gs['water_absorbing_ap'][()].flatten()[0]} Np/m/MHz")


if __name__ == "__main__":
    # Print k-Wave metadata
    print_kwave_metadata()

    # Analyze BM7-SC1 (focused bowl)
    results_sc1 = analyze_scenario("SC1")

    # Analyze BM7-SC2 (plane piston)
    results_sc2 = analyze_scenario("SC2")

    print("\n\nDONE.")
