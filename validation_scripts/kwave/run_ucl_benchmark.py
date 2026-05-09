#!/usr/bin/env python3
"""Run k-Wave simulation against the ITRUSST UCL Transcranial Benchmarks (BM7, BM8, BM9).

Uses k-wave-python to simulate the benchmark scenarios defined in:
  Aubry et al., "Benchmark problems for transcranial ultrasound simulation," 2022.

The script loads pre-rasterized skull/brain masks from the UCL benchmark data,
builds the heterogeneous medium, creates the appropriate source (focused bowl
or plane piston), runs the k-Wave FDTD simulation, and saves p_amp/p_phase
output in HDF5 format matching the benchmark reference.

Usage examples:
  # BM7 with focused bowl source, GPU, default 0.5mm grid
  python scripts/run_ucl_benchmark.py --benchmark 7 --source 1 --gpu

  # BM7 with coarser 1mm grid for faster testing
  python scripts/run_ucl_benchmark.py --benchmark 7 --source 1 --gpu --dx 1.0

  # BM8 with plane piston source
  python scripts/run_ucl_benchmark.py --benchmark 8 --source 2 --gpu

Data directory layout expected at DATA_DIR:
  SKULL-MAPS/skull_mask_bm{7,8,9}_dx_{dx}mm.mat  (HDF5)
  KWAVE/PH1-BM{7,8,9}-SC{1,2}_KWAVE.mat          (reference results)
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import pathlib
import sys
import time
from datetime import datetime

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path.home() / "Data/openlifu-validation/ucl-benchmarks/data"

# Material properties from the UCL benchmark specification (Aubry 2022)
# and confirmed from the KWAVE reference HDF5 general_settings.
MATERIALS = {
    "water": {
        "sound_speed": 1500.0,       # m/s
        "density": 1000.0,           # kg/m^3
        "alpha_coeff": 0.0,          # dB/(MHz^y cm)
    },
    "cortical_bone": {
        "sound_speed": 2800.0,       # m/s
        "density": 1850.0,           # kg/m^3
        # The reference file stores alpha at power-law exponent.
        # cortical_ap = 16 dB/cm at 500 kHz in the reference, but the
        # benchmark spec says 4 dB/cm at 500 kHz for the skull bone.
        # The reference value is for the power-law formulation:
        #   alpha = alpha_coeff * f^alpha_power [dB/(MHz^y cm)]
        # At 500 kHz with alpha_power ~ 2: alpha_coeff * 0.5^2 = 4 => alpha_coeff = 16
        # This matches cortical_ap = 16 in the reference.
        "alpha_coeff": 16.0,         # dB/(MHz^y cm)
    },
    "brain": {
        "sound_speed": 1560.0,       # m/s
        "density": 1040.0,           # kg/m^3
        # brain_ap = 1.2 in the reference; at 500 kHz with alpha_power ~ 2:
        #   1.2 * 0.5^2 = 0.3 dB/cm, matching the benchmark spec.
        "alpha_coeff": 1.2,          # dB/(MHz^y cm)
    },
}

# Transducer definitions
SOURCES = {
    1: {
        "name": "Focused Bowl (SC1)",
        "type": "bowl",
        "diameter": 0.064,           # 64 mm
        "radius_of_curvature": 0.064,  # 64 mm ROC
        "frequency": 500e3,          # 500 kHz
        "source_pressure": 60e3,     # 60 kPa source magnitude
    },
    2: {
        "name": "Plane Piston (SC2)",
        "type": "disc",
        "diameter": 0.020,           # 20 mm
        "frequency": 500e3,          # 500 kHz
        "source_pressure": 60e3,     # 60 kPa source magnitude
    },
}

# Benchmark output grid specifications (physical domain in mm).
# These define the comparison grid for the output; the computational grid
# is larger due to PML and source positioning requirements.
BENCHMARK_GRIDS = {
    7: {
        "x_range_mm": (0, 120),
        "y_range_mm": (-35, 35),
        "z_range_mm": (-35, 35),
        "output_shape": (241, 141, 141),  # (nx, ny, nz) at 0.5mm
    },
    8: {
        "x_range_mm": (0, 225),
        "y_range_mm": (-85, 85),
        "z_range_mm": (-95, 95),
        "output_shape": (451, 341, 381),
    },
    9: {
        "x_range_mm": (0, 212),
        "y_range_mm": (-112, 112),
        "z_range_mm": (-92, 92),
        "output_shape": (425, 449, 369),
    },
}

# Alpha power for the power-law absorption model.
# The benchmark uses alpha_power = 2 (quadratic frequency dependence for bone).
ALPHA_POWER = 2.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_skull_mask(benchmark: int, dx_mm: float) -> dict:
    """Load pre-rasterized skull and brain masks from HDF5.

    Returns arrays in (x, y, z) order after transposing from the file's
    (z, y, x) storage order. Also returns coordinate vectors xi, yi, zi
    (in mm) and grid spacing dx (in mm).
    """
    dx_str = f"{dx_mm:g}"
    fname = f"skull_mask_bm{benchmark}_dx_{dx_str}mm.mat"
    path = DATA_DIR / "SKULL-MAPS" / fname
    if not path.exists():
        raise FileNotFoundError(f"Skull mask not found: {path}")

    with h5py.File(str(path), "r") as f:
        result = {}
        for key in ["skull_mask", "brain_mask"]:
            if key in f:
                arr = np.array(f[key])
                # HDF5/MATLAB storage is (z, y, x); transpose to (x, y, z)
                result[key] = np.transpose(arr, (2, 1, 0))
        for key in ["xi", "yi", "zi", "dx"]:
            if key in f:
                result[key] = np.array(f[key]).flatten()

    logger.info(
        "Loaded skull mask: BM%d, dx=%.2f mm, shape=%s, "
        "skull voxels=%d, brain voxels=%d",
        benchmark, dx_mm,
        result["skull_mask"].shape,
        int((result["skull_mask"] > 0).sum()),
        int((result["brain_mask"] > 0).sum()),
    )
    return result


def load_reference_settings(benchmark: int, source: int) -> dict:
    """Load general and simulation settings from the k-Wave reference file."""
    fname = f"PH1-BM{benchmark}-SC{source}_KWAVE.mat"
    path = DATA_DIR / "KWAVE" / fname
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    settings = {}
    with h5py.File(str(path), "r") as f:
        if "general_settings" in f:
            gs = f["general_settings"]
            for key in gs.keys():
                settings[f"gs_{key}"] = float(np.array(gs[key]).flat[0])
        if "simulation_settings" in f:
            ss = f["simulation_settings"]
            for key in ss.keys():
                val = np.array(ss[key])
                if val.size == 1:
                    settings[f"ss_{key}"] = float(val.flat[0])
                else:
                    settings[f"ss_{key}"] = val.flatten()
    return settings


# ---------------------------------------------------------------------------
# Medium construction
# ---------------------------------------------------------------------------

def build_medium_arrays(
    skull_mask: np.ndarray,
    brain_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D arrays for sound_speed, density, and alpha_coeff.

    Regions:
      - skull_mask == 1: cortical bone properties
      - brain_mask == 1 (and skull_mask == 0): brain properties
      - everything else: water properties

    Returns:
        Tuple of (sound_speed, density, alpha_coeff) arrays, all same shape
        as the input masks.
    """
    shape = skull_mask.shape
    sound_speed = np.full(shape, MATERIALS["water"]["sound_speed"], dtype=np.float32)
    density = np.full(shape, MATERIALS["water"]["density"], dtype=np.float32)
    alpha_coeff = np.full(shape, MATERIALS["water"]["alpha_coeff"], dtype=np.float32)

    # Apply brain properties first (brain region includes non-skull interior)
    brain_only = (brain_mask > 0) & (skull_mask == 0)
    sound_speed[brain_only] = MATERIALS["brain"]["sound_speed"]
    density[brain_only] = MATERIALS["brain"]["density"]
    alpha_coeff[brain_only] = MATERIALS["brain"]["alpha_coeff"]

    # Apply skull/cortical bone properties (overwrites brain where skull exists)
    skull = skull_mask > 0
    sound_speed[skull] = MATERIALS["cortical_bone"]["sound_speed"]
    density[skull] = MATERIALS["cortical_bone"]["density"]
    alpha_coeff[skull] = MATERIALS["cortical_bone"]["alpha_coeff"]

    logger.info(
        "Medium arrays built: shape=%s, c_range=[%.0f, %.0f] m/s, "
        "rho_range=[%.0f, %.0f] kg/m^3",
        shape, sound_speed.min(), sound_speed.max(),
        density.min(), density.max(),
    )
    return sound_speed, density, alpha_coeff


# ---------------------------------------------------------------------------
# Grid and source setup
# ---------------------------------------------------------------------------

def compute_grid_params(
    benchmark: int,
    dx_m: float,
    pml_size: int,
) -> dict:
    """Compute the computational grid parameters.

    The grid encompasses the benchmark output domain plus PML layers.
    The transducer is placed at the -x edge of the grid (after PML).
    The coordinate system places x=0 at the first non-PML grid point,
    matching the benchmark convention.

    Returns dict with Nx, Ny, Nz (total including PML), dx, and offsets.
    """
    bm = BENCHMARK_GRIDS[benchmark]
    dx_mm = dx_m * 1e3

    # Output domain extents in mm
    x_min, x_max = bm["x_range_mm"]
    y_min, y_max = bm["y_range_mm"]
    z_min, z_max = bm["z_range_mm"]

    # Number of grid points in the output domain
    out_nx = int(round((x_max - x_min) / dx_mm)) + 1
    out_ny = int(round((y_max - y_min) / dx_mm)) + 1
    out_nz = int(round((z_max - z_min) / dx_mm)) + 1

    # Total grid with PML on each side
    # The source is placed at the -x boundary (inside PML region or just outside)
    # so we need extra space on the -x side for the source bowl.
    # Add some margin before the output domain for the source geometry.
    # The transducer is at x=0 in the benchmark coordinate system, so we need
    # some negative-x space for the bowl's rear surface.
    # For the focused bowl (64mm ROC), the rear-to-front depth is:
    #   depth = ROC - sqrt(ROC^2 - (D/2)^2) = 64 - sqrt(64^2 - 32^2) = 64 - 55.4 = 8.6 mm
    # We add ~20mm margin before x=0 for the source.
    source_margin_mm = 20.0
    source_margin_pts = int(np.ceil(source_margin_mm / dx_mm))

    # Total grid size (output domain + source margin + PML on each side)
    Nx = out_nx + source_margin_pts + 2 * pml_size
    Ny = out_ny + 2 * pml_size
    Nz = out_nz + 2 * pml_size

    # The output domain starts at grid index (pml_size + source_margin_pts)
    # in the x-direction and at pml_size in y and z.
    output_origin = (pml_size + source_margin_pts, pml_size, pml_size)

    return {
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "dx": dx_m,
        "out_nx": out_nx, "out_ny": out_ny, "out_nz": out_nz,
        "output_origin": output_origin,
        "source_margin_pts": source_margin_pts,
        "pml_size": pml_size,
        "x_range_mm": (x_min, x_max),
        "y_range_mm": (y_min, y_max),
        "z_range_mm": (z_min, z_max),
    }


def build_full_medium(
    grid_params: dict,
    skull_data: dict,
    dx_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build medium arrays sized for the full computational grid.

    The skull/brain masks are placed into the correct position within the
    larger computational grid (which includes PML and source margin).
    """
    Nx = grid_params["Nx"]
    Ny = grid_params["Ny"]
    Nz = grid_params["Nz"]
    ox, oy, oz = grid_params["output_origin"]

    # Build output-domain-sized medium from masks
    skull_mask = skull_data["skull_mask"]
    brain_mask = skull_data["brain_mask"]

    # The masks may need to be resampled if dx differs from the mask resolution.
    mask_dx_mm = float(skull_data["dx"][0])
    target_dx_mm = dx_m * 1e3

    if abs(mask_dx_mm - target_dx_mm) > 0.001:
        logger.info(
            "Resampling masks from %.3f mm to %.3f mm",
            mask_dx_mm, target_dx_mm,
        )
        from scipy.ndimage import zoom
        scale = mask_dx_mm / target_dx_mm
        skull_mask = (zoom(skull_mask.astype(np.float32), scale, order=0) > 0.5).astype(np.uint8)
        brain_mask = (zoom(brain_mask.astype(np.float32), scale, order=0) > 0.5).astype(np.uint8)
        logger.info("Resampled mask shape: %s", skull_mask.shape)

    # Build medium arrays for the output domain
    c_out, rho_out, alpha_out = build_medium_arrays(skull_mask, brain_mask)

    # Place into full grid (initialize with water)
    sound_speed = np.full((Nx, Ny, Nz), MATERIALS["water"]["sound_speed"], dtype=np.float32)
    density = np.full((Nx, Ny, Nz), MATERIALS["water"]["density"], dtype=np.float32)
    alpha_coeff = np.full((Nx, Ny, Nz), MATERIALS["water"]["alpha_coeff"], dtype=np.float32)

    # Determine how much of the output domain fits
    sx = min(c_out.shape[0], Nx - ox)
    sy = min(c_out.shape[1], Ny - oy)
    sz = min(c_out.shape[2], Nz - oz)

    sound_speed[ox:ox+sx, oy:oy+sy, oz:oz+sz] = c_out[:sx, :sy, :sz]
    density[ox:ox+sx, oy:oy+sy, oz:oz+sz] = rho_out[:sx, :sy, :sz]
    alpha_coeff[ox:ox+sx, oy:oy+sy, oz:oz+sz] = alpha_out[:sx, :sy, :sz]

    logger.info(
        "Full medium grid: (%d, %d, %d), output placed at offset (%d, %d, %d), "
        "output region size (%d, %d, %d)",
        Nx, Ny, Nz, ox, oy, oz, sx, sy, sz,
    )

    return sound_speed, density, alpha_coeff


def create_source(
    source_cfg: dict,
    grid_params: dict,
    kgrid,
    dt: float,
    Nt: int,
) -> "kSource":
    """Create a k-wave source for the benchmark transducer.

    The transducer is positioned at the -x edge of the output domain
    (x=0 in benchmark coordinates), centered in y and z, pointing in
    the +x direction.

    For the focused bowl, the geometric focus is at (ROC, 0, 0) in
    benchmark coordinates, which maps to (ROC + output_origin_x * dx)
    in grid coordinates.

    Returns a kSource with p_mask and p set for CW excitation.
    """
    from kwave.utils.kwave_array import kWaveArray
    from kwave.ksource import kSource

    dx = grid_params["dx"]
    Nx = grid_params["Nx"]
    Ny = grid_params["Ny"]
    Nz = grid_params["Nz"]
    ox = grid_params["output_origin"][0]

    # k-wave grid coordinates are centered at the grid center.
    # Grid center is at (Nx/2 * dx, Ny/2 * dy, Nz/2 * dz) from the grid origin.
    # We need the transducer position in these centered coordinates.

    # Benchmark x=0 corresponds to grid index ox.
    # In k-wave centered coordinates, grid index i maps to x = (i - Nx/2) * dx
    # So benchmark x=0 maps to kwave_x = (ox - Nx/2) * dx.
    # The bowl rear surface is at benchmark x=0.
    bowl_center_x = (ox - Nx / 2) * dx

    # y and z center of the benchmark domain corresponds to the grid center
    # (oy maps to benchmark y_min, and the transducer is at y=0, z=0).
    # In kwave coordinates, y=0 and z=0 are at the grid center, which IS
    # the benchmark y=0 and z=0 (since oy is the offset for y_min).
    bowl_center_y = 0.0
    bowl_center_z = 0.0

    # Create kWaveArray
    karray = kWaveArray(
        bli_tolerance=0.1,
        upsampling_rate=10,
        single_precision=True,
    )

    if source_cfg["type"] == "bowl":
        roc = source_cfg["radius_of_curvature"]
        diameter = source_cfg["diameter"]
        # Bowl position: center of rear surface
        position = [bowl_center_x, bowl_center_y, bowl_center_z]
        # Focus position: along +x axis at distance ROC from bowl center
        focus_pos = [bowl_center_x + roc, bowl_center_y, bowl_center_z]
        karray.add_bowl_element(position, roc, diameter, focus_pos)
        logger.info(
            "Bowl source: pos=(%.4f, %.4f, %.4f) m, focus=(%.4f, %.4f, %.4f) m, "
            "ROC=%.3f m, diameter=%.3f m",
            *position, *focus_pos, roc, diameter,
        )

    elif source_cfg["type"] == "disc":
        diameter = source_cfg["diameter"]
        # Disc position: center of disc surface at x=0
        position = [bowl_center_x, bowl_center_y, bowl_center_z]
        # Focus position: along +x axis (disc is flat, focus at infinity,
        # but kWaveArray needs a direction vector)
        focus_pos = [bowl_center_x + 1.0, bowl_center_y, bowl_center_z]
        karray.add_disc_element(position, diameter, focus_pos)
        logger.info(
            "Disc source: pos=(%.4f, %.4f, %.4f) m, diameter=%.3f m",
            *position, diameter,
        )

    # Get binary mask and distributed source signal.
    # For CW excitation, we use a continuous sinusoid for the full simulation.
    freq = source_cfg["frequency"]
    amplitude = source_cfg["source_pressure"]

    # CW source signal: single sinusoid for the entire simulation duration
    t_array = np.arange(Nt) * dt
    source_signal = amplitude * np.sin(2 * np.pi * freq * t_array)
    # kWaveArray expects (n_elements, n_timesteps)
    source_signal = source_signal.reshape(1, -1)

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, source_signal)

    n_source_pts = int(source.p_mask.sum())
    logger.info(
        "Source created: %d source grid points, %d timesteps, freq=%.0f kHz",
        n_source_pts, Nt, freq / 1e3,
    )

    return source


# ---------------------------------------------------------------------------
# Steady-state extraction
# ---------------------------------------------------------------------------

def extract_steady_state(
    p_timeseries: np.ndarray,
    dt: float,
    freq: float,
    n_periods: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract steady-state amplitude and phase from the last n_periods of CW data.

    Uses Fourier analysis at the driving frequency to extract the complex
    amplitude at each sensor point from the last n_periods of the time series.

    Args:
        p_timeseries: Pressure time series, shape (n_timesteps, n_sensor_points)
            or (n_timesteps,) for a single point.
        dt: Time step in seconds.
        freq: Driving frequency in Hz.
        n_periods: Number of periods to use from the end of the signal.

    Returns:
        Tuple of (p_amp, p_phase) arrays, each with shape (n_sensor_points,).
    """
    period = 1.0 / freq
    n_samples_per_period = int(round(period / dt))
    n_samples = n_periods * n_samples_per_period

    if p_timeseries.ndim == 1:
        p_timeseries = p_timeseries.reshape(-1, 1)

    n_total = p_timeseries.shape[0]
    n_pts = p_timeseries.shape[1]
    if n_samples > n_total:
        logger.warning(
            "Requested %d samples for steady-state extraction but only %d available. "
            "Using all available samples.",
            n_samples, n_total,
        )
        n_samples = n_total

    # Take the last n_samples
    p_end = p_timeseries[-n_samples:, :]

    # --- Diagnostics: dtype, shape, memory estimate ---
    p_end_f32 = p_end.astype(np.float32, copy=False)
    mem_p_end = p_end_f32.nbytes / (1024**3)
    mem_exp = n_samples * 8 / (1024**3)  # complex64 = 8 bytes
    mem_broadcast = n_samples * n_pts * 8 / (1024**3)  # complex64 intermediate
    logger.info(
        "extract_steady_state: p_end shape=%s dtype=%s (%.2f GB), "
        "exp_term complex64 (%.4f GB), broadcast product complex64 (%.2f GB)",
        p_end_f32.shape, p_end_f32.dtype, mem_p_end,
        mem_exp, mem_broadcast,
    )

    # --- Check raw data for NaN/Inf before extraction ---
    n_nan = np.count_nonzero(np.isnan(p_end_f32))
    n_inf = np.count_nonzero(np.isinf(p_end_f32))
    if n_nan > 0 or n_inf > 0:
        logger.warning(
            "Raw p_end contains %d NaN and %d Inf values out of %d total elements",
            n_nan, n_inf, p_end_f32.size,
        )

    # Compute the Fourier coefficient at the driving frequency.
    # Use complex64 (not complex128) to halve peak memory for large sensor counts.
    t = np.arange(n_samples, dtype=np.float32) * np.float32(dt)
    # Complex exponential at the driving frequency (complex64)
    exp_term = np.exp(
        np.float32(-2.0) * np.complex64(1j) * np.float32(np.pi) * np.float32(freq) * t
    )  # shape (n_samples,), dtype complex64

    # DFT at the target frequency: sum(p * exp(-2j*pi*f*t)) * dt / T
    # Normalized to give the complex amplitude.
    # Accumulate in chunks if sensor count is very large to avoid a single
    # (n_samples, n_pts) complex64 temporary.
    T = n_samples * dt
    scale = np.complex64((2.0 / T) * dt)

    CHUNK_PTS = max(1, int(2e9 / (n_samples * 8)))  # ~2 GB per chunk
    if n_pts > CHUNK_PTS:
        logger.info(
            "Chunked DFT: %d sensor points in chunks of %d (%.1f GB per chunk)",
            n_pts, CHUNK_PTS, CHUNK_PTS * n_samples * 8 / 1e9,
        )
        coeffs = np.empty(n_pts, dtype=np.complex64)
        for i0 in range(0, n_pts, CHUNK_PTS):
            i1 = min(i0 + CHUNK_PTS, n_pts)
            chunk = p_end_f32[:, i0:i1]  # (n_samples, chunk_size) float32
            coeffs[i0:i1] = scale * np.sum(
                chunk * exp_term[:, np.newaxis], axis=0,
            )
    else:
        coeffs = scale * np.sum(
            p_end_f32 * exp_term[:, np.newaxis], axis=0,
        )  # shape (n_pts,), dtype complex64

    p_amp = np.abs(coeffs).astype(np.float32)
    p_phase = np.angle(coeffs).astype(np.float32)

    # --- Fallback: if Fourier extraction produced NaN, use peak-to-peak ---
    n_nan_out = np.count_nonzero(np.isnan(p_amp))
    if n_nan_out > 0:
        logger.warning(
            "Fourier extraction produced %d NaN amplitudes out of %d points. "
            "Falling back to peak-to-peak amplitude for affected points.",
            n_nan_out, n_pts,
        )
        nan_mask = np.isnan(p_amp)
        # Compute peak-to-peak for columns where Fourier failed
        p_max = np.max(p_end_f32[:, nan_mask], axis=0)
        p_min = np.min(p_end_f32[:, nan_mask], axis=0)
        p_amp[nan_mask] = (p_max - p_min) / 2.0
        p_phase[nan_mask] = 0.0
        n_still_nan = np.count_nonzero(np.isnan(p_amp))
        if n_still_nan > 0:
            logger.error(
                "Peak-to-peak fallback still has %d NaN values", n_still_nan,
            )
        else:
            logger.info("Peak-to-peak fallback resolved all NaN amplitudes.")

    logger.info(
        "extract_steady_state result: p_amp range=[%.4g, %.4g], "
        "p_phase range=[%.4f, %.4f], NaN count=%d",
        p_amp.min(), p_amp.max(),
        p_phase.min(), p_phase.max(),
        np.count_nonzero(np.isnan(p_amp)),
    )

    return p_amp, p_phase


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark simulation."""
    benchmark = args.benchmark
    source_id = args.source
    use_gpu = args.gpu
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cfg = SOURCES[source_id]
    freq = source_cfg["frequency"]
    wavelength_water = MATERIALS["water"]["sound_speed"] / freq  # 3mm at 500kHz

    # Determine grid spacing
    if args.dx is not None:
        dx_mm = args.dx
    else:
        dx_mm = 0.5  # Default: 0.5mm (6 PPW in water at 500 kHz)
    dx_m = dx_mm * 1e-3

    ppw = wavelength_water / dx_m
    logger.info(
        "Grid spacing: %.3f mm (%.1f points per wavelength in water)",
        dx_mm, ppw,
    )

    # PML size
    pml_size = args.pml_size
    logger.info("PML size: %d grid points", pml_size)

    # CFL number
    cfl = args.cfl
    logger.info("CFL number: %.3f", cfl)

    # ----- Load skull mask -----
    # Find the best available mask resolution
    available_dx = []
    for f in (DATA_DIR / "SKULL-MAPS").glob(f"skull_mask_bm{benchmark}_dx_*mm.mat"):
        dx_str = f.stem.split("dx_")[1].replace("mm", "")
        try:
            available_dx.append(float(dx_str))
        except ValueError:
            pass

    # Pick the mask closest to our target dx (prefer exact match, then closest)
    if dx_mm in available_dx:
        mask_dx = dx_mm
    else:
        # Use the finest available mask and let build_full_medium resample
        mask_dx = min(available_dx) if available_dx else dx_mm

    logger.info(
        "Loading skull mask at %.3f mm resolution (target grid: %.3f mm)",
        mask_dx, dx_mm,
    )
    skull_data = load_skull_mask(benchmark, mask_dx)

    # ----- Load reference settings for comparison -----
    ref_settings = load_reference_settings(benchmark, source_id)
    logger.info(
        "Reference simulation used: dx=%.4f mm, grid=(%s), Nt=%.0f, dt=%.2e s",
        ref_settings.get("ss_grid_spacing", 0) * 1e3,
        ref_settings.get("ss_grid_size", "?"),
        ref_settings.get("ss_Nt", 0),
        ref_settings.get("ss_dt", 0),
    )

    # ----- Compute grid parameters -----
    grid_params = compute_grid_params(benchmark, dx_m, pml_size)
    Nx, Ny, Nz = grid_params["Nx"], grid_params["Ny"], grid_params["Nz"]
    logger.info(
        "Computational grid: (%d, %d, %d) = %.1f M voxels, "
        "dx=%.3f mm, physical size=(%.1f, %.1f, %.1f) mm",
        Nx, Ny, Nz, Nx * Ny * Nz / 1e6,
        dx_mm, Nx * dx_mm, Ny * dx_mm, Nz * dx_mm,
    )

    # ----- Build medium -----
    sound_speed, density, alpha_coeff = build_full_medium(
        grid_params, skull_data, dx_m,
    )

    # ----- Create k-wave grid -----
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    kgrid = kWaveGrid([Nx, Ny, Nz], [dx_m, dx_m, dx_m])

    # Time stepping: use the CFL condition
    c_max = float(sound_speed.max())
    dt = cfl * dx_m / c_max
    logger.info("Time step: %.4e s (CFL=%.3f, c_max=%.0f m/s)", dt, cfl, c_max)

    # Simulation duration: need enough cycles for CW steady state.
    # The wave must traverse the full grid at least once, plus extra cycles
    # for the steady state to develop. For a CW simulation, we need:
    #   t_end >= t_propagation + n_steady_state_periods * period
    # where t_propagation = grid_diagonal / c_min
    c_min = float(sound_speed[sound_speed > 0].min())
    grid_diagonal = np.sqrt((Nx * dx_m) ** 2 + (Ny * dx_m) ** 2 + (Nz * dx_m) ** 2)
    t_propagation = grid_diagonal / c_min
    period = 1.0 / freq

    # Number of steady-state periods to record after propagation
    n_record_periods = args.record_periods
    # Total number of periods including propagation time
    n_propagation_periods = int(np.ceil(t_propagation / period))
    n_total_periods = n_propagation_periods + n_record_periods + 2  # +2 for safety

    t_end = n_total_periods * period
    Nt = int(np.ceil(t_end / dt))
    kgrid.setTime(Nt, dt)

    logger.info(
        "Simulation time: %.2f us (%d timesteps), "
        "%.1f propagation periods + %d record periods",
        t_end * 1e6, Nt, n_propagation_periods, n_record_periods,
    )

    # ----- Build medium object -----
    medium = kWaveMedium(
        sound_speed=sound_speed,
        density=density,
        alpha_coeff=alpha_coeff,
        alpha_power=ALPHA_POWER,
        alpha_mode="no_dispersion",
    )

    # ----- Create source -----
    source = create_source(source_cfg, grid_params, kgrid, dt, Nt)

    # ----- Create sensor -----
    # Record pressure time series at the output grid points only.
    # For memory efficiency with large grids, we record at all points
    # in the output domain using a binary sensor mask.
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    ox, oy, oz = grid_params["output_origin"]
    out_nx = grid_params["out_nx"]
    out_ny = grid_params["out_ny"]
    out_nz = grid_params["out_nz"]

    # Only record the last n_record_periods for steady-state extraction.
    # Use record_start_index to skip the transient.
    n_skip = int((n_propagation_periods + 1) * period / dt)
    record_start = max(1, n_skip)

    if args.record_mode == "time_series":
        # Record full time series at output grid (memory intensive!)
        sensor_mask[ox:ox+out_nx, oy:oy+out_ny, oz:oz+out_nz] = 1
        sensor = kSensor(sensor_mask, record=["p"])
        sensor.record_start_index = record_start
        logger.info(
            "Sensor: %d output voxels, recording from step %d (%.1f us), "
            "mode=time_series",
            int(sensor_mask.sum()), record_start, record_start * dt * 1e6,
        )
    else:
        # Record p_max and p_min at the full grid (for amplitude estimation)
        # This is less memory intensive but gives less accurate phase information.
        sensor_mask[ox:ox+out_nx, oy:oy+out_ny, oz:oz+out_nz] = 1
        sensor = kSensor(sensor_mask, record=["p_max", "p_min"])
        sensor.record_start_index = record_start
        logger.info(
            "Sensor: %d output voxels, recording from step %d (%.1f us), "
            "mode=max_min",
            int(sensor_mask.sum()), record_start, record_start * dt * 1e6,
        )

    # ----- Run simulation -----
    logger.info("=" * 72)
    logger.info("Starting k-Wave simulation...")
    logger.info("  GPU: %s", use_gpu)
    logger.info("  Grid: (%d, %d, %d) at %.3f mm", Nx, Ny, Nz, dx_mm)
    logger.info("  Timesteps: %d, dt=%.2e s, t_end=%.2f us", Nt, dt, t_end * 1e6)
    logger.info("=" * 72)

    t_start = time.time()

    simulation_options = SimulationOptions(
        pml_inside=True,
        pml_size=[pml_size, pml_size, pml_size],
        pml_alpha=args.pml_alpha,
        save_to_disk=True,
        data_cast="single",
        smooth_c0=args.smooth,
        smooth_rho0=args.smooth,
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=use_gpu,
        show_sim_log=True,
    )

    try:
        output = kspaceFirstOrder3D(
            kgrid=kgrid,
            source=source,
            sensor=sensor,
            medium=medium,
            simulation_options=simulation_options,
            execution_options=execution_options,
        )
    finally:
        # Clean up temp files
        for fpath in [simulation_options.input_filename, simulation_options.output_filename]:
            with contextlib.suppress(OSError):
                pathlib.Path(fpath).unlink(missing_ok=True)

    t_elapsed = time.time() - t_start
    logger.info("Simulation complete in %.1f seconds (%.1f minutes)", t_elapsed, t_elapsed / 60)

    # ----- Extract results -----
    n_output_pts = int(sensor_mask.sum())

    if args.record_mode == "time_series":
        # output['p'] has shape (n_timesteps_recorded, n_sensor_points)
        p_raw = output["p"]
        if p_raw.ndim == 1:
            p_raw = p_raw.reshape(-1, 1)
        logger.info("Raw output shape: %s", p_raw.shape)

        # Extract steady-state amplitude and phase
        p_amp_flat, p_phase_flat = extract_steady_state(
            p_raw, dt, freq, n_periods=n_record_periods,
        )

        # Reshape to 3D output grid
        p_amp_3d = np.zeros((out_nx, out_ny, out_nz), dtype=np.float32)
        p_phase_3d = np.zeros((out_nx, out_ny, out_nz), dtype=np.float32)

        # The sensor data is ordered in Fortran (column-major) order of the
        # nonzero voxels in the sensor mask.
        # Since our sensor mask is a rectangular block, we can reshape directly.
        p_amp_3d = p_amp_flat.reshape((out_nx, out_ny, out_nz), order="F")
        p_phase_3d = p_phase_flat.reshape((out_nx, out_ny, out_nz), order="F")

    else:
        # max/min mode: p_amp ~ (p_max - p_min) / 2 for CW signals
        p_max_flat = output["p_max"]
        p_min_flat = output["p_min"]

        # Debug: inspect raw output from k-Wave
        logger.info(
            "DEBUG p_max: dtype=%s, shape=%s, range=[%s, %s], sample=%s",
            p_max_flat.dtype, p_max_flat.shape,
            np.min(p_max_flat), np.max(p_max_flat),
            p_max_flat[:5] if len(p_max_flat) >= 5 else p_max_flat,
        )
        logger.info(
            "DEBUG p_min: dtype=%s, shape=%s, range=[%s, %s], sample=%s",
            p_min_flat.dtype, p_min_flat.shape,
            np.min(p_min_flat), np.max(p_min_flat),
            p_min_flat[:5] if len(p_min_flat) >= 5 else p_min_flat,
        )

        # Cast to float64 before subtraction to avoid float32 overflow
        p_amp_flat = ((p_max_flat.astype(np.float64) - p_min_flat.astype(np.float64)) / 2.0).astype(np.float32)

        # No phase information available in max/min mode
        p_phase_flat = np.zeros_like(p_amp_flat)

        # Reshape to 3D
        p_amp_3d = p_amp_flat.reshape((out_nx, out_ny, out_nz), order="F")
        p_phase_3d = p_phase_flat.reshape((out_nx, out_ny, out_nz), order="F")

    logger.info(
        "Output: p_amp range=[%.1f, %.1f] Pa, peak at voxel %s",
        p_amp_3d.min(), p_amp_3d.max(),
        np.unravel_index(np.argmax(p_amp_3d), p_amp_3d.shape),
    )

    # ----- Save output -----
    # Transpose to (z, y, x) to match the UCL benchmark reference format
    p_amp_save = np.transpose(p_amp_3d, (2, 1, 0))
    p_phase_save = np.transpose(p_phase_3d, (2, 1, 0))

    output_fname = f"PH1-BM{benchmark}-SC{source_id}_OPENLIFU_dx{dx_mm:g}mm.mat"
    output_path = output_dir / output_fname

    with h5py.File(str(output_path), "w") as f:
        f.create_dataset("p_amp", data=p_amp_save, dtype=np.float32)
        f.create_dataset("p_phase", data=p_phase_save, dtype=np.float32)

        # Save simulation metadata
        gs = f.create_group("general_settings")
        gs.create_dataset("source_f0", data=np.array([[freq]]))
        gs.create_dataset("source_mag", data=np.array([[source_cfg["source_pressure"]]]))
        gs.create_dataset("water_cp", data=np.array([[MATERIALS["water"]["sound_speed"]]]))
        gs.create_dataset("water_rho", data=np.array([[MATERIALS["water"]["density"]]]))
        gs.create_dataset("cortical_cp", data=np.array([[MATERIALS["cortical_bone"]["sound_speed"]]]))
        gs.create_dataset("cortical_rho", data=np.array([[MATERIALS["cortical_bone"]["density"]]]))
        gs.create_dataset("cortical_ap", data=np.array([[MATERIALS["cortical_bone"]["alpha_coeff"]]]))
        gs.create_dataset("brain_cp", data=np.array([[MATERIALS["brain"]["sound_speed"]]]))
        gs.create_dataset("brain_rho", data=np.array([[MATERIALS["brain"]["density"]]]))
        gs.create_dataset("brain_ap", data=np.array([[MATERIALS["brain"]["alpha_coeff"]]]))

        ss = f.create_group("simulation_settings")
        ss.create_dataset("Nt", data=np.array([[Nt]], dtype=np.float64))
        ss.create_dataset("dt", data=np.array([[dt]]))
        ss.create_dataset("grid_size", data=np.array([[Nx], [Ny], [Nz]], dtype=np.float64))
        ss.create_dataset("grid_spacing", data=np.array([[dx_m]]))
        ss.create_dataset("cfl", data=np.array([[cfl]]))
        ss.create_dataset("pml_size", data=np.array([[pml_size]], dtype=np.float64))
        ss.create_dataset("record_start_index", data=np.array([[record_start]]))
        ss.create_dataset("record_mode", data=args.record_mode)
        ss.create_dataset("alpha_power", data=np.array([[ALPHA_POWER]]))
        ss.create_dataset("elapsed_time_s", data=np.array([[t_elapsed]]))

        # Save coordinate vectors (in mm, matching benchmark convention)
        bm = BENCHMARK_GRIDS[benchmark]
        xi = np.linspace(bm["x_range_mm"][0], bm["x_range_mm"][1], out_nx)
        yi = np.linspace(bm["y_range_mm"][0], bm["y_range_mm"][1], out_ny)
        zi = np.linspace(bm["z_range_mm"][0], bm["z_range_mm"][1], out_nz)
        f.create_dataset("xi", data=xi.reshape(-1, 1))
        f.create_dataset("yi", data=yi.reshape(-1, 1))
        f.create_dataset("zi", data=zi.reshape(-1, 1))

    logger.info("Results saved to: %s", output_path)
    logger.info("Output p_amp shape (in file): %s", p_amp_save.shape)

    # ----- Quick comparison with reference -----
    ref_path = DATA_DIR / "KWAVE" / f"PH1-BM{benchmark}-SC{source_id}_KWAVE.mat"
    if ref_path.exists():
        with h5py.File(str(ref_path), "r") as f:
            p_ref = np.array(f["p_amp"], dtype=np.float32)
            p_ref = np.transpose(p_ref, (2, 1, 0))  # (z,y,x) -> (x,y,z)

        # If our output is at a different resolution, we can still report peak values
        logger.info("\n" + "=" * 72)
        logger.info("  COMPARISON WITH k-WAVE REFERENCE")
        logger.info("=" * 72)
        logger.info("  Reference peak: %.1f Pa at %s",
                     p_ref.max(), np.unravel_index(np.argmax(p_ref), p_ref.shape))
        logger.info("  Our peak:       %.1f Pa at %s",
                     p_amp_3d.max(), np.unravel_index(np.argmax(p_amp_3d), p_amp_3d.shape))

        if p_amp_3d.shape == p_ref.shape:
            # Direct voxel comparison
            mask = p_ref > 0.01 * p_ref.max()
            l2_err = np.linalg.norm(p_ref[mask] - p_amp_3d[mask]) / np.linalg.norm(p_ref[mask]) * 100
            linf_err = np.max(np.abs(p_ref[mask] - p_amp_3d[mask])) / p_ref.max() * 100
            peak_diff = (p_amp_3d.max() - p_ref.max()) / p_ref.max() * 100

            ref_peak_idx = np.unravel_index(np.argmax(p_ref), p_ref.shape)
            our_peak_idx = np.unravel_index(np.argmax(p_amp_3d), p_amp_3d.shape)
            peak_dist = np.linalg.norm(
                np.array(our_peak_idx, dtype=float) - np.array(ref_peak_idx, dtype=float)
            ) * dx_mm

            logger.info("  Peak difference: %+.1f%%", peak_diff)
            logger.info("  Peak distance:   %.2f mm", peak_dist)
            logger.info("  L2 error:        %.1f%%", l2_err)
            logger.info("  Linf error:      %.1f%%", linf_err)
        else:
            logger.info("  (Grid shapes differ, skipping voxel-level comparison)")
            logger.info("  Reference shape: %s, our shape: %s",
                        p_ref.shape, p_amp_3d.shape)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run k-Wave simulation against UCL Transcranial Benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", "-b", type=int, default=7, choices=[7, 8, 9],
        help="Benchmark number (7, 8, or 9)",
    )
    parser.add_argument(
        "--source", "-s", type=int, default=1, choices=[1, 2],
        help="Source configuration: 1=focused bowl, 2=plane piston",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False,
        help="Use GPU acceleration (requires CUDA binary)",
    )
    parser.add_argument(
        "--dx", type=float, default=None,
        help="Grid spacing in mm (default: 0.5 mm, i.e. 6 PPW at 500 kHz)",
    )
    parser.add_argument(
        "--pml-size", type=int, default=20,
        help="PML thickness in grid points",
    )
    parser.add_argument(
        "--pml-alpha", type=float, default=2.0,
        help="PML absorption coefficient (Nepers per grid point)",
    )
    parser.add_argument(
        "--cfl", type=float, default=0.1,
        help="CFL stability number (0.1 is safe for heterogeneous skull; "
             "the UCL reference used ~0.047)",
    )
    parser.add_argument(
        "--record-periods", type=int, default=3,
        help="Number of periods to record for steady-state extraction",
    )
    parser.add_argument(
        "--record-mode", type=str, default="time_series",
        choices=["time_series", "max_min"],
        help="Recording mode: 'time_series' for full p(t) with Fourier extraction "
             "(accurate amplitude and phase), or 'max_min' for p_max/p_min "
             "(lower memory, amplitude only, no phase)",
    )
    parser.add_argument(
        "--smooth", action="store_true", default=False,
        help="Enable smoothing of sound speed and density distributions",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default=str(pathlib.Path.home() / "Data/openlifu-validation/ucl-benchmarks/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 72)
    logger.info(
        "  UCL Benchmark %d | Source %d (%s) | GPU: %s",
        args.benchmark, args.source,
        SOURCES[args.source]["name"],
        args.gpu,
    )
    logger.info("=" * 72)

    try:
        output_path = run_benchmark(args)
        logger.info("\nDone. Output saved to: %s", output_path)
    except Exception:
        logger.exception("Benchmark simulation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
