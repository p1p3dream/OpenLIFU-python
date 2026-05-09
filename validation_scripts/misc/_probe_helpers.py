"""Per-voxel water-calibrated gate helpers for time-gated probe analyses.

Standalone module (no package dependencies beyond numpy) so multiple
scripts can adopt it without contending on the same file. Load by
path when ``scripts/`` is not on sys.path, e.g.::

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "_probe_helpers",
        pathlib.Path(__file__).with_name("_probe_helpers.py"),
    )
    _probe_helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_probe_helpers)

Decision-grade recipe (per Codex):

    t_gate(v) = t_water_peak(target)
              + (|aperture - v| - |aperture - target|) / c0

The current time-gated spatial-search probe uses ONE water-calibrated
gate center (the target voxel's water-peak time) for every cube voxel.
With ~11^3 voxels in a 5 mm cube and TOF spreads of several
microseconds across the cube, that single gate bakes a timing bias
into a diagnostic whose purpose is to distinguish timing from spatial
displacement. Giving each voxel its own geometrically corrected gate
center removes that bias.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np


def per_voxel_water_calibrated_gate_center(
    sensor_world_mm: np.ndarray,
    target_world_mm: np.ndarray,
    aperture_center_world_mm: np.ndarray,
    t_water_peak_target: float,
    c0_mps: float = 1500.0,
) -> float:
    """Return the gate center time (seconds) for a given sensor voxel
    using the water-calibrated target peak as the reference, corrected
    for geometric TOF relative to target.

    Per Codex's decision-grade recipe::

        t_gate(v) = t_water_peak(target)
                  + (|aperture - v| - |aperture - target|) / c0

    When ``sensor == target``, returns ``t_water_peak_target`` exactly.

    Parameters
    ----------
    sensor_world_mm
        (3,) world position of this sensor voxel, in millimeters.
    target_world_mm
        (3,) world position of the target, in millimeters.
    aperture_center_world_mm
        (3,) world position of the aperture center, in millimeters.
    t_water_peak_target
        Water-calibrated peak time at the target (seconds).
    c0_mps
        Water sound speed used for the geometric TOF correction
        (meters per second). Defaults to 1500.0 m/s.

    Returns
    -------
    float
        Gate center time in seconds.
    """
    sensor = np.asarray(sensor_world_mm, dtype=float).reshape(3)
    target = np.asarray(target_world_mm, dtype=float).reshape(3)
    aperture = np.asarray(aperture_center_world_mm, dtype=float).reshape(3)

    d_sensor_mm = float(np.linalg.norm(aperture - sensor))
    d_target_mm = float(np.linalg.norm(aperture - target))

    # Convert mm to m before dividing by c0 (m/s) to get seconds.
    delta_seconds = (d_sensor_mm - d_target_mm) * 1e-3 / c0_mps
    return float(t_water_peak_target) + delta_seconds


def extract_focal_window_peak(
    time_series: np.ndarray,
    dt: float,
    gate_center_s: float,
    gate_half_width_s: float,
) -> float:
    """Return ``max(|time_series|)`` within
    ``[gate_center - gate_half_width, gate_center + gate_half_width]``.

    If the window partially overlaps the valid sample range it is
    clamped. If the window is fully outside the valid range, returns
    0.0 and emits a ``RuntimeWarning``.

    Parameters
    ----------
    time_series
        1-D array of samples (shape ``(Nt,)``).
    dt
        Sample period in seconds.
    gate_center_s
        Center of the window in seconds (measured from sample 0).
    gate_half_width_s
        Half-width of the window in seconds. Must be non-negative.

    Returns
    -------
    float
        Peak absolute amplitude within the clamped window, or 0.0 if
        the window is fully outside the sample range.
    """
    ts = np.asarray(time_series)
    if ts.ndim != 1:
        raise ValueError(f"time_series must be 1-D, got shape {ts.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if gate_half_width_s < 0.0:
        raise ValueError(
            f"gate_half_width_s must be non-negative, got {gate_half_width_s}"
        )

    n = ts.shape[0]
    t_lo = gate_center_s - gate_half_width_s
    t_hi = gate_center_s + gate_half_width_s

    # Convert time bounds to sample indices. Use floor/ceil so the full
    # requested window is covered (inclusive start, exclusive end).
    i_lo_raw = int(np.floor(t_lo / dt))
    i_hi_raw = int(np.ceil(t_hi / dt)) + 1  # +1 because slicing is exclusive

    i_lo = max(0, i_lo_raw)
    i_hi = min(n, i_hi_raw)

    if i_hi <= i_lo:
        warnings.warn(
            (
                "Focal window [{:g}s, {:g}s] falls fully outside the valid "
                "sample range [0, {:g}s); returning 0.0."
            ).format(t_lo, t_hi, n * dt),
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0

    return float(np.max(np.abs(ts[i_lo:i_hi])))


def compute_spatial_search_peaks(
    target_cube_time_series: np.ndarray,
    cube_world_mm: np.ndarray,
    target_world_mm: np.ndarray,
    aperture_center_world_mm: np.ndarray,
    t_water_peak_target: float,
    dt: float,
    gate_half_width_s: float,
    c0_mps: float = 1500.0,
) -> Dict[str, object]:
    """Compute per-voxel focal-window peaks across a spatial-search cube.

    For each of ``Nv`` cube voxels:

    1. Compute the per-voxel gate center via
       :func:`per_voxel_water_calibrated_gate_center`.
    2. Extract the focal-window peak amplitude via
       :func:`extract_focal_window_peak`.

    Parameters
    ----------
    target_cube_time_series
        (Nv, Nt) per-voxel time series.
    cube_world_mm
        (Nv, 3) world positions of the cube voxels, in millimeters.
    target_world_mm
        (3,) world position of the target, in millimeters.
    aperture_center_world_mm
        (3,) world position of the aperture center, in millimeters.
    t_water_peak_target
        Water-calibrated peak time at the target (seconds).
    dt
        Sample period of the time series (seconds).
    gate_half_width_s
        Half-width of the focal window (seconds).
    c0_mps
        Water sound speed for the geometric TOF correction (m/s).

    Returns
    -------
    dict
        ``{
            'per_voxel_peak': np.ndarray (Nv,),
            'argmax_voxel_index': int,
            'argmax_world_mm': np.ndarray (3,),
            'argmax_peak': float,
            'at_target_peak': float,   # peak from the voxel closest to target
            'spatial_offset_mm': float # ||argmax_world - target_world||
        }``
    """
    ts = np.asarray(target_cube_time_series)
    cube = np.asarray(cube_world_mm, dtype=float)
    target = np.asarray(target_world_mm, dtype=float).reshape(3)

    if ts.ndim != 2:
        raise ValueError(
            f"target_cube_time_series must be 2-D (Nv, Nt), got shape {ts.shape}"
        )
    if cube.ndim != 2 or cube.shape[1] != 3:
        raise ValueError(
            f"cube_world_mm must have shape (Nv, 3), got {cube.shape}"
        )
    if cube.shape[0] != ts.shape[0]:
        raise ValueError(
            "cube_world_mm and target_cube_time_series disagree on Nv: "
            f"{cube.shape[0]} vs {ts.shape[0]}"
        )

    n_v = cube.shape[0]
    per_voxel_peak = np.zeros(n_v, dtype=float)

    for i in range(n_v):
        gate_center = per_voxel_water_calibrated_gate_center(
            sensor_world_mm=cube[i],
            target_world_mm=target,
            aperture_center_world_mm=aperture_center_world_mm,
            t_water_peak_target=t_water_peak_target,
            c0_mps=c0_mps,
        )
        per_voxel_peak[i] = extract_focal_window_peak(
            time_series=ts[i],
            dt=dt,
            gate_center_s=gate_center,
            gate_half_width_s=gate_half_width_s,
        )

    argmax_idx = int(np.argmax(per_voxel_peak))
    argmax_world = cube[argmax_idx].copy()
    argmax_peak = float(per_voxel_peak[argmax_idx])

    # "at_target_peak" is the peak from the voxel closest to target
    # (the cube is discrete, so the target may not be a grid point).
    dists_to_target = np.linalg.norm(cube - target[None, :], axis=1)
    target_idx = int(np.argmin(dists_to_target))
    at_target_peak = float(per_voxel_peak[target_idx])

    spatial_offset_mm = float(np.linalg.norm(argmax_world - target))

    return {
        'per_voxel_peak': per_voxel_peak,
        'argmax_voxel_index': argmax_idx,
        'argmax_world_mm': argmax_world,
        'argmax_peak': argmax_peak,
        'at_target_peak': at_target_peak,
        'spatial_offset_mm': spatial_offset_mm,
    }
