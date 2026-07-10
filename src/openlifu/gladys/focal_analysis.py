"""Tissue-masked focal peak finding for GLADYS pressure fields.

Provides a reusable function for locating the pressure peak within an ROI
sphere around a target coordinate, optionally excluding skull and near-skull
voxels via an EDT-based tissue mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

FOCAL_ROI_RADIUS_MM: float = 15.0
"""Default radius (mm) of the spherical ROI around the target."""

SKULL_MARGIN_MM: float = 5.0
"""Default exclusion margin (mm) beyond the skull surface."""


@dataclass
class FocalStats:
    """Results of a focal peak analysis."""

    label: str
    peak_position_mm: np.ndarray
    peak_pressure: float
    focal_error_mm: float
    p_at_target: float
    global_max_pressure: float
    global_peak_position_mm: np.ndarray
    focal_volume_mm3: float
    used_tissue_mask: bool
    n_valid_voxels: int


def find_focal_peak(
    p_max: np.ndarray,
    coord_arrays: dict[str, np.ndarray],
    target_mm: np.ndarray,
    skull_mask: np.ndarray | None = None,
    roi_radius_mm: float = FOCAL_ROI_RADIUS_MM,
    skull_margin_mm: float = SKULL_MARGIN_MM,
    grid_spacing_mm: float | None = None,
    label: str = "",
) -> FocalStats:
    """Find the focal pressure peak within a tissue-masked ROI sphere.

    Parameters
    ----------
    p_max : np.ndarray
        3-D pressure field (e.g. maximum pressure over time).
    coord_arrays : dict[str, np.ndarray]
        Mapping of dimension names to 1-D coordinate arrays (mm), one per
        axis of *p_max*.  Iteration order determines the axis correspondence
        (first key = axis 0, etc.).
    target_mm : np.ndarray
        Target coordinate in mm, length 3, ordered to match *coord_arrays*.
    skull_mask : np.ndarray or None
        Boolean array with the same shape as *p_max*.  ``True`` marks bone
        voxels.  When provided, skull voxels and those within *skull_margin_mm*
        of the skull surface are excluded from the ROI.
    roi_radius_mm : float
        Radius of the spherical ROI centred on *target_mm*.
    skull_margin_mm : float
        Exclusion margin (mm) beyond the skull surface (computed via EDT).
    grid_spacing_mm : float or None
        Isotropic voxel spacing for the EDT.  If ``None``, the mean spacing
        across all axes is used.
    label : str
        Human-readable label for log messages.

    Returns
    -------
    FocalStats
        Dataclass containing peak location, pressures, focal error, focal
        volume, and masking metadata.
    """
    dims = list(coord_arrays.keys())
    arrays = [coord_arrays[d] for d in dims]

    # Pressure at the target coordinate
    target_idx = tuple(
        int(np.argmin(np.abs(arrays[ax] - target_mm[ax])))
        for ax in range(3)
    )
    p_at_target = float(p_max[target_idx])

    # Spherical ROI mask
    mg = np.meshgrid(*arrays, indexing="ij")
    dist_from_target = np.sqrt(sum((m - t) ** 2 for m, t in zip(mg, target_mm)))
    roi_mask = dist_from_target <= roi_radius_mm

    # Per-axis spacing for EDT and volume calculations
    per_axis_spacing = tuple(
        float(np.abs(np.diff(arrays[i])).mean()) for i in range(len(dims))
    )
    if grid_spacing_mm is not None:
        per_axis_spacing = (grid_spacing_mm, grid_spacing_mm, grid_spacing_mm)

    # Tissue masking: exclude skull + margin from the ROI
    used_tissue_mask = False
    if skull_mask is not None:
        spacing_tuple = per_axis_spacing
        # EDT gives distance (mm) from each non-skull voxel to the nearest skull voxel
        dt_mm = distance_transform_edt(~skull_mask, sampling=spacing_tuple)
        tissue_mask = ~skull_mask & (dt_mm >= skull_margin_mm)
        valid_mask = roi_mask & tissue_mask
        if not valid_mask.any():
            logger.warning(
                "%s: no valid tissue voxels in ROI (r=%.0f mm, margin=%.0f mm); "
                "falling back to unmasked ROI",
                label, roi_radius_mm, skull_margin_mm,
            )
            valid_mask = roi_mask
        else:
            used_tissue_mask = True
            logger.info(
                "%s: tissue-masked ROI: %d valid voxels (skull margin=%.0f mm)",
                label, int(valid_mask.sum()), skull_margin_mm,
            )
    else:
        valid_mask = roi_mask

    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        logger.warning("%s: ROI sphere contains no voxels; reporting target as peak", label)
        return FocalStats(
            label=label,
            peak_position_mm=np.array(target_mm, dtype=float),
            peak_pressure=p_at_target,
            focal_error_mm=0.0,
            p_at_target=p_at_target,
            global_max_pressure=float(p_max.max()),
            global_peak_position_mm=np.array([
                float(arrays[i][np.unravel_index(p_max.argmax(), p_max.shape)[i]])
                for i in range(len(dims))
            ]),
            focal_volume_mm3=0.0,
            used_tissue_mask=used_tissue_mask,
            n_valid_voxels=0,
        )

    # Peak within the (possibly masked) ROI
    pmax_roi = np.where(valid_mask, p_max, -np.inf)
    roi_idx = np.unravel_index(pmax_roi.argmax(), pmax_roi.shape)
    peak_mm = np.array([float(arrays[i][roi_idx[i]]) for i in range(len(dims))])
    focal_error = float(np.linalg.norm(peak_mm - target_mm))
    peak_pressure = float(p_max[roi_idx])

    # Global peak (over the entire field)
    global_max = float(p_max.max())
    global_idx = np.unravel_index(p_max.argmax(), p_max.shape)
    global_peak_mm = np.array([float(arrays[i][global_idx[i]]) for i in range(len(dims))])

    # -6 dB focal volume
    threshold_6db = peak_pressure / 2.0
    focal_region = valid_mask & (p_max >= threshold_6db)
    voxel_vol_mm3 = float(np.prod(per_axis_spacing))
    focal_vol_mm3 = int(focal_region.sum()) * voxel_vol_mm3

    logger.info(
        "%s: focal_peak=%.4g Pa @ (%s) mm, error=%.2f mm; p@target=%.4g Pa (ROI r=%.0f mm)",
        label, peak_pressure,
        ", ".join(f"{v:.1f}" for v in peak_mm),
        focal_error, p_at_target, roi_radius_mm,
    )
    logger.info(
        "%s: global_max=%.4g Pa @ (%s) mm; focal_vol_6dB=%.1f mm^3",
        label, global_max,
        ", ".join(f"{v:.1f}" for v in global_peak_mm),
        focal_vol_mm3,
    )

    return FocalStats(
        label=label,
        peak_position_mm=peak_mm,
        peak_pressure=peak_pressure,
        focal_error_mm=focal_error,
        p_at_target=p_at_target,
        global_max_pressure=global_max,
        global_peak_position_mm=global_peak_mm,
        focal_volume_mm3=focal_vol_mm3,
        used_tissue_mask=used_tissue_mask,
        n_valid_voxels=n_valid,
    )
