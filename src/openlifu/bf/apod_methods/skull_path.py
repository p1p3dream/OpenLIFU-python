from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer


# Conversion factor between decibels and nepers: dB = (20 / ln(10)) * Np,
# so 1 Np = 20 / ln(10) dB ~= 8.685889638 dB. Attenuation coefficients
# expressed in dB/cm must be divided by this factor before being used in
# an exp(-alpha * x) Beer's-law expression whose argument is in nepers.
DB_PER_NEPER = 20.0 / np.log(10.0)


@dataclass
class SkullPathApodization(ApodizationMethod):
    """Down-weight elements whose rays traverse long bone paths.

    For each element, cast a ray from the element's world position toward
    the target and count the number of skull voxels intersected along the
    way. The total path length through bone is converted into a per-element
    amplitude weight via Beer's law:

        path_cm = (voxels_in_skull * step_mm) / 10
        weight  = exp(-alpha_bone_db_per_cm * path_cm / DB_PER_NEPER)

    The ``alpha_bone_db_per_cm`` parameter is the amplitude attenuation
    coefficient of cortical bone at the operating frequency (approximately
    3.5 dB/cm at 500 kHz; frequency-dependent and tunable). Dividing by
    ``DB_PER_NEPER ~= 8.686`` converts dB into nepers so the exponential
    argument is dimensionally consistent with ``numpy.exp``.

    An optional hard cutoff is available via ``max_path_mm``: any element
    whose bone path exceeds this length is given weight 0 regardless of
    the Beer's-law value. This is useful when a small number of elements
    have unphysically long bone paths that dominate the apodization even
    after exponential weighting; defaults to ``None`` (no cutoff).

    Parameters
    ----------
    skull_mask:
        3D ``xarray.DataArray`` whose values are truthy inside the skull
        and 0 / False elsewhere. The three ``dims`` are assumed to be
        spatial axes with monotonic numeric ``coords`` in ``coord_units``.
    alpha_bone_db_per_cm:
        Amplitude attenuation coefficient of cortical bone in dB/cm at
        the operating frequency. Default ``3.5`` is a rough value for
        500 kHz; tune per frequency.
    max_path_mm:
        Optional hard cutoff on the cumulative bone path length. Elements
        whose path exceeds this value get weight 0. ``None`` (default)
        disables the cutoff.
    step_mm:
        Ray-march step size, in millimeters. This is both the sampling
        density along the ray and the per-step bone length contribution
        when a sample falls inside the skull. Should be at most half the
        minimum voxel size of ``skull_mask`` for accuracy.
    coord_units:
        Spatial units of the ``skull_mask`` coordinate axes (default
        ``"mm"``).

    Notes
    -----
    ``skull_mask`` is an ``xarray.DataArray`` and is not JSON-serializable
    in a lossless way through the simple ``to_dict`` / ``from_dict``
    pattern used by the other apodization methods. ``to_dict`` therefore
    omits the mask and ``from_dict`` expects it to be supplied separately
    by the caller. This class is currently a run-time-only dataclass.

    The angle-based ``SkullIncidenceApodization`` remains in the library
    as an exploratory tool, but incidence-angle histogram analysis on
    GU008 showed that skull path length is a near-perfect predictor of
    dead elements (AUC 0.988) while entry angle is essentially
    non-predictive (AUC 0.123). ``SkullPathApodization`` is therefore
    the physically motivated default for down-weighting elements that
    cannot deliver meaningful energy through thick bone.
    """

    skull_mask: Annotated[xa.DataArray, OpenLIFUFieldData(
        "Skull mask",
        "3D binary mask of skull voxels (xarray.DataArray with spatial coords)",
    )] = field(default=None)  # type: ignore[assignment]
    """3D binary mask of skull voxels."""

    alpha_bone_db_per_cm: Annotated[float, OpenLIFUFieldData(
        "Bone attenuation (dB/cm)",
        "Amplitude attenuation coefficient of cortical bone at the operating frequency",
    )] = 3.5
    """Amplitude attenuation coefficient of cortical bone in dB/cm."""

    max_path_mm: Annotated[float | None, OpenLIFUFieldData(
        "Max path (mm)",
        "Optional hard cutoff: elements with path > max_path_mm get weight 0",
    )] = None
    """Optional hard cutoff on cumulative bone path length (mm)."""

    step_mm: Annotated[float, OpenLIFUFieldData(
        "Ray-march step (mm)",
        "Ray-march step size in millimeters",
    )] = 0.25
    """Ray-march step size in millimeters."""

    coord_units: Annotated[str, OpenLIFUFieldData(
        "Coord units",
        "Spatial units of the skull_mask coordinate axes",
    )] = "mm"
    """Spatial units of the ``skull_mask`` coordinate axes."""

    def __post_init__(self):
        if self.skull_mask is None:
            raise ValueError("skull_mask is required.")
        if not isinstance(self.skull_mask, xa.DataArray):
            raise TypeError(
                f"skull_mask must be an xarray.DataArray, got {type(self.skull_mask).__name__}."
            )
        if self.skull_mask.ndim != 3:
            raise ValueError(
                f"skull_mask must be 3-dimensional, got {self.skull_mask.ndim} dims."
            )
        if not isinstance(self.alpha_bone_db_per_cm, int | float):
            raise TypeError(
                "alpha_bone_db_per_cm must be a number, got "
                f"{type(self.alpha_bone_db_per_cm).__name__}."
            )
        if self.alpha_bone_db_per_cm < 0:
            raise ValueError(
                f"alpha_bone_db_per_cm must be non-negative, got {self.alpha_bone_db_per_cm}."
            )
        if self.max_path_mm is not None:
            if not isinstance(self.max_path_mm, int | float):
                raise TypeError(
                    f"max_path_mm must be a number or None, got {type(self.max_path_mm).__name__}."
                )
            if self.max_path_mm <= 0:
                raise ValueError(
                    f"max_path_mm must be positive when given, got {self.max_path_mm}."
                )
        if not isinstance(self.step_mm, int | float):
            raise TypeError(
                f"step_mm must be a number, got {type(self.step_mm).__name__}."
            )
        if self.step_mm <= 0:
            raise ValueError(f"step_mm must be positive, got {self.step_mm}.")

        self._precompute_mask()

    # ----- Helper: voxel-frame geometry -----

    def _axis_coords(self):
        """Return (dims, coord_arrays) in the natural array-axis order."""
        dims = tuple(self.skull_mask.dims)
        coords = tuple(np.asarray(self.skull_mask.coords[d].values, dtype=float) for d in dims)
        return dims, coords

    def _precompute_mask(self):
        """Cache a boolean skull indicator and ascending-coord lookups.

        The mask is re-indexed so every axis is stored with ascending
        coordinates. This lets ``_world_to_index`` use a simple affine
        mapping, matching the convention in ``SkullIncidenceApodization``.
        """
        mask_bool = np.asarray(self.skull_mask.values).astype(bool)
        _, coords = self._axis_coords()

        spacing = tuple(
            float(abs(c[1] - c[0])) if len(c) > 1 else 1.0 for c in coords
        )
        self._voxel_spacing = spacing
        self._min_voxel_size = float(min(spacing))

        # Flip any descending axis so internal indexing can assume ascending.
        self._coord_ascending = []
        for axis_i, c in enumerate(coords):
            ascending = bool(len(c) <= 1 or c[-1] >= c[0])
            self._coord_ascending.append(ascending)
            if not ascending:
                mask_bool = np.flip(mask_bool, axis=axis_i)
        self._mask = mask_bool
        self._sorted_coords = tuple(
            (c if asc else c[::-1]).astype(float)
            for c, asc in zip(coords, self._coord_ascending, strict=False)
        )

    def _in_skull(self, point_coord: np.ndarray) -> bool:
        """Nearest-voxel lookup into the cached boolean skull mask.

        Points outside the mask bounding box return False. Counting voxels
        along the ray (rather than e.g. SDF sign) is the simplest and most
        direct definition of "bone path length" for this method, and
        nearest-neighbor is accurate enough because the step size is set
        to (at most) half a voxel.
        """
        shape = self._mask.shape
        idx = np.empty(3, dtype=np.int64)
        for j in range(3):
            c = self._sorted_coords[j]
            spacing_j = self._voxel_spacing[j] if self._voxel_spacing[j] != 0.0 else 1.0
            raw = (point_coord[j] - c[0]) / spacing_j
            raw_i = int(np.round(raw))
            if raw_i < 0 or raw_i >= shape[j]:
                return False
            idx[j] = raw_i
        return bool(self._mask[idx[0], idx[1], idx[2]])

    # ----- Main apodization calculation -----

    def calc_apodization(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ):
        # Transform convention: translation is in meters (world-frame SI).
        matrix = np.asarray(transform, dtype=float) if transform is not None else np.eye(4)

        # Everything below is in ``coord_units`` (default mm).
        m_to_coord = getunitconversion("m", self.coord_units)
        target_pos = np.asarray(target.get_position(units=self.coord_units), dtype=float)

        dims, coords = self._axis_coords()

        # Step size in coord_units.
        mm_to_coord = getunitconversion("mm", self.coord_units)
        step = float(self.step_mm) * mm_to_coord

        # Bounding box in the original coord frame, for short-circuit.
        mins = np.array([c.min() for c in coords], dtype=float)
        maxs = np.array([c.max() for c in coords], dtype=float)

        weights = np.ones(arr.numelements(), dtype=float)

        for i, el in enumerate(arr.elements):
            el_pos_m = np.asarray(el.get_position(units="m", matrix=matrix), dtype=float)
            el_pos = el_pos_m * m_to_coord

            ray_vec = target_pos - el_pos
            total_dist = float(np.linalg.norm(ray_vec))
            if total_dist == 0.0:
                weights[i] = 1.0
                continue
            ray_dir = ray_vec / total_dist

            ray_dir_axes = np.array(
                [ray_dir[list(target.dims).index(d)] if d in target.dims else ray_dir[j]
                 for j, d in enumerate(dims)],
                dtype=float,
            )
            el_pos_axes = np.array(
                [el_pos[list(target.dims).index(d)] if d in target.dims else el_pos[j]
                 for j, d in enumerate(dims)],
                dtype=float,
            )
            n_steps = max(2, int(np.ceil(total_dist / step)) + 1)
            ts = np.linspace(0.0, total_dist, n_steps)

            voxels_in_skull = 0
            for t in ts:
                p_axes = el_pos_axes + ray_dir_axes * t
                # Bounding-box short circuit.
                if np.any(p_axes < mins - step) or np.any(p_axes > maxs + step):
                    continue
                if self._in_skull(p_axes):
                    voxels_in_skull += 1

            # Effective step length along the ray between consecutive
            # samples (linspace endpoint-inclusive), expressed in
            # coord_units and then converted to cm. Falls back to the
            # requested step if n_steps is degenerate.
            if n_steps > 1:
                eff_step_coord = total_dist / (n_steps - 1)
            else:
                eff_step_coord = step
            coord_to_mm = getunitconversion(self.coord_units, "mm")
            eff_step_mm = eff_step_coord * coord_to_mm
            path_mm = voxels_in_skull * eff_step_mm
            path_cm = path_mm / 10.0

            if self.max_path_mm is not None and path_mm > self.max_path_mm:
                weights[i] = 0.0
                continue

            # Beer's law in natural log. alpha is dB/cm, so divide by
            # DB_PER_NEPER (~8.686) to convert to Np/cm before applying
            # to exp().
            weights[i] = float(np.exp(-self.alpha_bone_db_per_cm * path_cm / DB_PER_NEPER))

        return weights

    # ----- Serialization -----

    def to_dict(self):
        """Serialize to a dict, excluding ``skull_mask``.

        ``skull_mask`` is an ``xarray.DataArray`` and is not losslessly
        representable via the simple dict round-trip pattern used by the
        other apodization methods. Callers who need to reconstruct this
        method from a dict must re-supply ``skull_mask`` explicitly.
        """
        d = {
            "class": self.__class__.__name__,
            "alpha_bone_db_per_cm": self.alpha_bone_db_per_cm,
            "max_path_mm": self.max_path_mm,
            "step_mm": self.step_mm,
            "coord_units": self.coord_units,
        }
        return d

    def to_table(self) -> pd.DataFrame:
        """Get a table of the apodization method parameters."""
        records = [
            {"Name": "Type", "Value": "Skull Path", "Unit": ""},
            {"Name": "Bone Attenuation", "Value": self.alpha_bone_db_per_cm, "Unit": "dB/cm"},
            {"Name": "Max Path", "Value": self.max_path_mm, "Unit": "mm"},
            {"Name": "Ray Step", "Value": self.step_mm, "Unit": "mm"},
            {"Name": "Mask Units", "Value": self.coord_units, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)
