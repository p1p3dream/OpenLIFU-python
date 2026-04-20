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


@dataclass
class SkullIncidenceApodization(ApodizationMethod):
    """Down-weight elements whose rays strike the skull near-perpendicular.

    For each element, cast a ray from the element's world position toward the
    target. March along the ray in ``step_mm`` increments (converted to the
    coordinate units of ``skull_mask``) and find the first voxel where the
    segmentation mask is truthy ("skull"). Estimate the local outward skull
    surface normal from the 3D gradient of the mask, compute the angle between
    the incoming ray direction and that normal, and emit a per-element weight
    using a linear roll-off:

        * weight = 0 if angle < ``min_angle_deg``   (near-perpendicular)
        * weight = 1 if angle > ``rolloff_angle_deg`` (grazing / oblique)
        * linear between the two

    The physical intuition: normal-incidence rays reflect most strongly at the
    soft-tissue / skull interface (highest impedance mismatch reflection
    coefficient) and contribute the most to near-aperture standing-wave peaks.
    Grazing-incidence rays reflect less and transmit more, so they are kept.

    Parameters
    ----------
    skull_mask:
        3D ``xarray.DataArray`` whose values are truthy inside the skull and
        0 / False elsewhere. The three ``dims`` are assumed to be spatial
        axes with monotonic numeric ``coords`` in ``coord_units``.
    min_angle_deg:
        Angle (deg, between ray and local surface normal) below which the
        element is fully suppressed (weight 0).
    rolloff_angle_deg:
        Angle (deg) above which the element is fully kept (weight 1).
    step_mm:
        Ray-march step size, in millimeters.
    coord_units:
        Spatial units of the ``skull_mask`` coordinate axes (default ``"mm"``).

    Notes
    -----
    ``skull_mask`` is an ``xarray.DataArray`` and is not JSON-serializable in
    a lossless way through the simple ``to_dict`` / ``from_dict`` pattern
    used by the other apodization methods. ``to_dict`` therefore omits the
    mask and ``from_dict`` expects it to be supplied separately by the
    caller. This class is currently a run-time-only dataclass.
    """

    skull_mask: Annotated[xa.DataArray, OpenLIFUFieldData(
        "Skull mask",
        "3D binary mask of skull voxels (xarray.DataArray with spatial coords)",
    )] = field(default=None)  # type: ignore[assignment]
    """3D binary mask of skull voxels."""

    min_angle_deg: Annotated[float, OpenLIFUFieldData(
        "Min angle (deg)",
        "Angle below which the element is fully suppressed (weight 0)",
    )] = 20.0
    """Angle below which the element is fully suppressed (weight 0)."""

    rolloff_angle_deg: Annotated[float, OpenLIFUFieldData(
        "Rolloff angle (deg)",
        "Angle above which the element is fully kept (weight 1)",
    )] = 45.0
    """Angle above which the element is fully kept (weight 1)."""

    step_mm: Annotated[float, OpenLIFUFieldData(
        "Ray-march step (mm)",
        "Ray-march step size in millimeters",
    )] = 0.5
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
        if not isinstance(self.min_angle_deg, int | float):
            raise TypeError(
                f"min_angle_deg must be a number, got {type(self.min_angle_deg).__name__}."
            )
        if not isinstance(self.rolloff_angle_deg, int | float):
            raise TypeError(
                f"rolloff_angle_deg must be a number, got {type(self.rolloff_angle_deg).__name__}."
            )
        if self.min_angle_deg < 0:
            raise ValueError(
                f"min_angle_deg must be non-negative, got {self.min_angle_deg}."
            )
        if self.rolloff_angle_deg <= self.min_angle_deg:
            raise ValueError(
                "rolloff_angle_deg must be strictly greater than min_angle_deg, "
                f"got rolloff={self.rolloff_angle_deg}, min={self.min_angle_deg}."
            )
        if not isinstance(self.step_mm, int | float):
            raise TypeError(
                f"step_mm must be a number, got {type(self.step_mm).__name__}."
            )
        if self.step_mm <= 0:
            raise ValueError(f"step_mm must be positive, got {self.step_mm}.")

    # ----- Helper: voxel-frame geometry and surface normals -----

    def _axis_coords(self):
        """Return (dims, coord_arrays) in the natural array-axis order."""
        dims = tuple(self.skull_mask.dims)
        coords = tuple(np.asarray(self.skull_mask.coords[d].values, dtype=float) for d in dims)
        return dims, coords

    def _compute_normals(self):
        """Compute unit outward-normal components at every voxel, in mask axis order.

        Returns an array of shape (3, *skull_mask.shape) whose first axis is
        the gradient component along ``skull_mask.dims[i]``. Any voxel whose
        gradient magnitude is 0 has all three components = 0 (undefined
        normal); callers must check for that before normalizing.
        """
        mask = np.asarray(self.skull_mask.values, dtype=float)
        _, coords = self._axis_coords()
        # np.gradient accepts per-axis spacings. Pass coordinate arrays so
        # the gradient is per-unit-length in the mask's physical space,
        # not per-voxel. This keeps the normal direction correct even when
        # voxel spacing differs per axis.
        grads = np.gradient(mask, *coords)
        # np.gradient returns a list of arrays, one per axis, in axis order
        # that matches ``dims``. Stack into a (3, ...) array.
        g = np.stack(grads, axis=0)
        return g

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

        # Everything below is in ``coord_units`` (default mm). Convert
        # positions from meters into coord_units once.
        m_to_coord = getunitconversion("m", self.coord_units)
        target_pos = np.asarray(target.get_position(units=self.coord_units), dtype=float)

        dims, coords = self._axis_coords()
        mask_vals = np.asarray(self.skull_mask.values)
        normals = self._compute_normals()  # (3, *shape), in coord_units space

        # Step size in coord_units.
        mm_to_coord = getunitconversion("mm", self.coord_units)
        step = float(self.step_mm) * mm_to_coord

        # Per-axis bounds for the ray marcher, used to stop early if the ray
        # has left the mask volume before reaching the target.
        mins = np.array([c.min() for c in coords], dtype=float)
        maxs = np.array([c.max() for c in coords], dtype=float)
        shape = np.array(mask_vals.shape, dtype=int)

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

            # The ray direction in the coord axes of the mask, in the same
            # ordering as ``dims``. The ray lives in world coordinates with
            # axes named (x, y, z) by convention on Point.dims, which must
            # match ``skull_mask.dims`` for the dot product with normals to
            # be meaningful. We assume the caller has built the mask in the
            # same (x, y, z) world frame as the transform places the array.
            ray_dir_axes = np.array(
                [ray_dir[list(target.dims).index(d)] if d in target.dims else ray_dir[j]
                 for j, d in enumerate(dims)],
                dtype=float,
            )

            # Ray march
            n_steps = int(np.ceil(total_dist / step))
            hit = False
            hit_idx = None
            for k in range(1, n_steps + 1):
                p = el_pos + ray_dir * (k * step)
                # Snap to nearest voxel index along each axis.
                idx = []
                in_bounds = True
                for j in range(3):
                    c = coords[j]
                    # Find the nearest index by searching sorted coords.
                    # coords are monotonic but could be ascending or descending.
                    ascending = c[-1] >= c[0]
                    if ascending:
                        ix = int(np.clip(np.searchsorted(c, p[j]), 0, len(c) - 1))
                        # searchsorted returns insertion point; pick nearest
                        # of ix-1 and ix.
                        if ix > 0 and abs(c[ix - 1] - p[j]) < abs(c[ix] - p[j]):
                            ix = ix - 1
                    else:
                        # For descending coords, negate and search.
                        ix = int(np.clip(np.searchsorted(-c, -p[j]), 0, len(c) - 1))
                        if ix > 0 and abs(c[ix - 1] - p[j]) < abs(c[ix] - p[j]):
                            ix = ix - 1
                    if p[j] < mins[j] - 0.5 * step or p[j] > maxs[j] + 0.5 * step:
                        in_bounds = False
                        break
                    idx.append(ix)
                if not in_bounds:
                    continue
                ix, iy, iz = idx
                if mask_vals[ix, iy, iz]:
                    hit = True
                    hit_idx = (ix, iy, iz)
                    break

            if not hit:
                weights[i] = 1.0
                continue

            # Local outward normal at the hit voxel (in coord-axis order).
            n = normals[:, hit_idx[0], hit_idx[1], hit_idx[2]]
            n_norm = float(np.linalg.norm(n))
            if n_norm == 0.0:
                # Gradient-degenerate voxel: no defined normal. Treat as
                # no-skull-in-path and keep the element.
                weights[i] = 1.0
                continue
            n = n / n_norm

            cos_theta = float(np.clip(abs(np.dot(ray_dir_axes, n)), 0.0, 1.0))
            theta_deg = float(np.degrees(np.arccos(cos_theta)))

            if theta_deg <= self.min_angle_deg:
                w = 0.0
            elif theta_deg >= self.rolloff_angle_deg:
                w = 1.0
            else:
                w = (theta_deg - self.min_angle_deg) / (
                    self.rolloff_angle_deg - self.min_angle_deg
                )
            weights[i] = w

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
            "min_angle_deg": self.min_angle_deg,
            "rolloff_angle_deg": self.rolloff_angle_deg,
            "step_mm": self.step_mm,
            "coord_units": self.coord_units,
        }
        return d

    def to_table(self) -> pd.DataFrame:
        """Get a table of the apodization method parameters."""
        records = [
            {"Name": "Type", "Value": "Skull Incidence", "Unit": ""},
            {"Name": "Min Angle", "Value": self.min_angle_deg, "Unit": "deg"},
            {"Name": "Rolloff Angle", "Value": self.rolloff_angle_deg, "Unit": "deg"},
            {"Name": "Ray Step", "Value": self.step_mm, "Unit": "mm"},
            {"Name": "Mask Units", "Value": self.coord_units, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)
