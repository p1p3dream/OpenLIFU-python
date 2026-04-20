from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa
from scipy.ndimage import distance_transform_edt, map_coordinates

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer


@dataclass
class SkullIncidenceApodization(ApodizationMethod):
    """Down-weight elements whose rays strike the skull near-perpendicular.

    For each element, cast a ray from the element's world position toward the
    target. The ray is tested against the skull's signed distance field (SDF):
    the first sample where the SDF changes from positive (outside skull) to
    non-positive (inside skull) marks the intersection. Linear interpolation
    between the bracketing samples estimates a sub-voxel hit position, at
    which the cached SDF gradient is evaluated to obtain the local outward
    surface normal. The incidence angle between the ray and that normal is
    then mapped to a per-element weight using a linear roll-off:

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
        Ray-march step size, in millimeters. Used as the sampling density
        for sign-change detection along the ray; the final hit position is
        refined sub-voxel via linear interpolation of the SDF, so modest
        changes in ``step_mm`` should not materially change the weights.
    coord_units:
        Spatial units of the ``skull_mask`` coordinate axes (default ``"mm"``).
    safe_on_degenerate_normal:
        If True (default), voxels where the SDF gradient magnitude is
        effectively zero (``< 0.1 / min_voxel_size``) are treated as
        undefined normals and the element is assigned weight 0 (i.e.
        excluded from the aperture). This is the conservative / safety
        oriented default for a steering apodization. Set False to keep
        the previous behavior of assigning weight 1 to degenerate normals.

    Notes
    -----
    ``skull_mask`` is an ``xarray.DataArray`` and is not JSON-serializable in
    a lossless way through the simple ``to_dict`` / ``from_dict`` pattern
    used by the other apodization methods. ``to_dict`` therefore omits the
    mask and ``from_dict`` expects it to be supplied separately by the
    caller. This class is currently a run-time-only dataclass.

    Implementation
    --------------
    The signed distance field (positive outside the skull, negative inside)
    is computed once in ``__post_init__`` and its gradient is cached. This
    avoids recomputing a whole-volume EDT per ``calc_apodization`` call.
    Normals evaluated from the SDF gradient are smooth across the interface,
    unlike the step-function gradient of the raw binary mask, and the ray
    intersection is grid-step-independent up to the accuracy of the SDF.
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

    safe_on_degenerate_normal: Annotated[bool, OpenLIFUFieldData(
        "Safe on degenerate normal",
        "If True, rays whose intersection has an undefined normal are excluded (weight 0)",
    )] = True
    """If True, degenerate normals produce weight 0 (conservative); else weight 1."""

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

        self._precompute_sdf()

    # ----- Helper: voxel-frame geometry and surface normals -----

    def _axis_coords(self):
        """Return (dims, coord_arrays) in the natural array-axis order."""
        dims = tuple(self.skull_mask.dims)
        coords = tuple(np.asarray(self.skull_mask.coords[d].values, dtype=float) for d in dims)
        return dims, coords

    def _precompute_sdf(self):
        """Compute and cache the signed distance field and its gradient.

        Sign convention: SDF is positive outside the skull and negative
        inside the skull, in ``coord_units``. ``distance_transform_edt``'s
        ``sampling`` argument is used so distances are in the physical
        units of the mask's coords (per-axis spacing), not voxel counts.

        Gradient is cached in mask axis order: ``self._sdf_grad`` has shape
        ``(3, *mask.shape)``. Because the SDF is smooth across the skull
        boundary, its gradient gives a well-defined outward normal
        everywhere except in the interior / exterior far from the surface
        (where the gradient can approach zero numerically).
        """
        mask_bool = np.asarray(self.skull_mask.values).astype(bool)
        _, coords = self._axis_coords()
        # Per-axis physical spacing. Use absolute difference of the first
        # two coordinate samples; this handles both ascending and
        # descending monotonic coord arrays.
        spacing = tuple(
            float(abs(c[1] - c[0])) if len(c) > 1 else 1.0 for c in coords
        )
        self._voxel_spacing = spacing
        self._min_voxel_size = float(min(spacing))

        # distance_transform_edt returns the distance to the nearest False
        # voxel (for our inputs). Compose the two halves with opposite
        # signs to form a signed distance.
        outside = distance_transform_edt(~mask_bool, sampling=spacing)
        if mask_bool.any():
            inside = distance_transform_edt(mask_bool, sampling=spacing)
        else:
            inside = np.zeros_like(outside)
        sdf = outside - inside  # positive outside, negative inside
        self._sdf = sdf.astype(float)

        # Gradient with explicit per-axis physical spacings -> components
        # are d(sdf)/d(axis) in coord_units per coord_unit (i.e.
        # dimensionless for a proper distance field). Sign is "points
        # toward increasing SDF", i.e. outward from the skull.
        grads = np.gradient(self._sdf, *spacing)
        self._sdf_grad = np.stack(grads, axis=0)

        # Ascending-sorted coords make sub-voxel index conversion a
        # straightforward affine mapping. If an axis is descending, we
        # flip the SDF / gradient along that axis so internal indexing
        # can always assume ascending. Also flip the corresponding
        # gradient component sign since the axis direction inverted.
        self._coord_ascending = []
        for axis_i, c in enumerate(coords):
            ascending = bool(len(c) <= 1 or c[-1] >= c[0])
            self._coord_ascending.append(ascending)
            if not ascending:
                self._sdf = np.flip(self._sdf, axis=axis_i)
                self._sdf_grad = np.flip(self._sdf_grad, axis=axis_i + 1)
                # Flipping the axis inverts the gradient component along
                # that axis.
                self._sdf_grad[axis_i] = -self._sdf_grad[axis_i]
        # Ascending-sorted coord arrays (for affine index conversion).
        self._sorted_coords = tuple(
            (c if asc else c[::-1]).astype(float)
            for c, asc in zip(coords, self._coord_ascending, strict=False)
        )

    def _world_to_index(self, point_coord: np.ndarray) -> np.ndarray:
        """Convert a world-space point (coord_units, dims order) to float
        voxel indices into the internally-ascending SDF array.

        Uses an affine mapping based on the first coordinate value and
        per-axis spacing, which is exact for uniformly sampled coord
        arrays and approximately correct for mildly non-uniform ones.
        """
        idx = np.empty(3, dtype=float)
        for j in range(3):
            c = self._sorted_coords[j]
            spacing_j = self._voxel_spacing[j] if self._voxel_spacing[j] != 0.0 else 1.0
            idx[j] = (point_coord[j] - c[0]) / spacing_j
        return idx

    def _sample_sdf(self, point_coord: np.ndarray) -> float:
        """Sample the SDF at a world-space point via linear interpolation."""
        idx = self._world_to_index(point_coord)
        val = map_coordinates(
            self._sdf, idx[:, None], order=1, mode="nearest"
        )
        return float(val[0])

    def _sample_grad(self, point_coord: np.ndarray) -> np.ndarray:
        """Sample the 3-component SDF gradient at a world-space point.

        The returned vector is in ``(dim0, dim1, dim2)`` order, matching
        ``skull_mask.dims``. Components are expressed in the
        internally-ascending axis frame (so callers that built their ray
        in the original axis frame must use the "ascending-equivalent"
        direction; we handle this centrally in ``calc_apodization``).
        """
        idx = self._world_to_index(point_coord)
        out = np.empty(3, dtype=float)
        for a in range(3):
            out[a] = map_coordinates(
                self._sdf_grad[a], idx[:, None], order=1, mode="nearest"
            )[0]
        return out

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

        # Step size in coord_units.
        mm_to_coord = getunitconversion("mm", self.coord_units)
        step = float(self.step_mm) * mm_to_coord

        # Per-axis bounds (in the original, possibly descending coord
        # frame) for quick out-of-bounds short-circuiting.
        mins = np.array([c.min() for c in coords], dtype=float)
        maxs = np.array([c.max() for c in coords], dtype=float)

        # Threshold for "degenerate" gradient magnitude: below this, we
        # consider the SDF normal undefined. Chosen as 0.1 per minimum
        # voxel size, i.e. the gradient should change the SDF by at
        # least 0.1 * min_spacing over one voxel to count as "defined".
        grad_eps = 0.1 / max(self._min_voxel_size, 1e-12)

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

            # Express the ray direction in mask axis order. ``Point.dims``
            # is the spatial-axis naming that ``ray_dir`` indexes into.
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

            # If the ray lies entirely outside the mask bounding box,
            # nothing can hit.
            # (We still march and let sign-change detection handle edges,
            #  but clamp n_steps to avoid unbounded work.)
            n_steps = max(2, int(np.ceil(total_dist / step)) + 1)
            ts = np.linspace(0.0, total_dist, n_steps)

            hit_found = False
            hit_point = None

            prev_sdf = None
            prev_point = None
            for k, t in enumerate(ts):
                p_axes = el_pos_axes + ray_dir_axes * t
                # Bounding-box short circuit. Even slightly outside is
                # fine (map_coordinates "nearest" mode will clamp), but
                # skipping early saves work.
                if np.any(p_axes < mins - step) or np.any(p_axes > maxs + step):
                    prev_sdf = None
                    prev_point = None
                    continue
                s = self._sample_sdf(p_axes)
                if prev_sdf is not None and prev_sdf > 0.0 and s <= 0.0:
                    # Sign change between prev_point (outside) and
                    # p_axes (inside). Linearly interpolate to find the
                    # zero crossing.
                    denom = prev_sdf - s
                    if denom == 0.0:
                        alpha = 0.0
                    else:
                        alpha = prev_sdf / denom
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    hit_point = prev_point + alpha * (p_axes - prev_point)
                    hit_found = True
                    break
                prev_sdf = s
                prev_point = p_axes
                if k == 0 and s <= 0.0:
                    # Element position is already inside the skull. Treat
                    # as a perpendicular hit at the element itself.
                    hit_point = p_axes
                    hit_found = True
                    break

            if not hit_found:
                weights[i] = 1.0
                continue

            # Evaluate SDF gradient at the sub-voxel hit location.
            n = self._sample_grad(hit_point)
            n_norm = float(np.linalg.norm(n))
            if n_norm < grad_eps:
                # Degenerate normal: policy determines weight.
                weights[i] = 0.0 if self.safe_on_degenerate_normal else 1.0
                continue
            n = n / n_norm

            # The internally-stored gradient is in the ascending-axis
            # frame. If an axis was descending in the original mask,
            # flip the corresponding ray-direction component so the
            # dot product is computed in the same frame as the normal.
            ray_dir_for_dot = ray_dir_axes.copy()
            for axis_i, asc in enumerate(self._coord_ascending):
                if not asc:
                    ray_dir_for_dot[axis_i] = -ray_dir_for_dot[axis_i]

            cos_theta = float(np.clip(abs(np.dot(ray_dir_for_dot, n)), 0.0, 1.0))
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
            "safe_on_degenerate_normal": self.safe_on_degenerate_normal,
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
            {"Name": "Safe On Degenerate Normal", "Value": self.safe_on_degenerate_normal, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)
