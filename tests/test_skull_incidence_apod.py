"""Tests for SkullIncidenceApodization.

These tests build small synthetic transducers and skull masks to exercise
the ray-cast + surface-normal + incidence-angle pipeline. All geometry is
in world millimeters.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xa

from openlifu import xdc
from openlifu.bf.apod_methods import SkullIncidenceApodization
from openlifu.geo import Point


def _make_mask(values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> xa.DataArray:
    """Build a 3D xa.DataArray skull mask with (x, y, z) dims and mm coords."""
    return xa.DataArray(
        values.astype(float),
        coords={"x": x, "y": y, "z": z},
        dims=("x", "y", "z"),
    )


def _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0)) -> xdc.Transducer:
    el = xdc.Element(
        index=0,
        position=np.array(position_mm, dtype=float),
        orientation=np.array([0.0, 0.0, 0.0]),
        size=np.array([1.0, 1.0]),
        units="mm",
    )
    return xdc.Transducer(elements=[el], frequency=500_000, units="mm")


def _make_slab_mask(z_min_mm: float, z_max_mm: float, extent_mm: float = 120.0, spacing_mm: float = 1.0) -> xa.DataArray:
    """Build a skull mask that is a slab normal to the +z axis.

    The slab fills every (x, y) at z in [z_min_mm, z_max_mm]. Coords span
    [-extent_mm, extent_mm] on each axis.
    """
    axis = np.arange(-extent_mm, extent_mm + spacing_mm, spacing_mm, dtype=float)
    x, y, z = axis, axis, axis.copy()
    shape = (len(x), len(y), len(z))
    vals = np.zeros(shape, dtype=float)
    zmask = (z >= z_min_mm) & (z <= z_max_mm)
    # Broadcast along the x,y plane.
    vals[:, :, zmask] = 1.0
    return _make_mask(vals, x, y, z)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_elements_with_no_skull_in_path_get_weight_1():
    """Empty skull mask -> all rays miss -> weight 1 for every element."""
    axis = np.arange(-50.0, 51.0, 1.0)
    mask = _make_mask(np.zeros((len(axis), len(axis), len(axis))), axis, axis, axis)
    method = SkullIncidenceApodization(skull_mask=mask)

    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")
    weights = method.calc_apodization(arr, target, params=None, transform=None)
    assert weights.shape == (1,)
    assert weights[0] == pytest.approx(1.0)


def test_perpendicular_incidence_gets_weight_0():
    """Ray strikes slab head-on along +z -> angle ~0 -> weight 0."""
    mask = _make_slab_mask(z_min_mm=40.0, z_max_mm=45.0, extent_mm=60.0, spacing_mm=1.0)
    method = SkullIncidenceApodization(skull_mask=mask, min_angle_deg=20.0, rolloff_angle_deg=45.0)

    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")
    weights = method.calc_apodization(arr, target, params=None, transform=None)
    assert weights[0] == pytest.approx(0.0)


def test_oblique_incidence_gets_weight_1():
    """Ray at >=45 deg from slab normal -> angle >= rolloff -> weight 1."""
    mask = _make_slab_mask(z_min_mm=40.0, z_max_mm=45.0, extent_mm=120.0, spacing_mm=1.0)
    method = SkullIncidenceApodization(skull_mask=mask, min_angle_deg=20.0, rolloff_angle_deg=45.0)

    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    # Aim at 45 deg off the z axis: ray has equal x and z components. At the
    # slab (z=40 mm) it hits x=40 mm, which is well within extent.
    target = Point(position=np.array([100.0, 0.0, 100.0]), units="mm")
    weights = method.calc_apodization(arr, target, params=None, transform=None)
    assert weights[0] == pytest.approx(1.0)


def test_linear_rolloff_in_between():
    """Angle at midpoint of [min, rolloff] -> weight ~0.5.

    With min=20 deg and rolloff=45 deg the midpoint is 32.5 deg.
    Expected weight = (32.5 - 20) / (45 - 20) = 0.5. Some voxelization
    tolerance is allowed.
    """
    mask = _make_slab_mask(z_min_mm=40.0, z_max_mm=45.0, extent_mm=120.0, spacing_mm=0.5)
    method = SkullIncidenceApodization(
        skull_mask=mask,
        min_angle_deg=20.0,
        rolloff_angle_deg=45.0,
        step_mm=0.25,
    )

    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    theta = np.deg2rad(32.5)
    # Ray direction (sin(theta), 0, cos(theta)); target along that ray.
    target = Point(
        position=np.array([100.0 * np.sin(theta), 0.0, 100.0 * np.cos(theta)]),
        units="mm",
    )
    weights = method.calc_apodization(arr, target, params=None, transform=None)
    assert weights[0] == pytest.approx(0.5, abs=0.05)


def test_to_dict_excludes_skull_mask_and_validation():
    """to_dict omits the xarray mask; constructor validation fires as expected."""
    mask = _make_slab_mask(z_min_mm=40.0, z_max_mm=45.0, extent_mm=30.0, spacing_mm=1.0)
    method = SkullIncidenceApodization(
        skull_mask=mask,
        min_angle_deg=15.0,
        rolloff_angle_deg=50.0,
        step_mm=0.75,
        coord_units="mm",
    )

    d = method.to_dict()
    assert d["class"] == "SkullIncidenceApodization"
    assert d["min_angle_deg"] == 15.0
    assert d["rolloff_angle_deg"] == 50.0
    assert d["step_mm"] == 0.75
    assert d["coord_units"] == "mm"
    assert "skull_mask" not in d, "skull_mask is not JSON-serializable and is intentionally omitted"

    # Callers who want to rebuild the dataclass must re-supply the mask.
    rebuilt = SkullIncidenceApodization(
        skull_mask=mask,
        min_angle_deg=d["min_angle_deg"],
        rolloff_angle_deg=d["rolloff_angle_deg"],
        step_mm=d["step_mm"],
        coord_units=d["coord_units"],
    )
    assert rebuilt.min_angle_deg == 15.0
    assert rebuilt.rolloff_angle_deg == 50.0
    assert rebuilt.step_mm == 0.75

    # Validation checks
    with pytest.raises(ValueError):
        SkullIncidenceApodization(skull_mask=mask, min_angle_deg=30.0, rolloff_angle_deg=20.0)
    with pytest.raises(ValueError):
        SkullIncidenceApodization(skull_mask=mask, min_angle_deg=-1.0)
    with pytest.raises(ValueError):
        SkullIncidenceApodization(skull_mask=None)


def test_sdf_normal_is_stable_across_step_sizes():
    """Sub-voxel SDF intersection -> weights independent of step_mm.

    Previously, nearest-voxel ray marching made the reported hit voxel
    (and thus the weight) sensitive to small changes in ``step_mm``.
    With SDF-based sign-change detection + linear interpolation, the
    hit position is sub-voxel and weights should agree to within a few
    percent across different step sizes.
    """
    # Oblique ray (30 deg off normal) into a slab. Angle is within the
    # linear rolloff region so the weight is mid-scale and therefore
    # sensitive to any grid artifacts.
    mask = _make_slab_mask(z_min_mm=40.0, z_max_mm=45.0, extent_mm=120.0, spacing_mm=0.5)
    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    theta = np.deg2rad(30.0)
    target = Point(
        position=np.array([100.0 * np.sin(theta), 0.0, 100.0 * np.cos(theta)]),
        units="mm",
    )

    w_fine = SkullIncidenceApodization(
        skull_mask=mask, min_angle_deg=20.0, rolloff_angle_deg=45.0, step_mm=0.25,
    ).calc_apodization(arr, target, params=None, transform=None)[0]
    w_coarse = SkullIncidenceApodization(
        skull_mask=mask, min_angle_deg=20.0, rolloff_angle_deg=45.0, step_mm=0.5,
    ).calc_apodization(arr, target, params=None, transform=None)[0]

    # Both should be in the rolloff region and match closely.
    assert w_fine == pytest.approx(w_coarse, abs=0.05)
    # And should be near the analytic value (30 - 20)/(45 - 20) = 0.4.
    assert w_fine == pytest.approx(0.4, abs=0.1)


def test_degenerate_normal_gives_weight_zero_by_default():
    """Isolated single-voxel skull -> zero SDF gradient at the voxel ->
    safe_on_degenerate_normal=True (default) yields weight 0.

    With a single isolated skull voxel surrounded by empty voxels, the
    SDF still has a well-defined zero crossing at the voxel boundary
    and a non-zero gradient in the exterior. To guarantee a truly
    degenerate normal we place the element already inside the single
    voxel so the ray starts below the SDF zero and the hit is reported
    at the element position, where the gradient is effectively zero.
    """
    axis = np.arange(-10.0, 10.5, 1.0, dtype=float)
    vals = np.zeros((len(axis), len(axis), len(axis)), dtype=float)
    # Mark a single interior voxel at index corresponding to (0,0,0).
    ix = iy = iz = int(np.argmin(np.abs(axis - 0.0)))
    vals[ix, iy, iz] = 1.0
    mask = _make_mask(vals, axis, axis, axis)

    # Place the element at the skull-voxel center so the ray starts
    # inside the skull. map_coordinates at the center of a single
    # isolated voxel samples a gradient that is effectively zero by
    # symmetry of the SDF around the voxel center.
    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 5.0]), units="mm")

    method_safe = SkullIncidenceApodization(
        skull_mask=mask, safe_on_degenerate_normal=True,
    )
    w_safe = method_safe.calc_apodization(arr, target, params=None, transform=None)[0]
    assert w_safe == pytest.approx(0.0)

    method_unsafe = SkullIncidenceApodization(
        skull_mask=mask, safe_on_degenerate_normal=False,
    )
    w_unsafe = method_unsafe.calc_apodization(arr, target, params=None, transform=None)[0]
    assert w_unsafe == pytest.approx(1.0)


def test_smoothed_normal_on_realistic_skull_shape():
    """Ellipsoidal shell skull -> normal-incidence ray gets weight 0,
    oblique ray gets weight 1.

    Builds a thick ellipsoidal shell centered on the origin. From the
    origin, a ray straight up the z axis strikes the shell head-on
    (weight 0). A ray at 60 deg off the z axis hits the shell at a
    point where the outward normal is also roughly 60 deg off z, so
    the incidence angle is small and the weight should again be 0 or
    near zero. To get a genuinely oblique ray, shoot from a laterally
    offset element toward the origin so the ray crosses the shell at
    a grazing angle.
    """
    spacing = 1.0
    axis = np.arange(-60.0, 60.5, spacing, dtype=float)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r_outer = 50.0
    r_inner = 45.0
    a = (X * X + Y * Y + Z * Z)
    shell = (a <= r_outer * r_outer) & (a >= r_inner * r_inner)
    mask = _make_mask(shell.astype(float), axis, axis, axis)

    # Normal-incidence case: element at origin, target far up +z.
    # Ray goes straight up, hits the top of the shell head-on.
    arr_on_axis = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target_on_axis = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")
    method = SkullIncidenceApodization(
        skull_mask=mask,
        min_angle_deg=20.0,
        rolloff_angle_deg=45.0,
        step_mm=0.25,
    )
    w_normal = method.calc_apodization(arr_on_axis, target_on_axis, params=None, transform=None)[0]
    assert w_normal == pytest.approx(0.0, abs=0.05)

    # Oblique case: element offset in x, target at origin along +z such
    # that the ray direction is steeply inclined to the local shell
    # normal at the crossing point. With element at (0, 0, 45.5) just
    # grazing inside the outer shell, a target well off-axis yields
    # a strongly oblique crossing.
    el_off = xdc.Element(
        index=0,
        position=np.array([47.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        size=np.array([1.0, 1.0]),
        units="mm",
    )
    arr_off = xdc.Transducer(elements=[el_off], frequency=500_000, units="mm")
    # Target along the ellipsoid's y axis from the element position:
    # ray travels mostly in -x, +y -> at the shell crossing the outward
    # normal is close to +x while the ray is close to +y, so the angle
    # between ray and normal is near 90 deg -> weight 1.
    target_off = Point(position=np.array([47.0, 80.0, 0.0]), units="mm")
    w_oblique = method.calc_apodization(arr_off, target_off, params=None, transform=None)[0]
    assert w_oblique == pytest.approx(1.0, abs=0.05)
