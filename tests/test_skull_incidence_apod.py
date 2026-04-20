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
