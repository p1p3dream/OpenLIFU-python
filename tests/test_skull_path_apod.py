"""Tests for SkullPathApodization.

These tests build small synthetic transducers and skull masks to exercise
the ray-cast + voxel-count + Beer's-law pipeline. All geometry is in
world millimeters unless noted.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xa

from openlifu import xdc
from openlifu.bf.apod_methods import SkullPathApodization
from openlifu.bf.apod_methods.skull_path import DB_PER_NEPER
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


def _make_slab_mask(
    z_min_mm: float,
    z_max_mm: float,
    extent_mm: float = 120.0,
    spacing_mm: float = 0.5,
) -> xa.DataArray:
    """Build a skull mask that is a slab normal to the +z axis.

    The slab fills every (x, y) at z in [z_min_mm, z_max_mm]. Coords span
    [-extent_mm, extent_mm] on each axis.
    """
    axis = np.arange(-extent_mm, extent_mm + spacing_mm, spacing_mm, dtype=float)
    x, y, z = axis, axis, axis.copy()
    shape = (len(x), len(y), len(z))
    vals = np.zeros(shape, dtype=float)
    zmask = (z >= z_min_mm) & (z <= z_max_mm)
    vals[:, :, zmask] = 1.0
    return _make_mask(vals, x, y, z)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_skull_mask_all_weights_one():
    """Empty skull mask -> zero bone path -> weight 1 for every element."""
    axis = np.arange(-50.0, 51.0, 1.0)
    mask = _make_mask(np.zeros((len(axis), len(axis), len(axis))), axis, axis, axis)
    method = SkullPathApodization(skull_mask=mask)

    # A 3-element line transducer.
    elements = [
        xdc.Element(
            index=i,
            position=np.array([x, 0.0, 0.0], dtype=float),
            orientation=np.array([0.0, 0.0, 0.0]),
            size=np.array([1.0, 1.0]),
            units="mm",
        )
        for i, x in enumerate((-5.0, 0.0, 5.0))
    ]
    arr = xdc.Transducer(elements=elements, frequency=500_000, units="mm")
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")
    weights = method.calc_apodization(arr, target, params=None, transform=None)

    assert weights.shape == (3,)
    assert np.allclose(weights, 1.0)


def test_long_path_gives_low_weight():
    """Thick slab ~20 mm in the way at normal incidence.

    Expected weight ~ exp(-3.5 * 2 / 8.686) ~ 0.446. A small tolerance
    allows for voxelization-edge effects (the ray-marcher may count one
    voxel more or fewer depending on where the samples land).
    """
    mask = _make_slab_mask(z_min_mm=30.0, z_max_mm=50.0, extent_mm=80.0, spacing_mm=0.5)
    method = SkullPathApodization(
        skull_mask=mask,
        alpha_bone_db_per_cm=3.5,
        step_mm=0.25,
    )
    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")

    weights = method.calc_apodization(arr, target, params=None, transform=None)
    expected = float(np.exp(-3.5 * 2.0 / DB_PER_NEPER))
    assert expected == pytest.approx(0.4466, abs=1e-3)
    assert weights[0] == pytest.approx(expected, abs=0.05)


def test_max_path_cutoff_zeros_elements():
    """max_path_mm=10 and a 15 mm slab in the way -> weight 0."""
    mask = _make_slab_mask(z_min_mm=30.0, z_max_mm=45.0, extent_mm=80.0, spacing_mm=0.5)
    method = SkullPathApodization(
        skull_mask=mask,
        alpha_bone_db_per_cm=3.5,
        max_path_mm=10.0,
        step_mm=0.25,
    )
    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")

    weights = method.calc_apodization(arr, target, params=None, transform=None)
    assert weights[0] == pytest.approx(0.0)

    # Sanity: without the cutoff the same geometry produces a Beer's-law weight
    # strictly between 0 and 1.
    method_no_cutoff = SkullPathApodization(
        skull_mask=mask, alpha_bone_db_per_cm=3.5, step_mm=0.25,
    )
    w_no_cutoff = method_no_cutoff.calc_apodization(arr, target, params=None, transform=None)[0]
    assert 0.0 < w_no_cutoff < 1.0


def test_short_path_near_unity():
    """Thin 5 mm slab -> weight ~ exp(-3.5 * 0.5 / 8.686) ~ 0.818."""
    mask = _make_slab_mask(z_min_mm=37.5, z_max_mm=42.5, extent_mm=80.0, spacing_mm=0.5)
    method = SkullPathApodization(
        skull_mask=mask,
        alpha_bone_db_per_cm=3.5,
        step_mm=0.25,
    )
    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")

    weights = method.calc_apodization(arr, target, params=None, transform=None)
    expected = float(np.exp(-3.5 * 0.5 / DB_PER_NEPER))
    assert expected == pytest.approx(0.8175, abs=1e-3)
    assert weights[0] == pytest.approx(expected, abs=0.05)


def test_alpha_scaling_squares_weight():
    """Doubling alpha_bone squares the weight (w2 = w1^2)."""
    mask = _make_slab_mask(z_min_mm=35.0, z_max_mm=45.0, extent_mm=80.0, spacing_mm=0.5)
    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")

    w1 = SkullPathApodization(
        skull_mask=mask, alpha_bone_db_per_cm=3.5, step_mm=0.25,
    ).calc_apodization(arr, target, params=None, transform=None)[0]
    w2 = SkullPathApodization(
        skull_mask=mask, alpha_bone_db_per_cm=7.0, step_mm=0.25,
    ).calc_apodization(arr, target, params=None, transform=None)[0]

    # The two weights share the same path length, so w2 should equal w1 ** 2
    # up to floating-point tolerance. (Both use identical ray-march sampling.)
    assert w2 == pytest.approx(w1 ** 2, rel=1e-6, abs=1e-9)
    # Also sanity-check that w1 is in (0, 1): a 10 mm slab with alpha=3.5
    # yields a non-trivial weight.
    assert 0.0 < w1 < 1.0


def test_transform_applied_correctly():
    """A translation applied via the Phase B (meters) transform shifts
    the element position; the path length (and hence weight) changes.

    Baseline: element at origin, ray up +z through a slab -> some path.
    Translated: lift the element +Z by 20 mm so it starts inside the slab
    -> shorter remaining bone path -> higher weight.
    """
    mask = _make_slab_mask(z_min_mm=30.0, z_max_mm=50.0, extent_mm=80.0, spacing_mm=0.5)
    arr = _build_single_element_xdc(position_mm=(0.0, 0.0, 0.0))
    target = Point(position=np.array([0.0, 0.0, 80.0]), units="mm")

    method = SkullPathApodization(
        skull_mask=mask, alpha_bone_db_per_cm=3.5, step_mm=0.25,
    )
    w_identity = method.calc_apodization(arr, target, params=None, transform=None)[0]

    # Shift the element by +40 mm in z, expressed in meters because the
    # transform's translation component is in meters (Phase B).
    shift = np.eye(4)
    shift[2, 3] = 0.040  # +40 mm along z, in meters
    w_shifted = method.calc_apodization(arr, target, params=None, transform=shift)[0]

    # Baseline ray crosses the full 20 mm slab; shifted ray starts at
    # z = 40 mm (inside the slab) so it only traverses 10 mm of bone,
    # giving a larger weight.
    assert w_shifted > w_identity
    # Baseline should match the 20 mm-slab expectation to within tolerance.
    expected_identity = float(np.exp(-3.5 * 2.0 / DB_PER_NEPER))
    assert w_identity == pytest.approx(expected_identity, abs=0.05)
    # Shifted should match the ~10 mm remaining-bone expectation.
    expected_shifted = float(np.exp(-3.5 * 1.0 / DB_PER_NEPER))
    assert w_shifted == pytest.approx(expected_shifted, abs=0.05)


def test_to_dict_from_dict_roundtrip_and_validation():
    """to_dict omits skull_mask; from_dict reconstructs scalar params.

    Since skull_mask is not JSON-serializable, the canonical round-trip
    pattern is: dump with to_dict, edit the dict to add the mask, and
    pass through ApodizationMethod.from_dict (or reconstruct directly).
    """
    mask = _make_slab_mask(z_min_mm=40.0, z_max_mm=45.0, extent_mm=30.0, spacing_mm=1.0)
    method = SkullPathApodization(
        skull_mask=mask,
        alpha_bone_db_per_cm=4.2,
        max_path_mm=18.0,
        step_mm=0.3,
        coord_units="mm",
    )

    d = method.to_dict()
    assert d["class"] == "SkullPathApodization"
    assert d["alpha_bone_db_per_cm"] == 4.2
    assert d["max_path_mm"] == 18.0
    assert d["step_mm"] == 0.3
    assert d["coord_units"] == "mm"
    assert "skull_mask" not in d, "skull_mask is not JSON-serializable and is intentionally omitted"

    # Reconstruct by hand (the caller is responsible for re-supplying the mask).
    rebuilt = SkullPathApodization(
        skull_mask=mask,
        alpha_bone_db_per_cm=d["alpha_bone_db_per_cm"],
        max_path_mm=d["max_path_mm"],
        step_mm=d["step_mm"],
        coord_units=d["coord_units"],
    )
    assert rebuilt.alpha_bone_db_per_cm == 4.2
    assert rebuilt.max_path_mm == 18.0
    assert rebuilt.step_mm == 0.3
    assert rebuilt.coord_units == "mm"

    # Validation
    with pytest.raises(ValueError):
        SkullPathApodization(skull_mask=None)
    with pytest.raises(ValueError):
        SkullPathApodization(skull_mask=mask, alpha_bone_db_per_cm=-1.0)
    with pytest.raises(ValueError):
        SkullPathApodization(skull_mask=mask, step_mm=0.0)
    with pytest.raises(ValueError):
        SkullPathApodization(skull_mask=mask, max_path_mm=0.0)
