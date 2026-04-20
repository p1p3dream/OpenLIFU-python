"""Tests for transform plumbing through get_point_source.

These tests verify that ``openlifu.sim.kwave_if.get_point_source`` correctly
applies the ``transform`` kwarg to element positions before mapping them to
grid voxels.  They complement ``test_transform_pipeline.py``, which covers
``get_karray`` / ``run_simulation`` with the ``source_method='kwave_array'``
branch.  The ``source_method='point_source'`` branch had a parallel pose
hole prior to this change: ``get_point_source`` read element positions via
``el.get_position(units=coord_units)`` with no matrix, so any caller-supplied
transform passed to ``run_simulation`` was silently dropped.

No actual k-wave simulation is run here; we build minimal xarray and
transducer fixtures and inspect ``kSource.p_mask`` directly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xa

# The 'slow' marker is not registered in pyproject.toml and the project
# escalates PytestUnknownMarkWarning to an error via filterwarnings=["error"].
warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)

pytest.importorskip("kwave")  # get_point_source imports kwave.ksource

from openlifu import xdc  # noqa: E402
from openlifu.sim.kwave_if import get_point_source  # noqa: E402


def _build_params_grid(n=31, half_extent_mm=15.0):
    """Build a small cubic params dataset centered at origin (units=mm)."""
    coords = {}
    for dim in ("x", "y", "z"):
        cv = np.linspace(-half_extent_mm, half_extent_mm, n, endpoint=True)
        coords[dim] = xa.DataArray(cv, dims=[dim], attrs={"units": "mm"})
    shape = (n, n, n)
    sound_speed = xa.DataArray(
        np.full(shape, 1500.0, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "m/s", "ref_value": 1500.0},
    )
    density = xa.DataArray(
        np.full(shape, 1000.0, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "kg/m^3", "ref_value": 1000.0},
    )
    return xa.Dataset({"sound_speed": sound_speed, "density": density})


def _build_small_transducer(positions_mm):
    """Transducer with elements at the given local positions (mm)."""
    elements = [
        xdc.Element(
            position=np.array(p, dtype=float),
            size=np.array([1.0, 1.0]),
            units="mm",
        )
        for p in positions_mm
    ]
    return xdc.Transducer(elements=elements, frequency=500_000, units="mm")


def _nonzero_world_positions_mm(p_mask, params):
    """Extract nonzero voxel positions from a [x,y,z]-ordered p_mask.

    Returns an (N, 3) array of (x, y, z) coordinates in mm.
    """
    xs = params.coords["x"].to_numpy()
    ys = params.coords["y"].to_numpy()
    zs = params.coords["z"].to_numpy()
    ix, iy, iz = np.nonzero(p_mask)
    return np.stack([xs[ix], ys[iy], zs[iz]], axis=-1)


def test_identity_transform_gives_same_positions_as_no_transform():
    """transform=None and transform=np.eye(4) must produce identical p_mask."""
    params = _build_params_grid()
    # A few elements scattered near the grid center so they all land inside.
    arr = _build_small_transducer(
        [(0.0, 0.0, 0.0), (2.0, -1.0, 3.0), (-4.0, 2.0, -2.0)]
    )
    source_mat = np.zeros((arr.numelements(), 16), dtype=np.float32)
    source_mat[:, 0] = 1.0  # nonzero so signal_matrix isn't pathological

    src_none = get_point_source(arr, params, source_mat, transform=None)
    src_eye = get_point_source(arr, params, source_mat, transform=np.eye(4))

    assert src_none.p_mask.shape == src_eye.p_mask.shape
    np.testing.assert_array_equal(src_none.p_mask, src_eye.p_mask)
    # Signal matrix shape should also match (same number of source voxels)
    assert src_none.p.shape == src_eye.p.shape


def test_non_identity_transform_shifts_positions():
    """A +10 mm world-frame translation in x should shift nonzero voxels +10 mm in x."""
    params = _build_params_grid()
    # Elements in the transducer-local frame, clustered near origin.
    local_positions_mm = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)]
    arr = _build_small_transducer(local_positions_mm)
    source_mat = np.zeros((arr.numelements(), 16), dtype=np.float32)
    source_mat[:, 0] = 1.0

    src_identity = get_point_source(arr, params, source_mat, transform=np.eye(4))

    # Phase B convention: translation column is in meters. 10 mm = 0.01 m.
    shift_m = np.array([0.01, 0.0, 0.0])
    transform = np.eye(4)
    transform[0:3, 3] = shift_m

    src_shifted = get_point_source(arr, params, source_mat, transform=transform)

    pos_identity = _nonzero_world_positions_mm(src_identity.p_mask, params)
    pos_shifted = _nonzero_world_positions_mm(src_shifted.p_mask, params)

    # Same number of source voxels (no elements should fall off the grid for
    # this 20 mm half-extent grid with a 10 mm shift of 3 elements spanning
    # x in [-2, 2] mm).
    assert pos_identity.shape == pos_shifted.shape
    assert pos_identity.shape[0] == len(local_positions_mm)

    # Sort by x so we can pair identity voxels with shifted voxels.
    id_sorted = pos_identity[np.argsort(pos_identity[:, 0])]
    sh_sorted = pos_shifted[np.argsort(pos_shifted[:, 0])]

    delta = sh_sorted - id_sorted  # expected: (+10, 0, 0) mm per voxel

    # Voxel spacing in x is (20 mm) / 20 steps = 1 mm; shift resolves cleanly.
    np.testing.assert_allclose(delta[:, 0], 10.0, atol=1.0)
    np.testing.assert_allclose(delta[:, 1], 0.0, atol=1.0)
    np.testing.assert_allclose(delta[:, 2], 0.0, atol=1.0)
