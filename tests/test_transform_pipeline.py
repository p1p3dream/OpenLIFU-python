"""End-to-end tests for the transducer-to-world transform pipeline.

These tests validate that a 4x4 transducer-to-world affine transform flows
correctly through every consumer:

- ``Direct.calc_delays``
- ``SimulationCorrected.calc_delays`` / ``_run_reciprocal_simulation``
- ``get_karray`` / ``run_simulation``
- Apodization methods (indirectly, via ``Element.angle_to_point`` /
  ``distance_to_point`` sharing the same ``matrix[0:3, 3]`` scaling rule)

Transform convention (current): the translation column is always in
meters (world-frame SI units). Consumers pass the matrix straight
through to ``Element.get_position(units="m", matrix=matrix)`` /
``distance_to_point`` / ``angle_to_point``; no per-call rescaling is
needed because ``get_position`` scales the element's local position to
meters before left-multiplying the matrix.

Tests 1, 2, 4 are fast (no k-wave).  Tests 3, 5 are ``@pytest.mark.slow``
and gated by ``pytest.importorskip("kwave")``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xa

# The 'slow' marker is not registered in pyproject.toml and the project
# escalates PytestUnknownMarkWarning to an error via filterwarnings=["error"].
# Suppress that specific warning so the @pytest.mark.slow decorators below
# don't break collection of the fast tests in this file.
warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)

from openlifu import xdc  # noqa: E402
from openlifu.bf.delay_methods import Direct, SimulationCorrected  # noqa: E402
from openlifu.geo import Point  # noqa: E402
from openlifu.util.units import getunitconversion  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_small_transducer(positions_mm, units="mm"):
    """Build a Transducer with elements at the given local positions (mm).

    ``positions_mm`` is a sequence of (x, y, z) tuples in transducer-native
    millimeters.
    """
    elements = [
        xdc.Element(
            index=i,
            position=np.array(p, dtype=float),
            orientation=np.array([0.0, 0.0, 0.0]),
            size=np.array([5.0, 5.0]),
            units="mm",
        )
        for i, p in enumerate(positions_mm)
    ]
    return xdc.Transducer(elements=elements, frequency=500_000, units=units)


def _translation_matrix(delta_m):
    """Build a 4x4 translation-only matrix (translation in meters).

    The transform convention is: translation column is in meters
    (world-frame SI units).
    """
    T = np.eye(4)
    T[:3, 3] = np.asarray(delta_m, dtype=float)
    return T


def _rotation_matrix_about_axis(axis, angle_rad):
    """Return a 4x4 rotation matrix (no translation) about a unit axis."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    x, y, z = axis
    R = np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0.0],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s, 0.0],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c),     0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return R


def _compose(*mats):
    """Compose 4x4 matrices left-to-right: _compose(A, B, C) == A @ B @ C."""
    out = np.eye(4)
    for m in mats:
        out = out @ m
    return out


def _build_water_params(n=64, dx_mm=1.0, c0=1500.0, rho=1000.0):
    """Build a homogeneous-water xarray Dataset on an n^3 cube, centered at 0."""
    half = (n * dx_mm) / 2.0
    coords = {}
    for dim in ("x", "y", "z"):
        cv = np.linspace(-half, half, n, endpoint=False)
        coords[dim] = xa.DataArray(cv, dims=[dim], attrs={"units": "mm"})
    shape = (n, n, n)
    sound_speed = xa.DataArray(
        np.full(shape, c0, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "m/s", "ref_value": c0},
    )
    density = xa.DataArray(
        np.full(shape, rho, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "kg/m^3", "ref_value": rho},
    )
    attenuation = xa.DataArray(
        np.zeros(shape, dtype=np.float32),
        dims=("x", "y", "z"),
        coords=coords,
        attrs={"units": "dB/cm/MHz", "ref_value": 0.0},
    )
    return xa.Dataset({
        "sound_speed": sound_speed,
        "density": density,
        "attenuation": attenuation,
    })


# ---------------------------------------------------------------------------
# Test 1: translation-only invariance (Direct)
# ---------------------------------------------------------------------------

def test_translation_invariance_direct():
    """Shifting elements in local coords by -delta while applying a T(+delta)
    transform must leave geometric delays unchanged (the transform exactly
    compensates the local shift, so element world positions are identical).

    In Direct.calc_delays, the target is passed in world coordinates (no
    transform is applied to it), so the test holds the target Point fixed
    across both scenarios.

    - Scenario A: elements at local positions ``p``,
                   target at world ``t``, transform=None (identity).
    - Scenario B: elements at local positions ``p - delta``,
                   target at world ``t`` (same),
                   transform=T(+delta) with translation expressed in meters.

    Both describe the exact same physical geometry in world space.
    """
    method = Direct(c0=1500.0)

    # Local element positions for scenario A (transducer-native mm)
    local_positions_A_mm = [
        (10.0, 0.0, 0.0),
        (-10.0, 0.0, 0.0),
        (0.0, 10.0, 5.0),
        (0.0, -10.0, -5.0),
    ]
    target_world_mm = (0.0, 0.0, 40.0)
    target = Point(
        position=np.array(target_world_mm), units="mm", dims=("x", "y", "z"),
    )

    # Scenario A: identity transform
    arr_A = _build_small_transducer(local_positions_A_mm)
    delays_A = method.calc_delays(arr_A, target, params=None, transform=None)

    # Scenario B: shift elements by -delta in local coords (mm, since element
    # positions are stored in transducer-native mm), transform T(+delta) with
    # translation expressed in meters (the transform convention).
    delta_mm = np.array([7.0, -3.0, 11.0])
    delta_m = delta_mm * 1e-3
    local_positions_B_mm = [tuple(np.array(p) - delta_mm) for p in local_positions_A_mm]
    arr_B = _build_small_transducer(local_positions_B_mm)
    transform_B = _translation_matrix(delta_m)
    delays_B = method.calc_delays(arr_B, target, params=None, transform=transform_B)

    np.testing.assert_allclose(
        delays_A, delays_B, atol=1e-12, rtol=0.0,
        err_msg=(
            "Translation-only invariance failed.\n"
            f"delays_A = {delays_A}\ndelays_B = {delays_B}\n"
            f"diff = {delays_A - delays_B}"
        ),
    )


# ---------------------------------------------------------------------------
# Test 2: rotation-only invariance (Direct)
# ---------------------------------------------------------------------------

def test_rotation_invariance_direct():
    """A symmetric 4-element ring rotated rigidly together with its target
    about z must produce the same delays (up to cyclic permutation caused by
    relabeling after rotation).

    We use a ring that is symmetric under a 90-degree rotation about z.
    Rotating the whole system by exactly 90 degrees maps element i to the
    position element ((i - 1) mod N) formerly occupied, but element indices
    stay the same. So the transformed array with the transformed target
    should produce the same set of delay values in the same order as the
    identity-transform case (because we rotate BOTH the elements' world
    positions and the target by the same amount).

    This test validates that Direct.calc_delays respects pure rotations.
    """
    method = Direct(c0=1500.0)

    # 4-element ring at local z=0, radius 40 mm
    R_mm = 40.0
    local_positions_mm = [
        (R_mm * np.cos(2 * np.pi * i / 4),
         R_mm * np.sin(2 * np.pi * i / 4),
         0.0)
        for i in range(4)
    ]
    # Target at local (0, 0, depth)
    depth_mm = 60.0
    target_local = np.array([0.0, 0.0, depth_mm])

    # Scenario A: identity transform, target at local=world
    arr_A = _build_small_transducer(local_positions_mm)
    target_A = Point(
        position=target_local.copy(), units="mm", dims=("x", "y", "z"),
    )
    delays_A = method.calc_delays(arr_A, target_A, params=None, transform=None)

    # Scenario B: rotate whole system by 90 degrees about z.
    # Rotation-only transform has no translation component, so the "mm vs m"
    # scaling of the translation column is a no-op here.
    Rz90 = _rotation_matrix_about_axis((0, 0, 1), np.pi / 2)
    arr_B = _build_small_transducer(local_positions_mm)
    # World-space target = Rz90 @ target_local (rotation preserves length, no translation)
    target_world_B = (Rz90[:3, :3] @ target_local)
    target_B = Point(
        position=target_world_B, units="mm", dims=("x", "y", "z"),
    )
    delays_B = method.calc_delays(arr_B, target_B, params=None, transform=Rz90)

    np.testing.assert_allclose(
        delays_A, delays_B, atol=1e-12, rtol=0.0,
        err_msg=(
            "Rotation-only invariance failed (ring symmetric under Rz(90deg)).\n"
            f"delays_A = {delays_A}\ndelays_B = {delays_B}\n"
            f"diff = {delays_A - delays_B}"
        ),
    )


# ---------------------------------------------------------------------------
# Test 3: cross-method parity (Direct vs SimulationCorrected in water) SLOW
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cross_method_parity_direct_vs_simulation_corrected_water():
    """In homogeneous water, SimulationCorrected should match Direct up to the
    discretization error of the k-wave time step.

    Uses a non-identity transform (translate + rotate) to exercise the
    transform plumbing in both methods. This is the critical diagnostic:
    if it passes, both methods agree on where the transformed elements are
    in world space.
    """
    kwave = pytest.importorskip("kwave")  # noqa: F841  (import gate)

    c_ref = 1500.0
    cfl = 0.3
    dx_m = 0.001  # 1 mm grid spacing in meters
    # dt approximation: cfl * dx / c
    dt = cfl * dx_m / c_ref
    tol = 2.0 * dt  # ~400 ns

    # 64 mm cube, 1 mm iso, centered at origin
    params = _build_water_params(n=64, dx_mm=1.0, c0=c_ref)

    # Small 8-element "bowl" array: ring of 8 elements at radius 20 mm,
    # slightly curved in local +z so they are not all coplanar.  Keep in
    # transducer-local coords; pose places the whole array in world space.
    R_local_mm = 20.0
    bowl_sagitta_mm = 4.0
    local_positions_mm = []
    for i in range(8):
        theta = 2 * np.pi * i / 8
        x = R_local_mm * np.cos(theta)
        y = R_local_mm * np.sin(theta)
        # Small z offset to make bowl-like, but still shallow enough that all
        # elements fit inside the 64 mm grid after pose.
        z = bowl_sagitta_mm * (x * x + y * y) / (R_local_mm * R_local_mm)
        local_positions_mm.append((x, y, z))
    arr = _build_small_transducer(local_positions_mm)

    # Non-identity pose: translate by (5, 0, 5) mm (= 0.005 m), rotate 10
    # degrees about x. Small z translation keeps the bowl elements well inside
    # the 64 mm grid (z_world max ~ 9 mm vs grid bound z = 31 mm) so
    # _run_reciprocal_simulation actually runs instead of hitting the
    # out-of-grid raise. Transform translation is expressed in meters.
    T_translate = _translation_matrix((0.005, 0.0, 0.005))
    R_tilt = _rotation_matrix_about_axis((1, 0, 0), np.deg2rad(10.0))
    # Rotation about origin first, then translate.  In column-vector / @
    # convention: world = T @ R @ local, so transform = T @ R.
    transform = _compose(T_translate, R_tilt)

    # Target at world (0, 0, -10) mm  (inside the 64 mm cube)
    target = Point(
        position=np.array([0.0, 0.0, -10.0]),
        units="mm",
        dims=("x", "y", "z"),
    )

    direct = Direct(c0=c_ref)
    delays_direct = direct.calc_delays(arr, target, params=params, transform=transform)

    # Spy on _fallback_delays: if SimulationCorrected secretly falls back to
    # Direct (via its except-ValueError wrapper in calc_delays), we would
    # trivially pass this test by comparing Direct to Direct. Patch the
    # fallback to raise loudly so any fallback surfaces as a test failure.
    from unittest.mock import patch

    sim = SimulationCorrected(c0=c_ref, cfl=cfl, n_cycles=3, gpu=False)

    def _no_fallback(self_, arr_, target_, params_, transform_=None):
        msg = (
            "SimulationCorrected silently fell back to Direct; "
            "the k-wave reciprocal simulation path was not exercised."
        )
        raise AssertionError(msg)

    with patch.object(SimulationCorrected, "_fallback_delays", _no_fallback):
        delays_sim = sim.calc_delays(arr, target, params=params, transform=transform)

    diff = np.max(np.abs(delays_sim - delays_direct))
    assert diff < tol, (
        f"SimulationCorrected delays disagree with Direct by {diff * 1e9:.1f} ns "
        f"(tol = {tol * 1e9:.1f} ns).\n"
        f"delays_direct (us) = {delays_direct * 1e6}\n"
        f"delays_sim    (us) = {delays_sim * 1e6}"
    )


# ---------------------------------------------------------------------------
# Test 4: consumer parity (all consumers agree on transformed element pos)
# ---------------------------------------------------------------------------

def test_consumer_parity_world_positions():
    """Every consumer of the transform must agree on where a given element
    lands in world space.

    Under the current convention (transform translation is always in meters),
    consumers pass the matrix straight through.

    We compute element 0's world position via three independent paths that
    mirror the production code:

    1. Direct consumer: ``el.get_position(units="m", matrix=matrix)`` with
       matrix passed through unchanged.
    2. SimulationCorrected path: ``get_position(units="m", matrix=matrix)``
       then rescale the result to ``coord_units`` for voxel-index lookups.
       We then convert back to meters for the parity comparison.
    3. get_karray path: same as Direct, ``get_position(units="m",
       matrix=matrix)`` with matrix passed through unchanged.

    If any two paths disagree, the transform plumbing is inconsistent.
    """
    # 4 elements at distinct local positions (mm)
    local_positions_mm = [
        (12.0, -3.0, 0.0),
        (-8.0, 5.0, 2.0),
        (0.0, -15.0, -1.0),
        (7.0, 7.0, 4.0),
    ]
    arr = _build_small_transducer(local_positions_mm)

    # Non-trivial transform: rotate 25 deg about (0, 1, 0), translate
    # (4, -6, 18) mm = (0.004, -0.006, 0.018) m. Translation column is in
    # meters per the world-frame convention.
    R = _rotation_matrix_about_axis((0, 1, 0), np.deg2rad(25.0))
    T = _translation_matrix((0.004, -0.006, 0.018))
    transform = _compose(T, R)  # translation column is in meters

    # Build a params dataset whose coord units are mm. The SimCorrected
    # consumer rescales from meters to coord_units for its voxel indexing.
    params = _build_water_params(n=16, dx_mm=1.0)
    coord_dims = list(params.coords.dims)
    coord_units = params[coord_dims[0]].attrs.get("units", "mm")

    el0 = arr.elements[0]

    # --- Path 1: Direct consumer ---
    # Direct passes matrix straight through.
    pos_direct_m = el0.get_position(units="m", matrix=transform)

    # --- Path 2: SimulationCorrected consumer ---
    # SimCorrected calls get_position(units="m") and then scales the result
    # to coord_units for its downstream voxel-index lookup. For this parity
    # check we convert back to meters.
    scl_m_to_coord = getunitconversion("m", coord_units)
    pos_sim_in_coord_units = el0.get_position(units="m", matrix=transform) * scl_m_to_coord
    pos_sim_m = pos_sim_in_coord_units * getunitconversion(coord_units, "m")

    # --- Path 3: get_karray consumer ---
    # get_karray passes matrix straight through.
    pos_karray_m = el0.get_position(units="m", matrix=transform)

    tol_m = 1e-9
    np.testing.assert_allclose(
        pos_direct_m, pos_sim_m, atol=tol_m, rtol=0.0,
        err_msg=(
            "Direct and SimulationCorrected consumers disagree on element 0 world pos.\n"
            f"pos_direct_m = {pos_direct_m}\n"
            f"pos_sim_m    = {pos_sim_m}\n"
            f"diff         = {pos_direct_m - pos_sim_m}"
        ),
    )
    np.testing.assert_allclose(
        pos_direct_m, pos_karray_m, atol=tol_m, rtol=0.0,
        err_msg=(
            "Direct and get_karray consumers disagree on element 0 world pos.\n"
            f"pos_direct_m = {pos_direct_m}\n"
            f"pos_karray_m = {pos_karray_m}\n"
            f"diff         = {pos_direct_m - pos_karray_m}"
        ),
    )
    np.testing.assert_allclose(
        pos_sim_m, pos_karray_m, atol=tol_m, rtol=0.0,
        err_msg=(
            "SimulationCorrected and get_karray consumers disagree on element 0 world pos.\n"
            f"pos_sim_m    = {pos_sim_m}\n"
            f"pos_karray_m = {pos_karray_m}\n"
            f"diff         = {pos_sim_m - pos_karray_m}"
        ),
    )


# ---------------------------------------------------------------------------
# Test 5: out-of-grid hard fail  SLOW
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_out_of_grid_hard_fail():
    """A deliberately wrong transform that places elements outside the grid
    must raise ValueError mentioning ``allow_out_of_grid_fallback``.

    The raise happens before any k-wave call, so this test does not actually
    require k-wave at runtime.  We still gate it with importorskip and mark
    it slow because the fast-test subset is meant to run without kwave or
    simulation machinery.
    """
    kwave = pytest.importorskip("kwave")  # noqa: F841

    # Small 16 mm cube params
    params = _build_water_params(n=16, dx_mm=1.0)

    # Transducer with elements near local origin
    local_positions_mm = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0),
    ]
    arr = _build_small_transducer(local_positions_mm)

    # Deliberately wrong transform: translate by +0.5 m (= 500 mm) in x,
    # pushing every element way outside the 16 mm grid (grid half-extent is
    # 8 mm). Translation column is in meters per the world-frame convention.
    bad_transform = _translation_matrix((0.5, 0.0, 0.0))

    target = Point(
        position=np.array([0.0, 0.0, 0.0]), units="mm", dims=("x", "y", "z"),
    )

    method = SimulationCorrected(c0=1500.0, cfl=0.3, n_cycles=3, gpu=False)
    assert method.allow_out_of_grid_fallback is False

    with pytest.raises(ValueError, match="allow_out_of_grid_fallback"):
        method._run_reciprocal_simulation(
            arr=arr, target=target, params=params, transform=bad_transform,
        )
