"""Tests for the legacy mm->m migration shim in Solution.from_dict.

Phase B (commit c89e91b) changed the convention for Solution.transform:
the 4x4 transform's translation column is now always in meters. Solutions
serialized before that commit have translation in transducer-native units
(typically mm) and would be misinterpreted by 1000x under the new code.

Solution.from_dict heuristically detects the legacy format by translation
magnitude (>1 m is implausible for human-scale poses) and rescales.
"""
from __future__ import annotations

import logging

import numpy as np

from openlifu.plan.solution import Solution


def _default_solution_dict_with_transform(transform: np.ndarray) -> dict:
    """Build a minimal Solution dict suitable for from_dict, with a chosen transform.

    Solution.from_dict expects date_created as an isoformat string (as it
    would be after json.loads), but asdict/to_dict preserves it as a datetime
    object. Normalize here so we can exercise from_dict without going through JSON.
    """
    sol = Solution(transform=transform)
    d = sol.to_dict(include_simulation_data=False)
    d["date_created"] = d["date_created"].isoformat()
    return d


def test_meters_convention_round_trip_unchanged():
    """A transform already in meters (translation magnitude < 1) must pass through unchanged."""
    T = np.eye(4)
    T[0:3, 3] = [0.05, 0.0, 0.1]  # 5 cm, 0, 10 cm - realistic human-scale in meters

    d = _default_solution_dict_with_transform(T)
    loaded = Solution.from_dict(d)

    assert loaded.transform is not None
    np.testing.assert_allclose(loaded.transform, T, atol=1e-12, rtol=0.0)
    # Specifically the translation column
    np.testing.assert_allclose(loaded.transform[0:3, 3], [0.05, 0.0, 0.1], atol=1e-12)


def test_legacy_mm_convention_converted(caplog):
    """A transform with translation magnitude > 1 (legacy mm) must be scaled by 1e-3 with a warning."""
    # Hand-build a dict with legacy-mm translations: 50 mm, 0, 100 mm
    T_legacy = np.eye(4)
    T_legacy[0:3, 3] = [50.0, 0.0, 100.0]  # legacy mm numerics

    d = _default_solution_dict_with_transform(T_legacy)

    with caplog.at_level(logging.WARNING, logger="openlifu.plan.solution"):
        loaded = Solution.from_dict(d)

    assert loaded.transform is not None
    # Expect translation column to now be in meters: [0.05, 0.0, 0.1]
    np.testing.assert_allclose(
        loaded.transform[0:3, 3], [0.05, 0.0, 0.1], atol=1e-9, rtol=0.0
    )
    # Rotation/linear block (identity here) must be untouched
    np.testing.assert_allclose(loaded.transform[0:3, 0:3], np.eye(3), atol=1e-12)
    # The bottom row must remain [0, 0, 0, 1]
    np.testing.assert_allclose(loaded.transform[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    # A warning mentioning the legacy mm assumption should have been emitted
    warning_messages = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("legacy mm" in msg.lower() or "mm convention" in msg.lower() for msg in warning_messages), (
        f"Expected a legacy-mm warning, got: {warning_messages}"
    )


def test_identity_transform_unchanged():
    """Identity transform (translation magnitude 0) must round-trip unchanged."""
    T = np.eye(4)

    d = _default_solution_dict_with_transform(T)
    loaded = Solution.from_dict(d)

    assert loaded.transform is not None
    np.testing.assert_allclose(loaded.transform, np.eye(4), atol=1e-12, rtol=0.0)
