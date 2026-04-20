"""Tests for Solution.to_dict/from_dict date_created round-trip.

Regression: previously Solution.to_dict returned date_created as a raw
datetime (via asdict), while Solution.from_dict called
datetime.fromisoformat, which expects a string. Direct to_dict/from_dict
round-trips (skipping JSON) therefore failed. JSON round-trips worked by
accident because json.dumps stringified the datetime.

The fix: to_dict now serializes date_created via isoformat(); from_dict
accepts either a string or an already-constructed datetime.
"""
from __future__ import annotations

from datetime import datetime

from openlifu.plan.solution import Solution


def test_datetime_roundtrip_via_to_dict_from_dict():
    """A Solution round-tripped via to_dict/from_dict preserves date_created."""
    known_dt = datetime(2024, 5, 17, 12, 34, 56, 789000)
    sol = Solution(date_created=known_dt)

    d = sol.to_dict(include_simulation_data=False)

    # to_dict must emit date_created as a string (ISO format), so it is
    # directly consumable by from_dict without any caller-side massaging.
    assert isinstance(d["date_created"], str)
    assert d["date_created"] == known_dt.isoformat()

    loaded = Solution.from_dict(d)
    assert isinstance(loaded.date_created, datetime)
    assert loaded.date_created == known_dt


def test_datetime_string_input():
    """from_dict accepts date_created as an ISO-format string."""
    known_dt = datetime(2023, 1, 2, 3, 4, 5)
    sol = Solution(date_created=known_dt)
    d = sol.to_dict(include_simulation_data=False)

    # Force the string form (this is what json.loads would produce too).
    d["date_created"] = known_dt.isoformat()
    assert isinstance(d["date_created"], str)

    loaded = Solution.from_dict(d)
    assert isinstance(loaded.date_created, datetime)
    assert loaded.date_created == known_dt


def test_datetime_datetime_input():
    """from_dict is tolerant of date_created already being a datetime.

    This is the direct in-memory round-trip path that previously crashed.
    """
    known_dt = datetime(2022, 12, 31, 23, 59, 59)
    sol = Solution(date_created=known_dt)
    d = sol.to_dict(include_simulation_data=False)

    # Simulate a caller who replaced the serialized string with a raw
    # datetime (or who skipped the string stage some other way).
    d["date_created"] = known_dt
    assert isinstance(d["date_created"], datetime)

    loaded = Solution.from_dict(d)
    assert isinstance(loaded.date_created, datetime)
    assert loaded.date_created == known_dt
