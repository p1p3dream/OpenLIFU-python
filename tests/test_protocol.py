from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from openlifu import Protocol, Transducer
from openlifu.bf.focal_patterns import Wheel
from openlifu.db import Session
from openlifu.plan.protocol import OnPulseMismatchAction
from openlifu.plan.target_constraints import TargetConstraints


@pytest.fixture()
def example_protocol() -> Protocol:
    return Protocol.from_file(Path(__file__).parent/'resources/example_db/protocols/example_protocol/example_protocol.json')

@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/"resources/example_db/transducers/example_transducer/example_transducer.json")

@pytest.fixture()
def example_session() -> Session:
    return Session.from_file(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/example_session.json")

@pytest.fixture()
def example_wheel_pattern() -> Wheel:
    return Wheel(num_spokes=6)

def test_to_dict_from_dict(example_protocol: Protocol):
    proto_dict = example_protocol.to_dict()
    new_protocol = Protocol.from_dict(proto_dict)
    assert new_protocol == example_protocol

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_protocol(example_protocol : Protocol, compact_representation: bool):
    assert example_protocol.from_json(example_protocol.to_json(compact_representation)) == example_protocol

def test_default_protocol():
    """Ensure it is possible to construct a default protocol"""
    Protocol()

def test_to_table(example_protocol: Protocol):
    """Ensure that the protocol can be correctly converted to a table."""
    t = example_protocol.to_table()
    assert t is not None
    assert "Category" in t.columns
    assert "Name" in t.columns
    assert "Value" in t.columns
    assert "Unit" in t.columns
    tm =t.set_index(["Category", "Name"])
    assert tm.loc["","ID"]["Value"] == example_protocol.id
    assert tm.loc["Delay Method", "Default Sound Speed"]["Value"] == 1540.0
    assert tm.loc["Delay Method", "Default Sound Speed"]["Unit"] == "m/s"

@pytest.mark.parametrize(
    "target_constraints",
    [
        [
            TargetConstraints(dim="P", units="mm", min=0.0, max=float("inf")),
        ],
        [
            TargetConstraints(dim="P", units="m", min=-0.001, max=0.0),
        ],
        [
            TargetConstraints(dim="L", units="mm", min=-100.0, max=0.0),
            TargetConstraints(dim="P", units="mm", min=-100.0, max=0.0),
            TargetConstraints(dim="S", units="mm", min=-100.0, max=-10.0),
        ]
    ]
)
def test_check_target(example_protocol: Protocol, example_session: Session, target_constraints: TargetConstraints):
    """Ensure that the target can be correctly verified."""
    example_protocol.target_constraints = target_constraints
    with pytest.raises(ValueError, match="not within bounds"):
        example_protocol.check_target(example_session.targets[0])

@pytest.mark.parametrize("on_pulse_mismatch", [
            OnPulseMismatchAction.ERROR,
            OnPulseMismatchAction.ROUND,
            OnPulseMismatchAction.ROUNDUP,
            OnPulseMismatchAction.ROUNDDOWN
        ]
    )
def test_fix_pulse_mismatch(
        example_protocol: Protocol,
        example_session: Session,
        example_wheel_pattern: Wheel,
        on_pulse_mismatch: OnPulseMismatchAction
    ):
    """Test if sequence is correctly fixed for all pulse mismatch actions."""
    logging.disable(logging.CRITICAL)

    target = example_session.targets[0]
    foci = example_wheel_pattern.get_targets(target)
    num_foci = len(foci)
    if on_pulse_mismatch is OnPulseMismatchAction.ERROR:
        with pytest.raises(ValueError, match="not a multiple of the number of foci"):
            example_protocol.fix_pulse_mismatch(on_pulse_mismatch, foci)
    else:
        example_protocol.fix_pulse_mismatch(on_pulse_mismatch, foci)
        if on_pulse_mismatch is OnPulseMismatchAction.ROUND:
            assert example_protocol.sequence.pulse_count == num_foci
        elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDUP:
            assert example_protocol.sequence.pulse_count == 2*num_foci
        elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDDOWN:
            assert example_protocol.sequence.pulse_count == num_foci


# ---------------------------------------------------------------------------
# beamform() plumbing: delay methods that return amplitude weighting alongside
# delays must get their apod multiplicatively combined with the protocol's
# apod_method output. Backward-compatible delay methods (Direct,
# SimulationCorrected) only contribute delays; apod comes from apod_method
# alone.
# ---------------------------------------------------------------------------


class _DummyDelayMethodScalar:
    """Minimal non-ComplexWeighted delay method: returns scalar delays only."""

    def __init__(self, delays):
        self._delays = np.asarray(delays, dtype=float)

    def calc_delays(self, arr, target, params, transform=None):
        return self._delays

    def calc_delays_and_apod(self, arr, target, params, transform=None):
        # Base-class style: only delays, apod is None.
        return self.calc_delays(arr, target, params, transform=transform), None

    def to_table(self):  # pragma: no cover - not exercised here
        import pandas as pd
        return pd.DataFrame()


class _DummyComplexWeightedMethod:
    """Minimal delay method that contributes both delays and apod."""

    def __init__(self, delays, apod):
        self._delays = np.asarray(delays, dtype=float)
        self._apod = np.asarray(apod, dtype=float)

    def calc_delays(self, arr, target, params, transform=None):
        return self._delays

    def calc_delays_and_apod(self, arr, target, params, transform=None):
        return self._delays, self._apod

    def to_table(self):  # pragma: no cover - not exercised here
        import pandas as pd
        return pd.DataFrame()


class _DummyApodMethod:
    """Minimal apod method: returns a fixed per-element weight vector."""

    def __init__(self, apod):
        self._apod = np.asarray(apod, dtype=float)

    def calc_apodization(self, arr, target, params, transform=None):
        return self._apod

    def to_table(self):  # pragma: no cover - not exercised here
        import pandas as pd
        return pd.DataFrame()


class TestProtocolBeamformPlumbing:
    """Direct / scalar-return delay methods vs. tuple-return (ComplexWeighted)."""

    def test_scalar_delay_method_apod_from_apod_method_only(self):
        """A delay method that returns only delays should not influence apod;
        the apod_method output is passed through unchanged."""
        delays_in = np.array([0.0, 1e-6, 2e-6])
        apod_from_method = np.array([0.5, 1.0, 0.25])

        protocol = Protocol()
        protocol.delay_method = _DummyDelayMethodScalar(delays_in)
        protocol.apod_method = _DummyApodMethod(apod_from_method)

        delays, apod = protocol.beamform(arr=None, target=None, params=None)
        np.testing.assert_array_equal(delays, delays_in)
        np.testing.assert_array_equal(apod, apod_from_method)

    def test_complex_weighted_delay_method_multiplies_apod(self):
        """A delay method that returns (delays, apod) should have its apod
        multiplicatively combined with the apod_method output."""
        delays_in = np.array([0.0, 1e-6, 2e-6])
        delay_apod = np.array([0.5, 1.0, 0.25])
        apod_from_method = np.array([1.0, 0.8, 0.5])

        protocol = Protocol()
        protocol.delay_method = _DummyComplexWeightedMethod(delays_in, delay_apod)
        protocol.apod_method = _DummyApodMethod(apod_from_method)

        delays, apod = protocol.beamform(arr=None, target=None, params=None)
        np.testing.assert_array_equal(delays, delays_in)
        np.testing.assert_allclose(apod, delay_apod * apod_from_method)

    def test_direct_default_delay_method_backward_compatible(self):
        """Real Direct delay method should yield unmodified apod_method output
        (backward compat path: Direct's default calc_delays_and_apod returns
        (delays, None))."""
        from openlifu.bf.delay_methods import Direct

        apod_from_method = np.array([0.3, 0.7, 1.0])
        protocol = Protocol()
        protocol.delay_method = Direct(c0=1500.0)
        protocol.apod_method = _DummyApodMethod(apod_from_method)

        # Short-circuit the Direct call so we don't need a full transducer
        # fixture here; the plumbing we care about is downstream of it.
        from unittest.mock import patch
        delays_in = np.array([0.0, 1e-6, 2e-6])
        with patch.object(Direct, "calc_delays", return_value=delays_in):
            delays, apod = protocol.beamform(arr=None, target=None, params=None)

        np.testing.assert_array_equal(delays, delays_in)
        np.testing.assert_array_equal(apod, apod_from_method)
