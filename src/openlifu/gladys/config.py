"""GLADYS-specific constants and default protocol configuration.

Defines the sonodynamic therapy (SDT) parameters derived from the GLADYS
hardware platform and the Wu et al. 2025 optimal pulse characterization.
Provides factory functions to build a fully configured Protocol with
GLADYS defaults.
"""

from __future__ import annotations

from openlifu.bf import Pulse, Sequence
from openlifu.bf.delay_methods.complex_weighted import ComplexWeighted
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.bf.focal_patterns import SinglePoint
from openlifu.plan.protocol import Protocol
from openlifu.seg.seg_methods.nnunet_seg import NNUNetSegmentation
from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI
from openlifu.sim.sim_setup import SimSetup

# ---------------------------------------------------------------------------
# Physical / treatment constants
# ---------------------------------------------------------------------------

SDT_FREQUENCY_HZ: float = 500_000.0
"""Sonodynamic therapy carrier frequency (Hz). 500 kHz is the GLADYS operating point."""

SDT_PULSE_DURATION_S: float = 0.086
"""Optimal pulse duration (s). 86 ms per Wu et al. 2025."""

SDT_PULSE_INTERVAL_S: float = 0.2
"""Interval between successive pulses within a pulse train (s)."""

SDT_PULSE_TRAIN_INTERVAL_S: float = 1.0
"""Interval between successive pulse trains (s)."""

SDT_PULSE_TRAIN_COUNT: int = 300
"""Number of pulse trains in a full treatment sequence.
300 trains at 1 s interval = 5 min total treatment duration."""

SDT_PULSES_PER_TRAIN: int = 5
"""Number of pulses per pulse train. 5 pulses * 0.2 s interval = 1.0 s train."""

TREATMENT_DURATION_S: float = 300.0
"""Total treatment duration per target (s). 5 minutes."""

DEFAULT_VOLTAGE: float = 1.0
"""Default transmit voltage (V). Scaled by Protocol.calc_solution when simulation is enabled."""

# ---------------------------------------------------------------------------
# Simulation / beamforming constants
# ---------------------------------------------------------------------------

DEFAULT_SPEED_OF_SOUND: float = 1500.0
"""Reference speed of sound in water (m/s)."""

DEFAULT_CFL: float = 0.3
"""Courant-Friedrichs-Lewy number for k-wave time stepping."""

DEFAULT_SIM_SPACING_MM: float = 1.0
"""Default simulation grid spacing (mm)."""

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def default_pulse() -> Pulse:
    """Build the default GLADYS SDT pulse."""
    return Pulse(
        frequency=SDT_FREQUENCY_HZ,
        amplitude=1.0,
        duration=SDT_PULSE_DURATION_S,
    )


def default_sequence() -> Sequence:
    """Build the default GLADYS SDT pulse sequence.

    5 pulses per train at 0.2 s intervals, 300 trains at 1.0 s intervals,
    yielding a 5-minute treatment duration.
    """
    return Sequence(
        pulse_interval=SDT_PULSE_INTERVAL_S,
        pulse_count=SDT_PULSES_PER_TRAIN,
        pulse_train_interval=SDT_PULSE_TRAIN_INTERVAL_S,
        pulse_train_count=SDT_PULSE_TRAIN_COUNT,
    )


def default_seg_method() -> NNUNetSegmentation:
    """Build the default segmentation method for GLADYS.

    Uses NNUNetSegmentation with the fullhead model, which segments the
    T1-weighted MRI volume into water, air, CSF, gray matter, white matter,
    skull, and soft tissue using an ONNX-exported nnU-Net deep learning model.
    Validated in the N=180 transcranial study with substantially higher
    accuracy than ThresholdMRI.
    """
    return NNUNetSegmentation(model_type="fullhead")


def default_delay_method() -> ComplexWeighted:
    """Build the default delay method for GLADYS.

    Uses the ComplexWeighted narrowband phase correction method, which
    extracts per-element complex coefficients (amplitude and phase) from
    a reciprocal k-wave simulation at the operating frequency. Validated
    in the N=180 transcranial study with +2.9 dB focal gain over
    SimulationCorrected.
    """
    return ComplexWeighted(
        c0=DEFAULT_SPEED_OF_SOUND,
        cfl=DEFAULT_CFL,
        n_cycles=3,
        gpu=True,
    )


def default_sim_setup() -> SimSetup:
    """Build the default simulation setup for GLADYS."""
    return SimSetup(
        spacing=DEFAULT_SIM_SPACING_MM,
        units="mm",
        cfl=DEFAULT_CFL,
        c0=DEFAULT_SPEED_OF_SOUND,
    )


def default_protocol() -> Protocol:
    """Build a fully configured GLADYS treatment protocol.

    Assembles the SDT pulse, sequence, segmentation, delay method,
    simulation setup, and focal pattern into a single Protocol object
    ready for treatment planning.
    """
    return Protocol(
        id="gladys_sdt",
        name="GLADYS SDT Protocol",
        description=(
            "Sonodynamic therapy protocol for the GLADYS platform. "
            "500 kHz, 86 ms pulse (Wu et al. 2025), 5-min treatment."
        ),
        pulse=default_pulse(),
        sequence=default_sequence(),
        focal_pattern=SinglePoint(),
        sim_setup=default_sim_setup(),
        delay_method=default_delay_method(),
        seg_method=default_seg_method(),
    )
