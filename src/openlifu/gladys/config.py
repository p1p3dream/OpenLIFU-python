"""GLADYS-specific constants and default protocol configuration.

Defines the sonodynamic therapy (SDT) parameters derived from the GLADYS
hardware platform and the Wu et al. 2025 optimal pulse characterization.
Provides factory functions to build a fully configured Protocol with
GLADYS defaults.
"""

from __future__ import annotations

from openlifu.bf import Pulse, Sequence
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.bf.focal_patterns import SinglePoint
from openlifu.plan.protocol import Protocol
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


def default_seg_method() -> ThresholdMRI:
    """Build the default segmentation method for GLADYS.

    Uses ThresholdMRI with full-head tissue classification enabled, which
    differentiates skull, scalp, CSF, gray matter, and white matter from
    the T1-weighted MRI volume.
    """
    return ThresholdMRI(classify_brain_tissues=True)


def default_delay_method() -> SimulationCorrected:
    """Build the default delay method for GLADYS.

    Uses the SimulationCorrected reciprocal k-wave approach to compute
    aberration-corrected transmit delays through the heterogeneous skull.
    """
    return SimulationCorrected(
        c0=DEFAULT_SPEED_OF_SOUND,
        cfl=DEFAULT_CFL,
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
