from __future__ import annotations

import pytest

from openlifu.bf.delay_methods import SimulationCorrected
from openlifu.plan.protocol import Protocol
from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI

from openlifu.gladys.config import (
    DEFAULT_CFL,
    DEFAULT_SIM_SPACING_MM,
    DEFAULT_SPEED_OF_SOUND,
    SDT_FREQUENCY_HZ,
    SDT_PULSE_DURATION_S,
    SDT_PULSE_INTERVAL_S,
    SDT_PULSE_TRAIN_COUNT,
    SDT_PULSE_TRAIN_INTERVAL_S,
    SDT_PULSES_PER_TRAIN,
    TREATMENT_DURATION_S,
    default_delay_method,
    default_protocol,
    default_pulse,
    default_seg_method,
    default_sequence,
    default_sim_setup,
)
from openlifu.gladys.pipeline import GLADYSPipeline


# ---------------------------------------------------------------------------
# Default protocol construction
# ---------------------------------------------------------------------------


class TestDefaultProtocolConstruction:
    def test_default_protocol_construction(self) -> None:
        """default_protocol() should return a valid Protocol with all fields populated."""
        proto = default_protocol()
        assert isinstance(proto, Protocol)
        assert proto.id == "gladys_sdt"
        assert proto.name == "GLADYS SDT Protocol"
        assert "500 kHz" in proto.description
        assert "5-min" in proto.description

    def test_protocol_has_pulse_and_sequence(self) -> None:
        """The protocol should carry the GLADYS SDT pulse and sequence."""
        proto = default_protocol()
        assert proto.pulse.frequency == SDT_FREQUENCY_HZ
        assert proto.pulse.duration == SDT_PULSE_DURATION_S
        assert proto.sequence.pulse_interval == SDT_PULSE_INTERVAL_S
        assert proto.sequence.pulse_count == SDT_PULSES_PER_TRAIN
        assert proto.sequence.pulse_train_interval == SDT_PULSE_TRAIN_INTERVAL_S
        assert proto.sequence.pulse_train_count == SDT_PULSE_TRAIN_COUNT


# ---------------------------------------------------------------------------
# Segmentation method
# ---------------------------------------------------------------------------


class TestPipelineSegMethod:
    def test_pipeline_has_correct_seg_method(self) -> None:
        """The default pipeline should use ThresholdMRI with brain tissue classification."""
        proto = default_protocol()
        assert isinstance(proto.seg_method, ThresholdMRI)
        assert proto.seg_method.classify_brain_tissues is True

    def test_seg_method_materials(self) -> None:
        """With brain classification on, the seg method should have csf, gm, wm materials."""
        seg = default_seg_method()
        assert isinstance(seg, ThresholdMRI)
        assert "csf" in seg.materials
        assert "gray_matter" in seg.materials
        assert "white_matter" in seg.materials
        assert "tissue" not in seg.materials


# ---------------------------------------------------------------------------
# Delay method
# ---------------------------------------------------------------------------


class TestPipelineDelayMethod:
    def test_pipeline_has_correct_delay_method(self) -> None:
        """The default pipeline should use SimulationCorrected for aberration correction."""
        proto = default_protocol()
        assert isinstance(proto.delay_method, SimulationCorrected)
        assert proto.delay_method.c0 == DEFAULT_SPEED_OF_SOUND
        assert proto.delay_method.cfl == DEFAULT_CFL

    def test_delay_method_standalone(self) -> None:
        dm = default_delay_method()
        assert isinstance(dm, SimulationCorrected)
        assert dm.c0 == 1500.0
        assert dm.cfl == 0.3


# ---------------------------------------------------------------------------
# SDT parameter configuration
# ---------------------------------------------------------------------------


class TestConfigSDTParameters:
    def test_config_sdt_parameters(self) -> None:
        """Verify the SDT constants match the GLADYS operating specifications."""
        assert SDT_FREQUENCY_HZ == 500_000.0
        assert SDT_PULSE_DURATION_S == 0.086
        assert SDT_PULSE_INTERVAL_S == 0.2
        assert SDT_PULSE_TRAIN_INTERVAL_S == 1.0
        assert SDT_PULSE_TRAIN_COUNT == 300
        assert SDT_PULSES_PER_TRAIN == 5
        assert TREATMENT_DURATION_S == 300.0

    def test_sdt_treatment_duration_consistency(self) -> None:
        """300 pulse trains at 1 s interval should equal 5 minutes."""
        calculated_duration = SDT_PULSE_TRAIN_COUNT * SDT_PULSE_TRAIN_INTERVAL_S
        assert calculated_duration == TREATMENT_DURATION_S

    def test_pulse_factory(self) -> None:
        pulse = default_pulse()
        assert pulse.frequency == SDT_FREQUENCY_HZ
        assert pulse.amplitude == 1.0
        assert pulse.duration == SDT_PULSE_DURATION_S

    def test_sequence_factory(self) -> None:
        seq = default_sequence()
        assert seq.pulse_interval == SDT_PULSE_INTERVAL_S
        assert seq.pulse_count == SDT_PULSES_PER_TRAIN
        assert seq.pulse_train_interval == SDT_PULSE_TRAIN_INTERVAL_S
        assert seq.pulse_train_count == SDT_PULSE_TRAIN_COUNT

    def test_sim_setup_factory(self) -> None:
        ss = default_sim_setup()
        assert ss.spacing == DEFAULT_SIM_SPACING_MM
        assert ss.units == "mm"
        assert ss.cfl == DEFAULT_CFL
        assert ss.c0 == DEFAULT_SPEED_OF_SOUND


# ---------------------------------------------------------------------------
# GLADYSPipeline construction
# ---------------------------------------------------------------------------


class TestGLADYSPipelineConstruction:
    def test_default_pipeline_construction(self) -> None:
        """GLADYSPipeline with no arguments should use the default protocol."""
        pipeline = GLADYSPipeline()
        assert isinstance(pipeline.protocol, Protocol)
        assert pipeline.protocol.id == "gladys_sdt"
        assert pipeline.transducer is None

    def test_pipeline_requires_transducer_for_planning(self) -> None:
        """plan_treatment should raise if no transducer is provided."""
        pipeline = GLADYSPipeline()
        with pytest.raises(ValueError, match="Transducer must be provided"):
            pipeline.plan_treatment(
                mri_path="/fake/mri.nii.gz",
                target_position=[0.0, 0.0, 0.0],
            )
