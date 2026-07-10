"""Integration tests for the GLADYS treatment planning pipeline.

Tests cover transducer creation, default protocol/SDT parameters,
MRI loading, error handling, and a smoke test for the full
plan_treatment pipeline (without GPU simulation).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import xarray as xa

from openlifu.gladys.config import (
    SDT_FREQUENCY_HZ,
    SDT_PULSE_DURATION_S,
    SDT_PULSE_TRAIN_COUNT,
    TREATMENT_DURATION_S,
    default_protocol,
    default_pulse,
    default_sequence,
)
from openlifu.gladys.pipeline import GLADYSPipeline
from openlifu.gladys.transducer import (
    APERTURE_MM,
    FREQUENCY_HZ,
    N_ELEMENTS,
    RADIUS_MM,
    create_gladys_transducer,
)
from openlifu.plan.protocol import Protocol
from openlifu.xdc import Transducer

_has_nibabel = importlib.util.find_spec("nibabel") is not None


# ---------------------------------------------------------------------------
# Transducer creation
# ---------------------------------------------------------------------------


class TestCreateGladysTransducer:
    """Tests for create_gladys_transducer geometry and metadata."""

    def test_element_count(self) -> None:
        """The GLADYS transducer should have 64 elements."""
        arr = create_gladys_transducer()
        assert arr.numelements() == N_ELEMENTS
        assert arr.numelements() == 64

    def test_frequency(self) -> None:
        """Nominal frequency should be 500 kHz."""
        arr = create_gladys_transducer()
        assert arr.frequency == FREQUENCY_HZ
        assert arr.frequency == 500e3

    def test_units_mm(self) -> None:
        """Transducer and its elements should use millimeters."""
        arr = create_gladys_transducer()
        assert arr.units == "mm"
        for el in arr.elements:
            assert el.units == "mm"

    def test_element_positions_on_sphere(self) -> None:
        """Every element should lie on a sphere of radius R=90 mm."""
        arr = create_gladys_transducer()
        for el in arr.elements:
            r = np.linalg.norm(el.position)
            assert r == pytest.approx(RADIUS_MM, abs=1e-6), (
                f"Element {el.index} at distance {r:.4f} mm, expected {RADIUS_MM} mm"
            )

    def test_element_positions_within_aperture(self) -> None:
        """All elements should fall within the aperture half-angle."""
        arr = create_gladys_transducer()
        half_aperture = APERTURE_MM / 2.0
        theta_max = np.arcsin(half_aperture / RADIUS_MM)
        for el in arr.elements:
            r = np.linalg.norm(el.position)
            theta = np.arccos(el.position[2] / r)
            assert theta <= theta_max + 1e-9, (
                f"Element {el.index} polar angle {np.degrees(theta):.2f} deg "
                f"exceeds theta_max {np.degrees(theta_max):.2f} deg"
            )

    def test_returns_transducer_type(self) -> None:
        """Factory should return an openlifu Transducer instance."""
        arr = create_gladys_transducer()
        assert isinstance(arr, Transducer)

    def test_custom_n_elements(self) -> None:
        """Passing a custom n_elements should produce the requested count."""
        arr = create_gladys_transducer(n_elements=16)
        assert arr.numelements() == 16


# ---------------------------------------------------------------------------
# Default protocol and SDT parameters
# ---------------------------------------------------------------------------


class TestDefaultProtocolSDTParams:
    """Verify default protocol carries correct SDT treatment parameters."""

    def test_frequency_500khz(self) -> None:
        """Pulse frequency should be 500 kHz."""
        pulse = default_pulse()
        assert pulse.frequency == 500_000.0
        assert pulse.frequency == SDT_FREQUENCY_HZ

    def test_pulse_duration_86ms(self) -> None:
        """Pulse duration should be 86 ms per Wu et al. 2025."""
        pulse = default_pulse()
        assert pulse.duration == 0.086
        assert pulse.duration == SDT_PULSE_DURATION_S

    def test_sequence_300_trains(self) -> None:
        """Sequence should have 300 pulse trains for a 5-min treatment."""
        seq = default_sequence()
        assert seq.pulse_train_count == 300
        assert seq.pulse_train_count == SDT_PULSE_TRAIN_COUNT

    def test_treatment_duration_5_minutes(self) -> None:
        """300 trains at 1 s interval should equal 300 s (5 minutes)."""
        seq = default_sequence()
        total = seq.pulse_train_count * seq.pulse_train_interval
        assert total == 300.0
        assert total == TREATMENT_DURATION_S

    def test_protocol_is_valid(self) -> None:
        """default_protocol() should return a fully populated Protocol."""
        proto = default_protocol()
        assert isinstance(proto, Protocol)
        assert proto.id == "gladys_sdt"
        assert proto.pulse is not None
        assert proto.sequence is not None

    def test_protocol_pulse_matches_factory(self) -> None:
        """Protocol's embedded pulse should match default_pulse() values."""
        proto = default_protocol()
        assert proto.pulse.frequency == SDT_FREQUENCY_HZ
        assert proto.pulse.duration == SDT_PULSE_DURATION_S

    def test_protocol_sequence_matches_factory(self) -> None:
        """Protocol's embedded sequence should match default_sequence() values."""
        proto = default_protocol()
        assert proto.sequence.pulse_train_count == SDT_PULSE_TRAIN_COUNT


# ---------------------------------------------------------------------------
# MRI loading
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_nibabel, reason="nibabel not installed")
class TestLoadMRICreatesXarray:
    """Test GLADYSPipeline.load_mri with a synthetic NIfTI volume."""

    @staticmethod
    def _create_synthetic_nifti(path, shape=(10, 10, 10), spacing=(1.0, 1.0, 1.0)):
        """Write a tiny synthetic NIfTI file using nibabel."""
        import nibabel as nib

        data = np.random.default_rng(42).uniform(0, 1000, size=shape).astype(np.float32)
        affine = np.diag([*spacing, 1.0])
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(path))
        return data, affine

    def test_returns_xarray_dataarray(self, tmp_path) -> None:
        """load_mri should return an xarray.DataArray."""
        nii_path = tmp_path / "test_vol.nii.gz"
        self._create_synthetic_nifti(nii_path)
        vol = GLADYSPipeline.load_mri(nii_path)
        assert isinstance(vol, xa.DataArray)

    def test_correct_shape(self, tmp_path) -> None:
        """Loaded volume should have the same shape as the source data."""
        shape = (8, 12, 6)
        nii_path = tmp_path / "test_shape.nii.gz"
        self._create_synthetic_nifti(nii_path, shape=shape)
        vol = GLADYSPipeline.load_mri(nii_path)
        assert vol.shape == shape

    def test_dimension_names(self, tmp_path) -> None:
        """DataArray should have dimensions named (x, y, z)."""
        nii_path = tmp_path / "test_dims.nii.gz"
        self._create_synthetic_nifti(nii_path)
        vol = GLADYSPipeline.load_mri(nii_path)
        assert vol.dims == ("x", "y", "z")

    def test_coordinates_in_mm(self, tmp_path) -> None:
        """Coordinates should carry units='mm' attribute."""
        nii_path = tmp_path / "test_units.nii.gz"
        self._create_synthetic_nifti(nii_path)
        vol = GLADYSPipeline.load_mri(nii_path)
        for dim in ("x", "y", "z"):
            assert vol.coords[dim].attrs["units"] == "mm"

    def test_coordinate_values_from_affine(self, tmp_path) -> None:
        """Coordinate values should match affine origin + spacing * index."""
        spacing = (2.0, 3.0, 1.5)
        shape = (5, 7, 4)
        nii_path = tmp_path / "test_coords.nii.gz"
        self._create_synthetic_nifti(nii_path, shape=shape, spacing=spacing)
        vol = GLADYSPipeline.load_mri(nii_path)
        for axis, dim in enumerate(("x", "y", "z")):
            expected = np.arange(shape[axis]) * spacing[axis]
            np.testing.assert_allclose(vol.coords[dim].values, expected)

    def test_source_attr(self, tmp_path) -> None:
        """DataArray should carry the source file path as an attribute."""
        nii_path = tmp_path / "test_source.nii.gz"
        self._create_synthetic_nifti(nii_path)
        vol = GLADYSPipeline.load_mri(nii_path)
        assert "source" in vol.attrs
        assert str(nii_path) in vol.attrs["source"]

    def test_dtype_float32(self, tmp_path) -> None:
        """Loaded volume should have float32 dtype."""
        nii_path = tmp_path / "test_dtype.nii.gz"
        self._create_synthetic_nifti(nii_path)
        vol = GLADYSPipeline.load_mri(nii_path)
        assert vol.dtype == np.float32


# ---------------------------------------------------------------------------
# MRI loading error handling
# ---------------------------------------------------------------------------


class TestLoadMRIFileNotFound:
    """Test that load_mri raises FileNotFoundError for missing files."""

    def test_raises_on_nonexistent_path(self) -> None:
        """FileNotFoundError should be raised when the NIfTI file does not exist."""
        with pytest.raises(FileNotFoundError, match="MRI file not found"):
            GLADYSPipeline.load_mri("/nonexistent/path/to/mri.nii.gz")

    def test_raises_on_nonexistent_directory(self, tmp_path) -> None:
        """FileNotFoundError should be raised even when the parent dir exists."""
        fake = tmp_path / "does_not_exist.nii"
        with pytest.raises(FileNotFoundError, match="MRI file not found"):
            GLADYSPipeline.load_mri(fake)


# ---------------------------------------------------------------------------
# Plan treatment error handling
# ---------------------------------------------------------------------------


class TestPlanTreatmentNoTransducerRaises:
    """Verify that plan_treatment raises when no transducer is available."""

    def test_no_transducer_at_construction_or_call(self) -> None:
        """ValueError should be raised when transducer is None everywhere."""
        pipeline = GLADYSPipeline()
        with pytest.raises(ValueError, match="Transducer must be provided"):
            pipeline.plan_treatment(
                mri_path="/fake/mri.nii.gz",
                target_position=[0.0, 0.0, 0.0],
            )

    def test_transducer_none_explicitly(self) -> None:
        """Explicitly passing transducer=None should also raise."""
        pipeline = GLADYSPipeline(transducer=None)
        with pytest.raises(ValueError, match="Transducer must be provided"):
            pipeline.plan_treatment(
                mri_path="/fake/mri.nii.gz",
                target_position=[0.0, 0.0, 0.0],
                transducer=None,
            )


# ---------------------------------------------------------------------------
# Plan treatment smoke test (no GPU / no simulation)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_nibabel, reason="nibabel not installed")
class TestPlanTreatmentSmoke:
    """Smoke test: run plan_treatment(simulate=False) to verify the pipeline
    chain works end to end without needing a GPU or k-wave installation.

    This exercises: MRI loading, target Point construction, segmentation,
    and delay computation. Simulation and scaling are skipped.
    """

    @staticmethod
    def _create_synthetic_nifti(path, shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0)):
        """Write a tiny synthetic NIfTI with uniform intensity."""
        import nibabel as nib

        data = np.ones(shape, dtype=np.float32) * 500.0
        affine = np.diag([*spacing, 1.0])
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(path))

    @pytest.mark.slow
    def test_plan_treatment_runs_without_simulation(self, tmp_path) -> None:
        """plan_treatment(simulate=False) should return a Solution tuple."""
        nii_path = tmp_path / "smoke_mri.nii.gz"
        self._create_synthetic_nifti(nii_path)

        arr = create_gladys_transducer()
        pipeline = GLADYSPipeline(transducer=arr)

        # Target near the center of the volume
        target_pos = [8.0, 8.0, 8.0]

        solution, sim_result, analysis = pipeline.plan_treatment(
            mri_path=str(nii_path),
            target_position=target_pos,
            simulate=False,
            scale=False,
        )

        # Solution should be returned and have basic properties
        from openlifu.plan.solution import Solution
        assert isinstance(solution, Solution)
        assert solution.id is not None

    @pytest.mark.slow
    def test_plan_treatment_with_transducer_override(self, tmp_path) -> None:
        """Passing transducer= to plan_treatment should override the pipeline default."""
        nii_path = tmp_path / "smoke_override.nii.gz"
        self._create_synthetic_nifti(nii_path)

        # Pipeline created without transducer
        pipeline = GLADYSPipeline()
        arr = create_gladys_transducer()

        solution, sim_result, analysis = pipeline.plan_treatment(
            mri_path=str(nii_path),
            target_position=[8.0, 8.0, 8.0],
            transducer=arr,
            simulate=False,
            scale=False,
        )

        from openlifu.plan.solution import Solution
        assert isinstance(solution, Solution)

    def test_plan_treatment_bad_target_shape(self, tmp_path) -> None:
        """target_position with wrong number of elements should raise ValueError."""
        nii_path = tmp_path / "smoke_bad_target.nii.gz"
        self._create_synthetic_nifti(nii_path)

        arr = create_gladys_transducer()
        pipeline = GLADYSPipeline(transducer=arr)

        with pytest.raises(ValueError, match="3 elements"):
            pipeline.plan_treatment(
                mri_path=str(nii_path),
                target_position=[1.0, 2.0],
                simulate=False,
            )
