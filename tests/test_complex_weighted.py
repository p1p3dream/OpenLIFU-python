from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openlifu.bf.delay_methods import ComplexWeighted, DelayMethod, SimulationCorrected


class TestComplexWeightedConstruction:
    """Test basic construction of ComplexWeighted instances."""

    def test_default_construction(self):
        method = ComplexWeighted()
        assert isinstance(method, ComplexWeighted)
        assert isinstance(method, DelayMethod)
        assert method.c0 == 1500.0
        assert method.cfl == 0.3
        assert method.n_cycles == 3
        assert method.gpu is True
        assert method.allow_out_of_grid_fallback is False
        assert method.bandwidth_frac == 0.1
        assert method.window_cycles == 6.0

    def test_custom_params(self):
        method = ComplexWeighted(
            c0=1480.0,
            cfl=0.2,
            n_cycles=5,
            gpu=False,
            allow_out_of_grid_fallback=True,
            bandwidth_frac=0.2,
            window_cycles=8.0,
        )
        assert method.c0 == 1480.0
        assert method.cfl == 0.2
        assert method.n_cycles == 5
        assert method.gpu is False
        assert method.allow_out_of_grid_fallback is True
        assert method.bandwidth_frac == 0.2
        assert method.window_cycles == 8.0


class TestComplexWeightedValidation:
    """Test parameter validation."""

    def test_invalid_c0_negative(self):
        with pytest.raises(ValueError, match="greater than 0"):
            ComplexWeighted(c0=-100.0)

    def test_invalid_cfl_too_high(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ComplexWeighted(cfl=1.0)

    def test_invalid_n_cycles_zero(self):
        with pytest.raises(ValueError, match="at least 1"):
            ComplexWeighted(n_cycles=0)

    def test_invalid_bandwidth_frac_zero(self):
        with pytest.raises(ValueError, match="bandwidth_frac"):
            ComplexWeighted(bandwidth_frac=0.0)

    def test_invalid_bandwidth_frac_too_high(self):
        with pytest.raises(ValueError, match="bandwidth_frac"):
            ComplexWeighted(bandwidth_frac=2.0)

    def test_invalid_window_cycles_zero(self):
        with pytest.raises(ValueError, match="window_cycles"):
            ComplexWeighted(window_cycles=0.0)

    def test_invalid_window_cycles_type(self):
        with pytest.raises(TypeError, match="window_cycles"):
            ComplexWeighted(window_cycles="six")

    def test_n_cycles_float_int_coercion(self):
        method = ComplexWeighted(n_cycles=3.0)
        assert method.n_cycles == 3
        assert isinstance(method.n_cycles, int)


class TestComplexWeightedSerialization:
    """Test to_dict / from_dict round-trip serialization."""

    def test_round_trip(self):
        original = ComplexWeighted(
            c0=1450.0, cfl=0.35, n_cycles=5, gpu=False,
            bandwidth_frac=0.15, window_cycles=10.0,
        )
        d = original.to_dict()
        assert d['class'] == 'ComplexWeighted'
        restored = DelayMethod.from_dict(d)
        assert isinstance(restored, ComplexWeighted)
        assert restored.c0 == original.c0
        assert restored.cfl == original.cfl
        assert restored.n_cycles == original.n_cycles
        assert restored.gpu == original.gpu
        assert restored.bandwidth_frac == original.bandwidth_frac
        assert restored.window_cycles == original.window_cycles


class TestComplexWeightedTable:
    """Test the to_table method."""

    def test_to_table(self):
        method = ComplexWeighted()
        table = method.to_table()
        assert len(table) == 8
        assert table.iloc[0]['Name'] == 'Type'
        assert table.iloc[0]['Value'] == 'ComplexWeighted'
        assert table.iloc[-2]['Name'] == 'Bandpass Bandwidth Fraction'
        assert table.iloc[-1]['Name'] == 'DFT Window Cycles'


class TestWeightsFromCoefficients:
    """The pure-math helper that converts (a_i, phi_i) to (delays, apod)."""

    def test_delays_match_phase_over_2pi_f0(self):
        f0 = 500e3
        phases = np.array([0.0, np.pi / 4, -np.pi / 3, np.pi / 2])
        amplitudes = np.array([1.0, 0.5, 0.25, 0.75])
        delays, apod = ComplexWeighted._weights_from_coefficients(
            amplitudes, phases, f0,
        )
        expected_delays = -phases / (2 * np.pi * f0)
        np.testing.assert_allclose(delays, expected_delays, atol=1e-15)
        np.testing.assert_allclose(apod, amplitudes / amplitudes.max())

    def test_apod_normalized_to_unit_max(self):
        f0 = 500e3
        amplitudes = np.array([2.0, 4.0, 1.0])
        phases = np.zeros_like(amplitudes)
        _delays, apod = ComplexWeighted._weights_from_coefficients(
            amplitudes, phases, f0,
        )
        assert apod.max() == pytest.approx(1.0)
        np.testing.assert_allclose(apod, [0.5, 1.0, 0.25])

    def test_phase_wrapping(self):
        """Phases outside (-pi, pi] should be wrapped before dividing."""
        f0 = 500e3
        phases = np.array([2.5 * np.pi, -1.5 * np.pi, 4.0 * np.pi])
        amplitudes = np.ones_like(phases)
        delays, _ = ComplexWeighted._weights_from_coefficients(
            amplitudes, phases, f0,
        )
        # 2.5*pi wraps to 0.5*pi; -1.5*pi wraps to +0.5*pi;
        # 4.0*pi wraps to 0 (modulo 2*pi).
        expected = -np.array([0.5 * np.pi, 0.5 * np.pi, 0.0]) / (2 * np.pi * f0)
        np.testing.assert_allclose(delays, expected, atol=1e-12)

    def test_zero_amplitudes_gracefully_handled(self):
        f0 = 500e3
        amplitudes = np.zeros(3)
        phases = np.array([0.1, 0.2, 0.3])
        delays, apod = ComplexWeighted._weights_from_coefficients(
            amplitudes, phases, f0,
        )
        # With max(a) == 0 the code returns ones rather than NaN.
        np.testing.assert_array_equal(apod, np.ones(3))
        assert np.all(np.isfinite(delays))


class TestExtractNarrowbandCoefficient:
    """Unit test the single-bin DFT extractor on a synthetic sinusoid."""

    def test_recovers_phase_and_amplitude_of_pure_tone(self):
        f0 = 500e3
        fs = 20e6  # well above Nyquist
        dt = 1.0 / fs
        n = 4096
        t = np.arange(n) * dt
        A = 3.0
        phi0 = 0.7  # radians
        # Real tone: cos(2*pi*f0*t + phi0)
        signal = A * np.cos(2 * np.pi * f0 * t + phi0)

        arrival_idx = n // 2
        coef = ComplexWeighted._extract_narrowband_coefficient(
            signal, dt, f0, arrival_idx, window_cycles=20.0,
        )
        # For basis exp(-j*omega*t), a real cosine of amplitude A and phase phi0
        # gives a coefficient whose angle equals +phi0 and whose magnitude is
        # (A/2) * window_length_samples in the long-window limit.
        win_len = int(round(20.0 / f0 / dt))
        expected_mag = (A / 2.0) * win_len
        assert abs(coef) == pytest.approx(expected_mag, rel=0.05)
        assert np.angle(coef) == pytest.approx(phi0, abs=0.05)


class TestComplexWeightedBehavior:
    """Test the full pipeline by mocking _run_reciprocal_simulation_complex."""

    def test_calc_complex_weights_from_known_coefficients(self):
        f0 = 500e3
        amplitudes = np.array([0.5, 1.0, 0.25])
        phases = np.array([0.0, np.pi / 4, -np.pi / 3])

        method = ComplexWeighted()
        with patch.object(
            ComplexWeighted,
            "_run_reciprocal_simulation_complex",
            return_value=(amplitudes, phases, f0),
        ), patch("importlib.util.find_spec", return_value=True):
            delays, apod = method.calc_complex_weights(
                arr=None, target=None, params=None, transform=None,
            )

        np.testing.assert_allclose(delays, -phases / (2 * np.pi * f0))
        np.testing.assert_allclose(apod, amplitudes / amplitudes.max())
        assert apod.max() == pytest.approx(1.0)

    def test_calc_delays_returns_delays_only(self):
        """calc_delays must return a 1D delay array (backward compat)."""
        f0 = 500e3
        amplitudes = np.array([1.0, 0.5])
        phases = np.array([0.1, -0.2])

        method = ComplexWeighted()
        with patch.object(
            ComplexWeighted,
            "_run_reciprocal_simulation_complex",
            return_value=(amplitudes, phases, f0),
        ), patch("importlib.util.find_spec", return_value=True):
            delays = method.calc_delays(
                arr=None, target=None, params=None, transform=None,
            )
        assert delays.ndim == 1
        np.testing.assert_allclose(delays, -phases / (2 * np.pi * f0))

    def test_calc_delays_and_apod_alias(self):
        f0 = 500e3
        amplitudes = np.array([1.0, 0.5, 0.25])
        phases = np.array([0.0, 0.3, -0.4])
        method = ComplexWeighted()
        with patch.object(
            ComplexWeighted,
            "_run_reciprocal_simulation_complex",
            return_value=(amplitudes, phases, f0),
        ), patch("importlib.util.find_spec", return_value=True):
            d1, a1 = method.calc_complex_weights(None, None, None, None)
            d2, a2 = method.calc_delays_and_apod(None, None, None, None)
        np.testing.assert_allclose(d1, d2)
        np.testing.assert_allclose(a1, a2)

    def test_equal_amplitudes_gives_pure_phase_delays(self):
        """When all a_i are identical, apod should be all-ones and delays
        reduce to -phi_i/(2*pi*f0), i.e. the pure-phase case."""
        f0 = 500e3
        amplitudes = np.full(8, 0.73)
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, size=8)

        method = ComplexWeighted()
        with patch.object(
            ComplexWeighted,
            "_run_reciprocal_simulation_complex",
            return_value=(amplitudes, phases, f0),
        ), patch("importlib.util.find_spec", return_value=True):
            delays, apod = method.calc_complex_weights(
                arr=None, target=None, params=None, transform=None,
            )
        np.testing.assert_allclose(apod, np.ones_like(amplitudes))
        np.testing.assert_allclose(delays, -phases / (2 * np.pi * f0), atol=1e-12)

    def test_fallback_when_kwave_missing(self):
        """When k-wave is not installed, calc_complex_weights should fall
        back to Direct geometric delays and unit amplitudes."""
        method = ComplexWeighted()
        fallback_delays = np.array([0.0, 1e-6, 2e-6])
        with patch("importlib.util.find_spec", return_value=None), patch.object(
            ComplexWeighted,
            "_fallback_complex_weights",
            return_value=(fallback_delays, np.ones_like(fallback_delays)),
        ) as mock_fallback:
            delays, apod = method.calc_complex_weights(
                arr=None, target=None, params=None, transform=None,
            )
        mock_fallback.assert_called_once()
        np.testing.assert_array_equal(delays, fallback_delays)
        np.testing.assert_array_equal(apod, np.ones_like(fallback_delays))

    def test_fallback_on_simulation_failure(self):
        """If the reciprocal sim raises, fall back cleanly."""
        method = ComplexWeighted()
        fallback_delays = np.zeros(4)
        with patch.object(
            ComplexWeighted,
            "_run_reciprocal_simulation_complex",
            side_effect=RuntimeError("mocked failure"),
        ), patch("importlib.util.find_spec", return_value=True), patch.object(
            ComplexWeighted,
            "_fallback_complex_weights",
            return_value=(fallback_delays, np.ones_like(fallback_delays)),
        ) as mock_fallback:
            delays, apod = method.calc_complex_weights(
                arr=None, target=None, params=None, transform=None,
            )
        mock_fallback.assert_called_once()
        np.testing.assert_array_equal(delays, fallback_delays)
        np.testing.assert_array_equal(apod, np.ones_like(fallback_delays))
