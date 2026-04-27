from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.delay_methods import DelayMethod
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer

logger = logging.getLogger(__name__)


@dataclass
class ComplexWeighted(DelayMethod):
    """Narrowband complex-weighted phase correction at operating frequency f0.

    Like SimulationCorrected, runs a reciprocal k-wave simulation with a virtual
    point source at the target. Instead of extracting a scalar delay per element,
    extracts a complex narrowband coefficient a_i * exp(j * phi_i) at f0 via a
    single-bin DFT on each element's recorded time series (optionally
    bandpass-filtered around f0 first). The resulting complex weights are split
    into:

      - delays[i] = -phi_i / (2 * pi * f0)
      - apod[i]   = a_i / max(a_i)

    so that a downstream beamformer that accepts both delays and apodization
    reproduces the complex weight on transmit.

    The DelayMethod contract only promises scalar delays per element, so for
    backward compatibility ``calc_delays`` returns the delays alone. The new
    entry point ``calc_complex_weights`` returns the ``(delays, apod)`` pair.

    Notes:
      - Returned amplitudes are normalized so that max(a_i) == 1.
      - Returned delays are absolute transmit times: the narrowband phase
        delay is composed internally with the geometric TOF baseline from
        :class:`Direct` and biased up so ``min(delays) == 0``. Callers do not
        need to add any further geometric offset.
      - Requires a protocol wiring that accepts both delays and apodization to
        actually apply the amplitude component. :class:`openlifu.plan.protocol.Protocol.beamform`
        already honors the amplitude component returned by
        :meth:`calc_delays_and_apod`. See the design doc
        ``docs/design/phase_correction_approaches.md``.
    """

    c0: Annotated[
        float,
        OpenLIFUFieldData("Speed of Sound (m/s)", "Reference speed of sound in the medium (m/s)"),
    ] = 1500.0
    """Reference speed of sound in the medium (m/s)"""

    cfl: Annotated[float, OpenLIFUFieldData("CFL Number", "Courant-Friedrichs-Lewy number for time stepping")] = 0.3
    """Courant-Friedrichs-Lewy number for time stepping"""

    n_cycles: Annotated[int, OpenLIFUFieldData("Source Cycles", "Number of cycles in the source pulse")] = 3
    """Number of cycles in the source pulse"""

    gpu: Annotated[bool, OpenLIFUFieldData("Use GPU", "Whether to attempt GPU-accelerated simulation")] = True
    """Whether to attempt GPU-accelerated simulation"""

    allow_out_of_grid_fallback: Annotated[
        bool,
        OpenLIFUFieldData(
            "Allow Out-of-Grid Fallback",
            "If True, silently fall back to geometric time-of-flight for elements outside the simulation grid.",
        ),
    ] = False
    """Whether to silently fall back for elements outside the simulation grid."""

    bandwidth_frac: Annotated[
        float,
        OpenLIFUFieldData(
            "Bandpass Bandwidth Fraction",
            "Fractional bandwidth around f0 for the pre-DFT Butterworth bandpass (e.g. 0.1 means +-5%% around f0).",
        ),
    ] = 0.1
    """Fractional bandwidth for the narrowband bandpass filter around f0."""

    window_cycles: Annotated[
        float,
        OpenLIFUFieldData(
            "DFT Window Cycles",
            "Number of cycles of f0 in the single-bin DFT window centered on the arrival time.",
        ),
    ] = 6.0
    """Length of the single-bin DFT analysis window in cycles of f0."""

    def __post_init__(self):
        if not isinstance(self.c0, int | float):
            raise TypeError("Speed of sound must be a number")
        if self.c0 <= 0:
            raise ValueError("Speed of sound must be greater than 0")
        self.c0 = float(self.c0)

        if not isinstance(self.cfl, int | float):
            raise TypeError("CFL must be a number")
        if self.cfl <= 0 or self.cfl >= 1:
            raise ValueError("CFL must be between 0 and 1 (exclusive)")
        self.cfl = float(self.cfl)

        if not isinstance(self.n_cycles, int):
            if isinstance(self.n_cycles, float) and self.n_cycles == int(self.n_cycles):
                self.n_cycles = int(self.n_cycles)
            else:
                raise TypeError("n_cycles must be an integer")
        if self.n_cycles < 1:
            raise ValueError("n_cycles must be at least 1")

        if not isinstance(self.gpu, bool):
            raise TypeError("gpu must be a boolean")

        if not isinstance(self.allow_out_of_grid_fallback, bool):
            raise TypeError("allow_out_of_grid_fallback must be a boolean")

        if not isinstance(self.bandwidth_frac, int | float):
            raise TypeError("bandwidth_frac must be a number")
        if self.bandwidth_frac <= 0 or self.bandwidth_frac >= 2:
            raise ValueError("bandwidth_frac must be between 0 and 2 (exclusive)")
        self.bandwidth_frac = float(self.bandwidth_frac)

        if not isinstance(self.window_cycles, int | float):
            raise TypeError("window_cycles must be a number")
        if self.window_cycles <= 0:
            raise ValueError("window_cycles must be greater than 0")
        self.window_cycles = float(self.window_cycles)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calc_delays(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Backward-compatible entry point that returns delays only.

        Internally this calls :meth:`calc_complex_weights` and discards the
        amplitude component. Use :meth:`calc_complex_weights` directly if you
        want both delay and amplitude per element.
        """
        delays, _apod = self.calc_complex_weights(arr, target, params, transform)
        return delays

    def calc_complex_weights(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-element ``(delays, apod)`` from the narrowband complex weights.

        The returned ``delays`` are absolute transmit times: the narrowband
        phase *correction* (the difference between the measured DFT phase and
        the expected geometric phase, wrapped to ``(-pi, pi]``, so magnitude
        is at most ``1/(2*f0)``) is composed with the nominal geometric
        time-of-flight baseline from :class:`Direct` and biased up so
        ``min(delays) == 0``. This keeps all per-element delays non-negative,
        as required by the transmit path. In the pure-phase limit (all phases
        equal to their geometric expectation) the result matches
        :meth:`Direct.calc_delays` to within floating-point noise.

        Falls back to Direct (geometric) delays with unit amplitudes if k-wave
        is not available or the simulation raises.
        """
        try:
            import importlib
            if importlib.util.find_spec("kwave") is None:
                raise ImportError("k-wave not installed")
        except ImportError:
            logger.warning(
                "k-wave not available. ComplexWeighted falling back to Direct "
                "delay method with unit amplitudes.",
            )
            return self._fallback_complex_weights(arr, target, params, transform)

        try:
            amplitudes, phases, f0 = self._run_reciprocal_simulation_complex(
                arr, target, params, transform,
            )

            geometric_phases = self._compute_geometric_phases(
                arr, target, params, f0, transform,
            )

            phase_delays, apod = self._weights_from_coefficients(
                amplitudes, phases, f0, geometric_phases=geometric_phases,
            )
            delays = self._compose_with_geometric(
                phase_delays, arr, target, params, transform,
            )
            return delays, apod
        except (RuntimeError, ValueError, IndexError, OSError):
            logger.exception(
                "ComplexWeighted delay calculation failed. "
                "Falling back to Direct method with unit amplitudes.",
            )
            return self._fallback_complex_weights(arr, target, params, transform)

    def calc_delays_and_apod(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Explicit alias for :meth:`calc_complex_weights`."""
        return self.calc_complex_weights(arr, target, params, transform)

    def _compose_with_geometric(
        self,
        phase_delays: np.ndarray,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compose narrowband phase correction with a geometric TOF baseline.

        ``phase_delays[i]`` is the excess propagation time for element *i*
        relative to the geometric estimate: ``tau_measured - tau_geometric``.
        A positive value means the actual path is slower than geometric
        (e.g., skull slows the wave), so the element should fire *earlier*
        to compensate. We therefore SUBTRACT ``phase_delays`` from the
        geometric baseline.

        The geometric baseline is ``max(TOF_geo) - TOF_geo_i`` (from
        :class:`Direct`), which is non-negative. After subtracting the
        correction, the result is biased so ``min(delays) == 0``.
        """
        from openlifu.bf.delay_methods.direct import Direct

        geom_delays = Direct(c0=self.c0).calc_delays(
            arr, target, params, transform=transform,
        )
        delays = geom_delays - np.asarray(phase_delays, dtype=float)
        min_delay = float(np.min(delays))
        if min_delay < 0.0:
            delays = delays - min_delay
        return delays

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_geometric_phases(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset | None,
        f0: float,
        transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute expected geometric phase per element: ``-2*pi*f0*(dist/c)``.

        Uses the same reference sound speed that :class:`Direct` would select
        from ``params``, so the geometric phase is consistent with the
        geometric delay baseline added by :meth:`_compose_with_geometric`.
        """
        if params is not None and 'sound_speed' in params and 'ref_value' in params['sound_speed'].attrs:
            c_ref = float(params['sound_speed'].attrs['ref_value'])
        else:
            c_ref = self.c0
        matrix = np.asarray(transform, dtype=float) if transform is not None else np.eye(4)
        target_pos = target.get_position(units="m")
        dists_m = np.array([
            el.distance_to_point(target_pos, units="m", matrix=matrix)
            for el in arr.elements
        ])
        tof = dists_m / c_ref
        return -2.0 * np.pi * f0 * tof

    @staticmethod
    def _weights_from_coefficients(
        amplitudes: np.ndarray,
        phases: np.ndarray,
        f0: float,
        geometric_phases: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert per-element (a_i, phi_i) complex coefficients to (delays, apod).

        Parameters
        ----------
        amplitudes : array
            Per-element DFT coefficient magnitudes.
        phases : array
            Per-element DFT coefficient phases (radians). These encode the
            total propagation phase (geometric + aberration).
        f0 : float
            Operating frequency in Hz.
        geometric_phases : array or None
            Expected geometric phase per element: ``-2*pi*f0*(dist_i / c0)``.
            When provided, only the phase *correction* (measured minus
            geometric) is wrapped to (-pi, pi] before converting to a delay.
            This avoids aliasing multi-cycle geometric propagation into a
            single period, which is critical when the geometric time-of-flight
            spread exceeds half a period of f0.
        """
        amplitudes = np.asarray(amplitudes, dtype=float)
        phases = np.asarray(phases, dtype=float)

        max_a = float(np.max(amplitudes)) if amplitudes.size else 0.0
        if max_a > 0:
            apod = amplitudes / max_a
        else:
            apod = np.ones_like(amplitudes)

        if geometric_phases is not None:
            geometric_phases = np.asarray(geometric_phases, dtype=float)
            # Wrap only the small correction (measured - geometric) so that
            # multi-cycle geometric delays are not aliased into one period.
            correction = np.angle(np.exp(1j * (phases - geometric_phases)))
            delays = -correction / (2 * np.pi * f0)
        else:
            # Legacy path: wrap total phase (correct only when geometric TOF
            # spread is less than half a period of f0).
            wrapped = np.angle(np.exp(1j * phases))
            delays = -wrapped / (2 * np.pi * f0)
        return delays, apod

    @staticmethod
    def _extract_narrowband_coefficient(
        signal: np.ndarray,
        dt: float,
        f0: float,
        arrival_idx: int,
        window_cycles: float,
    ) -> complex:
        """Single-bin DFT of ``signal`` at ``f0`` over a window of
        ``window_cycles`` periods centered on ``arrival_idx``.

        Returns the raw complex coefficient (not normalized). The convention
        used is ``sum_n x[n] * exp(-j * 2 * pi * f0 * t[n])`` over the gated
        window. A narrowband real tone of the form ``cos(2*pi*f0*t + phi0)``
        therefore yields a coefficient whose phase equals ``+phi0`` in the
        long-window limit (because
        ``cos(w t + phi) * exp(-j w t) = 0.5*(exp(j phi) + exp(-j(2 w t + phi)))``
        and only the first term averages to non-zero). Callers treat
        ``angle(coef)`` as the received narrowband phase ``phi_i``; with the
        propagation model ``signal_i(t) = cos(omega*(t - tau_i))`` this gives
        ``phi_i = -omega * tau_i``, so ``tau_i = -phi_i/(2*pi*f0)``.
        """
        n = len(signal)
        if n == 0:
            return 0j
        win_len = max(2, int(round(window_cycles / f0 / dt)))
        half = win_len // 2
        start = max(0, arrival_idx - half)
        end = min(n, start + win_len)
        # If we clipped at the end, slide start back so we use the full window
        # when possible.
        if end - start < win_len:
            start = max(0, end - win_len)
        segment = signal[start:end]
        t = (np.arange(start, end)) * dt
        basis = np.exp(-1j * 2 * np.pi * f0 * t)
        return complex(np.sum(segment * basis))

    def _run_reciprocal_simulation_complex(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run the reciprocal k-wave sim and extract narrowband (a_i, phi_i).

        Returns:
            amplitudes: 1D array of |coefficient| per element.
            phases:     1D array of angle(coefficient) per element, in radians.
                        Positive phase means the received narrowband tone leads
                        the reference cosine at f0 (i.e. ``coef = a * exp(j*phi)``
                        with basis ``exp(-j*omega*t)``).
            f0:         Operating frequency extracted from the transducer (Hz).
        """
        from scipy.signal import butter, hilbert, sosfiltfilt

        from openlifu.sim.kwave_if import run_point_source_simulation

        # -- Reference sound speeds ------------------------------------------------
        if 'sound_speed' in params and 'ref_value' in params['sound_speed'].attrs:
            sound_speed_ref = params['sound_speed'].attrs['ref_value']
        else:
            sound_speed_ref = self.c0

        if 'sound_speed' in params:
            sound_speed_max = float(np.max(params['sound_speed'].to_numpy()))
        else:
            sound_speed_max = sound_speed_ref
        sound_speed_max = max(sound_speed_max, sound_speed_ref)

        # -- Operating frequency ---------------------------------------------------
        f0 = float(arr.frequency)

        # -- Grid / coordinate bookkeeping ----------------------------------------
        coord_dims = list(params.coords.dims)
        coord_units = params[coord_dims[0]].attrs.get('units', 'mm')
        _DIM_IDX = {'x': 0, 'y': 1, 'z': 2}

        matrix = np.asarray(transform, dtype=float) if transform is not None else np.eye(4)
        scl_m_to_coord = getunitconversion("m", coord_units)
        element_positions_raw = np.array([
            el.get_position(units="m", matrix=matrix) * scl_m_to_coord
            for el in arr.elements
        ])
        target_pos_raw = target.get_position(units=coord_units)

        coord_arrays = [params.coords[dim].to_numpy() for dim in coord_dims]
        grid_shape = tuple(len(c) for c in coord_arrays)

        # -- Element-to-voxel mapping (with out-of-grid handling) ------------------
        sensor_indices = []
        out_of_grid = set()
        for el_i, epos_xyz in enumerate(element_positions_raw):
            idx = []
            inside = True
            bad_axis: tuple[str, float, float] | None = None
            for dim_i, dim_name in enumerate(coord_dims):
                coord_vals = coord_arrays[dim_i]
                pos_component = epos_xyz[_DIM_IDX[dim_name]]
                cmin, cmax = float(coord_vals[0]), float(coord_vals[-1])
                if cmin > cmax:
                    cmin, cmax = cmax, cmin
                half_step = abs(float(coord_vals[1] - coord_vals[0])) / 2 if len(coord_vals) > 1 else 0
                if pos_component < cmin - half_step or pos_component > cmax + half_step:
                    if inside:
                        bad_axis = (dim_name, cmin, cmax)
                    inside = False
                nearest_idx = int(np.argmin(np.abs(coord_vals - pos_component)))
                idx.append(nearest_idx)
            if not inside:
                out_of_grid.add(el_i)
                if not self.allow_out_of_grid_fallback:
                    dim_name, cmin, cmax = bad_axis
                    raise ValueError(
                        f"Element {el_i} at position {epos_xyz} is outside the simulation grid "
                        f"with coord bounds along {dim_name}=[{cmin}, {cmax}]. "
                        "Set allow_out_of_grid_fallback=True to silently fall back to geometric "
                        "time-of-flight (not recommended for real pipelines)."
                    )
                logger.warning(
                    f"Element {el_i} at position {epos_xyz} is outside the simulation grid. "
                    "Using geometric/unit-amplitude fallback for this element.",
                )
            sensor_indices.append(tuple(idx))

        sensor_mask = np.zeros(grid_shape, dtype=int)
        for idx in sensor_indices:
            sensor_mask[idx] = 1

        # -- Target voxel ----------------------------------------------------------
        target_idx = []
        for dim_i, dim_name in enumerate(coord_dims):
            coord_vals = coord_arrays[dim_i]
            pos_component = target_pos_raw[_DIM_IDX[dim_name]]
            cmin, cmax = float(coord_vals[0]), float(coord_vals[-1])
            if cmin > cmax:
                cmin, cmax = cmax, cmin
            half_step = abs(float(coord_vals[1] - coord_vals[0])) / 2 if len(coord_vals) > 1 else 0
            if pos_component < cmin - half_step or pos_component > cmax + half_step:
                logger.warning(
                    "Target position %s is outside the simulation grid on axis %s.",
                    target_pos_raw, dim_name,
                )
                raise ValueError("Target outside simulation grid")
            nearest_idx = int(np.argmin(np.abs(coord_vals - pos_component)))
            target_idx.append(nearest_idx)
        target_idx = tuple(target_idx)

        source_mask = np.zeros(grid_shape, dtype=int)
        source_mask[target_idx] = 1

        # -- Simulation time budget ------------------------------------------------
        scl_to_m = getunitconversion(coord_units, 'm')
        dists_m = np.linalg.norm(
            element_positions_raw - target_pos_raw, axis=1,
        ) * scl_to_m
        max_dist_m = float(np.max(dists_m))
        t_end_needed = max_dist_m / sound_speed_ref * 1.5 + self.n_cycles / f0

        # -- Run sim ---------------------------------------------------------------
        sensor_data, dt = run_point_source_simulation(
            params=params,
            source_mask=source_mask,
            sensor_mask=sensor_mask,
            freq=f0,
            n_cycles=self.n_cycles,
            sound_speed_ref=sound_speed_ref,
            cfl=self.cfl,
            gpu=self.gpu,
            t_end=t_end_needed,
        )

        # -- Voxel-to-column lookup (Fortran order after xyz transpose) ------------
        perm_to_xyz = [coord_dims.index(d) for d in ['x', 'y', 'z']]
        sensor_mask_xyz = np.transpose(sensor_mask, perm_to_xyz)
        grid_shape_xyz = sensor_mask_xyz.shape

        nonzero_xyz = list(zip(*np.nonzero(sensor_mask_xyz)))

        def fortran_linear_index(idx, shape):
            lin = idx[0]
            stride = shape[0]
            for d in range(1, len(shape)):
                lin += idx[d] * stride
                stride *= shape[d]
            return lin

        nonzero_with_fortran = [(fortran_linear_index(idx, grid_shape_xyz), idx) for idx in nonzero_xyz]
        nonzero_with_fortran.sort(key=lambda x: x[0])
        sorted_nonzero = [item[1] for item in nonzero_with_fortran]
        voxel_to_row = {idx: row for row, idx in enumerate(sorted_nonzero)}

        # -- Optional narrowband bandpass ------------------------------------------
        fs = 1.0 / dt
        low = f0 * (1.0 - self.bandwidth_frac / 2.0)
        high = f0 * (1.0 + self.bandwidth_frac / 2.0)
        # Only filter if the band fits safely below Nyquist.
        do_filter = 0.0 < low < high < 0.5 * fs
        sos = None
        if do_filter:
            try:
                sos = butter(N=4, Wn=[low, high], btype="bandpass", fs=fs, output="sos")
            except ValueError:
                sos = None
                do_filter = False

        # -- Per-element narrowband extraction -------------------------------------
        n_elements = len(arr.elements)
        amplitudes = np.zeros(n_elements)
        phases = np.zeros(n_elements)

        for el_i, sensor_idx in enumerate(sensor_indices):
            if el_i in out_of_grid:
                # Fall back: use geometric time-of-flight phase and unit amplitude.
                dist_grid = np.linalg.norm(
                    element_positions_raw[el_i] - target_pos_raw,
                )
                dist_m = dist_grid * getunitconversion(coord_units, 'm')
                tof = dist_m / sound_speed_ref
                amplitudes[el_i] = 1.0
                # phase = -omega * tof so that delay = -phi/(2*pi*f0) = tof
                phases[el_i] = -2 * np.pi * f0 * tof
                continue

            sensor_idx_xyz = tuple(sensor_idx[i] for i in perm_to_xyz)
            col = voxel_to_row[sensor_idx_xyz]
            time_series = np.asarray(sensor_data[:, col], dtype=float)

            if do_filter and sos is not None:
                filtered = sosfiltfilt(sos, time_series)
            else:
                filtered = time_series

            # Locate the first-arrival sample using threshold detection (same
            # approach as SimulationCorrected) so the DFT window centers on the
            # direct wavefront, not late skull reverberations.
            analytic = hilbert(filtered)
            envelope = np.abs(analytic)
            earliest_arrival_s = (
                np.linalg.norm(element_positions_raw[el_i] - target_pos_raw)
                * scl_to_m
                / sound_speed_max
            )
            gate_start = max(0, int((earliest_arrival_s - 2 * dt) / dt))
            if gate_start >= len(envelope):
                gate_start = 0
            gated_env = envelope[gate_start:]
            max_env = float(np.max(gated_env)) if gated_env.size else 0.0
            if max_env == 0:
                arrival_idx = gate_start
            else:
                threshold = 0.1 * max_env
                above = np.where(gated_env >= threshold)[0]
                if len(above) == 0:
                    first_above = 0
                else:
                    first_above = int(above[0])
                win_half = 5
                win_start = max(0, first_above - win_half)
                win_end = min(len(gated_env), first_above + win_half + 1)
                local_peak = win_start + int(np.argmax(gated_env[win_start:win_end]))
                arrival_idx = gate_start + local_peak

            coef = self._extract_narrowband_coefficient(
                filtered, dt, f0, arrival_idx, self.window_cycles,
            )
            amplitudes[el_i] = abs(coef)
            phases[el_i] = float(np.angle(coef))

        return amplitudes, phases, f0

    def _fallback_complex_weights(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: use Direct (geometric) delays with unit amplitudes."""
        from openlifu.bf.delay_methods.direct import Direct
        direct = Direct(c0=self.c0)
        delays = direct.calc_delays(arr, target, params, transform)
        apod = np.ones_like(delays)
        return delays, apod

    def to_table(self) -> pd.DataFrame:
        """Get a table of the delay method parameters."""
        records = [
            {"Name": "Type", "Value": "ComplexWeighted", "Unit": ""},
            {"Name": "Default Sound Speed", "Value": self.c0, "Unit": "m/s"},
            {"Name": "CFL Number", "Value": self.cfl, "Unit": ""},
            {"Name": "Source Cycles", "Value": self.n_cycles, "Unit": ""},
            {"Name": "Use GPU", "Value": self.gpu, "Unit": ""},
            {"Name": "Allow Out-of-Grid Fallback", "Value": self.allow_out_of_grid_fallback, "Unit": ""},
            {"Name": "Bandpass Bandwidth Fraction", "Value": self.bandwidth_frac, "Unit": ""},
            {"Name": "DFT Window Cycles", "Value": self.window_cycles, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)
