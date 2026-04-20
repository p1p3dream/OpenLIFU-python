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
class SimulationCorrected(DelayMethod):
    """Delay method using k-wave simulation with reciprocity for phase correction.

    Places a virtual point source at the target and records pressure time series
    at all transducer element positions. The arrival time at each element encodes
    the true acoustic path through the heterogeneous skull model. Delays are
    computed as max(arrival_time) - arrival_time for each element.

    Uses a single k-wave simulation (via reciprocity) instead of one per element.
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
        OpenLIFUFieldData("Allow Out-of-Grid Fallback", "If True, silently fall back to geometric time-of-flight for elements outside the simulation grid. Default False raises an error, which catches pose / transform configuration bugs."),
    ] = False
    """Whether to silently fall back to geometric time-of-flight for elements
    outside the simulation grid. Default False raises an error so that pose /
    transform misconfiguration is caught instead of silently producing a large
    focal error."""

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

    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset, transform: np.ndarray | None = None):
        """Calculate delays using k-wave simulation with reciprocity.

        Fires a virtual point source at the target position and records the pressure
        time series at each transducer element location. Arrival times are extracted
        via the Hilbert envelope peak, and delays are computed so that all elements
        fire in phase at the target.

        Falls back to Direct (geometric) delays if k-wave is not available or
        if the simulation fails for any reason.

        Args:
        :param arr: The transducer array.
        :param target: The focal target point.
        :param params: Simulation grid dataset with sound_speed, density, attenuation fields.
        :param transform: Optional 4x4 affine transform for element positions.
        :returns: 1D numpy array of per-element delays in seconds.
        """
        try:
            import importlib
            if importlib.util.find_spec("kwave") is None:
                raise ImportError("k-wave not installed")
        except ImportError:
            logger.warning("k-wave not available. Falling back to Direct delay method.")
            return self._fallback_delays(arr, target, params, transform)

        try:
            arrival_times = self._run_reciprocal_simulation(arr, target, params, transform)
            delays = np.max(arrival_times) - arrival_times
            return delays
        except (RuntimeError, ValueError, IndexError, OSError):
            logger.exception("Simulation-corrected delay calculation failed. Falling back to Direct method.")
            return self._fallback_delays(arr, target, params, transform)

    def _run_reciprocal_simulation(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the reciprocal k-wave simulation and extract arrival times.

        Args:
            arr: The transducer array.
            target: The focal target point.
            params: Simulation grid dataset.
            transform: Optional 4x4 affine transform.

        Returns:
            arrival_times: 1D array of arrival times (seconds) per element.
        """
        from scipy.signal import hilbert

        from openlifu.sim.kwave_if import run_point_source_simulation

        # Get the reference sound speed from params if available
        if 'sound_speed' in params and 'ref_value' in params['sound_speed'].attrs:
            sound_speed_ref = params['sound_speed'].attrs['ref_value']
        else:
            sound_speed_ref = self.c0

        # Maximum sound speed in the grid (e.g. skull bone at ~3000 m/s).
        # Used below to gate the Hilbert envelope by the EARLIEST plausible
        # arrival time, since bone shortcuts can beat water-speed paths.
        if 'sound_speed' in params:
            sound_speed_max = float(np.max(params['sound_speed'].to_numpy()))
        else:
            sound_speed_max = sound_speed_ref
        # Safety: the earliest-arrival bound must not be later than c_ref would give.
        sound_speed_max = max(sound_speed_max, sound_speed_ref)

        # Get frequency from the transducer
        freq = arr.frequency

        # Compute element positions in the simulation coordinate frame.
        # get_position returns [x, y, z] but the grid dims may be in a
        # different order (e.g. [z, y, x]). We map dim names to position
        # component indices so each axis is compared correctly.
        coord_dims = list(params.coords.dims)
        coord_units = params[coord_dims[0]].attrs.get('units', 'mm')
        _DIM_IDX = {'x': 0, 'y': 1, 'z': 2}

        if transform is not None:
            matrix = np.asarray(transform, dtype=float).copy()
            matrix[0:3, 3] *= getunitconversion(arr.units, coord_units)
        else:
            matrix = np.eye(4)
        element_positions_raw = np.array([
            el.get_position(units=coord_units, matrix=matrix)
            for el in arr.elements
        ])

        # Get target position in grid units (also [x, y, z] order)
        target_pos_raw = target.get_position(units=coord_units)

        # Build the sensor mask: find nearest grid indices for each element
        coord_arrays = [params.coords[dim].to_numpy() for dim in coord_dims]
        grid_shape = tuple(len(c) for c in coord_arrays)

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
                        "This usually means the transducer pose transform is missing or wrong. "
                        "Set allow_out_of_grid_fallback=True to silently fall back to geometric "
                        "time-of-flight (not recommended for real pipelines)."
                    )
                logger.warning(
                    f"Element {el_i} at position {epos_xyz} is outside the simulation grid. "
                    "Using geometric time-of-flight estimate for this element."
                )
            sensor_indices.append(tuple(idx))

        # Build sensor mask (3D binary)
        sensor_mask = np.zeros(grid_shape, dtype=int)
        for idx in sensor_indices:
            sensor_mask[idx] = 1

        # Find the target voxel index, with out-of-grid check
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
                    "Target position %s is outside the simulation grid on axis %s. "
                    "Falling back to geometric delays.",
                    target_pos_raw, dim_name,
                )
                raise ValueError("Target outside simulation grid")
            nearest_idx = int(np.argmin(np.abs(coord_vals - pos_component)))
            target_idx.append(nearest_idx)
        target_idx = tuple(target_idx)

        # Build source mask (single voxel at target)
        source_mask = np.zeros(grid_shape, dtype=int)
        source_mask[target_idx] = 1

        # Compute the required simulation end time so the wave has enough
        # time to propagate from the point source (target) to the farthest
        # transducer element.  The auto-calculated time from k-wave is based
        # on grid extent alone, which can be too short for deep targets.
        scl_to_m = getunitconversion(coord_units, 'm')
        dists_m = np.linalg.norm(
            element_positions_raw - target_pos_raw, axis=1
        ) * scl_to_m
        max_dist_m = float(np.max(dists_m))
        # Propagation time with 1.5x safety margin (skull slows waves below
        # the reference speed) plus the source pulse duration.
        t_end_needed = max_dist_m / sound_speed_ref * 1.5 + self.n_cycles / freq

        # Run the point source simulation
        sensor_data, dt = run_point_source_simulation(
            params=params,
            source_mask=source_mask,
            sensor_mask=sensor_mask,
            freq=freq,
            n_cycles=self.n_cycles,
            sound_speed_ref=sound_speed_ref,
            cfl=self.cfl,
            gpu=self.gpu,
            t_end=t_end_needed,
        )

        # sensor_data is (n_timesteps, n_sensor_points).
        # Multiple elements may map to the same voxel if the grid is coarse.
        # We need to map sensor data columns back to elements.

        # k-wave receives the sensor mask transposed to [x,y,z] order
        # (done inside run_point_source_simulation). It returns data
        # columns in Fortran (column-major) order of that xyz mask. We
        # build the voxel-to-column lookup in xyz space, then convert
        # each sensor_idx (which is in coord_dims order) to xyz before
        # lookup. The dict is still called voxel_to_row for historical
        # reasons; its values are column indices into sensor_data.
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

        # Extract arrival time for each element
        n_elements = len(arr.elements)
        arrival_times = np.zeros(n_elements)

        for el_i, sensor_idx in enumerate(sensor_indices):
            if el_i in out_of_grid:
                # Element is outside the simulation grid; use geometric fallback.
                dist_grid = np.linalg.norm(
                    element_positions_raw[el_i] - target_pos_raw
                )
                dist_m = dist_grid * getunitconversion(coord_units, 'm')
                arrival_times[el_i] = dist_m / sound_speed_ref
                continue

            # Convert sensor_idx from coord_dims order to xyz order
            sensor_idx_xyz = tuple(sensor_idx[i] for i in perm_to_xyz)
            col = voxel_to_row[sensor_idx_xyz]
            time_series = sensor_data[:, col]
            # Compute the analytic signal envelope via the Hilbert transform
            analytic = hilbert(time_series)
            envelope = np.abs(analytic)
            # Gate out the early-time source-pulse leakage for elements near the
            # source voxel. Lower bound: earliest plausible arrival using c_max
            # (bone paths at ~3000 m/s can beat water-speed paths), minus a
            # small 2*dt buffer for numerical dispersion.
            earliest_arrival_s = (
                np.linalg.norm(element_positions_raw[el_i] - target_pos_raw)
                * scl_to_m
                / sound_speed_max
            )
            gate_start = max(0, int((earliest_arrival_s - 2 * dt) / dt))
            if gate_start >= len(envelope):
                gate_start = 0  # fallback, should not happen given t_end margin
            # The arrival time is the time of the envelope peak after the gate
            peak_sample = gate_start + int(np.argmax(envelope[gate_start:]))
            arrival_times[el_i] = peak_sample * dt

        return arrival_times

    def _fallback_delays(self, arr: Transducer, target: Point, params: xa.Dataset, transform: np.ndarray | None = None) -> np.ndarray:
        """Compute delays using the Direct (geometric) method as a fallback."""
        from openlifu.bf.delay_methods.direct import Direct
        direct = Direct(c0=self.c0)
        return direct.calc_delays(arr, target, params, transform)

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the delay method parameters

        :returns: Pandas DataFrame of the delay method parameters
        """
        records = [
            {"Name": "Type", "Value": "SimulationCorrected", "Unit": ""},
            {"Name": "Default Sound Speed", "Value": self.c0, "Unit": "m/s"},
            {"Name": "CFL Number", "Value": self.cfl, "Unit": ""},
            {"Name": "Source Cycles", "Value": self.n_cycles, "Unit": ""},
            {"Name": "Use GPU", "Value": self.gpu, "Unit": ""},
            {"Name": "Allow Out-of-Grid Fallback", "Value": self.allow_out_of_grid_fallback, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)
