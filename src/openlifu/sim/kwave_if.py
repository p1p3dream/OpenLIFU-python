from __future__ import annotations

import contextlib
import logging
import pathlib
from copy import deepcopy
from typing import List

import numpy as np
import xarray as xa

from openlifu import xdc
from openlifu.util.units import getunitconversion


def get_kgrid(coords: xa.Coordinates, t_end = 0, dt = 0, sound_speed_ref=1500, cfl=0.5):
    from kwave.kgrid import kWaveGrid
    units = [coords[dim].attrs['units'] for dim in coords.dims]
    if not all(unit == units[0] for unit in units):
        raise ValueError("All coordinates must have the same units")
    scl = getunitconversion(units[0], 'm')
    # kWaveGrid expects [Nx, Ny, Nz] = [x, y, z] order.
    # coords.dims may be in any order (e.g. z, y, x), so reorder by name.
    _dim_order = {'x': 0, 'y': 1, 'z': 2}
    dim_names = list(coords.dims)
    sz = [0, 0, 0]
    dx = [0.0, 0.0, 0.0]
    for dim in dim_names:
        idx = _dim_order[dim]
        sz[idx] = len(coords[dim])
        dx[idx] = float(np.diff(coords[dim].to_numpy())[0]) * scl
    kgrid = kWaveGrid(sz, dx)
    if dt == 0 or t_end == 0:
        kgrid.makeTime(sound_speed_ref, cfl)
    else:
        Nt = round(t_end / dt)
        kgrid.setTime(Nt, dt)
    return kgrid

def get_karray(arr: xdc.Transducer,
               bli_tolerance: float = 0.05,
               upsampling_rate: int = 5,
               translation: List[float] = [0.,0.,0.],
               rotation: List[float] = [0.,0.,0.],
               transform: np.ndarray | None = None):
    import kwave
    import kwave.data
    from kwave.utils.kwave_array import kWaveArray
    karray = kWaveArray(bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate,
                        single_precision=True)
    matrix = transform if transform is not None else np.eye(4)
    for el in arr.elements:
        ele_pos = list(el.get_position(units="m", matrix=matrix))
        ele_w, ele_l = el.get_size(units="m")
        ele_angle = list(el.get_angle(units="deg"))
        karray.add_rect_element(ele_pos, ele_w, ele_l, ele_angle)
    translation = kwave.data.Vector(translation)
    rotation = kwave.data.Vector(rotation)
    karray.set_array_position(translation, rotation)
    return karray

def _reorder_to_xyz(params: xa.Dataset, var_name: str) -> np.ndarray:
    """Reorder a 3D variable from params dim order to [x, y, z] for k-wave."""
    arr = params[var_name]
    target_order = ['x', 'y', 'z']
    current_order = list(arr.dims)
    if current_order == target_order:
        return arr.data
    return arr.transpose(*target_order).data

def get_medium(params: xa.Dataset, ref_values_only: bool = False):
    from kwave.kmedium import kWaveMedium
    if ref_values_only:
        medium = kWaveMedium(sound_speed=params['sound_speed'].attrs['ref_value'],
                             density=params['density'].attrs['ref_value'],
                             alpha_coeff=params['attenuation'].attrs['ref_value'],
                             alpha_power=0.9,
                             alpha_mode='no_dispersion')
    else:
        medium= kWaveMedium(sound_speed=_reorder_to_xyz(params, 'sound_speed'),
                        density=_reorder_to_xyz(params, 'density'),
                        alpha_coeff=_reorder_to_xyz(params, 'attenuation'),
                        alpha_power=0.9,
                        alpha_mode='no_dispersion')
    return medium

def get_sensor(kgrid, record=['p_max','p_min']):
    from kwave.ksensor import kSensor
    sensor_mask = np.ones([kgrid.Nx, kgrid.Ny, kgrid.Nz])
    sensor = kSensor(sensor_mask, record=record)
    return sensor

def get_source(kgrid, karray, source_sig):
    from kwave.ksource import kSource
    source = kSource()
    logging.info("Getting binary mask")
    source.p_mask = karray.get_array_binary_mask(kgrid)
    logging.info("Getting distributed source signal")
    source.p = karray.get_distributed_source_signal(kgrid, source_sig)
    return source


def get_point_source(
    arr: xdc.Transducer,
    params: xa.Dataset,
    source_mat: np.ndarray,
) -> 'kSource':
    """Build a k-wave source by placing each transducer element as a point source.

    Instead of using kWaveArray BLI (which fails for curved arrays), this maps
    each element to its nearest grid voxel and assigns its delayed source signal
    directly. Elements mapping to the same voxel have their signals summed.

    :param arr: Transducer with elements already in the simulation coordinate frame.
    :param params: Simulation grid dataset (provides coords and their ordering).
    :param source_mat: Source signals, shape (n_elements, n_timesteps), with delays
        and apodization already applied by Transducer.calc_output.
    :returns: kSource with p_mask and p set for k-wave simulation.
    """
    from collections import defaultdict
    from kwave.ksource import kSource

    coord_dims = list(params.dims)
    coord_units = params[coord_dims[0]].attrs.get('units', 'mm')
    _DIM_IDX = {'x': 0, 'y': 1, 'z': 2}

    coord_arrays = {dim: params.coords[dim].to_numpy() for dim in coord_dims}

    # Map each element to its nearest grid voxel (in params dim order)
    voxel_elements = defaultdict(list)  # voxel_tuple -> [element_indices]
    n_outside = 0
    for el_i, el in enumerate(arr.elements):
        pos_xyz = el.get_position(units=coord_units)
        idx = []
        inside = True
        for dim in coord_dims:
            cv = coord_arrays[dim]
            pc = pos_xyz[_DIM_IDX[dim]]
            cmin, cmax = float(cv.min()), float(cv.max())
            half_step = abs(float(cv[1] - cv[0])) / 2 if len(cv) > 1 else 0
            if pc < cmin - half_step or pc > cmax + half_step:
                inside = False
            idx.append(int(np.argmin(np.abs(cv - pc))))
        if not inside:
            n_outside += 1
            logging.warning(
                f"Element {el_i} at {pos_xyz} is outside grid, excluded from source."
            )
            continue
        voxel_elements[tuple(idx)].append(el_i)

    if n_outside > 0:
        logging.info(f"Point source: {len(voxel_elements)} voxels from "
                     f"{arr.numelements() - n_outside} elements ({n_outside} outside grid)")

    # Build source mask in params dim order, then transpose to xyz
    grid_shape = tuple(len(coord_arrays[d]) for d in coord_dims)
    mask_params = np.zeros(grid_shape, dtype=np.int32)
    for voxel in voxel_elements:
        mask_params[voxel] = 1

    # Transpose to [x, y, z] for k-wave
    perm_to_xyz = [coord_dims.index(d) for d in ['x', 'y', 'z']]
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm_to_xyz):
        inv_perm[p] = i
    mask_xyz = np.transpose(mask_params, inv_perm)

    # Build signal matrix ordered by k-wave's Fortran traversal of mask_xyz
    nonzero_xyz = list(zip(*np.nonzero(mask_xyz)))

    def fortran_linear_index(idx, shape):
        lin = idx[0]
        stride = shape[0]
        for d in range(1, len(shape)):
            lin += idx[d] * stride
            stride *= shape[d]
        return lin

    nonzero_sorted = sorted(nonzero_xyz, key=lambda idx: fortran_linear_index(idx, mask_xyz.shape))

    # Map xyz voxel tuples back to params-order voxel tuples for lookup
    def xyz_to_params(xyz_idx):
        return tuple(xyz_idx[p] for p in perm_to_xyz)

    n_timesteps = source_mat.shape[1]
    signal_matrix = np.zeros((len(nonzero_sorted), n_timesteps), dtype=source_mat.dtype)

    for row, xyz_voxel in enumerate(nonzero_sorted):
        params_voxel = xyz_to_params(xyz_voxel)
        element_indices = voxel_elements[params_voxel]
        # Sum signals from all elements at this voxel
        for el_i in element_indices:
            signal_matrix[row, :] += source_mat[el_i, :]

    source = kSource()
    source.p_mask = mask_xyz
    source.p = signal_matrix
    logging.info(f"Point source: {len(nonzero_sorted)} source voxels, "
                 f"{signal_matrix.shape[1]} timesteps")
    return source

def run_point_source_simulation(
    params: xa.Dataset,
    source_mask: np.ndarray,
    sensor_mask: np.ndarray,
    freq: float = 1e6,
    n_cycles: int = 3,
    sound_speed_ref: float = 1500.0,
    cfl: float = 0.3,
    gpu: bool = True,
    ref_values_only: bool = False,
    t_end: float = 0,
):
    """Run a k-wave simulation with a point source and sparse sensor mask.

    Used by the SimulationCorrected delay method to implement the reciprocity
    approach: a single point source is placed at the target and pressure time
    series are recorded at transducer element positions.

    :param params: Simulation grid dataset with sound_speed, density, attenuation.
    :param source_mask: 3D binary array with 1 at the source (target) voxel.
    :param sensor_mask: 3D binary array with 1s at sensor (element) voxels.
    :param freq: Source frequency in Hz.
    :param n_cycles: Number of cycles in the source pulse.
    :param sound_speed_ref: Reference speed of sound (m/s) for time stepping.
    :param cfl: Courant-Friedrichs-Lewy number for time stepping.
    :param gpu: Whether to use GPU acceleration.
    :param ref_values_only: If True, use reference (homogeneous) medium values.
    :param t_end: Minimum simulation end time in seconds.  When > 0 the kgrid
        time axis is extended to at least this value, overriding the automatic
        estimate that k-wave derives from the grid extent alone.  This is
        important for transcranial FUS where the propagation distance from the
        target to the farthest element can exceed what the grid size implies.
    :returns: Tuple of (sensor_data, dt) where sensor_data is a 2D array
        (n_sensor_points, n_timesteps) and dt is the time step in seconds.
    """
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    # Build kgrid with auto timing to obtain the CFL-derived dt.
    kgrid = get_kgrid(params.coords, sound_speed_ref=sound_speed_ref, cfl=cfl)
    dt = float(kgrid.dt)

    # Determine the minimum simulation end time.  The caller may supply an
    # explicit t_end (preferred, since _run_reciprocal_simulation has direct
    # access to element/target positions).  When t_end is not provided, fall
    # back to an estimate derived from source/sensor mask positions.
    min_t_end = 0.0
    if t_end > 0:
        min_t_end = t_end
    else:
        # Estimate from mask positions (legacy fallback).
        coord_dims = list(params.dims)
        coord_units = params[coord_dims[0]].attrs.get('units', 'mm')
        scl_to_m = getunitconversion(coord_units, 'm')
        coord_arrays_m = {
            dim: params.coords[dim].to_numpy() * scl_to_m for dim in coord_dims
        }
        src_nz = np.nonzero(source_mask)
        sen_nz = np.nonzero(sensor_mask)
        if len(src_nz[0]) > 0 and len(sen_nz[0]) > 0:
            src_pos = np.stack(
                [coord_arrays_m[coord_dims[ax]][src_nz[ax]] for ax in range(3)],
                axis=-1,
            )
            sen_pos = np.stack(
                [coord_arrays_m[coord_dims[ax]][sen_nz[ax]] for ax in range(3)],
                axis=-1,
            )
            from scipy.spatial.distance import cdist
            max_distance = float(cdist(src_pos, sen_pos).max())
            # 1.5x safety margin (skull slows waves) + source pulse duration
            min_t_end = max_distance / sound_speed_ref * 1.5 + n_cycles / freq

    # Extend kgrid time axis if the auto-calculated duration is too short.
    auto_t_end = float(kgrid.Nt * kgrid.dt)
    if min_t_end > 0 and auto_t_end < min_t_end:
        new_Nt = int(np.ceil(min_t_end / dt))
        logging.info(
            "Point source sim: extending time from %.1f us (%d steps) "
            "to %.1f us (%d steps) to ensure full propagation",
            auto_t_end * 1e6, int(kgrid.Nt),
            new_Nt * dt * 1e6, new_Nt,
        )
        kgrid.setTime(new_Nt, dt)
    else:
        logging.info(
            "Point source sim: auto time %.1f us (%d steps) is sufficient",
            auto_t_end * 1e6, int(kgrid.Nt),
        )

    # Build medium
    medium = get_medium(params, ref_values_only=ref_values_only)

    # Reorder masks from params dim order to k-wave [x,y,z] order.
    # SimulationCorrected builds masks in params.dims order (e.g. z,y,x).
    dim_names = list(params.dims)
    _dim_order = {'x': 0, 'y': 1, 'z': 2}
    perm = [_dim_order[d] for d in dim_names]
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    # Transpose from (dim0, dim1, dim2) to (x, y, z)
    source_mask_xyz = np.transpose(source_mask, inv_perm)
    sensor_mask_xyz = np.transpose(sensor_mask, inv_perm)

    # Build source: short sinusoidal burst at the target voxel
    source = kSource()
    source.p_mask = source_mask_xyz
    t_sig = np.arange(0, n_cycles / freq, dt)
    # Windowed tone burst: Hann window * sine
    if len(t_sig) > 1:
        window = np.hanning(len(t_sig))
    else:
        window = np.ones(len(t_sig))
    source_signal = window * np.sin(2 * np.pi * freq * t_sig)
    # kSource expects (n_source_points, n_timesteps) but for a single point
    # source, a 1D signal is acceptable. Reshape to (1, N) to be safe.
    source.p = source_signal.reshape(1, -1)

    # Build sensor
    sensor = kSensor(sensor_mask_xyz, record=['p'])

    # Run simulation
    logging.info("Running point source reciprocal simulation for delay correction")
    simulation_options = SimulationOptions(
        pml_auto=True,
        pml_inside=False,
        save_to_disk=True,
        data_cast='single',
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=gpu)
    try:
        output = kspaceFirstOrder3D(
            kgrid=kgrid,
            source=source,
            sensor=sensor,
            medium=medium,
            simulation_options=simulation_options,
            execution_options=execution_options,
        )
    finally:
        # Clean up temporary H5 files left by k-wave (#435)
        # Runs even if simulation raises, preventing temp file leaks.
        for fpath in [simulation_options.input_filename, simulation_options.output_filename]:
            with contextlib.suppress(OSError):
                pathlib.Path(fpath).unlink(missing_ok=True)
    logging.info("Point source reciprocal simulation complete")

    # output['p'] has shape (n_sensor_points, n_timesteps)
    sensor_data = output['p']
    if sensor_data.ndim == 1:
        sensor_data = sensor_data.reshape(1, -1)

    return sensor_data, dt


def run_simulation(arr: xdc.Transducer,
                   params: xa.Dataset,
                   delays: np.ndarray | None = None,
                   apod: np.ndarray | None = None,
                   freq: float = 1e6,
                   cycles: float = 20,
                   amplitude: float = 1,
                   dt: float = 0,
                   t_end: float = 0,
                   cfl: float = 0.5,
                   bli_tolerance: float = 0.05,
                   upsampling_rate: int = 5,
                   gpu: bool = True,
                   ref_values_only: bool = False,
                   return_kwave_outputs: bool = False,
                   return_kwave_inputs: bool = False,
                   sensor_record: List[str] = ['p_max', 'p_min'],
                   source_method: str = 'kwave_array',
                   _source = None,
                   _sensor = None,
                   transform: np.ndarray | None = None,
):
    """ Run a k-wave simulation for the given transducer array and parameters.
    Args:
        arr: The transducer array to simulate.
        params: The simulation parameters as an xarray Dataset. Must include 'sound_speed', '
        density', and 'attenuation' variables with appropriate units.
        delays: Optional array of time delays for each element in the transducer array, in seconds. If None, no delays will be applied.
        apod: Optional array of apodization values for each
            element in the transducer array. If None, no apodization will be applied.
        freq: The frequency of the input signal in Hz. Default is 1 MHz.
        cycles: The number of cycles in the input signal. Default is 20.
        amplitude: The amplitude of the input signal. Default is 1.
        dt: The time step for the simulation in seconds. If 0, it will be automatically calculated based on the CFL condition.
        t_end: The total time for the simulation in seconds. If 0, it will be automatically calculated based on the input signal duration and the CFL condition.
        cfl: The Courant-Friedrichs-Lewy (CFL) number for the simulation. Default is 0.5.
        bli_tolerance: The tolerance for the boundary layer integral method used in k-wave. Default
            is 0.05.
        upsampling_rate: The upsampling rate for the boundary layer integral method used in k-wave
            Default is 5.
        gpu: Whether to use GPU for the simulation. Default is True. If False, the simulation will run on the CPU. Note that running on CPU may be very slow for large simulations.
        ref_values_only: Whether to use only the reference values for the medium properties (sound speed, density, attenuation) instead of the full spatial maps. Default is False. Setting this to True can significantly speed up the simulation, but will not capture any spatial variations in the medium properties.
        return_kwave_outputs: Whether to return the raw outputs from k-wave in addition to the processed xarray Dataset. Default is False.
        return_kwave_inputs: Whether to return the inputs to k-wave (kgrid, source, sensor, medium) in addition to the processed xarray Dataset. Default is False.
        sensor_record: List of strings specifying which k-wave sensor outputs to record. Can include '
        p_max', 'p_min', and 'p'. Default is ['p_max', 'p_min'].

    Additional Args:
        _source: Optional kSource object to use for the simulation. If None, a source will be created based on the transducer array and input signal.
        _sensor: Optional kSensor object to use for the simulation. If None, a sensor

    Returns:
        An xarray Dataset containing the simulation results, with variables corresponding to the requested sensor outputs and
        coordinates corresponding to the spatial dimensions of the simulation. If return_kwave_outputs is True, also returns a dictionary containing the raw outputs from k-wave. If return_kwave_inputs is True, also returns a dictionary containing the inputs to k-wave.

            """
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())
    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end, cfl=cfl)

    # When t_end is auto (0), the default kgrid time is based on grid extent
    # alone. For transcranial FUS the needed time is larger because of element
    # delays, the full propagation path, and signal duration.  Check and
    # rebuild the kgrid with an explicit t_end when the auto value falls short.
    if t_end == 0:
        _coord_units = [params[dim].attrs['units'] for dim in params.dims]
        _scl_to_m = getunitconversion(_coord_units[0], 'm')
        _c_ref = float(params['sound_speed'].attrs.get('ref_value', 1500.0))
        _max_delay = float(np.max(np.abs(delays)))
        # Grid diagonal in metres (proxy for max propagation distance)
        _extents_sq = 0.0
        for dim in params.dims:
            cv = params.coords[dim].to_numpy()
            _extents_sq += ((float(cv[-1]) - float(cv[0])) * _scl_to_m) ** 2
        _grid_diagonal = float(np.sqrt(_extents_sq))
        _signal_duration = cycles / freq
        # Total needed time with a 10% safety margin
        _t_end_needed = (_max_delay + _grid_diagonal / _c_ref + _signal_duration) * 1.1
        _auto_t_end = float(kgrid.Nt * kgrid.dt)
        if _auto_t_end < _t_end_needed:
            logging.info(
                "run_simulation: auto t_end (%.1f us) too short for "
                "transcranial sim (need %.1f us); rebuilding kgrid.",
                _auto_t_end * 1e6, _t_end_needed * 1e6,
            )
            kgrid = get_kgrid(params.coords, dt=float(kgrid.dt), t_end=_t_end_needed, cfl=cfl)

    t = np.arange(0, np.min([cycles / freq, (kgrid.Nt-np.ceil(max(delays)/kgrid.dt))*kgrid.dt]), kgrid.dt)
    input_signal = amplitude * np.sin(2 * np.pi * freq * t)
    units = [params[dim].attrs['units'] for dim in params.dims]
    if not all(unit == units[0] for unit in units):
        raise ValueError("All dimensions must have the same units")
    scl = getunitconversion(units[0], 'm')
    # Build array offset in [x, y, z] order to match kWaveArray's convention.
    # params.coords may be in any order (e.g. z, y, x), so we map by name.
    _dim_order = {'x': 0, 'y': 1, 'z': 2}
    array_offset = [0.0, 0.0, 0.0]
    for dim in params.dims:
        array_offset[_dim_order[dim]] = -float(params.coords[dim].to_numpy().mean()) * scl

    medium = get_medium(params, ref_values_only=ref_values_only)
    if _sensor is not None:
        sensor = _sensor
    else:
        sensor = get_sensor(kgrid, sensor_record)
    if 'p_min' not in sensor_record:
        raise ValueError("p_min must be included in sensor_record")
    if _source is not None:
        source = _source
    else:
        source_mat = arr.calc_output(input_signal, kgrid.dt, delays, apod)
    if arr.crosstalk_frac != 0:
        # Simulate crosstalk by adding additional elements to the array for each element that
        #   is within the crosstalk distance, with the signal scaled by the crosstalk fraction.
        #   This is a simple model of crosstalk and may not capture all the complexities of real
        #   crosstalk, but it can be useful for testing and simulation purposes.
        crosstalk_arr = arr.copy()
        crosstalk_mat = source_mat
        positions = arr.get_positions(units="m")
        for src_idx in range(arr.numelements()):
            for dst_idx in range(arr.numelements()):
                if src_idx == dst_idx:
                    continue
                src_pos = np.array(positions[src_idx])
                dst_pos = np.array(positions[dst_idx])
                dist = np.linalg.norm(src_pos - dst_pos)
                if dist <= arr.crosstalk_dist:
                    crosstalk_arr.elements += [arr.elements[dst_idx].copy()]
                    crosstalk_mat = np.vstack((crosstalk_mat, arr.crosstalk_frac*source_mat[src_idx,:]))
        arr = crosstalk_arr
        source_mat = crosstalk_mat
    if source_method == 'point_source':
        source = get_point_source(arr, params, source_mat)
    else:
        karray = get_karray(arr,
                            translation=array_offset,
                            bli_tolerance=bli_tolerance,
                            upsampling_rate=upsampling_rate,
                            transform=transform)
        source = get_source(kgrid, karray, source_mat)
    logging.info("Running simulation")
    simulation_options = SimulationOptions(
                            pml_auto=True,
                            pml_inside=False,
                            save_to_disk=True,
                            data_cast='single'
                        )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=gpu)
    inputs = {'kgrid':kgrid, 'source':source, 'sensor':sensor, 'medium':medium,
              'simulation_options':simulation_options, 'execution_options':execution_options}
    try:
        output = kspaceFirstOrder3D(**deepcopy(inputs))
    finally:
        # Clean up temporary H5 files left by k-wave (#435)
        for fpath in [simulation_options.input_filename, simulation_options.output_filename]:
            with contextlib.suppress(OSError):
                pathlib.Path(fpath).unlink(missing_ok=True)
    logging.info('Simulation Complete')

    # k-wave output is in [Nx, Ny, Nz] = [x, y, z] order (Fortran).
    # Reshape to [x, y, z] then build xarray with named dims so it
    # aligns with params coords regardless of their original order.
    _dim_order = {'x': 0, 'y': 1, 'z': 2}
    sz_xyz = [0, 0, 0]
    for dim in params.dims:
        sz_xyz[_dim_order[dim]] = params.sizes[dim]
    xyz_dims = ['x', 'y', 'z']

    def _reshape_output(flat):
        """Reshape flat k-wave output to xarray with correct dim names."""
        arr_xyz = flat.reshape(sz_xyz, order='F')
        da = xa.DataArray(arr_xyz, dims=xyz_dims,
                         coords={d: params.coords[d] for d in xyz_dims})
        # Transpose to match params dim order
        return da.transpose(*params.dims)

    ds_dict = {}
    for record in sensor.record:
        if record == 'p_max':
            ds_dict['p_max'] = _reshape_output(output['p_max']).assign_attrs(
                                units='Pa', long_name='PPP')
            ds_dict['p_max'].name = 'p_max'
        elif record == 'p_min':
            ds_dict['p_min'] = (-1 * _reshape_output(output['p_min'])).assign_attrs(
                            units='Pa', long_name='PNP')
            ds_dict['p_min'].name = 'p_min'
            Z = params['density'] * params['sound_speed']
            pmin_reshaped = _reshape_output(output['p_min'])
            ds_dict['intensity'] = (1e-4 * pmin_reshaped**2 / (2 * Z))
            ds_dict['intensity'].attrs = {'units': 'W/cm^2', 'long_name': 'Intensity'}
            ds_dict['intensity'].name = 'I'
        elif record == 'p':
            pcoords = {d: params.coords[d] for d in ['x', 'y', 'z']}
            pcoords['t'] = np.arange(0, output['Nt']*kgrid.dt, kgrid.dt)
            p_xyz = output['p'].reshape([output['Nt'], *sz_xyz], order='F')
            da_p = xa.DataArray(p_xyz,
                         dims=['t', 'x', 'y', 'z'],
                         coords=pcoords,
                         attrs={'units':'Pa', 'long_name':'Pressure'})
            ds_dict['p'] = da_p.transpose('t', *params.dims)

    ds = xa.Dataset(ds_dict)
    if return_kwave_outputs and return_kwave_inputs:
        return ds, output, inputs
    elif return_kwave_outputs:
        return ds, output
    elif return_kwave_inputs:
        return ds, inputs
    return ds
