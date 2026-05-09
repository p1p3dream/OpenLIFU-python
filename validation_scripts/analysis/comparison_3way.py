"""3-way comparison: isolate where heterogeneous simulation diverges.

Runs three simulations with identical transducer, delays, grid:
  A) Homogeneous water via ref_values_only=True  (scalar medium path)
  B) Heterogeneous medium from nnU-Net segmentation (array medium path)
  C) Manual homogeneous water arrays (array medium path, uniform values)

If A works and C fails, the array code path itself is broken.
If C works and B fails, the segmented medium values are the problem.
"""

import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/OpenLIFU-python/src"))
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/openlifu-env/lib:") + os.environ.get("LD_LIBRARY_PATH", "")

import xarray as xa
import nibabel as nib
from copy import deepcopy

from openlifu.seg.seg_methods.nnunet_seg import NNUNetSegmentation
from openlifu.bf.delay_methods.direct import Direct
from openlifu.xdc import Transducer
from openlifu.geo import Point
from openlifu.sim.kwave_if import run_simulation

# ─── Configuration ───────────────────────────────────────────────────────────
MRI_PATH = os.path.expanduser(
    "~/Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
    "Anonymized_Subjects/T1-Weighted MRI/GU008_deface.nii"
)
MODEL_PATH = os.path.expanduser("~/.openlifu/models/fullhead_seg.onnx")
CFL = 0.1
FREQ = 500e3
CYCLES = 5


def report(label, result, target_mm):
    """Print max pressure, focal location, focal error."""
    p_max = result['p_max'].to_numpy()
    focal_idx = np.unravel_index(p_max.argmax(), p_max.shape)
    dims = list(result['p_max'].dims)
    focal_mm = np.array([
        result.coords[d].to_numpy()[focal_idx[i]]
        for i, d in enumerate(dims)
    ])
    error = np.linalg.norm(focal_mm - target_mm)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Max pressure : {p_max.max():.4f} Pa")
    print(f"  Min pressure : {p_max.min():.6f} Pa")
    print(f"  Focal spot   : {focal_mm} mm")
    print(f"  Target       : {target_mm} mm")
    print(f"  Focal error  : {error:.2f} mm")
    print(f"  Field shape  : {p_max.shape}")

    # Pressure stats
    print(f"  Mean pressure: {p_max.mean():.6f} Pa")
    print(f"  Std pressure : {p_max.std():.6f} Pa")
    print(f"  99th pctile  : {np.percentile(p_max, 99):.6f} Pa")

    return {
        'max_pressure': p_max.max(),
        'focal_mm': focal_mm,
        'focal_error': error,
    }


def main():
    print("=" * 70)
    print("3-WAY COMPARISON: Scalar vs. Heterogeneous vs. Manual Homogeneous")
    print("=" * 70)

    # ─── Step 1: Load MRI and Segment ────────────────────────────────────
    print("\n[1] Loading MRI and segmenting...")
    img = nib.load(MRI_PATH)
    data = np.asarray(img.dataobj, dtype=np.float32)
    spacing = np.abs(np.diag(img.affine)[:3])

    coords_zyx = {}
    for i, dim in enumerate(["z", "y", "x"]):
        coords_zyx[dim] = xa.Variable(
            dim, np.arange(data.shape[i]) * spacing[i], attrs={"units": "mm"}
        )
    volume = xa.DataArray(data, dims=["z", "y", "x"], coords=coords_zyx)

    seg = NNUNetSegmentation(
        model_path=MODEL_PATH, model_type="fullhead",
        use_gpu=False, use_mirroring=False,
    )
    t0 = time.time()
    full_seg = seg._segment(volume)
    print(f"  Segmentation done in {time.time()-t0:.1f}s")

    # ─── Step 2: Place transducer (pre-transformed) ──────────────────────
    print("\n[2] Placing transducer...")
    seg_arr = full_seg.to_numpy()
    midx = seg._material_indices()

    target_z = data.shape[0] * spacing[0] / 2
    target_y = data.shape[1] * spacing[1] / 2
    target_x = data.shape[2] * spacing[2] / 2

    skull_z_indices = np.where(
        np.any(seg_arr == midx['skull'], axis=(1, 2))
    )[0]
    skull_top_z = skull_z_indices.max() * spacing[0] + 5

    transform = np.eye(4)
    transform[0, 3] = target_x
    transform[1, 3] = target_y
    transform[2, 3] = skull_top_z

    arr_origin = Transducer.gen_matrix_array(
        nx=8, ny=8, pitch=6.0, kerf=0.5, frequency=500e3, units="mm",
    )
    arr_placed = deepcopy(arr_origin)
    for el in arr_placed.elements:
        new_pos = el.get_position(units=el.units, matrix=transform)
        el.position = new_pos

    positions = np.array([el.get_position(units="mm") for el in arr_placed.elements])
    print(f"  Transducer center: ({positions[:,0].mean():.1f}, "
          f"{positions[:,1].mean():.1f}, {positions[:,2].mean():.1f}) mm")
    print(f"  Target: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) mm")

    # ─── Step 3: Build simulation grid ───────────────────────────────────
    print("\n[3] Building simulation grid...")
    margin = 10
    x_min = min(positions[:, 0].min(), target_x) - margin
    x_max = max(positions[:, 0].max(), target_x) + margin
    y_min = min(positions[:, 1].min(), target_y) - margin
    y_max = max(positions[:, 1].max(), target_y) + margin
    z_min = target_z - margin
    z_max = skull_top_z + margin

    crop_vol = volume.sel(
        z=slice(z_min, z_max),
        y=slice(y_min, y_max),
        x=slice(x_min, x_max),
    )
    print(f"  Grid shape: {dict(zip(crop_vol.dims, crop_vol.shape))}")

    # Get the segmentation-based params (heterogeneous medium)
    sim_params_hetero = seg.seg_params(crop_vol)
    print(f"  sim_params dims: {dict(sim_params_hetero.dims)}")

    # ─── Diagnostic: inspect the heterogeneous medium arrays ─────────────
    print("\n[3b] Medium array diagnostics:")
    for var in ['sound_speed', 'density', 'attenuation']:
        arr_data = sim_params_hetero[var].to_numpy()
        ref_val = sim_params_hetero[var].attrs['ref_value']
        print(f"  {var}:")
        print(f"    ref_value={ref_val}, min={arr_data.min():.2f}, "
              f"max={arr_data.max():.2f}, mean={arr_data.mean():.2f}")
        unique_vals = np.unique(arr_data)
        if len(unique_vals) <= 10:
            print(f"    unique values: {unique_vals}")
        else:
            print(f"    unique count: {len(unique_vals)}")

    # ─── Step 4: Geometric delays ────────────────────────────────────────
    target = Point(
        position=np.array([target_x, target_y, target_z]),
        id="tumor", name="GBM Target", units="mm",
    )
    target_mm = target.get_position(units="mm")

    print("\n[4] Computing geometric delays...")
    direct = Direct(c0=1500.0)
    delays = direct.calc_delays(arr_placed, target, sim_params_hetero)
    apod = np.ones(arr_placed.numelements())
    print(f"  Delay range: {delays.min()*1e6:.1f} to {delays.max()*1e6:.1f} us")

    # ─── Step 5: Build manual homogeneous params ─────────────────────────
    # Same Dataset structure as sim_params_hetero, but with uniform water values
    print("\n[5] Building manual homogeneous water params (same shape as grid)...")
    sim_params_manual = deepcopy(sim_params_hetero)
    water_c = 1500.0  # m/s
    water_rho = 1000.0  # kg/m^3
    water_alpha = 0.002  # dB/cm/MHz

    sim_params_manual['sound_speed'].data[:] = water_c
    sim_params_manual['density'].data[:] = water_rho
    sim_params_manual['attenuation'].data[:] = water_alpha
    print(f"  Filled sound_speed={water_c}, density={water_rho}, attenuation={water_alpha}")

    # Verify
    for var in ['sound_speed', 'density', 'attenuation']:
        arr_data = sim_params_manual[var].to_numpy()
        print(f"  {var}: min={arr_data.min():.4f}, max={arr_data.max():.4f}")

    # ─── Step 6: Run simulations ─────────────────────────────────────────
    common_kwargs = dict(
        arr=arr_placed,
        delays=delays,
        apod=apod,
        freq=FREQ,
        cycles=CYCLES,
        amplitude=1.0,
        cfl=CFL,
        gpu=True,
    )

    # --- Sim A: Homogeneous water (scalar path, ref_values_only=True) ---
    print("\n" + "="*70)
    print("[SIM A] Homogeneous water (ref_values_only=True, scalar medium)")
    print("="*70)
    t0 = time.time()
    result_a = run_simulation(
        params=sim_params_hetero,
        ref_values_only=True,
        **common_kwargs,
    )
    dt_a = time.time() - t0
    print(f"  Completed in {dt_a:.1f}s")
    stats_a = report("SIM A: Homogeneous water (scalar)", result_a, target_mm)

    # --- Sim B: Heterogeneous from segmentation (array path) ---
    print("\n" + "="*70)
    print("[SIM B] Heterogeneous medium from segmentation (ref_values_only=False)")
    print("="*70)
    t0 = time.time()
    result_b = run_simulation(
        params=sim_params_hetero,
        ref_values_only=False,
        **common_kwargs,
    )
    dt_b = time.time() - t0
    print(f"  Completed in {dt_b:.1f}s")
    stats_b = report("SIM B: Heterogeneous segmentation", result_b, target_mm)

    # --- Sim C: Manual homogeneous arrays (array path, uniform water) ---
    print("\n" + "="*70)
    print("[SIM C] Manual homogeneous water arrays (ref_values_only=False)")
    print("="*70)
    t0 = time.time()
    result_c = run_simulation(
        params=sim_params_manual,
        ref_values_only=False,
        **common_kwargs,
    )
    dt_c = time.time() - t0
    print(f"  Completed in {dt_c:.1f}s")
    stats_c = report("SIM C: Manual homogeneous arrays", result_c, target_mm)

    # ─── Final summary ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Simulation':<45} {'Max P (Pa)':>12} {'Focal Err (mm)':>16}")
    print("-"*75)
    for label, stats in [
        ("A: Homogeneous water (scalar)", stats_a),
        ("B: Heterogeneous segmentation (array)", stats_b),
        ("C: Manual homo water (array)", stats_c),
    ]:
        print(f"{label:<45} {stats['max_pressure']:>12.4f} {stats['focal_error']:>16.2f}")

    print()
    if stats_a['max_pressure'] > 0.1 and stats_c['max_pressure'] < 0.01:
        print("DIAGNOSIS: Array code path is broken (A works, C fails with same values)")
    elif stats_c['max_pressure'] > 0.1 and stats_b['max_pressure'] < 0.01:
        print("DIAGNOSIS: Segmented medium values cause the failure (C works, B fails)")
    elif stats_a['max_pressure'] > 0.1 and stats_b['max_pressure'] > 0.1:
        print("DIAGNOSIS: Both paths produce reasonable results")
    else:
        print("DIAGNOSIS: Unclear, review numbers above")

    # Extra: check if A and C produce similar fields
    p_a = result_a['p_max'].to_numpy()
    p_c = result_c['p_max'].to_numpy()
    corr = np.corrcoef(p_a.ravel(), p_c.ravel())[0, 1]
    rmse = np.sqrt(np.mean((p_a - p_c)**2))
    print(f"\nA vs C correlation: {corr:.6f}")
    print(f"A vs C RMSE: {rmse:.6f}")
    print(f"A vs C max-pressure ratio: {stats_c['max_pressure']/stats_a['max_pressure']:.4f}"
          if stats_a['max_pressure'] > 0 else "")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
