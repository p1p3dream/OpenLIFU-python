#!/usr/bin/env python3
"""Parameterized GLADYS-nnU-Net sim for Birnbaum full-head subjects.

Forked from run_gladys_nnunet.py (which was hardcoded to GU008). Adds:
  - --subject <SubjectID> flag to derive MRI + label paths
  - Inline focal-gain metric (p@target vs aperture-band mean) at 10 mm radius
  - A final one-line machine-parseable summary:
      SUBJECT_SUMMARY subject=<ID> bone_pct=<X> skull_path_near=<Y>
      p_water=<Z> p_skull=<W> p_geom_skull=<V>
      gain_vs_mean_water=<G> gain_vs_mean_skull=<H> atten_db=<A>
  - Per-subject output filenames so runs don't clobber each other

This script is intentionally NOT committed; it's a local batch-driver tool.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Workaround for kwave v3 logging bug (same as parent script)
# ---------------------------------------------------------------------------
_orig_log = logging.log

def _patched_log(level, msg, *args, **kwargs):
    try:
        if args and isinstance(msg, str) and "%" not in msg:
            msg = msg + " " + " ".join(str(a) for a in args)
            args = ()
    except Exception:
        pass
    return _orig_log(level, msg, *args, **kwargs)

logging.log = _patched_log

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import xarray as xa
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openlifu.bf.delay_methods.direct import Direct
from openlifu.bf.delay_methods.simulation_corrected import SimulationCorrected
from openlifu.geo import Point
from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.seg_methods.nnunet_seg import LABEL_MAP_FULLHEAD
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER
from openlifu.sim.kwave_if import run_simulation
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_ELEMENTS = 64
RADIUS_MM = 90.0
APERTURE_MM = 80.0
FREQ_HZ = 500e3
ELEMENT_SIZE_MM = 5.0
GRID_SPACING_MM = 0.5
CFL = 0.1
CYCLES = 5
AMPLITUDE = 1.0
C0 = 1500.0
T_END_SAFETY = 2.0
GRID_MARGIN_MM = 10.0
APERTURE_BAND_RADIUS_MM = 10.0


def _default_fullhead_materials() -> dict[str, Material]:
    m = MATERIALS.copy()
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


@dataclass
class PreSegmented(SegmentationMethod):
    label_nifti_path: str = ""
    nnunet_label_map: dict[int, str] = field(default_factory=lambda: dict(LABEL_MAP_FULLHEAD))
    materials: dict[str, Material] = field(default_factory=_default_fullhead_materials)

    def __post_init__(self):
        super().__post_init__()
        if not self.label_nifti_path:
            raise ValueError("label_nifti_path is required")
        img = nib.load(self.label_nifti_path)
        data = np.asarray(img.dataobj).astype(np.int16)
        affine = img.affine
        dim_names = ("x", "y", "z")
        coords = {}
        for axis, dim in enumerate(dim_names):
            origin = float(affine[axis, 3])
            spacing = float(affine[axis, axis])
            coord_values = origin + np.arange(data.shape[axis]) * spacing
            coords[dim] = xa.Variable(dim, coord_values, attrs={"units": "mm"})
        self._labels = xa.DataArray(data, dims=dim_names, coords=coords)
        mat_idx = self._material_indices()
        for label_int, mat_key in self.nnunet_label_map.items():
            if mat_key not in mat_idx:
                raise ValueError(
                    f"nnunet_label_map references material '{mat_key}' "
                    f"which is not in self.materials (for label {label_int})."
                )

    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        mat_idx = self._material_indices()
        src = self._labels
        src_dims = list(src.dims)
        src_origin = {d: float(src.coords[d].to_numpy()[0]) for d in src_dims}
        src_spacing = {}
        for d in src_dims:
            cs = src.coords[d].to_numpy()
            src_spacing[d] = float(cs[1] - cs[0]) if len(cs) > 1 else 1.0
        tgt_dims = list(volume.dims)
        assert set(tgt_dims) == set(src_dims)
        tgt_coord_arrays = [volume.coords[d].to_numpy() for d in src_dims]
        mg = np.meshgrid(*tgt_coord_arrays, indexing="ij")
        frac_idx = []
        for i, d in enumerate(src_dims):
            fi = (mg[i] - src_origin[d]) / src_spacing[d]
            frac_idx.append(fi)
        frac_stack = np.stack(frac_idx, axis=0)
        resampled = map_coordinates(
            src.to_numpy().astype(np.float32), frac_stack, order=0,
            mode="constant", cval=0.0,
        ).astype(np.int16)
        water_idx = mat_idx["water"]
        output = np.full(resampled.shape, water_idx, dtype=int)
        for nn_label, mat_key in self.nnunet_label_map.items():
            output[resampled == nn_label] = mat_idx[mat_key]
        labels_da = xa.DataArray(
            output, dims=src_dims,
            coords={d: volume.coords[d] for d in src_dims},
        )
        return labels_da.transpose(*tgt_dims)

    def to_table(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"Name": "Type", "Value": "PreSegmented (nnU-Net label NIfTI)", "Unit": ""},
            {"Name": "Label NIfTI", "Value": self.label_nifti_path, "Unit": ""},
            {"Name": "Reference Material", "Value": self.ref_material, "Unit": ""},
        ])


def create_hemispherical_array(
    n_elements=64, radius_mm=90.0, aperture_mm=80.0,
    freq_hz=500e3, element_size_mm=5.0,
) -> Transducer:
    half_aperture = aperture_mm / 2.0
    theta_max = np.arcsin(half_aperture / radius_mm)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    elements = []
    for i in range(n_elements):
        cos_theta = 1.0 - (1.0 - np.cos(theta_max)) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = golden_angle * i
        x = radius_mm * np.sin(theta) * np.cos(phi)
        y = radius_mm * np.sin(theta) * np.sin(phi)
        z = radius_mm * np.cos(theta)
        nx, ny, nz = -x, -y, -z
        az = np.arctan2(nx, nz)
        el = -np.arctan2(ny, np.sqrt(nx**2 + nz**2))
        elements.append(Element(
            index=i + 1, pin=i + 1,
            position=np.array([x, y, z]),
            orientation=np.array([az, el, 0.0]),
            size=np.array([element_size_mm, element_size_mm]),
            units="mm",
        ))
    return Transducer(
        id="hemi64", name=f"Hemispherical {n_elements}-element array",
        elements=elements, frequency=freq_hz, units="mm",
    )


def load_nifti_as_xarray(nifti_path: Path) -> xa.DataArray:
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    dim_names = ("x", "y", "z")
    coords = {}
    for axis, dim in enumerate(dim_names):
        origin = float(affine[axis, 3])
        spacing = float(affine[axis, axis])
        coord_values = origin + np.arange(data.shape[axis]) * spacing
        coords[dim] = xa.Variable(dim, coord_values, attrs={"units": "mm"})
    return xa.DataArray(data, dims=dim_names, coords=coords)


def compute_focal_gain(pmax, coord_arrays_xyz, target_mm, positions_world,
                        band_radius_mm=10.0):
    """Return dict with p_at_target, p_aperture_mean, gain_vs_mean."""
    cx, cy, cz = coord_arrays_xyz
    xx, yy, zz = np.meshgrid(cx, cy, cz, indexing="ij")
    coord_stack = np.stack([xx, yy, zz], axis=-1)
    tidx = tuple(
        int(np.argmin(np.abs(coord_arrays_xyz[ax] - target_mm[ax])))
        for ax in range(3)
    )
    p_at_target = float(pmax[tidx])
    r2 = band_radius_mm ** 2
    band_mask = np.zeros(pmax.shape, dtype=bool)
    for pos in positions_world:
        d2 = np.sum((coord_stack - pos) ** 2, axis=-1)
        band_mask |= d2 <= r2
    n_band = int(band_mask.sum())
    if n_band == 0:
        return {
            "p_at_target": p_at_target, "p_aperture_mean": float("nan"),
            "gain_vs_mean": float("nan"), "n_band": 0,
        }
    vals = pmax[band_mask]
    p_ap_mean = float(vals.mean())
    gain = p_at_target / p_ap_mean if p_ap_mean > 0 else float("nan")
    return {
        "p_at_target": p_at_target, "p_aperture_mean": p_ap_mean,
        "gain_vs_mean": gain, "n_band": n_band,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True,
                    help="Subject ID, e.g., GU002, GU010, NC004")
    ap.add_argument("--mri-path", default=None,
                    help="Override MRI path (default: birnbaum <subject>_deface.nii)")
    ap.add_argument("--label-path", default=None,
                    help="Override label path (default: ~/Data/openlifu-validation/results/<subject>_nnunet_labels.nii.gz)")
    ap.add_argument("--results-dir", default=None,
                    help="Results dir (default: ~/Data/openlifu-validation/results)")
    args = ap.parse_args()

    subj = args.subject
    mri_path = Path(args.mri_path) if args.mri_path else (
        Path.home() / "Data/openlifu-validation/datasets/birnbaum-fullhead/Data/"
        / "Anonymized_Subjects" / "T1-Weighted MRI" / f"{subj}_deface.nii"
    )
    label_path = Path(args.label_path) if args.label_path else (
        Path.home() / "Data/openlifu-validation/results" / f"{subj}_nnunet_labels.nii.gz"
    )
    results_dir = Path(args.results_dir) if args.results_dir else (
        Path.home() / "Data/openlifu-validation/results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 72)
    print(f"GLADYS nnU-Net sim | subject={subj}")
    print(f"  MRI:    {mri_path}")
    print(f"  Labels: {label_path}")
    print("=" * 72)
    if not mri_path.exists():
        print(f"ERROR: MRI not found: {mri_path}")
        sys.exit(1)
    if not label_path.exists():
        print(f"ERROR: labels not found: {label_path}")
        sys.exit(1)

    volume = load_nifti_as_xarray(mri_path)
    seg_method = PreSegmented(label_nifti_path=str(label_path))
    print(f"MRI shape: {volume.shape}, label shape: {seg_method._labels.shape}")

    lab_arr = seg_method._labels.to_numpy()
    print("nnU-Net label histogram (raw label file):")
    for nn_label, mat_key in sorted(seg_method.nnunet_label_map.items()):
        n = int((lab_arr == nn_label).sum())
        pct = 100.0 * n / lab_arr.size
        print(f"  {nn_label} ({mat_key}): {n:,d} ({pct:.2f}%)")

    # Segment MRI grid to find target
    seg_labels = seg_method._segment(volume)
    material_idx = seg_method._material_indices()
    seg_arr = seg_labels.to_numpy()
    coord_arrays = {d: volume.coords[d].to_numpy() for d in volume.dims}
    dim_names = list(volume.dims)

    brain_mask = np.zeros(seg_arr.shape, dtype=bool)
    for k in ("csf", "gray_matter", "white_matter"):
        if k in material_idx:
            brain_mask |= seg_arr == material_idx[k]
    if brain_mask.sum() == 0:
        brain_mask = seg_arr == material_idx["tissue"]
    brain_indices = np.argwhere(brain_mask)
    target_mm = np.array([
        float(np.mean(coord_arrays[dim_names[ax]][brain_indices[:, ax]]))
        for ax in range(3)
    ])
    print(f"Target (brain center): ({target_mm[0]:.1f}, {target_mm[1]:.1f}, {target_mm[2]:.1f}) mm")

    skull_mask = seg_arr == material_idx["skull"]
    skull_indices = np.argwhere(skull_mask)
    if skull_indices.size == 0:
        print("ERROR: no skull voxels in segmentation; aborting.")
        print(f"SUBJECT_SUMMARY subject={subj} STATUS=no_skull")
        sys.exit(2)
    skull_mm = np.array([
        coord_arrays[dim_names[ax]][skull_indices[:, ax]]
        for ax in range(3)
    ])
    max_skull_per_axis = np.array([
        float(skull_mm[ax].max() - target_mm[ax]) for ax in range(3)
    ])
    approach_axis = int(np.argmax(max_skull_per_axis))
    print(f"Approach axis: {dim_names[approach_axis]} (axis {approach_axis})")

    # Array
    arr_local = create_hemispherical_array(
        n_elements=N_ELEMENTS, radius_mm=RADIUS_MM, aperture_mm=APERTURE_MM,
        freq_hz=FREQ_HZ, element_size_mm=ELEMENT_SIZE_MM,
    )
    transform = np.eye(4)
    transform[:3, 3] = target_mm
    if approach_axis == 0:
        angle = np.pi / 2
        transform[:3, :3] = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
    elif approach_axis == 1:
        angle = -np.pi / 2
        transform[:3, :3] = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
    transform[:3, 3] = target_mm
    arr = deepcopy(arr_local)
    for el in arr.elements:
        world_pos = el.get_position(units="mm", matrix=transform)
        el.position = world_pos
        direction = target_mm - world_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            n = direction / dist
            az = np.arctan2(n[0], n[2])
            el_angle = -np.arctan2(n[1], np.sqrt(n[0]**2 + n[2]**2))
            el.orientation = np.array([az, el_angle, 0.0])
    positions = arr.get_positions(units="mm")

    # Build sim grid
    all_points = np.vstack([positions, target_mm[np.newaxis, :]])
    grid_min = np.floor((all_points.min(axis=0) - GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    grid_max = np.ceil((all_points.max(axis=0) + GRID_MARGIN_MM) / GRID_SPACING_MM) * GRID_SPACING_MM
    sim_coords = {}
    for ax, dim in enumerate(["x", "y", "z"]):
        n_pts = int(np.round((grid_max[ax] - grid_min[ax]) / GRID_SPACING_MM)) + 1
        sim_coords[dim] = xa.Variable(
            dim, np.linspace(grid_min[ax], grid_max[ax], n_pts),
            attrs={"units": "mm"},
        )
    grid_shape = tuple(len(sim_coords[d]) for d in ["x", "y", "z"])
    print(f"Sim grid: {grid_shape}")

    from scipy.interpolate import RegularGridInterpolator
    orig_coords_list = [volume.coords[d].to_numpy() for d in volume.dims]
    interp = RegularGridInterpolator(
        orig_coords_list, volume.to_numpy(),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    sim_coord_arrays = [sim_coords[d].data for d in ["x", "y", "z"]]
    mg = np.meshgrid(*sim_coord_arrays, indexing="ij")
    query_pts = np.stack([m.ravel() for m in mg], axis=-1)
    resampled_data = interp(query_pts).reshape(grid_shape).astype(np.float32)
    sim_volume = xa.DataArray(
        resampled_data, dims=["x", "y", "z"],
        coords={d: sim_coords[d] for d in ["x", "y", "z"]},
    )

    sim_params = seg_method.seg_params(sim_volume)

    sim_seg = seg_method._segment(sim_volume)
    sim_seg_arr = sim_seg.to_numpy()
    air_mask = sim_seg_arr == material_idx["air"]
    skull_mask_sim = sim_seg_arr == material_idx["skull"]
    water_mat = seg_method.materials["water"]
    n_skull = int(skull_mask_sim.sum())
    total_vox = sim_seg_arr.size
    pct_skull = 100.0 * n_skull / total_vox
    print(f"SIM grid bone fraction: {pct_skull:.2f}% ({n_skull:,d}/{total_vox:,d} voxels)")
    if air_mask.any():
        sim_params["sound_speed"].data[air_mask] = water_mat.sound_speed
        sim_params["density"].data[air_mask] = water_mat.density
        sim_params["attenuation"].data[air_mask] = water_mat.attenuation

    # Skull path along target->aperture ray
    aperture_center_mm = positions.mean(axis=0)
    ray_dir = aperture_center_mm - target_mm
    ray_len_mm = float(np.linalg.norm(ray_dir))
    skull_path_near_mm = float("nan")
    skull_path_far_mm = float("nan")
    if ray_len_mm > 1e-6:
        ray_unit = ray_dir / ray_len_mm
        probe_len_mm = 1.3 * RADIUS_MM
        step_mm = GRID_SPACING_MM / 2.0
        sim_origins = np.array([sim_coord_arrays[i][0] for i in range(3)])
        sim_specs = np.array([sim_coord_arrays[i][1] - sim_coord_arrays[i][0] for i in range(3)])

        def _skull_path(direction):
            n = int(np.ceil(probe_len_mm / step_mm)) + 1
            ts = np.linspace(0.0, probe_len_mm, n)
            pts = target_mm[None, :] + ts[:, None] * direction[None, :]
            frac = ((pts - sim_origins[None, :]) / sim_specs[None, :]).T
            sampled = map_coordinates(
                sim_seg_arr.astype(np.float32), frac, order=0,
                mode="constant", cval=-1.0,
            ).astype(np.int16)
            return int((sampled == material_idx["skull"]).sum()) * step_mm

        skull_path_near_mm = _skull_path(ray_unit)
        skull_path_far_mm = _skull_path(-ray_unit)
    print(f"Skull path near-side: {skull_path_near_mm:.1f} mm, far-side: {skull_path_far_mm:.1f} mm")

    target = Point(position=target_mm.copy(), id="brain_center",
                   name="Brain Center Target", units="mm")
    direct = Direct(c0=C0)
    delays_geo = direct.calc_delays(arr, target, sim_params)

    from openlifu.sim.sim_setup import SimSetup
    sim_setup = SimSetup(
        spacing=GRID_SPACING_MM, units="mm",
        x_extent=(float(grid_min[0]), float(grid_max[0])),
        y_extent=(float(grid_min[1]), float(grid_max[1])),
        z_extent=(float(grid_min[2]), float(grid_max[2])),
        c0=C0, cfl=CFL,
    )
    max_dist_mm = sim_setup.get_max_distance(arr, units="mm")
    t_end = (max_dist_mm * 1e-3) / C0 * T_END_SAFETY
    c_max = float(sim_params["sound_speed"].to_numpy().max())
    dx_m = GRID_SPACING_MM * 1e-3
    dt = CFL * dx_m / c_max

    sim_corrected = SimulationCorrected(c0=C0, cfl=CFL, n_cycles=3, gpu=True)
    delays_corrected = sim_corrected.calc_delays(arr, target, sim_params)

    apod = np.ones(arr.numelements())
    common_kwargs = dict(
        arr=arr, apod=apod, freq=FREQ_HZ, cycles=CYCLES, amplitude=AMPLITUDE,
        dt=dt, t_end=t_end, cfl=CFL, gpu=True, source_method="point_source",
    )

    print("\n[SIM A] corrected + skull (nnU-Net)")
    t0 = time.time()
    result_a = run_simulation(params=sim_params, delays=delays_corrected,
                              ref_values_only=False, **common_kwargs)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[SIM B] geometric + skull")
    t0 = time.time()
    result_b = run_simulation(params=sim_params, delays=delays_geo,
                              ref_values_only=False, **common_kwargs)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[SIM C] geometric + water")
    t0 = time.time()
    result_c = run_simulation(params=sim_params, delays=delays_geo,
                              ref_values_only=True, **common_kwargs)
    print(f"  done in {time.time()-t0:.1f}s")

    cx = sim_coord_arrays[0]
    cy = sim_coord_arrays[1]
    cz = sim_coord_arrays[2]

    def _stats(result, label):
        pmax = result["p_max"].to_numpy()
        fg = compute_focal_gain(pmax, (cx, cy, cz), target_mm, positions,
                                 band_radius_mm=APERTURE_BAND_RADIUS_MM)
        raw_max = float(pmax.max())
        idx = np.unravel_index(pmax.argmax(), pmax.shape)
        raw_loc = np.array([float(cx[idx[0]]), float(cy[idx[1]]), float(cz[idx[2]])])
        err = float(np.linalg.norm(raw_loc - target_mm))
        print(f"  {label}: raw_max={raw_max:.4g} Pa (err={err:.1f}mm), "
              f"p@target={fg['p_at_target']:.4g} Pa, "
              f"p_ap_mean={fg['p_aperture_mean']:.4g} Pa, "
              f"gain_vs_mean={fg['gain_vs_mean']:.3f}")
        return fg, raw_max, err

    print("\n--- FOCAL STATS ---")
    stats_a, _, _ = _stats(result_a, "A corrected+skull")
    stats_b, _, _ = _stats(result_b, "B geometric+skull")
    stats_c, _, _ = _stats(result_c, "C geometric+water ")

    p_water = stats_c["p_at_target"]
    p_skull = stats_a["p_at_target"]
    p_geom_skull = stats_b["p_at_target"]
    if p_water > 0 and p_skull > 0:
        atten_db = 20.0 * np.log10(p_water / p_skull)
    else:
        atten_db = float("nan")

    # Save pmax NIfTIs with subject prefix
    for sim_label, result in [("corrected", result_a), ("geometric", result_b), ("water", result_c)]:
        p_max_data = result["p_max"].to_numpy()
        out_affine = np.diag([
            float(cx[1] - cx[0]) if len(cx) > 1 else 1.0,
            float(cy[1] - cy[0]) if len(cy) > 1 else 1.0,
            float(cz[1] - cz[0]) if len(cz) > 1 else 1.0,
            1.0,
        ])
        out_affine[0, 3] = float(cx[0])
        out_affine[1, 3] = float(cy[0])
        out_affine[2, 3] = float(cz[0])
        out_path = results_dir / f"{subj}_gladys_nnunet_{sim_label}_pmax.nii.gz"
        nib.save(nib.Nifti1Image(p_max_data.astype(np.float32), out_affine), str(out_path))

    t_elapsed = time.time() - t_total
    print(f"\n[{subj}] Total: {t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")
    print(
        f"SUBJECT_SUMMARY subject={subj} "
        f"bone_pct={pct_skull:.3f} "
        f"skull_path_near={skull_path_near_mm:.2f} "
        f"skull_path_far={skull_path_far_mm:.2f} "
        f"p_water={p_water:.6g} "
        f"p_skull={p_skull:.6g} "
        f"p_geom_skull={p_geom_skull:.6g} "
        f"gain_vs_mean_water={stats_c['gain_vs_mean']:.4f} "
        f"gain_vs_mean_skull={stats_a['gain_vs_mean']:.4f} "
        f"atten_db={atten_db:.2f}"
    )


if __name__ == "__main__":
    main()
