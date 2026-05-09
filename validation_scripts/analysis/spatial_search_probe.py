#!/usr/bin/env python3
"""Summarize spatial-search-probe sidecars for Birnbaum subjects.

Reads per-subject timegated JSON sidecars (produced by
run_gladys_nnunet_subject.py run with EXPANDED_TARGET_PROBE=1) and prints a
side-by-side table comparing target-only vs spatial-search attenuation for
the skull (corrected) sims, using the water sim as baseline.

Intentionally uncommitted local helper.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _fnum(x, fmt=".4g"):
    if x is None:
        return "nan"
    try:
        if not math.isfinite(float(x)):
            return "nan"
        return format(float(x), fmt)
    except Exception:
        return "nan"


def _db_atten(p_baseline, p_skull):
    try:
        p_baseline = float(p_baseline)
        p_skull = float(p_skull)
        if not (math.isfinite(p_baseline) and math.isfinite(p_skull)):
            return float("nan")
        if p_baseline <= 0 or p_skull <= 0:
            return float("nan")
        return 20.0 * math.log10(p_baseline / p_skull)
    except Exception:
        return float("nan")


def summarize(subject: str, results_dir: Path, tag: str):
    sidecar_corr = results_dir / f"{subject}_{tag}gladys_nnunet_corrected_timegated.json"
    sidecar_geom = results_dir / f"{subject}_{tag}gladys_nnunet_geometric_timegated.json"
    sidecar_water = results_dir / f"{subject}_{tag}gladys_nnunet_water_timegated.json"
    missing = [p for p in (sidecar_corr, sidecar_geom, sidecar_water) if not p.exists()]
    if missing:
        return {
            "subject": subject,
            "error": f"missing: {[str(m) for m in missing]}",
        }

    with open(sidecar_corr) as f:
        corr = json.load(f)
    with open(sidecar_geom) as f:
        geom = json.load(f)
    with open(sidecar_water) as f:
        water = json.load(f)

    # Target-only water-gate metric (from skull corrected)
    p_fw_target_corr = corr.get("p_focal_window_water_gate_at_target_Pa", float("nan"))
    p_fw_target_geom = geom.get("p_focal_window_water_gate_at_target_Pa", float("nan"))

    # Water baseline: water sim's target via its own geom gate
    p_fw_water_target = water.get("p_focal_window_at_target_Pa", float("nan"))
    p_fw_water_cube = water.get("spatial_max_p_focal_window_Pa", float("nan"))

    # Spatial max for skull sims (water-gate applied within cube)
    p_fw_cube_corr = corr.get("spatial_max_p_focal_window_water_gate_Pa", float("nan"))
    p_fw_cube_geom = geom.get("spatial_max_p_focal_window_water_gate_Pa", float("nan"))

    off_corr = corr.get("spatial_max_offset_from_target_mm", float("nan"))
    off_geom = geom.get("spatial_max_offset_from_target_mm", float("nan"))
    off_water = water.get("spatial_max_offset_from_target_mm", float("nan"))

    spmax_corr_pos = corr.get("spatial_max_world_mm")
    spmax_corr_name = corr.get("spatial_max_sensor_name")
    target_mm = corr.get("target_mm")

    n_cube = corr.get("cube_n_sensors", 0)
    half = corr.get("cube_half_extent_mm")

    # Attenuation comparisons
    atten_target_only = _db_atten(p_fw_water_target, p_fw_target_corr)
    atten_spatial_vs_water_target = _db_atten(p_fw_water_target, p_fw_cube_corr)
    atten_spatial_vs_water_cube = _db_atten(p_fw_water_cube, p_fw_cube_corr)

    return {
        "subject": subject,
        "n_cube": n_cube,
        "cube_half_mm": half,
        "target_mm": target_mm,
        "p_fw_water_target": p_fw_water_target,
        "p_fw_water_cube": p_fw_water_cube,
        "p_fw_corr_target": p_fw_target_corr,
        "p_fw_corr_cube_max": p_fw_cube_corr,
        "p_fw_geom_target": p_fw_target_geom,
        "p_fw_geom_cube_max": p_fw_cube_geom,
        "offset_corr_mm": off_corr,
        "offset_geom_mm": off_geom,
        "offset_water_mm": off_water,
        "spatial_max_corr_world_mm": spmax_corr_pos,
        "spatial_max_corr_name": spmax_corr_name,
        "atten_target_only_dB": atten_target_only,
        "atten_spatial_vs_water_target_dB": atten_spatial_vs_water_target,
        "atten_spatial_vs_water_cube_dB": atten_spatial_vs_water_cube,
        "delta_dB": atten_target_only - atten_spatial_vs_water_target
        if math.isfinite(atten_target_only) and math.isfinite(atten_spatial_vs_water_target)
        else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+",
                    default=["GU008", "GU002", "GU010", "NC004"])
    ap.add_argument("--results-dir", default=str(Path.home() / "Data/openlifu-validation/results"))
    ap.add_argument("--tag", default="spatial_",
                    help="OUTPUT_TAG prefix used when the sims were run")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    rows = [summarize(s, rd, args.tag) for s in args.subjects]

    # Print table
    hdr = (
        f"{'Subject':<8} "
        f"{'n_cube':>7} {'half_mm':>7} "
        f"{'p_water_tgt':>12} {'p_water_cube':>13} "
        f"{'p_skull_tgt':>12} {'p_skull_cube':>13} "
        f"{'off_mm':>7} "
        f"{'att_tgt_dB':>11} {'att_sp_v_tgt':>13} {'att_sp_v_cub':>13} "
        f"{'delta_dB':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['subject']:<8}  ERROR: {r['error']}")
            continue
        print(
            f"{r['subject']:<8} "
            f"{r['n_cube']:>7d} {r['cube_half_mm']:>7} "
            f"{_fnum(r['p_fw_water_target']):>12} "
            f"{_fnum(r['p_fw_water_cube']):>13} "
            f"{_fnum(r['p_fw_corr_target']):>12} "
            f"{_fnum(r['p_fw_corr_cube_max']):>13} "
            f"{_fnum(r['offset_corr_mm'], '.2f'):>7} "
            f"{_fnum(r['atten_target_only_dB'], '.2f'):>11} "
            f"{_fnum(r['atten_spatial_vs_water_target_dB'], '.2f'):>13} "
            f"{_fnum(r['atten_spatial_vs_water_cube_dB'], '.2f'):>13} "
            f"{_fnum(r['delta_dB'], '.2f'):>9}"
        )

    # Also emit JSON for machine parsing
    print("\nJSON:")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
