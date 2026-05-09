#!/usr/bin/env python3
"""Diagnose whether the 2*dt Hilbert envelope gate in SimulationCorrected is sufficient.

Runs a reciprocal point-source sim through the Ernie skull and compares
the Hilbert envelope peak at gate margins of 0, 2, 5, 10 dt. If peaks
shift when the gate changes, the 2*dt margin may clip real arrivals.

Usage: cd ~/OpenLIFU-python && PYTHONPATH=src python3 scripts/diagnose_hilbert_gate.py
"""
import argparse, json, logging, os, sys, time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import nibabel as nib, numpy as np, pandas as pd, xarray as xa
from scipy.ndimage import map_coordinates
from scipy.signal import hilbert
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.seg_methods.nnunet_seg import LABEL_MAP_FULLHEAD
from openlifu.seg.seg_methods.threshold_mri import CSF, GRAY_MATTER, WHITE_MATTER
from openlifu.sim.kwave_if import run_point_source_simulation
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
N_EL, R_MM, AP_MM, FREQ, EL_SZ = 64, 90.0, 80.0, 500e3, 5.0
CFL, C0, N_CYC = 0.3, 1500.0, 3
MARGINS = [0, 2, 5, 10]
LABEL_P = Path.home() / "Data/openlifu-validation/results/ernie_nnunet_labels.nii.gz"
T1_P = Path.home() / "Data/openlifu-validation/results/ernie_T1_ras.nii.gz"
OUT_P = Path.home() / "Data/openlifu-validation/results/hilbert_gate_diagnostic.json"
_DIM = {'x': 0, 'y': 1, 'z': 2}

def _fh_mats():
    m = MATERIALS.copy(); m["csf"], m["gray_matter"], m["white_matter"] = CSF, GRAY_MATTER, WHITE_MATTER; return m

@dataclass
class PreSeg(SegmentationMethod):
    label_nifti_path: str = ""
    nnunet_label_map: dict = field(default_factory=lambda: dict(LABEL_MAP_FULLHEAD))
    materials: dict = field(default_factory=_fh_mats)
    def __post_init__(self):
        super().__post_init__()
        img = nib.load(self.label_nifti_path); d = np.asarray(img.dataobj).astype(np.int16); a = img.affine
        c = {dim: xa.Variable(dim, a[i,3]+np.arange(d.shape[i])*a[i,i], attrs={"units":"mm"}) for i,dim in enumerate(("x","y","z"))}
        self._labels = xa.DataArray(d, dims=("x","y","z"), coords=c)
    def _segment(self, vol):
        mi = self._material_indices(); s = self._labels; sd = list(s.dims)
        o = {d: float(s.coords[d].to_numpy()[0]) for d in sd}
        sp = {d: float(s.coords[d].to_numpy()[1]-s.coords[d].to_numpy()[0]) for d in sd}
        mg = np.meshgrid(*[vol.coords[d].to_numpy() for d in sd], indexing="ij")
        fi = np.stack([(mg[i]-o[d])/sp[d] for i,d in enumerate(sd)])
        rs = map_coordinates(s.to_numpy().astype(np.float32), fi, order=0, mode="constant", cval=0).astype(np.int16)
        out = np.full(rs.shape, mi["water"], dtype=int)
        for nl, mk in self.nnunet_label_map.items(): out[rs==nl] = mi[mk]
        return xa.DataArray(out, dims=sd, coords={d: vol.coords[d] for d in sd}).transpose(*list(vol.dims))
    def to_table(self): return pd.DataFrame([{"Name":"PreSeg","Value":self.label_nifti_path,"Unit":""}])

def load_nii(p):
    img = nib.load(str(p)); d = np.asarray(img.dataobj, dtype=np.float32); a = img.affine
    c = {dim: xa.Variable(dim, a[i,3]+np.arange(d.shape[i])*a[i,i], attrs={"units":"mm"}) for i,dim in enumerate(("x","y","z"))}
    return xa.DataArray(d, dims=("x","y","z"), coords=c)

def make_array():
    tmax = np.arcsin(AP_MM/2/R_MM); ga = np.pi*(3-np.sqrt(5)); els = []
    for i in range(N_EL):
        ct = 1-(1-np.cos(tmax))*(i+0.5)/N_EL; t = np.arccos(ct); p = ga*i
        x,y,z = R_MM*np.sin(t)*np.cos(p), R_MM*np.sin(t)*np.sin(p), R_MM*np.cos(t)
        az, el = np.arctan2(-x,-z), -np.arctan2(-y, np.sqrt(x**2+z**2))
        els.append(Element(index=i+1, pin=i+1, position=np.array([x,y,z]),
                           orientation=np.array([az,el,0.0]), size=np.array([EL_SZ,EL_SZ]), units="mm"))
    return Transducer(id="hemi64", name="Hemi64", elements=els, frequency=FREQ, units="mm")

def fortran_lin(idx, shape):
    lin, stride = idx[0], shape[0]
    for d in range(1, len(shape)): lin += idx[d]*stride; stride *= shape[d]
    return lin

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", dest="gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false"); args = ap.parse_args()
    print("="*72+"\nHilbert Gate Diagnostic: Ernie skull\n"+"="*72)
    for p,n in [(LABEL_P,"Labels"),(T1_P,"T1")]:
        if not p.exists(): print(f"ERROR: {n} not found at {p}"); sys.exit(1)
    vol = load_nii(T1_P); seg = PreSeg(label_nifti_path=str(LABEL_P))
    print(f"T1 {vol.shape}, Labels {seg._labels.shape}")
    t0=time.time(); params = seg.seg_params(vol); print(f"Params computed in {time.time()-t0:.1f}s")
    c_ref = float(params['sound_speed'].attrs.get('ref_value', C0))
    c_max = max(float(np.max(params['sound_speed'].to_numpy())), c_ref)
    print(f"c_ref={c_ref:.0f}, c_max={c_max:.0f} m/s")
    # Brain center + approach
    sl = seg._segment(vol); mi = seg._material_indices(); sa = sl.to_numpy()
    bm = np.zeros(sa.shape, dtype=bool)
    for k in ("csf","gray_matter","white_matter"):
        if k in mi: bm |= (sa==mi[k])
    dn = list(vol.dims); ca = {d: vol.coords[d].to_numpy() for d in dn}
    bi = np.argwhere(bm)
    tgt = np.array([float(np.mean(ca[dn[a]][bi[:,a]])) for a in range(3)])
    print(f"Target: ({tgt[0]:.1f}, {tgt[1]:.1f}, {tgt[2]:.1f}) mm")
    ski = np.argwhere(sa==mi["skull"])
    msd = [float(ca[dn[a]][ski[:,a]].max()-tgt[a]) for a in range(3)]
    aa = int(np.argmax(msd)); print(f"Approach axis: {dn[aa]}")
    # Transform
    xf = np.eye(4)
    if aa==0:
        a=np.pi/2; xf[:3,:3]=[[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]
    elif aa==1:
        a=-np.pi/2; xf[:3,:3]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
    xf[:3,3] = tgt
    arr = deepcopy(make_array())
    for el in arr.elements: el.position = el.get_position(units="mm", matrix=xf)
    epos = np.array([el.position for el in arr.elements])
    # Crop params to bounding box around elements + target with padding
    PAD_MM = 25.0
    all_pts = np.vstack([epos, tgt.reshape(1,3)])
    cd = list(params.dims); cu = params[cd[0]].attrs.get('units','mm')
    slices = {}
    for di, d in enumerate(cd):
        cv = params.coords[d].to_numpy()
        ax = _DIM[d]
        lo, hi = float(all_pts[:, ax].min()) - PAD_MM, float(all_pts[:, ax].max()) + PAD_MM
        mask = (cv >= lo) & (cv <= hi)
        if mask.sum() < 10: mask[:] = True  # safety: don't crop to nothing
        slices[d] = mask
    params = params.isel({d: np.where(slices[d])[0] for d in cd})
    print(f"Cropped params to {dict(params.sizes)} (from full volume)")
    # Masks on cropped grid
    cdarrs = [params.coords[d].to_numpy() for d in cd]; gs = tuple(len(c) for c in cdarrs)
    sidx = [tuple(int(np.argmin(np.abs(cdarrs[di]-epos[ei][_DIM[cd[di]]]))) for di in range(3)) for ei in range(N_EL)]
    smask = np.zeros(gs, dtype=int)
    for ix in sidx: smask[ix] = 1
    tidx = tuple(int(np.argmin(np.abs(cdarrs[di]-tgt[_DIM[cd[di]]]))) for di in range(3))
    srcm = np.zeros(gs, dtype=int); srcm[tidx] = 1
    s2m = getunitconversion(cu, 'm')
    dists_m = np.linalg.norm(epos-tgt, axis=1)*s2m
    t_end = float(np.max(dists_m))/c_ref*1.5 + N_CYC/FREQ
    print(f"Grid {gs}, {int(smask.sum())} sensors, t_end={t_end*1e6:.1f}us, GPU={args.gpu}")
    print("\nRunning point source simulation...")
    t0=time.time()
    sd, dt = run_point_source_simulation(params=params, source_mask=srcm, sensor_mask=smask,
        freq=FREQ, n_cycles=N_CYC, sound_speed_ref=c_ref, cfl=CFL, gpu=args.gpu, t_end=t_end)
    selap = time.time()-t0; print(f"Done in {selap:.1f}s, shape={sd.shape}, dt={dt*1e9:.1f}ns")
    # Voxel-to-column mapping
    p2x = [cd.index(d) for d in ['x','y','z']]; smx = np.transpose(smask, p2x); gsx = smx.shape
    nzx = sorted(list(zip(*np.nonzero(smx))), key=lambda i: fortran_lin(i, gsx))
    v2c = {ix: c for c,ix in enumerate(nzx)}
    # Analyze
    results = []
    for ei in range(N_EL):
        dm = float(np.linalg.norm(epos[ei]-tgt)); d_m = dm*s2m
        gtof = d_m/c_ref; ea_s = d_m/c_max
        sx = tuple(sidx[ei][i] for i in p2x); col = v2c[sx]
        env = np.abs(hilbert(sd[:, col]))
        peaks = {}
        for mg in MARGINS:
            gs_ = max(0, int((ea_s - mg*dt)/dt))
            if gs_ >= len(env): gs_ = 0
            ps = gs_ + int(np.argmax(env[gs_:])); amp = float(env[ps])
            peaks[mg] = {"gate_start": gs_, "peak_sample": ps, "peak_time_us": round(ps*dt*1e6,3), "peak_amplitude": round(amp,6)}
        sens = len(set(peaks[m]["peak_sample"] for m in MARGINS)) > 1
        results.append({"element": ei, "distance_mm": round(dm,2), "geometric_tof_us": round(gtof*1e6,3),
                        "earliest_arrival_us": round(ea_s*1e6,3), "peaks_by_margin": peaks, "sensitive": sens})
    # Save
    out = {"dataset":"ernie","dt_ns":round(dt*1e9,2),"n_elements":N_EL,"sound_speed_ref":c_ref,
           "sound_speed_max":c_max,"grid_shape":list(gs),"sim_elapsed_s":round(selap,1),
           "gate_margins_tested":MARGINS,"elements":results}
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_P,"w") as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT_P}")
    # Summary
    ns = sum(1 for r in results if r["sensitive"])
    print(f"\n{'='*100}\nSUMMARY: {ns}/{N_EL} elements are gate-sensitive\n{'='*100}")
    hdr = f"{'El':>3} {'Dist':>7} {'GeomTOF':>9} {'G2dt':>5} {'P2dt':>5} {'P0dt':>5} {'Sens':>5} {'Amp':>10}"
    print(hdr+"\n"+"-"*len(hdr))
    for r in results:
        p2,p0 = r["peaks_by_margin"][2], r["peaks_by_margin"][0]
        print(f"{r['element']:>3} {r['distance_mm']:>7.1f} {r['geometric_tof_us']:>9.3f} "
              f"{p2['gate_start']:>5} {p2['peak_sample']:>5} {p0['peak_sample']:>5} "
              f"{'YES' if r['sensitive'] else 'no':>5} {p2['peak_amplitude']:>10.4f}")
    if ns:
        print(f"\nSensitive elements detail:")
        print(f"{'El':>3} | "+" | ".join(f"m={m}dt(gate->pk)" for m in MARGINS))
        for r in results:
            if not r["sensitive"]: continue
            print(f"{r['element']:>3} | "+" | ".join(f"{r['peaks_by_margin'][m]['gate_start']:>4}->{r['peaks_by_margin'][m]['peak_sample']:>4}" for m in MARGINS))

if __name__ == "__main__":
    main()
