# Phase correction approaches for transcranial FUS

Status: design doc, no code yet
Owner: gladys pipeline
Target: close the ~19 dB phase-decoherence gap measured on GU008

## 1. Problem statement

On GU008 the best current delay method, `SimulationCorrected` (reciprocal
k-wave + envelope-peak arrival times, see
`src/openlifu/bf/delay_methods/simulation_corrected.py`), leaves a residual
focal attenuation dominated by phase decoherence at the common focal time.
Measured values:

- Corrected coherence factor: **CF = 0.107**
- Equivalent pressure loss: **-19.4 dB** relative to a coherent ideal sum
- Amplitude-weighted per-element phase standard deviation at f0: **~63 deg**

A scalar per-element delay (what `_run_reciprocal_simulation` currently
produces) can only compensate first-order path-length differences. The
heterogeneous skull introduces effects that scalar delays cannot represent:

1. **Subwavelength path-length variability** that no single scalar delay
   can fix. At 500 kHz the water wavelength is ~3 mm; the skull thickness
   varies by ~1-5 mm across the aperture, and sound speed varies from
   ~1500 m/s (marrow) to ~2900 m/s (cortical bone).
2. **Frequency-dependent phase shifts** (dispersion in bone: phase velocity
   depends on frequency, and attenuation is roughly linear in frequency).
3. **Amplitude distortion** that is not just phase: transmission coefficient
   varies element-to-element by ~10-20 dB depending on incidence angle and
   local thickness.
4. **Multi-path**: the signal arriving at the target from a given element
   is a sum of direct + reflected + refracted components, each with its
   own phase and amplitude. Envelope-peak picking collapses this to a
   single arrival time and throws the rest away.

The arrival-time picker in `_run_reciprocal_simulation` (Hilbert envelope
peak, gated by earliest-plausible arrival via `sound_speed_max`) is a
reasonable first-order estimator, but the 63 deg phase scatter shows it is
not enough. Any method that outputs only one scalar delay per element
inherits this ceiling.

## 2. Approaches table

| # | Approach | Impl complexity | Expected dB recovery | Runtime compute | Inputs | `SimulationCorrected` compat |
|---|----------|-----------------|----------------------|-----------------|--------|------------------------------|
| 1 | Full time-reversal (broadband) | High: new waveform-per-channel API, per-element time-series storage, transmit hardware that plays arbitrary waveforms | 15-19 dB (gold standard) | 1 reciprocal sim (same as today); storage Nt * Nelem samples per plan | MRI-derived skull model (same as today); no CT | Breaks API: `calc_delays` returns one scalar per element. Needs a new `calc_waveforms` return type |
| 2 | Frequency-domain phase correction (narrowband) | Low: ~80 LoC, reuse `sensor_data`, FFT + extract arg at f0 | 8-12 dB | Trivial (1 FFT per element) | Same as today | Drop-in: returns delays via phase/omega; or extended return with phase-only |
| 3 | Iterative delay tuning (gradient ascent on focal pressure) | Medium: ~300 LoC, needs forward-sim wrapper + optimizer loop | 6-10 dB (improves delays but still scalar; bounded by same ceiling) | N_iter * forward sims (~10-50 k-wave runs; hours on GPU) | Same as today, plus a forward-sim harness | Drop-in: still returns scalar delays |
| 4 | Complex per-element weighting at f0 (narrowband, amp+phase) | Low-medium: ~150 LoC, extract complex amplitude at f0 from `sensor_data`, new return type `(delays, apod_weights)` or `complex_weights` | 10-14 dB | Trivial (1 FFT per element) | Same as today | Extension: new `ComplexWeighted` delay method; downstream beamformer must accept complex weights |
| 5 | Model-based CT aberration correction (ray trace) | High: ~600 LoC, requires CT pipeline, skull segmentation, ray tracer, phase integrator | 8-14 dB (literature: 70-90% of full TR) | Fast at runtime (ray trace in seconds) | **CT required**, not just MRI | Replace `SimulationCorrected` with a CT-driven method; heavy preprocessing |

Rough dB-recovery estimates are anchored to the literature cited in the
deep-dives below and to the observed 19.4 dB decoherence ceiling. These
are first-principles guesses, not guarantees.

## 3. Deep dive on each approach

### 3.1 Full time-reversal (Fink, 1992; Thomas and Fink, 1996; Aubry et al., 2003)

The classical approach: the reciprocal simulation already records a full
pressure time series at each element (`sensor_data[:, col]` in
`_run_reciprocal_simulation`). Full time-reversal fires each element with
a time-reversed copy of its own recorded signal, amplitude-scaled so the
total transmit power matches the transducer budget. This handles
multi-path, dispersion, and amplitude variation in one shot because it
replays the exact physics the virtual target source would have produced.

Implementation sketch: keep the `sensor_data` array; define a new
`WaveformMethod.calc_waveforms(arr, target, params, transform)` that
returns shape `(n_elements, n_samples)`; downstream, the transmit
controller must accept arbitrary per-channel waveforms. For a DIY 64-element
array driven by square-wave pulsers this is not available; it requires
per-channel arbitrary waveform generators (AWGs). Estimated effort:
~8 developer-days for the Python side, plus hardware work that is
out of scope for this repo.

### 3.2 Frequency-domain phase correction (Clement and Hynynen, 2000; Hynynen and Jolesz, 1998)

Fourier-transform each element's recorded signal, extract the complex
value at f0 = 500 kHz, invert its phase, and fire a narrowband burst with
that phase at unit amplitude. This is the standard workhorse in MR-guided
FUS (ExAblate, for instance, does a version of this). It is limited to
narrowband correction and ignores amplitude variation, but it is cheap
and eliminates the envelope-peak picking error that likely dominates the
current 63 deg phase scatter.

Implementation sketch: in `_run_reciprocal_simulation`, after building
`sensor_data`, compute `X_i = sum_t sensor_data[t, col_i] * exp(-1j*2*pi*f0*t*dt)`
(effectively a single DFT bin). Define `phase_i = -arg(X_i)`. Convert
back to delay via `delay_i = phase_i / (2*pi*f0)` wrapped appropriately,
or return `phase_i` directly. Estimated effort: ~2 developer-days
including tests against a known analytical aberrator.

### 3.3 Iterative delay tuning (Herbert et al., 2009; Larrat et al., 2010)

Start with the scalar geometric delay, run a forward simulation, measure
focal pressure, perturb one element's delay by `+/- dt`, keep if focal
pressure rises, iterate. Variants use coordinate descent, simulated
annealing, or gradient estimates from MR-ARFI displacements in the
experimental setting.

Implementation sketch: wrap the existing forward-sim pipeline in a
`FocalPressureObjective`, use `scipy.optimize` with a finite-difference
gradient, cap iterations at ~50. At 64 elements and one forward sim per
two function evals, this is ~50-100 k-wave runs per target, which at
~30 s/sim on GPU is 25-50 minutes per plan. Worst-case pathology: the
objective is noisy in regions where multi-path dominates. Estimated
effort: ~6 developer-days, most of it on the forward-sim harness and
convergence diagnostics.

The fundamental limit is that this method still outputs a single scalar
delay per element, so it cannot beat the same ceiling that bounds
approach #2. In practice it reaches that ceiling more reliably than
envelope-peak picking because it optimizes the quantity we actually care
about.

### 3.4 Complex per-element weighting at f0 (Tanter et al., 2007; Marquet et al., 2009)

Generalize the "delay" to a complex weight `w_i = |w_i| * exp(j*phi_i)`
applied at f0. `phi_i` comes from the single-bin DFT as in approach #2;
`|w_i|` comes from the magnitude of that same DFT bin (or from a
per-element transmission estimate). The element fires a narrowband
burst with amplitude `|w_i|` and phase `phi_i`.

This is half of full time-reversal: narrowband-only, but captures both
the amplitude and phase of the Green's function at f0. Hardware needs:
per-channel narrowband bursts with programmable amplitude and phase,
which most clinical arrays support and which a well-designed 64-channel
DIY array can support with 8-bit phase DACs per channel.

Implementation sketch: new `ComplexWeighted` delay method under
`src/openlifu/bf/delay_methods/`, returning a tuple `(delays, apod)` or
a `complex_weights` array. The beamformer in
`src/openlifu/bf/` needs a small extension to apply apodization on
transmit; currently transmit apodization is implicitly unity. Estimated
effort: ~5 developer-days including beamformer plumbing and a
sanity-check forward-sim test.

### 3.5 Model-based CT aberration correction (Clement and Hynynen, 2002; Aubry et al., 2003; Marsac et al., 2017)

Given a CT skull model (Hounsfield units per voxel, mapped to density
and sound speed), ray-trace each element's central ray through the
skull, integrate phase along the ray
(`phi_i = 2*pi*f0 * integral(1/c(s) ds)`), and compute amplitude from
Fresnel transmission at each bone interface plus attenuation along the
path. This is predictive rather than measurement-based: no k-wave
simulation needed at runtime.

Implementation sketch: a CT preprocessing pipeline (HU -> density ->
sound speed via Aubry 2003 calibration), a segmented skull mesh, a ray
tracer through the voxel grid, and a per-element phase integrator. Needs
CT data we do not have for most of the validation set. Estimated effort:
~12 developer-days. The gladys/main branch is MRI-only today, so this
is a major scope expansion.

## 4. Recommendation

**Prototype approach #4: complex per-element weighting at f0.**

Ranked by effort/reward:

- Approach #2 (phase-only narrowband) is cheaper but leaves amplitude
  weighting on the table, so its dB ceiling is visibly lower than #4.
- Approach #4 re-uses the exact `sensor_data` array that
  `_run_reciprocal_simulation` already computes. No new simulations.
  The only meaningful new code is a single-bin DFT, a new delay-method
  class, and a small beamformer extension to apply transmit apodization.
- Approach #3 is a respectable fallback if #4 underperforms, but it
  cannot beat the scalar-delay ceiling and costs 50-100x more compute.
- Approach #1 needs transmit hardware we do not have on the DIY 64-element
  array.
- Approach #5 needs CT data we do not have for most subjects.

Expected dB recovery for approach #4: **10-14 dB**, taking the CF from
0.107 toward the 0.35-0.55 range. This is a best guess, bracketed by
Marquet 2009 (narrowband complex weighting reached ~80% of time-reversal
optimum in skull experiments) and by the physics: a 63 deg phase scatter
collapsing to a few degrees should recover most but not all of the
phase-coherence loss, and per-element amplitude weighting additionally
suppresses off-focus sidelobes.

## 5. Implementation plan (approach #4)

1. **Single-bin DFT extractor** (1 day). Add a helper in
   `src/openlifu/bf/delay_methods/` that takes `sensor_data`, `dt`, and
   `f0`, and returns complex amplitudes per element. Unit-test against a
   synthetic aberrator with known phase and amplitude. Verify against a
   brute-force `numpy.fft.rfft` at the nearest bin.
2. **New `ComplexWeighted` delay method** (1 day). Subclass `DelayMethod`,
   return `(delays, apodization)` where `delays = -phase / (2*pi*f0)` and
   `apodization = |X_i| / max_j |X_j|`. Factor shared k-wave setup out of
   `SimulationCorrected` into a helper so we do not duplicate the
   source-mask / sensor-mask / target-idx logic.
3. **Beamformer plumbing for transmit apodization** (1 day). The current
   transmit path assumes unit amplitude per element. Add a per-element
   scaling hook and make sure downstream forward-sims see it. Add a unit
   test showing that `ComplexWeighted` with uniform `|X_i|` reduces to
   `SimulationCorrected` up to envelope-peak vs phase picking noise.
4. **Validation on GU008** (1 day). Run the full gladys pipeline with the
   new method, recompute corrected CF and focal pressure, and compare
   against the 0.107 baseline. Log per-element amplitude and phase
   histograms. Update `THRESHOLD_MRI_VALIDATION_MARCH_2026.md` with
   results.
5. **Sweep across validation set and write-up** (1 day). Run on the rest
   of the ~/Data/openlifu-validation/ subjects, tabulate CF before and
   after, and report mean dB recovery. If CF improves by >6 dB mean we
   ship it; if <3 dB, pivot to approach #3 (iterative delay tuning) or
   investigate why.

## 6. Open questions and risks

- **Envelope-peak gating interaction.** The single-bin DFT is computed
  over the full `sensor_data` window, which includes the source-pulse
  ring-up and any post-focal reverberation. We may need to window to the
  steady-state portion of the received signal before the DFT, or use a
  matched filter at f0. Risk that naive windowing inherits the same
  arrival-time picking problem the current method has.
- **Element-to-voxel aliasing.** `_run_reciprocal_simulation` already
  warns that multiple elements can land in the same voxel for coarse
  grids. Complex weighting makes this worse because amplitude is now
  load-bearing. Need to confirm grid resolution is fine enough at 500 kHz
  for all elements to resolve into distinct voxels, or add a sub-voxel
  interpolation step.
- **Transmit-hardware fidelity.** A DIY 64-element array with 8-bit phase
  DACs has ~1.4 deg quantization at f0, which is well inside budget, but
  amplitude control is often coarser (3-4 bits of apodization). If the
  transmit chain cannot realize the computed `|w_i|`, the simulated
  recovery will not match the measured recovery. Need a quantization
  model and a sensitivity study.
- **Narrowband assumption.** We correct at f0 only. If the skull's phase
  response is strongly dispersive across the transmit bandwidth (say
  more than ~30 deg across the 10-20% bandwidth of a 3-cycle burst), we
  will see bandwidth-averaging loss that approach #4 does not address.
  At 500 kHz in cortical bone this is probably small but needs to be
  checked; fall back is full time-reversal (approach #1).
- **CF vs focal-pressure disagreement.** CF is a good proxy for phase
  coherence but not a perfect proxy for focal pressure in the presence
  of amplitude weighting; with nonuniform `|w_i|` the Cauchy-Schwarz
  relationship between CF and focal pressure shifts. Need to report
  both metrics and not just CF.
- **Reference for dB recovery estimate.** The 10-14 dB guess is
  literature-anchored but GU008-specific physics (skull thickness
  distribution, incidence angle spread) may make us over- or under-shoot.
  Milestone 5 above is how we find out.

## References

- Aubry, J.-F., Tanter, M., Pernot, M., Thomas, J.-L., Fink, M. (2003).
  Experimental demonstration of noninvasive transskull adaptive focusing
  based on prior computed tomography scans. JASA 113(1).
- Clement, G. T., Hynynen, K. (2000). A noninvasive method for focusing
  ultrasound through the human skull. Phys. Med. Biol. 47.
- Clement, G. T., Hynynen, K. (2002). Correlation of ultrasound phase with
  physical skull properties. Ultrasound Med. Biol. 28.
- Fink, M. (1992). Time reversal of ultrasonic fields. IEEE UFFC 39.
- Herbert, E., Pernot, M., Montaldo, G., Fink, M., Tanter, M. (2009).
  Energy-based adaptive focusing of waves: application to noninvasive
  aberration correction of ultrasonic wavefields. JASA 126.
- Hertzberg, Y., Volovick, A., Zur, Y., Medan, Y., Vitek, S., Navon, G.
  (2010). Ultrasound focusing using magnetic resonance acoustic radiation
  force imaging: application to ultrasound transcranial therapy. Med.
  Phys. 37.
- Hynynen, K., Jolesz, F. A. (1998). Demonstration of potential
  noninvasive ultrasound brain therapy through an intact skull. UMB 24.
- Larrat, B., Pernot, M., Aubry, J.-F., Dervishi, E., Sinkus, R., Seilhean,
  D., Marie, Y., Tanter, M., Fink, M. (2010). MR-guided transcranial brain
  HIFU in small animal models. Phys. Med. Biol. 55.
- Marquet, F., Pernot, M., Aubry, J.-F., Montaldo, G., Marsac, L., Tanter,
  M., Fink, M. (2009). Non-invasive transcranial ultrasound therapy based
  on a 3D CT scan: protocol validation and in vitro results. Phys. Med.
  Biol. 54.
- Marsac, L., Chauvet, D., La Greca, R., Boch, A.-L., Chaumoitre, K.,
  Tanter, M., Aubry, J.-F. (2017). Ex vivo optimisation of a heterogeneous
  speed of sound model of the human skull for non-invasive transcranial
  focused ultrasound at 1 MHz. Int. J. Hyperthermia 33.
- Tanter, M., Pernot, M., Aubry, J.-F., Montaldo, G., Marquet, F., Fink, M.
  (2007). Compensating for bone interfaces and respiratory motion in HIFU.
  Int. J. Hyperthermia 23.
- Thomas, J.-L., Fink, M. (1996). Ultrasonic beam focusing through tissue
  inhomogeneities with a time reversal mirror: application to transskull
  therapy. IEEE UFFC 43.
