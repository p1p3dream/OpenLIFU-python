# Validation Scripts

Scratch scripts for GLADYS validation experiments, segmentation evaluation,
acoustic simulation, and debugging. Organized into subdirectories by purpose.

## batch/ - Batch runners

| Script | Description |
|--------|-------------|
| `batch_jwave_all.sh` | Run j-Wave simulations across all subjects |
| `batch_jwave_helmholtz.sh` | Run j-Wave Helmholtz solver across subjects |
| `batch_pipeline.sh` | Run the full GLADYS pipeline in batch |
| `batch_two_class_bone.sh` | Run two-class bone simulations for all subjects |
| `batch_two_class_bone_remaining.sh` | Continue two-class bone batch for remaining subjects |
| `launch_256element_stonkbot.sh` | Launch 256-element array simulations on stonkbot |
| `parallel_runner.py` | GPU-aware parallel runner with filesystem locking |
| `sweep_cortical_thickness.sh` | Sweep cortical thickness parameter for two-class bone sims |

## jwave/ - j-Wave simulations

| Script | Description |
|--------|-------------|
| `jwave_kwave_compare.py` | Compare j-Wave and k-Wave simulation results side by side |
| `jwave_poc.py` | Proof-of-concept j-Wave simulation on a simple geometry |

## kwave/ - k-Wave simulation runners

| Script | Description |
|--------|-------------|
| `run_gladys_256element.py` | Run GLADYS simulation with the 256-element hemispherical array |
| `run_gladys_nnunet.py` | Run GLADYS with nnU-Net segmentation (single-class bone) |
| `run_gladys_nnunet_kwavearray.py` | Run GLADYS using KWaveArray element definitions |
| `run_gladys_nnunet_maxangle.py` | Run GLADYS with max-angle skull-incidence apodization |
| `run_gladys_nnunet_subject.py` | Run GLADYS per-subject with expanded target probe and time gating |
| `run_gladys_nnunet_timegated.py` | Run GLADYS with time-gated focal metrics |
| `run_gladys_pipeline.py` | Run full pipeline on GU008: segmentation, materials, acoustic params |
| `run_gladys_simulation.py` | Run a standalone GLADYS k-Wave simulation |
| `run_nnunet_external_validation.py` | Run trained nnU-Net skull model on external datasets (IXI, SHARM) |
| `run_pseudoct_birnbaum.py` | Run PseudoCT segmentation on all Birnbaum subjects |
| `run_ucl_benchmark.py` | Run the UCL intercomparison benchmark simulation |

## analysis/ - Analysis and comparison

| Script | Description |
|--------|-------------|
| `analyze_bm7.py` | Analyze benchmark 7 simulation results |
| `analyze_gladys_pmax.py` | Analyze peak pressure (p_max) across GLADYS runs |
| `analyze_outlier_skulls.py` | Investigate subjects with anomalous skull attenuation |
| `analyze_skull_path.py` | Analyze per-element skull path lengths and attenuation |
| `bone_density_analysis.py` | Extract per-element MRI intensity (bone density proxy) along ray paths |
| `coherence_factor_cw_batch.py` | Compute coherence factor for ComplexWeighted across subjects in batch |
| `coherence_factor_gu008.py` | Compute per-element coherence factor for GU008 |
| `coherence_factor_subject.py` | Compute per-element coherence factor for an arbitrary subject |
| `compare_delay_methods.py` | Compare Direct, SimulationCorrected, and ComplexWeighted delays |
| `compare_first_arrival_vs_argmax.py` | Compare first-arrival vs argmax peak-picking for delay extraction |
| `compare_ucl_bm7.py` | Compare OpenLIFU results against UCL benchmark 7 reference |
| `comparison_3way.py` | Three-way comparison of delay/apodization methods |
| `complex_weighted_water_validation.py` | Validate ComplexWeighted delays and phases in homogeneous water |
| `dilated_path_analysis_gu008.py` | Compare skull path analysis with raw vs 1mm-dilated skull mask |
| `generate_results_7mm.py` | Generate VALIDATION_RESULTS_7mm.md from JSON result files |
| `head_to_head_ants.py` | Compare ThresholdMRI vs ANTs Atropos segmentation on BrainWeb data |
| `incidence_angle_amplitude_gu008.py` | Per-element amplitude vs skull incidence angle diagnostic |
| `metrics.py` | Shared focal quality metrics (attenuation, focal error, FWHM) |
| `narrowband_dft_metric_gu008.py` | Compare narrowband DFT vs time-domain focal metric on GU008 |
| `optimize_placement.py` | Find optimal array orientation via spherical search over skull rays |
| `per_element_transmission_scatter_gu008.py` | Per-element observed vs predicted transmission scatter on GU008 |
| `per_element_transmission_scatter_subject.py` | Per-element transmission scatter for an arbitrary subject |
| `skull_thickness_map.py` | Map skull thickness over full sphere and find optimal placement window |
| `spatial_search_probe.py` | Summarize spatial-search-probe sidecars for Birnbaum subjects |
| `validate_delay_equality.py` | Validate that SimulationCorrected delays match Direct in water |

## investigation/ - Debugging and one-off investigation

| Script | Description |
|--------|-------------|
| `alpha_bone_sweep_gu010.py` | Sweep skull absorption coefficient on GU010 to explain attenuation residual |
| `cfl_diag_alpha12_gu010_cfl003_only.py` | CFL diagnostic on GU010 with alpha=12 (CFL003 run only) |
| `cfl_diagnostic_alpha12_gu010.py` | Full CFL diagnostic on GU010 with alpha=12 |
| `diagnose_hilbert_gate.py` | Debug Hilbert envelope time gating for delay extraction |
| `diagnose_reverberation.py` | Investigate reverberation artifacts in reciprocal simulations |
| `gu010_investigation.py` | Deep-dive investigation of GU010 anomalous results |
| `gu010_segmentation_uncertainty.py` | Quantify segmentation uncertainty contribution for GU010 |
| `investigate_ernie_refinement.py` | Investigate Ernie dataset segmentation refinement (v1) |
| `investigate_ernie_refinement2.py` | Investigate Ernie dataset segmentation refinement (v2) |
| `investigate_ernie_refinement3.py` | Investigate Ernie dataset segmentation refinement (v3) |
| `investigate_ernie_refinement4.py` | Investigate Ernie dataset segmentation refinement (v4) |
| `investigate_ernie_refinement5.py` | Investigate Ernie dataset segmentation refinement (v5) |
| `ray_sanity_gu008.py` | Sanity check ray-casting geometry on GU008 skull |
| `skull_path_apod_alpha_sweep_gu010.py` | Sweep SkullPathApodization alpha on GU010 |

## segmentation/ - Segmentation validation

| Script | Description |
|--------|-------------|
| `eval_sharm_dice.py` | Evaluate Dice scores against SHARM ForkNet skull labels |
| `export_onnx.py` | Export trained nnU-Net model to ONNX format |
| `prepare_nnunet_dataset.py` | Prepare nnU-Net training dataset from Birnbaum scans |
| `remap_ernie_labels.py` | Remap SimNIBS Ernie tissue labels to OpenLIFU format |
| `split_skull_two_class.py` | Split single-class skull labels into cortical + trabecular bone |
| `validate_100_scans.py` | Run segmentation validation across 100 scans |
| `validate_brainweb.py` | Validate segmentation on BrainWeb synthetic data |
| `validate_colin27.py` | Validate segmentation on the Colin27 atlas |
| `validate_ernie.py` | Validate segmentation on the SimNIBS Ernie dataset |
| `validate_ibsr.py` | Validate segmentation on the IBSR dataset |
| `validate_mni152.py` | Validate segmentation on the MNI152 atlas |
| `validate_nnunet_birnbaum.py` | Validate nnU-Net skull model against Birnbaum ground truth |
| `validate_skull_model_gu008.py` | Validate skull_seg.onnx on GU008 and compare to fullhead Dice |
| `validate_two_class_bone.py` | Validate two-class bone modeling on Ernie Extended dataset |

## benchmarks/ - Performance benchmarks

| Script | Description |
|--------|-------------|
| `benchmark_15.py` | Benchmark simulation with 15 elements |
| `benchmark_before_after.py` | A/B benchmark comparing performance before and after a change |
| `benchmark_scaling.py` | Benchmark simulation scaling with grid size or element count |
| `benchmark_workers.py` | Benchmark parallel worker configurations |
| `profile_phases.py` | Profile individual pipeline phases (seg, materials, sim) |
| `profile_segment_internals.py` | Profile internal segmentation subroutines |
| `test_edt_optimizations.py` | Test EDT (Euclidean distance transform) optimization strategies |

## misc/ - Helpers and assets

| Script | Description |
|--------|-------------|
| `_gpu_flock.py` | GPU coordination via filesystem lock for serializing CUDA calls |
| `_probe_helpers.py` | Per-voxel water-calibrated gate helpers for time-gated probe analyses |
| `generate_256element_array.py` | Generate and validate a 256-element hemispherical transducer array |
| `hemi256_500khz.json` | 256-element array definition at 500 kHz |
| `hemi256_layout.png` | Visual layout of the 256-element array |
