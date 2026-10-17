# P1 paper-final metric protocol (2026-09-08)

Status: case/frame audit passed; no training, renderer, adaptation or metric implementation changes.
Production source: P0.5 freeze_v2, SHA-256 `b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.

## Cases and frames

The authoritative inclusion list is `paper_final_dataset_protocol.tsv`: Liver 6, Spine 9,
Arm 33 QC segments, Breast 30 volumes. Main comparison uses measured file poses (MP)
for all methods. PPP is evaluated separately, with the identical retained frame IDs.
All seven baselines and legacy Ours have successful final-30k results for all 78 cases.
`baseline_case_frame_audit.json` records every resolved-config hash and actual test-frame
index sequence: all 624 runs agree with the frozen per-case loading, metric config and
holdout indices. No method-specific missing-case exclusion is permitted.

Use all frames of each processed case, no online resize/subsample. Holdout uses
`test_mode=holdout_interval`, `holdout_num_samples=0`, `holdout_test_ratio=0.2`.
Let N be case frames, t=8 for Spine and 10 otherwise. Test count is
`max(1,min(N-2*t-1,round(0.2*N)))`; indices are unique rounded linspace points from
t through N-1-t. All other frames are training frames, including boundary frames;
the trim is a restriction on test eligibility, not removal from training.
Seed 42; 30,000 training iterations; Ours starts from 1,000,000 Gaussians.

## Quality aggregation

Use `test/psnr`, `test/l1`, `test/ssim`, `test/lpips`, `test/dists`, `test/gmsd`
at step **30000**, mapped to `final_psnr`, `final_l1`, `final_ssim`, `final_lpips`,
`final_dists`, `final_gmsd`. Never use best checkpoint fields or an earlier final step.
These are arithmetic means of per-frame metrics within a sweep; dataset metrics
are arithmetic means of sweep metrics, **not frame-weighted**, not PSNR of pooled MSE.
For multi-sweep PPP report paired per-sweep values, dataset mean and sample std (ddof=1).
QC segments are the Arm aggregation units, not independent subjects; no subject-level
inference or significance claims from this single seed.
Use the frozen `utils/metric` implementation unchanged (PSNR max_val=1.0, six metrics).
Save `eval/per_frame_metrics_iteration_030000.tsv`; validate finite values, step and IDs.

Baselines come from the ordered source tables in `legacy_main_selected_rows.json`:
GS fixed subset, GS full sweep, accepted UltraGauss Arm chi2=5.5 replacement, implicit
full sweep. Later source has precedence, never choose max metric over duplicates.
This reproduces the existing published numerical table. Legacy Ours is comparison-only.
Missing/failed/nonfinite corrected cases block a final dataset aggregate; report explicit
incompleteness and retry the same config, never silently average available cases.
Best/second-best are ranked on unrounded means, higher for PSNR/SSIM, lower otherwise.

## Population, time and FPS

`final_num_gaussians` is the final checkpoint population; verify against last
`train/num_gaussians`. Use the existing scalar/DC logs for population trajectories;
no extra training. A zero-prune or capacity-increasing control is not forced to match
the final population of Full CGA.

Report end-to-end run wall seconds separately from any iteration-only timing.
Historical `train_wall_time_sec` measures the training Python subprocess, including
setup, evaluation and checkpoint writes, excluding queueing/postprocessing. Preserve
that measurement boundary for new runs. Anchor wrapper wall seconds exclude imports
before its timer; flag this small timing-boundary difference rather than implying exact
speedup. Dataset time = sweep arithmetic mean; a combined two-dataset table time =
equal-weight mean of dataset means, not pooled sweep mean. Report concurrency/hardware.
Historical shared-worker timings are not controlled speed comparisons with exclusive GPUs.

Run the frozen `benchmark_render_fps.py` on retained corrected final checkpoints,
one benchmark worker per GPU with no simultaneous training on that GPU. Defaults:
5 warmup frames, up to 20 held-out frames, 3 timed passes, pure forward CUDA timing,
excluding setup, disk I/O, metrics and backward. Per-run FPS = rendered frames / seconds.
Save FPS for all corrected main cases. For comparative paper FPS use the **same two
cases per dataset for every method** as
`render_fps_complete_cached_ours_per_run.tsv`; aggregate arithmetic mean of per-case
FPS, then equal-weight dataset means for combined tables. Do not compare all-case Ours
FPS to two-case baseline FPS. Baseline FPS is reused as historical evidence, with its
hardware/build provenance retained; no baseline training is authorized.

## Qualitative and claims

Freeze existing TMI-v2 selections before corrected training: Liver l3/item27,
Spine left3_2/item17, Arm 010_RH_Per_S_PtD_qc_0165_0364_s02/item18,
Breast validation_116/item39. Keep existing crop/display ranges; require identical GT
hashes against baseline panels. These are inherited illustrative selections (not random
samples); do not re-select cases to maximize corrected advantage. All saved frames remain
available for inspection. Generate comparison reports and proposed Ours table rows only;
do not update Results claims until the Batch 1 scientific checkpoint is reviewed.

## Batch boundary

Batch 1 only: corrected Main, Joint/PPP/LGR/CGA structural, existing frozen 16-case
PPP multi-sweep subset (6 Liver, 5 Arm, 5 Breast), representative seven-point rho.
The 16-case subset is explicitly not all 69 Liver/Arm/Breast main cases. It retains the
pre-existing validation selection, before corrected results, consistent with the approved
approximately 14 extra runs. No multi-sweep rho curve, LGR grid or CGA grid is submitted.
