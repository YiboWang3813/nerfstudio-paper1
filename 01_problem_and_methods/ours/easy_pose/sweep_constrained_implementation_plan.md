# Sweep-Constrained Pose-Free Ours Implementation Plan

Date: 2026-06-18
Branch: `dev/ours-pose-free-sweep-20260618`

## 1. Goal

The current Ours pipeline has two validated innovations:

- local-global ultrasound renderer;
- responsibility-based ultrasound density control.

This plan develops a third innovation around pose-free, or more precisely tracking-free, one-sweep ultrasound reconstruction. The target scenario is a freehand ultrasound sweep without an external optical or electromagnetic tracker attached to the probe.

The goal is not generic arbitrary pose recovery from unordered images. For one-sweep ultrasound, the data already contains strong acquisition structure:

- frames are temporally ordered;
- adjacent frames are close in probe motion;
- the dominant motion is a sweep progression through slice direction;
- rotation and in-plane drift should be smooth;
- pixel spacing and approximate slice spacing can be calibrated or estimated.

Therefore the proposed method converts pose-free reconstruction into a constrained sweep pose field estimation problem.

## 2. Core Idea

Represent each frame pose as a physically constrained sweep trajectory plus a small bounded residual:

```text
T_i = T_base(i; theta_sweep) * Exp(xi_i)
```

where:

- `T_i` is the probe-to-world pose of frame `i` in a canonical sweep coordinate system;
- `T_base(i; theta_sweep)` is a low-dimensional sweep trajectory defined by frame index and physical acquisition prior;
- `theta_sweep` contains global sweep parameters, such as slice step scale, sweep axis, initial orientation, and optional low-order trajectory curvature;
- `xi_i` is a small per-frame SE(3) residual;
- `Exp(.)` maps the residual twist to SE(3).

This design intentionally avoids letting GS photometric loss solve a full unconstrained pose estimation problem. Previous FIMA and joint pose experiments showed that GS reconstruction loss alone cannot reliably correct large pose errors, and unconstrained pose/Gaussian joint optimization has strong ambiguity. The new method limits pose freedom to the range that ultrasound GS can plausibly refine.

## 3. Paper Inspiration and What We Keep

The papers under `docs/literature/idea_sources/pose_free` mainly target RGB pose-free or COLMAP-free 3DGS. Their full pipelines are not directly portable to ultrasound, but several design patterns are useful.

### FreeSplatter and NoPoSplat

Useful:

- canonical reference frame instead of externally supplied camera coordinate frame;
- pose-free reconstruction can be formulated around a learned or optimized canonical space.

Not directly used:

- feed-forward large reconstruction models;
- PnP from predicted Gaussian/depth maps;
- large RGB pretraining assumptions.

For Ours, the first or middle ultrasound frame can define the canonical sweep coordinate system.

### NopeRoomGS and LongSplat

Useful:

- local-to-global optimization;
- short overlapping clips/windows;
- pose and scene parameters should not be fully optimized together at all times;
- alternating optimization reduces pose/geometry gradient interference.

For Ours, a sweep can be divided into overlapping local windows. Each window estimates a locally consistent trajectory, then overlapping frames constrain global stitching.

### PCR-GS

Useful:

- pose refinement should include cross-frame feature or structure consistency, not only rendering loss;
- frequency-aware optimization helps pose stability.

For ultrasound, DINO/RGB features are not reliable. Use ultrasound-compatible structure terms:

- low-frequency B-mode image consistency;
- wavelet low-band consistency;
- gradient or edge map consistency;
- local NCC or mutual information for adjacent frames/windows.

### Coarse-Pose GS Paper

Useful:

- optimize residual poses around a coarse pose;
- use bounded updates and feasible-region penalties;
- use different learning rates or preconditioning for pose and scene parameters.

For Ours, the canonical sweep trajectory is the coarse pose. Per-frame residuals must be bounded by mm/degree limits related to adjacent slice spacing.

### Robust Pose Estimation / Active GS

Useful:

- freeze or slow scene parameters during pose refinement;
- use visible Gaussian reprojection and robust losses instead of raw photometric loss alone.

For Ours, this motivates an alternating schedule:

- update GS with fixed pose field;
- update pose residuals with fixed or low-LR GS;
- repeat only if stable.

## 4. Proposed Method

### 4.1 Canonical Sweep Pose Field

Define a canonical coordinate frame:

- frame `i0` is the reference frame;
- image `u, v` lies in the probe image plane;
- sweep progression is initially along canonical `z`;
- frame `i` starts at `z = (i - i0) * delta_z`.

Initial pose:

```text
T_base(i) =
[
  R0,
  (i - i0) * delta_z * d + p0
]
```

where:

- `delta_z` is calibrated, estimated from metadata, or swept in a small set of candidate values;
- `d` is the sweep direction;
- `R0` is the canonical probe orientation;
- `p0` is fixed to remove gauge freedom.

Gauge handling:

- absolute world pose is not recoverable without a tracker;
- evaluation must align predicted trajectory to GT with Procrustes/Umeyama or first-frame alignment;
- the method should be described as tracking-free canonical sweep reconstruction.

### 4.2 Bounded Residual Pose

Each frame gets a residual twist:

```text
xi_i = [rho_i, phi_i]
```

where:

- `rho_i` is translation residual;
- `phi_i` is rotation residual.

Constraints:

```text
||rho_i|| <= tau_t
||phi_i|| <= tau_r
||xi_i - xi_{i-1}|| small
||xi_{i+1} - 2 xi_i + xi_{i-1}|| small
```

Default starting bounds should be conservative:

- translation residual bound: no larger than a fraction of adjacent slice spacing;
- rotation residual bound: small degrees, then tune only if Stage C is stable.

Use either hard clamping through parameterization or soft log-barrier penalties. Soft barriers are preferred first because they expose instability during experiments.

### 4.3 Ultrasound Structure Loss for Pose

Pose optimization should not rely on full-resolution speckle photometric loss only.

Candidate losses:

```text
L_render = L1/SSIM(render_i, image_i)
L_low = L1(lowpass(render_i), lowpass(image_i))
L_wavelet = L1(W_low(render_i), W_low(image_i))
L_grad = L1(grad_low(render_i), grad_low(image_i))
L_ncc = 1 - NCC(patch_low(render_i), patch_low(image_i))
```

Early experiments should start with `L_render + L_low + L_smooth + L_barrier`.

Do not introduce all structure losses at once. Add them only after the sweep pose field itself is verified.

### 4.4 Local-to-Global Window Optimization

Split the sweep into overlapping windows:

```text
window length: 5, 9, or 15 frames
overlap: 2 to 4 frames
```

Window stage:

- initialize each window from canonical sweep pose;
- optimize only local `theta_sweep_window` and small residuals;
- keep GS frozen or use a lightweight temporary local GS if needed.

Global stage:

- stitch window trajectories with overlap consistency;
- optimize global `theta_sweep` and residuals;
- train final Ours GS with the learned trajectory.

This stage should only be implemented after simpler global sweep residual experiments produce stable gains.

### 4.5 Alternating Optimization

Recommended schedule:

1. Warm up GS with fixed canonical sweep pose.
2. Freeze or strongly slow Gaussian parameters, optimize pose field with structure and barrier losses.
3. Freeze pose field, optimize GS with local-global renderer and responsibility DC.
4. Optionally repeat pose/GS blocks if pose residual stays bounded and validation improves.

The design goal is to avoid the known ambiguity where moving Gaussians and moving poses both explain the same image error.

## 5. Implementation Stages

### Stage A: Canonical Sweep Baseline

Purpose:

- establish the tracking-free lower bound;
- verify whether index-based pseudo-pose is usable at all.

Implementation:

- add a mode that ignores dataset GT extrinsics during training;
- generate pseudo poses from frame index, pixel spacing, and configurable `delta_z`;
- keep the existing renderer unchanged.

Experiments:

- first use one clean TUS-REC-2024 sweep;
- run with no pose optimization;
- compare against GT-pose Ours as upper bound.

Success signal:

- training does not collapse;
- rendered train frames are plausible;
- test frame quality is not completely broken after using interpolated pseudo poses.

Failure signal:

- canonical spacing is too wrong to reconstruct any coherent volume;
- metrics are far below a meaningful baseline and visual results are unusable.

### Stage B: Global Sweep Parameter Optimization

Purpose:

- test whether optimizing only low-dimensional global sweep parameters helps.

Trainable parameters:

- `delta_z` scale;
- optional sweep direction correction;
- optional global small orientation correction.

Keep per-frame residual disabled.

Success signal:

- improves over Stage A;
- optimized `delta_z` remains physically plausible;
- no unstable rotation/translation drift.

### Stage C: Bounded Per-Frame Residual

Purpose:

- test whether small residual pose refinement can close the gap to GT-pose Ours.

Implementation:

- add per-frame SE(3) residuals;
- add smoothness and barrier penalties;
- expose residual statistics in logs.

Metrics to log:

- max/mean translation residual;
- max/mean rotation residual;
- barrier violation count;
- adjacent relative translation error after GT alignment;
- reconstruction PSNR/SSIM/LPIPS.

Success signal:

- better reconstruction than Stage B;
- pose residuals stay below adjacent slice spacing scale;
- pose metrics improve after alignment.

Failure signal:

- pose residuals saturate at barrier;
- reconstruction improves only by unphysical pose distortion;
- pose metrics worsen while image metrics improve.

### Stage D: Low-Frequency Pose Loss

Purpose:

- reduce speckle noise influence on pose refinement.

Add one structure term at a time:

1. lowpass L1;
2. wavelet low-band;
3. low-frequency gradient.

Success signal:

- more stable pose residuals;
- better test frame metrics;
- less dependence on exact initialization.

### Stage E: Local-to-Global Windows

Purpose:

- handle long sweeps where global residual optimization is too underconstrained.

Implementation:

- overlapping windows;
- local pose refinement;
- global stitching through overlap constraints;
- final global Ours training.

Success signal:

- better long-sweep stability;
- lower trajectory drift;
- consistent improvements over Stage C/D.

### Stage F: Four-Dataset Validation

Only after Stage C or E has a clear positive result:

- synthetic_liver;
- spine_phantom;
- tus-rec-2024;
- tdsc_abus_2024.

Protocol:

- one representative sweep per dataset first;
- then expand if the signal is strong;
- compare fixed canonical, proposed pose field, and GT-pose upper bound.

## 6. Evaluation Protocol

### Pose Metrics

Because tracking-free reconstruction has gauge ambiguity, align predicted and GT trajectories before reporting:

- first-frame alignment for relative motion analysis;
- Procrustes/Umeyama alignment for global ATE;
- RPE translation;
- RPE rotation;
- adjacent frame displacement error.

Important threshold:

- if translation error is larger than adjacent slice spacing, the pose prior is probably not usable by GS.

### Reconstruction Metrics

Report:

- PSNR;
- SSIM;
- LPIPS if available;
- training time;
- renderer time if changed;
- final Gaussian count;
- visual slices.

Test pose handling:

- for pose-free protocol, do not render test frames with GT pose as the main metric;
- infer test frame pose from the learned sweep trajectory by frame index or local interpolation;
- GT-pose test can be reported only as a diagnostic upper-bound view.

### Baselines

Required:

- GT-pose Ours: upper bound;
- fixed canonical sweep: tracking-free lower baseline;
- global sweep parameter only;
- bounded residual full method.

Optional:

- unconstrained pose residual, expected to fail or overfit;
- photometric-only pose residual;
- low-frequency structure loss ablation.

## 7. Expected Risks

### Risk 1: Canonical Sweep Is Too Weak

If fixed index-based poses are too poor, Stage B/C may not recover.

Mitigation:

- sweep candidate `delta_z` values;
- use metadata spacing when available;
- start with TUS-REC-2024 because it is cleaner.

### Risk 2: Pose Residual Overfits Image Loss

Pose can deform to compensate Gaussian errors.

Mitigation:

- bounded residual;
- smoothness;
- alternating schedule;
- pose metrics after GT alignment;
- reject variants where image metrics improve but trajectory becomes unphysical.

### Risk 3: Speckle Misleads Pose Loss

Ultrasound speckle is not stable enough for raw photometric pose supervision.

Mitigation:

- low-frequency loss;
- wavelet low-band;
- local NCC/MI on smoothed images;
- avoid high-frequency pose loss early.

### Risk 4: Innovation Becomes Too Weak

If the method only fixes `delta_z`, reviewers may see it as a simple heuristic.

Mitigation:

- emphasize constrained pose field, not fixed spacing;
- include residual bounds, physical sweep prior, low-frequency structure loss, and local-to-global window consistency;
- show pose and reconstruction ablations.

## 8. Initial Development Tasks

1. Locate Ours dataset pose loading and renderer camera construction path.
2. Add pseudo-pose generation from frame index without changing GT-pose training.
3. Add config flags:
   - `pose_mode=gt|canonical_sweep|sweep_opt`;
   - `sweep_delta_mm`;
   - `sweep_ref_frame`;
   - `pose_residual_enabled`;
   - `pose_residual_bound_t_mm`;
   - `pose_residual_bound_r_deg`;
   - `pose_smooth_weight`;
   - `pose_barrier_weight`;
   - `pose_lowfreq_weight`.
4. Implement Stage A canonical sweep training.
5. Add pose trajectory dump:
   - predicted poses;
   - aligned pose metrics if GT exists;
   - residual statistics.
6. Run Stage A on one TUS-REC-2024 sweep.
7. Decide whether Stage B is viable before adding per-frame residual.

## 9. First Experiment Target

Dataset:

- TUS-REC-2024.

Reason:

- cleaner one-sweep setting;
- more suitable for testing whether tracking-free sweep prior can work.

Initial comparison:

```text
A0: GT-pose Ours upper bound
A1: fixed canonical sweep, no pose optimization
A2: fixed canonical sweep with delta_z grid search
```

Only if A1/A2 are non-degenerate should we implement Stage B/C.

## 10. Current Status

- 2026-06-18: Plan created on `dev/ours-pose-free-sweep-20260618`.
- 2026-06-18: Stage A implementation started.
  - Added `scene.loading.pose_mode=canonical_sweep` as a container-level pose override.
  - Canonical poses use frame index, `DatasetMeta.delta_elevational_mm`, configurable delta scale, and a signed canonical axis.
  - Default remains `pose_mode=file`, so existing GT-pose experiments are unchanged.
  - Added parallel submit script: `scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_a_tus_qc.sh`.

Stage A first run:

```text
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_a_tus_qc_20260618
dataset = tus_rec_2024_qc
sweep = 010_LH_Per_L_DtP_qc_0005_0370
iterations = 5000
num_init = 300000
renderer = final local-global renderer
dc = final responsibility DC, short schedule 1000/4000/1000
```

Parallel variants:

| Variant | Pose mode | Purpose | Job |
| --- | --- | --- | --- |
| `gt_pose_upper` | `file` | Short-run upper bound under the same renderer/DC/training budget. | `64440` |
| `canonical_s100` | `canonical_sweep`, delta scale `1.00` | Metadata-spacing tracking-free baseline. | `64441` |
| `canonical_s075` | `canonical_sweep`, delta scale `0.75` | Coarse spacing sensitivity. | `64442` |
| `canonical_s125` | `canonical_sweep`, delta scale `1.25` | Coarse spacing sensitivity. | `64443` |

Interpretation rule:

- If all canonical variants collapse far below the GT-pose upper bound, pure index-based pseudo-pose is not sufficient and Stage B must add optimized global sweep parameters before per-frame residuals.
- If one canonical scale gives a non-degenerate reconstruction, proceed to Stage B/C on top of that initialization.

Submission status:

- Submitted with `scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_a_tus_qc.sh`.
- Jobs `64440-64443` were submitted without dependencies and started concurrently on `gpu3`.

Stage A result:

| Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Gaussians | Wall min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gt_pose_upper` | 31.2398 | 0.2194 | 0.015642 | 0.83663 | 177,495 | 1.07 |
| `canonical_s075` | 31.5728 | 0.1920 | 0.014589 | 0.84404 | 159,951 | 1.13 |
| `canonical_s100` | 31.6178 | 0.2059 | 0.014730 | 0.84246 | 172,575 | 1.20 |
| `canonical_s125` | 31.5368 | 0.2252 | 0.015070 | 0.83822 | 182,195 | 1.22 |

Delta versus `gt_pose_upper`:

| Variant | PSNR | LPIPS | L1 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| `canonical_s075` | +0.3331 | -0.0274 | -0.001053 | +0.00742 |
| `canonical_s100` | +0.3780 | -0.0135 | -0.000912 | +0.00584 |
| `canonical_s125` | +0.2970 | +0.0058 | -0.000572 | +0.00160 |

Immediate interpretation:

- Stage A passes the non-degeneration check. A pure index-based canonical sweep is not only trainable; on this TUS-QC holdout split it is stronger than the short-run tracked-pose row.
- This does not mean canonical pose is universally better than GT. The short-run `gt_pose_upper` label is only the same-budget tracked-pose reference, not the final 30k upper bound.
- The likely signal is that the TUS tracked pose contains local jitter/noise, while a parallel canonical sweep acts as a strong physical denoising prior for one-sweep reconstruction.
- `canonical_s100` is the PSNR winner. `canonical_s075` has the best perceptual/structural metrics and is close to the measured first-step translation norm observed in the smoke check.

Next decision:

- Proceed to Stage B/C. The next useful experiment is not to abandon pose optimization, but to add a small, physically bounded pose field on top of the canonical sweep and check whether it improves over `canonical_s075/s100` without losing trajectory plausibility.
- Before full residual optimization, add pose-diagnostic export and aligned trajectory metrics, because image metrics alone can reward an unphysical pose field.

Stage B/C implementation update:

- Fix `OursPoseManager` diagnostics so canonical-sweep runs use the original file poses as GT, while the active scene poses are treated as the prior.
- Add gauge-aware trajectory diagnostics:
  - similarity-aligned center error;
  - relative pose error rotation/translation;
  - predicted and GT adjacent-step scale;
  - residual rotation/translation statistics.
- Add hard residual clamps:
  - `model.pose_optimization.residual_bound_trans_mm`;
  - `model.pose_optimization.residual_bound_rot_deg`.

Stage B/C short run:

```text
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_bc_rerun_tus_qc_20260618
dataset = tus_rec_2024_qc
sweep = 010_LH_Per_L_DtP_qc_0005_0370
iterations = 5000
num_init = 300000
renderer = final local-global renderer
dc = final responsibility DC, short schedule 1000/4000/1000
pose diagnostics = full 365-frame train trajectory
```

Parallel variants:

| Variant | Delta scale | Pose update | Purpose |
| --- | ---: | --- | --- |
| `s075_diag` | 0.75 | no trainable residual | Diagnostic rerun for the perceptual/structural Stage A winner. |
| `s100_diag` | 1.00 | no trainable residual | Diagnostic rerun for the PSNR Stage A winner. |
| `s075_trans010` | 0.75 | translation residual only, bound `0.10 mm` | Test whether small residual refinement helps a near-GT step prior. |
| `s100_trans010` | 1.00 | translation residual only, bound `0.10 mm` | Test whether small residual refinement helps metadata spacing. |

Pose residual settings:

```text
start_iter = 1000
end_iter = 4000
optimize_translation = true
optimize_rotation = false
lr_trans = 1e-4
lambda_prior_trans = 0.20
lambda_smooth_trans = 0.05
residual_bound_trans_mm = 0.10
```

Interpretation:

- If `trans010` improves image metrics and reduces RPE/center error without hitting the residual bound, Stage C is promising.
- If `trans010` improves image metrics but worsens RPE or saturates residual bounds, the residual is overfitting and Stage B should stay as a fixed/low-dimensional sweep parameter method.
- If `trans010` is neutral or negative, the third innovation can still be framed around tracking-free canonical sweep denoising, but the trainable residual branch should be treated as optional or moved to later work.

Execution note:

- First submission jobs `64444-64447` failed around 500 iterations before summary export.
- Cause: CUDA/NVRTC error from pose diagnostic linalg/reduction on 4090 with the current torch stack: `nvrtc: error: invalid value for --gpu-architecture (-arch)`.
- Fix: pose diagnostics are scalar logging only, so metrics now detach and run on CPU. This avoids changing training/render gradients.
- Rerun uses a fresh output root to avoid mixing partial failed scalars with valid summaries.

Stage B/C rerun result:

```text
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_bc_rerun_tus_qc_20260618
jobs = 64448-64451
status = finished
```

Image metrics:

| Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Gaussians | Wall min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s075_diag` | 31.5734 | 0.1923 | 0.014602 | 0.84400 | 160,142 | 1.22 |
| `s075_trans010` | 31.5753 | 0.1916 | 0.014597 | 0.84417 | 160,061 | 1.53 |
| `s100_diag` | 31.6373 | 0.2051 | 0.014721 | 0.84254 | 172,719 | 1.23 |
| `s100_trans010` | 31.6283 | 0.2058 | 0.014714 | 0.84259 | 172,524 | 1.65 |

Pose diagnostics at iteration 5000:

| Variant | Center aligned mean mm | Center aligned max mm | RPE trans mm | Pred step mm | GT step mm | Residual trans max mm | Residual trans mean mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s075_diag` | 5.2423 | 11.4362 | 0.3318 | 0.3030 | 0.5296 | 0.0000 | 0.0000 |
| `s075_trans010` | 5.2416 | 11.4312 | 0.3313 | 0.3029 | 0.5296 | 0.0164 | 0.0029 |
| `s100_diag` | 5.2423 | 11.4361 | 0.3164 | 0.4040 | 0.5296 | 0.0000 | 0.0000 |
| `s100_trans010` | 5.2420 | 11.4360 | 0.3160 | 0.4039 | 0.5296 | 0.0171 | 0.0026 |

Delta from diagnostic-only prior:

| Pair | PSNR | LPIPS | L1 | SSIM | Center mean mm | RPE trans mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s075_trans010 - s075_diag` | +0.0019 | -0.0007 | -0.000005 | +0.00017 | -0.0007 | -0.0005 |
| `s100_trans010 - s100_diag` | -0.0090 | +0.0007 | -0.000007 | +0.00005 | -0.0003 | -0.0004 |

Interpretation:

- The fixed canonical sweep prior remains the useful signal. It is image-stable and beats the same-budget tracked-pose reference from Stage A on this TUS-QC sweep.
- The bounded translation residual is numerically safe: it does not saturate the `0.10 mm` clamp and keeps residuals around `0.003 mm` mean / `0.017 mm` max.
- The residual branch is currently too weak to claim as the main innovation. It gives a tiny positive result for `s075`, a mixed result for `s100`, and only changes RPE by about `0.0004-0.0005 mm`.
- The measured GT adjacent-step mean is `0.5296 mm`, while the metadata prior is `0.4040 mm` and `s075` is `0.3030 mm`. The best image metrics do not directly correspond to the closest tracker trajectory, which supports a denoising/prior narrative but requires careful wording.

Next recommended validation:

- Do not spend more effort on per-frame residuals until the fixed canonical sweep has been validated at production scale.
- Run the canonical sweep prior as the primary tracking-free method under the same article-level setting as the renderer/DC work: `1M` initialization, `30k` iterations, standard responsibility DC, best-checkpoint evaluation.
- Use at least `tus_rec_2024_qc` first, then extend to one sweep each from `synthetic_liver`, `spine_phantom`, and `tdsc_abus_2023/2024` if the first full run remains positive.
- Include `file pose`, `canonical_s075`, and `canonical_s100`. Add a small scale-grid variant only if the full run shows sensitivity.

Stage D production-setting TUS-QC validation:

Purpose:

- Confirm whether the Stage A/B/C short-run canonical sweep signal survives the article-level training setting.
- Compare against tracked file poses under the same final renderer and responsibility DC protocol.
- Keep the first production validation narrow before spending GPU on all four datasets.

Planned setting:

```text
script = scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_d_tus_qc_30k.sh
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_d_tus_qc_30k_20260618
dataset = tus_rec_2024_qc
sweep = 010_LH_Per_L_DtP_qc_0005_0370
iterations = 30000
num_init = 1000000
renderer = final local-global renderer
dc = final responsibility DC, fixed protocol 1000/25000/2000
test protocol = holdout_interval, trim 10, 40 samples
```

Variants:

| Variant | Pose mode | Purpose |
| --- | --- | --- |
| `file_pose` | loaded tracked poses | Full-setting tracked-pose reference. |
| `canonical_s075` | canonical sweep, delta scale `0.75` | Stage A perceptual/structural winner. |
| `canonical_s100` | canonical sweep, delta scale `1.00` | Metadata-spacing canonical baseline and Stage A PSNR winner. |

Decision rule:

- If either canonical variant remains comparable to or better than `file_pose`, the third innovation should focus on fixed/low-dimensional tracking-free canonical sweep reconstruction.
- If both canonical variants collapse at 30k, the short-run signal was likely a regularization artifact and this pose-free direction should be downgraded.
- If canonical is viable but scale-sensitive, the next step should be a low-dimensional global sweep scale/axis calibration, not per-frame residual optimization.

Stage D result:

```text
jobs = 64454-64456
status = finished
err logs = empty
```

| Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Gaussians | Wall min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `file_pose` | 32.3123 | 0.1734 | 0.013715 | 0.85577 | 220,072 | 4.12 |
| `canonical_s075` | 32.3734 | 0.1775 | 0.013372 | 0.85642 | 156,151 | 4.10 |
| `canonical_s100` | 32.4687 | 0.1750 | 0.013388 | 0.85749 | 201,491 | 4.02 |

Delta versus `file_pose`:

| Variant | PSNR | LPIPS | L1 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| `canonical_s075` | +0.0611 | +0.0041 | -0.000344 | +0.00064 |
| `canonical_s100` | +0.1564 | +0.0016 | -0.000328 | +0.00171 |

Interpretation:

- The Stage A/B/C signal survives full `1M+30k` training on TUS-QC.
- `canonical_s100` is the strongest overall row by PSNR/SSIM and is close on L1; `file_pose` only keeps a small LPIPS advantage.
- This supports the main pose-free direction as fixed or low-dimensional canonical sweep reconstruction, not high-DOF per-frame residual optimization.
- Because the tracked-pose reference is not clearly superior, the narrative should not call file pose a strict upper bound for noisy handheld sweeps. It is better framed as externally tracked reference pose, while canonical sweep is a tracking-free denoising/regularizing prior.

Stage E next validation:

- Extend the same `file_pose / canonical_s075 / canonical_s100` comparison to one sweep each from `synthetic_liver`, `spine_phantom`, `tus_rec_2024_qc`, and `tdsc_abus_2023`.
- Keep identical renderer, DC, training budget, and dataset split settings to the renderer/DC article-level experiments.
- Submit in parallel without Slurm dependencies; if queue pressure appears, split by index ranges manually.

Stage E result:

```text
script = scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_e_four_dataset_30k.sh
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_e_four_dataset_30k_20260618
jobs = 64457-64468
status = finished
err logs = empty
submission note = cluster QOS limited active submitted jobs, so tasks were submitted in three waves without dependencies
```

Best-checkpoint metrics:

| Dataset | Sweep | Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Best Gaussians |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `synthetic_liver` | `l1` | `file_pose` | 32.1387 | 0.2660 | 0.017397 | 0.80162 | 94,338 |
| `synthetic_liver` | `l1` | `canonical_s075` | 22.9936 | 0.4632 | 0.053665 | 0.62299 | 66,435 |
| `synthetic_liver` | `l1` | `canonical_s100` | 22.9622 | 0.4548 | 0.053777 | 0.62040 | 88,170 |
| `spine_phantom` | `left1` | `file_pose` | 30.6422 | 0.2126 | 0.015076 | 0.73947 | 84,347 |
| `spine_phantom` | `left1` | `canonical_s075` | 29.0656 | 0.3690 | 0.017434 | 0.67551 | 30,444 |
| `spine_phantom` | `left1` | `canonical_s100` | 29.3289 | 0.3322 | 0.017227 | 0.67832 | 46,689 |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | `file_pose` | 32.3017 | 0.1742 | 0.013727 | 0.85583 | 220,239 |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | `canonical_s075` | 32.3744 | 0.1768 | 0.013373 | 0.85634 | 155,957 |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | `canonical_s100` | 32.4746 | 0.1742 | 0.013383 | 0.85758 | 202,323 |
| `tdsc_abus_2023` | `validation_100` | `file_pose` | 27.4489 | 0.3060 | 0.027519 | 0.77822 | 376,160 |
| `tdsc_abus_2023` | `validation_100` | `canonical_s075` | 27.0428 | 0.3061 | 0.028218 | 0.77036 | 317,366 |
| `tdsc_abus_2023` | `validation_100` | `canonical_s100` | 27.4627 | 0.3084 | 0.027527 | 0.77791 | 375,690 |

Best canonical delta versus `file_pose`:

| Dataset | Best canonical | PSNR | LPIPS | L1 | SSIM | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `synthetic_liver` | `canonical_s075` | -9.1451 | +0.1973 | +0.036268 | -0.17862 | Fails. Canonical straight-sweep pose is not compatible with this sweep geometry/split. |
| `spine_phantom` | `canonical_s100` | -1.3133 | +0.1196 | +0.002150 | -0.06114 | Fails or weak. Some structure remains, but tracked pose is clearly needed. |
| `tus_rec_2024_qc` | `canonical_s100` | +0.1729 | +0.0001 | -0.000344 | +0.00175 | Passes. Canonical sweep is better on PSNR/L1/SSIM and essentially tied on LPIPS. |
| `tdsc_abus_2023` | `canonical_s100` | +0.0138 | +0.0024 | +0.000008 | -0.00031 | Borderline. PSNR ties/slightly wins; other metrics are effectively tied or slightly worse. |

Stage E interpretation:

- The method should not be claimed as universal pose-free reconstruction.
- The viable claim is narrower and more defensible: for offline one-sweep ultrasound protocols where frames are acquired as a regular or near-regular ordered sweep, a canonical sweep pose prior can replace external tracking and can even denoise tracker jitter.
- `tus_rec_2024_qc` is the strongest support case because the canonical prior beats file poses under full `1M+30k` training.
- `tdsc_abus_2023` is a weak support/borderline case: canonical_s100 is approximately tied with file pose, which is still useful for a tracking-free setting if the narrative emphasizes comparable quality without tracker hardware.
- `synthetic_liver` and `spine_phantom` are negative controls. They show that the assumption breaks when sweep geometry is not close to the canonical parallel-slice model.
- The third innovation should therefore be written with an explicit validity condition and a geometry-aware fallback: use canonical sweep for regular one-sweep scans; use tracked poses or a low-dimensional sweep calibration when the scan contains stronger pose variation.

Next implementation direction:

- Add a production config for `canonical_s100` on regular sweeps.
- Keep `canonical_s075` as an ablation/scale sensitivity row, not the default.
- Do not revive high-DOF per-frame residual optimization as a main claim.
- If another experiment is needed, test a low-dimensional global calibration: optimize only global scale and small global tilt/axis parameters, not per-frame poses.

Stage F pose-geometry failure diagnosis:

```text
script = scripts/methods/ours/easy_pose/analyze_canonical_sweep_pose_geometry.py
report = docs/methods/ours/easy_pose/canonical_geometry_diagnosis.md
json = outputs/preproduction/ideas/ours/easy_pose/pose_free_geometry_diagnosis_20260618/metrics.json
```

Key diagnostics:

| Dataset | Stage E best canonical - file PSNR | Step CV | Median step / meta delta | Elevational fraction | Non-elev / elev | Linear residual mean / max | Main diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `synthetic_liver` | -9.1451 | 0.0003 | 0.9999 | 0.9726 | 0.1882 | 1.1425 / 4.2967 mm | Train trajectory is mostly regular, but external `test_relative_poses.npy` is sampled at about 5x train spacing. Current canonical test loading compresses the 40 test poses to one-step spacing. |
| `spine_phantom` | -1.3133 | 0.6361 | 0.9214 | 0.9452 | 0.3451 | 5.3491 / 13.2691 mm | Trajectory is line-like but frame spacing is highly non-uniform. Fixed-index canonical sweep erases real variable spacing, so tracked poses remain better. |
| `tus_rec_2024_qc` | +0.1729 | 0.4626 | 1.1924 | 0.8598 | 1.0571 | 5.2423 / 11.4361 mm | QC crop keeps mostly elevational one-sweep motion, but tracked poses contain local in-plane jitter. Canonical sweep acts as a useful denoising prior. |
| `tdsc_abus_2023` | +0.0138 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 / 0.0000 mm | Exported poses are already ideal parallel slices; canonical_s100 matches file poses as expected. |

Revised interpretation:

- `synthetic_liver` should not be used as evidence that canonical sweep fails on regular trajectories. It exposes a protocol bug/limitation: `interpolated_file` test poses carry a larger global frame spacing, while current canonical test construction only sees the compressed test sequence length.
- `spine_phantom` is the real negative case for the current fixed-index canonical model. Its pose centers are nearly collinear, but the scan speed/spacing is not uniform; therefore a uniform frame-index prior is too restrictive.
- `tus_rec_2024_qc` remains the main positive freehand case because canonical sweep suppresses tracker jitter rather than matching every tracked in-plane displacement.
- `tdsc_abus_2023` remains the ideal/generated parallel-slice sanity check.

Next step after diagnosis:

- Fix or extend canonical pose generation for `interpolated_file` protocols by preserving original/global frame spacing, then rerun `synthetic_liver` as a sanity check.
- For `spine_phantom`, test a low-dimensional non-uniform spacing model, e.g. monotonic cumulative spacing or piecewise-linear global spacing, before considering any per-frame 6-DoF pose residual.

Stage G protocol correction:

The Stage E `synthetic_liver` rows used the historical synthetic-liver benchmark protocol:

```text
scene=liver
scene.loading.test_mode=interpolated_file
```

This is inconsistent with the other Stage E datasets, which used `holdout_interval`. It is also not a fair protocol for evaluating `canonical_sweep`, because `interpolated_file` creates a separate 40-frame test container. For `synthetic_liver/l1`, the train sequence has median adjacent step `0.2685 mm`, while `test_relative_poses.npy` has median adjacent step `1.3423 mm`, about `5x` larger. The current canonical loader rebuilds canonical poses from the compressed 40-frame test index sequence, so it assigns one-step spacing to those test frames and compresses the test trajectory.

Therefore:

- Stage E `synthetic_liver` canonical rows should be marked invalid for the pose-free conclusion.
- A corrected synthetic-liver comparison must use the same `holdout_interval` protocol as the other datasets.
- The corrected run should compare `file_pose`, `canonical_s075`, and `canonical_s100` under `1M+30k`, final renderer, and responsibility DC.

Corrected experiment:

```text
script = scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_g_synthetic_liver_holdout_30k.sh
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_g_synthetic_liver_holdout_30k_20260618
dataset = synthetic_liver
sweep = l1
test_mode = holdout_interval
holdout_trim_each = 10
holdout_num_samples = 40
```

Stage G result:

```text
jobs = 64471-64473
status = finished
err logs = empty
```

| Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Best Gaussians | Best step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `file_pose` | 29.6384 | 0.3124 | 0.023207 | 0.67739 | 100,877 | 30000 |
| `canonical_s075` | 29.8144 | 0.3351 | 0.022833 | 0.68103 | 69,233 | 25000 |
| `canonical_s100` | 29.6595 | 0.3223 | 0.023123 | 0.67753 | 93,607 | 30000 |

Corrected delta versus `file_pose`:

| Variant | PSNR | LPIPS | L1 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| `canonical_s075` | +0.1760 | +0.0227 | -0.000374 | +0.00364 |
| `canonical_s100` | +0.0211 | +0.0098 | -0.000083 | +0.00014 |

Corrected interpretation:

- The original Stage E `synthetic_liver` negative result was invalid for the pose-free conclusion because it mixed `interpolated_file` synthetic testing with `holdout_interval` testing on the other datasets.
- Under the same holdout protocol, `synthetic_liver` becomes a positive case for canonical sweep on PSNR/L1/SSIM, although LPIPS is worse.
- The current dataset-level conclusion becomes:
  - positive: `synthetic_liver` under corrected holdout, `tus_rec_2024_qc`;
  - near tie: `tdsc_abus_2023`;
  - true negative/limitation: `spine_phantom`.
- The remaining failure mechanism to study is therefore mainly non-uniform spacing in `spine_phantom`, not synthetic_liver.

## Stage H: Non-Uniform Canonical Spacing for Spine Phantom

Motivation:

- After the Stage G protocol correction, the remaining hard failure is `spine_phantom`.
- Geometry diagnosis shows that `spine_phantom` is close to a one-dimensional sweep path, but adjacent frame spacing is highly non-uniform.
- The current `canonical_sweep` model assumes a fixed offset `t_i = (i - i_ref) * delta`, so it can only model constant-speed sweeps.
- This is too restrictive for manual freehand ultrasound. If the method only works for uniform spacing, the pose-free claim is easy to challenge.

Implementation:

- Keep the canonical sweep pose family: identity rotations plus translation along a signed canonical axis.
- Add `scene.loading.canonical_sweep_spacing_mode`:
  - `uniform`: current behavior, fixed `frame_index * delta`.
  - `file_step_norm`: oracle diagnostic, cumulative adjacent file-pose translation norm.
  - `file_axis_projection`: oracle diagnostic, cumulative adjacent file-pose translation projected onto the canonical axis.
- The two file-based modes are not final pose-free production modes. They are causal diagnostics: if they recover `spine_phantom`, the failure is primarily non-uniform spacing rather than the whole canonical-sweep idea.
- If Stage H succeeds, the production direction is a low-dimensional tracking-free spacing prior, e.g. monotonic/piecewise cumulative spacing calibrated by acquisition metadata or weak image consistency, not per-frame 6-DoF pose optimization.

Experiment:

```text
script = scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_h_spine_nonuniform_30k.sh
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_h_spine_nonuniform_30k_20260618
dataset = spine_phantom
sweep = left1
test_mode = holdout_interval
setting = 1M init + 30k + local-global renderer + responsibility DC
```

Variants:

| Variant | pose_mode | spacing_mode | Purpose |
| --- | --- | --- | --- |
| `file_pose` | `file` | n/a | Tracked-pose reference. |
| `canonical_uniform_s100` | `canonical_sweep` | `uniform` | Current failed canonical baseline. |
| `canonical_stepnorm_s100` | `canonical_sweep` | `file_step_norm` | Tests whether non-uniform scalar spacing fixes the failure. |
| `canonical_axisproj_s100` | `canonical_sweep` | `file_axis_projection` | Tests whether signed elevational spacing is enough. |

Expected interpretation:

- If `file_step_norm` or `file_axis_projection` closes most of the gap to `file_pose`, the third innovation should become a non-uniform canonical sweep prior.
- If both remain far below `file_pose`, the failure is not just spacing; it likely requires rotation/in-plane motion or dataset-specific imaging differences, and the pose-free claim should stay limited to regular sweeps.

Stage H result:

```text
jobs = 64476, 64480, 64481, 64482
status = finished
err logs = empty
note = jobs 64477-64479 were early failed submissions before the Hydra scene YAML field was added; ignore them.
```

| Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Best step | Best Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `file_pose` | 30.6441 | 0.2138 | 0.015079 | 0.73903 | 5000 | 84,485 |
| `canonical_uniform_s100` | 29.3541 | 0.3327 | 0.017230 | 0.67855 | 15000 | 46,494 |
| `canonical_stepnorm_s100` | 28.8183 | 0.3424 | 0.017951 | 0.66127 | 15000 | 53,151 |
| `canonical_axisproj_s100` | 28.8153 | 0.3405 | 0.017941 | 0.66164 | 15000 | 53,367 |

Delta versus `file_pose`:

| Variant | PSNR | LPIPS | L1 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| `canonical_uniform_s100` | -1.2900 | +0.1189 | +0.002152 | -0.06049 |
| `canonical_stepnorm_s100` | -1.8258 | +0.1286 | +0.002872 | -0.07776 |
| `canonical_axisproj_s100` | -1.8288 | +0.1267 | +0.002863 | -0.07739 |

Stage H interpretation:

- The `spine_phantom` failure is not solved by replacing uniform spacing with oracle adjacent-step spacing.
- In fact, the current uniform canonical sweep remains the best canonical variant on this sweep.
- Therefore the failure is not merely "scan speed is non-uniform"; it likely also involves rotation, in-plane/lateral motion, or dataset-specific image/pose coupling that cannot be represented by identity-rotation straight-line canonical poses.
- The result narrows the viable pose-free claim:
  - strong for regular or near-regular one-sweep data (`tus_rec_2024_qc`, corrected `synthetic_liver`);
  - acceptable/tied for generated parallel slices (`tdsc_abus_2023`);
  - not sufficient for `spine_phantom` without a stronger low-dimensional pose model.
- A production pose-free extension should not simply consume file-pose step lengths. If we continue, the next meaningful model is not "non-uniform spacing only", but a low-dimensional sweep calibration with global tilt/axis plus monotonic spacing, still much lower-DoF than per-frame 6-DoF pose optimization.

## Stage I: Spine Phantom Sweep-QC Instead of Complex Calibration

Decision:

- Do not pursue complex calibration as the main line.
- Standardize `spine_phantom` into a regular-sweep QC subset, analogous to the production choice made for `tus_rec_2024_qc`.
- This keeps the third innovation focused on a tracking-free canonical sweep prior for offline one-sweep reconstruction, rather than turning it into a separate pose-estimation/calibration method.

QC window analysis:

```text
script = scripts/methods/ours/easy_pose/analyze_spine_phantom_qc_windows.py
report = docs/datasets/spine_phantom/qc_window_analysis.md
```

Selected windows:

| Window | Original frame range | Frames | Rationale |
| --- | --- | ---: | --- |
| `qc80_0039_0119` | `[39, 119)` | 80 | Best length-80 window; better regularity than the full sweep while retaining enough train frames. |
| `qc100_0026_0126` | `[26, 126)` | 100 | Longer QC candidate; weaker regularity than `qc80`, but more training coverage. |

Experiment:

```text
script = scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_i_spine_qc_windows_30k.sh
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_i_spine_qc_windows_30k_20260618
dataset = spine_phantom
sweep = left1
test_mode = holdout_interval
holdout_trim_each = 5
holdout_num_samples = 20
setting = 1M init + 30k + local-global renderer + responsibility DC
```

Variants:

| Window | Variant | Purpose |
| --- | --- | --- |
| `qc80_0039_0119` | `file_pose` | Tracked reference on the QC subset. |
| `qc80_0039_0119` | `canonical_s100` | Tests whether QC makes tracking-free canonical sweep viable. |
| `qc100_0026_0126` | `file_pose` | Tracked reference on the longer QC subset. |
| `qc100_0026_0126` | `canonical_s100` | Tests the coverage/regularity tradeoff. |

Expected interpretation:

- If canonical closes the file-pose gap after QC, the pose-free contribution should include a sweep-quality condition: canonical reconstruction is valid for near-regular one-sweep segments, and QC is part of the tracking-free protocol.
- If canonical still fails after QC, `spine_phantom` should remain a limitation/negative-control dataset rather than forcing a high-DoF calibration method into the paper.

Stage I result:

```text
jobs = 64485-64488
status = finished
err logs = empty
```

| Window | Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Best step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qc80_0039_0119` | `file_pose` | 29.0250 | 0.2017 | 0.017922 | 0.70930 | 5000 |
| `qc80_0039_0119` | `canonical_s100` | 27.7217 | 0.3292 | 0.020244 | 0.64796 | 30000 |
| `qc100_0026_0126` | `file_pose` | 29.5023 | 0.2273 | 0.017300 | 0.70457 | 15000 |
| `qc100_0026_0126` | `canonical_s100` | 28.1205 | 0.3654 | 0.019518 | 0.65355 | 30000 |

Stage I interpretation:

- Simple QC windowing does not close the gap on `spine_phantom`.
- `qc100` is the better window among the two because both file-pose and canonical scores are higher than `qc80`.
- However, the canonical model is still far below file pose:
  - `qc80`: `-1.3033 dB` PSNR, LPIPS `+0.1275`, SSIM `-0.06133`;
  - `qc100`: `-1.3818 dB` PSNR, LPIPS `+0.1382`, SSIM `-0.05101`.
- One remaining non-complex-QC explanation is spacing scale: the selected windows have median adjacent steps larger than the metadata delta (`qc80`: about `1.098/0.79=1.39`; `qc100`: about `0.998/0.79=1.26`), while Stage I only tested scale `1.00`.

## Stage J: Spine QC Canonical Spacing-Scale Sweep

Motivation:

- Stay within choice 1: standardize the sweep/QC protocol, not complex calibration.
- Test whether the Stage I failure is simply because canonical_s100 compresses the QC windows along the sweep axis.

Experiment:

```text
script = scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_j_spine_qc_scale_sweep_30k.sh
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_j_spine_qc_scale_sweep_30k_20260618
setting = same as Stage I
```

Stage J result:

```text
jobs = 64489-64492
status = finished
err logs = empty
```

| Window | Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Best step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qc80_0039_0119` | `canonical_s125` | 27.7142 | 0.3095 | 0.020357 | 0.64571 | 30000 |
| `qc80_0039_0119` | `canonical_s140` | 27.6642 | 0.2825 | 0.020662 | 0.64118 | 15000 |
| `qc100_0026_0126` | `canonical_s125` | 28.2031 | 0.3442 | 0.019453 | 0.65316 | 30000 |
| `qc100_0026_0126` | `canonical_s140` | 28.2098 | 0.3259 | 0.019564 | 0.65257 | 25000 |

Stage J interpretation:

- Spacing-scale correction also does not solve `spine_phantom`.
- On `qc80`, larger spacing does not improve PSNR over `canonical_s100`.
- On `qc100`, spacing scale gives a small improvement (`28.1205 -> 28.2098` dB), but the gap to `file_pose` remains large (`29.5023 - 28.2098 = 1.2925` dB).
- Therefore the spine failure is not explained by full-sweep non-uniformity, QC-window irregularity, or a wrong scalar slice spacing alone.
- The most likely missing factors are rotations/in-plane motion and/or dataset-specific image-pose coupling. Solving this would require a more complex low-dimensional or per-frame calibration method, which we intentionally do not pursue in the current pose-free innovation line.

Current decision:

- Keep `spine_phantom` as a limitation/negative-control dataset for pose-free canonical sweep.
- Do not continue into complex calibration for this paper branch.
- Write the pose-free contribution as valid for regular or near-regular one-sweep protocols with a QC condition, supported by corrected `synthetic_liver`, `tus_rec_2024_qc`, and the generated-parallel `tdsc_abus_2023` sanity case.

## Stage K: Spine Arc-Length Uniform Real-Frame Subset

Question from the failure analysis: can `spine_phantom` become a usable validation case if we do not alter any image/pose labels, but select real frames whose tracked pose centers are closer to a uniform sweep in arc length?

Implementation:

- Added `scene.loading.frame_indices_path` as an optional loader-level subset file.
- The file stores original frame ids; images and poses are indexed together, so no raw pose or image label is modified.
- Added `scripts/methods/ours/easy_pose/select_arclength_uniform_frames.py`.
- Generated:
  - `scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform100.json`
  - `scripts/methods/ours/easy_pose/slurm/spine_phantom_frame_splits/left1/arclength_uniform80.json`

Subset geometry:

| subset | selected frames | original step CV | selected step mean | selected step median | selected step CV | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| arc100 | 100 | 0.6361 | 1.2014 mm | 1.1436 mm | 0.3355 | Matches Ultra-NeRF's 100-frame regularized-test idea more closely. |
| arc80 | 80 | 0.6361 | 1.5055 mm | 1.5123 mm | 0.3035 | More regular spacing but fewer training frames after holdout. |

Experiment:

```text
script = scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_k_spine_arclength_subset_30k.sh
output_root = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_k_spine_arclength_subset_30k_20260618
common = 1M init, 30k, final local-global renderer, responsibility DC, holdout_interval, trim=5, holdout=20
```

Rows:

| subset | variant | pose setting |
| --- | --- | --- |
| arc100 | file_pose | tracked file pose upper/reference row |
| arc100 | canonical_calibrated | canonical_sweep with `delta_mm=1.2013612499640334` |
| arc80 | file_pose | tracked file pose upper/reference row |
| arc80 | canonical_calibrated | canonical_sweep with `delta_mm=1.5055033385625227` |

Interpretation rule:

- If canonical gap closes strongly, spine failure was mainly a data-regularity/protocol issue.
- If canonical remains far below file pose, spine is not a clean support dataset for this pose-free prior even after legal frame regularization.

Stage K result:

```text
jobs = 64495-64498
status = finished
err logs = empty
summary = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_k_spine_arclength_subset_30k_20260618/summary.tsv
```

| Subset | Variant | Best PSNR | Best LPIPS | Best L1 | Best SSIM | Best step | Final PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arc100 | file_pose | 29.3174 | 0.2200 | 0.017303 | 0.70466 | 5000 | 29.1453 |
| arc100 | canonical_calibrated | 27.9277 | 0.3270 | 0.019922 | 0.63604 | 30000 | 27.9277 |
| arc80 | file_pose | 28.6217 | 0.2110 | 0.018734 | 0.68552 | 5000 | 28.3269 |
| arc80 | canonical_calibrated | 26.9522 | 0.2734 | 0.022217 | 0.60317 | 25000 | 26.9401 |

Gaps to file-pose reference:

| Subset | Delta PSNR | Delta LPIPS | Delta L1 | Delta SSIM |
| --- | ---: | ---: | ---: | ---: |
| arc100 canonical - file | -1.3897 dB | +0.1070 | +0.002619 | -0.06862 |
| arc80 canonical - file | -1.6695 dB | +0.0625 | +0.003483 | -0.08236 |

Stage K interpretation:

- Arc-length nearest-frame selection improves the pose spacing statistics but does not make `spine_phantom` compatible with the canonical pose-free prior.
- The `arc100` subset is slightly better than prior QC windows for canonical PSNR (`27.93 dB` vs `27.72-28.21 dB` depending on QC/scale), but the tracked-pose reference remains much higher.
- The failure is therefore not just an artifact of selecting uneven frame indices. The remaining mismatch likely involves rotations/in-plane motion, physical image-plane calibration, and the original UltraNeRF multi-sweep/robot-tracked acquisition design.
- Keep `spine_phantom` out of the positive pose-free evidence set. It can remain useful for renderer/DC validation and as a limitation case for canonical tracking-free reconstruction.
