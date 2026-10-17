# Ours + FiMA joint feasibility: Milestone C/D smoke and 1k validation

Date: 2026-06-15

Branch: `ours-fima-joint-feasibility`

## Goal

Continue the Ours + FiMA pose-free validation after Milestone A/B:

- Milestone C: run a real two-stage baseline, `FiMA predicted pose + Ours fixed pose`.
- Milestone D: test whether Ours reconstruction loss can repair the FiMA pose prior through existing per-frame delta optimization.

Dataset:

- `scene=tus_rec_2024_qc`
- sweep: `010_LH_Per_L_DtP_qc_0005_0370`
- absolute pose protocol

GPU:

- Slurm node `gpu3` RTX 4090

## Implementation

Added runner:

- `scripts/methods/ours/joint_pose_estimate_and_reconstruction/slurm/slurm_ours_fima_joint_cd_runner.sh`

The runner performs:

1. FiMA adjacent-pair PEM training with protocol-safe holdout.
2. Absolute-anchored FiMA pose export.
3. FiMA prior evaluation against absolute GT poses.
4. Ours GT-pose upper-bound run.
5. Ours fixed FiMA-prior run.
6. Ours trainable-delta run.

Important runner details:

- Uses `preproduction/baseline_reproduction/fima/paper_edges` but explicitly overrides:
  - `scene=tus_rec_2024_qc`
  - `scene.dataset_name=tus_rec_2024`
  - `model.sequence.respect_scene_holdout=true`
  - `model.pem.input_mode=adjacent_pair`
- Uses `preproduction/ideas/ours/pose_uncertainty_prior/pose_prior` for Ours, because this template already declares `model.pose_optimization`.
- Supports skipping GT/fixed baselines and reusing an existing prior for extra delta variants.

## Configuration smoke

Jobs:

- `63358`: failed, because the first runner version used `preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_validation_2k`, whose raw Hydra model block did not contain `pose_optimization`.
- `63359`: completed after switching Ours to `preproduction/ideas/ours/pose_uncertainty_prior/pose_prior`.

Smoke settings:

- FiMA: `20` iterations
- Ours: `20` iterations
- init Gaussians: `20,000`
- holdout samples: `8`

Purpose:

- Validate train/export/fixed/delta pipeline wiring, not method quality.

Outcome:

- FiMA export succeeded.
- Ours GT/fixed/delta all ran to completion.
- Summary TSV was written.

## 1k experiment

Job:

- `63360`
- State: `COMPLETED`
- Node: `gpu3`
- Elapsed: `00:01:53`

Settings:

- FiMA: `1000` iterations
- Ours: `1000` iterations
- init Gaussians: `100,000`
- holdout samples: `20`
- Ours density control: start `250`, end `800`, freq `250`

FiMA prior quality:

| metric | value |
|---|---:|
| abs trans mean | 17.6018 mm |
| abs trans p95 | 33.0459 mm |
| abs rot mean | 13.9623 deg |
| abs rot p95 | 25.4072 deg |
| step trans absdiff mean | 0.1923 mm |
| step rot absdiff mean | 0.0368 deg |

Interpretation:

- Local adjacent motion prediction is not terrible at 1k.
- Global chained pose still has large drift, especially rotation.

Ours reconstruction results:

| ablation | PSNR | LPIPS | L1 | SSIM | pose rot err | pose trans err |
|---|---:|---:|---:|---:|---:|---:|
| GT pose + Ours | 17.7916 | 0.4656 | 0.07703 | 0.4960 | - | - |
| FiMA prior + Ours fixed | 17.4167 | 0.4698 | 0.07998 | 0.4649 | 13.9060 deg | 17.5353 mm |
| FiMA prior + Ours trans-only delta | 17.4119 | 0.4695 | 0.07997 | 0.4654 | 13.9060 deg | 17.5363 mm |

Interpretation:

- Milestone C is feasible: FiMA predicted poses can initialize Ours and produce held-out render metrics.
- The 1k FiMA prior trails the GT-pose upper bound, as expected.
- The trans-only per-frame delta did not repair pose error.
- Rendering metrics are effectively unchanged relative to fixed prior.

## Extra D variant: full-SE3 pose-only delta

Job:

- `63361`
- State: `COMPLETED`
- Node: `gpu3`

Settings:

- Reused the 1k FiMA prior from job `63360`.
- Ran only the delta Ours variant.
- Rotation and translation both trainable.
- Pose-only optimization from iter `500` to `900`.
- `lr_rot=1e-4`, `lr_trans=5e-4`.
- `lambda_prior_rot=1e-4`, `lambda_prior_trans=1e-4`.

Result:

| ablation | PSNR | LPIPS | L1 | SSIM | pose rot err | pose trans err |
|---|---:|---:|---:|---:|---:|---:|
| FiMA prior + Ours full-SE3 pose-only delta | 16.7496 | 0.4727 | 0.08791 | 0.3864 | 13.9340 deg | 17.5376 mm |

Interpretation:

- Full-SE3 pose-only delta worsened held-out rendering.
- Pose error did not improve.
- This suggests the current weak-joint per-frame delta path is not yet a positive result for the pose-free narrative.

## User-observed issue: FiMA pose scale is too large for adjacent TUS-REC slices

After reviewing `scene/meta/dataset_meta/tus_rec_2024.py` and the actual sweep poses, the adjacent-frame motion scale is very small:

| GT adjacent motion statistic | value |
|---|---:|
| step translation mean | 0.5296 mm |
| step translation median | 0.4818 mm |
| step translation p95 | 0.9441 mm |
| step rotation mean | 0.0882 deg |
| step rotation p95 | 0.1766 deg |

This makes the previous FiMA prior error unacceptable for reconstruction:

- FiMA 1k absolute translation mean error: `17.60 mm`
- FiMA 1k absolute rotation mean error: `13.96 deg`

The local step error is smaller:

- step translation absdiff mean: `0.1923 mm`
- step rotation absdiff mean: `0.0368 deg`

But even small biased adjacent-frame errors accumulate over 365 frames. A free SE(3) adjacent regressor is therefore a poor fit for this near-linear, sub-millimetre sweep unless it is strongly constrained.

## Diagnostic: remove FiMA-predicted adjacent rotations

A post-hoc diagnostic prior was built from the 1k FiMA chain:

- keep FiMA-predicted adjacent translations;
- replace every adjacent rotation with identity;
- keep the first absolute pose anchor.

Prior quality:

| prior | abs trans mean | abs trans p95 | abs rot mean | abs rot p95 |
|---|---:|---:|---:|---:|
| FiMA 1k SE(3) | 17.6018 mm | 33.0459 mm | 13.9623 deg | 25.4072 deg |
| FiMA translation + zero rotation | 16.2139 mm | 23.5525 mm | 5.3672 deg | 10.7854 deg |

Ours fixed-prior result:

| ablation | PSNR | LPIPS | L1 | SSIM | pose rot err | pose trans err |
|---|---:|---:|---:|---:|---:|---:|
| FiMA prior + Ours fixed | 17.4167 | 0.4698 | 0.07998 | 0.4649 | 13.9060 deg | 17.5353 mm |
| FiMA translation + zero rotation + Ours fixed | 17.6938 | 0.4692 | 0.07637 | 0.5075 | 5.3559 deg | 16.0748 mm |

This is the strongest signal so far:

- Removing FiMA's predicted rotations substantially improves pose error.
- Ours reconstruction improves strongly, nearly matching the 1k GT-pose smoke on PSNR and exceeding it on L1/SSIM.
- The remaining weakness is LPIPS and residual translation drift.

Interpretation:

The pose-free line should not use unconstrained FiMA SE(3) edge prediction as the pose prior. The better direction is a physics-constrained near-linear sweep model:

```text
P_{i+1} = P_i @ T_physical_step @ T_small_residual
```

where:

- `T_physical_step` is a metadata / sweep-statistics prior, e.g. mainly elevational translation around `0.4-0.5 mm`;
- FiMA predicts only a small residual, preferably translation-first;
- rotations should be fixed to zero initially or heavily regularized / low-pass constrained.

## Follow-up: 5k FiMA with automatic zero-rotation postprocess

Implementation changes:

- Added `scripts/methods/baselines/fima/postprocess_pose_prior.py`.
- Added runner support for `PRIOR_POSTPROCESS_MODE=zero_rotation`.
- When this mode is enabled, FiMA export writes a raw pose chain first, then the postprocessor:
  - keeps each predicted local adjacent translation;
  - replaces each adjacent rotation with identity;
  - chains from the exported first-pose anchor.

Job:

- `63364`
- State: `COMPLETED`
- Node: `gpu3`

Settings:

- FiMA: `5000` iterations
- Postprocess: `zero_rotation`
- Ours: fixed prior, `1000` iterations
- init Gaussians: `100,000`
- holdout samples: `20`

Local adjacent translation check:

| source | local edge mean xyz |
|---|---:|
| GT | `[0.160, -0.134, 0.456] mm` |
| FiMA 5k raw | `[0.135, -0.118, 0.454] mm` |

The 5k FiMA local translation has become almost constant and is close to the GT mean step. This suggests the remaining issue is not local translation scale, but accumulated absolute-frame drift and rotation handling.

Prior quality:

| prior | abs trans mean | abs trans p95 | abs rot mean | abs rot p95 | step trans absdiff mean |
|---|---:|---:|---:|---:|---:|
| FiMA 1k SE(3) | 17.6018 mm | 33.0459 mm | 13.9623 deg | 25.4072 deg | 0.1923 mm |
| FiMA 1k translation + zero rotation | 16.2139 mm | 23.5525 mm | 5.3672 deg | 10.7854 deg | 0.1923 mm |
| FiMA 5k translation + zero rotation | 13.3611 mm | 19.5764 mm | 5.3672 deg | 10.7854 deg | 0.1918 mm |

Ours fixed-prior result:

| ablation | PSNR | LPIPS | L1 | SSIM | pose rot err | pose trans err |
|---|---:|---:|---:|---:|---:|---:|
| GT pose + Ours | 17.7916 | 0.4656 | 0.07703 | 0.4960 | - | - |
| FiMA 1k translation + zero rotation + Ours fixed | 17.6938 | 0.4692 | 0.07637 | 0.5075 | 5.3559 deg | 16.0748 mm |
| FiMA 5k translation + zero rotation + Ours fixed | 17.5065 | 0.4679 | 0.07848 | 0.4888 | 5.3559 deg | 13.2487 mm |

Interpretation:

- Longer FiMA training improves absolute translation error from `16.07 mm` to `13.25 mm`.
- Held-out reconstruction does not improve with the lower translation error in this short 1k Ours setting.
- This reinforces that the useful signal is the constrained local motion prior, not unconstrained SE(3) prediction.

## Controlled noisy-GT stress test for Ours pose delta

Purpose:

Before claiming a weak-joint pose-repair mechanism, test whether the current Ours per-frame delta can repair a known, small synthetic pose error.

Implementation changes:

- Added runner support for skipping FiMA with `RUN_FIMA_STAGE=false`.
- Added runner support for `RUN_NOISY_GT_FIXED` and `RUN_NOISY_GT_DELTA`.
- These branches use the GT train poses as the base prior and apply `OursPoseManager` synthetic local residual noise.

Jobs:

- `63366`: 0.5 mm translation-only noisy GT, fixed and delta
- `63367`: same noisy GT, high-LR delta stress test

Common settings:

- Synthetic pose noise: `noise_rot_deg=0.0`, `noise_trans_mm=0.5`
- Ours: `1000` iterations
- init Gaussians: `100,000`
- holdout samples: `20`

Delta settings:

| variant | pose-only window | lr_trans | lambda_prior_trans |
|---|---:|---:|---:|
| standard | `500-900` | `3e-4` | `0.001` |
| high-LR stress | `500-900` | `1e-2` | `0.0` |

Results:

| ablation | PSNR | LPIPS | L1 | SSIM | pose trans err | prior trans err |
|---|---:|---:|---:|---:|---:|---:|
| Noisy GT fixed, 0.5 mm | 17.7579 | 0.4711 | 0.07710 | 0.4931 | 0.5000 mm | 0.5000 mm |
| Noisy GT delta, standard | 16.8801 | 0.4739 | 0.08653 | 0.4028 | 0.4988 mm | 0.5000 mm |
| Noisy GT delta, high-LR | 16.8769 | 0.4745 | 0.08655 | 0.4027 | 0.4934 mm | 0.5000 mm |

Interpretation:

- A fixed `0.5 mm` translation-only pose perturbation is tolerable for this short Ours reconstruction.
- The current per-frame delta path barely changes the pose even when the true synthetic error is small and controlled.
- Pose-only delta hurts held-out reconstruction because Gaussian updates pause while pose correction remains ineffective.
- Increasing `lr_trans` by more than 30x still does not recover the synthetic pose perturbation.

This is a strong negative control for Milestone D. The existing GS reconstruction loss should not be used as the main evidence for pose correction. For the pose-free narrative, the next viable route is a better constrained FiMA/physics prior, not the current per-frame Ours delta.

## Current conclusion

Milestone C is validated as a runnable two-stage baseline.

Milestone D is currently negative:

- reconstruction loss did not meaningfully improve the FiMA pose prior;
- trans-only delta was neutral;
- full-SE3 pose-only delta was worse;
- pose metrics stayed essentially at prior quality.
- controlled noisy-GT tests show the current delta path also cannot repair a known `0.5 mm` translation perturbation.

This does not invalidate the framework-level idea, but it blocks any claim that Ours reconstruction already provides useful pose correction through the existing `OursPoseManager` delta path.

## Recommended next step

Do not move to full online FiMA-Ours joint training yet.

First improve the two-stage pose prior and the weak-joint baseline:

1. Replace free SE(3) FiMA edge prediction with a near-linear physical prior plus small residual.
2. Add a translation-only or rotation-clamped FiMA export path.
3. Train FiMA longer, e.g. `5k` or `10k`, but only after the output parameterization is constrained.
4. Treat the current per-frame Ours delta path as negative for pose correction unless its gradient/parameterization is redesigned.
5. Add direct pose-delta diagnostics before any further weak-joint claim:
   - delta norm over time,
   - pose gradient norm,
   - per-frame current-vs-prior pose error.
6. If continuing the pose-free narrative, prioritize constrained pose prior design:
   - metadata step + FiMA residual translation;
   - rotation clamped or low-pass bounded;
   - sequence-level smoothness / drift penalties at export or training time.
7. Only after weak-joint repair works should Milestone E full network-joint training be attempted.
