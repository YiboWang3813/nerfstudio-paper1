# FiMA Pose Protocol Stage 1

## Goal

Check whether the current GS `holdout_interval` protocol is the main reason FiMA produces poor pose priors, before adding uncertainty ideas on top of FiMA poses.

## Code Changes

- Added FiMA-only `model.sequence.split_protocol`.
  - `auto`: old behavior, follow Scene `holdout_interval` when available.
  - `block_holdout`: group validation frames into contiguous blocks and leave longer train-contiguous regions.
- Added Hydra defaults for the new FiMA sequence fields and previously added uncertainty fields.
- Exposed FiMA split protocol knobs in `scripts/methods/ours/joint_pose_estimate_and_reconstruction/slurm/slurm_ours_fima_joint_cd_runner.sh`.
- Fixed FiMA chain-eval anchor in `methods/fima/callbacks.py`; training logs should use the first absolute pose when the scene uses absolute poses.

## Split Sanity

Primary sweep: `010_LH_Per_L_DtP_qc_0005_0370`, `N=365`, 20 validation frames.

| protocol | validation pattern | train edges kept | phase-A W8 windows kept |
| --- | --- | ---: | ---: |
| scene interval holdout | 20 isolated frames | 324 / 364 | 50 / 90 |
| block holdout, 4 blocks x 5 frames | 4 contiguous blocks | 340 / 364 | 78 / 90 |

This confirms the protocol change does what it is supposed to do: it reduces fragmentation of FiMA training windows.

## 3k FiMA-Only Runs

Output root: `outputs/preproduction/ideas/ours/pose_uncertainty_prior/fima_pose_protocol_20260615`

| job | protocol | prior | abs trans mean mm | abs trans p95 mm | abs rot mean deg | abs rot p95 deg | step trans absdiff mean mm | step rot absdiff mean deg |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `63429` | scene interval | raw FiMA | 16.7124 | 26.7488 | 12.5284 | 23.6034 | 0.1926 | 0.0361 |
| `63432` | block holdout | raw FiMA | 19.1079 | 33.9207 | 17.9738 | 32.7392 | 0.1922 | 0.0392 |
| postprocess | scene interval | zero rotation | 11.5916 | 17.6035 | 5.3672 | 10.7854 | 0.1926 | 0.0884 |
| postprocess | block holdout | zero rotation | 15.0621 | 22.5666 | 5.3672 | 10.7854 | 0.1922 | 0.0884 |

GT adjacent-step p95 on this sweep:

- translation: `0.9441 mm`
- rotation: `0.1779 deg`

## Interpretation

1. The FiMA split protocol was hurting continuity, but it is not the dominant failure mode in this setup. `block_holdout` preserves many more W8 train windows, yet the raw chained prior is worse than the current scene interval split.
2. Both raw protocols learn a similar adjacent translation step error, about `0.192 mm`, but chained absolute pose error remains large. This points to accumulated bias/drift rather than only sparse supervision.
3. Rotation remains the clearest unstable component. Removing local predicted rotations again improves the prior substantially, matching the earlier physical-prior finding.
4. Even the best short-run prior here, scene interval + zero rotation, is still not "good enough" as a pose prior for a pose-free or uncertainty-weighted story: `11.59 mm` mean absolute translation error is far above the adjacent-frame motion scale.

## Decision

Do not merge uncertainty ideas into raw FiMA pose priors yet. The next step should make the pose prior physically constrained first:

1. Use zero/small-rotation FiMA edge composition by default.
2. Add a translation bias correction or physical residual correction, evaluated directly by pose metrics before GS.
3. Only after pose error drops to a sweep-scale acceptable range, test FiMA-pose-fed GS and uncertainty weighting.

## Translation Postprocess Probe

After the protocol comparison, the best raw prior (`scene interval`, job `63429`) was postprocessed without retraining.

| postprocess | base source | abs trans mean mm | abs trans p95 mm | note |
| --- | --- | ---: | ---: | --- |
| zero rotation | n/a | 11.5916 | 17.6035 | best among tested deployable variants |
| physical step | input median edge | 11.6099 | 17.6065 | slightly worse than zero rotation |
| physical residual, clamp 0.05-0.25mm | input median edge | 11.5916 | 17.6035 | effectively identical to zero rotation |
| physical step | GT median edge | 14.7129 | 22.0667 | diagnostic upper-bound only; worse |
| physical residual | GT median edge | 11.5916 | 17.6035 | identical to zero rotation |

This rules out a simple global-step-vector fix for the current FiMA prior. The remaining translation error is mainly accumulated directional/bias drift in the predicted local translations, not just wrong global step magnitude.

## Updated Next Step

The next useful FiMA-pose experiment should be a supervised pose-prior correction stage, not another holdout split:

1. Fit a small affine/bias correction on FiMA local translation edges using train-safe adjacent edges only, then evaluate on held-out edges/blocks.
2. Keep rotation zero or strongly clamped during chain composition.
3. If corrected edge error drops substantially, export that prior into Ours fixed-pose GS. If it does not, stop treating FiMA as a pose prior and keep the physical trajectory prior separate from FiMA.

## Translation Calibration Probe

Implemented `scripts/methods/baselines/fima/calibrate_translation_edges.py` to fit a translation-edge correction using only train-safe adjacent edges, then evaluate corrected zero-rotation chains.

| source prior | split used for calibration | fit | abs trans mean mm | abs trans p95 mm | held-out edge trans mean mm | held-out edge trans p95 mm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| scene interval raw | interval | diagonal scale+bias | 7.6555 | 12.1895 | 0.2195 | 0.4029 |
| scene interval raw | interval | affine | 5.9156 | 11.1556 | 0.2110 | 0.4133 |
| block raw | block | diagonal scale+bias | 6.0726 | 10.4119 | 0.1779 | 0.3145 |
| block raw | block | affine | 5.2435 | 10.1972 | 0.1773 | 0.3142 |

This is the first sign that FiMA translations contain recoverable signal: a train-edge affine calibration improves both chain error and held-out edge error. However, the best calibrated chain still has `5.24 mm` mean absolute translation error, which is far larger than the `0.944 mm` GT adjacent-step p95. This is not yet good enough for the uncertainty-weighted GS story.

Practical conclusion: continue improving/calibrating FiMA edge predictions before running expensive GS combinations. A short GS fixed-prior smoke can be run later, but image metrics should not be treated as pose-prior validation.
