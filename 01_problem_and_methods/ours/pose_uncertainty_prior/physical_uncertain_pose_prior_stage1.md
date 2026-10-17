# Physical Uncertain Pose Prior Stage 1 Results

## Setup

- branch: `ours-physical-uncertain-pose-prior`
- jobs: `63404` (1k), `63409` (10k follow-up)
- GPU: `gpu3` / RTX 4090 node
- sweep: `010_LH_Per_L_DtP_qc_0005_0370`
- Ours setting: 100k init Gaussians, 1000 iterations, 20 holdout test frames
- output root: `outputs/preproduction/ideas/ours/pose_uncertainty_prior/physical_uncertain_prior`
- summary: `outputs/preproduction/ideas/ours/pose_uncertainty_prior/physical_uncertain_prior/ours_stage1_summary.tsv`

## Implemented Artifacts

- `scripts/methods/baselines/fima/postprocess_pose_prior.py`
  - `zero_rotation`
  - `physical_step`
  - `physical_residual`
  - optional frame scores/weights and markdown report
- `scripts/methods/ours/pose_uncertainty_prior/slurm/slurm_physical_uncertain_prior_stage1.sh`
- `tests/methods/fima/test_postprocess_pose_prior.py`

## Offline Pose Diagnostics

Input FiMA raw prior:

- abs translation mean: `15.0051 mm`
- abs rotation mean: `11.1549 deg`
- adjacent edge translation mean: `0.4882 mm`
- adjacent edge rotation mean: `0.0824 deg`

Postprocess results:

| prior | abs trans mean mm | abs rot mean deg | note |
| --- | ---: | ---: | --- |
| raw FiMA | 15.0051 | 11.1549 | free chained SE(3) |
| zero rotation | 13.3245 | 5.3525 | keep local FiMA translation, remove adjacent rotation |
| physical residual c015 | 13.3245 | 5.3525 | same as zero rotation; 0.15mm clamp did not activate |
| physical residual c005 | 17.3986 | 5.3525 | too strong translation clamp hurts accumulated pose |
| physical step | 24.9473 | 5.3525 | constant 0.404mm step badly underfits drift |

Interpretation:

- The useful first-order constraint is removing FiMA local rotations.
- Existing FiMA local translations should mostly be preserved.
- Forcing every edge close to the dataset-level `0.404mm` step is too rigid for this sweep.

## 1000-Iter Ours Results

| ablation | PSNR | L1 | SSIM | LPIPS | pose trans mm | pose rot deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT pose | 17.7960 | 0.0770 | 0.4962 | 0.4658 | n/a | n/a |
| raw FiMA fixed | 17.4048 | 0.0805 | 0.4583 | 0.4643 | 14.9896 | 11.1390 |
| zero rotation fixed | 17.5046 | 0.0785 | 0.4881 | 0.4673 | 13.2487 | 5.3559 |
| physical step fixed | 18.1955 | 0.0710 | 0.5527 | 0.4671 | 24.8631 | 5.3559 |
| physical residual c015 fixed | 17.4987 | 0.0786 | 0.4879 | 0.4681 | 13.2487 | 5.3559 |
| physical residual c005 fixed | 17.7339 | 0.0759 | 0.5117 | 0.4663 | 17.3116 | 5.3559 |
| physical residual c015 weighted | 17.5027 | 0.0786 | 0.4874 | 0.4675 | 13.2487 | 5.3559 |

## Conclusions

1. Raw FiMA is still a poor prior for Ours because chained rotation error is large.
2. Zeroing local rotations is consistently better than raw FiMA in both pose and reconstruction metrics.
3. `physical_residual_c015` is currently equivalent to zero-rotation because the residual clamp is loose.
4. `physical_residual_c005` worsens pose error but improves 1k reconstruction metrics versus c015/zero-rotation. This suggests short-run Ours metrics may favor smoother geometry even when absolute pose is worse.
5. `physical_step` gives the best 1k PSNR/SSIM despite the worst translation error. This is not enough to claim better pose prior; it is likely a smoothness/easier-optimization effect and must be checked at longer iterations.
6. Physical-deviation frame weights are almost constant (`min=0.3182`, `max=0.3329`, `mean=0.3252`), so weighting currently just rescales image loss. It does not provide meaningful selective uncertainty.

## 10000-Iter Follow-Up

Because the 1k result gave a suspiciously high score to `physical_step` despite poor pose error, job `63409` repeated the key variants for 10k iterations with density control from 1k to 8k.

| ablation | PSNR | L1 | SSIM | LPIPS | pose trans mm | pose rot deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT pose | 28.4729 | 0.0224 | 0.7716 | 0.3606 | n/a | n/a |
| raw FiMA fixed | 28.7850 | 0.0208 | 0.7701 | 0.3799 | 14.9896 | 11.1390 |
| zero rotation fixed | 28.9625 | 0.0199 | 0.7824 | 0.3582 | 13.2487 | 5.3559 |
| physical step fixed | 30.4269 | 0.0170 | 0.8078 | 0.3128 | 24.8631 | 5.3559 |
| physical residual c005 fixed | 29.6223 | 0.0186 | 0.7930 | 0.3407 | 17.3116 | 5.3559 |

Important implementation detail:

- `OursPoseManager._replace_scene_poses()` only replaces `scene.train_container`.
- In `scene.loading.test_mode=holdout_interval`, `Scene` sets `test_container = train_container` and test frames are read by held-out indices from the same container.
- Therefore full-container pose replacement also changes holdout test-frame poses. These image metrics are reconstruction metrics under the replaced pose list, not proof that the predicted pose chain is anatomically/physically accurate.

Interpretation:

1. Zeroing FiMA rotations is still the most defensible pose-prior improvement: it reduces pose error and improves Ours metrics versus raw FiMA.
2. `physical_step` and `physical_residual_c005` are reconstruction-friendly despite worse absolute pose. This is useful but should be framed as a geometry/trajectory regularization effect, not as a better pose estimator.
3. The current physical-deviation uncertainty remains weak because it is almost constant across frames. It should be replaced by learned FiMA log-variance, MC dropout, or ensemble disagreement.

## Next Decision

The next high-value experiment is no longer another GS-loss pose-refine run. The direction should split into two tracks:

1. A publishable pose-prior track:
   - constrained FiMA with zero/small rotation
   - bounded local translation residual
   - learned heteroscedastic uncertainty
   - report pose error and reconstruction metrics separately

2. A reconstruction-regularization track:
   - physical-step or soft physical-step priors as a trajectory regularizer
   - evaluate whether smoother priors improve geometry even when absolute pose error is worse
   - avoid claiming pose accuracy from image metrics alone
