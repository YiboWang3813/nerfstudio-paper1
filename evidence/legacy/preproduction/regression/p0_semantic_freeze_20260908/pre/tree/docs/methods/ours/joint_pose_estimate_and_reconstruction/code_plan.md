# Ours + FiMA Joint Pose-Free Reconstruction Code Plan

## Goal

Build and evaluate a pose-free single-sweep ultrasound reconstruction path:

```text
input B-mode sweep
  -> FiMA pose estimator
  -> coarse probe poses
  -> Ours Gaussian scene reconstruction
  -> rendering losses
  -> gradients improve scene and, in the joint setting, FiMA pose estimation
```

This is an exploratory research line. It must not disrupt the current production
known-pose benchmark pipeline.

## Fixed Task Protocol

Do not change the current single-sweep train/test protocol to fit FiMA.
FiMA and any joint model must adapt to the existing protocol.

- Each experiment uses one sweep at a time.
- Only linear or near-linear single-sweep tasks are in scope for the four current
  datasets.
- Do not use COLMAP for these tasks. Linear slice sweeps do not satisfy the
  usual multi-view feature-consistency assumptions required by COLMAP.
- The existing holdout protocol remains fixed:
  - trim unstable head/tail frames as configured;
  - choose held-out frames by the current holdout interval / linspace policy;
  - train frames are used for optimization;
  - held-out test frames are used for novel-slice rendering and metrics.
- Test-frame image reconstruction loss must not backpropagate during training.

Important distinction:

- For pose-estimation inference, the pose network may receive a test frame as
  input if the task is defined as "given the full test-time sweep images, predict
  the sweep poses." This is normal trackerless inference.
- For scene reconstruction training, test images must not contribute rendering
  loss or update Gaussian / pose-network parameters.

## Current Code Audit

### Ours

`methods/ours` already has pose-gradient support.

- `methods/ours/pose_optimization.py`
  - `OursPoseManager` supports external pose prior via
    `model.pose_optimization.prior_poses_abspath`.
  - Current optimized pose is `prior @ delta`.
  - The delta is per-frame trainable `pose_delta_rot` and `pose_delta_trans`.
  - This supports weak joint optimization but does not update a pose network.
- `methods/ours/render.py`
  - `render(..., pose_override=...)` can render with a differentiable pose.
- `ops/ours_renderer/ours_renderer/autograd.py`
  - CUDA backward returns `d_pose` when the pose input requires gradient.
- Existing renderer tests already validate useful pose gradients:
  - `tests/ops/ours_renderer/validate_us_focus_pose_se3_gradient.py`
  - `tests/ops/ours_renderer/validate_us_focus_pose_recovery.py`
  - `tests/ops/ours_renderer/validate_us_focus_joint_pose_gauss.py`

Conclusion: the renderer is ready for differentiable pose optimization.

### FiMA

`methods/fima` is currently an offline pose-estimation pipeline.

- `methods/fima/pose_estimator.py`
  - `FiMAPoseEstimator` predicts 6DoF pose from an image sequence window.
  - Supports `last_frame_rel0`, `window_frames_rel0`, and `window_edges`.
- `methods/fima/pose_estimator_adjacent.py`
  - `FiMAAdjacentPairEstimator` predicts adjacent-frame relative transform.
- `methods/fima/sequence_dataset.py`
  - Builds causal windows and adjacent-pair training batches.
  - Can align with `Scene.train_indices` / `Scene.test_indices`.
- `methods/fima/pem_pose_export.py`
  - Exports chained `(N, 4, 4)` pose priors from edge predictions.
- `methods/fima/README.md`
  - Explicitly says online rendering integration is not implemented.

Conclusion: FiMA can provide a stage-1 pose prior today, but it is not yet wired
into Ours as a trainable module receiving reconstruction gradients.

## Known Gaps To Fix First

### 1. Absolute Pose Export

Current FiMA chained export uses `R0 = I`, which is only correct for
relative-pose datasets.

Required behavior:

- If scene pose type is relative, keep `P_pred[0] = I`.
- If scene pose type is absolute, use the sweep's first loaded pose:

```text
P_pred[0] = P_gt[0] or chosen reference pose
P_pred[i + 1] = P_pred[i] @ T_i
```

This is required for `spine_phantom`, `tus_rec_2024_qc`, and any absolute-pose
scene.

### 2. Holdout Window Semantics

The train/test frame protocol must remain unchanged. FiMA must not compress
`train_indices` into a fake continuous timeline.

Bad pattern:

```text
original frames: 0 1 2 3 4 5
test frame:      3
train sequence:  0 1 2 4 5
then treating 2 -> 4 as adjacent
```

Required behavior:

- A supervised adjacent edge `(i, i + 1)` is valid only if both original global
  frame indices are consecutive and satisfy the selected split rule.
- A training render target frame must be a train frame.
- A multi-frame context window must preserve original global time indices.
- If a context window crosses held-out test frames, either:
  - reject the window for training; or
  - keep the temporal gap explicit and do not supervise fake adjacent edges
    across the gap.

For the first implementation, use the conservative option: reject train windows
that include test frames.

### 3. Joint Model Wiring

Hydra currently selects one `model.name`. A real joint model should not try to
enable `/model: fima` and `/model: ours` at the same time.

Recommended implementation:

- Add `methods/fima_ours/`.
- Add `configs_hydra/model/fima_ours.yaml`.
- Add experiment configs under `configs_hydra/experiment/ours_fima_joint/`.
- Add `FiMAOursJointState` and callbacks.

This keeps exploratory code isolated from production `ours` and `fima`.

### 4. Gaussian Initialization Without GT Pose Leakage

For pose-free claims, Gaussian initialization must use the pose prior available
to the tested method, not GT poses.

Allowed:

- GT-pose Ours upper bound initializes from GT poses.
- FIMA+Ours initializes from FIMA predicted poses.
- Noisy-pose ablation initializes from noisy poses.

Not allowed:

- Initialize Gaussians from GT pose, then claim pose-free reconstruction.

### 5. Degenerate Joint Optimization

A powerful Gaussian scene can fit images under wrong poses. Joint training must
prevent this failure mode.

Required regularizers / diagnostics:

- pose supervision loss when GT poses are available;
- pose prior / correction magnitude loss;
- temporal smoothness on relative pose increments;
- monotonic or near-linear sweep motion prior where appropriate;
- held-out frame rendering metrics;
- explicit pose error / drift / frame-corner error reporting.

## Implementation Milestones

### Milestone A: Protocol-Safe FiMA Export

Purpose: make FiMA compatible with the current held-out frame protocol and all
four dataset pose conventions.

Tasks:

1. Add a protocol-safe FiMA window / edge builder.
2. Ensure training windows preserve original global frame adjacency.
3. Add absolute-pose export support.
4. Export:
   - `pred_poses.npy`
   - `gt_poses.npy`
   - `edge_pose6.npy`
   - `frame_pose_errors.npy`
   - optional `frame_uncertainty.npy` / `frame_weights.npy`
5. Add tests with synthetic frame indices to catch fake adjacency across
   holdout frames.

Success criteria:

- Exported pose shape matches the active train sweep.
- Relative-pose datasets still export correctly.
- Absolute-pose datasets preserve the correct first-frame anchor.
- No supervised training edge crosses a held-out frame.

### Milestone B: Renderer-to-FiMA Gradient Bridge Test

Purpose: verify that reconstruction loss can update FiMA parameters.

Create a small CUDA test:

```text
two or short-window images
  -> FiMA pose estimator
  -> 6DoF to SE(3)
  -> Ours renderer
  -> image loss
  -> backward
```

Success criteria:

- FiMA head gradients are non-zero and finite.
- At least one earlier FiMA stem/mixer parameter has non-zero finite gradient.
- A few optimizer steps reduce the controlled render loss.
- No test image is used as train loss input in the test fixture.

This test should be small and deterministic, not a full experiment.

### Milestone C: Two-Stage Baseline

Purpose: establish the weak baseline before claiming joint learning.

Pipeline:

1. Train FiMA PEM on train frames / train sweeps.
2. Export `pred_poses.npy`.
3. Run Ours with:
   - `pose_optimization.enabled=true`
   - `pose_optimization.prior_poses_abspath=<pred_poses.npy>`
   - `pose_optimization.trainable=false`
4. Evaluate held-out test rendering.

Report:

- FIMA pose error:
  - adjacent edge translation / rotation;
  - chain drift;
  - frame-corner error if implemented.
- Ours rendering:
  - PSNR / SSIM / LPIPS where available;
  - visual compose images;
  - failure cases.

This is not the main method. It is the reference "pose prior only" baseline.

### Milestone D: Weak Joint With Per-Frame Delta

Purpose: use existing `OursPoseManager` to test whether reconstruction loss can
repair a FIMA pose prior.

Pipeline:

1. Use the same exported `pred_poses.npy`.
2. Run Ours with trainable residual delta:
   - `pose_optimization.enabled=true`
   - `pose_optimization.trainable=true`
   - tune `lambda_prior_rot`, `lambda_prior_trans`;
   - tune `lambda_smooth_rot`, `lambda_smooth_trans`.
3. Compare against two-stage fixed-pose baseline.

Report:

- whether pose error improves from prior to optimized pose;
- whether rendering metrics improve;
- whether optimized pose overfits train frames and hurts held-out frames.

This is a necessary baseline, but still not full joint network learning.

### Milestone E: Full FiMA-Ours Joint Training

Purpose: implement the actual research idea.

New module:

```text
methods/fima_ours/
  configs.py
  state.py
  callbacks.py
  pose_adapter.py
```

Core loop:

1. Build `Scene` with fixed holdout split.
2. Build FiMA pose estimator.
3. Build Ours Gaussian container and renderer.
4. For each training step:
   - sample a protocol-valid train frame or train window;
   - run FiMA on its image context;
   - convert predicted 6DoF edge / rel0 pose to `probe->world`;
   - optionally compose with a small learnable per-frame or per-edge delta;
   - render train frame(s);
   - compute image reconstruction loss;
   - add pose supervision / prior / smoothness losses;
   - backprop into Gaussian parameters and selected FiMA parameters.

Recommended training schedule:

1. Warm-start FiMA with supervised pose training.
2. Freeze FiMA, train Gaussians from predicted poses.
3. Unfreeze only FiMA head and last mixer layers.
4. Enable small residual delta.
5. Optionally reduce supervised pose loss weight to test reconstruction-guided
   improvement.

Avoid unfreezing the full pose network from iteration 0.

## Loss Design

Use a weighted sum:

```text
L = L_render_train
  + lambda_pose_sup * L_pose_sup
  + lambda_prior * L_delta_prior
  + lambda_smooth * L_temporal_smooth
  + lambda_cycle * L_pose_cycle
  + lambda_gauss_reg * L_gauss_regularization
```

Recommended terms:

- `L_render_train`: L1 + SSIM on train frames only.
- `L_pose_sup`: supervised relative pose loss where tracker pose labels are
  allowed.
- `L_delta_prior`: keeps residual correction small.
- `L_temporal_smooth`: second-difference smoothness over translation and
  rotation.
- `L_pose_cycle`: optional local consistency, e.g. composed short edges match
  direct longer edge.
- Existing Ours scale / opacity / intensity regularization should stay enabled.

Do not use held-out test render loss for training.

## Experiment Matrix

Start with `tus_rec_2024_qc`, because it is the most natural pose-free dataset
among the current four. Then extend to the others.

For each selected sweep:

1. `GT pose + Ours`
   - upper bound.
2. `FIMA pose + Ours fixed pose`
   - two-stage baseline.
3. `FIMA pose + Ours per-frame delta`
   - weak joint baseline using existing `OursPoseManager`.
4. `FIMA + Ours full joint`
   - main exploratory method.
5. `Noisy GT pose + Ours per-frame delta`
   - controlled sanity check for pose repair capacity.

Optional negative control:

- COLMAP pseudo pose, only if needed to show why it is not suitable for linear
  ultrasound slice sweeps.

## Metrics

Report both pose and reconstruction metrics.

Pose metrics:

- adjacent edge translation error;
- adjacent edge rotation error;
- accumulated chain drift;
- global translation / rotation error;
- frame-corner reconstruction error, matching the TUS-REC style if possible.

Rendering metrics:

- train PSNR / SSIM for overfit diagnosis;
- held-out test PSNR / SSIM / LPIPS if available;
- visual compose panels;
- pose-error-vs-render-error scatter if useful.

Runtime / stability:

- GPU memory;
- iteration speed;
- number of Gaussians;
- pose gradient norm;
- FiMA gradient norm;
- NaN / Inf checks on predicted pose matrices.

## Dataset Scope

Current production datasets:

- `synthetic_liver`
- `spine_phantom`
- `tus_rec_2024_qc`
- `tdsc_abus_2023`

Important caveats:

- `synthetic_liver` and `spine_phantom` have known poses and are useful for
  controlled supervised pose experiments.
- `tus_rec_2024_qc` is the most relevant target for trackerless freehand
  reconstruction.
- `tdsc_abus_2023` has generated parallel slice poses, not real tracked
  freehand poses. Use it carefully for pose-free claims.
- UltraGS Wild / Clinical should not be mixed into this plan. They are
  sensorless / COLMAP-pseudo-pose style datasets and do not match the current
  physical DatasetMeta benchmark.

## First Minimal Run

Recommended first real experiment:

Dataset:

- `scene=tus_rec_2024_qc`
- one QC sweep already known to render correctly under GT pose.

Run order:

1. FiMA supervised pose training on that sweep with protocol-safe windows.
2. Export predicted poses.
3. Ours fixed-predicted-pose smoke test.
4. Ours predicted-pose + trainable delta smoke test.
5. Full gradient bridge test.
6. Only then attempt full joint training.

Stop early if:

- FiMA pose error is too poor for Gaussians to initialize;
- Ours with fixed predicted pose cannot render anything meaningful;
- per-frame delta improves train frames but damages held-out frames;
- FiMA receives unstable or zero gradients from render loss.

## Deliverables

For the exploratory agent:

1. Protocol-safe FiMA export code and tests.
2. Renderer-to-FiMA gradient bridge test.
3. New isolated `methods/fima_ours` joint prototype.
4. Hydra configs for:
   - FiMA pose pretraining;
   - Ours fixed FIMA prior;
   - Ours trainable delta with FIMA prior;
   - full joint prototype.
5. A short report with:
   - pose metrics;
   - reconstruction metrics;
   - visual outputs;
   - whether joint training improves FiMA pose estimates over supervised-only
     or two-stage baselines.

The key claim to validate is not merely "FIMA can provide poses to Ours." The
key claim is:

```text
Ours reconstruction loss provides useful feedback that improves pose estimation
or improves held-out reconstruction beyond a simple two-stage pose-prior setup.
```
