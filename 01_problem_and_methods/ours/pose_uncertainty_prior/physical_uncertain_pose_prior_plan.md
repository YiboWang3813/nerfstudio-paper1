# Physical Uncertain Pose Prior Implementation Plan

## Goal

Move the Ours + FiMA story away from direct GS-loss pose repair and toward an offline one-sweep ultrasound prior:

```text
physical adjacent-step prior + bounded FiMA residual + pose uncertainty-aware Ours reconstruction
```

The premise is that the sweep has strong motion structure. Adjacent slices should move by a small, smooth step near the dataset's elevational spacing, while a learned pose estimator can provide residual corrections and confidence. Ours should consume this prior probabilistically instead of treating the pose estimate as ground truth or relying on ambiguous photometric gradients to fix it.

## Stage 1: No-Retrain Physical Postprocess

Input: existing FiMA chained pose prior.

Output:

- constrained pose chain for `model.pose_optimization.prior_poses_abspath`
- frame uncertainty scores / weights for `model.pose_uncertainty.uncertainty_abspath`
- markdown diagnostics against GT when available

Implemented in:

- `scripts/methods/baselines/fima/postprocess_pose_prior.py`

Modes:

- `zero_rotation`: keep FiMA local translations and remove all adjacent rotations.
- `physical_step`: replace every adjacent edge with a constant physical step.
- `physical_residual`: use a constant physical step plus a bounded FiMA translation residual.

Default non-oracle policy:

- estimate the physical step direction from FiMA local edge median
- normalize the step magnitude to the known TUS-REC spacing when requested, e.g. `--step-mm 0.404`
- remove adjacent rotations in the first pass
- derive uncertainty from physical-model deviation:

```text
score_i = ||t_i^FiMA - t_base|| / trans_scale + rot_i^FiMA / rot_scale
weight_i = min_weight + (max_weight - min_weight) * exp(-alpha * score_i)
```

## Stage 2: Ours Short Validation

Run short Ours comparisons on the same TUS-REC sweep:

1. raw FiMA prior
2. zero-rotation FiMA prior
3. physical-step prior
4. physical-residual prior
5. physical-residual prior + frame uncertainty weighting

Expected interpretation:

- If physical-residual improves pose metrics and does not damage Ours reconstruction, it supports the physical-prior direction.
- If uncertainty weighting improves robustness only when some frames are low-confidence, it supports uncertainty-aware consumption.
- If fixed constrained pose beats trainable pose refinement, keep the narrative away from GS-loss pose repair.

## Stage 3: Heteroscedastic FiMA

Once Stage 1 is promising, train FiMA to output residual mean plus uncertainty:

```text
mu_i = residual edge pose
logvar_i = translation/rotation uncertainty
loss = supervised pose loss + lambda * heteroscedastic NLL
```

The repo already has scaffolding:

- `methods/fima/pose_estimator.py` has a `predict_uncertainty` log-variance head.
- `methods/fima/losses.py` has `heteroscedastic_pose_loss`.

Remaining work:

- wire heteroscedastic loss into `methods/fima/callbacks.py`
- add uncertainty export from FiMA edge inference
- calibrate uncertainty against edge and accumulated-chain pose errors

## Stage 4: Ours-Side Uncertainty Consumption

Start with existing frame weighting:

- `methods/ours/gs_only_callbacks.py`
- `model.pose_uncertainty.enabled=true`
- `model.pose_uncertainty.weight_mode=direct` or `exp`

Possible later extensions:

- density-control decoupling for uncertain frames
- MC pose marginalization from edge uncertainty
- support/residual uncertainty to avoid densifying geometry from unreliable slices

## Current Branch

`ours-physical-uncertain-pose-prior`
