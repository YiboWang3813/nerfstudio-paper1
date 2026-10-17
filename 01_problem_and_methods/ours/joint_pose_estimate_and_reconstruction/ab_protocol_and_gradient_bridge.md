# Ours + FiMA joint feasibility: Milestone A/B validation

Date: 2026-06-15

Branch: `ours-fima-joint-feasibility`

## Goal

Validate the first two feasibility gates for the Ours + FiMA pose-free reconstruction idea:

- Milestone A: protocol-safe FiMA pose export under the existing single-sweep holdout protocol.
- Milestone B: renderer-to-FiMA gradient bridge, proving Ours reconstruction loss can update a FiMA pose estimator.

This does not yet claim full joint training works. It only validates that the prerequisites are real.

## Milestone A: protocol-safe FiMA export

Implemented changes:

- `methods/fima/pem_pose_export.py`
  - Chain export now accepts an initial pose anchor.
  - Relative-pose scenes keep `P_pred[0] = I`.
  - Absolute-pose scenes use the first loaded sweep pose as the chain anchor.
- `methods/fima/sequence_dataset.py`
  - Multiframe FiMA under Scene holdout now builds windows only from original global-consecutive train frames.
  - It no longer compresses `train_indices` into a fake continuous timeline.
  - Target-statistics computation also uses only protocol-safe global windows.
  - The legacy compressed helper now rejects non-consecutive windows.

Tests added:

- `tests/methods/fima/test_protocol_safe_export.py`

Coverage:

- absolute chain composition preserves first-pose anchor;
- relative export keeps identity anchor;
- multiframe holdout rejects compressed fake timelines such as `[0, 1, 2, 4]`;
- multiframe holdout samples only globally consecutive windows;
- adjacent-pair holdout never creates fake cross-test edges such as `(2, 4)`.

## Milestone B: renderer-to-FiMA gradient bridge

Implemented test:

- `tests/methods/fima/test_renderer_gradient_bridge.py`

The test builds a minimal differentiable path:

```text
two B-mode-like rendered images
  -> FiMAAdjacentPairEstimator
  -> predicted 6DoF pose
  -> differentiable SE(3)
  -> Ours CUDA renderer
  -> image MSE loss
  -> backward into FiMA
```

Success criteria checked by the test:

- render loss is finite;
- predicted pose is finite;
- FiMA head gradients are non-zero and finite;
- FiMA stem gradients are non-zero and finite;
- several Adam steps reduce the controlled render loss.

No held-out test frame contributes training loss in this fixture.

## Verification

Local conda environment:

```bash
source ~/.bashrc
conda activate gaussian_splatting
PYTHONPATH=. pytest -q -p no:cacheprovider \
  tests/methods/fima/test_protocol_safe_export.py \
  tests/methods/fima/test_renderer_gradient_bridge.py
```

Result:

```text
6 passed in 4.14s
```

Python compile check:

```bash
python -m py_compile \
  methods/fima/pem_pose_export.py \
  methods/fima/sequence_dataset.py \
  tests/methods/fima/test_protocol_safe_export.py \
  tests/methods/fima/test_renderer_gradient_bridge.py
```

Result: passed.

Slurm RTX 4090 validation:

- Job: `63352`
- Node: `gpu3`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:00:39`

## Interpretation

Milestone A is feasible and fixed at the unit-test level. The biggest protocol risk was real: absolute-pose export previously anchored at identity, and multiframe holdout could use compressed train-index subsequences. Both are now guarded.

Milestone B is feasible at the minimal gradient-bridge level. Ours renderer loss can produce useful gradients through differentiable SE(3) into FiMA adjacent-pair estimator parameters, including early image features.

This supports moving to Milestone C/D:

- C: two-stage `FIMA predicted pose + Ours fixed pose`;
- D: `FIMA predicted pose + Ours trainable per-frame delta`.

It does not yet validate Milestone E full online joint training.

## Next recommended step

Run a small real-sweep two-stage baseline on `tus_rec_2024_qc`:

1. train FiMA adjacent-pair PEM with protocol-safe train edges;
2. export absolute-anchored `pred_poses.npy`;
3. run Ours with fixed FiMA prior;
4. run Ours with trainable per-frame delta;
5. report pose error, held-out render metrics, and whether delta improves held-out reconstruction or only overfits train frames.
