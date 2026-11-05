# Ours pose-gradient diagnosis for FiMA/Ours joint training

Date: 2026-06-15

Branch: `ours-fima-joint-feasibility`

## Goal

Diagnose why Ours reconstruction loss does not meaningfully refine FiMA or synthetic noisy-GT poses.

Hypotheses:

1. The CUDA renderer produces very weak gradients with respect to pose translation.
2. Even when the image-loss pose gradient exists, its descent direction may not correspond to the true pose correction.
3. Joint Gaussian + pose optimization is ambiguous: Gaussian attributes/geometry can explain image loss without moving poses toward ground truth.

## Implementation

Added:

- `scripts/methods/ours/joint_pose_estimate_and_reconstruction/diagnose_pose_gradients.py`
- `scripts/methods/ours/joint_pose_estimate_and_reconstruction/slurm/slurm_pose_gradient_diagnosis.sh`

The diagnostic:

1. Loads the real TUS-REC sweep `010_LH_Per_L_DtP_qc_0005_0370`.
2. Uses either fresh Ours Gaussian initialization or a trained Ours container checkpoint.
3. Applies a known local pose residual: `0.5 mm` translation, `0 deg` rotation.
4. Runs one CUDA Ours renderer backward pass on frames `40,120,220`.
5. Reports:
   - pose residual gradient norms;
   - Gaussian raw-parameter gradient norms;
   - learning-rate-scaled update magnitudes;
   - cosine between the translation update direction and the true GT correction direction;
   - normalized pose-step loss changes.

## CUDA gradient path

The Ours CUDA backward path does return `grad_pose_probe_to_world`.

Relevant implementation:

- `ops/ours_renderer/ours_renderer/autograd.py`
  - `ProjectedSliceBackwardCUDA(..., need_pose_grad=ctx.needs_input_grad[5])`
  - returns `d_pose` to PyTorch autograd.
- `ops/ours_renderer/src/shader/backward.cu`
  - translation gradient:
    - `grad_pose[3,7,11] += -g_delta`
  - rotation-column gradient:
    - accumulated through normalized pose axes and Gaussian world offsets.

Interpretation:

The implementation matches the expected structure: image loss first produces gradients on projected Gaussian center/covariance terms, then those gradients are transformed back through the probe pose basis. Pose gradients exist, but translation is a thin downstream signal from the same rendering residuals that Gaussian parameters can also explain.

## Jobs

All jobs ran on `gpu3` RTX 4090.

| job | purpose | state |
|---:|---|---|
| `63385` | first fresh-init diagnostic | completed |
| `63387` | fresh-init diagnostic with normalized pose steps | completed |
| `63388` | train GT-pose Ours for 500 iters and save checkpoint | completed |
| `63389` | trained-checkpoint pose-gradient diagnostic | completed |

Main JSON outputs:

- `outputs/preproduction/ideas/ours/joint_pose_estimate_and_reconstruction/cd_feasibility/pose_gradient_diagnosis/pose_grad_trans05_normsteps_100k_frames40_120_220.json`
- `outputs/preproduction/ideas/ours/joint_pose_estimate_and_reconstruction/cd_feasibility/pose_gradient_diagnosis/pose_grad_trans05_gt500ckpt_100k_frames40_120_220.json`

## Results

### Fresh Ours initialization

Settings:

- `100,000` initial Gaussians
- frames: `40,120,220`
- loss: L1
- synthetic pose residual: `0.5 mm` translation

| metric | value |
|---|---:|
| base L1 | 0.1002756 |
| pose translation grad RMS | 3.5908e-05 |
| pose translation grad max | 9.7936e-05 |
| pose rotation grad RMS | 3.3989e-03 |
| current lr-scaled trans RMS (`3e-4`) | 1.0772e-08 mm/step |
| high lr-scaled trans RMS (`1e-2`) | 3.5908e-07 mm/step |
| cosine(update, true correction) | -0.0178 |

Per-frame cosine between the pose-gradient update and the true correction:

```text
[0.0269, -0.2242, -0.0128]
```

Normalized translation-step loss change:

| forced translation step along `-grad` | L1 delta |
|---:|---:|
| 0.01 mm | -1.49e-06 |
| 0.10 mm | -1.54e-05 |
| 0.50 mm | -7.42e-05 |

### 500-iteration trained Ours checkpoint

Checkpoint:

```text
outputs/preproduction/ideas/ours/joint_pose_estimate_and_reconstruction/cd_feasibility/ours_fima_posegrad_gt500_010_LH_Per_L_DtP_qc_0005_0370_gt_500/checkpoints/iteration_500_container.pth
```

Settings are otherwise identical to the fresh-init diagnostic.

| metric | value |
|---|---:|
| base L1 | 0.0933691 |
| pose translation grad RMS | 1.9108e-04 |
| pose translation grad max | 4.9109e-04 |
| pose rotation grad RMS | 1.7719e-02 |
| current lr-scaled trans RMS (`3e-4`) | 5.7323e-08 mm/step |
| high lr-scaled trans RMS (`1e-2`) | 1.9108e-06 mm/step |
| cosine(update, true correction) | 0.0249 |

Per-frame cosine:

```text
[0.0170, 0.1410, 0.0263]
```

Normalized translation-step loss change:

| forced translation step along `-grad` | L1 delta |
|---:|---:|
| 0.01 mm | -8.43e-06 |
| 0.10 mm | -8.21e-05 |
| 0.50 mm | -3.89e-04 |

Normalized rotation-step loss change:

| forced rotation step along `-grad` | L1 delta |
|---:|---:|
| 0.01 deg | -1.35e-05 |
| 0.10 deg | -1.33e-04 |
| 1.00 deg | -1.05e-03 |

## Interpretation

Pose gradients are present and can reduce the current image loss if forced to take normalized steps. The CUDA backward path is therefore not broken.

However, the gradients are not useful as a pose-refinement signal under the current optimizer setup:

- The actual learning-rate-scaled translation update is extremely small.
  - At `lr_trans=3e-4`, trained-checkpoint RMS update is only `5.7e-08 mm/step`.
  - Even at `lr_trans=1e-2`, it is only `1.9e-06 mm/step`.
- The translation gradient direction is almost orthogonal to the true correction direction.
  - Fresh init cosine: `-0.0178`.
  - 500-iter checkpoint cosine: `0.0249`.
- Rotation gradients are roughly two orders of magnitude larger than translation gradients, even when the injected noise is translation-only.
- A forced `0.5 mm` translation step along the image-loss gradient reduces L1 by only `3.9e-4` after 500 iters, but this direction is not aligned with the actual GT pose correction.

This explains the previous noisy-GT experiments:

- Standard pose delta barely changed `0.5 mm` synthetic noise.
- High-LR pose delta also barely changed it.
- Pose-only windows hurt reconstruction because Gaussian updates pause while pose updates do not move toward the true pose.

## Conclusion

The current Ours reconstruction loss is not a reliable pose-refinement objective for FiMA/Ours joint training.

More precisely:

- CUDA pose gradients exist.
- They can descend image loss locally.
- But translation gradients are tiny under the configured optimizer and poorly aligned with the true pose correction.
- The image-loss descent direction is dominated by ambiguous rendering explanations, especially rotation and Gaussian adaptation.

Therefore, the current per-frame `OursPoseManager` delta path should be treated as a negative result for pose correction.

## Recommended direction

For the paper idea, do not claim that GS reconstruction loss repairs FiMA pose drift in this setup.

Better options:

1. Use a physics-constrained pose prior as the core contribution:
   - metadata / sweep-statistics step;
   - FiMA predicts only small translation residuals;
   - rotation is clamped, low-pass bounded, or strongly regularized.
2. Replace per-frame pose deltas with a low-dimensional sweep model:
   - global step vector;
   - low-order drift;
   - small smooth residual field.
3. If pose refinement is revisited:
   - freeze Gaussian means/scales/intensities during pose steps;
   - use a larger, normalized pose optimizer rather than raw Adam on tiny gradients;
   - add explicit temporal smoothness and pose-prior losses;
   - diagnose cosine-to-known-correction on noisy-GT before trying FiMA priors.
