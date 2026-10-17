# TDSc Echo Saturation Tuning Notes

Date: 2026-06-25

Dataset and sweep: `tdsc_abus_2023/validation_100`

Purpose: diagnose and repair the TDSc representative sweep where Ours underperformed GaussianPile and Slice-adapted-3DGS under the fair GS protocol.

## Protocol

All runs in this note used the same comparison-side constraints:

- Dataset split: `holdout_interval`, trimmed middle 80/20 holdout.
- Pose: local file/absolute pose.
- Initial Gaussians: `1,000,000`.
- Training iterations: `30,000`.
- Checkpoint selection: final checkpoint.
- Metrics: `psnr`, `l1`, `ssim`, `lpips`, `dists`, `gmsd`.
- Loss: unified L1 + SSIM comparison loss.
- DC timing: `start=1000`, `end=25000`, `freq=2000`.
- Scale activation: bounded sigmoid, `scale_min_mm=0.03`, `scale_max_mm=3.89`.
- Renderer: Ours local/global renderer, no `probe3d`, no `plane_conv2d`.
- Attenuation path: enabled, source `attenuation`, `ray_attenuation_formula=slice_depth`.

## Baseline Problem

Before the TDSc-specific echo tuning, the best Ours candidate was:

`ours_b8_no_intprune_slice_depth_b080_s055_min035`

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours previous best | 27.8322 | 0.02577 | 0.78835 | 0.24955 | 0.19177 | 0.06714 | 919,337 |

Against the same representative sweep, this was still below:

| Method | PSNR | LPIPS | DISTS |
|---|---:|---:|---:|
| GaussianPile | 28.3473 | 0.23257 | 0.18675 |
| Slice-adapted-3DGS | 28.1656 | 0.22612 | 0.19694 |
| UltraGray | 27.9915 | 0.23207 | 0.20898 |

Visual and per-frame diagnosis showed that Ours preserved broad structure but lost local high-frequency speckle and shallow edge sharpness. The worst PSNR gap against GaussianPile concentrated in the later test-frame region, especially item `51-68`, with item `1` also poor.

Diagnostics:

- `outputs/preproduction/diagnostics/tdsc_abus_2023_validation_100_ours_gap/`
- `outputs/preproduction/diagnostics/tdsc_abus_2023_validation_100_ours_cause/`

## Cause Diagnosis

The issue was not that the echo branch failed to learn. It was that the echo branch saturated.

Final attribute statistics for the previous best:

| Attribute | Mean | P50 | P75 | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| `echo_alpha` | 0.0818 | 0.0254 | 0.1990 | 0.2000 | 0.2000 | 0.2000 |
| `attenuation` | 0.0835 | 0.0203 | 0.0232 | 0.1114 | 0.9950 | 1.0000 |
| `scale_mm` | 0.5728 | 0.4762 | 0.7364 | 1.0563 | 2.1105 | 3.8900 |

Important fractions:

- `echo_alpha >= 0.19`: `30.2%`.
- `attenuation >= 0.5`: `6.5%`.

Attenuation itself was useful and should not be disabled:

- Previous b8 attenuation-on improved over b8 attenuation-off by `+0.2695 dB` PSNR overall.
- On the worst item `51-68` cluster, b8 attenuation-on still improved over b8 attenuation-off by `+0.1858 dB`.

Conclusion: the failure mode is a coupling between strong attenuation and saturated local echo. The model explains tissue response with a small high-attenuation tail plus many saturated echo Gaussians, but this allocation is too coarse for fine speckle/edge fidelity.

## Failed or Weak Directions

These directions did not solve the issue:

- Increasing `echo_alpha_max` above `0.2`: worsened metrics.
- Lowering attenuation learning rate alone: reduced strong attenuation tail but did not improve PSNR.
- Turning attenuation off: significantly worse than attenuation-on.
- Continuing to tune DC thresholds alone: did not address the saturation failure mode.

## Effective Direction

Reducing `model.gauss_container.lr.echo_alphas` was effective.

First echo saturation fix sweep:

Output root: `outputs/preproduction/ideas/ours/tdsc_echo_saturation_fix/`

| Variant | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final G | `echo>=0.19` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old best | 27.8322 | 0.02577 | 0.78835 | 0.24955 | 0.19177 | 0.06714 | 919,337 | 0.302 |
| `echo_init005_lr004_attlr010` | 27.9525 | 0.02545 | 0.79210 | 0.23092 | 0.18350 | 0.06517 | 1,000,058 | 0.201 |
| `echo_init003_lr004_attlr010` | 27.9314 | 0.02548 | 0.79184 | 0.23385 | 0.18464 | 0.06535 | 999,328 | 0.198 |
| `echo_init005_lr002_attlr010` | 28.0087 | 0.02533 | 0.79325 | 0.21771 | 0.17779 | 0.06379 | 1,029,228 | 0.097 |
| `echo_init005_lr004_attlr005` | 27.9255 | 0.02552 | 0.79180 | 0.23072 | 0.18438 | 0.06518 | 1,000,052 | 0.203 |

The `echo_lr=0.02` run confirmed that suppressing echo saturation improves accuracy and perceptual metrics.

Second echo learning-rate refinement:

Output root: `outputs/preproduction/ideas/ours/tdsc_echo_lr_refine/`

| Echo LR | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final G | `echo>=0.19` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.010` | 28.0179 | 0.02516 | 0.79339 | 0.21207 | 0.17276 | 0.06312 | 1,031,524 | 0.037 |
| `0.015` | 28.0200 | 0.02525 | 0.79331 | 0.21378 | 0.17498 | 0.06348 | 1,030,412 | 0.065 |
| `0.020` | 28.0157 | 0.02531 | 0.79322 | 0.21844 | 0.17702 | 0.06391 | 1,029,225 | 0.097 |
| `0.025` | 27.9929 | 0.02538 | 0.79293 | 0.22067 | 0.17929 | 0.06438 | 1,026,915 | 0.130 |

Diagnostics:

- `outputs/preproduction/diagnostics/tdsc_abus_2023_validation_100_ours_echo_fix/`
- `outputs/preproduction/diagnostics/tdsc_abus_2023_validation_100_ours_echo_lr_refine/`

## Current Recommendation

For TDSc production candidate:

```yaml
model:
  gauss_container:
    constraints:
      scale_activation: bounded_sigmoid
      scale_min_mm: 0.03
      scale_max_mm: 3.89
      echo_alpha_max: 0.2
    init_val:
      echo_alphas: 0.05
    lr:
      echo_alphas: 0.015  # PSNR-best TDSc candidate
      attenuations: 0.01
  offline:
    density_control:
      thresholds:
        prune_intensity: 0.0
  slice_render:
    ray_attenuation_enabled: true
    ray_attenuation_source: attenuation
    ray_attenuation_formula: slice_depth
    ray_attenuation_beta: 0.80
    ray_attenuation_strength: 0.55
    ray_attenuation_min_transmission: 0.35
    ray_attenuation_quadrature_points: 1
```

Use `echo_alphas=0.015` for PSNR-oriented production ranking. Use `echo_alphas=0.010` if perceptual robustness is preferred, because it has better L1, SSIM, LPIPS, DISTS, and GMSD while being only `0.0020 dB` lower in PSNR.

## Remaining Gap

The best TDSc Ours candidate now beats UltraGray on this sweep:

| Method | PSNR | LPIPS | DISTS |
|---|---:|---:|---:|
| Ours, `echo_lr=0.015` | 28.0200 | 0.21378 | 0.17498 |
| UltraGray | 27.9915 | 0.23207 | 0.20898 |

But it still trails:

| Method | PSNR Gap vs Ours |
|---|---:|
| Slice-adapted-3DGS | `+0.1456 dB` |
| GaussianPile | `+0.3273 dB` |

The biggest remaining gap is still in the later frames, especially item `51-68`. Further gains likely require a small two-factor search around the selected echo LR:

- `echo_alpha_max`: e.g. `0.16`, `0.18`, `0.20`.
- attenuation strength or min transmission around the current slice-depth setting.

Do not return to broad DC-threshold sweeps until the echo/attenuation allocation is stable.

## Related Commits

- `f488294`: Add TDSC ours echo saturation fix sweep.
- `3dc75b8`: Add TDSC ours echo LR refine sweep.
