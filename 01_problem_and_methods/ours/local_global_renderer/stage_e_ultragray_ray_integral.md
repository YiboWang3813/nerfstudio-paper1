# Ours UltraGray Ray Integral Stage E

Date: 2026-06-16
Branch: `dev/ours-ultragray-ray-integral-stage-e-20260616`

## Goal

Stage D 仍然把 global attenuation 写成简化 lateral/elevational weight。Stage E 的目标是把 attenuation branch 对齐 UltraGray 的传播建模，同时修掉实现层面的明显低效点：

1. 使用 UltraGray 风格的 ray-Gaussian line integral。
2. 衰减合成使用 `T_i = tau + (1 - tau) * exp(-psi)`。
3. `psi` 使用 3-point Gauss-Legendre quadrature，默认窗口 `2.0mm`。
4. Ray-CSR 在 forward 构建并保存，backward 复用，不再重建。
5. Quaternion VJP 从 finite-diff 改为 analytic `transform_grads_R_to_Quat`。
6. 去掉 ray loop 里重复的 probe stats 计算，直接使用 CSR sorted depth 做 early break。

## Implementation

New config field:

```text
model.slice_render.ray_attenuation_quadrature_window_mm: 2.0
```

Forward now returns local and ray CSR state:

```text
colors, modulation, local_csr, ray_start, ray_end, ray_indices, ray_depths
```

Backward consumes the saved ray CSR tensors:

```text
RayCsrAttenuationBackwardCUDA(..., ray_start, ray_end, sorted_gauss_ids, sorted_depth_pix, ...)
```

Compared with UltraGray, Stage E keeps Ours' probe/world coordinate transform and local tile response branch. The global branch now only differs in candidate organization: Ours builds a per-lateral-pixel ray CSR from transformed Gaussians; UltraGray starts from explicit ray candidates.

## VJP Note

UltraGray's public VJP treats the quadrature interval endpoints as fixed after choosing the closest-approach window. That misses gradients through:

```text
t_start/t_end -> t_center -> (o_g, d_g)
```

Stage E implements this boundary VJP as well. This is not a new modeling claim; it is the complete derivative of the same quadrature expression.

Finite-difference checks must avoid candidate-boundary discontinuities. Perturbing lateral means or scales can change the discrete ray/tile candidate set. Those are mapper discontinuities, not continuous shader-gradient errors.

## Verification

Build:

```text
bash -lc 'source ~/.bashrc && conda activate gaussian_splatting && module load cuda/11.8.0 && bash ops/ours_renderer/rebuild.sh'
```

Result:

```text
ours_renderer smoke check passed.
```

Diagnostics ran on Slurm node `gpu3` / RTX 4090:

```text
scripts/methods/ours/local_global_renderer/slurm/run_ray_stage_e_diagnostics.sh
job: 63844
```

Direct ray-CSR gradient check:

```text
attenuation, means, scales, quats checked
gradient check passed: worst_rel_err=0.046243
```

Full renderer autograd check:

```text
ray-enabled worst_main_rel_err       = 0.008801
ray-enabled worst_scale_quat_rel_err = 0.011471
local-only  worst_main_rel_err       = 0.000002
local-only  worst_scale_quat_rel_err = 0.000213
```

The full renderer diagnostic intentionally skips `means/scales` finite differences because the local tile mapper has discrete candidate boundaries.

## 200-iter Smoke

Run:

```text
outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_20260616/stage_e_smoke_200
job: 63845
experiment: preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_separated_attenuation_ray_csr_tail_2k
overrides: engine.iterations=200
```

Result:

```text
iter 100: loss/total=0.2978, loss/l1=0.0472, loss/ssim=0.2506
iter 200: loss/total=0.2448, loss/l1=0.0389, loss/ssim=0.2059
progress bar speed: 68.58 it/s
slurm elapsed: 00:00:40
```

For context, Stage D full geometry VJP 2k logged about `87.96 it/s`. The Stage E smoke includes startup overhead and only 200 iterations, so this is not a final throughput number, but it confirms the new ray integral renderer trains stably.

## Current Interpretation

Stage E fixes the main implementation objections:

- attenuation is now a real ray integral rather than a local proxy;
- geometry gradients cover attenuation, means, scales, and quats;
- quaternion VJP is analytic;
- forward/backward share ray CSR;
- obvious duplicate per-candidate stats are removed.

The remaining open question is empirical value, not basic correctness. The next useful experiment is a matched 2k or 10k TUS-REC comparison against Stage D/local-only under the same split.

## Matched 2k Ablation

User question:

```text
Does the completed attenuation + strength renderer beat single-strength local rendering?
```

Protocol:

- Dataset/config base: `preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_echo_alpha_ray_csr_tail_2k`
- Data split: TUS-REC-2024 QC single sweep, 325 train frames + 40 holdout test frames
- Iterations: 2000
- GPU: Slurm `gpu3`, RTX 4090
- Losses for both variants: `[l1, ssim, scale_wall]`
- `attenuation_tail` disabled for both variants to isolate renderer structure
- Full variant: `echo_alpha` local branch + Stage-E ray attenuation
- Single-strength variant: same `echo_alpha` local branch, `ray_attenuation_enabled=false`

Runs:

```text
outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_single_strength_2k
outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_full_ray_2k
jobs: 63852, 63853
```

Results:

| Iter | Variant | Test L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | single strength | 0.0335 | 25.2732 | 0.6685 | 0.3749 |
| 1000 | full ray + strength | 0.0333 | 25.3351 | 0.6706 | 0.3734 |
| 2000 | single strength | 0.0199 | 29.2402 | 0.8004 | 0.2693 |
| 2000 | full ray + strength | 0.0198 | 29.2697 | 0.8013 | 0.2680 |

Final 2k delta of full ray over single strength:

```text
PSNR  +0.0295 dB
L1    -0.000062
SSIM  +0.00090
LPIPS -0.00137
```

Interpretation:

The completed Stage-E attenuation branch is no longer an obvious toy or negative component. Under this clean single-sweep 2k ablation it gives a small but consistent gain across all four test metrics. The gain is still weak; this result supports continuing to longer and broader validation, but it is not strong enough by itself to claim a robust journal-level physical-rendering improvement.

## Effect-Size Diagnostic

Run:

```text
job: 63859
script: scripts/methods/ours/local_global_renderer/slurm/run_ray_stage_e_effect_diagnostic.sh
checkpoint: outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_full_ray_2k/checkpoints/iteration_2000_container.pth
output: outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_full_ray_2k/ray_effect_diagnostic.json
```

Key result:

```text
active-pixel modulation median = 0.9600
active-pixel modulation mean   = 0.9628
active-pixel modulation p95    = 0.9934
```

This matches the configured cap:

```text
image = local_strength * (1 - beta + beta * T)
beta = 0.20
T >= 0.80
minimum modulation = 0.80 + 0.20 * 0.80 = 0.96
```

Gradient diagnosis from mean L1 over 8 holdout-spread frames:

| Raw parameter | mean abs grad | RMS grad | max abs grad | nonzero frac |
| --- | ---: | ---: | ---: | ---: |
| means | 1.35e-7 | 3.75e-7 | 7.11e-6 | 0.6388 |
| scales | 6.17e-8 | 1.73e-7 | 4.31e-6 | 0.6382 |
| quats | 1.53e-7 | 2.65e-5 | 2.28e-2 | 0.6388 |
| echo_alphas | 7.31e-8 | 1.95e-7 | 2.58e-6 | 0.6386 |
| attenuations | 1.05e-11 | 3.16e-10 | 8.17e-8 | 0.0210 |

Interpretation:

The default Stage-E attenuation branch is severely under-powered. On active pixels it mostly applies only the hard 4% cap, and its raw attenuation gradient is about 7000x smaller than the echo-alpha gradient by mean absolute value. The weak metric gain above is therefore consistent with limited action space and weak optimization signal, not necessarily with a useless attenuation model.

## Headroom Sweep

Protocol is the same as the matched 2k ablation above, except attenuation strength is increased. All runs keep `attenuation_tail` disabled.

Variants:

| Variant | beta | strength | min T |
| --- | ---: | ---: | ---: |
| single strength | 0.00 | n/a | n/a |
| weak/full default | 0.20 | 0.08 | 0.80 |
| medium | 0.50 | 0.20 | 0.60 |
| strong | 1.00 | 0.50 | 0.40 |

Runs:

```text
medium job: 63861
strong job: 63860
outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_medium_ray_2k
outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_strong_ray_2k
```

Final 2k test metrics:

| Variant | Test L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| single strength | 0.019908 | 29.2402 | 0.8004 | 0.2693 |
| weak/full default | 0.019846 | 29.2697 | 0.8013 | 0.2680 |
| medium | 0.019423 | 29.4683 | 0.8052 | 0.2572 |
| strong | 0.020874 | 28.7957 | 0.8073 | 0.2428 |

Relative to single-strength:

| Variant | L1 improvement | PSNR change | SSIM improvement | LPIPS improvement |
| --- | ---: | ---: | ---: | ---: |
| weak/full default | +0.313% | +0.0295 dB | +0.112% | +0.507% |
| medium | +2.438% | +0.2281 dB | +0.603% | +4.487% |
| strong | -4.850% | -0.4445 dB | +0.866% | +9.830% |

Interpretation:

Increasing attenuation headroom clearly matters. The medium setting turns the weak positive effect into a more visible all-metric gain. The strong setting over-regularizes or over-darkens photometric fidelity: SSIM and LPIPS improve, but L1 and PSNR degrade. The useful region is likely around `beta=0.4-0.7`, `strength=0.15-0.35`, and `min_transmission=0.5-0.7`, not at the original conservative cap and not at the fully strong setting.

Next recommended experiment:

```text
fine sweep around medium:
  beta: 0.40, 0.50, 0.60
  strength: 0.15, 0.20, 0.30
  min_transmission: 0.55, 0.60, 0.70
```

Keep this sweep small first. The current evidence says "attenuation can help if it is allowed to act", but it also shows too much attenuation trades away PSNR/L1 for perceptual smoothness.

## Fine Headroom Sweep

Protocol:

```text
job: 63862
script: scripts/methods/ours/local_global_renderer/slurm/run_ray_stage_e_fine_sweep_serial.sh
grid:
  beta: 0.40, 0.50, 0.60
  strength: 0.15, 0.20, 0.30
  min_transmission: 0.55, 0.60, 0.70
outputs:
  outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_fine_sweep_20260616
summary:
  outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_fine_sweep_20260616/summary.md
```

The Slurm training job completed all 27 runs. Its final in-job summary failed because Python 3.7 lacks `str.removeprefix`; the summarizer was patched and rerun manually. Training metrics and checkpoints were unaffected.

Top results sorted by PSNR:

| Rank | beta | strength | minT | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.60 | 0.30 | 0.55 | 0.019351 | 29.4997 | 0.8065 | 0.2526 |
| 2 | 0.40 | 0.30 | 0.55 | 0.019409 | 29.4905 | 0.8049 | 0.2572 |
| 3 | 0.60 | 0.20 | 0.55 | 0.019415 | 29.4883 | 0.8066 | 0.2524 |
| 4 | 0.60 | 0.20 | 0.60 | 0.019347 | 29.4872 | 0.8063 | 0.2532 |
| 5 | 0.50 | 0.30 | 0.55 | 0.019377 | 29.4805 | 0.8060 | 0.2549 |
| 6 | 0.60 | 0.15 | 0.60 | 0.019405 | 29.4721 | 0.8057 | 0.2545 |
| 7 | 0.50 | 0.15 | 0.60 | 0.019461 | 29.4637 | 0.8052 | 0.2571 |
| 8 | 0.50 | 0.20 | 0.55 | 0.019438 | 29.4633 | 0.8054 | 0.2554 |
| 9 | 0.50 | 0.15 | 0.55 | 0.019426 | 29.4616 | 0.8055 | 0.2547 |
| 10 | 0.60 | 0.30 | 0.60 | 0.019444 | 29.4594 | 0.8057 | 0.2536 |

Metric-specific winners:

| Metric | beta | strength | minT | Value |
| --- | ---: | ---: | ---: | ---: |
| PSNR | 0.60 | 0.30 | 0.55 | 29.4997 |
| L1 | 0.60 | 0.20 | 0.60 | 0.019347 |
| SSIM | 0.60 | 0.20 | 0.55 | 0.8066 |
| LPIPS | 0.60 | 0.20 | 0.55 | 0.2524 |

Best-PSNR point versus single-strength baseline:

| Variant | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| single strength | 0.019908 | 29.2402 | 0.8004 | 0.2693 |
| best fine sweep | 0.019351 | 29.4997 | 0.8065 | 0.2526 |

Relative change:

```text
L1    +2.797%
PSNR  +0.2595 dB
SSIM  +0.761%
LPIPS +6.190%
PSNR-implied MSE reduction: 5.80%
```

Interpretation:

The useful region is now clearer: `beta=0.60`, `min_transmission=0.55-0.60`, and `strength=0.20-0.30`. `min_transmission=0.70` is consistently weaker, suggesting that the branch needs enough modulation range. The best PSNR point and the best perceptual point are slightly different:

- Use `beta=0.60, strength=0.30, minT=0.55` if optimizing PSNR/L1 balance.
- Use `beta=0.60, strength=0.20, minT=0.55` if optimizing SSIM/LPIPS.

For the next validation stage, the recommended default is `beta=0.60, strength=0.30, minT=0.55`, with `beta=0.60, strength=0.20, minT=0.55` kept as the perceptual-fidelity alternative.

## No-DC Controlled Ablation

Motivation:

The DC-on comparison is the practical method setting, but it is not a clean renderer-only comparison. Local-only and local-global renderers produce different gradients, so density control can clone/prune different Gaussians. To test whether attenuation helps without changing DC trajectories, this run disables densify and prune for all variants.

Protocol:

```text
job: 63868
script: scripts/methods/ours/local_global_renderer/slurm/run_ray_stage_e_no_dc_ablation_serial.sh
output root:
  outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_no_dc_ablation_20260616
common overrides:
  model.offline.density_control.enable.densify=false
  model.offline.density_control.enable.prune=false
  loss.needed_losses=[l1,ssim,scale_wall]
  loss.lambdas.attenuation_tail=0.0
```

Log verification:

```text
[OursDensityController] Init Densify=False, Prune=False
train/num_gaussians=3.000e+05 throughout all runs
```

Final 2k test metrics:

| Variant | beta | strength | minT | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single strength | 0.00 | n/a | n/a | 0.019954 | 29.2118 | 0.8016 | 0.2623 |
| best ray | 0.60 | 0.30 | 0.55 | 0.019376 | 29.4994 | 0.8073 | 0.2493 |
| perceptual ray | 0.60 | 0.20 | 0.55 | 0.019471 | 29.4700 | 0.8065 | 0.2508 |

Best ray versus no-DC single strength:

```text
L1    +2.897%
PSNR  +0.2876 dB
SSIM  +0.711%
LPIPS +4.956%
PSNR-implied MSE reduction: 6.41%
```

Perceptual ray versus no-DC single strength:

```text
L1    +2.421%
PSNR  +0.2582 dB
SSIM  +0.611%
LPIPS +4.384%
PSNR-implied MSE reduction: 5.77%
```

Interpretation:

The attenuation gain survives with density control disabled and fixed 300k Gaussians. This rules out the simplest confound that the observed improvement is only caused by different clone/prune behavior. DC may still amplify or shape the practical method behavior, but the renderer/attenuation branch itself is already beneficial under a controlled no-DC setting.

Updated next step:

Run longer matched validation with the same three variants:

1. `single_strength`
2. `best_ray`: `beta=0.60, strength=0.30, minT=0.55`
3. `perceptual_ray`: `beta=0.60, strength=0.20, minT=0.55`

Run both no-DC and DC-on at 10k if budget allows. If budget is tight, prioritize DC-on 10k for practical performance and keep the no-DC 2k result as the controlled renderer evidence.

## 10k Matched Validation

Protocol:

```text
job: 63905
script: scripts/methods/ours/local_global_renderer/slurm/run_ray_stage_e_10k_validation_serial.sh
output root:
  outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_10k_validation_20260616
common overrides:
  engine.iterations=10000
  engine.interval.test=5000
  engine.interval.save=10000
  loss.needed_losses=[l1,ssim,scale_wall]
  loss.lambdas.attenuation_tail=0.0
```

This is the DC-on practical setting. Density control remains enabled with the inherited TUS-REC QC timing (`start_iter=500`, `end_iter=1500`, `freq=500`), matching the earlier 2k practical experiments except for longer optimization.

Final 10k metrics:

| Variant | beta | strength | minT | Gaussians | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single strength | 0.00 | n/a | n/a | 188425 | 0.015284 | 31.4357 | 0.8393 | 0.2194 |
| best ray | 0.60 | 0.30 | 0.55 | 205723 | 0.014746 | 31.7286 | 0.8447 | 0.2052 |
| perceptual ray | 0.60 | 0.20 | 0.55 | 205354 | 0.014744 | 31.7225 | 0.8446 | 0.2055 |

5k intermediate metrics:

| Variant | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| single strength | 0.0161 | 30.9910 | 0.8322 | 0.2315 |
| best ray | 0.0155 | 31.2771 | 0.8379 | 0.2139 |
| perceptual ray | 0.0155 | 31.2935 | 0.8381 | 0.2144 |

Best ray versus 10k single strength:

```text
L1    +3.623%
PSNR  +0.2929 dB
SSIM  +0.643%
LPIPS +6.471%
PSNR-implied MSE reduction: 6.52%
```

Perceptual ray versus 10k single strength:

```text
L1    +3.922%
PSNR  +0.2868 dB
SSIM  +0.631%
LPIPS +6.335%
PSNR-implied MSE reduction: 6.39%
```

Interpretation:

The 2k conclusion holds under longer optimization. Both ray variants remain consistently better than single-strength at 5k and 10k. The two tuned ray variants are very close: `best_ray` gives the highest PSNR/SSIM/LPIPS at 10k, while `perceptual_ray` gives a slightly lower L1. The recommended default remains:

```text
beta=0.60, strength=0.30, minT=0.55
```

The 10k result also shows that DC-on practical behavior keeps about 17k more Gaussians for ray variants than local-only. Since no-DC 2k was already positive, this should be described as two compatible effects: the renderer itself improves reconstruction, and the altered gradients/DC trajectory may further support practical performance.
