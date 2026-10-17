# Frequency-Edge High-Pass Loss Idea Validation

Date: 2026-06-11
Branch: `ours-idea-frequency-edge`
GPU: RTX 4090 node `gpu3`
Scene protocol: offline one-sweep US focus reconstruction, `preproduction/ideas/ours/renderer_baseline/main_10k`, sweep `l1`, 1M initialization, holdout interval test frames.

## Idea

Dynamic-GS frequency-aware losses were adapted to the offline ultrasound setting as a high-pass residual consistency term. Instead of modeling temporal frequency, the method compares the high-frequency residuals of the rendered frame and target frame:

`L_highpass = |(I_pred - G(I_pred)) - (I_gt - G(I_gt))|_1`

where `G` is a fixed Gaussian low-pass filter. The intended ultrasound-specific role is to preserve fine speckle and boundary detail without changing the offline one-sweep renderer or density-control schedule.

Implementation is default-off:

- Loss registry key: `highpass_l1`
- Default config lambda: `0.0`
- Candidate config: `configs_hydra/experiment/preproduction/ideas/ours/frequency_edge/frequency_edge.yaml`
- Candidate settings: `kernel_size=9`, `sigma=2.0`, `lambda=0.01`

## 1k Lambda Probe

Output root: `outputs/preproduction/ideas/ours/frequency_edge/idea_frequency_edge_20260611`

| Variant | Seed | PSNR | LPIPS | L1 | SSIM |
|---|---:|---:|---:|---:|---:|
| baseline | 42 | 27.230255 | 0.258992 | 0.032491 | 0.561682 |
| highpass 0.005 | 42 | 27.227702 | 0.258975 | 0.032517 | 0.561744 |
| highpass 0.01 | 42 | 27.233055 | 0.258176 | 0.032504 | 0.561718 |
| highpass 0.02 | 42 | 27.223161 | 0.257944 | 0.032546 | 0.561655 |

`lambda=0.01` was selected for 10k validation because it was the only 1k setting with both a small PSNR gain and a clear LPIPS gain. `lambda=0.02` improved LPIPS in the short probe but hurt PSNR/L1.

## 10k Validation

Output root: `outputs/preproduction/ideas/ours/frequency_edge/idea_frequency_edge_20260611_10k`

| Seed | Variant | PSNR | LPIPS | L1 | SSIM | Gaussians |
|---:|---|---:|---:|---:|---:|---:|
| 42 | baseline | 28.266765 | 0.192777 | 0.027523 | 0.590892 | 697409 |
| 42 | highpass 0.01 | 28.268541 | 0.192321 | 0.027511 | 0.591065 | 697276 |
| 7 | baseline | 28.268522 | 0.193219 | 0.027516 | 0.590993 | 697238 |
| 7 | highpass 0.01 | 28.269102 | 0.193056 | 0.027514 | 0.590990 | 697836 |
| 123 | baseline | 28.261655 | 0.193039 | 0.027532 | 0.590731 | 697632 |
| 123 | highpass 0.01 | 28.266178 | 0.193180 | 0.027514 | 0.590823 | 697177 |

Mean deltas for highpass 0.01 minus baseline:

| Metric | Mean delta | Direction | Seeds improved |
|---|---:|---|---:|
| PSNR | +0.002293 | better | 3/3 |
| LPIPS | -0.000159 | better | 2/3 |
| L1 | -0.0000106 | better | 3/3 |
| SSIM | +0.0000873 | better | 2/3 |
| Gaussians | +3.3 | neutral | n/a |

The `lambda=0.02` 10k check on seed 42 was not selected further: PSNR/L1/SSIM were slightly positive, but LPIPS worsened to `0.193335` versus baseline `0.192777`.

## Decision

This idea is provisionally viable but weak. It is not a standalone journal-level contribution by itself, because the absolute metric gains are very small. It is worth keeping as a low-risk auxiliary loss for the final method stack because:

- It improves PSNR and L1 on all three tested seeds.
- It improves the mean of all four tracked metrics.
- It does not materially change the Gaussian count, so the gain is not coming from an obvious density-control side effect.
- It is conceptually aligned with ultrasound speckle/detail preservation and can be described as frequency-aware residual supervision for offline one-sweep reconstruction.

Recommended next use: keep `highpass_l1` default-off in base loss config, use `preproduction/ideas/ours/frequency_edge/frequency_edge.yaml` as the reproducible candidate, and only promote it into the main paper method if it remains positive when combined with other accepted ideas.

## Verification

- Unit tests: `tests/methods/ours/test_highpass_l1_loss.py`
- Dry run: Slurm job `62193`, 2 iterations on `gpu3`, confirmed `Loss Pipeline assembled: ['l1', 'ssim', 'scale_wall', 'highpass_l1']`
- 1k jobs: `62196`, `62197`, `62199`, `62200`
- 10k jobs: `62201`, `62202`, `62203`, `62204`, `62205`, `62206`, `62207`
