# Spine Phantom Ours Production Preflight, 2026-06-12

This note records the exploratory/near-production Slurm runs for `spine_phantom` using the production Ours pipeline.

## Setup

- Dataset: `/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom`
- Sweeps tested: `left1`, `right1`
- Default scene config: `configs_hydra/scene/spine_phantom.yaml`
- Main experiment config: `configs_hydra/experiment/preproduction/dataset_preflight/spine_phantom/spine_phantom_preflight_10k.yaml`
- Data preprocessing: online axial downsample only, `1300x164 -> 650x164`, with `spine_phantom_shifted_float`
- Gaussian initialization: 1M unless stated otherwise
- Eval split: holdout interval, trim 10 frames from each end, 40 test frames
- GPU: `gpu3` RTX 4090 via Slurm

## 10k Preflight

Summary path:

`outputs/preproduction/ideas/ours/easy_pose/spine_phantom_production_preflight_20260612/summary.tsv`

| job | run | best step | best PSNR | best LPIPS | best SSIM | best DISTS | best GMSD | final PSNR | final LPIPS | final gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 62494 | left1 axial2 1M ref | 7500 | 31.2071 | 0.1624 | 0.7859 | 0.2144 | 0.1037 | 31.1717 | 0.1638 | 279410 |
| 62495 | right1 axial2 1M | 7500 | 30.8796 | 0.1575 | 0.7810 | 0.2093 | 0.1021 | 30.8415 | 0.1584 | 282701 |
| 62496 | left1 axial2 300k | 7500 | 31.0121 | 0.1923 | 0.7783 | 0.2312 | 0.1114 | 30.9700 | 0.1927 | 171873 |
| 62497 | left1 fullres 1M | 7500 | 30.0809 | 0.2721 | 0.7642 | 0.2779 | 0.1299 | 30.0590 | 0.2719 | 276005 |
| 62498 | left1 axial2 1M elev0.79 | 7500 | 31.2060 | 0.1629 | 0.7857 | 0.2139 | 0.1039 | 31.1730 | 0.1629 | 279473 |

Findings:

- `axial_downsample_factor=2` is the right default for current production work. Full resolution is much worse in this setup: best PSNR drops by about 1.13 dB and LPIPS worsens from about 0.162 to 0.272 on `left1`.
- 1M initialization is better than 300k even though density control prunes both to much smaller final sets. The 300k run loses about 0.20 dB PSNR and has much worse LPIPS/DISTS/GMSD.
- `elev_corridor_half_mm=0.79` is effectively tied with the default `0.395`; it does not provide a clear improvement. Keep the narrower/default `0.395` for production unless a later ablation targets this specifically.
- `left1` and `right1` are consistent enough for production use under this config.

## 30k Confirm

Summary path:

`outputs/preproduction/ideas/ours/easy_pose/spine_phantom_30k_confirm_20260612/summary.tsv`

| job | run | best step | best PSNR | best LPIPS | best SSIM | best DISTS | final step | final PSNR | final LPIPS | final SSIM | final DISTS | final gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 62499 | left1 axial2 1M 30k | 10000 | 31.2076 | 0.1603 | 0.7861 | 0.2094 | 30000 | 31.0132 | 0.1703 | 0.7830 | 0.2136 | 279870 |
| 62500 | right1 axial2 1M 30k | 5000 | 30.8779 | 0.1591 | 0.7822 | 0.2084 | 30000 | 30.7373 | 0.1644 | 0.7795 | 0.2089 | 283487 |

Per-step right1 note:

- `right1` best PSNR is at 5k, but LPIPS/DISTS improve through 10k-20k before later degrading. For PSNR-selected reporting, use the best checkpoint. For perceptual-only comparisons, inspect the selected metric explicitly rather than assuming the final checkpoint is best.

## Production Recommendation

- Use `spine_phantom` with online axial downsample factor 2, 1M initialization, and the default `elev_corridor_half_mm=0.395`.
- Do not use full-resolution `1300x164` for the current production sweep. It is slower and empirically worse here.
- Do not reduce initialization to 300k for production. Let density control prune the 1M initialization to about 280k final Gaussians.
- For this dataset, do not rely on the 30k final checkpoint. Run with periodic evaluation and select best checkpoint by the target metric. If the production comparison needs a fixed budget, 10k is a better default cap than 30k for PSNR/SSIM.
- Suggested production protocol for `spine_phantom`: train up to 10k or 15k, evaluate every 2.5k or 5k, and report best-by-PSNR with the corresponding LPIPS/DISTS/GMSD. If using a 30k budget for consistency with other datasets, still report best checkpoint, not final checkpoint.
