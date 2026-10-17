# TUS-REC-2024 Ours Production Preflight, 2026-06-12

## Scope

This preflight validates the current production-path `ours_us_focus` setup on the
local TUS-REC-2024 export. The renderer lateral-offset fix is required for these
results; earlier black/low-coverage TUS runs before that fix are not comparable.

Data:

- Raw root: `/home/boywang25/ultrasound-scene-reconstruction/data/TUS_REC_2024`
- Export root: `/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024`
- Representative sweeps:
  - `010_LH_Par_C_DtP`
  - `010_RH_Per_S_PtD`
  - `011_LH_Par_C_DtP`

Configs:

- Minimal: `experiment=preproduction/dataset_preflight/tus_rec_2024/tus_rec_2024_validation_2k`
- 10k preflight: `experiment=preproduction/dataset_preflight/tus_rec_2024/tus_rec_2024_preflight_10k`
- 30k confirmation: `experiment=preproduction/dataset_preflight/tus_rec_2024/tus_rec_2024_confirm_30k`

Default candidate setting:

- Online image downsample: `scene.loading.image_downsample_factor=2`
- Initial gaussians: `1,000,000`
- Iterations: `30,000`
- Holdout: `holdout_interval`, trim each side `10`, samples `40`
- Elevational corridor: `model.slice_render.elev_corridor_half_mm=0.202`

## Minimal 2k Validation

Output root:
`outputs/preproduction/ideas/ours/easy_pose/tus_rec_2024_minimal_20260612`

| Sweep | Init | Iter | Downsample | PSNR | SSIM | LPIPS | L1 | Final Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `010_LH_Par_C_DtP` | 300k | 2k | 2x | 27.218 | 0.799 | 0.203 | 0.0263 | 241,569 |
| `010_RH_Per_S_PtD` | 300k | 2k | 2x | 31.177 | 0.863 | 0.203 | 0.0150 | 267,034 |
| `011_LH_Par_C_DtP` | 300k | 2k | 2x | 26.330 | 0.792 | 0.204 | 0.0269 | 234,340 |

Result: all three sweeps train and evaluate without black predictions or
coverage collapse. The cross-subject sweep is lower but still stable.

## 10k Preflight

Output root:
`outputs/preproduction/ideas/ours/easy_pose/tus_rec_2024_production_preflight_20260612`

| Case | Sweep | Init | Iter | Downsample | Elev Half | PSNR | SSIM | LPIPS | L1 | Final Gaussians |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | `010_LH_Par_C_DtP` | 1M | 10k | 2x | 0.202 | 29.165 | 0.822 | 0.169 | 0.0221 | 515,469 |
| Second sweep | `010_RH_Per_S_PtD` | 1M | 10k | 2x | 0.202 | 35.128 | 0.904 | 0.098 | 0.0097 | 695,042 |
| Capacity | `010_LH_Par_C_DtP` | 300k | 10k | 2x | 0.202 | 28.901 | 0.819 | 0.192 | 0.0227 | 213,807 |
| Elev corridor | `010_LH_Par_C_DtP` | 1M | 10k | 2x | 0.404 | 29.153 | 0.822 | 0.169 | 0.0221 | 515,945 |

Failed full-resolution controls:

- `010_LH_Par_C_DtP`, 1M, 10k, full-res: CUDA OOM during initialization sampling.
- `010_LH_Par_C_DtP`, 300k, 10k, full-res: CUDA OOM during initialization sampling.

Both full-resolution failures occurred after full-res SliceMeta precompute and
before gaussian injection, inside `UnifiedSampler._sample_importance`. GPU memory
was already around 22 GB on a 24 GB RTX 4090. Full-resolution full-sweep TUS is
therefore not a reliable production target for the current pipeline.

## 30k Confirmation

Output root:
`outputs/preproduction/ideas/ours/easy_pose/tus_rec_2024_30k_confirm_20260612`

| Sweep | Init | Downsample | Best Step | PSNR | SSIM | LPIPS | L1 | Final Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `010_LH_Par_C_DtP` | 1M | 2x | 30k | 29.553 | 0.828 | 0.165 | 0.0214 | 514,993 |
| `010_RH_Per_S_PtD` | 1M | 2x | 30k | 35.376 | 0.907 | 0.093 | 0.0094 | 696,527 |

Intermediate test sequence:

| Sweep | 5k PSNR | 10k PSNR | 15k PSNR | 20k PSNR | 25k PSNR | 30k PSNR |
|---|---:|---:|---:|---:|---:|---:|
| `010_LH_Par_C_DtP` | 28.829 | 29.213 | 29.276 | 29.368 | 29.517 | 29.553 |
| `010_RH_Per_S_PtD` | 34.772 | 35.175 | 35.288 | 35.363 | 35.307 | 35.376 |

Unlike the earlier spine phantom confirmation, TUS continues to benefit from
training beyond 10k. The gain from 10k to 30k is about `+0.39 dB` on the
reference sweep and `+0.25 dB` on the second sweep.

## Visual Check

Checked compose outputs from the two 30k confirmation runs:

- `outputs/preproduction/ideas/ours/easy_pose/tus_rec_2024_30k_confirm_20260612/010_lh_par_c_dtp_img2_1m_30k_ref_62526/vis/item_1/compose.png`
- `outputs/preproduction/ideas/ours/easy_pose/tus_rec_2024_30k_confirm_20260612/010_rh_per_s_ptd_img2_1m_30k_62527/vis/item_1/compose.png`

Both show aligned fan-shaped GT/pred regions. No black prediction, gross drift,
or obvious coordinate-frame mismatch was observed. Residuals are mostly speckle
and fine texture differences.

## Conclusion

TUS-REC-2024 is ready for production experiments with the current flat
image-coordinate calibration path, provided that the renderer lateral-offset fix
is included.

Recommended production config:

- `experiment=preproduction/dataset_preflight/tus_rec_2024/tus_rec_2024_confirm_30k`
- `scene.loading.image_downsample_factor=2`
- `model.gauss_init.num_points=1000000`
- `scene.sampler.num_init=1000000`
- `engine.iterations=30000`
- `model.slice_render.elev_corridor_half_mm=0.202`

Rationale:

- 2x downsample is stable across three sweeps and avoids full-resolution OOM.
- Full-resolution full-sweep training is not currently feasible on a 24 GB RTX
  4090, even with 300k initial gaussians.
- 1M gives modest but consistent improvement over 300k, especially LPIPS.
- `0.202` and `0.404` elevational corridor are effectively equivalent on the
  reference sweep, so keep the narrower pose-derived default.
- 30k improves over 10k on both confirmed sweeps, so TUS should not reuse the
  spine phantom 10k stopping rule.
