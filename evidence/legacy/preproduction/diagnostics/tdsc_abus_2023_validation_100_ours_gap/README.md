# TDSc validation_100 Ours Gap Diagnosis

Inputs: production Stage 1 GS runs plus best preproduction Ours variant `ours_b8_no_intprune_slice_depth_b080_s055_min035`.

## Mean Metrics

| method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD |
|---|---:|---:|---:|---:|---:|---:|
| gaussianpile | 28.3473 | 0.02527 | 0.78779 | 0.23257 | 0.18675 | 0.06782 |
| slice_adapted_3dgs | 28.1656 | 0.02477 | 0.79765 | 0.22612 | 0.19694 | 0.05961 |
| ultragray | 27.9915 | 0.02558 | 0.78571 | 0.23207 | 0.20898 | 0.06570 |
| ours_production | 27.7509 | 0.02596 | 0.78622 | 0.25508 | 0.19534 | 0.06876 |
| ours_best | 27.8322 | 0.02577 | 0.78835 | 0.24955 | 0.19177 | 0.06714 |

## Key Gaps

- Ours-best minus gaussianpile: PSNR -0.5151, DISTS +0.00502.
- Ours-best minus slice_adapted_3dgs: PSNR -0.3334, DISTS -0.00517.
- Ours-best minus ultragray: PSNR -0.1593, DISTS -0.01721.

## Files

- `per_frame_gap.tsv`: full per-frame metric deltas.
- `worst_psnr_vs_gaussianpile.tsv`, `worst_psnr_vs_slice_adapted_3dgs.tsv`, `worst_psnr_vs_ultragray.tsv`: worst PSNR frames.
- `worst_dists_vs_*.tsv`: worst perceptual-DISTS frames.
- `tdsc_vis_gap_grid.png`: 10 saved visualization frames with GT/baselines/ours.
