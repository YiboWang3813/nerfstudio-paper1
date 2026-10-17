# TDSc Ours Echo LR Refine Results

Best PSNR variant: `ours_b8_no_intprune_slice_depth_echo_init005_lr0015`. Best DISTS variant: `ours_b8_no_intprune_slice_depth_echo_init005_lr0010`.

| method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | final G | echo>=0.19 | atten>=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_b8_no_intprune_slice_depth_echo_init005_lr0015 | 28.0200 | 0.02525 | 0.79331 | 0.21378 | 0.17498 | 0.06348 | 1030412 | 0.065 | 0.070 |
| ours_b8_no_intprune_slice_depth_echo_init005_lr0010 | 28.0179 | 0.02516 | 0.79339 | 0.21207 | 0.17276 | 0.06312 | 1031524 | 0.037 | 0.073 |
| ours_b8_no_intprune_slice_depth_echo_init005_lr0020 | 28.0157 | 0.02531 | 0.79322 | 0.21844 | 0.17702 | 0.06391 | 1029225 | 0.097 | 0.068 |
| ours_b8_no_intprune_slice_depth_echo_init005_lr0025 | 27.9929 | 0.02538 | 0.79293 | 0.22067 | 0.17929 | 0.06438 | 1026915 | 0.130 | 0.066 |

## Decision

- `echo_lr=0.015` gives the highest PSNR, but `echo_lr=0.01` gives the best L1/LPIPS/DISTS/GMSD and is only 0.0020 dB lower in PSNR.
- Both keep echo saturation around 5-6%, much lower than the previous 30% saturation failure mode.
- For production ranking by PSNR, select `echo_lr=0.015`; for perceptual robustness, prefer `echo_lr=0.01`.
