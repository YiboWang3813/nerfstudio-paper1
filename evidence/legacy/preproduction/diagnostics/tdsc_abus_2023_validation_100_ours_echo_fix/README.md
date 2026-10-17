# TDSc Ours Echo Saturation Fix Results

Best variant: `ours_b8_no_intprune_slice_depth_echo_init005_lr002_attlr010`.

| method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | final G | echo>=0.19 | atten>=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_b8_no_intprune_slice_depth_echo_init005_lr002_attlr010 | 28.0087 | 0.02533 | 0.79325 | 0.21771 | 0.17779 | 0.06379 | 1029228 | 0.097 | 0.068 |
| ours_b8_no_intprune_slice_depth_echo_init005_lr004_attlr010 | 27.9525 | 0.02545 | 0.79210 | 0.23092 | 0.18350 | 0.06517 | 1000058 | 0.201 | 0.064 |
| ours_b8_no_intprune_slice_depth_echo_init003_lr004_attlr010 | 27.9314 | 0.02548 | 0.79184 | 0.23385 | 0.18464 | 0.06535 | 999328 | 0.198 | 0.065 |
| ours_b8_no_intprune_slice_depth_echo_init005_lr004_attlr005 | 27.9255 | 0.02552 | 0.79180 | 0.23072 | 0.18438 | 0.06518 | 1000052 | 0.203 | 0.033 |
| old_best_slice_depth_b080_s055_min035 | 27.8322 | 0.02577 | 0.78835 | 0.24955 | 0.19177 | 0.06714 | 919337 | 0.302 | 0.065 |

## Interpretation

- Lowering `echo_alphas` LR from 0.04/0.08-class behavior to 0.02 gave the best result and reduced echo saturation while keeping attenuation enabled.
- The best fix improves PSNR by +0.1766 dB over the previous Ours best and also improves LPIPS/DISTS/GMSD.
- The result now beats UltraGray on this sweep, but still trails Slice-3DGS and GaussianPile in PSNR.
