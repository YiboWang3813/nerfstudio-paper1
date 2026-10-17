# LGR 11-case structural confirmation

55 slots: 19 exact-config reuse + 36 newly trained; MP, seed42, 1M, 30k, same frozen split. No hyperparameter experiments.

No-global variants use local-aware CGA fallback; all other training settings unchanged. Differences are whole-system trained contrasts, not isolated fixed-topology renderer effects.

Means and sample standard deviations (ddof=1) are sweep-wise. W/T/L is per metric, direction-aware, with numerical tie tolerance 1e-6; wins are NOT significance or clinical/practical effect claims. Single seed; Arm segments are not independent subjects.

## Variant means

| dataset | variant | metric | n | mean | std |
|---|---|---|---|---|---|
| synthetic_liver | Plane | psnr | 6 | 25.0916 | 1.16846 |
| synthetic_liver | Plane | ssim | 6 | 0.59964 | 0.0242603 |
| synthetic_liver | Plane | lpips | 6 | 0.427217 | 0.0591366 |
| synthetic_liver | Fixed | psnr | 6 | 29.1606 | 0.479984 |
| synthetic_liver | Fixed | ssim | 6 | 0.642769 | 0.0236503 |
| synthetic_liver | Fixed | lpips | 6 | 0.169183 | 0.00349061 |
| synthetic_liver | Adaptive | psnr | 6 | 29.3386 | 0.484782 |
| synthetic_liver | Adaptive | ssim | 6 | 0.653032 | 0.0228569 |
| synthetic_liver | Adaptive | lpips | 6 | 0.195312 | 0.00469069 |
| synthetic_liver | Simple | psnr | 6 | 29.26 | 0.484146 |
| synthetic_liver | Simple | ssim | 6 | 0.648613 | 0.0230784 |
| synthetic_liver | Simple | lpips | 6 | 0.179733 | 0.00489897 |
| synthetic_liver | Full | psnr | 6 | 29.2239 | 0.479694 |
| synthetic_liver | Full | ssim | 6 | 0.648545 | 0.0230896 |
| synthetic_liver | Full | lpips | 6 | 0.179857 | 0.00499061 |
| tus_rec_2024_qc | Plane | psnr | 5 | 21.4376 | 0.637861 |
| tus_rec_2024_qc | Plane | ssim | 5 | 0.676345 | 0.0287725 |
| tus_rec_2024_qc | Plane | lpips | 5 | 0.343759 | 0.0228315 |
| tus_rec_2024_qc | Fixed | psnr | 5 | 33.6737 | 0.819511 |
| tus_rec_2024_qc | Fixed | ssim | 5 | 0.884308 | 0.0145517 |
| tus_rec_2024_qc | Fixed | lpips | 5 | 0.125563 | 0.00669602 |
| tus_rec_2024_qc | Adaptive | psnr | 5 | 33.4112 | 0.742098 |
| tus_rec_2024_qc | Adaptive | ssim | 5 | 0.879569 | 0.0138185 |
| tus_rec_2024_qc | Adaptive | lpips | 5 | 0.136359 | 0.00641798 |
| tus_rec_2024_qc | Simple | psnr | 5 | 33.6759 | 0.794595 |
| tus_rec_2024_qc | Simple | ssim | 5 | 0.882107 | 0.0142693 |
| tus_rec_2024_qc | Simple | lpips | 5 | 0.123543 | 0.00503405 |
| tus_rec_2024_qc | Full | psnr | 5 | 33.7693 | 0.787297 |
| tus_rec_2024_qc | Full | ssim | 5 | 0.882733 | 0.0141593 |
| tus_rec_2024_qc | Full | lpips | 5 | 0.123012 | 0.00462638 |


## Paired transitions

| dataset | before | after | metric | mean_delta | std_paired_delta | wins | ties | losses | min_delta | max_delta |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_liver | Plane | Fixed | psnr | 4.06904 | 0.757888 | 6 | 0 | 0 | 3.16376 | 4.91126 |
| synthetic_liver | Plane | Fixed | ssim | 0.043129 | 0.00549575 | 6 | 0 | 0 | 0.0355195 | 0.0512917 |
| synthetic_liver | Plane | Fixed | lpips | -0.258034 | 0.0603646 | 6 | 0 | 0 | -0.320798 | -0.170569 |
| synthetic_liver | Fixed | Adaptive | psnr | 0.177949 | 0.0123889 | 6 | 0 | 0 | 0.167647 | 0.201636 |
| synthetic_liver | Fixed | Adaptive | ssim | 0.0102622 | 0.000985381 | 6 | 0 | 0 | 0.00923015 | 0.011574 |
| synthetic_liver | Fixed | Adaptive | lpips | 0.0261285 | 0.00167835 | 0 | 0 | 6 | 0.0242658 | 0.0287664 |
| synthetic_liver | Adaptive | Simple | psnr | -0.0785983 | 0.00985732 | 0 | 0 | 6 | -0.0900238 | -0.0634711 |
| synthetic_liver | Adaptive | Simple | ssim | -0.00441853 | 0.000439162 | 0 | 0 | 6 | -0.00483905 | -0.00364813 |
| synthetic_liver | Adaptive | Simple | lpips | -0.0155784 | 0.00122043 | 6 | 0 | 0 | -0.0173202 | -0.0141249 |
| synthetic_liver | Simple | Full | psnr | -0.0360729 | 0.0131863 | 0 | 0 | 6 | -0.0577894 | -0.0228356 |
| synthetic_liver | Simple | Full | ssim | -6.75518e-05 | 0.000141727 | 1 | 0 | 5 | -0.000224627 | 0.000174164 |
| synthetic_liver | Simple | Full | lpips | 0.000123774 | 0.000445411 | 2 | 0 | 4 | -0.000469126 | 0.000770851 |
| tus_rec_2024_qc | Plane | Fixed | psnr | 12.2362 | 0.784394 | 5 | 0 | 0 | 11.1201 | 13.1848 |
| tus_rec_2024_qc | Plane | Fixed | ssim | 0.207963 | 0.0159444 | 5 | 0 | 0 | 0.181316 | 0.222418 |
| tus_rec_2024_qc | Plane | Fixed | lpips | -0.218196 | 0.0166024 | 5 | 0 | 0 | -0.24463 | -0.199047 |
| tus_rec_2024_qc | Fixed | Adaptive | psnr | -0.262558 | 0.0891353 | 0 | 0 | 5 | -0.385812 | -0.157163 |
| tus_rec_2024_qc | Fixed | Adaptive | ssim | -0.00473911 | 0.00096243 | 0 | 0 | 5 | -0.00570718 | -0.00349671 |
| tus_rec_2024_qc | Fixed | Adaptive | lpips | 0.0107958 | 0.00127199 | 0 | 0 | 5 | 0.00974232 | 0.0129888 |
| tus_rec_2024_qc | Adaptive | Simple | psnr | 0.264724 | 0.103787 | 5 | 0 | 0 | 0.107057 | 0.378556 |
| tus_rec_2024_qc | Adaptive | Simple | ssim | 0.00253864 | 0.00109092 | 5 | 0 | 0 | 0.00101808 | 0.00361179 |
| tus_rec_2024_qc | Adaptive | Simple | lpips | -0.012816 | 0.00266291 | 5 | 0 | 0 | -0.0164409 | -0.00969409 |
| tus_rec_2024_qc | Simple | Full | psnr | 0.0934159 | 0.0299443 | 5 | 0 | 0 | 0.0489883 | 0.127394 |
| tus_rec_2024_qc | Simple | Full | ssim | 0.000625587 | 0.000224193 | 5 | 0 | 0 | 0.000364607 | 0.000876331 |
| tus_rec_2024_qc | Simple | Full | lpips | -0.000530779 | 0.000516286 | 4 | 0 | 1 | -0.00112376 | 0.000138881 |


See per_case.tsv and paired_deltas.tsv for every individual case. No default retuning or Batch2 launch is authorized by these results.