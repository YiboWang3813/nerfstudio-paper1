# P2 final Fixed refresh results

Main78 and LGR55slots are exact-config reuse. New structural runs use P2; reused rows preserve their original execution source hashes with a separate P2 equivalence certification. No Batch2 launched.

## Joint

| dataset | label | final_psnr | final_ssim | final_lpips | final_num_gaussians |
|---|---|---|---|---|---|
| synthetic_liver | L0C0P0 | 25.7467 | 0.616474 | 0.399011 | 202893 |
| synthetic_liver | L0C0P1 | 25.6076 | 0.617297 | 0.565421 | 253360 |
| synthetic_liver | L0C1P0 | 25.7949 | 0.615747 | 0.387236 | 569320 |
| synthetic_liver | L0C1P1 | 25.5954 | 0.616877 | 0.573077 | 648933 |
| synthetic_liver | L1C0P0 | 29.4488 | 0.660492 | 0.158422 | 628109 |
| synthetic_liver | L1C0P1 | 29.6483 | 0.667251 | 0.182677 | 606159 |
| synthetic_liver | L1C1P0 | 29.4522 | 0.660275 | 0.157044 | 688611 |
| synthetic_liver | L1C1P1 | 29.6484 | 0.667444 | 0.182353 | 672079 |
| tus_rec_2024_qc | L0C0P0 | 21.092 | 0.643446 | 0.349698 | 189307 |
| tus_rec_2024_qc | L0C0P1 | 24.1362 | 0.760127 | 0.32196 | 322552 |
| tus_rec_2024_qc | L0C1P0 | 21.3978 | 0.649532 | 0.33987 | 559193 |
| tus_rec_2024_qc | L0C1P1 | 24.1532 | 0.760131 | 0.321697 | 615874 |
| tus_rec_2024_qc | L1C0P0 | 32.785 | 0.86541 | 0.111759 | 679542 |
| tus_rec_2024_qc | L1C0P1 | 33.2628 | 0.871507 | 0.10804 | 654499 |
| tus_rec_2024_qc | L1C1P0 | 32.8167 | 0.865764 | 0.11204 | 713263 |
| tus_rec_2024_qc | L1C1P1 | 33.2709 | 0.871553 | 0.108021 | 682586 |


| dataset | module | metric | mean_effect | min_effect | max_effect |
|---|---|---|---|---|---|
| synthetic_liver | LGR | psnr | 3.8633 | 3.6573 | 4.0531 |
| synthetic_liver | LGR | ssim | 0.0472667 | 0.0440186 | 0.0505667 |
| synthetic_liver | LGR | lpips | -0.311062 | -0.390724 | -0.230192 |
| synthetic_liver | CGA | psnr | 0.00988892 | -0.0122611 | 0.0482361 |
| synthetic_liver | CGA | ssim | -0.000292946 | -0.000726744 | 0.000192702 |
| synthetic_liver | CGA | lpips | -0.00145517 | -0.0117754 | 0.00765637 |
| synthetic_liver | PPP | psnr | 0.01429 | -0.199548 | 0.199511 |
| synthetic_liver | PPP | ssim | 0.00397033 | 0.000823651 | 0.00716886 |
| synthetic_liver | PPP | lpips | 0.100454 | 0.0242556 | 0.185841 |
| tus_rec_2024_qc | LGR | psnr | 10.3391 | 9.11772 | 11.693 |
| tus_rec_2024_qc | LGR | ssim | 0.16525 | 0.11138 | 0.221964 |
| tus_rec_2024_qc | LGR | lpips | -0.223341 | -0.237939 | -0.213676 |
| tus_rec_2024_qc | CGA | psnr | 0.0906474 | 0.0081566 | 0.305723 |
| tus_rec_2024_qc | CGA | ssim | 0.00162242 | 4.229e-06 | 0.00608545 |
| tus_rec_2024_qc | CGA | lpips | -0.00245716 | -0.00982767 | 0.000280468 |
| tus_rec_2024_qc | PPP | psnr | 1.68287 | 0.454212 | 3.04411 |
| tus_rec_2024_qc | PPP | ssim | 0.0597917 | 0.0057886 | 0.116681 |
| tus_rec_2024_qc | PPP | lpips | -0.0134123 | -0.0277382 | -0.0037192 |


LGR off is Plane(no conditioning,no global), not the former Adaptive local-only setting. CGA off is gradient+opacity; no-global CGA on degrades to local-aware.

## PPP representative

| dataset | label | final_psnr | final_ssim | final_lpips | final_num_gaussians |
|---|---|---|---|---|---|
| synthetic_liver | MP | 29.4522 | 0.660275 | 0.157044 | 688611 |
| synthetic_liver | PP-S | 29.6415 | 0.66723 | 0.182063 | 671361 |
| synthetic_liver | PP-A | 29.6805 | 0.668584 | 0.184835 | 665249 |
| synthetic_liver | PPP | 29.6484 | 0.667444 | 0.182353 | 672079 |
| tus_rec_2024_qc | MP | 32.8167 | 0.865764 | 0.11204 | 713263 |
| tus_rec_2024_qc | PP-S | 33.033 | 0.868796 | 0.110448 | 710350 |
| tus_rec_2024_qc | PP-A | 32.7787 | 0.865395 | 0.126326 | 602565 |
| tus_rec_2024_qc | PPP | 33.2709 | 0.871553 | 0.108021 | 682586 |


## PPP multi-sweep

| dataset | label | n | PSNR_mean | PSNR_std | SSIM_mean | SSIM_std |
|---|---|---|---|---|---|---|
| synthetic_liver | MP | 6 | 29.0251 | 0.483665 | 0.635463 | 0.0244167 |
| synthetic_liver | PPP | 6 | 29.1963 | 0.485163 | 0.639873 | 0.0263214 |
| tdsc_abus_2023 | MP | 5 | 29.2089 | 1.12671 | 0.812052 | 0.0358189 |
| tdsc_abus_2023 | PPP | 5 | 29.2119 | 1.12969 | 0.812105 | 0.0358103 |
| tus_rec_2024_qc | MP | 5 | 34.125 | 0.873084 | 0.887679 | 0.0149723 |
| tus_rec_2024_qc | PPP | 5 | 34.6381 | 0.959594 | 0.892855 | 0.0151313 |


Only PSNR/SSIM are claimed here; no all-perceptual-invariance claim.

## Rho

| dataset | label | final_psnr | final_ssim | final_lpips | final_num_gaussians |
|---|---|---|---|---|---|
| synthetic_liver | 0.6 | 29.988 | 0.678577 | 0.218973 | 594292 |
| synthetic_liver | 0.75 | 29.8571 | 0.67483 | 0.200965 | 625602 |
| synthetic_liver | 0.9 | 29.7284 | 0.6701 | 0.188888 | 654682 |
| synthetic_liver | 1.0 | 29.6484 | 0.667444 | 0.182353 | 672079 |
| synthetic_liver | 1.1 | 29.566 | 0.664371 | 0.17716 | 688038 |
| synthetic_liver | 1.25 | 29.43 | 0.659704 | 0.171521 | 708619 |
| synthetic_liver | 1.5 | 29.1704 | 0.651123 | 0.165992 | 738932 |
| tus_rec_2024_qc | 0.6 | 32.9503 | 0.867364 | 0.121327 | 596413 |
| tus_rec_2024_qc | 0.75 | 33.072 | 0.86903 | 0.114789 | 635543 |
| tus_rec_2024_qc | 0.9 | 33.2066 | 0.870899 | 0.10963 | 665573 |
| tus_rec_2024_qc | 1.0 | 33.2709 | 0.871553 | 0.108021 | 682586 |
| tus_rec_2024_qc | 1.1 | 33.3105 | 0.871932 | 0.107028 | 697038 |
| tus_rec_2024_qc | 1.25 | 33.3399 | 0.872187 | 0.105319 | 717317 |
| tus_rec_2024_qc | 1.5 | 33.3051 | 0.8718 | 0.106287 | 740651 |


rho1 is fixed nominal spacing, not a test-selected optimum.

## CGA

| dataset | label | final_G | final_G_k | total_spawned | total_pruned | path_protected_candidate_decisions | PSNR | LPIPS |
|---|---|---|---|---|---|---|---|---|
| synthetic_liver | GD | 628109 | 628.109 | 588 | 372479 | 0 | 29.4488 | 0.158422 |
| synthetic_liver | RGD | 1.05067e+06 | 1050.67 | 50672 | 0 | 0 | 29.4344 | 0.15681 |
| synthetic_liver | RGD+OP | 643294 | 643.294 | 45624 | 402330 | 0 | 29.4432 | 0.158375 |
| synthetic_liver | RGD+LP | 644939 | 644.939 | 45614 | 400675 | 0 | 29.4541 | 0.157542 |
| synthetic_liver | Full | 688611 | 688.611 | 46496 | 357885 | 234708 | 29.4522 | 0.157044 |
| tus_rec_2024_qc | GD | 679542 | 679.542 | 0 | 320458 | 0 | 32.785 | 0.111759 |
| tus_rec_2024_qc | RGD | 1.02259e+06 | 1022.59 | 22586 | 0 | 0 | 32.7085 | 0.112262 |
| tus_rec_2024_qc | RGD+OP | 689866 | 689.866 | 21086 | 331220 | 0 | 32.773 | 0.112035 |
| tus_rec_2024_qc | RGD+LP | 690845 | 690.845 | 21026 | 330181 | 0 | 32.7781 | 0.111338 |
| tus_rec_2024_qc | Full | 713263 | 713.263 | 21156 | 307893 | 119549 | 32.8167 | 0.11204 |


| dataset | extra_final_G | delta_PSNR | delta_LPIPS |
|---|---|---|---|
| synthetic_liver | 43672 | -0.0018494 | -0.000497626 |
| tus_rec_2024_qc | 22418 | 0.0386384 | 0.000701491 |


Protected decisions are event-summed, not unique Gaussian identities.

## LGR

| dataset | label | n | psnr | ssim | lpips |
|---|---|---|---|---|---|
| synthetic_liver | Plane | 6 | 25.0916 | 0.59964 | 0.427217 |
| synthetic_liver | Fixed | 6 | 29.1606 | 0.642769 | 0.169183 |
| synthetic_liver | Adaptive | 6 | 29.3386 | 0.653032 | 0.195312 |
| synthetic_liver | Fixed+Simple | 6 | 29.0572 | 0.635632 | 0.156491 |
| synthetic_liver | Fixed+Full | 6 | 29.0251 | 0.635463 | 0.15669 |
| tus_rec_2024_qc | Plane | 5 | 21.4376 | 0.676345 | 0.343759 |
| tus_rec_2024_qc | Fixed | 5 | 33.6737 | 0.884308 | 0.125563 |
| tus_rec_2024_qc | Adaptive | 5 | 33.4112 | 0.879569 | 0.136359 |
| tus_rec_2024_qc | Fixed+Simple | 5 | 34.0237 | 0.887108 | 0.112989 |
| tus_rec_2024_qc | Fixed+Full | 5 | 34.125 | 0.887679 | 0.112477 |


Plane→Fixed tests finite slice. Fixed/Adaptive compares alternatives, not progressive modules. Fixed→Simple→Full tests global formulations; no monotonicity or independent acoustic-attribute claim.

All raw data, configs,source lineage and exported images are preserved. Await scientific review. No automatic Hyperparameter Batch2.
