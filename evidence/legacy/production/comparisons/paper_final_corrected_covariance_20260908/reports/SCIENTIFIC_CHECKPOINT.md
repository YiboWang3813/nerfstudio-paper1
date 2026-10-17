# Batch 1 scientific checkpoint

All quality numbers below are corrected-covariance final30k, seed42. Baselines are reused on identical case/frame IDs. Legacy Ours is impact-reference only. Single-seed contrasts are not significance tests. Batch2 is NOT authorized.

## Main comparison

| dataset | method | n | final_psnr | final_l1 | final_ssim | final_lpips | final_dists | final_gmsd | train_wall_time_sec |
|---|---|---|---|---|---|---|---|---|---|
| spine_phantom | ours_corrected | 9 | 30.8063 | 0.0144627 | 0.784377 | 0.145393 | 0.1656 | 0.104891 | 578.812 |
| spine_phantom | gaussianpile | 9 | 28.2243 | 0.0197013 | 0.716831 | 0.178897 | 0.215297 | 0.128603 | 384.333 |
| spine_phantom | nerfus_no_prior | 9 | 27.4567 | 0.018983 | 0.69206 | 0.555071 | 0.35868 | 0.226776 | 422.444 |
| spine_phantom | ours_legacy | 9 | 31.0188 | 0.0139841 | 0.789836 | 0.170764 | 0.2145 | 0.110085 | 358.222 |
| spine_phantom | slice_adapted_3dgs | 9 | 27.5968 | 0.0215108 | 0.712966 | 0.26723 | 0.255464 | 0.162895 | 372.222 |
| spine_phantom | ultragauss | 9 | 28.2479 | 0.0209699 | 0.626396 | 0.227204 | 0.198358 | 0.148534 | 172.556 |
| spine_phantom | ultragray | 9 | 30.6262 | 0.0157092 | 0.755087 | 0.131455 | 0.130089 | 0.10859 | 160.444 |
| spine_phantom | ultrags_baseline | 9 | 30.4708 | 0.0155814 | 0.736993 | 0.186837 | 0.233144 | 0.11946 | 340.111 |
| spine_phantom | ultranerf | 9 | 27.1349 | 0.0194934 | 0.684494 | 0.510281 | 0.331565 | 0.224143 | 403.333 |
| synthetic_liver | ours_corrected | 6 | 29.2239 | 0.0244273 | 0.648545 | 0.179857 | 0.150655 | 0.100859 | 545.522 |
| synthetic_liver | gaussianpile | 6 | 25.2303 | 0.0390457 | 0.4684 | 0.331105 | 0.229253 | 0.165321 | 448 |
| synthetic_liver | nerfus_no_prior | 6 | 22.0717 | 0.0416466 | 0.597131 | 0.724642 | 0.487021 | 0.207481 | 245.833 |
| synthetic_liver | ours_legacy | 6 | 29.348 | 0.0241051 | 0.655221 | 0.237876 | 0.187581 | 0.100575 | 351 |
| synthetic_liver | slice_adapted_3dgs | 6 | 26.7288 | 0.0318704 | 0.572487 | 0.278623 | 0.202416 | 0.127686 | 475.333 |
| synthetic_liver | ultragauss | 6 | 27.56 | 0.0298699 | 0.567031 | 0.171042 | 0.141816 | 0.112513 | 167.667 |
| synthetic_liver | ultragray | 6 | 27.658 | 0.0294396 | 0.56352 | 0.153566 | 0.125603 | 0.108583 | 247 |
| synthetic_liver | ultrags_baseline | 6 | 28.0158 | 0.0292361 | 0.593675 | 0.237926 | 0.200874 | 0.102043 | 456.5 |
| synthetic_liver | ultranerf | 6 | 21.6785 | 0.0639181 | 0.564779 | 0.452876 | 0.282829 | 0.116274 | 546.833 |
| tdsc_abus_2023 | ours_corrected | 30 | 28.7395 | 0.0236351 | 0.804734 | 0.176347 | 0.156934 | 0.0673528 | 410.98 |
| tdsc_abus_2023 | gaussianpile | 30 | 26.8541 | 0.0294953 | 0.748596 | 0.326235 | 0.215707 | 0.102919 |  |
| tdsc_abus_2023 | nerfus_no_prior | 30 | 22.0047 | 0.0471172 | 0.650348 | 0.562959 | 0.27624 | 0.152863 | 471.167 |
| tdsc_abus_2023 | ours_legacy | 30 | 28.4809 | 0.024037 | 0.801281 | 0.230551 | 0.180306 | 0.067303 | 320.333 |
| tdsc_abus_2023 | slice_adapted_3dgs | 30 | 27.0838 | 0.0274498 | 0.792647 | 0.241425 | 0.199547 | 0.0741601 | 448.8 |
| tdsc_abus_2023 | ultragauss | 30 | 27.7615 | 0.025172 | 0.781601 | 0.190263 | 0.180197 | 0.0764391 | 166.967 |
| tdsc_abus_2023 | ultragray | 30 | 27.1599 | 0.0306766 | 0.734418 | 0.272265 | 0.210562 | 0.103496 | 163.2 |
| tdsc_abus_2023 | ultrags_baseline | 30 | 26.2911 | 0.033261 | 0.72029 | 0.351318 | 0.237847 | 0.103136 | 425.067 |
| tdsc_abus_2023 | ultranerf | 30 | 19.4499 | 0.0757878 | 0.568706 | 0.569151 | 0.290029 | 0.162019 | 447.633 |
| tus_rec_2024_qc | ours_corrected | 33 | 33.7694 | 0.0113981 | 0.882168 | 0.111152 | 0.133941 | 0.0473666 | 487.034 |
| tus_rec_2024_qc | gaussianpile | 33 | 32.6258 | 0.0139345 | 0.853255 | 0.153016 | 0.142129 | 0.0615174 | 468.636 |
| tus_rec_2024_qc | nerfus_no_prior | 33 | 29.4267 | 0.0167516 | 0.836057 | 0.347558 | 0.288523 | 0.0847665 | 477.061 |
| tus_rec_2024_qc | ours_legacy | 33 | 34.2494 | 0.0109167 | 0.890305 | 0.129638 | 0.150582 | 0.045149 | 354.818 |
| tus_rec_2024_qc | slice_adapted_3dgs | 33 | 33.9551 | 0.0115227 | 0.889401 | 0.109049 | 0.135241 | 0.045406 | 470.394 |
| tus_rec_2024_qc | ultragauss | 33 | 31.8598 | 0.0152183 | 0.802228 | 0.200963 | 0.19065 | 0.0676428 | 175.576 |
| tus_rec_2024_qc | ultragray | 33 | 33.8162 | 0.0129626 | 0.832439 | 0.132098 | 0.138721 | 0.0576403 | 218.091 |
| tus_rec_2024_qc | ultrags_baseline | 33 | 31.4713 | 0.0196829 | 0.79552 | 0.199425 | 0.183088 | 0.0731133 | 455.636 |
| tus_rec_2024_qc | ultranerf | 33 | 26.3681 | 0.0251281 | 0.795413 | 0.273882 | 0.248937 | 0.101818 | 453.788 |

| dataset | metric | best | second | corrected_rank |
|---|---|---|---|---|
| spine_phantom | psnr | ours_corrected | ultragray | 1 |
| spine_phantom | l1 | ours_corrected | ultrags_baseline | 1 |
| spine_phantom | ssim | ours_corrected | ultragray | 1 |
| spine_phantom | lpips | ultragray | ours_corrected | 2 |
| spine_phantom | dists | ultragray | ours_corrected | 2 |
| spine_phantom | gmsd | ours_corrected | ultragray | 1 |
| synthetic_liver | psnr | ours_corrected | ultrags_baseline | 1 |
| synthetic_liver | l1 | ours_corrected | ultrags_baseline | 1 |
| synthetic_liver | ssim | ours_corrected | nerfus_no_prior | 1 |
| synthetic_liver | lpips | ultragray | ultragauss | 3 |
| synthetic_liver | dists | ultragray | ultragauss | 3 |
| synthetic_liver | gmsd | ours_corrected | ultrags_baseline | 1 |
| tdsc_abus_2023 | psnr | ours_corrected | ultragauss | 1 |
| tdsc_abus_2023 | l1 | ours_corrected | ultragauss | 1 |
| tdsc_abus_2023 | ssim | ours_corrected | slice_adapted_3dgs | 1 |
| tdsc_abus_2023 | lpips | ours_corrected | ultragauss | 1 |
| tdsc_abus_2023 | dists | ours_corrected | ultragauss | 1 |
| tdsc_abus_2023 | gmsd | ours_corrected | slice_adapted_3dgs | 1 |
| tus_rec_2024_qc | psnr | slice_adapted_3dgs | ultragray | 3 |
| tus_rec_2024_qc | l1 | ours_corrected | slice_adapted_3dgs | 1 |
| tus_rec_2024_qc | ssim | slice_adapted_3dgs | ours_corrected | 2 |
| tus_rec_2024_qc | lpips | slice_adapted_3dgs | ours_corrected | 2 |
| tus_rec_2024_qc | dists | ours_corrected | slice_adapted_3dgs | 1 |
| tus_rec_2024_qc | gmsd | slice_adapted_3dgs | ours_corrected | 2 |

## Corrected minus legacy Ours

| dataset | psnr | l1 | ssim | lpips | dists | gmsd |
|---|---|---|---|---|---|---|
| spine_phantom | -0.212441 | 0.000478613 | -0.00545905 | -0.0253709 | -0.0488991 | -0.00519391 |
| synthetic_liver | -0.124092 | 0.000322188 | -0.00667514 | -0.0580188 | -0.0369257 | 0.000284422 |
| tdsc_abus_2023 | 0.258559 | -0.000401907 | 0.00345226 | -0.0542033 | -0.0233713 | 4.97421e-05 |
| tus_rec_2024_qc | -0.480001 | 0.000481364 | -0.00813638 | -0.0184866 | -0.0166404 | 0.00221755 |

These deltas compare the old published main-table row, not an isolated covariance intervention. The newer accumulated-A_g legacy main configs match the corrected configs and are separately compared in main_matched_legacy_per_run_deltas.tsv. Neither historical binary is treated as strictly reproducible; no causal/statistical significance claim is made.

## Numerical-claim audit

All old exact Ours values/percentages require replacement even when the direction survives. The following table recomputes every dataset/category/metric contrast against its strongest baseline. Best-PSNR/L1/SSIM and GMSD/perceptual claims must follow the explicit ranks above, not the old narrative.

| dataset | comparison | metric | strongest_baseline | baseline | corrected | delta | relative_reduction_pct | directional_advantage_supported |
|---|---|---|---|---|---|---|---|---|
| spine_phantom | NeRF | psnr | nerfus_no_prior | 27.4567 | 30.8063 | 3.34966 |  | True |
| spine_phantom | NeRF | l1 | nerfus_no_prior | 0.018983 | 0.0144627 | -0.00452033 | 23.8125 | True |
| spine_phantom | NeRF | ssim | nerfus_no_prior | 0.69206 | 0.784377 | 0.0923168 |  | True |
| spine_phantom | NeRF | lpips | ultranerf | 0.510281 | 0.145393 | -0.364888 | 71.5072 | True |
| spine_phantom | NeRF | dists | ultranerf | 0.331565 | 0.1656 | -0.165964 | 50.0549 | True |
| spine_phantom | NeRF | gmsd | ultranerf | 0.224143 | 0.104891 | -0.119252 | 53.2035 | True |
| spine_phantom | ultrasound_Gaussian | psnr | ultragray | 30.6262 | 30.8063 | 0.180136 |  | True |
| spine_phantom | ultrasound_Gaussian | l1 | ultrags_baseline | 0.0155814 | 0.0144627 | -0.00111876 | 7.18006 | True |
| spine_phantom | ultrasound_Gaussian | ssim | ultragray | 0.755087 | 0.784377 | 0.0292892 |  | True |
| spine_phantom | ultrasound_Gaussian | lpips | ultragray | 0.131455 | 0.145393 | 0.0139385 | -10.6033 | False |
| spine_phantom | ultrasound_Gaussian | dists | ultragray | 0.130089 | 0.1656 | 0.0355116 | -27.298 | False |
| spine_phantom | ultrasound_Gaussian | gmsd | ultragray | 0.10859 | 0.104891 | -0.00369848 | 3.40592 | True |
| spine_phantom | adapted_Gaussian | psnr | gaussianpile | 28.2243 | 30.8063 | 2.58205 |  | True |
| spine_phantom | adapted_Gaussian | l1 | gaussianpile | 0.0197013 | 0.0144627 | -0.0052386 | 26.5902 | True |
| spine_phantom | adapted_Gaussian | ssim | gaussianpile | 0.716831 | 0.784377 | 0.0675452 |  | True |
| spine_phantom | adapted_Gaussian | lpips | gaussianpile | 0.178897 | 0.145393 | -0.0335032 | 18.7277 | True |
| spine_phantom | adapted_Gaussian | dists | gaussianpile | 0.215297 | 0.1656 | -0.0496967 | 23.0828 | True |
| spine_phantom | adapted_Gaussian | gmsd | gaussianpile | 0.128603 | 0.104891 | -0.0237121 | 18.4382 | True |
| synthetic_liver | NeRF | psnr | nerfus_no_prior | 22.0717 | 29.2239 | 7.15221 |  | True |
| synthetic_liver | NeRF | l1 | nerfus_no_prior | 0.0416466 | 0.0244273 | -0.0172192 | 41.3461 | True |
| synthetic_liver | NeRF | ssim | nerfus_no_prior | 0.597131 | 0.648545 | 0.0514146 |  | True |
| synthetic_liver | NeRF | lpips | ultranerf | 0.452876 | 0.179857 | -0.273018 | 60.2855 | True |
| synthetic_liver | NeRF | dists | ultranerf | 0.282829 | 0.150655 | -0.132174 | 46.7328 | True |
| synthetic_liver | NeRF | gmsd | ultranerf | 0.116274 | 0.100859 | -0.0154142 | 13.2568 | True |
| synthetic_liver | ultrasound_Gaussian | psnr | ultrags_baseline | 28.0158 | 29.2239 | 1.20813 |  | True |
| synthetic_liver | ultrasound_Gaussian | l1 | ultrags_baseline | 0.0292361 | 0.0244273 | -0.00480881 | 16.4482 | True |
| synthetic_liver | ultrasound_Gaussian | ssim | ultrags_baseline | 0.593675 | 0.648545 | 0.0548709 |  | True |
| synthetic_liver | ultrasound_Gaussian | lpips | ultragray | 0.153566 | 0.179857 | 0.026291 | -17.1203 | False |
| synthetic_liver | ultrasound_Gaussian | dists | ultragray | 0.125603 | 0.150655 | 0.0250527 | -19.946 | False |
| synthetic_liver | ultrasound_Gaussian | gmsd | ultrags_baseline | 0.102043 | 0.100859 | -0.00118347 | 1.15977 | True |
| synthetic_liver | adapted_Gaussian | psnr | slice_adapted_3dgs | 26.7288 | 29.2239 | 2.49507 |  | True |
| synthetic_liver | adapted_Gaussian | l1 | slice_adapted_3dgs | 0.0318704 | 0.0244273 | -0.00744304 | 23.3541 | True |
| synthetic_liver | adapted_Gaussian | ssim | slice_adapted_3dgs | 0.572487 | 0.648545 | 0.0760586 |  | True |
| synthetic_liver | adapted_Gaussian | lpips | slice_adapted_3dgs | 0.278623 | 0.179857 | -0.098766 | 35.4479 | True |
| synthetic_liver | adapted_Gaussian | dists | slice_adapted_3dgs | 0.202416 | 0.150655 | -0.0517612 | 25.5716 | True |
| synthetic_liver | adapted_Gaussian | gmsd | slice_adapted_3dgs | 0.127686 | 0.100859 | -0.0268271 | 21.0101 | True |
| tdsc_abus_2023 | NeRF | psnr | nerfus_no_prior | 22.0047 | 28.7395 | 6.73485 |  | True |
| tdsc_abus_2023 | NeRF | l1 | nerfus_no_prior | 0.0471172 | 0.0236351 | -0.0234821 | 49.8376 | True |
| tdsc_abus_2023 | NeRF | ssim | nerfus_no_prior | 0.650348 | 0.804734 | 0.154385 |  | True |
| tdsc_abus_2023 | NeRF | lpips | nerfus_no_prior | 0.562959 | 0.176347 | -0.386611 | 68.6749 | True |
| tdsc_abus_2023 | NeRF | dists | nerfus_no_prior | 0.27624 | 0.156934 | -0.119306 | 43.1892 | True |
| tdsc_abus_2023 | NeRF | gmsd | nerfus_no_prior | 0.152863 | 0.0673528 | -0.0855098 | 55.939 | True |
| tdsc_abus_2023 | ultrasound_Gaussian | psnr | ultragauss | 27.7615 | 28.7395 | 0.977994 |  | True |
| tdsc_abus_2023 | ultrasound_Gaussian | l1 | ultragauss | 0.025172 | 0.0236351 | -0.00153689 | 6.10556 | True |
| tdsc_abus_2023 | ultrasound_Gaussian | ssim | ultragauss | 0.781601 | 0.804734 | 0.0231325 |  | True |
| tdsc_abus_2023 | ultrasound_Gaussian | lpips | ultragauss | 0.190263 | 0.176347 | -0.0139154 | 7.31379 | True |
| tdsc_abus_2023 | ultrasound_Gaussian | dists | ultragauss | 0.180197 | 0.156934 | -0.0232624 | 12.9094 | True |
| tdsc_abus_2023 | ultrasound_Gaussian | gmsd | ultragauss | 0.0764391 | 0.0673528 | -0.00908628 | 11.887 | True |
| tdsc_abus_2023 | adapted_Gaussian | psnr | slice_adapted_3dgs | 27.0838 | 28.7395 | 1.65575 |  | True |
| tdsc_abus_2023 | adapted_Gaussian | l1 | slice_adapted_3dgs | 0.0274498 | 0.0236351 | -0.00381471 | 13.897 | True |
| tdsc_abus_2023 | adapted_Gaussian | ssim | slice_adapted_3dgs | 0.792647 | 0.804734 | 0.0120868 |  | True |
| tdsc_abus_2023 | adapted_Gaussian | lpips | slice_adapted_3dgs | 0.241425 | 0.176347 | -0.0650773 | 26.9555 | True |
| tdsc_abus_2023 | adapted_Gaussian | dists | slice_adapted_3dgs | 0.199547 | 0.156934 | -0.042613 | 21.3548 | True |
| tdsc_abus_2023 | adapted_Gaussian | gmsd | slice_adapted_3dgs | 0.0741601 | 0.0673528 | -0.00680726 | 9.17915 | True |
| tus_rec_2024_qc | NeRF | psnr | nerfus_no_prior | 29.4267 | 33.7694 | 4.34262 |  | True |
| tus_rec_2024_qc | NeRF | l1 | nerfus_no_prior | 0.0167516 | 0.0113981 | -0.00535352 | 31.9583 | True |
| tus_rec_2024_qc | NeRF | ssim | nerfus_no_prior | 0.836057 | 0.882168 | 0.0461116 |  | True |
| tus_rec_2024_qc | NeRF | lpips | ultranerf | 0.273882 | 0.111152 | -0.16273 | 59.4162 | True |
| tus_rec_2024_qc | NeRF | dists | ultranerf | 0.248937 | 0.133941 | -0.114996 | 46.1947 | True |
| tus_rec_2024_qc | NeRF | gmsd | nerfus_no_prior | 0.0847665 | 0.0473666 | -0.0373999 | 44.1211 | True |
| tus_rec_2024_qc | ultrasound_Gaussian | psnr | ultragray | 33.8162 | 33.7694 | -0.0468816 |  | False |
| tus_rec_2024_qc | ultrasound_Gaussian | l1 | ultragray | 0.0129626 | 0.0113981 | -0.00156454 | 12.0696 | True |
| tus_rec_2024_qc | ultrasound_Gaussian | ssim | ultragray | 0.832439 | 0.882168 | 0.0497293 |  | True |
| tus_rec_2024_qc | ultrasound_Gaussian | lpips | ultragray | 0.132098 | 0.111152 | -0.0209459 | 15.8564 | True |
| tus_rec_2024_qc | ultrasound_Gaussian | dists | ultragray | 0.138721 | 0.133941 | -0.00477962 | 3.4455 | True |
| tus_rec_2024_qc | ultrasound_Gaussian | gmsd | ultragray | 0.0576403 | 0.0473666 | -0.0102737 | 17.8239 | True |
| tus_rec_2024_qc | adapted_Gaussian | psnr | slice_adapted_3dgs | 33.9551 | 33.7694 | -0.185781 |  | False |
| tus_rec_2024_qc | adapted_Gaussian | l1 | slice_adapted_3dgs | 0.0115227 | 0.0113981 | -0.000124643 | 1.08171 | True |
| tus_rec_2024_qc | adapted_Gaussian | ssim | slice_adapted_3dgs | 0.889401 | 0.882168 | -0.00723282 |  | False |
| tus_rec_2024_qc | adapted_Gaussian | lpips | slice_adapted_3dgs | 0.109049 | 0.111152 | 0.00210236 | -1.9279 | False |
| tus_rec_2024_qc | adapted_Gaussian | dists | slice_adapted_3dgs | 0.135241 | 0.133941 | -0.00129977 | 0.961075 | True |
| tus_rec_2024_qc | adapted_Gaussian | gmsd | slice_adapted_3dgs | 0.045406 | 0.0473666 | 0.00196063 | -4.318 | False |

Training-time advantage claims are not hardware-controlled: new training uses4080SUPER, anchors/legacy use4090, and concurrency differs. Old240–353FPS is not corrected evidence. Fair corrected FPS is separately gated on4090 measurements.

## Joint factorial

| dataset | PPP | LGR | CGA | final_psnr | final_ssim | final_lpips | final_num_gaussians |
|---|---|---|---|---|---|---|---|
| synthetic_liver | 0 | 0 | 0 | 29.7824 | 0.677221 | 0.194015 | 549728 |
| synthetic_liver | 0 | 0 | 1 | 29.7903 | 0.67741 | 0.194717 | 586524 |
| synthetic_liver | 0 | 1 | 0 | 29.669 | 0.672488 | 0.180254 | 539572 |
| synthetic_liver | 0 | 1 | 1 | 29.6569 | 0.672849 | 0.179989 | 616314 |
| synthetic_liver | 1 | 0 | 0 | 29.9529 | 0.681933 | 0.24071 | 521750 |
| synthetic_liver | 1 | 0 | 1 | 29.9543 | 0.682 | 0.240806 | 560085 |
| synthetic_liver | 1 | 1 | 0 | 29.8452 | 0.678064 | 0.223573 | 511500 |
| synthetic_liver | 1 | 1 | 1 | 29.8382 | 0.678226 | 0.223402 | 595627 |
| tus_rec_2024_qc | 0 | 0 | 0 | 32.3699 | 0.860472 | 0.133184 | 620585 |
| tus_rec_2024_qc | 0 | 0 | 1 | 32.3607 | 0.86048 | 0.133754 | 634845 |
| tus_rec_2024_qc | 0 | 1 | 0 | 32.5592 | 0.861907 | 0.120826 | 619438 |
| tus_rec_2024_qc | 0 | 1 | 1 | 32.5951 | 0.862374 | 0.120477 | 660659 |
| tus_rec_2024_qc | 1 | 0 | 0 | 32.7565 | 0.866708 | 0.136118 | 589751 |
| tus_rec_2024_qc | 1 | 0 | 1 | 32.7541 | 0.866685 | 0.136267 | 605452 |
| tus_rec_2024_qc | 1 | 1 | 0 | 32.9697 | 0.867841 | 0.122473 | 584094 |
| tus_rec_2024_qc | 1 | 1 | 1 | 32.9844 | 0.868027 | 0.122429 | 619915 |

| dataset | module | metric | mean_effect | min_effect | max_effect |
|---|---|---|---|---|---|
| synthetic_liver | PPP | psnr | 0.172983 | 0.163995 | 0.181311 |
| synthetic_liver | PPP | ssim | 0.0050639 | 0.00459003 | 0.00557573 |
| synthetic_liver | PPP | lpips | 0.0448791 | 0.0433191 | 0.0466947 |
| synthetic_liver | PPP | num_gaussians | -25794 | -28072 | -20687 |
| synthetic_liver | LGR | psnr | -0.117635 | -0.13338 | -0.107652 |
| synthetic_liver | LGR | ssim | -0.00423379 | -0.00473226 | -0.00377327 |
| synthetic_liver | LGR | lpips | -0.0157569 | -0.0174034 | -0.0137605 |
| synthetic_liver | LGR | num_gaussians | 11231.5 | -10250 | 35542 |
| synthetic_liver | CGA | psnr | -0.00244868 | -0.0120865 | 0.0078491 |
| synthetic_liver | CGA | ssim | 0.000194469 | 6.6447e-05 | 0.000360443 |
| synthetic_liver | CGA | lpips | 9.04598e-05 | -0.000265203 | 0.000701699 |
| synthetic_liver | CGA | num_gaussians | 59000 | 36796 | 84127 |
| tus_rec_2024_qc | PPP | psnr | 0.394948 | 0.386584 | 0.410497 |
| tus_rec_2024_qc | PPP | ssim | 0.00600663 | 0.00565224 | 0.00623573 |
| tus_rec_2024_qc | PPP | lpips | 0.00226169 | 0.00164695 | 0.00293436 |
| tus_rec_2024_qc | PPP | num_gaussians | -34078.8 | -40744 | -29393 |
| tus_rec_2024_qc | LGR | psnr | 0.21681 | 0.189304 | 0.23445 |
| tus_rec_2024_qc | LGR | ssim | 0.00145071 | 0.0011323 | 0.00189441 |
| tus_rec_2024_qc | LGR | lpips | -0.0132795 | -0.0138378 | -0.0123579 |
| tus_rec_2024_qc | LGR | num_gaussians | 8368.25 | -5657 | 25814 |
| tus_rec_2024_qc | CGA | psnr | 0.0097384 | -0.0092427 | 0.035904 |
| tus_rec_2024_qc | CGA | ssim | 0.000159464 | -2.2777e-05 | 0.000466989 |
| tus_rec_2024_qc | CGA | lpips | 8.15875e-05 | -0.000348952 | 0.000570055 |
| tus_rec_2024_qc | CGA | num_gaussians | 26750.8 | 14260 | 41221 |

Effects are paired across four settings of the other modules, not independent replicates. LGR-off means adaptive local without global transmission; CGA-on then uses local-aware fallback, not an all-protected path rule.

## PPP

| dataset | label | final_psnr | final_ssim | final_lpips |
|---|---|---|---|---|
| synthetic_liver | MP | 29.6569 | 0.672849 | 0.179989 |
| synthetic_liver | PP-S | 29.8463 | 0.67837 | 0.224325 |
| synthetic_liver | PP-A | 29.8603 | 0.678722 | 0.226285 |
| synthetic_liver | PPP | 29.8382 | 0.678226 | 0.223402 |
| tus_rec_2024_qc | MP | 32.5951 | 0.862374 | 0.120477 |
| tus_rec_2024_qc | PP-S | 32.9355 | 0.867681 | 0.11959 |
| tus_rec_2024_qc | PP-A | 32.51 | 0.861811 | 0.142989 |
| tus_rec_2024_qc | PPP | 32.9844 | 0.868027 | 0.122429 |

| dataset | n | MP_PSNR_mean | MP_PSNR_std | PPP_PSNR_mean | PPP_PSNR_std | MP_SSIM_mean | PPP_SSIM_mean | delta_PSNR_mean | delta_PSNR_std |
|---|---|---|---|---|---|---|---|---|---|
| synthetic_liver | 6 | 29.2239 | 0.479694 | 29.4074 | 0.473684 | 0.648545 | 0.652746 | 0.183539 | 0.0258496 |
| tdsc_abus_2023 | 5 | 28.9691 | 1.1118 | 28.9654 | 1.1022 | 0.808926 | 0.808923 | -0.00368078 | 0.00975441 |
| tus_rec_2024_qc | 5 | 33.7693 | 0.787297 | 34.2654 | 0.897438 | 0.882733 | 0.888396 | 0.496096 | 0.144504 |

Geometry-use claim is confined to pseudo-trajectory construction. Arm subset preprocessing used measured-pose QC.

## LGR structural

| dataset | label | final_psnr | final_ssim | final_lpips | final_num_gaussians |
|---|---|---|---|---|---|
| synthetic_liver | Plane | 25.7949 | 0.615747 | 0.387236 | 569320 |
| synthetic_liver | Fixed | 29.5886 | 0.667039 | 0.167928 | 673117 |
| synthetic_liver | Adaptive | 29.7903 | 0.67741 | 0.194717 | 586524 |
| synthetic_liver | Simple | 29.7003 | 0.672675 | 0.179914 | 605446 |
| synthetic_liver | Full | 29.6569 | 0.672849 | 0.179989 | 616314 |
| tus_rec_2024_qc | Plane | 21.3978 | 0.649532 | 0.33987 | 559193 |
| tus_rec_2024_qc | Fixed | 32.5178 | 0.863977 | 0.123253 | 690492 |
| tus_rec_2024_qc | Adaptive | 32.3607 | 0.86048 | 0.133754 | 634845 |
| tus_rec_2024_qc | Simple | 32.4677 | 0.861498 | 0.12097 | 650561 |
| tus_rec_2024_qc | Full | 32.5951 | 0.862374 | 0.120477 | 660659 |

| dataset | transition | delta_PSNR | delta_SSIM | delta_LPIPS |
|---|---|---|---|---|
| synthetic_liver | Plane -> Fixed | 3.79375 | 0.0512917 | -0.219308 |
| synthetic_liver | Fixed -> Adaptive | 0.201636 | 0.0103709 | 0.0267888 |
| synthetic_liver | Adaptive -> Simple | -0.0900238 | -0.00473483 | -0.0148026 |
| synthetic_liver | Simple -> Full | -0.0433564 | 0.000174164 | 7.5126e-05 |
| tus_rec_2024_qc | Plane -> Fixed | 11.1201 | 0.214445 | -0.216617 |
| tus_rec_2024_qc | Fixed -> Adaptive | -0.157163 | -0.00349671 | 0.0105007 |
| tus_rec_2024_qc | Adaptive -> Simple | 0.107057 | 0.00101808 | -0.012784 |
| tus_rec_2024_qc | Simple -> Full | 0.127394 | 0.000876331 | -0.000492919 |

No monotonic improvement is assumed. Plane/Fixed isolates finite-slice conditioning; Fixed/Adaptive isolates depth dependence. Global-on also enables contribution-based path protection, so Adaptive/Simple is a whole-system structural contrast, not a renderer-only fixed-topology estimate. Simple/Full compares the specified preceding-depth approximation to causal integration. No attribute-decoupling ablation is claimed.

## CGA structural and population

| dataset | label | final_psnr | final_lpips | final_num_gaussians |
|---|---|---|---|---|
| synthetic_liver | GD | 29.669 | 0.180254 | 539572 |
| synthetic_liver | RGD | 29.6232 | 0.179506 | 1.07324e+06 |
| synthetic_liver | RGD+OP | 29.6659 | 0.180232 | 557778 |
| synthetic_liver | RGD+LP | 29.6707 | 0.181362 | 562701 |
| synthetic_liver | Full | 29.6569 | 0.179989 | 616314 |
| tus_rec_2024_qc | GD | 32.5592 | 0.120826 | 619438 |
| tus_rec_2024_qc | RGD | 32.4614 | 0.121303 | 1.02796e+06 |
| tus_rec_2024_qc | RGD+OP | 32.5598 | 0.121304 | 631202 |
| tus_rec_2024_qc | RGD+LP | 32.5485 | 0.12091 | 632560 |
| tus_rec_2024_qc | Full | 32.5951 | 0.120477 | 660659 |

| dataset | transition | delta_G | delta_G_pct | delta_PSNR | delta_LPIPS |
|---|---|---|---|---|---|
| synthetic_liver | GD -> RGD | 533670 | 98.9062 | -0.0458153 | -0.000748529 |
| synthetic_liver | RGD -> RGD+OP | -515464 | -48.0287 | 0.0427006 | 0.00072629 |
| synthetic_liver | RGD -> RGD+LP | -510541 | -47.57 | 0.0474911 | 0.00185611 |
| synthetic_liver | RGD+LP -> Full | 53613 | 9.5278 | -0.0137623 | -0.00137279 |
| tus_rec_2024_qc | GD -> RGD | 408520 | 65.9501 | -0.0977806 | 0.00047713 |
| tus_rec_2024_qc | RGD -> RGD+OP | -396756 | -38.5965 | 0.0983266 | 9.09e-07 |
| tus_rec_2024_qc | RGD -> RGD+LP | -395398 | -38.4644 | 0.087051 | -0.000392606 |
| tus_rec_2024_qc | RGD+LP -> Full | 28099 | 4.44211 | 0.0466336 | -0.000433476 |

RGD vs GD also removes conventional pruning by definition. RGD+OP restores it; RGD+LP tests contribution-aware pruning; Full adds global protection. Final population differences are net training outcomes, not a count of the exact same Gaussians preserved.

## Representative rho

| dataset | rho | final_psnr | final_ssim | final_lpips |
|---|---|---|---|---|
| synthetic_liver | 0.6 | 30.0339 | 0.682524 | 0.26641 |
| synthetic_liver | 0.75 | 29.9821 | 0.681525 | 0.247244 |
| synthetic_liver | 0.9 | 29.8965 | 0.679717 | 0.231846 |
| synthetic_liver | 1 | 29.8382 | 0.678226 | 0.223402 |
| synthetic_liver | 1.1 | 29.7736 | 0.676451 | 0.216805 |
| synthetic_liver | 1.25 | 29.6816 | 0.673855 | 0.208182 |
| synthetic_liver | 1.5 | 29.4912 | 0.669135 | 0.195462 |
| tus_rec_2024_qc | 0.6 | 32.7 | 0.863818 | 0.140968 |
| tus_rec_2024_qc | 0.75 | 32.8492 | 0.866042 | 0.132368 |
| tus_rec_2024_qc | 0.9 | 32.9358 | 0.867301 | 0.125727 |
| tus_rec_2024_qc | 1 | 32.9844 | 0.868027 | 0.122429 |
| tus_rec_2024_qc | 1.1 | 33.0226 | 0.868561 | 0.119754 |
| tus_rec_2024_qc | 1.25 | 33.0936 | 0.869471 | 0.116426 |
| tus_rec_2024_qc | 1.5 | 33.1787 | 0.870608 | 0.112994 |

| dataset | metric | best_sampled_rho | default | best_sampled | default_minus_best |
|---|---|---|---|---|---|
| synthetic_liver | psnr | 0.6 | 29.8382 | 30.0339 | -0.195657 |
| synthetic_liver | ssim | 0.6 | 0.678226 | 0.682524 | -0.00429762 |
| synthetic_liver | lpips | 1.5 | 0.223402 | 0.195462 | 0.0279407 |
| tus_rec_2024_qc | psnr | 1.5 | 32.9844 | 33.1787 | -0.194367 |
| tus_rec_2024_qc | ssim | 1.5 | 0.868027 | 0.870608 | -0.00258132 |
| tus_rec_2024_qc | lpips | 1.5 | 0.122429 | 0.112994 | 0.0094353 |

Best sampled is descriptive, not a retuned paper default. No LGR/CGA robustness/plateau claim is validated by Batch1; those grids remain Batch2 decisions.

## Artifacts and next gate

See qualitative_paths.json (fixed inherited cases, identical baseline GT hashes), population_dynamics.pdf/png, rho_sensitivity.pdf/png, per-run tables, exact config/provenance in each output directory. Do not launch Batch2 automatically.
