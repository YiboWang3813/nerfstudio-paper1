# Paper-wide paired Fixed+Full candidate validation

78/78 Fixed and 78/78 Adaptive complete:11 Fixed reuse+67 new runs; all Adaptive reused. Corrected P0.5 source, MP,Full global,Full CGA,1M,30k,seed42. Only local conditioning mode differs.

Fixed minus Adaptive; sample standard deviation(ddof1), median paired difference; ties abs<=1e-6. PSNR/SSIM higher is better; L1/LPIPS/DISTS/GMSD lower is better. No significance tests. Arm segments are not independent subjects.

## Six-metric paired results

| dataset | metric | fixed_mean | adaptive_mean | n | mean_delta | std_paired_delta | median_delta | wins | ties | losses | tie_absolute_tolerance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_liver | psnr | 29.0251 | 29.2239 | 6 | -0.198836 | 0.0102893 | -0.198604 | 0 | 0 | 6 | 1e-06 |
| synthetic_liver | l1 | 0.0250499 | 0.0244273 | 6 | 0.000622532 | 4.32466e-05 | 0.000611954 | 0 | 0 | 6 | 1e-06 |
| synthetic_liver | ssim | 0.635463 | 0.648545 | 6 | -0.0130822 | 0.00148217 | -0.0124878 | 0 | 0 | 6 | 1e-06 |
| synthetic_liver | lpips | 0.15669 | 0.179857 | 6 | -0.0231669 | 0.000908789 | -0.0229686 | 6 | 0 | 0 | 1e-06 |
| synthetic_liver | dists | 0.138797 | 0.150655 | 6 | -0.011858 | 0.00140163 | -0.0116054 | 6 | 0 | 0 | 1e-06 |
| synthetic_liver | gmsd | 0.0987445 | 0.100859 | 6 | -0.0021148 | 0.000442113 | -0.00200421 | 6 | 0 | 0 | 1e-06 |
| spine_phantom | psnr | 31.0426 | 30.8063 | 9 | 0.23627 | 0.0972265 | 0.216737 | 9 | 0 | 0 | 1e-06 |
| spine_phantom | l1 | 0.0141386 | 0.0144627 | 9 | -0.000324072 | 0.000163789 | -0.00030701 | 9 | 0 | 0 | 1e-06 |
| spine_phantom | ssim | 0.796865 | 0.784377 | 9 | 0.0124882 | 0.00330012 | 0.0132832 | 9 | 0 | 0 | 1e-06 |
| spine_phantom | lpips | 0.130524 | 0.145393 | 9 | -0.0148697 | 0.00401152 | -0.0135636 | 9 | 0 | 0 | 1e-06 |
| spine_phantom | dists | 0.15677 | 0.1656 | 9 | -0.00883026 | 0.00292409 | -0.00845725 | 9 | 0 | 0 | 1e-06 |
| spine_phantom | gmsd | 0.0984123 | 0.104891 | 9 | -0.00647895 | 0.00186214 | -0.00671533 | 9 | 0 | 0 | 1e-06 |
| tus_rec_2024_qc | psnr | 34.1123 | 33.7694 | 33 | 0.342906 | 0.144582 | 0.327835 | 31 | 0 | 2 | 1e-06 |
| tus_rec_2024_qc | l1 | 0.0110763 | 0.0113981 | 33 | -0.000321727 | 0.000157159 | -0.000293783 | 32 | 0 | 1 | 1e-06 |
| tus_rec_2024_qc | ssim | 0.886903 | 0.882168 | 33 | 0.00473503 | 0.00189775 | 0.00436774 | 33 | 0 | 0 | 1e-06 |
| tus_rec_2024_qc | lpips | 0.0998362 | 0.111152 | 33 | -0.0113157 | 0.00294973 | -0.011598 | 33 | 0 | 0 | 1e-06 |
| tus_rec_2024_qc | dists | 0.125734 | 0.133941 | 33 | -0.00820714 | 0.00283151 | -0.0074269 | 33 | 0 | 0 | 1e-06 |
| tus_rec_2024_qc | gmsd | 0.0446209 | 0.0473666 | 33 | -0.00274568 | 0.000966696 | -0.002685 | 33 | 0 | 0 | 1e-06 |
| tdsc_abus_2023 | psnr | 28.9908 | 28.7395 | 30 | 0.251325 | 0.126568 | 0.238909 | 30 | 0 | 0 | 1e-06 |
| tdsc_abus_2023 | l1 | 0.0231398 | 0.0236351 | 30 | -0.000495261 | 0.000363397 | -0.000441083 | 30 | 0 | 0 | 1e-06 |
| tdsc_abus_2023 | ssim | 0.808505 | 0.804734 | 30 | 0.00377105 | 0.0035322 | 0.00291962 | 28 | 0 | 2 | 1e-06 |
| tdsc_abus_2023 | lpips | 0.174541 | 0.176347 | 30 | -0.00180605 | 0.00234228 | -0.00160139 | 23 | 0 | 7 | 1e-06 |
| tdsc_abus_2023 | dists | 0.159328 | 0.156934 | 30 | 0.00239335 | 0.000826487 | 0.0023659 | 0 | 0 | 30 | 1e-06 |
| tdsc_abus_2023 | gmsd | 0.0646853 | 0.0673528 | 30 | -0.00266753 | 0.00175289 | -0.00287857 | 30 | 0 | 0 | 1e-06 |


## Representation and engineering observations

| dataset | local | field | n | mean | std | median | minimum | maximum |
|---|---|---|---|---|---|---|---|---|
| synthetic_liver | fixed | final_G | 6 | 676511 | 13454.4 | 678075 | 652145 | 688611 |
| synthetic_liver | fixed | training_wrapper_wall_seconds | 6 | 514.493 | 93.8969 | 508.915 | 427.095 | 606.68 |
| synthetic_liver | fixed | peak_local_tile_gaussian_pairs | 6 | 3.28089e+06 | 76301.4 | 3.29264e+06 | 3158576 | 3362031 |
| synthetic_liver | fixed | peak_global_ray_gaussian_pairs | 6 | 332058 | 14482.5 | 329452 | 315492 | 358232 |
| synthetic_liver | fixed | peak_allocated_bytes | 6 | 3.09957e+09 | 3.80378e+06 | 3.10014e+09 | 3094163456 | 3102967808 |
| synthetic_liver | fixed | peak_reserved_bytes | 6 | 3117416448 | 1.14866e+06 | 3.11742e+09 | 3116367872 | 3118465024 |
| synthetic_liver | adaptive | final_G | 6 | 604772 | 12641.1 | 606650 | 582245 | 616314 |
| synthetic_liver | adaptive | training_wrapper_wall_seconds | 6 | 542.762 | 57.8202 | 562.072 | 425.979 | 578.701 |
| synthetic_liver | adaptive | peak_local_tile_gaussian_pairs | 6 | 3.153e+06 | 65906.8 | 3.16545e+06 | 3051712 | 3220422 |
| synthetic_liver | adaptive | peak_global_ray_gaussian_pairs | 6 | 267239 | 10348 | 264970 | 253713 | 282754 |
| synthetic_liver | adaptive | peak_allocated_bytes | 6 | 3.09957e+09 | 3.80378e+06 | 3.10014e+09 | 3094163456 | 3102967808 |
| synthetic_liver | adaptive | peak_reserved_bytes | 6 | 3117416448 | 1.14866e+06 | 3.11742e+09 | 3116367872 | 3118465024 |
| spine_phantom | fixed | final_G | 9 | 663761 | 8941.69 | 662567 | 645523 | 673910 |
| spine_phantom | fixed | training_wrapper_wall_seconds | 9 | 506.039 | 70.7057 | 466.943 | 444.761 | 616.68 |
| spine_phantom | fixed | peak_local_tile_gaussian_pairs | 9 | 3.79431e+06 | 101482 | 3814103 | 3599830 | 3921110 |
| spine_phantom | fixed | peak_global_ray_gaussian_pairs | 9 | 261668 | 14397 | 259712 | 235769 | 283145 |
| spine_phantom | fixed | peak_allocated_bytes | 9 | 2110825984 | 5.05548e+07 | 2123979776 | 2005082624 | 2162478080 |
| spine_phantom | fixed | peak_reserved_bytes | 9 | 2225078272 | 4.00798e+07 | 2222981120 | 2174746624 | 2290089984 |
| spine_phantom | adaptive | final_G | 9 | 603763 | 11365 | 601094 | 588634 | 621939 |
| spine_phantom | adaptive | training_wrapper_wall_seconds | 9 | 575.451 | 13.3731 | 575.745 | 558.104 | 596.041 |
| spine_phantom | adaptive | peak_local_tile_gaussian_pairs | 9 | 3.56464e+06 | 111393 | 3602964 | 3368853 | 3708417 |
| spine_phantom | adaptive | peak_global_ray_gaussian_pairs | 9 | 238029 | 14100.4 | 238319 | 210930 | 253698 |
| spine_phantom | adaptive | peak_allocated_bytes | 9 | 2110825984 | 5.05548e+07 | 2123979776 | 2005082624 | 2162478080 |
| spine_phantom | adaptive | peak_reserved_bytes | 9 | 2.1603e+09 | 5.27262e+07 | 2174746624 | 2051014656 | 2222981120 |
| tus_rec_2024_qc | fixed | final_G | 33 | 676188 | 20255 | 678369 | 629432 | 720916 |
| tus_rec_2024_qc | fixed | training_wrapper_wall_seconds | 33 | 456.667 | 158.854 | 421.807 | 339.249 | 1187.88 |
| tus_rec_2024_qc | fixed | peak_local_tile_gaussian_pairs | 33 | 3.32526e+06 | 320249 | 3413156 | 2598646 | 3783610 |
| tus_rec_2024_qc | fixed | peak_global_ray_gaussian_pairs | 33 | 388600 | 345056 | 252794 | 156165 | 1956737 |
| tus_rec_2024_qc | fixed | peak_allocated_bytes | 33 | 5.48854e+09 | 2.86179e+09 | 6059995136 | 1492000256 | 11126240768 |
| tus_rec_2024_qc | fixed | peak_reserved_bytes | 33 | 5.67159e+09 | 2.91875e+09 | 6247415808 | 1658847232 | 11460935680 |
| tus_rec_2024_qc | adaptive | final_G | 33 | 633049 | 37371.8 | 629197 | 590149 | 813185 |
| tus_rec_2024_qc | adaptive | training_wrapper_wall_seconds | 33 | 483.412 | 170.919 | 419.146 | 326.891 | 1289.19 |
| tus_rec_2024_qc | adaptive | peak_local_tile_gaussian_pairs | 33 | 3.18971e+06 | 263937 | 3240073 | 2646878 | 3625973 |
| tus_rec_2024_qc | adaptive | peak_global_ray_gaussian_pairs | 33 | 352695 | 380213 | 212873 | 127576 | 2202690 |
| tus_rec_2024_qc | adaptive | peak_allocated_bytes | 33 | 5.48876e+09 | 2.86148e+09 | 6059995136 | 1499149312 | 11126240768 |
| tus_rec_2024_qc | adaptive | peak_reserved_bytes | 33 | 5.67165e+09 | 2.91867e+09 | 6247415808 | 1658847232 | 11460935680 |
| tdsc_abus_2023 | fixed | final_G | 30 | 766467 | 56156.6 | 787291 | 660230 | 836839 |
| tdsc_abus_2023 | fixed | training_wrapper_wall_seconds | 30 | 375.469 | 39.442 | 367.933 | 326.713 | 427.675 |
| tdsc_abus_2023 | fixed | peak_local_tile_gaussian_pairs | 30 | 4.32929e+06 | 404086 | 4.62559e+06 | 3600494 | 4673745 |
| tdsc_abus_2023 | fixed | peak_global_ray_gaussian_pairs | 30 | 115877 | 11274 | 119266 | 86341 | 130662 |
| tdsc_abus_2023 | fixed | peak_allocated_bytes | 30 | 5.41495e+09 | 2.61578e+08 | 5.573e+09 | 4835329024 | 5660161536 |
| tdsc_abus_2023 | fixed | peak_reserved_bytes | 30 | 5.4491e+09 | 2.63723e+08 | 5.60988e+09 | 4861198336 | 5685379072 |
| tdsc_abus_2023 | adaptive | final_G | 30 | 737606 | 53412.8 | 757840 | 637142 | 804267 |
| tdsc_abus_2023 | adaptive | training_wrapper_wall_seconds | 30 | 407.26 | 14.8167 | 413.227 | 376.087 | 430.072 |
| tdsc_abus_2023 | adaptive | peak_local_tile_gaussian_pairs | 30 | 4.22779e+06 | 407197 | 4.5241e+06 | 3483746 | 4573273 |
| tdsc_abus_2023 | adaptive | peak_global_ray_gaussian_pairs | 30 | 106686 | 10179.6 | 109934 | 80035 | 118845 |
| tdsc_abus_2023 | adaptive | peak_allocated_bytes | 30 | 5.41495e+09 | 2.61578e+08 | 5.573e+09 | 4835329024 | 5660161536 |
| tdsc_abus_2023 | adaptive | peak_reserved_bytes | 30 | 5.4491e+09 | 2.63723e+08 | 5.60988e+09 | 4861198336 | 5685379072 |


Training-wrapper wall time includes shared-worker/hardware differences and is only an engineering observation. No FPS comparison has been performed. Candidate pair counts are maxima per run; their dataset mean/std and maximum are explicitly distinguished.

## Candidate ranks

| dataset | metric | adaptive_mean | fixed_mean | adaptive_rank | fixed_rank |
|---|---|---|---|---|---|
| synthetic_liver | psnr | 29.2239 | 29.0251 | 1 | 1 |
| synthetic_liver | l1 | 0.0244273 | 0.0250499 | 1 | 1 |
| synthetic_liver | ssim | 0.648545 | 0.635463 | 1 | 1 |
| synthetic_liver | lpips | 0.179857 | 0.15669 | 3 | 2 |
| synthetic_liver | dists | 0.150655 | 0.138797 | 3 | 2 |
| synthetic_liver | gmsd | 0.100859 | 0.0987445 | 1 | 1 |
| spine_phantom | psnr | 30.8063 | 31.0426 | 1 | 1 |
| spine_phantom | l1 | 0.0144627 | 0.0141386 | 1 | 1 |
| spine_phantom | ssim | 0.784377 | 0.796865 | 1 | 1 |
| spine_phantom | lpips | 0.145393 | 0.130524 | 2 | 1 |
| spine_phantom | dists | 0.1656 | 0.15677 | 2 | 2 |
| spine_phantom | gmsd | 0.104891 | 0.0984123 | 1 | 1 |
| tus_rec_2024_qc | psnr | 33.7694 | 34.1123 | 3 | 1 |
| tus_rec_2024_qc | l1 | 0.0113981 | 0.0110763 | 1 | 1 |
| tus_rec_2024_qc | ssim | 0.882168 | 0.886903 | 2 | 2 |
| tus_rec_2024_qc | lpips | 0.111152 | 0.0998362 | 2 | 1 |
| tus_rec_2024_qc | dists | 0.133941 | 0.125734 | 1 | 1 |
| tus_rec_2024_qc | gmsd | 0.0473666 | 0.0446209 | 2 | 1 |
| tdsc_abus_2023 | psnr | 28.7395 | 28.9908 | 1 | 1 |
| tdsc_abus_2023 | l1 | 0.0236351 | 0.0231398 | 1 | 1 |
| tdsc_abus_2023 | ssim | 0.804734 | 0.808505 | 1 | 1 |
| tdsc_abus_2023 | lpips | 0.176347 | 0.174541 | 1 | 1 |
| tdsc_abus_2023 | dists | 0.156934 | 0.159328 | 1 | 1 |
| tdsc_abus_2023 | gmsd | 0.0673528 | 0.0646853 | 1 | 1 |


Frozen main qualitative cases/crops are used without reselection; see contact_sheets/manifest.json. Await visual/scientific review and F1/F2/F3 decision. No automatic default change, dataset-specific mode, Methods edit, or Batch2 submission.
