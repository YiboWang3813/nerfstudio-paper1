# Final Fixed main claim audit

Only Fixed+Full quality is final Ours; same frozen6/9/33/30 cases and seven FINAL-REUSE baselines. No78-case retraining. Ranking uses unrounded values.

| dataset | method | n | final_psnr | final_l1 | final_ssim | final_lpips | final_dists | final_gmsd | final_num_gaussians | train_wall_time_sec |
|---|---|---|---|---|---|---|---|---|---|---|
| spine_phantom | ours_fixed_final | 9 | 31.0426 | 0.0141386 | 0.796865 | 0.130524 | 0.15677 | 0.0984123 | 663761 | 509.286 |
| synthetic_liver | ours_fixed_final | 6 | 29.0251 | 0.0250499 | 0.635463 | 0.15669 | 0.138797 | 0.0987445 | 676511 | 517.405 |
| tdsc_abus_2023 | ours_fixed_final | 30 | 28.9908 | 0.0231398 | 0.808505 | 0.174541 | 0.159328 | 0.0646853 | 766467 | 378.761 |
| tus_rec_2024_qc | ours_fixed_final | 33 | 34.1123 | 0.0110763 | 0.886903 | 0.0998362 | 0.125734 | 0.0446209 | 676188 | 459.982 |


| dataset | method | metric | raw_mean | displayed | rank | best | second_best |
|---|---|---|---|---|---|---|---|
| spine_phantom | ours_fixed_final | psnr | 31.0426 | 31.0426 | 1 | ours_fixed_final | ultragray |
| spine_phantom | ours_fixed_final | l1 | 0.0141386 | 0.0141 | 1 | ours_fixed_final | ultrags_baseline |
| spine_phantom | ours_fixed_final | ssim | 0.796865 | 0.7969 | 1 | ours_fixed_final | ultragray |
| spine_phantom | ours_fixed_final | lpips | 0.130524 | 0.1305 | 1 | ours_fixed_final | ultragray |
| spine_phantom | ours_fixed_final | dists | 0.15677 | 0.1568 | 2 | ultragray | ours_fixed_final |
| spine_phantom | ours_fixed_final | gmsd | 0.0984123 | 0.0984 | 1 | ours_fixed_final | ultragray |
| synthetic_liver | ours_fixed_final | psnr | 29.0251 | 29.0251 | 1 | ours_fixed_final | ultrags_baseline |
| synthetic_liver | ours_fixed_final | l1 | 0.0250499 | 0.0250 | 1 | ours_fixed_final | ultrags_baseline |
| synthetic_liver | ours_fixed_final | ssim | 0.635463 | 0.6355 | 1 | ours_fixed_final | nerfus_no_prior |
| synthetic_liver | ours_fixed_final | lpips | 0.15669 | 0.1567 | 2 | ultragray | ours_fixed_final |
| synthetic_liver | ours_fixed_final | dists | 0.138797 | 0.1388 | 2 | ultragray | ours_fixed_final |
| synthetic_liver | ours_fixed_final | gmsd | 0.0987445 | 0.0987 | 1 | ours_fixed_final | ultrags_baseline |
| tdsc_abus_2023 | ours_fixed_final | psnr | 28.9908 | 28.9908 | 1 | ours_fixed_final | ultragauss |
| tdsc_abus_2023 | ours_fixed_final | l1 | 0.0231398 | 0.0231 | 1 | ours_fixed_final | ultragauss |
| tdsc_abus_2023 | ours_fixed_final | ssim | 0.808505 | 0.8085 | 1 | ours_fixed_final | slice_adapted_3dgs |
| tdsc_abus_2023 | ours_fixed_final | lpips | 0.174541 | 0.1745 | 1 | ours_fixed_final | ultragauss |
| tdsc_abus_2023 | ours_fixed_final | dists | 0.159328 | 0.1593 | 1 | ours_fixed_final | ultragauss |
| tdsc_abus_2023 | ours_fixed_final | gmsd | 0.0646853 | 0.0647 | 1 | ours_fixed_final | slice_adapted_3dgs |
| tus_rec_2024_qc | ours_fixed_final | psnr | 34.1123 | 34.1123 | 1 | ours_fixed_final | slice_adapted_3dgs |
| tus_rec_2024_qc | ours_fixed_final | l1 | 0.0110763 | 0.0111 | 1 | ours_fixed_final | slice_adapted_3dgs |
| tus_rec_2024_qc | ours_fixed_final | ssim | 0.886903 | 0.8869 | 2 | slice_adapted_3dgs | ours_fixed_final |
| tus_rec_2024_qc | ours_fixed_final | lpips | 0.0998362 | 0.0998 | 1 | ours_fixed_final | slice_adapted_3dgs |
| tus_rec_2024_qc | ours_fixed_final | dists | 0.125734 | 0.1257 | 1 | ours_fixed_final | slice_adapted_3dgs |
| tus_rec_2024_qc | ours_fixed_final | gmsd | 0.0446209 | 0.0446 | 1 | ours_fixed_final | slice_adapted_3dgs |


## Supported / update numbers

PSNR and L1 rank1 on all4 datasets. SSIM rank1 on Liver/Spine/Breast, rank2 on Arm. LPIPS rank1 on Spine/Arm/Breast, rank2 on Liver. DISTS rank1 on Arm/Breast, rank2 on Liver/Spine. GMSD rank1 on all4. All old Adaptive/legacy numerical values and derived percentages must be replaced.

## Weaken / do not claim

Do not claim all metrics rank1. Fixed design trades lower Liver PSNR/L1/SSIM for better LPIPS/DISTS/GMSD; Breast DISTS is systematically worse than Adaptive despite remaining rank1 among baselines. Fixed usually retains more Gaussians; no efficiency-superiority or all-perceptual-gains claim.

## Efficiency

| dataset | time_seconds | fps |
|---|---|---|
| spine_phantom | 509.286 | 201.319 |
| synthetic_liver | 517.405 | 208.56 |
| tdsc_abus_2023 | 378.761 | 307.425 |
| tus_rec_2024_qc | 459.982 | 262.771 |


| panel | time_seconds | fps |
|---|---|---|
| synthetic_phantom | 513.345 | 204.94 |
| real | 419.371 | 285.098 |


FPS uses separately reconstructed Fixed models for the same two frozen cases per dataset,5warmup/20frames/3passes,pureforward on4090. Replica-vs-original deltas are retained; quality comes from the original78 models, not the replicas. Time uses outer training-process wall seconds and equal-dataset panel means; shared-worker/hardware heterogeneity precludes controlled speedup claims. Frozen baseline efficiency is reused verbatim, including historical aggregation caveats.

Canonical LaTeX is not modified; this report is the final numerical source.


## Approved numerical claim template

\methodname{} achieves the best PSNR, L1, and GMSD on all four datasets,
while attaining best or second-best SSIM, LPIPS, and DISTS throughout.

This is 100% accurate for the frozen eight-method, within-sweep Main comparison,
using unrounded means and MP for Ours. Do not remove this experimental scope or
imply these78runs used PPP. FPS uses separately rebuilt Fixed model replicas on
the frozen two cases/dataset; it is pure forward, not end-to-end throughput.
