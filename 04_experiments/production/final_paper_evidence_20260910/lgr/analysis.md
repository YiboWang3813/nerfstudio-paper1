# LGR: structural and numeric interpretation

## Final structural table (Liver6 / Arm5 means)

See final_lgr_table.tsv and structural_sources.tsv. Plane/Fixed/Adaptive are
local-only; Simple/Full both use Fixed, MP and contribution-aware adaptation.
Global-path availability also controls CGA global protection: these are complete
system comparisons, not isolation with a frozen Gaussian population.

| dataset | transition | delta_psnr | delta_ssim | delta_lpips |
|---|---|---|---|---|
| Liver | Plane -> Fixed | 4.06904 | 0.043129 | -0.258034 |
| Liver | Fixed -> Adaptive | 0.177949 | 0.0102622 | 0.0261285 |
| Liver | Fixed -> Fixed+Simple | -0.103442 | -0.00713754 | -0.0126927 |
| Liver | Fixed+Simple -> Fixed+Full | -0.0321163 | -0.000168484 | 0.000199736 |
| Arm | Plane -> Fixed | 12.2362 | 0.207963 | -0.218196 |
| Arm | Fixed -> Adaptive | -0.262558 | -0.00473911 | 0.0107958 |
| Arm | Fixed -> Fixed+Simple | 0.349972 | 0.00280013 | -0.0125737 |
| Arm | Fixed+Simple -> Fixed+Full | 0.101267 | 0.000570981 | -0.000512741 |


Plane→Fixed has the largest, consistent benefit (PSNR/SSIM/LPIPS improved on all
11 matched cases). Fixed→Adaptive increases Liver PSNR/SSIM but worsens LPIPS;
Arm worsens all three. Choose the shared simpler Fixed design, not a progressive
five-row addition story. Fixed→Simple: Arm improves all three aggregate metrics;
Liver PSNR/SSIM fall slightly while LPIPS improves. Simple→Full: Arm aggregate
PSNR/SSIM/LPIPS further improve; Liver PSNR slightly decreases and SSIM/LPIPS
are nearly unchanged with small unfavorable mean deltas. “Significant” must not
mean statistical significance; no unplanned significance test was conducted.

## Numerical interpretation

- **sigma_ele: materially sensitive, not unstable.** Widths .8/1/1.5/2mm;
  .8 is the existing implementation clamp minimum. Liver PSNR/SSIM rise
  monotonically with width, while LPIPS worsens; Arm all three metrics worsen
  with width. PSNR ranges .5873178/.7704966dB; LPIPS ranges .035017312/.021516102.
  Liver PSNR-best is2mm, Arm PSNR-best .8mm; both LPIPS-best .8mm. At1mm,
  Arm loses only .0446587dB and .002940552 LPIPS relative to.8mm; Liver gains
  .2024098dB over.8 at a .004937364 LPIPS cost. Wider2mm degrades Arm by
  .7258379dB relative to1mm. No tested narrow-width catastrophic failure and no
  evidence below.8mm. Supported: finite-slice width materially affects quality,
  while fixed1mm is a shared cross-domain operating point, not a universal optimum.
- **beta: moderately sensitive, with domain/metric trade-off.** .4/.6/.8/1;
  PSNR ranges .0912602/.2570081dB and LPIPS .016957068/.014544701.
  Increasing beta monotonically decreases Liver PSNR/SSIM but improves LPIPS;
  Arm PSNR/SSIM/LPIPS all improve. Default.8 is an interior shared setting,
  not the best on every metric. Do not lump beta with near-flat lambda/window.
- **lambda_ray: stable over the tested .2–1 range.** PSNR ranges
  .0171847/.0541477dB; LPIPS .001888730/.001020012. Changes are small and
  not uniformly monotonic. Default.55 lies in this stable observed region.
  “Reconstruction remains stable over a broad tested attenuation-strength range”
  is supported for these two cases; avoid absolute insensitivity claims.
- **Nq: non-monotonic; NO convergence/saturation claim.** Nq1 is close to3;
  Nq2 is worse on both datasets. 2→3 yields +.4096548/+.7780779dB,
  +.006553606/+.005368477 SSIM, and -.008574640/-.003282091 LPIPS.
  1→3 yields only +.0093911/+.0207719dB; Liver SSIM slightly falls and Arm
  LPIPS slightly worsens. Nq3 is the PSNR-best of the sampled settings but these
  trained-model endpoints do not establish convergence to an exact integral.
  CUDA forward/backward use standard Gauss–Legendre nodes/weights:1point at0,
  2points at±1/sqrt(3),3points at0 and±sqrt(3/5). No explicit node/weight bug
  was found. Center-sampling parity for sharply localized peaks is a plausible
  mechanism, not an experimentally identified cause; learned geometry and CGA
  co-adapt. Do not write “very low order is insufficient, then saturates.”
- **Half-window: stable across1–4mm; no established cost penalty.** PSNR
  ranges .0206915/.0244875dB; LPIPS .001247229/.001559514. Default2mm is
  reasonable; no monotonic quality improvement with larger windows. Mean global
  associations vary only ~.19% Liver/~.94% Arm across all values (denominator=min).
  These counts mix training/evaluation forwards and co-adapted populations.
  Nq remains3; increasing the window does not increase quadrature sample count.
  No controlled per-point FPS measurement exists; shared-worker wall time is not
  speed evidence. Supported: a larger window was not necessary for comparable
  quality in these cases. NOT supported: larger windows substantially raise cost.

All range endpoints/default-relative deltas are saved unrounded in data files.
No test-selected retuning, new integration order or extra experiment is proposed.

## sigma_ele

| dataset | point | is_default | final_psnr | final_ssim | final_lpips | delta_psnr | delta_ssim | delta_lpips |
|---|---|---|---|---|---|---|---|---|
| synthetic_liver | 0.8 | False | 29.2498 | 0.651978 | 0.152107 | -0.20241 | -0.00829655 | -0.00493736 |
| synthetic_liver | 1 | True | 29.4522 | 0.660275 | 0.157044 | 0 | 0 | 0 |
| synthetic_liver | 1.5 | False | 29.7074 | 0.671222 | 0.172535 | 0.255198 | 0.0109475 | 0.0154907 |
| synthetic_liver | 2 | False | 29.8371 | 0.67603 | 0.187124 | 0.384908 | 0.0157552 | 0.0300799 |
| tus_rec_2024_qc | 0.8 | False | 32.8614 | 0.866248 | 0.109099 | 0.0446587 | 0.000483213 | -0.00294055 |
| tus_rec_2024_qc | 1 | True | 32.8167 | 0.865764 | 0.11204 | 0 | 0 | 0 |
| tus_rec_2024_qc | 1.5 | False | 32.4778 | 0.860798 | 0.121091 | -0.338864 | -0.00496678 | 0.00905081 |
| tus_rec_2024_qc | 2 | False | 32.0909 | 0.8547 | 0.130615 | -0.725838 | -0.0110643 | 0.0185756 |


## beta

| dataset | point | is_default | final_psnr | final_ssim | final_lpips | delta_psnr | delta_ssim | delta_lpips |
|---|---|---|---|---|---|---|---|---|
| synthetic_liver | 0.4 | False | 29.5381 | 0.665149 | 0.16573 | 0.0858828 | 0.0048742 | 0.0086857 |
| synthetic_liver | 0.6 | False | 29.4822 | 0.66274 | 0.16289 | 0.0299921 | 0.00246535 | 0.00584582 |
| synthetic_liver | 0.8 | True | 29.4522 | 0.660275 | 0.157044 | 0 | 0 | 0 |
| synthetic_liver | 1 | False | 29.4468 | 0.65746 | 0.148773 | -0.0053774 | -0.00281516 | -0.00827137 |
| tus_rec_2024_qc | 0.4 | False | 32.6617 | 0.865037 | 0.118836 | -0.154961 | -0.000727398 | 0.00679586 |
| tus_rec_2024_qc | 0.6 | False | 32.7224 | 0.8651 | 0.116945 | -0.0943071 | -0.000664007 | 0.00490498 |
| tus_rec_2024_qc | 0.8 | True | 32.8167 | 0.865764 | 0.11204 | 0 | 0 | 0 |
| tus_rec_2024_qc | 1 | False | 32.9188 | 0.866548 | 0.104291 | 0.102047 | 0.000783298 | -0.00774885 |


## lambda_ray

| dataset | point | is_default | final_psnr | final_ssim | final_lpips | delta_psnr | delta_ssim | delta_lpips |
|---|---|---|---|---|---|---|---|---|
| synthetic_liver | 0.2 | False | 29.4473 | 0.660189 | 0.158221 | -0.0049119 | -8.6125e-05 | 0.00117741 |
| synthetic_liver | 0.35 | False | 29.4423 | 0.660395 | 0.158097 | -0.009853 | 0.00012035 | 0.00105303 |
| synthetic_liver | 0.55 | True | 29.4522 | 0.660275 | 0.157044 | 0 | 0 | 0 |
| synthetic_liver | 0.75 | False | 29.4574 | 0.660472 | 0.157681 | 0.0051957 | 0.000197353 | 0.000637292 |
| synthetic_liver | 1 | False | 29.4595 | 0.660629 | 0.158933 | 0.0073317 | 0.000354375 | 0.00188873 |
| tus_rec_2024_qc | 0.2 | False | 32.7697 | 0.865587 | 0.111602 | -0.047005 | -0.000177813 | -0.000438022 |
| tus_rec_2024_qc | 0.35 | False | 32.785 | 0.865613 | 0.111062 | -0.0317083 | -0.00015172 | -0.00097768 |
| tus_rec_2024_qc | 0.55 | True | 32.8167 | 0.865764 | 0.11204 | 0 | 0 | 0 |
| tus_rec_2024_qc | 0.75 | False | 32.8238 | 0.865827 | 0.111419 | 0.0071427 | 6.2538e-05 | -0.000620592 |
| tus_rec_2024_qc | 1 | False | 32.8207 | 0.86575 | 0.11102 | 0.0039576 | -1.4161e-05 | -0.00102001 |


## Nq

| dataset | point | is_default | final_psnr | final_ssim | final_lpips | delta_psnr | delta_ssim | delta_lpips |
|---|---|---|---|---|---|---|---|---|
| synthetic_liver | 1 | False | 29.4428 | 0.660401 | 0.158427 | -0.0093911 | 0.00012624 | 0.00138295 |
| synthetic_liver | 2 | False | 29.0425 | 0.653721 | 0.165619 | -0.409655 | -0.00655361 | 0.00857464 |
| synthetic_liver | 3 | True | 29.4522 | 0.660275 | 0.157044 | 0 | 0 | 0 |
| tus_rec_2024_qc | 1 | False | 32.7959 | 0.865486 | 0.11176 | -0.0207719 | -0.00027882 | -0.000279233 |
| tus_rec_2024_qc | 2 | False | 32.0386 | 0.860396 | 0.115322 | -0.778078 | -0.00536848 | 0.00328209 |
| tus_rec_2024_qc | 3 | True | 32.8167 | 0.865764 | 0.11204 | 0 | 0 | 0 |


## half_window

| dataset | point | is_default | final_psnr | final_ssim | final_lpips | delta_psnr | delta_ssim | delta_lpips |
|---|---|---|---|---|---|---|---|---|
| synthetic_liver | 1 | False | 29.4409 | 0.660325 | 0.158156 | -0.0113128 | 5.0418e-05 | 0.00111186 |
| synthetic_liver | 2 | True | 29.4522 | 0.660275 | 0.157044 | 0 | 0 | 0 |
| synthetic_liver | 3 | False | 29.4315 | 0.660091 | 0.158291 | -0.0206915 | -0.000184208 | 0.00124723 |
| synthetic_liver | 4 | False | 29.4395 | 0.660437 | 0.157854 | -0.0127161 | 0.000161669 | 0.000809571 |
| tus_rec_2024_qc | 1 | False | 32.7947 | 0.865527 | 0.111625 | -0.0220064 | -0.000236958 | -0.000414755 |
| tus_rec_2024_qc | 2 | True | 32.8167 | 0.865764 | 0.11204 | 0 | 0 | 0 |
| tus_rec_2024_qc | 3 | False | 32.8187 | 0.865787 | 0.111206 | 0.0019565 | 2.2602e-05 | -0.000834168 |
| tus_rec_2024_qc | 4 | False | 32.7942 | 0.865724 | 0.11048 | -0.022531 | -4.0242e-05 | -0.00155951 |


## Complete ranges / default positions

| dataset | panel | n | default_point | psnr_min | psnr_max | psnr_range | psnr_default | ssim_min | ssim_max | ssim_range | ssim_default | lpips_min | lpips_max | lpips_range | lpips_default | G_min | G_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_liver | sigma_ele | 4.000000 | 1.000000 | 29.249792 | 29.837110 | 0.587318 | 29.452202 | 0.651978 | 0.676030 | 0.024052 | 0.660275 | 0.152107 | 0.187124 | 0.035017 | 0.157044 | 591920.000000 | 724160.000000 |
| tus_rec_2024_qc | sigma_ele | 4.000000 | 1.000000 | 32.090867 | 32.861364 | 0.770497 | 32.816705 | 0.854700 | 0.866248 | 0.011547 | 0.865764 | 0.109099 | 0.130615 | 0.021516 | 0.112040 | 614276.000000 | 742257.000000 |
| synthetic_liver | beta | 4.000000 | 0.800000 | 29.446825 | 29.538085 | 0.091260 | 29.452202 | 0.657460 | 0.665149 | 0.007689 | 0.660275 | 0.148773 | 0.165730 | 0.016957 | 0.157044 | 683014.000000 | 704516.000000 |
| tus_rec_2024_qc | beta | 4.000000 | 0.800000 | 32.661744 | 32.918752 | 0.257008 | 32.816705 | 0.865037 | 0.866548 | 0.001511 | 0.865764 | 0.104291 | 0.118836 | 0.014545 | 0.112040 | 704465.000000 | 731788.000000 |
| synthetic_liver | lambda_ray | 5.000000 | 0.550000 | 29.442349 | 29.459534 | 0.017185 | 29.452202 | 0.660189 | 0.660629 | 0.000441 | 0.660275 | 0.157044 | 0.158933 | 0.001889 | 0.157044 | 687324.000000 | 688797.000000 |
| tus_rec_2024_qc | lambda_ray | 5.000000 | 0.550000 | 32.769700 | 32.823848 | 0.054148 | 32.816705 | 0.865587 | 0.865827 | 0.000240 | 0.865764 | 0.111020 | 0.112040 | 0.001020 | 0.112040 | 711157.000000 | 713263.000000 |
| synthetic_liver | Nq | 3.000000 | 3.000000 | 29.042547 | 29.452202 | 0.409655 | 29.452202 | 0.653721 | 0.660401 | 0.006680 | 0.660275 | 0.157044 | 0.165619 | 0.008575 | 0.157044 | 641666.000000 | 688611.000000 |
| tus_rec_2024_qc | Nq | 3.000000 | 3.000000 | 32.038627 | 32.816705 | 0.778078 | 32.816705 | 0.860396 | 0.865764 | 0.005368 | 0.865764 | 0.111760 | 0.115322 | 0.003561 | 0.112040 | 655788.000000 | 713263.000000 |
| synthetic_liver | half_window | 4.000000 | 2.000000 | 29.431510 | 29.452202 | 0.020691 | 29.452202 | 0.660091 | 0.660437 | 0.000346 | 0.660275 | 0.157044 | 0.158291 | 0.001247 | 0.157044 | 686604.000000 | 690898.000000 |
| tus_rec_2024_qc | half_window | 4.000000 | 2.000000 | 32.794174 | 32.818661 | 0.024488 | 32.816705 | 0.865527 | 0.865787 | 0.000260 | 0.865764 | 0.110480 | 0.112040 | 0.001560 | 0.112040 | 712137.000000 | 713677.000000 |

