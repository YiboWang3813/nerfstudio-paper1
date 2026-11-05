# Pre-Freeze GS-Strict Probe Log

Date: 2026-06-27 HKT

Status: historical evidence archive; superseded by the 2026-06-26 production
baseline freeze in `docs/production/multi_comparisons.md`.

This note records pre-freeze and preproduction GS-strict probe results that were
previously present only as TSV tables under:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/
outputs/preproduction/baseline_reproduction/ultrags_baseline/
```

The purpose is preservation: if those output folders are deleted, the
experiment intent, measured results, and conclusions should remain recoverable.

## Scope and Caveats

These runs were useful for protocol selection, bug diagnosis, and baseline
calibration. They are not paper-facing production results.

Important caveats:

- Several early tables still contain row-level `output_dir` fields pointing to
  old `outputs/production/.../runs/dc_tuning` locations. The tables themselves
  were moved into `outputs/preproduction/...` after the production/preproduction
  scope split.
- Some rows used pre-freeze protocol variants such as best-checkpoint selection,
  native `tus_rec_2024`, experimental Ours losses, or method parameter probes.
  The fixed production protocol later standardized final-checkpoint reporting,
  `0.8 L1 + 0.2 SSIM`, method-specific production baselines, and dataset-QC
  choices.
- Do not use these rows as the final comparison table. Use them only to
  understand why later production aliases were frozen the way they were.

## Final Decisions Preserved From These Probes

- Native `tus_rec_2024/010_LH_Par_C_DtP` is not a valid production Stage-1
  representative. It was replaced by `tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370`.
- Ours should use the method-native bounded scale setting in the fixed
  production protocol; pure `exp` scale is not required to make Ours competitive.
- Loss input clamping is not a hidden decisive factor. Keep it disabled in the
  primary protocol for consistency.
- The first TUS-QC UltraGray lead came from a method/protocol choice that was
  later superseded by the fixed local-window baseline. UltraGray-specific
  window-size evidence is recorded separately in
  `docs/production/problems_during_production/ultragray_local_window_size_ablation.md`.
- UltraGauss can be very strong on the TUS-QC representative under some
  learning-rate/init probes, but those probe rows were not adopted as the frozen
  six-GS production baseline.
- UltraGS baseline needed explicit slice-thickness and DC calibration. The
  detailed production-facing calibration is recorded in
  `docs/production/problems_during_production/ultrags_baseline_slice_thickness_260626.md`.

## 1. Early Four-Dataset Ours Calibration

Source:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/stage1_calibration_ours_newdc_per_run.tsv
```

Purpose: check the then-current Ours row with a new density-control setting on
one representative sweep per dataset.

| Dataset | Sweep | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | 32.4227 | 0.01353 | 0.8606 | 0.1470 | 0.1725 | 0.0589 | 871,905 |
| `spine_phantom` | `left1` | 31.0971 | 0.01366 | 0.7912 | 0.1746 | 0.2189 | 0.1114 | 852,542 |
| `synthetic_liver` | `l1` | 29.6441 | 0.02320 | 0.6764 | 0.2463 | 0.1960 | 0.0961 | 810,808 |
| `tdsc_abus_2023` | `validation_100` | 27.6138 | 0.02703 | 0.7826 | 0.2781 | 0.2159 | 0.0724 | 912,758 |

Conclusion:

- The Ours row was viable across all four representative sweeps, but TUS-QC and
  TDSC were not strong enough to close later baseline gaps.
- This table is superseded by the corrected production freeze. It should be
  read as an early Ours sanity pass, not as the final Ours method row.

## 2. Spine Phantom Stage-1 and DC Tuning

Sources:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/stage1_calibration_spine_phantom_left1_per_run.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/dc_tuning_spine_phantom_left1_gaussianpile_ultrags.tsv
```

### Stage-1 Ranking

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 31.0954 | 0.01366 | 0.7908 | 0.1753 | 0.2181 | 0.1117 | 851,543 |
| `ultragray` | 30.8929 | 0.01486 | 0.7696 | 0.1439 | 0.1784 | 0.1096 | 895,949 |
| `slice_adapted_3dgs` | 29.0576 | 0.01780 | 0.7582 | 0.2168 | 0.2194 | 0.1377 | 536,297 |
| `ultragauss` | 28.9346 | 0.01886 | 0.6541 | 0.1841 | 0.1563 | 0.1325 | 529,387 |
| `gaussianpile` | 28.0611 | 0.01863 | 0.6716 | 0.4633 | 0.3359 | 0.1986 | 17,071 |
| `ultrags_baseline` | 25.7479 | 0.02224 | 0.7018 | 0.3681 | 0.2983 | 0.2011 | 274,773 |

### GaussianPile DC Tuning

Best GaussianPile rows in the DC sweep:

| Tag | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gp_p002_obs4_score025` | 31.0104 | 0.01444 | 0.7926 | 0.1389 | 0.2061 | 0.0975 | 927,172 |
| `gp_p002_obs8_score015` | 31.0062 | 0.01444 | 0.7927 | 0.1405 | 0.2062 | 0.0977 | 926,894 |
| `gp_p005_obs8_score025` | 30.9322 | 0.01476 | 0.7837 | 0.1471 | 0.2120 | 0.0989 | 740,055 |
| `gp_p005_obs4_score025` | 30.9321 | 0.01476 | 0.7835 | 0.1477 | 0.2127 | 0.0990 | 739,846 |

Best UltraGS baseline rows in the same DC sweep:

| Tag | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ugs_scale01_dist100_reset_off` | 26.9212 | 0.01993 | 0.7076 | 0.2888 | 0.2261 | 0.1861 | 1,076,525 |
| `ugs_op001_reset_off` | 26.6306 | 0.02034 | 0.7195 | 0.2636 | 0.2215 | 0.1771 | 1,072,078 |
| `ugs_scale01_op001_reset_off` | 26.6306 | 0.02035 | 0.7193 | 0.2637 | 0.2190 | 0.1772 | 1,071,873 |

Conclusion:

- The initial GaussianPile Stage-1 row collapsed to only 17k Gaussians and was
  not representative.
- Low-opacity GaussianPile pruning around `p002` recovered the method to about
  31.01 PSNR and about 927k Gaussians on `spine_phantom/left1`.
- UltraGS baseline remained much weaker on this sweep even after local tuning,
  and also retained too many Gaussians in the best probe rows.
- These rows supported the later decision to treat baseline DC and thickness
  settings as protocol-sensitive rather than method-intrinsic conclusions.

## 3. Synthetic Liver Stage-1 and Local DC Probes

Sources:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/stage1_calibration_synthetic_liver_per_run.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/dc_tuning_synthetic_liver_l1_ours_dc.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/gs_threshold_fix_synthetic_liver_l1_per_run.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/slice_rerun_synthetic_liver_l1_per_run.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/ultragray_dc_tuning_synthetic_liver_l1_per_run.tsv
```

### Stage-1 Ranking

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 29.6441 | 0.02320 | 0.6764 | 0.2463 | 0.1960 | 0.0961 | 810,808 |
| `slice_adapted_3dgs` | 28.7385 | 0.02559 | 0.6366 | 0.1653 | 0.1491 | 0.0992 | 769,742 |
| `ultragauss` | 28.6055 | 0.02649 | 0.6036 | 0.1490 | 0.1344 | 0.0987 | 898,846 |
| `ultragray` | 28.0642 | 0.02818 | 0.5874 | 0.1614 | 0.1364 | 0.1087 | 898,193 |
| `gaussianpile` | 26.4533 | 0.03373 | 0.5405 | 0.3412 | 0.2207 | 0.1436 | 841,468 |
| `ultrags_baseline` | 25.3704 | 0.03716 | 0.6190 | 0.4707 | 0.2836 | 0.1654 | 932,960 |

### Ours DC Sweep

The Ours DC sweep changed pruning/low-confidence/attenuation-protection
thresholds. Representative top rows by PSNR:

| Tag | PSNR | LPIPS | DISTS | Final GS |
| --- | ---: | ---: | ---: | ---: |
| `ours_dc_A_p0010_lcq010_paq090` | 29.6534 | 0.2898 | 0.2157 | 132,827 |
| `ours_dc_B_p0005_lcq010_paq090` | 29.6510 | 0.2654 | 0.2056 | 219,810 |
| `ours_dc_C_p0010_lcq005_paq090` | 29.6488 | 0.2897 | 0.2140 | 132,588 |
| `ours_dc_H_p00025_lcq010_paq080` | 29.6487 | 0.2527 | 0.2012 | 370,627 |
| `ours_dc_P_p000015_lcq010_paq090` | 29.6451 | 0.2476 | 0.1981 | 755,167 |
| `ours_dc_O_p000010_lcq010_paq080` | 29.6442 | 0.2477 | 0.1977 | 811,569 |

Conclusion:

- The highest-PSNR Ours DC rows pruned too aggressively, leaving only about
  133k-220k Gaussians. That was not a healthy production setting.
- The row near 810k Gaussians preserved the healthier production budget while
  losing only about 0.009 dB versus the top PSNR row.
- This supported using a stable, budget-aware Ours DC setting rather than
  optimizing only representative-sweep PSNR.

### Other Synthetic Liver Probes

| Probe | Result | Conclusion |
| --- | --- | --- |
| UltraGS baseline threshold fix | 25.3704 PSNR, 932,960 Gaussians | Healthy count did not imply competitive quality. Later UltraGS work moved to slice-thickness calibration. |
| Slice-adapted rerun | 28.7385 final PSNR, best PSNR 28.8096 at 15k, 769,742 Gaussians | Validated a stable slice-adapted row, but final checkpoint was below Ours on this representative sweep. |
| UltraGray DC tuning | Best-checkpoint PSNR around 28.67 at 5k, final PSNR around 28.05-28.07 | UltraGray degraded after the best checkpoint under these settings; final-checkpoint reporting made this row weaker than best-checkpoint reporting. |

## 4. Invalid Native TUS-REC Stage-1

Source:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/stage1_calibration_tus_rec_2024_per_run.tsv
```

This used native `tus_rec_2024/010_LH_Par_C_DtP`, not the QC dataset. It was
later rejected as a production Stage-1 row because the raw native sweep contains
motion behavior outside the production QC protocol.

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slice_adapted_3dgs` | 30.3189 | 0.01968 | 0.8348 | 0.1714 | 0.1785 | 0.0861 | 646,944 |
| `ultragray` | 30.2881 | 0.02060 | 0.8315 | 0.1495 | 0.1513 | 0.0881 | 926,176 |
| `gaussianpile` | 30.0526 | 0.02091 | 0.8250 | 0.1486 | 0.1531 | 0.0900 | 983,402 |
| `ultragauss` | 29.3312 | 0.02251 | 0.7910 | 0.1905 | 0.1971 | 0.0979 | 811,510 |
| `ours` | 28.5930 | 0.02307 | 0.8073 | 0.2278 | 0.2178 | 0.1032 | 754,928 |
| `ultrags_baseline` | 21.8419 | 0.04103 | 0.6941 | 0.3326 | 0.3069 | 0.1590 | 1,173,943 |

Conclusion:

- Ours looked weak on this native TUS sweep, but the row should not be used for
  production ranking.
- The replacement production representative is
  `tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370`, where the QC crop matches
  the dataset README production protocol.

## 5. TUS-QC Stage-1 Before Final Freeze

Source:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/stage1_tus_rec_2024_qc_per_run.tsv
```

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ultragray` | 32.7623 | 0.01328 | 0.8570 | 0.1161 | 0.1371 | 0.0580 | 845,799 |
| `slice_adapted_3dgs` | 32.5032 | 0.01379 | 0.8675 | 0.1228 | 0.1553 | 0.0537 | 812,399 |
| `ours` | 32.4218 | 0.01352 | 0.8607 | 0.1463 | 0.1731 | 0.0590 | 872,289 |
| `gaussianpile` | 30.6329 | 0.01838 | 0.8240 | 0.1847 | 0.1604 | 0.0748 | 943,913 |
| `ultragauss` | 28.2898 | 0.02490 | 0.6459 | 0.2608 | 0.2512 | 0.1009 | 883,160 |
| `ultrags_baseline` | 24.2182 | 0.03244 | 0.7049 | 0.2799 | 0.2648 | 0.1505 | 1,078,531 |

Conclusion:

- This was the first QC-corrected TUS Stage-1 table. Ours recovered strongly
  versus the invalid native TUS run, but UltraGray and slice-adapted 3DGS were
  still above Ours on PSNR.
- The table triggered follow-up probes for loss alignment, scale activation,
  attenuation/integral settings, UltraGray behavior, and UltraGauss stability.
- This row was superseded by later GS-strict freeze decisions.

## 6. TUS-QC Loss and Scale Alignment

Sources:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_exp_scale_aligned.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_gs_loss_aligned_stage1.tsv
```

### Exp-Scale Aligned Table

| Method | Tag | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | `ours_exp_scale_aligned` | 32.7650 | 0.01282 | 0.8662 | 0.1470 | 0.1619 | 0.0562 | 870,402 |
| `slice_adapted_3dgs` | `slice_adapted_3dgs_exp_scale_aligned` | 32.5069 | 0.01378 | 0.8675 | 0.1233 | 0.1555 | 0.0537 | 812,215 |
| `ultragray` | `ultragray_exp_scale_aligned` | 32.4477 | 0.01369 | 0.8510 | 0.1312 | 0.1534 | 0.0593 | 863,448 |
| `ultragauss` | `ultragauss_exp_scale_aligned_lrfixed` | 32.0697 | 0.01418 | 0.8374 | 0.1583 | 0.1731 | 0.0637 | 799,840 |
| `gaussianpile` | `gaussianpile_exp_scale_aligned` | 30.9978 | 0.01710 | 0.8334 | 0.1661 | 0.1522 | 0.0718 | 952,649 |
| `ultrags_baseline` | `ultrags_baseline_exp_scale_aligned` | 24.1899 | 0.03246 | 0.7043 | 0.2795 | 0.2644 | 0.1505 | 1,076,762 |

### Loss-Aligned Table

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ultragray` | 32.9156 | 0.01308 | 0.8587 | 0.1133 | 0.1364 | 0.0567 | 860,900 |
| `ours` | 32.6555 | 0.01299 | 0.8652 | 0.1548 | 0.1737 | 0.0575 | 867,015 |
| `slice_adapted_3dgs` | 32.4988 | 0.01380 | 0.8674 | 0.1226 | 0.1550 | 0.0537 | 812,056 |
| `gaussianpile` | 30.6127 | 0.01834 | 0.8242 | 0.1819 | 0.1585 | 0.0757 | 937,182 |
| `ultrags_baseline` | 24.2462 | 0.03240 | 0.7053 | 0.2796 | 0.2639 | 0.1502 | 1,078,015 |

Conclusion:

- Aligning Ours to the exp-scale probe lifted Ours to 32.7650 PSNR and put it
  above the other rows in that specific table.
- Under the loss-aligned table, UltraGray remained ahead at 32.9156 PSNR.
- The two tables showed that the apparent ranking was very sensitive to the
  exact method/protocol knobs, so a fresh freeze with explicit method-native
  scale activation and baseline-specific settings was required.

## 7. TUS-QC Ours Attenuation, DC, Loss, LR, and Perceptual Probes

Sources:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_attn_dc_probe.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_attn_fine_probe.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_attn_window_probe.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_tuning.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_loss_tuning.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_scale_tuning.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_dc_protect_tuning.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_lr_tuning.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ours_perceptual_tuning.tsv
```

### Attenuation and Integral Probes

| Probe | Best PSNR | Key secondary metrics | Conclusion |
| --- | ---: | --- | --- |
| `ours_attn_dc_probe` | 32.7416 (`integral_3pt`) | LPIPS 0.1457, DISTS 0.1641, 859k GS | Integral attenuation improved over the first QC Stage-1 Ours row, but did not clearly beat the best later Ours window probe. |
| `ours_attn_fine_probe` | 32.7575 (`int_strength055`) | LPIPS 0.1439, DISTS 0.1622, 861k GS | Slightly stronger integral attenuation was directionally useful. |
| `ours_attn_window_probe` | 32.7680 (`int_min035_s055`) | LPIPS 0.1409, DISTS 0.1583, 861k GS | Best Ours TUS-QC probe in this family. `q2` was poor at 32.3082, so more quadrature was not automatically better. |

Detailed Ours window-probe rows:

| Tag | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours_int_min035_s055` | 32.7680 | 0.01286 | 0.8661 | 0.1409 | 0.1583 | 0.0561 | 861,315 |
| `ours_int_win4_s055` | 32.7403 | 0.01286 | 0.8658 | 0.1441 | 0.1605 | 0.0564 | 861,074 |
| `ours_int_attnlr004_s055` | 32.7364 | 0.01289 | 0.8657 | 0.1437 | 0.1615 | 0.0564 | 859,591 |
| `ours_int_q2_s055` | 32.3082 | 0.01334 | 0.8639 | 0.1425 | 0.1559 | 0.0583 | 826,828 |

### DC, Loss, Scale, LR, and Perceptual Probes

| Probe table | Best row | Best PSNR | Conclusion |
| --- | --- | ---: | --- |
| `tus_rec_2024_qc_ours_tuning.tsv` | `tusq_dc_a_grad75_sq98_mf05_lcq10_paq80` | 32.4251 | DC-only tuning barely improved the original QC Ours row; not enough. |
| `tus_rec_2024_qc_ours_loss_tuning.tsv` | `tusq_loss_b_l80_s20_base_dc` | 32.5041 | `0.8 L1 + 0.2 SSIM` beat `0.5/0.5`; helped define the later fair GS loss. |
| `tus_rec_2024_qc_ours_scale_tuning.tsv` | `tusq_scale_a_l65_s35_sw0` | 32.5478 | Scale-wall style variants were not decisive; some hurt PSNR. |
| `tus_rec_2024_qc_ours_dc_protect_tuning.tsv` | `tusq_dcprot_c_lcq02_paq90_sw0` | 32.5580 | DC protection settings were nearly flat around 32.554-32.558. |
| `tus_rec_2024_qc_ours_lr_tuning.tsv` | `tusq_lr_b_int010_hp010` | 32.5679 | Small LR refinements helped but did not match the best attenuation/window probes. |
| `tus_rec_2024_qc_ours_perceptual_tuning.tsv` | `tusq_perc_c_hp010` | 32.5607 | Auxiliary perceptual/high-pass losses did not give a decisive gain and were not retained in the fair comparison. |

Conclusion:

- The most useful Ours direction on this TUS-QC representative was the
  attenuation/integral/window setting, not broad DC threshold tuning.
- Additional auxiliary losses and perceptual probes were not adopted because
  they would change the comparison objective and did not provide a decisive
  improvement.
- These rows informed later diagnostics but were superseded by the fixed
  production Ours baseline.

## 8. TUS-QC Scale Fix, Scale Tail, Loss Clamp, UltraGray, and UltraGauss

Sources:

```text
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_protocol_sanity.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_scale_fix.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_scale_tail_diag.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_scale_tail_tune.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_mechanism_probe.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ultragauss_stability.tsv
outputs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/tables/tus_rec_2024_qc_ultragray_no_sh.tsv
```

### Protocol Sanity

| Row | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours_lossclamp` | 32.7566 | 0.01283 | 0.8662 | 0.1473 | 0.1616 | 0.0562 | 869,882 |
| `ultragray_lossclamp` | 32.5029 | 0.01351 | 0.8541 | 0.1236 | 0.1483 | 0.0589 | 862,626 |
| `slice_lossclamp` | 32.4988 | 0.01380 | 0.8673 | 0.1230 | 0.1552 | 0.0538 | 812,376 |
| `ultragray_scale03` | 32.4162 | 0.01374 | 0.8520 | 0.1267 | 0.1484 | 0.0591 | 878,385 |

Conclusion: loss input clamping did not materially explain the ranking. Keep
the primary protocol unclamped for consistency.

### Ours and UltraGauss Scale Fix

| Row | Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours_bounded_s3` | `ours` | 32.7662 | 0.01285 | 0.8660 | 0.1413 | 0.1591 | 0.0560 | 860,462 |
| `ug_lr0040` | `ultragauss` | 32.0311 | 0.01429 | 0.8342 | 0.1604 | 0.1747 | 0.0642 | 799,502 |
| `ug_lr0035` | `ultragauss` | 31.2587 | 0.01611 | 0.8013 | 0.2057 | 0.2022 | 0.0734 | 830,683 |
| `ug_lr0025` | `ultragauss` | 28.2910 | 0.02421 | 0.6724 | 0.3139 | 0.2632 | 0.1075 | 871,786 |

Conclusion: bounded Ours scale was stable and competitive. UltraGauss was very
sensitive to LR.

### Scale Tail Diagnostics and Tuning

| Row | Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours_scale_tail_diag_v2` | `ours` | 32.7633 | 0.01282 | 0.8662 | 0.1472 | 0.1621 | 0.0561 | 869,655 |
| `ultragauss_scale_tail_diag_v2` | `ultragauss` | 32.0443 | 0.01427 | 0.8346 | 0.1595 | 0.1729 | 0.0639 | 799,205 |
| `ours_sclr0025_max8` | `ours` | 32.7096 | 0.01291 | 0.8657 | 0.1384 | 0.1548 | 0.0564 | 851,156 |
| `ours_sclr0025_max5` | `ours` | 32.7036 | 0.01292 | 0.8656 | 0.1386 | 0.1556 | 0.0562 | 850,576 |
| `ultragauss_ptril00375` | `ultragauss` | 31.7085 | 0.01498 | 0.8207 | 0.1840 | 0.1882 | 0.0677 | 816,353 |
| `ultragauss_ptril0035` | `ultragauss` | 31.1867 | 0.01622 | 0.8001 | 0.2086 | 0.2044 | 0.0737 | 834,357 |

Conclusion: tighter Ours scale-tail constraints improved LPIPS/DISTS but cost
about 0.05-0.06 dB PSNR relative to `ours_bounded_s3`. The production decision
favored method-native bounded scale without adding extra scale-wall losses.

### UltraGray Mechanism and No-SH Probe

| Row | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ultragray_ug_attn_frozen` | 32.9223 | 0.01305 | 0.8593 | 0.1136 | 0.1358 | 0.0567 | 860,067 |
| `ultragray_no_sh` | 32.9135 | 0.01305 | 0.8589 | 0.1136 | 0.1370 | 0.0567 | 861,685 |
| `ultragray_ug_full_transwin` | 32.4288 | 0.01380 | 0.8510 | 0.1318 | 0.1538 | 0.0594 | 862,778 |

Conclusion: on this TUS-QC representative, UltraGray's strong row was not
explained by SH alone. Frozen attenuation / no-SH variants stayed high, while
the full-transmission-window variant was lower. This justified treating
UltraGray window/transmission behavior as a dedicated production problem rather
than a generic loss or SH artifact.

### UltraGauss Stability Probe

| Row | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ug_lr001` | 32.9726 | 0.01279 | 0.8608 | 0.1080 | 0.1377 | 0.0564 | 545,410 |
| `ug_init56` | 32.9691 | 0.01282 | 0.8595 | 0.1128 | 0.1411 | 0.0564 | 539,048 |
| `ug_lr0005` | 32.5983 | 0.01329 | 0.8526 | 0.1218 | 0.1473 | 0.0591 | 746,202 |

Conclusion: UltraGauss could exceed 32.9 PSNR on this representative with
specific LR/init probes, but the strongest rows retained only about 539k-545k
Gaussians. These were stability probes, not the frozen production alias.

## 9. UltraGS Baseline Natural-DC Synthetic Liver Evidence

Source:

```text
outputs/preproduction/baseline_reproduction/ultrags_baseline/natural_dc_full_30k/tables/stage1_calibration_synthetic_liver_per_run.tsv
```

| Method | Dataset | Sweep | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ultrags_baseline` | `synthetic_liver` | `l1` | 29.9949 | 0.02257 | 0.6716 | 0.1853 | 0.1837 | 0.0965 | 874,458 |

Conclusion:

- This natural-DC UltraGS baseline row was unexpectedly strong on
  `synthetic_liver/l1` and even exceeded the early Ours synthetic row.
- It was not accepted as final evidence because subsequent UltraGS investigation
  found slice-thickness/culling behavior that made small configured ratios
  effectively thicker than intended.
- The corrected conclusion and production recommendation are in
  `docs/production/problems_during_production/ultrags_baseline_slice_thickness_260626.md`.

## 10. Overall Lessons

- A single representative sweep can flip method ranking under small protocol
  changes. The fixed production protocol must be explicit about loss,
  checkpoint selection, scale activation, slice thickness/window parameters,
  density-control timing, and dataset QC.
- Best-checkpoint reporting hid some final-checkpoint degradation, especially
  for UltraGray on synthetic liver. The primary production comparison should
  use final 30k checkpoint rows unless a separate best-checkpoint supplement is
  explicitly declared.
- Gaussian count is a real health signal. Rows that maximize PSNR while pruning
  to 100k-200k Gaussians, or retaining only about 540k in a row expected to keep
  more capacity, should not automatically become production aliases.
- Ours TUS-QC improvements came mainly from physically meaningful attenuation
  and scale parameterization choices. Broad auxiliary-loss or perceptual-loss
  tuning was not compelling enough for the fair primary comparison.
- Baselines with physical slice/window assumptions need their own calibrated
  settings. This led directly to the separate thickness and window ablation
  documents for GaussianPile/slice-adapted/UltraGS/UltraGray.

## Supersession Map

| Historical output | Preserved conclusion | Superseded by |
| --- | --- | --- |
| `stage1_calibration_tus_rec_2024_per_run.tsv` | Native TUS row is invalid for production. | `tus_rec_2024_qc` dataset protocol and `multi_comparisons.md` decision log. |
| `stage1_tus_rec_2024_qc_per_run.tsv` | QC row restored Ours but exposed close baseline competition. | 2026-06-26 fixed GS production baseline. |
| `tus_rec_2024_qc_ours_*` | Ours best pre-freeze direction was attenuation/window tuning; auxiliary losses were not worth adopting. | Fixed Ours production alias and later GS-strict diagnostics. |
| `tus_rec_2024_qc_protocol_sanity.tsv` | Loss clamp is not decisive. | Primary unclamped protocol in `multi_comparisons.md`. |
| `tus_rec_2024_qc_ultragray_no_sh.tsv` and mechanism probe | UltraGray behavior is tied to transmission/window settings, not just SH. | UltraGray local-window ablation document. |
| `tus_rec_2024_qc_ultragauss_stability.tsv` | UltraGauss is LR/init sensitive on TUS-QC. | Fixed method-native production alias; do not promote probe rows directly. |
| `dc_tuning_spine_phantom_left1_gaussianpile_ultrags.tsv` | GaussianPile needed DC recovery; UltraGS remained weak. | Later thickness/DC calibration documents. |
| `natural_dc_full_30k/stage1_calibration_synthetic_liver_per_run.tsv` | UltraGS natural-DC synthetic result was suspiciously strong. | UltraGS slice-thickness/cull calibration document. |
