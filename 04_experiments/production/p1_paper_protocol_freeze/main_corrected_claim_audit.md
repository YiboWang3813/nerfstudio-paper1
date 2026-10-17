# Corrected main claim audit

Audit target is the supplied Results exchange copy, not a confirmed canonical LaTeX source. No manuscript was edited. Source SHA256: 90397171e1c451477642323323405992897c6f047d895bd8c11b09ffebb4ecac

Quality uses all frozen 6/9/33/30 cases, eight methods, equal sweep weights and unrounded ranks. Baselines quality are FINAL-REUSE.

## A. Still supported

- PSNR first on Liver, Spine and Breast; L1 first on all four. SSIM first on Liver/Spine/Breast, second on Arm. Breast wins all six metrics.
- Reconstruction accuracy remains above both NeRF methods on the evaluated datasets.
- Perceptual distances improve against legacy Ours, but are not best on every dataset.
- Pure-forward rendering exceeds a stated 30 FPS reference threshold on all measured cases; this does not establish end-to-end clinical/display latency.

## B. Exact numbers must change

All historical Ours values, PSNR increments, L1 percentages and 240–353 FPS text require replacement. The following rows cover the named baseline contrasts in the three comparison subsections; raw values are saved, not inferred from displayed rounding.

| dataset | baseline | baseline_PSNR | corrected_PSNR | PSNR_delta | baseline_SSIM | corrected_SSIM | L1_reduction_percent |
|---|---|---|---|---|---|---|---|
| synthetic_liver | nerfus_no_prior | 22.0717 | 29.2239 | 7.15221 | 0.597131 | 0.648545 | 41.3461 |
| synthetic_liver | ultrags_baseline | 28.0158 | 29.2239 | 1.20813 | 0.593675 | 0.648545 | 16.4482 |
| synthetic_liver | slice_adapted_3dgs | 26.7288 | 29.2239 | 2.49507 | 0.572487 | 0.648545 | 23.3541 |
| spine_phantom | nerfus_no_prior | 27.4567 | 30.8063 | 3.34966 | 0.69206 | 0.784377 | 23.8125 |
| spine_phantom | ultragray | 30.6262 | 30.8063 | 0.180136 | 0.755087 | 0.784377 | 7.93492 |
| spine_phantom | gaussianpile | 28.2243 | 30.8063 | 2.58205 | 0.716831 | 0.784377 | 26.5902 |
| tus_rec_2024_qc | nerfus_no_prior | 29.4267 | 33.7694 | 4.34262 | 0.836057 | 0.882168 | 31.9583 |
| tus_rec_2024_qc | ultragray | 33.8162 | 33.7694 | -0.0468816 | 0.832439 | 0.882168 | 12.0696 |
| tus_rec_2024_qc | slice_adapted_3dgs | 33.9551 | 33.7694 | -0.185781 | 0.889401 | 0.882168 | 1.08171 |
| tdsc_abus_2023 | nerfus_no_prior | 22.0047 | 28.7395 | 6.73485 | 0.650348 | 0.804734 | 49.8376 |
| tdsc_abus_2023 | ultragauss | 27.7615 | 28.7395 | 0.977994 | 0.781601 | 0.804734 | 6.10556 |
| tdsc_abus_2023 | slice_adapted_3dgs | 27.0838 | 28.7395 | 1.65575 | 0.792647 | 0.804734 | 13.897 |


Dataset counts must read 6 Liver / 9 Spine / 33 evaluated Arm QC segments / 30 Breast, separating 48 raw Arm sweeps. Single4090 training wording is incorrect: Batch1 used4080SUPER except two4090 anchors. d0 is fixed nominal/calibrated spacing, not verified runtime device metadata.

## C. Must weaken

- "Strongest fidelity across all four datasets" -> specify metric/dataset wins, including Arm exceptions.
- "Consistently sharper and richer than all baselines" -> inherited-case visual observations only; UltraGray/3DGS remain competitive. See qualitative audit.
- PPP preserves multi-sweep PSNR/SSIM, not all quality metrics; Liver representative LPIPS changes about .180 to .223.
- CGA quality stays in a narrow band; no significant-PSNR or LP-always-better claim.
- Full LGR and Adaptive advantages await the 11-case confirmation; no universal monotonic claim.
- rho=1 is the nominal fixed spacing, not an observed optimum/plateau.

## D. No longer supported / do not use

- Best PSNR/L1/SSIM on every dataset: false for Arm PSNR and SSIM.
- Arm PSNR exceeds UltraGray by .43dB: now lower by .0468816030303dB.
- Arm PSNR exceeds adapted3DGS: now lower by .1857811dB; Ours is third.
- Outperforms both adapted Gaussian methods on every dataset/metric: false.
- Less training time than both adapted baselines: not supported by current recorded times and not hardware-controlled.
- Covariance correction accelerates training: not established by shared-worker wall time.
- Final caption claiming uniformly finer structures than every baseline: not established by the inherited images.

## Full rank and display audit

| dataset | metric | raw_mean | displayed | rank | best | second_best |
|---|---|---|---|---|---|---|
| spine_phantom | psnr | 30.806319300000002 | 30.8063 | 1 | ours_corrected | ultragray |
| spine_phantom | l1 | 0.014462665477777779 | 0.0145 | 1 | ours_corrected | ultrags_baseline |
| spine_phantom | ssim | 0.7843765595555555 | 0.7844 | 1 | ours_corrected | ultragray |
| spine_phantom | lpips | 0.14539348744444444 | 0.1454 | 2 | ultragray | ours_corrected |
| spine_phantom | dists | 0.16560046355555555 | 0.1656 | 2 | ultragray | ours_corrected |
| spine_phantom | gmsd | 0.10489121444444445 | 0.1049 | 1 | ours_corrected | ultragray |
| synthetic_liver | psnr | 29.2239058 | 29.2239 | 1 | ours_corrected | ultrags_baseline |
| synthetic_liver | l1 | 0.024427324466666667 | 0.0244 | 1 | ours_corrected | ultrags_baseline |
| synthetic_liver | ssim | 0.6485454526666666 | 0.6485 | 1 | ours_corrected | nerfus_no_prior |
| synthetic_liver | lpips | 0.17985726116666667 | 0.1799 | 3 | ultragray | ultragauss |
| synthetic_liver | dists | 0.15065532383333333 | 0.1507 | 3 | ultragray | ultragauss |
| synthetic_liver | gmsd | 0.10085934368333334 | 0.1009 | 1 | ours_corrected | ultrags_baseline |
| tdsc_abus_2023 | psnr | 28.739507246666665 | 28.7395 | 1 | ours_corrected | ultragauss |
| tdsc_abus_2023 | l1 | 0.023635099243333332 | 0.0236 | 1 | ours_corrected | ultragauss |
| tdsc_abus_2023 | ssim | 0.8047335447666667 | 0.8047 | 1 | ours_corrected | slice_adapted_3dgs |
| tdsc_abus_2023 | lpips | 0.1763473747 | 0.1763 | 1 | ours_corrected | ultragauss |
| tdsc_abus_2023 | dists | 0.15693430233333333 | 0.1569 | 1 | ours_corrected | ultragauss |
| tdsc_abus_2023 | gmsd | 0.06735278896333334 | 0.0674 | 1 | ours_corrected | slice_adapted_3dgs |
| tus_rec_2024_qc | psnr | 33.769359575757576 | 33.7694 | 3 | slice_adapted_3dgs | ultragray |
| tus_rec_2024_qc | l1 | 0.011398074354545455 | 0.0114 | 1 | ours_corrected | slice_adapted_3dgs |
| tus_rec_2024_qc | ssim | 0.882168350030303 | 0.8822 | 2 | slice_adapted_3dgs | ours_corrected |
| tus_rec_2024_qc | lpips | 0.11115182277575758 | 0.1112 | 2 | slice_adapted_3dgs | ours_corrected |
| tus_rec_2024_qc | dists | 0.13394115415151514 | 0.1339 | 1 | ours_corrected | slice_adapted_3dgs |
| tus_rec_2024_qc | gmsd | 0.04736658716363636 | 0.0474 | 2 | slice_adapted_3dgs | ours_corrected |


## Corrected efficiency

| dataset | num_training_cases | training_seconds | all_case_FPS_mean | all_case_min_FPS | paper_matched_two_case_FPS | paper_FPS_cases |
|---|---|---|---|---|---|---|
| spine_phantom | 9 | 578.8118999534183 | 224.66436593403347 | 212.12621886990698 | 229.26347675774932 | left1_1;left3_2 |
| synthetic_liver | 6 | 545.5217044918487 | 243.88570146088205 | 236.57842789016985 | 240.16120006622583 | l1;l3 |
| tdsc_abus_2023 | 30 | 410.97991751829784 | 327.6825685206878 | 304.3498318133725 | 316.24683267012847 | validation_101;validation_116 |
| tus_rec_2024_qc | 33 | 487.0339401248052 | 275.77437746966575 | 56.896082461342445 | 293.1422933782769 | 010_RH_Per_C_PtD_qc_0006_0427;010_RH_Per_S_PtD_qc_0165_0364_s02 |


| panel | method | training_seconds | FPS | status | source | source_sha256 |
|---|---|---|---|---|---|---|
| synthetic_phantom | UltraNeRF | 475.1 | 315.2 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| synthetic_phantom | NeRF-US w/o prior | 334.1 | 344.9 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| synthetic_phantom | UltraGauss | 170.6 | 1269.4 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| synthetic_phantom | UltraGray | 195.1 | 940.4 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| synthetic_phantom | UltraGS baseline | 386.7 | 417.5 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| synthetic_phantom | Slice-adapted 3DGS | 413.5 | 291.4 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| synthetic_phantom | GaussianPile | 409.8 | 265.0 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| real | UltraNeRF | 450.7 | 314.7 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| real | NeRF-US w/o prior | 474.2 | 327.1 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| real | UltraGauss | 171.5 | 1321.0 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| real | UltraGray | 192.0 | 1190.3 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| real | UltraGS baseline | 441.1 | 403.1 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| real | Slice-adapted 3DGS | 460.1 | 307.4 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| real | GaussianPile | 438.2 | 252.7 | FINAL-REUSE_frozen_display_precision | /home/boywang25/ultrasound-scene-reconstruction/omniultra/docs/production/comparison_final_outputs/quantitative_tabs.md | 882d01c4f5729b34537a555772e6ea20c4eabfd645f94d6ccd569b644d7557c4 |
| synthetic_phantom | Ours corrected | 562.1668022226336 | 234.71233841198756 | corrected_equal_dataset_mean | corrected_efficiency_datasets.tsv |  |
| real | Ours corrected | 449.0069288215515 | 304.6945630242027 | corrected_equal_dataset_mean | corrected_efficiency_datasets.tsv |  |


Paper FPS uses the same two cases per dataset as every baseline, then equal-weight dataset means for Liver+Spine and Arm+Breast. All78-case FPS means/minima are provided separately and are not substituted into a two-case comparison. Timing means cover all frozen cases. FPS uses verified4090, 5warmup,20heldout frames,3passes, pure forward.

Baseline efficiency panel cells are reused verbatim at their frozen published precision. Historical Time aggregation was inconsistent (GS pooled sweeps, implicit equal datasets); this audit does not silently redefine baseline cells. Corrected Ours uses the P1 equal-dataset rule. These Time columns must carry the heterogeneity caveat, not support a controlled efficiency superiority claim. Raw baseline missing-time coverage is separately reported in main_complete_dataset_aggregates.tsv; missing is never zero.

No Batch2 or algorithm modification is authorized by this report.
