# CGA: capacity regulation, not an accuracy-boost claim

## Structural changes
| dataset | transition | delta_G | population_percent | delta_psnr | delta_lpips |
|---|---|---|---|---|---|
| Liver | GD -> RGD | 422563 | 67.2754 | -0.0143943 | -0.00161188 |
| Liver | RGD -> RGD+OP | -407378 | -38.7731 | 0.0087957 | 0.0015654 |
| Liver | RGD -> RGD+LP | -405733 | -38.6165 | 0.0196663 | 0.000731995 |
| Liver | RGD+LP -> Full | 43672 | 6.77149 | -0.0018494 | -0.000497626 |
| Arm | GD -> RGD | 343044 | 50.4816 | -0.0765283 | 0.000502935 |
| Arm | RGD -> RGD+OP | -332720 | -32.5371 | 0.0645369 | -0.000226849 |
| Arm | RGD -> RGD+LP | -331741 | -32.4414 | 0.0695633 | -0.000923958 |
| Arm | RGD+LP -> Full | 22418 | 3.24501 | 0.0386384 | 0.000701491 |


RGD grows population relative to GD; OP and LP reduce it relative to unpruned
RGD. Full retains additional primitives relative to LP. Full−LP PSNR is
-.0018494dB Liver and +.0386384dB Arm; LPIPS -.000497626 and +.000701491.
SUPPORTED: “path-aware pruning selectively retains additional primitives while
keeping reconstruction quality within a narrow range.” NOT “improves all metrics.”
Protected counts are event-summed decisions, not distinct Gaussian identities.

## qd x B: effective selection controls

PSNR ranges .0144313/.0131710dB; LPIPS .001205274/.001137021;
G ranges675559–711076 /707735–724242. Default(.98,.05) lies in the observed
stable-quality region. qd sets the positive-responsibility quantile; B adds an
approximately capped candidate budget relative to total population. Both feed
a threshold with >= ties, NOT strict TopK.

selection_behavior_summary.tsv classifies every point's12 adaptation events.
Limiter attribution is inferred from logged positive-score count, total N,
selected count, code quantile/topk rule and a2-count tolerance; original full
score vectors/pre-budget quantile thresholds are not archived. Do not claim
exact selected Gaussian identity equivalence across independent runs.

Liver: B=.001 is budget-limited12/12 at qd.96/.98 and10/12 at.99;
(.96,.005) budget9/12, quantile3/12. All other points quantile12/12.
Arm: B=.001 budget12/12 at.96,11/12 at.98,0/12 at.99;
all B=.005/.05 points quantile12/12. No unexplained count-classification cases.
Where the cap is loose, changing B does not change the selection rule for a
given score vector. Small differences across separately trained models are not
evidence of an active budget effect. Where the cap binds, qd changes can be masked.
Thus quality stability is supported, but not equivalence of all combinations.

## qp x qA: local saturation vs genuinely active path protection

PSNR ranges .0144032/.0090822dB; LPIPS .001576368/.001072631;
G ranges661306–722969 /701034–728740. All18 runs x12 events have local
contribution threshold exactly0 for qp=.05/.10/.15. The <= rule selects tied
zero-C primitives, so qp does NOT continuously increase pruning aggressiveness
in this sampled interval. This is local quantile saturation, not evidence that
an active qp control has no effect.

qA is NOT saturated: protection fractions track~30%,20%,10% for qA=.7,.8,.9.
At default qp=.1, path-protected candidate decisions are429680/234708/79070
Liver and216056/119549/50881 Arm. Raising qA narrows protection and lowers
final population. Its thresholds and decisions genuinely change. Lower qA means
more protection / less aggressive pruning, not more pruning. Default(.1,.8)
is in the stable-quality region, but the qp claim must acknowledge zero ties.

## tau_alpha

PSNR ranges .0159422/.0117541dB; LPIPS .002303379/.001112590;
G ranges565867–732376 /639733–738377. Raising threshold5e-5→5e-4 monotonically
reduces population while PSNR changes little. Default1e-4 is intermediate in
population, not guaranteed metric optimum. Together the grids show capacity
regulation without a large reconstruction penalty for the sampled settings.

## Quality–population: KEEP

The scatter adds the unpruned RGD populations (~1.051M Liver /1.023M Arm),
which are outside the hyperparameter grid populations, and connects structural
and numeric controls. It illustrates a broad population span with a narrow PSNR
band, not a fitted causal relation, optimum, efficiency ratio or Pareto proof.
No connecting trend line; datasets separate; independent axis scales labeled.
LPIPS counterpart is retained for review. H3 keeps four evidence classes.

## Claim adjudication

A quality stability: **SUPPORTED** for tested settings/cases, not universal.
B tunable population: **SUPPORTED**, especially tau_alpha/qA and active budgets.
C controls work as expected: **PARTIALLY SUPPORTED** as a blanket claim:
qd/B and qA show intended behavior, but qp lies on a zero-threshold plateau.
Residual responsibility targets weighted reconstruction error by code; final
metrics/scatter alone do not spatially prove that every added primitive lies in
an under-reconstructed region. Use formulation plus event/structural evidence,
not a standalone sensitivity proof of spatial reallocation.

## qd_by_B raw points

| dataset | point | final_psnr | final_lpips | final_num_gaussians | delta_psnr | delta_lpips |
|---|---|---|---|---|---|---|
| synthetic_liver | 0.96_0.001 | 29.4561 | 0.157492 | 675713 | 0.0038586 | 0.00044811 |
| synthetic_liver | 0.96_0.005 | 29.4505 | 0.157351 | 703443 | -0.0016635 | 0.000307317 |
| synthetic_liver | 0.96_0.05 | 29.4416 | 0.158069 | 711076 | -0.0105727 | 0.00102535 |
| synthetic_liver | 0.98_0.001 | 29.4527 | 0.158007 | 675737 | 0.0004661 | 0.000962862 |
| synthetic_liver | 0.98_0.005 | 29.4533 | 0.157401 | 688560 | 0.0010634 | 0.000356976 |
| synthetic_liver | 0.98_0.05 | 29.4522 | 0.157044 | 688611 | 0 | 0 |
| synthetic_liver | 0.99_0.001 | 29.452 | 0.156864 | 675559 | -0.0002218 | -0.000179923 |
| synthetic_liver | 0.99_0.005 | 29.4507 | 0.157191 | 678828 | -0.0015447 | 0.000146892 |
| synthetic_liver | 0.99_0.05 | 29.4543 | 0.157597 | 678026 | 0.0021039 | 0.000552959 |
| tus_rec_2024_qc | 0.96_0.001 | 32.8142 | 0.111529 | 711101 | -0.0024937 | -0.000510866 |
| tus_rec_2024_qc | 0.96_0.005 | 32.8124 | 0.111184 | 724242 | -0.0042852 | -0.000855374 |
| tus_rec_2024_qc | 0.96_0.05 | 32.8116 | 0.111739 | 724108 | -0.0051351 | -0.000300889 |
| tus_rec_2024_qc | 0.98_0.001 | 32.8107 | 0.111711 | 711098 | -0.0060518 | -0.000328668 |
| tus_rec_2024_qc | 0.98_0.005 | 32.8143 | 0.111268 | 713979 | -0.0024427 | -0.000771528 |
| tus_rec_2024_qc | 0.98_0.05 | 32.8167 | 0.11204 | 713263 | 0 | 0 |
| tus_rec_2024_qc | 0.99_0.001 | 32.8139 | 0.112015 | 707877 | -0.0028507 | -2.5152e-05 |
| tus_rec_2024_qc | 0.99_0.005 | 32.8238 | 0.110903 | 707735 | 0.0071192 | -0.00113702 |
| tus_rec_2024_qc | 0.99_0.05 | 32.8229 | 0.110959 | 708435 | 0.0062296 | -0.00108082 |


## qp_by_qA raw points

| dataset | point | final_psnr | final_lpips | final_num_gaussians | delta_psnr | delta_lpips |
|---|---|---|---|---|---|---|
| synthetic_liver | 0.05_0.7 | 29.4484 | 0.157682 | 722969 | -0.0038469 | 0.000637692 |
| synthetic_liver | 0.05_0.8 | 29.4438 | 0.157238 | 688566 | -0.0083901 | 0.000194303 |
| synthetic_liver | 0.05_0.9 | 29.4465 | 0.15849 | 661306 | -0.0056806 | 0.00144576 |
| synthetic_liver | 0.1_0.7 | 29.4429 | 0.157712 | 722679 | -0.0092843 | 0.0006677 |
| synthetic_liver | 0.1_0.8 | 29.4522 | 0.157044 | 688611 | 0 | 0 |
| synthetic_liver | 0.1_0.9 | 29.4561 | 0.156913 | 661460 | 0.0038832 | -0.000130608 |
| synthetic_liver | 0.15_0.7 | 29.4442 | 0.157002 | 722517 | -0.0080358 | -4.17e-05 |
| synthetic_liver | 0.15_0.8 | 29.4456 | 0.157313 | 688682 | -0.0066058 | 0.000268884 |
| synthetic_liver | 0.15_0.9 | 29.4573 | 0.157955 | 662097 | 0.0051189 | 0.000911161 |
| tus_rec_2024_qc | 0.05_0.7 | 32.82 | 0.111255 | 728740 | 0.0032528 | -0.000785167 |
| tus_rec_2024_qc | 0.05_0.8 | 32.8161 | 0.111518 | 713655 | -0.0005754 | -0.00052181 |
| tus_rec_2024_qc | 0.05_0.9 | 32.8233 | 0.111322 | 701034 | 0.0066189 | -0.000717617 |
| tus_rec_2024_qc | 0.1_0.7 | 32.8142 | 0.111088 | 728551 | -0.0024633 | -0.000951302 |
| tus_rec_2024_qc | 0.1_0.8 | 32.8167 | 0.11204 | 713263 | 0 | 0 |
| tus_rec_2024_qc | 0.1_0.9 | 32.8214 | 0.11138 | 702094 | 0.0046608 | -0.000659868 |
| tus_rec_2024_qc | 0.15_0.7 | 32.8216 | 0.111328 | 727557 | 0.0048676 | -0.000711797 |
| tus_rec_2024_qc | 0.15_0.8 | 32.8224 | 0.111765 | 713286 | 0.0056868 | -0.000274257 |
| tus_rec_2024_qc | 0.15_0.9 | 32.8175 | 0.110967 | 702379 | 0.0008189 | -0.00107263 |


## tau_alpha raw points

| dataset | point | final_psnr | final_lpips | final_num_gaussians | delta_psnr | delta_lpips |
|---|---|---|---|---|---|---|
| synthetic_liver | 5e-05 | 29.4462 | 0.157011 | 732376 | -0.0059823 | -3.2517e-05 |
| synthetic_liver | 0.0001 | 29.4522 | 0.157044 | 688611 | 0 | 0 |
| synthetic_liver | 0.0005 | 29.4622 | 0.159315 | 565867 | 0.0099599 | 0.00227086 |
| tus_rec_2024_qc | 5e-05 | 32.8128 | 0.111071 | 738377 | -0.0039076 | -0.000968948 |
| tus_rec_2024_qc | 0.0001 | 32.8167 | 0.11204 | 713263 | 0 | 0 |
| tus_rec_2024_qc | 0.0005 | 32.8246 | 0.112183 | 639733 | 0.0078465 | 0.000143642 |


## Event audit

| dataset | panel | point | row_id | events | budget_events | quantile_events | tie_or_other_events | zero_C_threshold_events | selected_min | selected_max | path_protection_fraction_min | path_protection_fraction_max | path_protected_candidate_decisions | total_pruned | total_spawned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Liver | qd_by_B | 0.96_0.001 | h2_016 | 12 | 12 | 0 | 0 | 12 | 692 | 1000 | 0.2 | 0.200001 | 225761 | 339317 | 15030 |
| Liver | qd_by_B | 0.96_0.005 | h2_017 | 12 | 9 | 3 | 0 | 12 | 2849 | 5037 | 0.2 | 0.200001 | 240732 | 374999 | 78442 |
| Liver | qd_by_B | 0.96_0.05 | h2_018 | 12 | 0 | 12 | 0 | 12 | 2852 | 6813 | 0.2 | 0.200001 | 245998 | 387098 | 98174 |
| Liver | qd_by_B | 0.98_0.001 | h2_019 | 12 | 12 | 0 | 0 | 12 | 692 | 1000 | 0.2 | 0.200002 | 223534 | 339282 | 15019 |
| Liver | qd_by_B | 0.98_0.005 | h2_020 | 12 | 0 | 12 | 0 | 12 | 1400 | 3405 | 0.2 | 0.200001 | 233256 | 358058 | 46618 |
| Liver | qd_by_B | 0.98_0.05 | h2_001 | 12 | 0 | 12 | 0 | 12 | 1395 | 3410 | 0.2 | 0.200001 | 234708 | 357885 | 46496 |
| Liver | qd_by_B | 0.99_0.001 | h2_021 | 12 | 10 | 2 | 0 | 12 | 690 | 1000 | 0.2 | 0.200001 | 225820 | 339329 | 14888 |
| Liver | qd_by_B | 0.99_0.005 | h2_022 | 12 | 0 | 12 | 0 | 12 | 692 | 1705 | 0.2 | 0.200001 | 228124 | 343559 | 22387 |
| Liver | qd_by_B | 0.99_0.05 | h2_023 | 12 | 0 | 12 | 0 | 12 | 692 | 1704 | 0.2 | 0.200001 | 227138 | 344295 | 22321 |
| Liver | qp_by_qA | 0.05_0.7 | h2_024 | 12 | 0 | 0 | 0 | 12 | 1449 | 3409 | 0.3 | 0.300001 | 428801 | 323935 | 46904 |
| Liver | qp_by_qA | 0.05_0.8 | h2_025 | 12 | 0 | 0 | 0 | 12 | 1398 | 3408 | 0.2 | 0.200001 | 233750 | 357857 | 46423 |
| Liver | qp_by_qA | 0.05_0.9 | h2_026 | 12 | 0 | 0 | 0 | 12 | 1340 | 3404 | 0.1 | 0.100001 | 79381 | 384582 | 45888 |
| Liver | qp_by_qA | 0.1_0.7 | h2_027 | 12 | 0 | 0 | 0 | 12 | 1455 | 3410 | 0.3 | 0.300001 | 429680 | 324313 | 46992 |
| Liver | qp_by_qA | 0.1_0.8 | h2_001 | 12 | 0 | 0 | 0 | 12 | 1395 | 3410 | 0.2 | 0.200001 | 234708 | 357885 | 46496 |
| Liver | qp_by_qA | 0.1_0.9 | h2_028 | 12 | 0 | 0 | 0 | 12 | 1345 | 3401 | 0.1 | 0.100001 | 79070 | 384424 | 45884 |
| Liver | qp_by_qA | 0.15_0.7 | h2_029 | 12 | 0 | 0 | 0 | 12 | 1442 | 3410 | 0.3 | 0.300001 | 427384 | 324377 | 46894 |
| Liver | qp_by_qA | 0.15_0.8 | h2_030 | 12 | 0 | 0 | 0 | 12 | 1398 | 3407 | 0.2 | 0.200001 | 232434 | 357817 | 46499 |
| Liver | qp_by_qA | 0.15_0.9 | h2_031 | 12 | 0 | 0 | 0 | 12 | 1349 | 3402 | 0.1 | 0.100001 | 79477 | 383920 | 46017 |
| Liver | tau_alpha | 5e-05 | h2_032 | 12 | 0 | 0 | 0 | 12 | 1484 | 3412 | 0.2 | 0.200001 | 208423 | 315489 | 47865 |
| Liver | tau_alpha | 0.0001 | h2_001 | 12 | 0 | 0 | 0 | 12 | 1395 | 3410 | 0.2 | 0.200001 | 234708 | 357885 | 46496 |
| Liver | tau_alpha | 0.0005 | h2_033 | 12 | 0 | 0 | 0 | 12 | 1163 | 3393 | 0.2 | 0.200001 | 277062 | 475643 | 41510 |
| Arm | qd_by_B | 0.96_0.001 | h2_050 | 12 | 12 | 0 | 0 | 12 | 718 | 1001 | 0.2 | 0.200001 | 120128 | 306376 | 17477 |
| Arm | qd_by_B | 0.96_0.005 | h2_051 | 12 | 0 | 12 | 0 | 12 | 1462 | 2782 | 0.2 | 0.200001 | 121511 | 318891 | 43133 |
| Arm | qd_by_B | 0.96_0.05 | h2_052 | 12 | 0 | 12 | 0 | 12 | 1459 | 2771 | 0.2 | 0.200001 | 123507 | 319044 | 43152 |
| Arm | qd_by_B | 0.98_0.001 | h2_053 | 12 | 11 | 1 | 0 | 12 | 718 | 1001 | 0.2 | 0.200001 | 120060 | 306188 | 17286 |
| Arm | qd_by_B | 0.98_0.005 | h2_054 | 12 | 0 | 12 | 0 | 12 | 728 | 1385 | 0.2 | 0.200001 | 120197 | 307228 | 21207 |
| Arm | qd_by_B | 0.98_0.05 | h2_035 | 12 | 0 | 12 | 0 | 12 | 724 | 1393 | 0.2 | 0.200001 | 119549 | 307893 | 21156 |
| Arm | qd_by_B | 0.99_0.001 | h2_055 | 12 | 0 | 12 | 0 | 12 | 362 | 698 | 0.2 | 0.200001 | 118669 | 302533 | 10410 |
| Arm | qd_by_B | 0.99_0.005 | h2_056 | 12 | 0 | 12 | 0 | 12 | 362 | 692 | 0.2 | 0.200001 | 120386 | 302779 | 10514 |
| Arm | qd_by_B | 0.99_0.05 | h2_057 | 12 | 0 | 12 | 0 | 12 | 362 | 698 | 0.2 | 0.200001 | 120357 | 301996 | 10431 |
| Arm | qp_by_qA | 0.05_0.7 | h2_058 | 12 | 0 | 0 | 0 | 12 | 727 | 1386 | 0.3 | 0.300001 | 215290 | 292612 | 21352 |
| Arm | qp_by_qA | 0.05_0.8 | h2_059 | 12 | 0 | 0 | 0 | 12 | 721 | 1386 | 0.2 | 0.200001 | 121248 | 307554 | 21209 |
| Arm | qp_by_qA | 0.05_0.9 | h2_060 | 12 | 0 | 0 | 0 | 12 | 720 | 1384 | 0.1 | 0.100001 | 50963 | 320100 | 21134 |
| Arm | qp_by_qA | 0.1_0.7 | h2_061 | 12 | 0 | 0 | 0 | 12 | 727 | 1391 | 0.3 | 0.300002 | 216056 | 292726 | 21277 |
| Arm | qp_by_qA | 0.1_0.8 | h2_035 | 12 | 0 | 0 | 0 | 12 | 724 | 1393 | 0.2 | 0.200001 | 119549 | 307893 | 21156 |
| Arm | qp_by_qA | 0.1_0.9 | h2_062 | 12 | 0 | 0 | 0 | 12 | 723 | 1389 | 0.1 | 0.100002 | 50881 | 319026 | 21120 |
| Arm | qp_by_qA | 0.15_0.7 | h2_063 | 12 | 0 | 0 | 0 | 12 | 729 | 1395 | 0.3 | 0.300001 | 213253 | 293798 | 21355 |
| Arm | qp_by_qA | 0.15_0.8 | h2_064 | 12 | 0 | 0 | 0 | 12 | 720 | 1394 | 0.2 | 0.200001 | 121179 | 307835 | 21121 |
| Arm | qp_by_qA | 0.15_0.9 | h2_065 | 12 | 0 | 0 | 0 | 12 | 724 | 1388 | 0.1 | 0.100001 | 51252 | 318673 | 21052 |
| Arm | tau_alpha | 5e-05 | h2_066 | 12 | 0 | 0 | 0 | 12 | 741 | 1407 | 0.2 | 0.200001 | 107102 | 283338 | 21715 |
| Arm | tau_alpha | 0.0001 | h2_035 | 12 | 0 | 0 | 0 | 12 | 724 | 1393 | 0.2 | 0.200001 | 119549 | 307893 | 21156 |
| Arm | tau_alpha | 0.0005 | h2_067 | 12 | 0 | 0 | 0 | 12 | 676 | 1293 | 0.2 | 0.200001 | 151733 | 379872 | 19605 |


## Complete ranges / default positions

| dataset | panel | n | default_point | psnr_min | psnr_max | psnr_range | psnr_default | ssim_min | ssim_max | ssim_range | ssim_default | lpips_min | lpips_max | lpips_range | lpips_default | G_min | G_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_liver | qd_by_B | 9.000000 | 0.98_0.05 | 29.441629 | 29.456060 | 0.014431 | 29.452202 | 0.660275 | 0.660822 | 0.000547 | 0.660275 | 0.156864 | 0.158069 | 0.001205 | 0.157044 | 675559.000000 | 711076.000000 |
| tus_rec_2024_qc | qd_by_B | 9.000000 | 0.98_0.05 | 32.810653 | 32.823824 | 0.013171 | 32.816705 | 0.865679 | 0.865805 | 0.000126 | 0.865764 | 0.110903 | 0.112040 | 0.001137 | 0.112040 | 707735.000000 | 724242.000000 |
| synthetic_liver | qp_by_qA | 9.000000 | 0.1_0.8 | 29.442918 | 29.457321 | 0.014403 | 29.452202 | 0.660199 | 0.660614 | 0.000415 | 0.660275 | 0.156913 | 0.158490 | 0.001576 | 0.157044 | 661306.000000 | 722969.000000 |
| tus_rec_2024_qc | qp_by_qA | 9.000000 | 0.1_0.8 | 32.814242 | 32.823324 | 0.009082 | 32.816705 | 0.865668 | 0.865819 | 0.000152 | 0.865764 | 0.110967 | 0.112040 | 0.001073 | 0.112040 | 701034.000000 | 728740.000000 |
| synthetic_liver | tau_alpha | 3.000000 | 0.000100 | 29.446220 | 29.462162 | 0.015942 | 29.452202 | 0.660275 | 0.660851 | 0.000577 | 0.660275 | 0.157011 | 0.159315 | 0.002303 | 0.157044 | 565867.000000 | 732376.000000 |
| tus_rec_2024_qc | tau_alpha | 3.000000 | 0.000100 | 32.812797 | 32.824551 | 0.011754 | 32.816705 | 0.865607 | 0.865832 | 0.000225 | 0.865764 | 0.111071 | 0.112183 | 0.001113 | 0.112040 | 639733.000000 | 738377.000000 |



## Combined quality–population extent

| dataset | unique_points | G_min | G_max | PSNR_min | PSNR_max | LPIPS_min | LPIPS_max |
|---|---|---|---|---|---|---|---|
| Liver | 23 | 565867 | 1.05067e+06 | 29.4344 | 29.4622 | 0.15681 | 0.159315 |
| Arm | 23 | 639733 | 1.02259e+06 | 32.7085 | 32.8246 | 0.110903 | 0.112262 |
