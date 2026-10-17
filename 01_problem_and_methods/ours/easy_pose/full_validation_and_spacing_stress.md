# Pose-Free Canonical Sweep Full Validation and Stress Plan

Date: 2026-06-18

Branch: `dev/ours-pose-free-sweep-20260618`

## Purpose

The pose-free contribution is not an accuracy-improvement module. Its value is
reducing the pose requirement from per-frame high-accuracy 6-DoF tracking to a
low-dimensional sweep protocol prior, mainly a calibrated inter-slice spacing,
for regular or near-regular offline one-sweep ultrasound acquisition.

The validation therefore reports pose-free retention and tracked-pose gap:

```text
PSNR gap = PSNR(canonical) - PSNR(file_pose)
LPIPS gap = LPIPS(canonical) - LPIPS(file_pose)
L1 gap = L1(canonical) - L1(file_pose)
SSIM gap = SSIM(canonical) - SSIM(file_pose)
```

Interpretation:

- strong pass: `PSNR gap >= -0.3 dB` without large LPIPS/L1/SSIM degradation;
- acceptable pass: `-0.6 dB <= PSNR gap < -0.3 dB`;
- fail: `PSNR gap < -0.8 dB` or obvious LPIPS/SSIM collapse.

`spine_phantom` is excluded from positive validation because Stage I/J/K showed
that its canonical failure is not fixed by QC windowing, scalar spacing, or
arc-length nearest-frame subset selection. It remains a limitation/negative case.

## Common Training Protocol

All production validation rows use:

- `1M` Gaussian initialization;
- `30k` iterations;
- final local-global renderer;
- responsibility density control;
- `holdout_interval` testing;
- best-checkpoint reporting by PSNR;
- Slurm on `gpu3` RTX 4090.

## Stage L: Multi-Sweep Full Validation

Script:

```text
scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_l_full_validation_30k.sh
```

Output:

```text
outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_l_full_validation_30k_20260618
```

Datasets and sweeps:

| dataset | sweeps | variants |
| --- | --- | --- |
| synthetic_liver | `l1,l2,l3,r1,r2,r3` | `file_pose`, `canonical_s100` |
| tus_rec_2024_qc | `010_LH_Per_L_DtP_qc_0005_0370`, `010_RH_Per_L_DtP_qc_0005_0574`, `011_LH_Per_S_PtD_qc_0008_0447`, `011_RH_Per_L_DtP_qc_0007_0584`, `010_LH_Per_C_DtP_qc_0140_0514` | `file_pose`, `canonical_s100` |
| tdsc_abus_2023 | `validation_100` to `validation_104` | `file_pose`, `canonical_s100` |

Total jobs: `32`.

Primary output:

- per-sweep gap table;
- dataset-level mean/median gap;
- count of strong pass / acceptable pass / fail.

Stage L result:

```text
status = finished
summary = outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_l_full_validation_30k_20260618/summary.tsv
rows = 32 jobs = 16 paired sweeps
```

Per-sweep best-checkpoint gaps:

| dataset | sweep | file PSNR | canonical PSNR | PSNR gap | LPIPS gap | L1 gap | SSIM gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver | l1 | 29.6503 | 29.6609 | +0.0105 | +0.0085 | -0.000073 | -0.00009 |
| synthetic_liver | l2 | 29.4464 | 29.4601 | +0.0137 | +0.0093 | -0.000090 | +0.00048 |
| synthetic_liver | l3 | 29.6729 | 29.6032 | -0.0697 | +0.0033 | +0.000106 | -0.00232 |
| synthetic_liver | r1 | 29.3959 | 29.3948 | -0.0012 | +0.0104 | -0.000044 | -0.00196 |
| synthetic_liver | r2 | 28.7366 | 28.7359 | -0.0007 | +0.0162 | -0.000064 | -0.00236 |
| synthetic_liver | r3 | 28.6185 | 28.5389 | -0.0795 | +0.0149 | +0.000115 | -0.00523 |
| tus_rec_2024_qc | 010_LH_Per_C_DtP_qc_0140_0514 | 33.8373 | 34.0610 | +0.2237 | +0.0056 | -0.000232 | +0.00110 |
| tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | 32.3188 | 32.4737 | +0.1549 | +0.0017 | -0.000317 | +0.00158 |
| tus_rec_2024_qc | 010_RH_Per_L_DtP_qc_0005_0574 | 32.9236 | 33.1790 | +0.2554 | +0.0051 | -0.000214 | +0.00146 |
| tus_rec_2024_qc | 011_LH_Per_S_PtD_qc_0008_0447 | 34.0558 | 34.4563 | +0.4005 | +0.0014 | -0.000299 | +0.00233 |
| tus_rec_2024_qc | 011_RH_Per_L_DtP_qc_0007_0584 | 32.9989 | 33.3241 | +0.3251 | +0.0032 | -0.000203 | +0.00229 |
| tdsc_abus_2023 | validation_100 | 27.4565 | 27.4492 | -0.0073 | -0.0003 | +0.000020 | -0.00011 |
| tdsc_abus_2023 | validation_101 | 29.6648 | 29.6727 | +0.0078 | -0.0004 | +0.000056 | -0.00039 |
| tdsc_abus_2023 | validation_102 | 29.2955 | 29.3052 | +0.0096 | +0.0010 | -0.000029 | +0.00016 |
| tdsc_abus_2023 | validation_103 | 28.0089 | 28.0083 | -0.0006 | -0.0000 | +0.000000 | +0.00002 |
| tdsc_abus_2023 | validation_104 | 27.5995 | 27.5913 | -0.0082 | -0.0010 | +0.000025 | -0.00016 |

Dataset-level PSNR gaps:

| dataset | pairs | mean gap | median gap | min gap | max gap | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic_liver | 6 | -0.0211 | -0.0009 | -0.0795 | +0.0137 | Strong retention; all well inside the -0.3 dB threshold. |
| tus_rec_2024_qc | 5 | +0.2719 | +0.2554 | +0.1549 | +0.4005 | Strong positive retention; canonical is consistently better. |
| tdsc_abus_2023 | 5 | +0.0003 | -0.0006 | -0.0082 | +0.0096 | Near-perfect tie; tracking-free canonical poses do not degrade quality. |
| all | 16 | +0.0771 | n/a | -0.0795 | +0.4005 | All 16 paired sweeps pass the strong-retention criterion. |

Stage L interpretation:

- The pose-free canonical prior is validated as a pose-dependency reduction module on the three non-spine datasets.
- It should not be claimed as a universal accuracy booster. The evidence is stronger: under regular/near-regular sweep protocols, per-frame 6-DoF pose can be replaced by a canonical sweep prior with negligible or positive PSNR gap.
- Synthetic liver and TDSC mainly show retention; TUS-QC shows consistent positive gains, likely because the canonical prior removes small tracker/sweep irregularities in already-regular QC sweeps.

## Stage M: Spacing Calibration Tolerance

Script:

```text
scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_m_spacing_tolerance_30k.sh
```

Output:

```text
outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_m_spacing_tolerance_30k_20260618
```

Representative sweeps:

| dataset | sweeps |
| --- | --- |
| synthetic_liver | `l1`, `r2` |
| tus_rec_2024_qc | `010_LH_Per_L_DtP_qc_0005_0370`, `011_LH_Per_S_PtD_qc_0008_0447` |
| tdsc_abus_2023 | `validation_100`, `validation_101` |

Variants:

```text
canonical scale = 0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50
```

File-pose references are reused from Stage L when possible.

Primary output:

- spacing-error curve per dataset;
- acceptable spacing-error range under the gap criteria.

## Stage N: Real Non-Uniform Subset Stress

This stage is planned after Stage L/M results. It uses real pose centers only to
select/evaluate subsets, not to provide canonical poses to the renderer.

Subset families:

- `uniform_arc`: nearest real frames to a uniform arc-length grid;
- `mild_jitter`: small random deviations from uniform arc positions;
- `medium_jitter`: larger deviations;
- `burst_gap`: locally dense frames plus one or two large gaps;
- `severe_gap`: failure-boundary subset.

Diagnostics to record:

```text
step_cv
max_step / median_step
p90_step / p10_step
rotation_per_step
non_elevational_motion_ratio
```

Rows:

```text
file_pose
canonical calibrated by mean/median selected spacing
```

Interpretation target:

Estimate an empirical regularity boundary, for example:

```text
step_cv <= 0.25: usually usable
0.25 < step_cv <= 0.45: dataset-dependent
step_cv > 0.50: unreliable, requires QC rejection or a stronger pose model
```

## Stage M Results: Spacing Calibration Tolerance

Date: 2026-06-18

Output:

```text
outputs/preproduction/ideas/ours/easy_pose/ours_pose_free_stage_m_spacing_tolerance_30k_20260618
```

Protocol:

- renderer/reconstruction: local-global best renderer, 1M init, 30k iterations, standard density control;
- pose-free prior: canonical one-sweep pose sequence with scalar spacing multiplier;
- comparison reference: Stage L file-pose result for the same sweep;
- strong retention: PSNR gap >= -0.3 dB;
- acceptable retention: PSNR gap >= -0.6 dB.

Best-PSNR gap against file-pose reference:

| dataset | sweep | 0.60 | 0.75 | 0.90 | 1.00 | 1.10 | 1.25 | 1.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver | l1 | +0.240 | +0.174 | +0.076 | +0.004 | -0.062 | -0.189 | -0.408 |
| synthetic_liver | r2 | +0.254 | +0.164 | +0.068 | -0.002 | -0.071 | -0.183 | -0.403 |
| tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | -0.065 | +0.062 | +0.116 | +0.163 | +0.176 | +0.171 | +0.103 |
| tus_rec_2024_qc | 011_LH_Per_S_PtD_qc_0008_0447 | +0.160 | +0.287 | +0.369 | +0.390 | +0.401 | +0.381 | +0.322 |
| tdsc_abus_2023 | validation_100 | -0.895 | -0.397 | -0.099 | -0.007 | +0.063 | +0.128 | +0.089 |
| tdsc_abus_2023 | validation_101 | -0.672 | -0.302 | -0.105 | -0.005 | +0.049 | +0.041 | -0.095 |

Pass-band summary:

| dataset | sweep | strong retention | acceptable retention |
| --- | --- | --- | --- |
| synthetic_liver | l1 | 0.60-1.25 | 0.60-1.50 |
| synthetic_liver | r2 | 0.60-1.25 | 0.60-1.50 |
| tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | 0.60-1.50 | 0.60-1.50 |
| tus_rec_2024_qc | 011_LH_Per_S_PtD_qc_0008_0447 | 0.60-1.50 | 0.60-1.50 |
| tdsc_abus_2023 | validation_100 | 0.90-1.50 | 0.75-1.50 |
| tdsc_abus_2023 | validation_101 | 0.90-1.50 | 0.75-1.50 |

Interpretation:

- The pose-free canonical sweep prior is not brittle to a small spacing calibration error.
- On all six representative sweeps, the calibrated setting `1.00` is within -0.01 dB to +0.39 dB of file-pose reconstruction.
- TUS-QC is highly tolerant: both representative sweeps strongly retain quality over the whole tested spacing range `0.60-1.50`.
- Synthetic liver tolerates underspecified spacing and moderate overspecified spacing; `1.50` remains acceptable but leaves the strong-retention band.
- TDSC is asymmetric: too-small spacing is harmful, especially `0.60`; however, `0.90-1.50` strongly retains quality on both representatives.
- A conservative paper claim can state that a simple calibrated spacing prior works reliably near the correct sweep spacing and remains robust over roughly `0.90-1.25` across all tested non-spine datasets, with wider tolerance on cleaner/regular sweeps.
