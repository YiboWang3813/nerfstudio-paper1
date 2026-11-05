# GS Thickness Ablation Stage 1

Date: 2026-06-26

## Purpose

Test whether a dataset-normalized slice-thickness ratio is a controllable
hyperparameter for GaussianPile and slice-adapted 3DGS on the Stage 1
representative TUS and TDSC sweeps.

The hypothesis is that too-thin settings under-cover the slice and too-thick
settings blur across neighboring slices, while ratios near the dataset
`delta_elevational_mm` can improve or stabilize reconstruction quality.

## Scope

Benchmark root:

```text
outputs/production/comparisons/us_recon_four_dataset_p100k_v1
```

Stage:

```text
thickness_ablation_stage1
```

Datasets and sweeps:

| Dataset | Sweep | delta_elevational_mm |
| --- | --- | ---: |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | `0.404000` |
| `tdsc_abus_2023` | `validation_100` | `0.475674` |

Methods:

```text
gaussianpile
slice_adapted_3dgs
```

Ratios:

```text
0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00
```

Effective parameterization:

```text
thickness_mm = ratio * dataset_meta.delta_elevational_mm
GaussianPile override: model.slice_render.sigma_z_mm=<thickness_mm>
slice_adapted_3dgs override: model.render.slice_thickness_mm=<thickness_mm>
```

All runs use the GS-strict production protocol inherited from the method aliases:
1M initial Gaussians, 30k iterations, final checkpoint evaluation, file poses,
trimmed-middle holdout, six production metrics, and shared DC timing.

## Launch

Runner:

```text
scripts/production/comparisons/us_recon_four_dataset_p100k_v1/run_thickness_ablation_task.sh
```

Submitter:

```text
scripts/production/comparisons/us_recon_four_dataset_p100k_v1/submit_thickness_ablation_stage1.sh
```

Default task count:

```text
2 datasets x 2 methods x 7 ratios = 28 runs
```

Default Slurm target:

```text
RTX 4090 / gpu3, array 0-27%2
```

Summary table:

```text
outputs/production/comparisons/us_recon_four_dataset_p100k_v1/tables/thickness_ablation_stage1_per_run.tsv
```

## Decision Rule

Use the completed table to check whether quality follows a peaked curve over
the ratio. A useful controllable thickness hyperparameter should show:

1. degraded metrics at one or both extremes;
2. stable or improved metrics near a middle ratio;
3. interpretable changes in Gaussian count, coverage, or black-frame health.

If the best ratio differs systematically from `1.0`, promote a dataset-normalized
ratio override for the affected method only after checking that the change does
not violate the frozen GS-strict protocol.

## Results

Completed: 2026-06-26.

All 28 runs reached `final_step=30000` and wrote summary rows to:

```text
outputs/production/comparisons/us_recon_four_dataset_p100k_v1/tables/thickness_ablation_stage1_per_run.tsv
```

The last four TDSC slice-adapted runs were re-run with `LIGHT_OUTPUTS=1`
after the first attempt hit user disk quota while saving checkpoint/visual
artifacts. This disabled checkpoint and visualization writes for the re-run
only; scalar metrics and per-frame metrics were still produced.

### Final PSNR / SSIM

| Dataset | Method | r050 | r075 | r100 | r125 | r150 | r200 | r300 | Best PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tus_rec_2024_qc` | `gaussianpile` | 27.713 / 0.782 | 29.446 / 0.808 | 30.930 / 0.831 | 31.871 / 0.846 | 32.433 / 0.855 | 32.982 / 0.864 | 33.154 / 0.869 | r300 |
| `tus_rec_2024_qc` | `slice_adapted_3dgs` | 32.047 / 0.864 | 32.510 / 0.867 | 32.885 / 0.870 | 33.066 / 0.871 | 33.148 / 0.871 | 33.141 / 0.870 | 32.899 / 0.867 | r150 |
| `tdsc_abus_2023` | `gaussianpile` | 23.048 / 0.647 | 26.913 / 0.751 | 28.350 / 0.788 | 28.782 / 0.800 | 28.972 / 0.807 | 29.090 / 0.812 | 28.337 / 0.799 | r200 |
| `tdsc_abus_2023` | `slice_adapted_3dgs` | 27.823 / 0.794 | 28.330 / 0.799 | 28.467 / 0.801 | 28.487 / 0.801 | 28.445 / 0.801 | 28.252 / 0.796 | 27.527 / 0.781 | r125 |

### Interpretation

The normalized thickness ratio is a controllable hyperparameter for both
methods on these sweeps.

- GaussianPile strongly benefits from thicker-than-delta slices. On TUS-QC it
  keeps improving through r300; on TDSC it peaks at r200 and drops at r300.
- slice-adapted 3DGS has a clear peaked curve. TUS-QC peaks around r150/r200,
  while TDSC peaks around r100/r125 and degrades when pushed to r200/r300.
- The useful ratio is related to `delta_elevational_mm` because the tested
  value is `ratio * delta_elevational_mm`, but the best ratio is not universally
  `1.0`; it is method- and dataset-dependent.
- For a controllable knob: increasing ratio generally raises scores from
  under-thick settings until a method-specific optimum, then can suppress scores
  once cross-slice blurring dominates. GaussianPile is less suppressed by large
  ratios on TUS-QC than slice-adapted 3DGS.

## r025 Follow-Up

After comparing the Stage 1 diagnostic table, a follow-up run tested
`slice_adapted_3dgs` with a deliberately thinner r025 setting on the same two
representative sweeps.

Runner:

```text
scripts/production/comparisons/us_recon_four_dataset_p100k_v1/run_slice3dgs_r025_task.sh
```

Submitter:

```text
scripts/production/comparisons/us_recon_four_dataset_p100k_v1/submit_slice3dgs_r025_stage1.sh
```

Canonical stage:

```text
slice3dgs_r025_stage1_sharedckpt
```

Canonical summary:

```text
outputs/production/comparisons/us_recon_four_dataset_p100k_v1/tables/slice3dgs_r025_stage1_sharedckpt_per_run.tsv
```

Canonical checkpoint root:

```text
outputs/production/comparisons/us_recon_four_dataset_p100k_v1/external_checkpoints/runs/slice3dgs_r025_stage1_sharedckpt
```

The first r025 submission wrote checkpoint symlinks to
`/tmp/boywang25/omniultra_checkpoints`. The scalar metrics completed, but the
login node could not resolve those node-local `/tmp` targets after the Slurm
tasks exited. That stage should not be promoted as canonical. The production
runner now defaults `CHECKPOINT_ROOT` to `${OUTPUT_ROOT}/external_checkpoints`,
which is a shared project path under the benchmark output tree. Each run writes
a `checkpoint_manifest.yaml` beside the retained final container and solver
weights.

### r025 Metrics

| Dataset | Sweep | Thickness mm | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | 0.101000 | 31.6762 | 0.01506 | 0.8618 | 0.1320 | 0.1614 | 0.0584 | 814507 |
| `tdsc_abus_2023` | `validation_100` | 0.1189185 | 26.9210 | 0.02803 | 0.7848 | 0.2508 | 0.2098 | 0.0707 | 813243 |

Compared with the earlier r050 rows, r025 is worse on both sweeps across all
six production metrics:

| Dataset | Metric | r025 | r050 | Direction |
| --- | --- | ---: | ---: | --- |
| `tus_rec_2024_qc` | PSNR | 31.6762 | 32.0468 | worse |
| `tus_rec_2024_qc` | L1 | 0.01506 | 0.01450 | worse |
| `tus_rec_2024_qc` | SSIM | 0.8618 | 0.8643 | worse |
| `tus_rec_2024_qc` | LPIPS | 0.1320 | 0.1276 | worse |
| `tus_rec_2024_qc` | DISTS | 0.1614 | 0.1584 | worse |
| `tus_rec_2024_qc` | GMSD | 0.0584 | 0.0564 | worse |
| `tdsc_abus_2023` | PSNR | 26.9210 | 27.8227 | worse |
| `tdsc_abus_2023` | L1 | 0.02803 | 0.02562 | worse |
| `tdsc_abus_2023` | SSIM | 0.7848 | 0.7940 | worse |
| `tdsc_abus_2023` | LPIPS | 0.2508 | 0.2343 | worse |
| `tdsc_abus_2023` | DISTS | 0.2098 | 0.2020 | worse |
| `tdsc_abus_2023` | GMSD | 0.0707 | 0.0628 | worse |

This confirms the lower side of the control curve: for `slice_adapted_3dgs`,
making the effective slice thickness too thin can suppress quality rather than
rescue a weak run. The useful search range for this method on these two sweeps
starts around r050 and peaks near r100-r150 depending on dataset.

### Table Impact If Using r025 Slice and r050 GaussianPile

If the Stage 1 diagnostic table is recomputed with `slice_adapted_3dgs=r025`
and `gaussianpile=r050`, the PSNR-sorted rows become:

TUS-QC / `010_LH_Per_L_DtP_qc_0005_0370`

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 32.7664 | 0.01283 | 0.8661 | 0.1471 | 0.1626 | 0.0561 |
| `ultragauss` | 32.0335 | 0.01434 | 0.8330 | 0.1645 | 0.1749 | 0.0644 |
| `slice_adapted_3dgs_r025` | 31.6762 | 0.01506 | 0.8618 | 0.1320 | 0.1614 | 0.0584 |
| `gaussianpile_r050` | 27.7133 | 0.02574 | 0.7816 | 0.2605 | 0.1998 | 0.0998 |
| `ultragray` | 27.0853 | 0.03137 | 0.5610 | 0.3201 | 0.2592 | 0.1230 |

TDSC / `validation_100`

| Method | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours` | 27.7341 | 0.02601 | 0.7859 | 0.2558 | 0.1944 | 0.0690 |
| `ultragauss` | 27.2054 | 0.02631 | 0.7793 | 0.1925 | 0.1931 | 0.0743 |
| `slice_adapted_3dgs_r025` | 26.9210 | 0.02803 | 0.7848 | 0.2508 | 0.2098 | 0.0707 |
| `gaussianpile_r050` | 23.0483 | 0.04673 | 0.6471 | 0.4568 | 0.2879 | 0.1543 |
| `ultragray` | 20.1852 | 0.07981 | 0.3833 | 0.5316 | 0.3272 | 0.2258 |

The table would make `ours` PSNR-first on both selected Stage 1 sweeps, but it
would do so by choosing a suppressed `slice_adapted_3dgs` setting and an
under-thick `gaussianpile` setting. That is useful as a diagnostic stress test,
not as a fair replacement for each baseline's best or default thickness.

### Production Baseline Freeze

The production baseline table fixed on 2026-06-26 supersedes the earlier
`gaussianpile_r050` table-impact example:

```text
slice_adapted_3dgs = r025
gaussianpile = r075
```

`r075` is still below GaussianPile's quality-best larger-thickness rows on these
two diagnostic sweeps, but it is the selected production baseline row for the
current six-GS objective comparison.

## Operational Lessons

- Treat `/tmp` as node-local unless the cluster explicitly documents it as
  shared. It is acceptable for scratch files but not for production checkpoint
  references that must be inspected after a Slurm job exits.
- For production or pre-production benchmark jobs, checkpoint roots should live
  on a persistent shared path. In this benchmark, use
  `${OUTPUT_ROOT}/external_checkpoints` by default.
- Keep run output directories and weight storage decoupled with
  `system.link_checkpoint_dir=true`, but always write a manifest containing the
  resolved checkpoint path and the retained final weight filenames.
- When validating a new checkpoint policy, check both scalar summary success and
  the physical target of the `checkpoints` symlink. A successful TSV row is not
  enough to prove that final weights are retrievable.
- For thickness ablations, record the absolute millimeter value and the
  normalized ratio. The ratio is interpretable across datasets only because it
  is multiplied by each dataset's `delta_elevational_mm`.
