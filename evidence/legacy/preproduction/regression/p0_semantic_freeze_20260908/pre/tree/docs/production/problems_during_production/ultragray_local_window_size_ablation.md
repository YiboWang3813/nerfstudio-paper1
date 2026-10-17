# UltraGray Local Window Size Ablation

Date: 2026-06-26

## Problem

The current UltraGray production diagnostic used a very small local transmission window:

```yaml
model.render.local_transmission_window_only: true
model.render.local_transmission_max_window_mm: 0.5
```

This made the renderer fast enough, but the reconstruction quality collapsed on the two representative production-problem sweeps:

| Dataset | Sweep | Window | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | 0.5 mm | 27.0853 | 0.03137 | 0.5610 | 0.3201 | 0.2592 | 0.1230 | 399,431 |
| `tdsc_abus_2023` | `validation_100` | 0.5 mm | 20.1852 | 0.07981 | 0.3833 | 0.5316 | 0.3272 | 0.2258 | 156,107 |

The goal of this ablation was to find a slightly larger local window that restores usable quality without returning to the old full scan-line transmission behavior.

## Experiment Setup

Hydra alias:

```text
preproduction/baseline_reproduction/ultragray/acc_speed_balance
```

Slurm scripts:

```text
scripts/methods/baselines/ultragray/slurm/05_acc_speed_balance_task.sh
scripts/methods/baselines/ultragray/slurm/06_submit_acc_speed_balance.sh
```

Output root:

```text
outputs/preproduction/ultragray_acc_speed_balance
```

Summary table:

```text
outputs/preproduction/ultragray_acc_speed_balance/tables/ultragray_acc_speed_balance.tsv
```

All ablation rows keep the GS-strict UltraGray protocol unchanged except for `model.render.local_transmission_max_window_mm`:

- 1M initial Gaussians
- 30k iterations
- final 30k checkpoint selection
- file poses
- full-resolution p100k loading fields
- UltraGray threshold-only DC timing from the GS-strict production alias
- metrics: PSNR, L1, SSIM, LPIPS, DISTS, GMSD

Jobs were run on `gpu3` / RTX 4090.

## Results

### TUS-QC

Sweep: `010_LH_Per_L_DtP_qc_0005_0370`

| Job | Window | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS | Run dir |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 65886 | 0.5 mm | 27.0853 | 0.03137 | 0.5610 | 0.3201 | 0.2592 | 0.1230 | 399,431 | `outputs/production/comparisons/us_recon_four_dataset_p100k_v1/runs/stage1_diagnostic/nvs/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/ultragray` |
| 66018 | 0.75 mm | 30.2186 | 0.02112 | 0.6870 | 0.2398 | 0.2052 | 0.0956 | 590,301 | `outputs/preproduction/ultragray_acc_speed_balance/runs/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/lw0.75mm` |
| 66010 | 1.0 mm | 31.6121 | 0.01713 | 0.7692 | 0.1895 | 0.1772 | 0.0778 | 689,680 | `outputs/preproduction/ultragray_acc_speed_balance/runs/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/lw1mm` |
| 66015 | 2.0 mm | 32.8285 | 0.01339 | 0.8531 | 0.1176 | 0.1394 | 0.0577 | 830,437 | `outputs/preproduction/ultragray_acc_speed_balance/runs/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/lw2mm` |

### TDSC-ABUS-2023

Sweep: `validation_100`

| Job | Window | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS | Run dir |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 65891 | 0.5 mm | 20.1852 | 0.07981 | 0.3833 | 0.5316 | 0.3272 | 0.2258 | 156,107 | `outputs/production/comparisons/us_recon_four_dataset_p100k_v1/runs/stage1_diagnostic/nvs/tdsc_abus_2023/validation_100/ultragray` |
| 66019 | 0.75 mm | 25.8683 | 0.03694 | 0.6854 | 0.3222 | 0.2470 | 0.1182 | 376,023 | `outputs/preproduction/ultragray_acc_speed_balance/runs/tdsc_abus_2023/validation_100/lw0.75mm` |
| 66014 | 1.0 mm | 27.2981 | 0.02988 | 0.7537 | 0.2416 | 0.2148 | 0.0881 | 529,572 | `outputs/preproduction/ultragray_acc_speed_balance/runs/tdsc_abus_2023/validation_100/lw1mm` |
| 66016 | 2.0 mm | 28.5818 | 0.02430 | 0.7980 | 0.2043 | 0.1934 | 0.0595 | 733,396 | `outputs/preproduction/ultragray_acc_speed_balance/runs/tdsc_abus_2023/validation_100/lw2mm` |

## Findings

1. `0.5 mm` is too small. It prunes/collapses the effective field badly and leaves UltraGray as an obvious failure row on both datasets.
2. `0.75 mm` is a meaningful recovery over `0.5 mm`, but it is still not strong enough. TDSC reaches only `25.8683` PSNR, below the intended usable range.
3. `1.0 mm` crosses the quality floor on both datasets: TUS-QC reaches `31.6121` PSNR and TDSC reaches `27.2981` PSNR.
4. `2.0 mm` is the best tested setting on both datasets: TUS-QC reaches `32.8285` PSNR and TDSC reaches `28.5818` PSNR, with the best L1/SSIM/LPIPS/DISTS/GMSD among the tested windows.
5. Final Gaussian count increases with window size, which is expected: a larger local transmission neighborhood gives more useful gradient support and avoids the severe topology shrinkage seen at `0.5 mm`.

## Production Baseline Freeze

The quality-best tested UltraGray local window in this ablation is `2.0 mm`,
but the production baseline table frozen on 2026-06-26 intentionally uses the
faster balanced local-window row:

```yaml
model.render.local_transmission_window_only: true
model.render.local_transmission_max_window_mm: 0.75
```

This is the UltraGray row used in the current six-GS objective comparison table
for `tus_rec_2024_qc` and `tdsc_abus_2023`.

## Ablation Quality-Best Candidate

Use `2.0 mm` only as the quality-best UltraGray local-window candidate:

```yaml
model.render.local_transmission_window_only: true
model.render.local_transmission_max_window_mm: 2.0
```

If speed becomes the limiting concern in a broader full-sweep relaunch, `1.0 mm`
is the fallback candidate. Do not confuse this quality-best recommendation with
the 2026-06-26 production baseline freeze, which uses `0.75 mm` by decision.

## Follow-Up

Before changing the production alias globally, rerun the selected candidate on the next staged production set and compare wall time / GPU memory from `logs/efficiency_summary.yaml` against the current `0.5 mm` row.
