# UltraGray Accuracy/Speed Balance

Date: 2026-06-26

## Goal

Current production UltraGray uses a very small local transmission window:

```yaml
model.render.local_transmission_window_only: true
model.render.local_transmission_max_window_mm: 0.5
```

This keeps speed acceptable but severely hurts TUS-QC and TDSC accuracy. This preproduction sweep tests slightly larger local windows while keeping the rest of the GS-strict production UltraGray protocol unchanged.

## Scope

Output root:

```text
outputs/preproduction/ultragray_acc_speed_balance
```

Hydra alias:

```text
preproduction/baseline_reproduction/ultragray/acc_speed_balance
```

Slurm scripts:

```text
scripts/methods/baselines/ultragray/slurm/05_acc_speed_balance_task.sh
scripts/methods/baselines/ultragray/slurm/06_submit_acc_speed_balance.sh
```

Datasets and sweeps:

| Dataset | Sweep | Current UltraGray PSNR target context |
| --- | --- | --- |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | Current 0.5mm row is about 27.1 PSNR; useful target is near 29 PSNR. |
| `tdsc_abus_2023` | `validation_100` | Current 0.5mm row is about 20.2 PSNR; useful target is near 26 PSNR. |

Initial sweep:

Before submitting the matrix, rebuild the renderer once if `_C` is missing. Do not rebuild concurrently with other training jobs.

| Variant | Override |
| --- | --- |
| `lw1mm` | `model.render.local_transmission_max_window_mm=1` |
| `lw2mm` | `model.render.local_transmission_max_window_mm=2` |

All variants keep:

- 1M initial Gaussians
- 30k iterations
- threshold-only UltraGray DC timing from the GS-strict production alias
- final 30k checkpoint selection
- metrics: PSNR, L1, SSIM, LPIPS, DISTS, GMSD
- file poses and full-resolution p100k protocol fields

## Results

Summary TSV:

```text
outputs/preproduction/ultragray_acc_speed_balance/tables/ultragray_acc_speed_balance.tsv
```

All four jobs completed successfully on `gpu3`:

| Job | Dataset | Sweep | Window | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final GS |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66018 | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | 0.75 mm | 30.2186 | 0.02112 | 0.6870 | 0.2398 | 0.2052 | 0.0956 | 590,301 |
| 66010 | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | 1 mm | 31.6121 | 0.01713 | 0.7692 | 0.1895 | 0.1772 | 0.0778 | 689,680 |
| 66015 | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | 2 mm | 32.8285 | 0.01339 | 0.8531 | 0.1176 | 0.1394 | 0.0577 | 830,437 |
| 66019 | `tdsc_abus_2023` | `validation_100` | 0.75 mm | 25.8683 | 0.03694 | 0.6854 | 0.3222 | 0.2470 | 0.1182 | 376,023 |
| 66014 | `tdsc_abus_2023` | `validation_100` | 1 mm | 27.2981 | 0.02988 | 0.7537 | 0.2416 | 0.2148 | 0.0881 | 529,572 |
| 66016 | `tdsc_abus_2023` | `validation_100` | 2 mm | 28.5818 | 0.02430 | 0.7980 | 0.2043 | 0.1934 | 0.0595 | 733,396 |

## Decision

`2 mm` is the clear preferred balance point from this sweep. It improves both datasets over `0.75 mm` and `1 mm`, remains much faster than the old full scan-line setting in the training loop, and clears the requested quality floor by a wide margin:

- TUS-QC improves from the current small-window diagnostic `27.0853` PSNR to `32.8285` PSNR.
- TDSC improves from the current small-window diagnostic `20.1852` PSNR to `28.5818` PSNR.

Recommended production-pre candidate override:

```yaml
model.render.local_transmission_window_only: true
model.render.local_transmission_max_window_mm: 2.0
```

The run directories are under:

```text
outputs/preproduction/ultragray_acc_speed_balance/runs/
```
