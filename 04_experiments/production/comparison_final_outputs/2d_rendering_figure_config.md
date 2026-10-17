# Frozen 2D Rendering Figure Production Config

Status: **completed and frozen**

Date frozen: 2026-06-29

This file records the final production configuration for the paper-facing 2D rendered-plane comparison figures. Do not change the figure script or regenerate these figures with different visual rules unless a new config version is created.

## Outputs

| Group | PDF | PNG preview | Manifest |
| --- | --- | --- | --- |
| Simulated | `figures/paper_simulated_2d_rendering.pdf` | `figures/paper_simulated_2d_rendering.png` | `manifests/paper_simulated_2d_rendering_manifest.json` |
| Real | `figures/paper_real_2d_rendering.pdf` | `figures/paper_real_2d_rendering.png` | `manifests/paper_real_2d_rendering_manifest.json` |

## Frozen Script

```text
scripts/production/comparisons/us_recon_four_dataset_p100k_v1/make_paper_2d_rendering_figures.py
```

Regeneration command:

```bash
MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/comparisons/us_recon_four_dataset_p100k_v1/make_paper_2d_rendering_figures.py
```

## Data Sources

Benchmark root:

```text
outputs/production/comparisons/us_recon_four_dataset_p100k_v1
```

Merged summary TSV priority, lowest to highest:

```text
gs_stage2_fixed_subset_per_run.tsv
gs_full_sweep_per_run.tsv
gs_ultragauss_tus_qc_chi2_5500_replacement.tsv
```

Per-panel images are loaded from each method run's:

```text
vis/item_<item>/pred.png
vis/item_<item>/gt.png
eval/per_frame_metrics_latest.tsv
```

## Fixed Columns

Use fixed paper-facing method columns:

```text
UltraGauss
UltraGray
Slice-3DGS
GaussianPile
Ours
GT
```

Corresponding method ids:

```text
ultragauss
ultragray
slice_adapted_3dgs
gaussianpile
ours
gt
```

The 2D figures do not use per-row rank-2/3/4/5 baseline columns.

## Row Selection

For each dataset, select the two sweeps where `ours` is rank-1 by final PSNR and has the highest final PSNR among Ours-winning sweeps.

Frozen selected rows:

| Figure | Dataset | Sweep | Display label | Item | Crop xywh |
| --- | --- | --- | --- | ---: | --- |
| Simulated | `synthetic_liver` | `l3` | `Liver_L1` | 27 | `[19, 0, 53, 107]` |
| Simulated | `synthetic_liver` | `l1` | `Liver_R1` | 31 | `[19, 0, 53, 107]` |
| Simulated | `spine_phantom` | `left3_2` | `Spine_L1` | 17 | `[59, 0, 36, 215]` |
| Simulated | `spine_phantom` | `left1_1` | `Spine_L11` | 18 | `[8, 378, 36, 215]` |
| Real | `tus_rec_2024_qc` | `010_RH_Per_S_PtD_qc_0165_0364_s02` | `TUS_S02` | 18 | `[129, 24, 92, 69]` |
| Real | `tus_rec_2024_qc` | `010_RH_Per_C_PtD_qc_0006_0427` | `TUS_0427` | 66 | `[129, 24, 92, 69]` |
| Real | `tdsc_abus_2023` | `validation_116` | `TDSC_116` | 39 | `[150, 23, 85, 67]` |
| Real | `tdsc_abus_2023` | `validation_101` | `TDSC_101` | 32 | `[90, 23, 85, 67]` |

## Frame And Crop Selection

Frame selection:

```text
score = (ours_psnr - max_baseline_psnr) + 5.0 * (ours_ssim - max_baseline_ssim)
```

Crop selection:

```text
mean_baseline_error = mean(abs(baseline_pred - gt))
ours_error = abs(ours_pred - gt)
advantage = mean_baseline_error - ours_error
texture = abs(gt - mean(gt))
score_map = advantage + 0.15 * texture
```

The crop search uses a deterministic 10 by 10 grid over valid crop origins.

Crop size:

```text
crop_w = max(36, int(image_width * 0.24))
crop_h = max(36, int(image_height * 0.24))
```

## Visual Style

Figure layout:

```text
rows = 4
columns = 6
figsize = (12.2, 7.2)
dpi = 300
wspace = 0.035
hspace = 0.08
```

Font:

```text
font.family = serif
font.serif = Times New Roman, Times, DejaVu Serif
pdf.fonttype = 42
ps.fonttype = 42
```

Text sizing:

```text
method header = 16 pt
sweep label = 16 pt
metric overlay = 8 pt
```

Panel visualization:

1. Show the full grayscale rendered image.
2. Draw a yellow source rectangle over the selected crop.
3. Place an overlapping magnified patch in the panel at the least-overlapping candidate corner.
4. Draw yellow connector lines from the source crop to the magnified patch.
5. Overlay `PSNR xx.xx` and `SSIM x.xxx` for non-GT method panels only.
6. Do not draw any in-image text in the GT panels.

## Completion

The 2D visualization deliverable is complete:

```text
docs/production/comparison_final_outputs/figures/paper_simulated_2d_rendering.pdf
docs/production/comparison_final_outputs/figures/paper_real_2d_rendering.pdf
```
