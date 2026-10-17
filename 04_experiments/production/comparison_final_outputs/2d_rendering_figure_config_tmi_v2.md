# TMI v2 Four-Dataset 2D Rendering Figure

Status: generated and frozen-sample.

This version preserves the original frozen figures and uses the four fixed samples specified in `docs/production/new_comparison_ablation_figures/readme.md`.

## Regeneration

```bash
MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/comparisons/us_recon_four_dataset_p100k_v1/make_tmi_all_datasets_2d_rendering_v2.py
```

## Outputs

- `figures/paper_all_datasets_2d_rendering.pdf`
- `figures/paper_all_datasets_2d_rendering.png`
- `manifests/paper_all_datasets_2d_rendering_manifest.json`

Layout: 4 datasets by 8 methods plus GT. Images use a shared `[0, 1]` grayscale display range and one shared crop/inset placement within each row. No metrics or sweep identifiers are overlaid. Compact row labels are `Liver`, `Spine`, `Arm`, and `Breast`; full dataset ids remain recorded in the manifest.
