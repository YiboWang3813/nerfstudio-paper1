# TMI v2 Ablation Figure Configuration

Status: generated from frozen production manifests. No training results were changed.

## Regeneration

```bash
MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/ablations/ours/make_tmi_ablation_joint_visual_compact_v2.py
MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/ablations/ours/make_tmi_ablation_sensitivity_summary_v2.py
```

## Outputs

- `figures/paper_ablation_joint_visual_compact.pdf` and `.png`
- `manifests/paper_ablation_joint_visual_compact_manifest.json`
- `figures/paper_ablation_parameter_sensitivity.pdf` and `.png`
- `manifests/paper_ablation_parameter_sensitivity_manifest.json`
- `figures/paper_ablation_parameter_sensitivity_single_column.pdf` and `.png`
- `manifests/paper_ablation_parameter_sensitivity_single_column_manifest.json`

Terminology: LGR is the physics-aware local-global Gaussian renderer; CGA is contribution-aware Gaussian adaptation; PP is acquisition-guided pseudo-pose parameterization. The sensitivity figure uses differences relative to the corresponding production defaults.
