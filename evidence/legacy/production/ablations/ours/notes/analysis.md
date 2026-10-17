# Ours GS-strict ablation rerun analysis

Source summary: `outputs/production/ablations/ours/summary.tsv`

Slurm job: `66353`

Rows completed: `116/116`

Checkpoint payload files retained: `0`

## Conclusions

- The full rerun completed without failed workers.
- Local-global rendering remains the strongest positive module: it improves mean PSNR on `3/4` core datasets under the factorial average.
- Responsibility DC is retained as an innovation because it is performance-neutral with weak positive evidence and changes allocation behavior; it is not claimed as a large final-quality booster under the fair loss.
- Easy-pose remains strongly retained: all positive-scope canonical/file PSNR gaps are at least `-0.1050 dB`, above the `-0.3 dB` strong-retention threshold.
- The negative-control spine rows still justify the easy-pose scope boundary.
- T07 spacing tolerance suggests rho/corridor can be dataset-tuned, but rho `1.00` remains acceptable and the gains are too small to explain the broader TDSC comparison gap alone.

## Artifact Map

- `tables/metrics_by_run.tsv`: complete 116-row per-run table.
- `tables/t02_renderer_dc_factorial.md`: factorial metrics.
- `tables/t03_renderer_component.md`: derived renderer effect.
- `tables/t04_responsibility_dc_component.md`: derived DC effect.
- `tables/t05_dc_subcomponents.md`: responsibility sub-component rows.
- `tables/t06_easy_pose_pairs.md`: positive easy-pose pairs.
- `tables/t07_spacing_tolerance.md`: spacing/rho sweep.
- `tables/t08_negative_control.md`: spine negative control.
- `tables/t09_seed_stability.md`: seed means.
- `tables/t09_seed_gain_vs_std.md`: seed-stability pass/fail.
- `tables/t10_runtime_gaussians.md`: runtime and Gaussian-count summary.
