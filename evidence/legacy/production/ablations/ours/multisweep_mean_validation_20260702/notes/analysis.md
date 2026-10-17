# Multi-Sweep Mean Validation Results
Source summaries:
- Historical source rows: `outputs/production/ablations/ours/summary.tsv`
- Expanded representative rows: `outputs/production/ablations/ours/expanded_paper_plan_20260702/summary.tsv`
- Multi-sweep newly run rows: `outputs/production/ablations/ours/multisweep_mean_validation_20260702/summary.tsv`
- Slurm job 66969 completed 60/60 new tasks with `failed=0`; checkpoint payload files retained under this output root: 0.

## Generated Files
- `tables/multisweep_mean_std.tsv`
- `tables/multisweep_per_sweep_metrics.tsv`
- `tables/t01_key_path_synthetic_mean_std.md`
- `tables/t01_key_path_tus_qc_mean_std.md`
- `tables/t02_renderer_internal_synthetic_mean_std.md`
- `tables/t02_renderer_internal_tus_qc_mean_std.md`
- `tables/t04_easy_pose_internal_synthetic_mean_std.md`
- `tables/t04_easy_pose_internal_tus_qc_mean_std.md`
- `tables/tdsc_key_path_mean_std.md`

## Core Mean Trends
### Synthetic liver, 5 sweeps
- T01 key path: +LG changes PSNR by -0.094 dB over Base; +LG+DC changes PSNR by -0.002 dB over +LG; +LG+DC+EP changes PSNR by -0.006 dB over +LG+DC.
- T02 renderer: integral q=3 changes PSNR by -0.098 dB over local-only and LPIPS by -0.037.
- T04 Easy-Pose: uniform canonical changes PSNR by -0.006 dB over file pose and LPIPS by +0.010.
### TUS-QC, 5 sweeps
- T01 key path: +LG changes PSNR by +0.439 dB over Base; +LG+DC changes PSNR by -0.003 dB over +LG; +LG+DC+EP changes PSNR by +0.348 dB over +LG+DC.
- T02 renderer: integral q=3 changes PSNR by +0.434 dB over local-only and LPIPS by -0.022.
- T04 Easy-Pose: uniform canonical changes PSNR by +0.348 dB over file pose and LPIPS by -0.000.
### TDSC Stress Check, 3 Sweeps
- T01 key path on TDSC: +LG changes PSNR by +0.347 dB over Base; +LG+DC changes PSNR by -0.028 dB over +LG; +LG+DC+EP changes PSNR by -0.004 dB over +LG+DC.

## Interpretation
- The averaged TUS-QC trend supports local-global rendering as the dominant real-data quality contributor.
- Synthetic liver remains a tradeoff: local/global variants can improve perceptual quality while not always improving PSNR. This matches the representative-sweep tables.
- Easy-Pose uniform canonical sweep is stable on average for both synthetic and TUS-QC; it does not collapse under multi-sweep validation.
- TDSC stress check confirms +LG helps over Base, but the full +DC/+EP stack is less consistently beneficial there. This should be discussed as dataset-dependent behavior rather than hidden in the main claim.
