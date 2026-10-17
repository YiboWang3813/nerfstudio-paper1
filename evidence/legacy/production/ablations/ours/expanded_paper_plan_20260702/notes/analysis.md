# Expanded Ours Paper Ablation Results
Source summaries:
- Historical default/source rows: `outputs/production/ablations/ours/summary.tsv`
- Expanded supplemental rows: `outputs/production/ablations/ours/expanded_paper_plan_20260702/summary.tsv`
- Slurm job 66881 completed 128/128 expanded tasks with `failed=0`; checkpoint payload files retained under expanded root: 0.

## Generated Tables
- `tables/t01_three_module_joint.md`
- `tables/t02_renderer_internal.md`
- `tables/t03_dc_internal.md`
- `tables/t04_easy_pose_internal.md`
- `tables/t14_renderer_sensitivity_delta_psnr.md`
- `tables/t14_renderer_sensitivity_summary.md`
- `tables/t15_dc_sensitivity_delta_psnr.md`
- `tables/t15_dc_sensitivity_summary.md`

## Main Conclusions
- T01 joint module table is now complete. Best synthetic PSNR is row c (29.854 dB); best real PSNR is row f (33.004 dB). Real data strongly favors LG+EP, while DC remains neutral to very slightly negative in this joint table. The full row h is essentially tied with row f on real PSNR (-0.006 dB) and keeps the complete method configuration. Synthetic remains a tradeoff where local-only rows keep slightly higher PSNR but worse LPIPS.
- T02 renderer internal table supports retaining integral q=3 on real data: local-only to full q=3 improves TUS PSNR by +0.239 dB and LPIPS by -0.022; synthetic PSNR drops -0.108 dB but LPIPS improves.
- T03 DC internal table confirms responsibility DC is not a large PSNR booster. Its value is mechanism/neutrality: responsibility variants stay close to gradient DC, and attenuation protection is defensible.
- T04 Easy-Pose internal table supports the uniform calibrated canonical sweep: on TUS it improves over file pose by +0.247 dB PSNR while retaining similar LPIPS. Step-norm/axis-projection rows are intermediate diagnostics.
- T14 renderer sensitivity shows `ray_attenuation_quadrature_points` is the dominant renderer parameter by PSNR range; production q=3 is the best setting on both standard datasets. Strength, beta, min_transmission, and candidate_k_sigma have smaller ranges; their production defaults are stable on the real representative dataset.
- T15 DC sensitivity shows very small PSNR ranges overall; DC thresholds form a stable plateau rather than a brittle single point. `prune_opacity` mainly changes final Gaussian count with small metric changes.

## Next Reporting Use
- Use T01-T04 for paper tables.
- Use T11-A/T11-B visual figures after selecting representative frames from the completed runs.
- Use T12/T14/T15 for parameter sensitivity figures; T13 remains deferred.
