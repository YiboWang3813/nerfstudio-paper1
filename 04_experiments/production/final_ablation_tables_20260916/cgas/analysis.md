# CGAS internal audit

All paths below are relative to `docs/production/final_paper_evidence_20260910/`. New raw files carry exact absolute source/config/output paths, row IDs and execution/equivalence hashes. Values were recomputed from frozen per-run final metrics, not copied from old manuscript values. This is arithmetic reaggregation, not a new evaluation. No per-frame rerendering or metric recomputation was performed.

Sources: `cga/structural_sources.tsv`, `cga/event_totals.tsv`; cross-check: `cga/final_cga_table.tsv`. Legacy directory names only.

Representative protocol: Liver `l1`; Arm `010_LH_Per_L_DtP_qc_0005_0370`; 1M initial Gaussians, 30k iterations, seed 42, unchanged held-out split. Single-seed descriptive comparisons, not statistical significance. Arm sweeps/segments must not be treated as independent subjects.

All rows use Fixed+Full PLGR and MP. GD = gradient densification plus conventional opacity pruning; RGD = residual-guided densification with NO pruning; RGD + OP adds opacity pruning; RGD + LP uses local-aware pruning; Full CGAS uses accumulated-path-contribution-aware pruning. These are strategy comparisons, not five isolated one-switch causal effects.

## Recomputed transitions

| dataset | transition | delta_G | population_change_percent | delta_PSNR | delta_LPIPS |
| --- | --- | --- | --- | --- | --- |
| Liver | GD -> RGD | 422563 | 67.275425125 | -0.0143943 | -0.001611883 |
| Liver | RGD -> RGD + OP | -407378 | -38.773089984 | 0.0087957 | 0.001565403 |
| Liver | RGD -> RGD + LP | -405733 | -38.61652352 | 0.0196663 | 0.000731995 |
| Liver | RGD + OP -> RGD + LP | 1645 | 0.255715116 | 0.0108706 | -0.000833408 |
| Liver | RGD + LP -> Full CGAS | 43672 | 6.771493118 | -0.0018494 | -0.000497626 |
| Arm | GD -> RGD | 343044 | 50.481647933 | -0.0765283 | 0.000502935 |
| Arm | RGD -> RGD + OP | -332720 | -32.537116683 | 0.0645369 | -0.000226849 |
| Arm | RGD -> RGD + LP | -331741 | -32.441379014 | 0.0695633 | -0.000923958 |
| Arm | RGD + OP -> RGD + LP | 979 | 0.141911618 | 0.0050264 | -0.000697109 |
| Arm | RGD + LP -> Full CGAS | 22418 | 3.245011544 | 0.0386384 | 0.000701491 |

## Claim audit

- SUPPORTED descriptively: RGD retains substantially more capacity than GD. However GD -> RGD changes both densification and pruning, so do not attribute the entire final-population difference to residual guidance alone. Event totals separate spawned and pruned counts; RGD's final growth over the 1M initialization is much smaller than its relative increase over pruned GD.
- SUPPORTED: OP/LP sharply reduce the RGD population with narrow quality variation. Final populations and quality ranges are saved unrounded.
- NOT SUPPORTED as stated: “local-aware pruning is not consistently better than opacity-only in PSNR.” In these final two cases LP has HIGHER PSNR on both: Liver +0.0108706 dB; Arm +0.0050264 dB. These tiny single-seed differences do not establish a general accuracy benefit. LP also has slightly lower LPIPS on both.
- SUPPORTED: Full retains 43,672 additional primitives on Liver (+6.771493% relative to LP) and 22,418 on Arm (+3.245012%; 3.25% at two decimal places). This is a final population difference, not the number of distinct path-protected primitives.
- SUPPORTED: Full does not improve every quality metric. Relative to LP, Liver PSNR changes -0.0018494 dB while LPIPS improves -0.000497626; Arm PSNR improves +0.0386384 dB while LPIPS worsens +0.000701491.
- Safe interpretation: path-aware pruning selectively retains additional primitives while keeping reconstruction quality within a narrow range. Repeated path-protected candidate decisions in event_totals.tsv are not unique Gaussian counts.

#G is descriptive, with no bold/underline or better-direction. Raw integer counts and exact #G(k) are retained; manuscript display rounds #G(k) to integers, following the existing table.

Across all five variants, PSNR ranges are 0.0196663 dB (Liver) and 0.1082017 dB (Arm); LPIPS ranges are 0.001611883 and 0.000923958. GD -> RGD final population increases are 67.2754% / 50.4816%, whereas RGD growth over the common 1M initialization is only 5.0672% / 2.2586%. OP reduces RGD population by 38.7731% / 32.5371%; LP reduces it by 38.6165% / 32.4414%. This distinction prevents conflating suppressed pruning with newly spawned capacity.
