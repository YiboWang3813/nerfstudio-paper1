# Batch1 postprocessing closeout (2026-09-09)

Quality aggregation now succeeds for all134 slots, including all78 main cases.
The historical GaussianPile/Breast/validation_104 missing time is not converted to
zero, does not exclude a quality case, and does not block FPS. The dataset raw-time
summary explicitly records29/30 available cases and does not silently publish that
partial mean as the paper aggregate. Baseline panel efficiency values are reused
verbatim from the old frozen table at its published precision.

Important timing caveat: the earlier P1 audit found that legacy GS panel Time used
pooled sweeps whereas implicit methods used equal dataset means. At the user's
September9 instruction to prioritize frozen baseline efficiency reuse, those panel
cells are now preserved, rather than silently changed. Corrected Ours uses equal
dataset means. Report this heterogeneity and hardware/concurrency differences;
do not claim a controlled training-speed advantage. Original baseline time/FPS
values are not recomputed or overwritten. Quality aggregation remains uniform.

## Verified FPS

Old blocked jobs79943/79945 were canceled before execution after Slurm rejected
updating the unsatisfiable dependency. Replacement80075 completed successfully
in5min26sec on a verified RTX4090. All78 cases measured, 5warmup, up to20eligible
heldout frames,3passes, pure forward, unchanged benchmark and production algorithm.

| Dataset | Cases for time/all-case FPS | Mean run seconds | All-case FPS mean | Paper matched2-case FPS |
|---|---:|---:|---:|---:|
| Liver | 6 | 545.5217044918487 | 243.88570146088205 | 240.16120006622583 |
| Spine | 9 | 578.8118999534183 | 224.66436593403347 | 229.26347675774932 |
| Arm | 33 | 487.0339401248052 | 275.77437746966575 | 293.1422933782769 |
| Breast | 30 | 410.97991751829784 | 327.6825685206878 | 316.24683267012847 |

Paper panel means (equal dataset weight): Synthetic/phantom562.1668022226336sec,
234.71233841198756FPS; Real449.0069288215515sec,304.6945630242027FPS.
All-case minimum56.896082461342445FPS (Arm010_RH_Par_S_DtP_qc_0005_0091), not hidden
by the representative panel mean. Relative to an explicitly stated30FPS reference,
all measured cases exceed it; pure-forward throughput is not end-to-end system latency.

## CGA accounting

Per-event data and population curves are in the corrected main `reports` directory.
Every variant satisfies initial1M + total_spawned - total_pruned = final_G exactly.
Twelve adaptation events per representative; pruned counts use the final union mask,
not the sum of overlapping weak and scale masks. Spawn counts use the actual
before/after population change plus union-pruned count (includes both children).

| Dataset | Variant | Spawned | Pruned | Final G | Path-protected candidate decisions |
|---|---|---:|---:|---:|---:|
| Liver | GD | 1628 | 462056 | 539572 | 0 |
| Liver | RGD | 73242 | 0 | 1073242 | 0 |
| Liver | RGD+OP | 64446 | 506668 | 557778 | 0 |
| Liver | RGD+LP | 64691 | 501990 | 562701 | 0 |
| Liver | Full | 65866 | 449552 | 616314 | 303097 |
| Arm | GD | 0 | 380562 | 619438 | 0 |
| Arm | RGD | 27958 | 0 | 1027958 | 0 |
| Arm | RGD+OP | 25234 | 394032 | 631202 | 0 |
| Arm | RGD+LP | 25278 | 392718 | 632560 | 0 |
| Arm | Full | 25497 | 364838 | 660659 | 163936 |

Protected decisions are summed event-level decisions; the same Gaussian can be
counted at multiple events. They are not unique primitives, nor equal to Full-minus-LP
population. Full-minus-LP final_G =53,613/28,099; dPSNR=-.0137623/+.0466336;
dLPIPS=-.001372787/-.000433476 for Liver/Arm. No significant-PSNR improvement claimed.

## Outputs and storage

Corrected output root:
`outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports`.

- `main_complete_dataset_aggregates.tsv`: unrounded quality means and time coverage.
- `main_corrected_values_display_ranks.tsv`: all192 method/dataset/metric values,
  four-decimal displays, ranks and best/second-best labels.
- `corrected_main_fps.tsv`: all78 raw FPS measurements and timer protocol fields.
- `corrected_efficiency_datasets.tsv`, `main_corrected_efficiency_panels.tsv`.
- `ppp_internal_candidate.tsv`, `ppp_multisweep_paired.tsv`, `ppp_multisweep_mean_std.tsv`.
- `cga_structural_candidate.tsv`, `cga_event_accounting.tsv`, `cga_population_accounting.tsv`.
- `population.tsv`, `population_dynamics.pdf/png`, `rho_sensitivity.pdf/png`.
- `contact_sheets/`: main replacement figure and12 contact sheets; frozen GT/crop hashes.

`main_corrected_claim_audit.md` and `qualitative_audit_20260909.md` are beside this file.
No canonical/exchange Results content was changed.

After FPS/report/image checks,76 newly trained main model files were permanently
deleted (2,438,731,076bytes). Numerical results, exports, manifests and source hashes
remain. Both reused anchors remain untouched. Ledger records every deleted file's
hash/size; recovering deleted weights requires retraining. No LGR active model touched.

Fourteen reporting/cleanup/paired-statistic tests pass.524 production/build hashes
match the corrected freeze; no PPP, LGR, CGA, training-loop or metric source changed.
LGR targeted confirmation is a separate authorized follow-up; Batch2 remains blocked
pending its scientific checkpoint and user approval.
