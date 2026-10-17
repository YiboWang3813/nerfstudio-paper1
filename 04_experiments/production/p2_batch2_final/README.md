# Final Hyperparameter Batch 2 — authorized numeric analysis

Structural design is closed. No algorithm/default edits, new structural variants,
PPP reruns, split-threshold runs, or automatic follow-on training are permitted.
P2 source SHA256:
`f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27`.

MP, Fixed+Full, residual-guided / accumulated-A_g path-aware CGA, 1M initial,
30k, seed42. Liver l1; Arm 010_LH_Per_L_DtP_qc_0005_0370. Same frozen split,
optimizer, loss and metric definitions. Every resolved config is compared with
the corresponding final Main MP anchor: only the studied key(s) may differ.

| Parameter | Values | Default | Reused slots | New slots |
|---|---|---|---:|---:|
| sigma_ele (mm) | .8,1,1.5,2 | 1 | 2 | 6 |
| beta | .4,.6,.8,1 | .8 | 2 | 6 |
| lambda_ray | .2,.35,.55,.75,1 | .55 | 2 | 8 |
| Nq | 1,2,3 | 3 | 2 | 4 |
| half-window (mm) | 1,2,3,4 | 2 | 2 | 6 |
| qd x B | [.96,.98,.99] x [.001,.005,.05] | .98,.05 | 2 | 16 |
| qp x qA | [.05,.10,.15] x [.70,.80,.90] | .10,.80 | 2 | 16 |
| tau_alpha | 5e-5,1e-4,5e-4 | 1e-4 | 2 | 4 |
| Total | 82 logical slots | 2 unique defaults | 16 | 66 |

68 unique configurations; 66 NEW + 2 FINAL-REUSE. No duplicates across workers.
Main/structural/PPP data are unchanged. The former52-run proposal was never
executed and provides no additional results for reuse.

## Range choices, not test-set tuning

- Existing renderer clips width to [.8,5]mm even in Fixed mode. A nominal .5
  would silently mean .8; choose .8 explicitly, preserving all other knobs.
  Do not relabel the effective width or lower the clamp for this single-axis study.
- Previous responsibility-selected fractions were .00174–.00340(Liver) and
  .00091–.00156(Arm). B=.02/.05/.08 is likely inactive throughout. The prior
  proposal's B=.001/.005/.05 tests a possibly binding budget and the unchanged
  loose default. qd follows the user's .96/.98/.99. A plateau when the budget
  is inactive is an implementation observation, not evidence of unrestricted
  sensitivity to an active cap. Preserve raw event thresholds/counts.
- Tau uses 5e-5/1e-4/5e-4. Read-only inspection of retained Fixed FPS reconstruction
  checkpoints, alpha=.2*sigmoid(raw), found fractions below these thresholds:
  Liver l1 fr_002: .08413/.10665/.17435;
  Arm 010_RH_Per_S_PtD_qc_0165_0364_s02 fr_007: .06248/.07718/.12450.
  The Arm diagnostic is a frozen FPS case, not the training representative.
  Final-state survivor distributions do not predict event-time pruning directly;
  they justify a nontrivial lower-tail range, not an expected quality outcome.
- No beta/lambda zero or changed global path semantics; no depth-adaptive law.

## Budget and stopping

Historical anchor estimates:8.50 worker-hours; max four-worker partition2.15h;
allow30% headroom (~2.8h), plus queue time. Actual GPU/parameter costs can differ.
No claim of controlled speedup from shared-worker wall time. Each worker stops
on a failed run; no silent retries, no extra searches. Postprocessing waits for
successful completion of both GPU jobs. No per-point formal FPS benchmark.

Existing output settings save final checkpoint only (not iteration histories).
Retain metrics, frame metrics, configs, images, logs, hashes and manifests.
Checkpoints may be cleaned only after output validation; protected FPS models
are outside this batch. No legacy or final-main output is overwritten.

## Outputs

`batch2_exact_matrix.json`, `panel_points.json`, `exact_configs/`, partition,
submission hashes and freeze reference are the executable protocol.
Reports go to
`outputs/production/ablations/ours/P2_fixed_full_final_20260909/batch2/reports`.
H1 reuses rho; H2 has five delta-PSNR panels plus an LPIPS companion;
H3 includes both grids, population heatmaps, tau and quality–population scatter.
These are draft layouts. Default points are marked; no default is retuned.
Final qualitative audit and scientific interpretation must be reviewed before
the completeness audit can mark every category COMPLETE.
