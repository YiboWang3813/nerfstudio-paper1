# Joint component audit

All paths below are relative to `docs/production/final_paper_evidence_20260910/`. New raw files carry exact absolute source/config/output paths, row IDs and execution/equivalence hashes. Values were recomputed from frozen per-run final metrics, not copied from old manuscript values. This is arithmetic reaggregation, not a new evaluation. No per-frame rerendering or metric recomputation was performed.

Authoritative raw: `joint/final_joint_sources.tsv`; frozen cross-check: `joint/final_joint_table.tsv`.

Representative protocol: Liver `l1`; Arm `010_LH_Per_L_DtP_qc_0005_0370`; 1M initial Gaussians, 30k iterations, seed 42, unchanged held-out split. Single-seed descriptive comparisons, not statistical significance. Arm sweeps/segments must not be treated as independent subjects.

## Design meaning

- AFGM off = MP, full measured 6-DoF geometry; on = frame-order/nominal-spacing construction.
- PLGR off = Plane local/no global, NOT Fixed local-only. On = Fixed 1-mm local + Full causal integral global.
- CGAS off = gradient densification + opacity pruning. On = residual-guided/path-aware adaptation; without global statistics, pruning is local-aware. Some reused off rows explicitly encode local-aware; others request path-aware and use the implemented fallback. Their effective semantics agree.
- The three switches form the full eight backgrounds. Table columns are AFGM/PLGR/CGAS; old L/C/P codes were translated, not blindly reordered.

## Mean on-minus-off effects across four paired backgrounds

| dataset | module | mean_delta_PSNR | mean_delta_SSIM | mean_delta_LPIPS |
| --- | --- | --- | --- | --- |
| Liver | AFGM | 0.014290025 | 0.003970332 | 0.100453703 |
| Liver | PLGR | 3.863299175 | 0.047266704 | -0.311062344 |
| Liver | CGAS | 0.009888925 | -0.000292946 | -0.001455174 |
| Arm | AFGM | 1.682869925 | 0.059791668 | -0.013412255 |
| Arm | PLGR | 10.339061125 | 0.165249544 | -0.223341182 |
| Arm | CGAS | 0.090647425 | 0.001622425 | -0.002457159 |

PLGR is the dominant PSNR contributor. These are four factorial backgrounds, not four independent replicates. CGAS chiefly concerns capacity allocation; the quality table alone does not prove spatial allocation, so consult its internal population/event evidence. AFGM removes measured geometry from trajectory construction, rather than acting as a universal accuracy booster: Liver LPIPS becomes worse when enabled, including the complete method. The full framework supports complementary roles, not a claim of positive synergy for every metric.

No joint qualitative figure is included or regenerated. Existing historical figures were not deleted or modified.
