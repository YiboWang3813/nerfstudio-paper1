# AFGM internal audit

All paths below are relative to `docs/production/final_paper_evidence_20260910/`. New raw files carry exact absolute source/config/output paths, row IDs and execution/equivalence hashes. Values were recomputed from frozen per-run final metrics, not copied from old manuscript values. This is arithmetic reaggregation, not a new evaluation. No per-frame rerendering or metric recomputation was performed.

Sources: `ppp/final_ppp_sources.tsv`, `ppp/multisweep_sources.tsv`; cross-checks: `ppp/final_ppp_table.tsv`, `ppp/final_ppp_multisweep.tsv`. These are legacy path names only.

Representative protocol: Liver `l1`; Arm `010_LH_Per_L_DtP_qc_0005_0370`; 1M initial Gaussians, 30k iterations, seed 42, unchanged held-out split. Single-seed descriptive comparisons, not statistical significance. Arm sweeps/segments must not be treated as independent subjects.

## Scientifically distinct rows

- MP: measured translations and rotations, complete 6-DoF geometry.
- PP-S: partial-oracle diagnostic. Adjacent measured translation Euclidean magnitudes accumulated along the predefined signed scan axis; rotations not used.
- PP-A: partial-oracle diagnostic. Signed adjacent measured translation projection on the predefined axis; no automatic axis estimation; rotations not used.
- AFGM: retained frame index, dataset-specific nominal spacing times rho=1, predefined signed axis and reference frame; identity rotations. Measured probe geometry is not used to construct its trajectory. A loader may still read a pose file for frame bookkeeping.

PP-S and PP-A are diagnostics, not submodules of AFGM. All four rows use final Fixed+Full PLGR and Full CGAS. Three metrics are retained in the diagnostic panel to expose perceptual trade-offs.

## Multi-sweep result

Liver 6, Arm 5, Breast 5: paired identical case IDs, equal-sweep arithmetic mean; sample std (ddof=1) across sweeps. The std is not a seed confidence interval. PSNR/SSIM only in this panel; no winner highlighting because its question is accuracy preservation.

| dataset | transition | metric | n | mean_delta | std_delta | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm | MP -> AFGM | PSNR | 5 | 0.5131599 | 0.128026517 | 5 | 0 | 0 |
| Arm | MP -> AFGM | SSIM | 5 | 0.005176001 | 0.000886251 | 5 | 0 | 0 |
| Breast | MP -> AFGM | PSNR | 5 | 0.00308156 | 0.004602805 | 3 | 0 | 2 |
| Breast | MP -> AFGM | SSIM | 5 | 0.000053108 | 0.000062863 | 4 | 0 | 1 |
| Liver | MP -> AFGM | PSNR | 6 | 0.1712783 | 0.031606716 | 6 | 0 | 0 |
| Liver | MP -> AFGM | SSIM | 6 | 0.004409665 | 0.002678033 | 6 | 0 | 0 |

SUPPORTED within this protocol: AFGM removes measured-pose geometry dependence from trajectory construction while preserving PSNR/SSIM reconstruction performance across the evaluated sweeps. Do not turn this into a formal equivalence or unseen-domain theorem.

Not supported: all perceptual metrics are preserved or AFGM is a universal accuracy booster. Representative Liver LPIPS rises from 0.157043992 (MP) to 0.182352949 (AFGM), a +0.025308957 deterioration, even though PSNR/SSIM improve. PP-S/PP-A are not strictly better or worse in every metric.
