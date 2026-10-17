# Final Experiment Completeness

| Evidence category | Status | Basis |
|---|---|---|
| Main comparison | COMPLETE |78Fixed+Full MP cases;7unchanged baselines;same cases;raw ranks |
| Joint component ablation | COMPLETE |16slots,8combinations x2representatives,final semantics |
| PPP internal | COMPLETE |MP/PP-S/PP-A/PPP x2,Fixed+Full |
| PPP multi-sweep | COMPLETE |6Liver+5Arm+5Breast MP/PPP,PSNR/SSIM |
| PPP rho sensitivity | COMPLETE |7values x2,existing final runs |
| LGR internal | COMPLETE |5rows,6Liver+5Arm multi-sweep |
| LGR hyperparameters | COMPLETE |5parameter axes,raw metrics/deltas;Nq non-monotonicity disclosed |
| CGA internal | COMPLETE |5variants x2,PSNR/LPIPS/population |
| CGA population dynamics | COMPLETE |Curves,event/spawn/prune/protection statistics |
| CGA hyperparameters | COMPLETE |Two3x3 grids,tau3points;active/saturated controls audited |
| FPS | COMPLETE |Fixed models,5warmup/20frames/3passes,2frozen cases/dataset |
| Qualitative | COMPLETE |4Main datasets+8internal sheets;GT/crop/config/source integrity |

**Are there any scientifically necessary additional training experiments? NO.**

No explicit implementation bug was identified by this final audit. Nq's
non-monotonic trained-model performance is an observed limitation, not sufficient
evidence of a coding error and not a reason to launch new runs. Completeness is
conditional on the restrained supported claims in final_claim_matrix.md, NOT on
proving every previously proposed optimistic claim. No multi-seed/generalization
beyond this protocol, clinical validation or exact numerical convergence claim.

Main deliberately retains MP, even though the final system also offers/uses PPP
in its acquisition-guided formulation. State the Main pose protocol transparently;
do not relabel MP numbers as PPP results. This accepted scope is not a missing run.

No more training, no expanded search, no test-driven default changes. Production
remains the P2 Fixed freeze. Canonical LaTeX source is still unconfirmed; assets
and prose/patch suggestions are prepared without editing03_method.tex or04_results.
Human manuscript integration/layout is editorial work, not incomplete experiments.
