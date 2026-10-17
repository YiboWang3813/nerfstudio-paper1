# Final targeted local-global interaction study

Authorized2026-09-09: same six Liver sweeps and five frozen Arm sweeps as the prior
LGR confirmation. Local fixed/adaptive x global none/simple/full =66 slots.
Exact semantic+metric config dedup against completed resolved configs:44 REUSE,
22 NEW (Fixed+Simple and Fixed+Full only). No Batch2, no new PPP/CGA/rho variants.

MP,initial1M,30k,seed42,same split/optimizer/evaluation. Fixed sigma=1mm through
the existing fixed-mode interface, not a new renderer. Full corrected CGA, with
the existing local-aware fallback only when global is absent. No default changes.

Production source archive SHA256 remains
`b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.
Original guarded Batch1 runner is copied textually with only protocol directory,
matrix filename and batch-label substitutions; execution still verifies unchanged
production/binary hashes. `submission_hashes.json` pins submission code/matrix/
exact configs. This is not a new method freeze.

Jobs80101(gpu2,two4080SUPER),80102(gpu3,two4090). Partition balanced using previous
same-case/global run wall times, not quality. ActualGPU/source/config stored per run.
No training run is duplicated; the output root is a new `lgr_interaction` subfolder
under the existing corrected ablation namespace. Existing Batch1/main/FPS/figures
and previous LGR results are unchanged.

Postprocessing only after both jobs succeed: report all66 slots, paired mean/sample
std(ddof1)/numeric WTL(tolerance1e-6, LPIPS direction reversed), and paired
difference-in-differences. Differences are trained whole-system interactions with
CGA responses, not fixed-topology renderer-only causal derivatives. MixedGPU and
single-seed effects are not statistical significance evidence.

Inherited Renderer contact-sheet selections: Liver l1/item5/crop[0,0,53,107];
Arm010_LH_Per_L_DtP_qc_0005_0370/item57/crop[194,97,92,69]. No selection by new
quality. Full fixed/adaptive and six-combination contact sheets require identical
GT hashes. Models/solver files may be deleted only after completed metrics/frame/
export validation; all reused artifacts stay untouched. Numerical data/images remain.

Seventeen reporting/statistics/cleanup unit tests passed before submission.
No automatic production selection or Batch2 submission after the report. Human review
must classify CaseA/B/C using absolute quality as well as interaction strength.
