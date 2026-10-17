# P2 Fixed+Full final method refresh

User-selected final design: fixed finite-slice sigma_ele1mm + full causal
ray-integrated global transmission + independent tau + corrected full CGA.
Adaptive remains an alternative only. No Hyperparameter Batch2 execution.

Production changes are restricted to2 Hydra YAML defaults: `model/ours_us_focus`
and the formal `production/ablations/ours/paper_semantic_freeze` entrypoint.
Generic low-level dataclass defaults remain compatibility behavior for older
probe3d/plane_cond2d families; changing those would invalidate the legacy-family
validator or require unrelated renderer logic changes. Formal Hydra defaults
explicitly select Fixed. The depth slope .01 remains in the config for the
Adaptive alternative, but the fixed-mode parameter vector sets it to zero.

All other constants and algorithm files, including CUDA binary, remain unchanged.
Regression tests compare the default resolved production config to existing
Fixed candidates, and preserve explicit None/Fixed/Adaptive reference/gradient,
corrected covariance, pruning/adaptation/no-global fallback and PPP tests.
CUDA compilation uses a separate build directory; the runtime binary is retained.

Exact-source freeze: see `freeze.json`, which points to the immutable archive and
SHA after regression validation. Complete dirty source/patches, binary/environment,
representative MP/PPP configs and tests accompany that archive. Final submission
matrix/configs/runner are separately sealed because their provenance includes
the archive hash itself.

Main78: no retraining, all FINAL-REUSE after exact config equivalence.
Baselines: FINAL-REUSE on unchanged6/9/33/30 cases. Keep original execution hashes
and separately label validated P2 equivalence; do not forge P2 training provenance.
The8 expressly approved FPS reconstructions are separate from this refresh budget.

Final Joint: Plane/no global when LGRoff; Fixed+Full when LGRon; GD+opacity when
CGAoff; RGD+path-aware when CGAon, falling back to local-aware without global.
PPPoff=MP; PPPon=uniform nominal acquisition-guided trajectory.16 slots on Liverl1
and Arm010_LH_Per_L_DtP_qc_0005_0370.

PPP representative MP/PP-S/PP-A/PPP; multi-sweep6Liver/5Arm/5Breast MPvsPPP;
representative rho .60/.75/.90/1/1.10/1.25/1.50. Shared defaults are deduplicated.
CGA GD/RGD/RGD+OP/RGD+LP/Full on MP+Fixed+Full. Extract population and event
statistics from the same runs; no new population-only training.

Final LGR: Plane/Fixed/Adaptive/Fixed+Simple/Fixed+Full,6Liver/5Arm means;0new runs.
Fixed vs Adaptive compares formulations, not progressive module additions.
Global-on/off includes the prescribed CGA protection interaction.

New outputs live under `outputs/production/{comparisons,ablations/ours}/P2_fixed_full_final_20260909`.
Legacy and Adaptive evidence is retained as history, not final Main Ours.
All refresh model files are initially retained; any future removal must follow
verification of required figures/metrics/provenance. FPS8 models remain retained.

Canonical LaTeX is still unconfirmed. `../p0_semantic_freeze/paper_alignment_p2.patch`
is an apply-ready semantic proposal; no exchange file is modified. Tables/figures
are generated under P2 and listed in the final data-source manifest.

After final reports and a proposed exact Batch2 plan, pause for user review.
