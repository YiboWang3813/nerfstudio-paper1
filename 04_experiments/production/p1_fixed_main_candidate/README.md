# Paper-wide Fixed+Full candidate validation

Authorized 2026-09-09: exactly the frozen main 6 Liver / 9 Spine / 33 Arm /
30 Breast cases. This is a paired local-design decision, not an official main-table
replacement, new baseline run or hyperparameter matrix.

`fixed_full_main_candidate_matrix.json` contains 78 Fixed slots and their exact
Adaptive source mappings. Expected dedup: 11 Fixed reuse (6 Liver, 5 Arm), 67 new
(9 Spine, 28 Arm, 30 Breast). All 78 Adaptive results are reused. Each candidate's
normalized resolved training+metric config must equal its Adaptive counterpart
after changing only `local_conditioning_mode` to `fixed`. Fixed uses the existing
1-mm sigma_ele_0 interface; all other formulation settings remain frozen.

MP / full integral global / independent learnable tau / full corrected CGA /
initial 1M / 30k / seed42 / same split and evaluation. Group `structural` in the
runner is a bookkeeping/retention label for this candidate design study, not a
different training configuration. No FPS is required in this phase.

Production source archive SHA-256:
`b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.
Production/binary files are checked against this freeze before every run. The
original guarded runner is copied with only protocol-directory, matrix-filename
and manifest-label substitutions. No production code or default is edited.
`submission_hashes.json` seals new orchestration/reporting/configs; this is not
a new algorithm freeze. Existing outputs are never overwritten.

Four disjoint workers are balanced by previous same-case runtime, not quality.
GPU names are verified inside Slurm allocations before launching training.
Failure records STOP; other workers finish their active run but start no new one.

Quality protocol: frame means from final 30k test metrics, then equal case weights.
Six metrics; paired Fixed minus Adaptive means/sample std/median/win-tie-loss.
Numerical tie threshold is 1e-6; reverse the direction for L1/LPIPS/DISTS/GMSD.
No p-values. Rankings separately substitute Fixed or Adaptive into the same
seven-baseline comparison, never treating both as a ninth method.

Engineering: mean/std final population, training-wrapper time, per-run peak
local/global candidate counts and memory. Save both mean and maximum of peaks.
Shared-worker wall time is not a controlled speed claim. No new FPS measurements.

Inherited main qualitative cases/crops (no result-dependent selection):

| Dataset | Case | Item | Crop xywh |
|---|---|---:|---|
| Liver | l3 | 27 | 19,0,53,107 |
| Spine | left3_2 | 17 | 59,0,36,215 |
| Arm | 010_RH_Per_S_PtD_qc_0165_0364_s02 | 18 | 129,24,92,69 |
| Breast | validation_116 | 39 | 150,23,85,67 |

Only after all final metrics, frame IDs, image exports, configs, population and
provenance validate may the new model/optimizer files be removed, with SHA256
deletion ledgers. All numerical results/logs/images and reused artifacts remain.
Removing models means this batch cannot be re-rendered/FPS-tested from those
files later; such future work is not part of the current authorization.

After the paired report/contact sheets, review F1/F2/F3 and pause. No automatic
default/Methods changes, no dataset-specific choice, and no Joint/PPP/CGA/rho or
Hyperparameter Batch 2 submissions.
