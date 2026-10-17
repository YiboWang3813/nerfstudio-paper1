# P0 semantic freeze and ablation infrastructure

This milestone implements ablation interfaces while preserving the existing full
production algorithm. **It is not permission to submit the formal 30k matrix.**

## Canonical paths

The actual final LaTeX compilation project has not been located. The repository
only contains `docs/production/experiment_results_io/{03_method.tex,
04_experiments_and_results.tex,tab_*.tex}`. Their `tables/main/...` inputs do not
resolve within an available main document. These exchange files are untouched.
The user has been asked to identify the real main.tex/Overleaf project. Proposed
precise corrections live in [paper_alignment.md](paper_alignment.md).

Canonical code: `methods/ours/`, `ops/ours_renderer/`; PPP remains in
`scene/container/single_sweep_container.py` (unchanged). The explicit paper code
anchor is `configs_hydra/experiment/production/ablations/ours/paper_semantic_freeze.yaml`.
Its default is uniform canonical PPP, adaptive local conditioning, full integral
global rendering, independent tau, residual-guided densification and path-aware
pruning. MP/partial-oracle rows must override `experiment.sensorless_required=false`.

## Interfaces

`model.slice_render.local_conditioning_mode = adaptive | fixed | none`:

- adaptive: unchanged production formula and computations;
- fixed: same pipeline with the depth slope zeroed (1 mm at current sigma0);
- none: unconditioned probe-space matrix and mean, eta=1, same marginal footprint,
  AABB mapper, determinant compensation, and additive shader family.

The covariance discrepancy described below applies to the legacy probe-space
matrix itself. The bypass does not silently repair that shared production math.

`model.offline.density_control.densification_policy = gradient | residual_guided`
and `pruning_policy = opacity | none | local_aware | path_aware` are independent.
Unset fields retain the legacy strategy/enable/responsibility mappings so existing
aliases/scripts continue to work. The paper alias sets both explicitly.
The legacy enable flags still act as master switches.

For path-aware pruning, an inactive global pathway (disabled, beta=0, or strength=0)
uses local-aware pruning and does not collect path statistics. A malformed or
missing A_g buffer in an otherwise active path remains the existing fail-safe
case; it is not silently called a valid no-LGR experiment. Explicit path-aware
policy uses A_g, not a legacy high-tau mode.

Global `slice_depth` vs `integral` switches and all production numbers are unchanged.
The detached opacity fallback has not been relabeled as a shared trainable control.
Acoustic controls A/B are compared in paper_alignment.md; B (frozen initialized tau)
is recommended but not implemented pending a decision.

## Critical finding

[covariance_blocker.md](covariance_blocker.md) documents a pre-existing
R S² R versus R S² Rᵀ discrepancy. It prevents declaring complete mathematical
Methods/code alignment. Default preservation and equivalence of old runs are
separate from mathematical validity; the latter cannot be established by a
wording change to a nonsymmetric matrix. No production covariance correction
has been made in this milestone.

## Source/evidence package

Base commit: `2834e7ec24a1346b2819fbb27b7efb3981f5c98c`; pre-existing dirty changes
include accumulated-A_g CUDA collection, controller integration, production
configuration promotion, and paper outputs. P0 preserves these changes. Existing
Stage-2 manifests name this commit with `git_dirty=true`; their historical patch
was not archived, so their exact original binary cannot be retrospectively proven
solely by that hash.

Evidence root: `outputs/preproduction/regression/p0_semantic_freeze_20260908/`.
It contains pre-P0 source archive, complete dirty diff/status, per-file hashes,
pre/post CUDA samples, config snapshots, a 538-run inventory, normalized config
reuse comparisons, and the two proposed matrices. P0 does not commit unrelated
user changes. A separate final infrastructure snapshot records the P0 additions.

## Reuse audit (conditional on retaining the existing full algorithm)

| Source / target | Decision | Explanation |
|---|---|---|
| Stage-2 T03, all 8 results | REUSE | Equivalent policies/topology, unchanged defaults |
| Stage-2 Joint, all 16 results | REUSE | Old no-LGR+CGA already explicitly used LP; new fallback has same behavior |
| PPP representative MP/PPP | REUSE | Use matching Stage-2 Joint full-CGA rows |
| PPP PP-S/PP-A | RERUN, 4 | Old rows were trained under obsolete CGA |
| LGR Adaptive and Full | REUSE | Matching Stage-2 local+LP and full file-pose rows |
| LGR Simple | RERUN, 2 | Old simple runs used old protection |
| LGR Plane/Fixed | NEW, 4 | New local controls |
| Acoustic control | NEW, 2 conditional | B proposed; scientific claim changes to transmission learnability |
| CGA RGD+OP | NEW, 2 | Newly isolated pruning policy |
| Stage-1 qA, 8 results | REUSE | Actual accumulated A_g; legacy tau quantile is inactive |
| Multi-sweep MP, 16 selected | REUSE | Latest main A_g runs, same holdout protocol |
| Remaining multi-sweep PPP | RERUN, 14 | Excludes the 2 representative PPP rows already available |
| Old qd/B, qp/protect, alpha grids | RERUN if selected | Old high-tau/P3 semantics do not test current A_g method |
| beta=0 / lambda=0 | REUSE local-only | Same output and local-pruning fallback; gradients have legacy reduction noise |

The `reuse_matrix.json` checks all 48 latest candidate sources using relevant
model/scene/loss/engine/seed config fields and effective-policy normalization.
Normalization explicitly removes only the inactive legacy tau quantile for the
A_g branch. It does not discard active numerical hyperparameters.

## Proposed budgets — not submitted

| New 30k runs | Full evidence | Minimal main-text |
|---|---:|---:|
| Structural (including 2 conditional acoustic runs) | 14 | 14 |
| Multi-sweep PPP | 14 | 14 |
| Representative rho (7 values, reuse defaults) | 12 | 12 |
| LGR parameters | 38 | 24 |
| CGA parameters | 76 | 36 |
| Total | **154** | **100** |

Without the unapproved acoustic control: 152 and 98. If a covariance correction
is chosen, recompute reuse and these budgets before submission.

Minimal qd/B = `{.94,.98,.99} x {.02,.05,.12}`; minimal qp/qA =
`{.05,.10,.20} x {.60,.80,.90}`. This samples around defaults and both sides with
9 points per dataset; it can show an observed plateau but cannot promise one in
advance. Full grids use the original 4x4 and 5x4 choices. Minimal alpha thresholds
are 1e-5/1e-4/1e-3; split thresholds .4/.6/1 mm. LGR minimal panels use three
values each; full choices and exact per-row overrides are in the JSON matrices.

Multi-sweep MP/PPP establishes cross-sweep validity and the representative rho
curve addresses numerical sensitivity. The 42-run historical multi-sweep rho
design need not be repeated by default. If desired later, using the new paired
validation defaults and existing representative curves reduces its additional
cost to 24 non-default runs.

From measured shared-worker 30k times (8–12 min/run), 154 runs cost approximately
20.5–30.8 worker-hours; 100 runs cost 13.3–20 worker-hours. At two workers/GPU,
rough allocation estimates are 10.3–15.4 and 6.7–10 GPU-hours; on three GPUs/six
workers, about 3.4–5.1 and 2.2–3.3 elapsed hours before overhead/queueing. These
are throughput estimates from shared-GPU logs, not exclusive-GPU benchmarks;
Plane control costs require the smoke evidence below.

## Completed validation (2026-09-08)

CUDA 11.8 extension compile passed on gpu3 / RTX 4090 (job 79902). The final
validation/smoke job 79905 completed with exit 0. The suite reports **41 passed,
1 xfailed**: the strict expected failure exposes the uncorrected rotated-covariance
mathematical discrepancy, not an accepted proof of mathematical correctness.
Tests cover the three local modes, finite-difference CUDA gradients, accumulated
path statistics, real-container adaptation events, independent OP/LP/path pruning,
no-global fallback, and PPP trajectory semantics. Existing pre-P0 adaptation
controller comparisons use eight policy/global cases with identical topology and
parameter results. Acoustic-control gradients are not tested because that control
has deliberately not been implemented.

The final paired default sample has exact CPU-reference/image/support equality;
the largest pre/post difference across gradients/path statistics is
**7.450580596923828e-6**, also observed in repeats of the unchanged pre-P0 build.
CUDA atomic reductions and the legacy shape-gradient finite-difference step
(1e-3) account for this variation. An earlier paired capture was bitwise equal
across all entries; deterministic rendering does not imply deterministic atomic
statistics/backward reductions. See `default_renderer_regression.json` for each
tensor. PPP and the ray integration source hashes are unchanged from the dirty
pre-P0 snapshot. These are scoped numerical/semantic regression results, not a
proof for every possible input or an endorsement of the covariance math.

Each real-data smoke uses 1M initial Gaussians, 100 iterations, and adaptation
events at iterations 50/100. Adaptive uses full global; Fixed/Plane disable global
and exercise local-aware fallback. No smoke metric is a paper result.

| Dataset / local mode | Peak tile–Gaussian pairs | Peak allocated GiB | Whole-process seconds | Final #G |
|---|---:|---:|---:|---:|
| Liver adaptive | 2,372,444 | 2.882 | 11.59 | 1,011,658 |
| Liver fixed | 2,372,053 | 2.882 | 6.74 | 1,007,311 |
| Liver none / Plane | 2,371,471 | 2.882 | 6.84 | 1,040,402 |
| Arm adaptive | 2,235,743 | 5.757 | 8.64 | 1,003,970 |
| Arm fixed | 2,236,691 | 5.757 | 7.32 | 1,002,758 |
| Arm none / Plane | 2,236,117 | 5.757 | 7.48 | 1,040,400 |

All six runs completed with finite rendered outputs, no crash/OOM, and two
adaptation events. Unit tests exercise actual pruning/protection because these
early smokes mostly clone rather than prune. The candidate count is the mapper's
CSR tile–Gaussian pair count, **not** pixel–Gaussian interactions; existing
candidate/corridor rules remain in place. Capacity is sufficient for these 100
iterations on a 24 GiB GPU; this does not establish late-30k capacity. Process
memory includes initialization, so equal peaks do not imply identical renderer
working sets. Whole-process times include loading/warmup and are not a controlled
runtime ablation. Earlier failed compile/test/entrypoint attempts remain in logs;
the failed Hydra smoke is preserved as `liver_adaptive_hydra_failed`.

## Final snapshot and handoff gates

`final/source.tar.gz`, `final/hashes.json`, `final/worktree.patch`,
`final/git_status.txt`, and `final/snapshot.sha256` under the evidence root record
the final infrastructure source state. `p0_only.patch` includes newly added files
as well as modifications relative to the pre-P0 dirty source. Exact typed full
anchors are `anchor_synthetic_liver.yaml` and `anchor_tus_rec_2024_qc.yaml`; each
smoke retains its own `resolved_config.yaml` and `run_manifest.yaml`.
`p0_provenance.py` verifies critical code/config hashes and writes a sidecar with
source snapshot, exact configuration and compiled renderer hashes. The resulting
anchor/smoke sidecars are additional provenance; historical manifests are not
retroactively changed. A future experiment manifest must reference this source
snapshot and its own exact config hash, and should refuse mismatched sources.
No future formal runner has been enabled by P0.

**Status: infrastructure preparation and bounded regression completed; final
paper/code semantic freeze blocked.** Before calling this paper-final, identify
the actual LaTeX compilation project and decide how to handle the pre-existing
covariance defect. Correcting it would be a production algorithm change, require
new default validation, and invalidate the unconditional use of the present reuse
budget. The acoustic A/B decision is separately pending. No 30k matrix was run
or submitted; all proposed run rows carry `execution_authorized=false`.
