# Ours Production Ablation Lifecycle

Date created: 2026-06-22

Last reset: 2026-06-27

Status: **expanded paper-facing ablation package completed; final figures generated except deferred T13**

Legacy rerun status: **116/116 GS-strict rerun rows completed on 2026-06-27**

Owner: Ours method / production ablation

This document is the lifecycle controller for production-grade Ours ablations.
Agents should read this file before creating aliases, launching Slurm jobs,
clearing outputs, reporting tables, or deciding whether an old result can be
used.

Related rules:

```text
docs/operations/pipeline_skills.md
docs/operations/scope_skills.md
docs/operations/git_workflow_skills.md
docs/operations/slurm_skills.md
docs/production/multi_comparisons.md
```

Validated method and diagnostic evidence:

```text
docs/methods/ours/validated_paper_innovations.md
docs/methods/ours/local_global_renderer/four_dataset_30k_validation.md
docs/methods/ours/responsibility_density_control/fixed_protocol_validation.md
docs/methods/ours/easy_pose/full_validation_and_spacing_stress.md
docs/production/problems_during_production/ours_gs_strict_diagnostics.md
```

## 1. Reset Rationale

The previous Ours production ablation package was completed under an older
protocol. It used:

```text
checkpoint selection = best test/psnr
loss = 0.65 L1 + 0.35 SSIM + 50.0 scale_wall
metrics = PSNR, L1, SSIM, LPIPS
scale activation = not consistently aligned to the final comparison Ours row
```

The latest multi-method comparison freezes a different Ours production row:

```text
checkpoint selection = final 30k checkpoint
loss = 0.8 L1 + 0.2 SSIM
disabled auxiliary training losses = scale_wall, highpass, LPIPS, DISTS, GMSD, and non-approved method-specific losses
metrics = PSNR, L1, SSIM, LPIPS, DISTS, GMSD
scale activation = bounded_sigmoid
pose source = file poses for the main comparison row
density-control timing = start 1000, end 25000, freq 2000
```

Policy after the 2026-06-27 reset:

```text
If an experiment is not absolutely safe to reuse, rerun it.
Paper-facing Ours ablation numbers must be generated under the latest
GS-strict Ours production config.
Old production ablation numbers are historical evidence only and must not be
copied into current result tables.
```

## 2. Reuse Classification

### 2.1 Absolutely Safe to Reuse From Pre-Reset Work

For work produced before the 2026-06-27 protocol reset, only these non-result
assets can be reused without rerunning. Completed rows from the 2026-06-27
GS-strict rerun are governed by Section 2.2 instead.

| Asset | Reuse status | Reason |
| --- | --- | --- |
| Method descriptions in `docs/methods/ours/**` | Reuse as design history | They document why modules exist, not current production numeric claims. |
| Ours GS-strict diagnostics conclusions | Reuse as configuration rationale | A1 corridor, B2 bounded scale, and B3 soft attenuation were run against the newer GS-strict family and support current config choices. |
| Dataset and sweep scopes | Reuse | The same four-dataset and representative-sweep scopes remain appropriate. |
| Existing script structure | Reuse after verification | Submitters and summarizers may be adapted, but must be checked against the latest config before launch. |
| Old git history | Reuse as audit trail | Old numbers remain recoverable from history but are not active table data. |

### 2.2 Rerun Requirement After Protocol Reset

The 2026-06-27 reset required all old paper-facing production ablation tables
to be regenerated under the latest GS-strict Ours row. That baseline rerun is
now complete and remains valid as historical/current evidence where its rows
match the expanded paper plan.

Current rule:

```text
Do not rerun a completed row merely because it came from the 2026-06-27
GS-strict rerun.

Do rerun or newly run any row that is missing from the expanded paper-facing
plan, any row that used a non-standard dataset for a main paper table, or any
row whose required overrides differ from the final comparison Ours config.
```

Reset-era status table:

| Table | Reset-era data status | Current action |
| --- | --- | --- |
| Stage 0 dry run | completed under current config | No action unless scripts change. |
| T01 module ladder | partially covered by completed T02/T06 rows | Run only missing 2x2x2 combinations b/d/f. |
| T02 renderer x DC factorial | completed under current config but not the new renderer-internal table | Run simple and q=1 renderer rows on the standard datasets. |
| T03 renderer component | completed as historical source | Superseded by new T02 renderer-internal table. |
| T04 responsibility DC component | completed as historical source | Superseded by new T03 DC-internal table. |
| T05 responsibility DC sub-components | completed on spine + TUS | Run missing synthetic rows needed by new T03; keep old spine rows as supplement. |
| T06 easy-pose file vs canonical | completed under current config | Reuse file/uniform rows; run missing step/axis rows for new T04. |
| T07 easy-pose spacing tolerance | completed under current config | Reuse for T12 rho sensitivity figure. |
| T08 easy-pose negative control | completed under current config | Supplement only unless discussed in limitations. |
| T09 seed stability | completed under current config | Supplement/robustness evidence. |
| T10 efficiency and mechanism | generated from completed rerun logs | Regenerate only if final paper figures require updated formatting. |

## 3. Latest Frozen Protocol

Unless a row explicitly says otherwise, all reruns must match the latest
GS-strict Ours production comparison row:

```text
source alias = configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml
num_init = 1000000
model.gauss_init.num_points = 1000000
iterations = 30000
test interval = 30000
save interval = 30000 only as a transient runtime need
checkpoint selection = final 30k checkpoint
loss = 0.8 L1 + 0.2 SSIM
disabled auxiliary training losses = scale_wall, highpass_l1, LPIPS, DISTS, GMSD, scale_aniso, opacity, attenuation losses, pose losses
loss input clamp = false
metrics = PSNR, L1, SSIM, LPIPS, DISTS, GMSD
scale activation = bounded_sigmoid
pose_type = absolute
pose_mode = file, except easy-pose rows
density-control timing = start 1000, end 25000, freq 2000
checkpoint retention = none after successful metric summary
primary GPU = gpu3 / RTX 4090 when available
WANDB_MODE = offline
```

Current Ours renderer / DC production anchor:

```text
model.slice_render.ray_attenuation_formula=integral
model.slice_render.ray_attenuation_quadrature_points=3
model.slice_render.ray_attenuation_candidate_k_sigma=2.0
model.slice_render.ray_attenuation_beta=0.80
model.slice_render.ray_attenuation_strength=0.55
model.slice_render.ray_attenuation_min_transmission=0.35
model.offline.density_control.strategy=contribution_residual
model.offline.density_control.thresholds.prune_opacity=0.0001
model.offline.density_control.responsibility.low_contribution_quantile=0.10
model.offline.density_control.responsibility.protect_attenuation_enabled=true
model.offline.density_control.responsibility.protect_attenuation_quantile=0.80
```

Dataset-specific overrides inherited from the comparison runner:

```text
tus_rec_2024_qc: ++model.slice_render.elev_corridor_half_mm=0.202
tdsc_abus_2023: ++model.slice_render.elev_corridor_half_mm=0.238
```

## 4. Storage Policy

Effective 2026-06-27:

```text
Do not retain checkpoint payloads for Ours ablation runs managed by the Ours
ablation agent.

Required retained artifacts are result summaries, per-run metric tables,
per-frame metrics when produced, scalar logs, efficiency logs, manifests,
Hydra/config snapshots, analysis notes, and generated tables/figures.

Checkpoint files under checkpoints/ are storage-only intermediates. After a run
has successfully written metrics and summaries, delete checkpoint payloads or
run with a no-retention cleanup path. Future Ours ablation reporting must record
the results in detail instead of relying on checkpoint retention.
```

Current cleanup status:

| Date | Path | Action | Status |
| --- | --- | --- | --- |
| 2026-06-27 | `outputs/production/ablations/ours/**/checkpoints/*` | Deleted old checkpoint payloads | Completed |
| 2026-06-27 | `outputs/production/ablations/ours/*` | Cleared old production ablation outputs for rerun prep | Completed |

## 5. Production Roots

```text
configs_hydra/experiment/production/ablations/ours/
scripts/production/ablations/ours/
outputs/production/ablations/ours/
```

The output root is reset. New stages should recreate this structure:

```text
README.md
MANIFEST.yaml
summary.tsv
tables/metrics_by_run.tsv
tables/paired_deltas.tsv
tables/paper_table.md
notes/analysis.md
```

## 6. Dataset Scope

### 6.0 Paper-Facing Standard Ablation Scope

Effective 2026-07-02, all paper-facing standard ablation tables must use the
same one synthetic dataset and one real dataset:

| Role | Dataset | Sweep | Use |
| --- | --- | --- | --- |
| Synthetic | `synthetic_liver` | `l1` | All standard ablation tables. |
| Real | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | All standard ablation tables. |

Standard paper-facing ablation tables should report only:

```text
PSNR, SSIM, LPIPS
```

Thus a standard two-dataset metric block has six metric columns:

```text
Syn PSNR, Syn SSIM, Syn LPIPS, Real PSNR, Real SSIM, Real LPIPS
```

Earlier rows on `spine_phantom`, `tdsc_abus_2023`, or additional TUS/synthetic
sweeps remain useful as supplemental robustness, negative-control, or
generalization evidence, but they should not define the main standard ablation
tables. In particular, old T05 rows on `spine_phantom / left1` are now
supplemental hard-case evidence; a paper-facing DC sub-component table still
needs matching `synthetic_liver / l1` rows if it is promoted to the main
standard ablation set.

### 6.1 Supplemental Four-Dataset Core Scope

Used for supplemental robustness and historical renderer/DC interpretation when
pose input is file-pose:

| Dataset | Scene config | Sweep | Test protocol | Role |
| --- | --- | --- | --- | --- |
| `synthetic_liver` | `liver` | `l1` | holdout_interval, trimmed 80/20 split | Synthetic controlled one-sweep. |
| `spine_phantom` | `spine_phantom` | `left1` | holdout_interval, trimmed 80/20 split | Hard sparse/fragmented topology case. |
| `tus_rec_2024_qc` | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | holdout_interval, trimmed 80/20 split | Clean freehand QC case. |
| `tdsc_abus_2023` | `tdsc_abus_2023` | `validation_100` | holdout_interval, trimmed 80/20 split | ABUS-style independent sweep. |

### 6.2 Easy-Pose Positive / Rho-Sensitivity Scope

Used for broad Easy-Pose pose-dependency reduction and rho-sensitivity
evidence. The main paper-facing Easy-Pose tables still use the Section 6.0
standard synthetic + real pair. Do not use `spine_phantom` as positive evidence.

| Dataset | Sweeps | Pairing |
| --- | --- | --- |
| `synthetic_liver` | `l1,l2,l3,r1,r2,r3` | file pose vs canonical sweep |
| `tus_rec_2024_qc` | `010_LH_Per_C_DtP_qc_0140_0514`, `010_LH_Per_L_DtP_qc_0005_0370`, `010_RH_Per_L_DtP_qc_0005_0574`, `011_LH_Per_S_PtD_qc_0008_0447`, `011_RH_Per_L_DtP_qc_0007_0584` | file pose vs canonical sweep |
| `tdsc_abus_2023` | `validation_100` through `validation_104` | file pose vs canonical sweep |

### 6.3 Supplemental / Legacy Fine-Ablation Scope

| Dataset | Sweep | Why |
| --- | --- | --- |
| `spine_phantom` | `left1` | Strongest topology/DC stress case. |
| `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | Clean real freehand case. |

This scope is no longer the default for main paper-facing ablation tables.
Use it only for supplemental robustness, historical interpretation, or
hard-case DC discussion. The main paper standard remains Section 6.0:
`synthetic_liver/l1` plus
`tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370`.

## 7. Required Script Updates Before Relaunch

Existing scripts may be reused only after verification against the latest
protocol:

```text
configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml
scripts/production/ablations/ours/run_ours_ablation_task_1m30k.sh
scripts/production/ablations/ours/submit_all_rerun_multiworker_1m30k.sh
scripts/production/ablations/ours/submit_stage0_dry_run_1m30k.sh
scripts/production/ablations/ours/submit_t02_renderer_dc_factorial_1m30k.sh
scripts/production/ablations/ours/submit_t05_dc_subcomponents_1m30k.sh
scripts/production/ablations/ours/submit_t06_easy_pose_pairs_1m30k.sh
scripts/production/ablations/ours/submit_t08_easy_pose_negative_control_1m30k.sh
scripts/production/ablations/ours/submit_t09_seed_stability_1m30k.sh
scripts/production/ablations/ours/submit_t10_efficiency_mechanism.sh
scripts/production/ablations/ours/summarize_ablation_run.py
scripts/production/ablations/ours/generate_ablation_report.py
```

Before any Slurm launch, verify:

| Check | Required value |
| --- | --- |
| Loss | `0.8 L1 + 0.2 SSIM`, no `scale_wall` |
| Metrics | `psnr,l1,ssim,lpips,dists,gmsd` |
| Scale activation | `bounded_sigmoid` |
| Checkpoint rule | final 30k only |
| Output cleanup | delete checkpoint payloads after summary |
| TUS override | `elev_corridor_half_mm=0.202` |
| TDSC override | `elev_corridor_half_mm=0.238` |

## 8. Completed Historical Rerun Package

The following package was completed before the expanded paper-facing ablation
plan was designed. It is still valid for completed rows that match the latest
GS-strict production config, but it is no longer the full paper ablation plan.

Historical full rerun job:

```text
Slurm job = 66353
Node = gpu3
GPU request = 2 RTX 4090 GPUs
Internal parallelism = 4 workers, 2 workers per GPU
Completed rows = 116 / 116
Worker failures = 0
Checkpoint payload files retained = 0
Summary = outputs/production/ablations/ours/summary.tsv
Generated tables = outputs/production/ablations/ours/tables/
Analysis note = outputs/production/ablations/ours/notes/analysis.md
```

| Table | Purpose | Scope | Runs | Rerun status |
| --- | --- | --- | ---: | --- |
| Stage 0 | Config/script smoke | TUS-QC file + easy-pose | 2 short runs | completed |
| T02 | Renderer x DC factorial | 4 core datasets x 4 variants | 16 | completed |
| T01 | Old module ladder | Derived from new T02 + new T06 representative rows | 0 extra | superseded by expanded 8-row T01; 3 combinations still missing |
| T03 | Old renderer component | Derived from new T02 | 0 extra | historical source only; superseded by new T02 |
| T04 | Old responsibility DC component | Derived from new T02 | 0 extra | historical source only; superseded by new T03 |
| T05 | DC sub-components | spine + TUS-QC x 4 variants | 8 | completed |
| T06 | Easy-pose file vs canonical | 16 positive-scope pairs | 32 | completed |
| T07 | Easy-pose spacing tolerance | 3 datasets, 6 sweeps stress grid | 42 | completed |
| T08 | Easy-pose negative control | spine file/canonical rows | 4 | completed |
| T09 | Seed stability | 2 datasets x 2 variants x 3 seeds | 12 | completed |
| T10 | Efficiency and mechanism | generated from new summaries/logs | 0 extra | generated from completed rows |

Minimum rerun package before paper-facing claims was completed:

```text
Stage 0 + T02 + T05 + T06 + T08 + T09 + T10 generation
```

Historical full rerun package was completed:

```text
Stage 0 + T02 + T05 + T06 + T07 + T08 + T09 + T10 generation
```

## 9. Historical Completed Result Registries

The following registries are generated from the 2026-06-27 GS-strict rerun.
They are valid sources for rows that match the expanded paper-facing plan, but
they are not by themselves the complete final paper ablation package.
Detailed per-run values are retained in:

```text
outputs/production/ablations/ours/summary.tsv
outputs/production/ablations/ours/tables/metrics_by_run.tsv
```

### Stage 0 Registry

| Job ID | Script | Output dir | Status | Notes |
| --- | --- | --- | --- | --- |
| 66353 | `scripts/production/ablations/ours/submit_all_rerun_multiworker_1m30k.sh` | `outputs/production/ablations/ours/runs/stage0` | completed | 500-step file-pose and canonical-sweep smoke rows completed; summarizer and checkpoint cleanup validated. |

### T01 Three-Module Joint Ablation

T01 is the paper-facing three-module joint ablation table. It should report only
PSNR, SSIM, and LPIPS on one synthetic dataset and one real dataset:

Motivation and paper role:

```text
T01 answers the highest-level reviewer question: which proposed components are
actually responsible for the final Ours behavior? It treats local-global
rendering (LG), responsibility density control (DC), and Easy-Pose (EP) as
three switchable modules and evaluates their joint contribution under the same
fair production loss used in comparisons.
```

What this experiment should prove:

```text
1. The final Ours stack is not an arbitrary bundle of tricks.
2. LG is the main reconstruction-quality contributor, especially on real TUS-QC.
3. EP can be added without degrading the final stack and can improve regular
   real sweeps by replacing file poses with a canonical sweep prior.
4. DC is retained as a topology-control mechanism that is near-neutral in final
   image metrics, not as the primary PSNR source.
```

Writing guidance:

```text
When writing from T01, first compare Base vs +LG to establish the largest
module-level effect. Then compare +LG vs +LG+DC to show that DC does not create
a hidden unfair accuracy gain and does not materially damage reconstruction.
Finally compare +LG+DC vs +LG+DC+EP to argue that Easy-Pose preserves or
improves quality while removing dependence on per-frame file poses. If synthetic
and real trends differ, phrase the synthetic result as a PSNR/perceptual
tradeoff and use TUS-QC as the main real-data evidence.
```

```text
Synthetic row = synthetic_liver / l1
Real row = tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370
Metrics = PSNR, SSIM, LPIPS
Table shape = 8 rows x 10 columns
Columns = index, LG, DC, EP, Syn PSNR, Syn SSIM, Syn LPIPS, Real PSNR, Real SSIM, Real LPIPS
LG = local-global renderer
DC = responsibility density control
EP = Easy-Pose canonical sweep
```

The table is a full 2x2x2 module factorial. Current status: **8/8 rows
completed**.

| index | LG | DC | EP | Synthetic source | Real source | Syn PSNR | Syn SSIM | Syn LPIPS | Real PSNR | Real SSIM | Real LPIPS | Status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a |  |  |  | T02 `local_gradient` | T02 `local_gradient` | 29.849 | 0.682 | 0.270 | 32.516 | 0.864 | 0.164 | completed |
| b |  |  | check | expanded T01 | expanded T01 | 29.836 | 0.681 | 0.279 | 32.725 | 0.866 | 0.166 | completed |
| c |  | check |  | T02 `local_responsibility` | T02 `local_responsibility` | 29.854 | 0.682 | 0.268 | 32.525 | 0.864 | 0.164 | completed |
| d |  | check | check | expanded T01 | expanded T01 | 29.842 | 0.681 | 0.279 | 32.730 | 0.866 | 0.166 | completed |
| e | check |  |  | T02 `localglobal_gradient` | T02 `localglobal_gradient` | 29.749 | 0.678 | 0.234 | 32.768 | 0.866 | 0.141 | completed |
| f | check |  | check | expanded T01 | expanded T01 | 29.753 | 0.677 | 0.244 | 33.004 | 0.868 | 0.142 | completed |
| g | check | check |  | T02/T06 file-pose full Ours | T02/T06 file-pose full Ours | 29.746 | 0.677 | 0.233 | 32.764 | 0.866 | 0.141 | completed |
| h | check | check | check | T06 `canonical_sweep` | T06 `canonical_sweep` | 29.755 | 0.677 | 0.244 | 32.998 | 0.868 | 0.142 | completed |

Completed result source:

```text
outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t01_three_module_joint.md
```

Current conclusion: the local-global renderer is the dominant real-data quality
contributor, responsibility DC is neutral to weakly positive, and Easy-Pose
retains or improves quality when added to the full Ours stack.

### T02 Local-Global Renderer Internal Ablation

T02 is the paper-facing renderer-internal ablation table. It is a follow-up to
T01: after T01 shows that the local-global renderer is the dominant module, T02
opens the renderer and compares progressively stronger renderer modes.

Motivation and paper role:

```text
T02 opens the strongest module from T01. A reviewer should be able to see that
the improvement is not merely caused by enabling any global branch, but by the
specific local-global rendering formulation and its integral attenuation design.
```

What this experiment should prove:

```text
1. Local-only rendering is a meaningful baseline because it removes the proposed
   global attenuation pathway while keeping the rest of Ours fixed.
2. Simple global attenuation tests whether a coarse global correction is enough.
3. Integral q=1 and q=3 test whether the proposed ray-aware integral treatment
   adds value beyond a one-sample approximation.
4. The production q=3 setting should be strongest or clearly defensible on the
   real representative dataset, even if synthetic PSNR shows a tradeoff.
```

Writing guidance:

```text
Use T02 to explain the renderer mechanism after the T01 module result. The
preferred narrative is: local-only can be competitive on synthetic PSNR, but the
local-global variants reduce perceptual artifacts and q=3 gives the best
real-data reconstruction quality. If the synthetic row favors local-only PSNR,
do not hide it; describe it as evidence that the renderer is not overfit to the
synthetic metric, while the real row and LPIPS support the production design.
```

Use the standard ablation scope:

```text
Synthetic row = synthetic_liver / l1
Real row = tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370
Metrics = PSNR, SSIM, LPIPS
Fixed modules = responsibility DC on, Easy-Pose off / file pose
Table shape = 4 rows x 8 columns
Columns = index, Renderer mode, Syn PSNR, Syn SSIM, Syn LPIPS, Real PSNR, Real SSIM, Real LPIPS
```

Recommended rows:

| index | Renderer mode | Synthetic source | Real source | Syn PSNR | Syn SSIM | Syn LPIPS | Real PSNR | Real SSIM | Real LPIPS | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a | local-only | T02 `local_responsibility` | T02 `local_responsibility` | 29.854 | 0.682 | 0.268 | 32.525 | 0.864 | 0.164 | completed |
| b | local + simple global attenuation | expanded T02 | expanded T02 | 29.782 | 0.678 | 0.232 | 32.660 | 0.866 | 0.141 | completed |
| c | local + integral attenuation, q=1 | expanded T02 | expanded T02 | 29.741 | 0.678 | 0.235 | 32.759 | 0.866 | 0.142 | completed |
| d | local + integral attenuation, q=3 | T02 `localglobal_responsibility` | T02 `localglobal_responsibility` | 29.746 | 0.677 | 0.233 | 32.764 | 0.866 | 0.141 | completed |

Definitions:

```text
local-only:
  model.slice_render.ray_attenuation_enabled=false

local + simple global attenuation:
  model.slice_render.ray_attenuation_enabled=true
  model.slice_render.ray_attenuation_formula=slice_depth
  model.slice_render.ray_attenuation_quadrature_points=1
  keep production beta/strength/min_transmission

local + integral attenuation, q=1:
  model.slice_render.ray_attenuation_enabled=true
  model.slice_render.ray_attenuation_formula=integral
  model.slice_render.ray_attenuation_quadrature_points=1
  keep production beta/strength/min_transmission

local + integral attenuation, q=3:
  model.slice_render.ray_attenuation_enabled=true
  model.slice_render.ray_attenuation_formula=integral
  model.slice_render.ray_attenuation_quadrature_points=3
  keep production beta/strength/min_transmission
```

Completed result source:

```text
outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t02_renderer_internal.md
```

Do not use a `global-only` row in the paper table. In the current renderer,
global attenuation is a modulation of the local slice renderer, not an
independent image-producing branch. A forced global-only proxy would be less
faithful to the implementation and harder to defend.

### T02 Historical Renderer x DC Factorial Source

Detailed table:

```text
outputs/production/ablations/ours/tables/t02_renderer_dc_factorial.md
```

Key decision: local-global rendering is strongly positive. The production
anchor `localglobal_responsibility` is tied with the best row on spine and TUS,
near-best on TDSC, and trades a small synthetic PSNR drop for better perceptual
metrics.

### T03 Responsibility Density Control Internal Ablation

T03 is the paper-facing DC-internal ablation table. It is a follow-up to T01:
after T01 establishes responsibility DC as a retained module, T03 opens the DC
strategy and compares progressively more complete density-control mechanisms.

Motivation and paper role:

```text
T03 prevents a misleading interpretation of DC. DC is not introduced as another
loss term or a renderer shortcut; it is a density-control policy. The internal
ablation separates gradient-only DC, responsibility-based densification,
responsibility pruning without attenuation protection, and the full protected
responsibility DC.
```

What this experiment should prove:

```text
1. Responsibility scores can replace pure gradient triggers without causing a
   large quality collapse.
2. The pruning/protection pieces are stable enough to keep the full DC policy
   near the best internal variants.
3. DC should be described as mechanism-oriented and performance-neutral to
   weakly positive under the fair L1+SSIM loss.
```

Writing guidance:

```text
When writing T03, do not oversell DC as a PSNR booster. The correct argument is
that responsibility DC changes how Gaussians are allocated while preserving
final reconstruction quality. If gradient DC is slightly higher on one metric,
state that the gap is tiny and then point to the full DC row being comparable
across PSNR/SSIM/LPIPS. Use this table to justify retaining DC as a stable
topology-control component rather than as the dominant accuracy module.
```

Use the standard ablation scope:

```text
Synthetic row = synthetic_liver / l1
Real row = tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370
Metrics = PSNR, SSIM, LPIPS
Fixed renderer = local + integral attenuation, q=3
Fixed pose = file pose / Easy-Pose off
Table shape = 4 rows x 8 columns
Columns = index, DC strategy, Syn PSNR, Syn SSIM, Syn LPIPS, Real PSNR, Real SSIM, Real LPIPS
```

Recommended rows:

| index | DC strategy | Synthetic source | Real source | Syn PSNR | Syn SSIM | Syn LPIPS | Real PSNR | Real SSIM | Real LPIPS | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a | gradient DC | T02 `localglobal_gradient` | T02 `localglobal_gradient` | 29.749 | 0.678 | 0.234 | 32.768 | 0.866 | 0.141 | completed |
| b | responsibility densify only | expanded T03 | T05 `responsibility_densify_only` | 29.742 | 0.678 | 0.234 | 32.730 | 0.866 | 0.142 | completed |
| c | responsibility + prune, no attenuation protection | expanded T03 | T05 `responsibility_no_attenuation_protect` | 29.752 | 0.678 | 0.232 | 32.761 | 0.866 | 0.141 | completed |
| d | full responsibility DC | T02 `localglobal_responsibility` | T02/T05 `responsibility_full` | 29.746 | 0.677 | 0.233 | 32.764 | 0.866 | 0.141 | completed |

Definitions:

```text
gradient DC:
  model.offline.density_control.strategy=gradient
  model.offline.density_control.responsibility.enabled=false

responsibility densify only:
  model.offline.density_control.strategy=contribution_residual
  model.offline.density_control.responsibility.enabled=true
  model.offline.density_control.enable.prune=false

responsibility + prune, no attenuation protection:
  model.offline.density_control.strategy=contribution_residual
  model.offline.density_control.responsibility.enabled=true
  model.offline.density_control.enable.prune=true
  model.offline.density_control.responsibility.protect_attenuation_enabled=false

full responsibility DC:
  model.offline.density_control.strategy=contribution_residual
  model.offline.density_control.responsibility.enabled=true
  model.offline.density_control.enable.prune=true
  model.offline.density_control.responsibility.protect_attenuation_enabled=true
  model.offline.density_control.responsibility.protect_attenuation_quantile=0.80
```

Completed result source:

```text
outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t03_dc_internal.md
```

Do not use the old `responsibility_no_max_hit` row as a paper-facing DC
internal ablation. The current rerun script did not make it a reliable
independent row from `responsibility_full`; it can only be revisited after the
override is corrected.

Expected paper conclusion: responsibility DC is a mechanism-oriented
density-control strategy. It should be reported as performance-neutral to weakly
positive under the fair loss while providing a more structured
densification/pruning signal than gradient-only DC. Do not claim it as a large
PSNR booster.

### T03 Historical Renderer Component Source

| Dataset / sweep | Local-only PSNR | Local-global PSNR | Delta PSNR | Delta L1 | Delta SSIM | Delta LPIPS | Delta DISTS | Delta GMSD | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| spine_phantom / left1 | 30.9277 | 31.0293 | +0.1016 | -0.00006 | +0.0013 | n/a | n/a | -0.0021 | pass |
| synthetic_liver / l1 | 29.8515 | 29.7474 | -0.1042 | +0.00018 | -0.0046 | n/a | n/a | -0.0020 | tradeoff |
| tdsc_abus_2023 / validation_100 | 27.5581 | 27.7980 | +0.2399 | -0.00086 | +0.0068 | n/a | n/a | -0.0067 | pass |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | 32.5206 | 32.7660 | +0.2454 | -0.00033 | +0.0022 | n/a | n/a | -0.0028 | pass |

Detailed table:

```text
outputs/production/ablations/ours/tables/t03_renderer_component.md
```

### T04 Easy-Pose Internal Ablation

T04 is the paper-facing Easy-Pose internal ablation table. It is a follow-up to
T01: after T01 shows that Easy-Pose can replace file poses in the full Ours
stack, T04 progressively removes pose information from the pose source to show
which part of the pose signal matters.

Motivation and paper role:

```text
T04 tests the core Easy-Pose claim: Ours should not require full per-frame 6DoF
poses for regular or near-regular sweeps. The table gradually weakens the pose
source, from full file pose to file-derived spacing diagnostics and finally to
uniform calibrated canonical spacing.
```

What this experiment should prove:

```text
1. Full file pose is not the only way to obtain good reconstructions.
2. File-derived step/axis variants diagnose which one-dimensional pose cues
   matter, but they are not the final pose-free setting.
3. Uniform calibrated Easy-Pose is the final claim: it keeps or improves quality
   on the validated regular real sweep while removing per-frame file-pose
   dependence.
```

Writing guidance:

```text
Write T04 as a progressive information-removal study. First state that file pose
is the strongest-information reference. Then show that simplified canonical
variants remain close, and emphasize the uniform row as the final Easy-Pose
setting. If synthetic PSNR is nearly tied but LPIPS is slightly worse, present
that as acceptable retention; if TUS-QC improves, use it as direct evidence that
canonical sweep regularization can help real tracked sequences.
```

Use the standard ablation scope:

```text
Synthetic row = synthetic_liver / l1
Real row = tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370
Metrics = PSNR, SSIM, LPIPS
Fixed renderer = local + integral attenuation, q=3
Fixed DC = full responsibility DC
Table shape = 4 rows x 8 columns
Columns = index, Pose prior, Syn PSNR, Syn SSIM, Syn LPIPS, Real PSNR, Real SSIM, Real LPIPS
```

Recommended rows:

| index | Pose prior | Synthetic source | Real source | Syn PSNR | Syn SSIM | Syn LPIPS | Real PSNR | Real SSIM | Real LPIPS | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a | full file pose | T06 `file_pose` | T06 `file_pose` | 29.748 | 0.678 | 0.234 | 32.750 | 0.866 | 0.142 | completed |
| b | canonical + file step norm | expanded T04 | expanded T04 | 29.745 | 0.677 | 0.245 | 32.817 | 0.866 | 0.145 | completed |
| c | canonical + file axis projection | expanded T04 | expanded T04 | 29.774 | 0.678 | 0.246 | 32.565 | 0.863 | 0.155 | completed |
| d | canonical + uniform calibrated spacing | T06 `canonical_sweep` | T06 `canonical_sweep` | 29.755 | 0.677 | 0.244 | 32.998 | 0.868 | 0.142 | completed |

Definitions:

```text
full file pose:
  scene.loading.pose_mode=file

canonical + file step norm:
  scene.loading.pose_mode=canonical_sweep
  scene.loading.canonical_sweep_axis=z
  scene.loading.canonical_sweep_spacing_mode=file_step_norm
  scene.loading.canonical_sweep_delta_scale=1.00

canonical + file axis projection:
  scene.loading.pose_mode=canonical_sweep
  scene.loading.canonical_sweep_axis=z
  scene.loading.canonical_sweep_spacing_mode=file_axis_projection
  scene.loading.canonical_sweep_delta_scale=1.00

canonical + uniform calibrated spacing:
  scene.loading.pose_mode=canonical_sweep
  scene.loading.canonical_sweep_axis=z
  scene.loading.canonical_sweep_spacing_mode=uniform
  scene.loading.canonical_sweep_delta_scale=1.00
```

Completed result source:

```text
outputs/production/ablations/ours/expanded_paper_plan_20260702/tables/t04_easy_pose_internal.md
```

Rows b and c are diagnostic rows, not final pose-free methods, because they
still derive one-dimensional spacing information from file poses. Row d is the
final Easy-Pose setting. This table is different from T07: T04 studies pose
information removal, while T07 studies scalar spacing/rho sensitivity.

Expected paper conclusion: Easy-Pose progressively removes 6DoF pose
information and retains reconstruction quality with only a uniform canonical
sweep prior under the validated regular/near-regular sweep scope.

### T04 Historical Responsibility DC Component Source

| Dataset / sweep | Gradient PSNR | Responsibility PSNR | Delta PSNR | Delta L1 | Delta SSIM | Delta LPIPS | Delta DISTS | Delta GMSD | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| spine_phantom / left1 | 30.9879 | 30.9691 | -0.0188 | +0.00003 | -0.0003 | n/a | n/a | n/a | neutral/slight negative |
| synthetic_liver / l1 | 29.7991 | 29.7998 | +0.0007 | -0.00001 | -0.0000 | n/a | n/a | n/a | weak positive |
| tdsc_abus_2023 / validation_100 | 27.6824 | 27.6737 | -0.0087 | +0.00001 | -0.0000 | n/a | n/a | n/a | neutral/slight negative |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | 32.6419 | 32.6447 | +0.0028 | -0.00001 | +0.0001 | n/a | n/a | n/a | weak positive |

Decision: responsibility DC is retained as performance-neutral with weak
positive evidence, not as a large final-quality booster. It is acceptable
because it does not materially damage final metrics, is slightly positive on
synthetic_liver and TUS-QC, and produces a different Gaussian allocation signal
from pure gradient DC.

Detailed table:

```text
outputs/production/ablations/ours/tables/t04_responsibility_dc_component.md
```

### T05 Responsibility DC Sub-Components

Motivation and paper role:

```text
T05 is a supplemental diagnostic for DC. It predates the final paper-facing T03
layout but remains useful for explaining why the full responsibility DC policy
was retained: individual DC sub-pieces are near-tied, and the protected full
variant is stable on TUS-QC.
```

Writing guidance:

```text
Use T05 only as supporting evidence, not as the main ablation table. In text,
cite it to say that turning off individual responsibility-DC protections does
not reveal a hidden dominant subcomponent; the full policy is selected because
it is stable and interpretable, not because it cherry-picks the highest PSNR.
```

| Dataset / sweep | Variant | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final Gaussians | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| spine_phantom / left1 | best: `responsibility_no_attenuation_protect` | 31.0504 | 0.01363 | 0.7881 | 0.1746 | 0.2127 | 0.1143 | see table | sub-components are near-tied |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | best: `responsibility_full` for PSNR/L1/SSIM/GMSD | 32.7613 | 0.01286 | 0.8661 | 0.1414 | 0.1589 | 0.0560 | see table | full DC retained |

Detailed table:

```text
outputs/production/ablations/ours/tables/t05_dc_subcomponents.md
```

### T06 Easy-Pose Pose Dependency

Motivation and paper role:

```text
T06 is the broad positive-scope validation for Easy-Pose. Unlike T04, which is a
compact paper table on the standard synthetic/real pair, T06 checks many
regular or near-regular sweeps and asks whether canonical Easy-Pose broadly
retains file-pose reconstruction quality.
```

Writing guidance:

```text
Use T06 to support generality within the stated Easy-Pose scope. The key text is
not that Easy-Pose wins every frame, but that the file-vs-canonical gap remains
small across all positive sweeps. This lets the paper claim reduced pose
dependency under regular acquisition without implying universal robustness to
arbitrary freehand or curved sweeps.
```

| Dataset | Sweep | File PSNR | Canonical PSNR | PSNR gap | LPIPS gap | DISTS gap | GMSD gap | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 positive sweeps | synthetic_liver, TUS-QC, TDSC | see table | see table | min -0.1050, mean +0.0997 | see table | see table | see table | all strong retention |

Detailed table:

```text
outputs/production/ablations/ours/tables/t06_easy_pose_pairs.md
```

### T07 Easy-Pose Spacing Tolerance

Motivation and paper role:

```text
T07 is the numeric sensitivity study for Easy-Pose spacing. It asks whether the
canonical sweep spacing scale rho is a fragile hyperparameter and whether the
default rho=1.00 is defensible without per-dataset tuning.
```

Writing guidance:

```text
Use T07 together with Figure T12. The right interpretation is that rho has a
visible but modest effect: synthetic may prefer smaller spacing for PSNR, while
TUS-QC is stable around rho=1.00 and only gains slightly from tuning. Therefore
rho=1.00 is a reasonable production default, and spacing tuning is not the main
explanation for comparison gaps.
```

| Dataset | Sweep | Scale | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic_liver | l1 | best PSNR rho 0.60 | 30.0221 | see table | see table | best LPIPS rho 1.50 | see table | best GMSD rho 1.25 | dataset-tunable |
| synthetic_liver | r2 | best PSNR rho 0.60 | 29.1202 | see table | see table | best LPIPS rho 1.50 | see table | best GMSD rho 1.25 | dataset-tunable |
| tdsc_abus_2023 | validation_100 | best PSNR rho 1.25 | 27.8725 | see table | see table | best LPIPS rho 0.90 | see table | best GMSD rho 1.25 | rho 1.00 retained |
| tdsc_abus_2023 | validation_101 | best PSNR rho 1.00 | 30.1352 | see table | see table | best LPIPS rho 0.90 | see table | best GMSD rho 1.10 | rho 1.00 retained |
| tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | best PSNR rho 1.25 | 33.0684 | see table | see table | best LPIPS rho 1.25 | see table | best GMSD rho 1.50 | rho 1.00 retained |
| tus_rec_2024_qc | 011_LH_Per_S_PtD_qc_0008_0447 | best PSNR rho 1.50 | 35.5303 | see table | see table | best LPIPS rho 1.10 | see table | best GMSD rho 1.50 | rho 1.00 retained |

Decision: spacing/corridor tuning has real but modest signal. It can improve
selected TUS/TDSC rows by about 0.07-0.11 dB and synthetic PSNR by about
0.28-0.29 dB at rho 0.60, but these gains are not large enough to explain the
full TDSC comparison gap by themselves.

Detailed table:

```text
outputs/production/ablations/ours/tables/t07_spacing_tolerance.md
```

### T08 Easy-Pose Negative Control

Motivation and paper role:

```text
T08 defines the boundary of Easy-Pose. It intentionally applies Easy-Pose where
the sweep geometry violates the regular/near-regular assumption, so that the
method scope is falsifiable rather than overstated.
```

Writing guidance:

```text
Use T08 to write a limitation-aware sentence: Easy-Pose is designed for regular
or near-regular sweeps, and the negative control confirms that the canonical
prior should not be used as a universal pose replacement. This strengthens the
paper by making the method assumption explicit.
```

| Dataset / sweep | Pose mode | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| spine_phantom / left1 | file vs canonical | see table | see table | see table | see table | see table | see table | negative control confirmed |
| spine_phantom / left1_arc100 | file vs canonical | see table | see table | see table | see table | see table | see table | negative control confirmed |

Detailed table:

```text
outputs/production/ablations/ours/tables/t08_negative_control.md
```

### T09 Seed Stability

Motivation and paper role:

```text
T09 checks that the main gains are larger than normal training randomness. It
compares base/full variants over multiple seeds on representative real rows.
```

Writing guidance:

```text
Use T09 when discussing reliability. The writing pattern is: the observed
base-to-full gains exceed pooled seed standard deviations by a large margin, so
the main ablation conclusions are not artifacts of one lucky optimization run.
This is supporting evidence; do not make T09 a central table unless reviewers
ask about variance.
```

| Dataset / sweep | Variant | Seed | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tdsc_abus_2023 / validation_100 | base vs full | 0,1,2 | base mean 27.5565, full mean 27.7971 | see table | see table | see table | see table | see table | see table |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | base vs full | 0,1,2 | base mean 32.5252, full mean 32.7555 | see table | see table | see table | see table | see table | see table |

Seed-stability pass/fail:

```text
TDSC gain = +0.2406 dB, pooled std = 0.0163
TUS-QC gain = +0.2303 dB, pooled std = 0.0062
```

Detailed tables:

```text
outputs/production/ablations/ours/tables/t09_seed_stability.md
outputs/production/ablations/ours/tables/t09_seed_gain_vs_std.md
```

### T10 Efficiency and Mechanism

Motivation and paper role:

```text
T10 records runtime, Gaussian-count, scalar-log, and mechanism artifacts. It is
not a primary accuracy ablation, but it supports the engineering claim that the
final method remains practical and that mechanism-level behavior can be audited
without retaining checkpoints.
```

Writing guidance:

```text
Use T10 only in supplement or implementation discussion. It can justify that the
method's extra modules do not make the system operationally opaque: logs,
counts, and qualitative outputs are retained, while checkpoint payloads are not
kept under the storage policy.
```

| Artifact | Source | Status | Notes |
| --- | --- | --- | --- |
| Runtime / Gaussian-count table | New Stage/T runs | generated | `outputs/production/ablations/ours/tables/t10_runtime_gaussians.md` |
| Qualitative panels | New final outputs | available | Retained under selected `runs/**/vis/`; no checkpoints retained. |
| Error maps / residual maps | New final outputs | available | Retained under selected `runs/**/vis/` and `eval/`. |
| Responsibility/DC traces | New scalar logs | available | Use `logs/scalars.tsv` and config snapshots instead of checkpoints. |

### T11 Ablation Visual Figures

T11 is the paper-facing qualitative visualization package for the standard
Ours ablations. It should use the same visual language as the frozen comparison
2D rendering figures:

Motivation and paper role:

```text
T11 turns the T01-T04 numeric conclusions into visual evidence. The tables show
metric changes, but the paper also needs to show that the proposed modules
produce visibly cleaner ultrasound renderings in the same regions where PSNR
and SSIM improve.
```

Writing guidance:

```text
Use T11 immediately after the objective ablation tables. For T11-A, describe
the additive path: the +LG and full rows should reduce visible artifacts and
better match GT in the zoomed ROI, especially on TUS-QC. For T11-B, use each row
group as visual support for the corresponding internal table: renderer variants
should show the clearest visual improvement, DC variants should look similar
with no collapse, and Easy-Pose variants should retain anatomical structures
despite reduced pose information. Do not claim visual evidence alone proves the
method; use it to make the metric trends interpretable.
```

```text
Reference config = docs/production/comparison_final_outputs/2d_rendering_figure_config.md
Style = grayscale prediction, yellow ROI box, yellow zoom inset, connector lines
Metric overlay = PSNR and SSIM for non-GT panels only
GT panels = no metric overlay
Crop rule = same crop within each row across all variants
Input images = outputs/production/ablations/ours/runs/**/vis/item_*/pred.png and gt.png
Per-frame metrics = outputs/production/ablations/ours/runs/**/eval/per_frame_metrics_latest.tsv
```

Generated output root:

```text
docs/production/ablation_final_outputs/
```

Generated deliverables:

| Figure | Output | Purpose | Status |
| --- | --- | --- | --- |
| T11-A | `figures/paper_ablation_joint_visual.pdf` / `.png` | Qualitative support for T01 three-module joint ablation. | generated |
| T11-B | `figures/paper_ablation_internal_visual.pdf` / `.png` | Qualitative support for T02/T03/T04 internal ablations. | generated |
| Config | `2d_ablation_figure_config.md` | Frozen reproducibility config after generation. | generated |
| Manifest | `manifests/*.json` | Selected items, crops, and source run paths. | generated |

#### T11-A: Joint Module Ablation Visual

Purpose: visually support T01 by showing the main module-addition path. T01
contains the full 8-row module factorial, but the figure should show the
paper-facing progression:

```text
base -> +LG -> +LG+DC -> +LG+DC+EP -> GT
```

Layout:

```text
rows = 4
cols = 5
row 1 = synthetic_liver / l1 prediction
row 2 = synthetic_liver / l1 error or residual map
row 3 = tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 prediction
row 4 = tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 error or residual map
```

Columns:

| Col | Label | Source |
| --- | --- | --- |
| 1 | Base | T01 index a: LG off, DC off, EP off |
| 2 | +LG | T01 index e: LG on, DC off, EP off |
| 3 | +LG+DC | T01 index g: LG on, DC on, EP off |
| 4 | +LG+DC+EP | T01 index h: LG on, DC on, EP on |
| 5 | GT | Ground truth |

Current source status:

| Source | Status |
| --- | --- |
| T01 a synthetic + real | completed |
| T01 e synthetic + real | completed |
| T01 g synthetic + real | completed |
| T01 h synthetic + real | completed |

T11-A has been generated after T01 completion:

```text
docs/production/ablation_final_outputs/figures/paper_ablation_joint_visual.pdf
docs/production/ablation_final_outputs/figures/paper_ablation_joint_visual.png
```

Expected writing from T11-A:

```text
The visual progression should be discussed as qualitative confirmation of T01.
On the real TUS-QC row, adding LG and then the full stack should preserve the
bright superficial layer and reduce texture/noise mismatch in the zoomed ROI.
The synthetic row may show smaller PSNR movement; use it to show that the full
stack does not introduce obvious structural artifacts even when synthetic PSNR
is a tradeoff.
```

#### T11-B: Internal Module Ablation Visual

Purpose: visually support T02, T03, and T04 by showing each module's internal
progression on both standard ablation datasets.

Layout:

```text
rows = 6
cols = 5
row 1 = T02 renderer internal, synthetic_liver / l1
row 2 = T02 renderer internal, tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370
row 3 = T03 DC internal, synthetic_liver / l1
row 4 = T03 DC internal, tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370
row 5 = T04 Easy-Pose internal, synthetic_liver / l1
row 6 = T04 Easy-Pose internal, tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370
```

Columns:

| Row group | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
| --- | --- | --- | --- | --- | --- |
| T02 renderer | Local | Simple | Integral-1 | Integral-3 | GT |
| T03 DC | Grad | Resp-D | Resp-P | Full DC | GT |
| T04 Easy-Pose | File | Step | Axis | Uniform | GT |

Source status:

| Row group | Source coverage | Status |
| --- | --- | --- |
| T02 renderer | Local, Simple, Integral-1, Integral-3 on synthetic + real | completed |
| T03 DC | Grad, Resp-D, Resp-P, Full DC on synthetic + real | completed |
| T04 Easy-Pose | File, Step, Axis, Uniform on synthetic + real | completed |

T11-B has been generated after T02-T04 completion:

```text
docs/production/ablation_final_outputs/figures/paper_ablation_internal_visual.pdf
docs/production/ablation_final_outputs/figures/paper_ablation_internal_visual.png
```

Expected writing from T11-B:

```text
Use one sentence per module group. Renderer: the integral renderer is visually
closest to GT on real data and matches the largest metric gain. DC: all DC rows
remain visually close, supporting the claim that responsibility DC is stable
rather than a visual-quality shortcut. Easy-Pose: the uniform canonical row
keeps the same main anatomical structures as file pose, supporting pose
dependency reduction.
```

#### T11 Generation Notes

The visualization script should be separate from the multi-method comparison
script:

```text
scripts/production/ablations/ours/make_paper_ablation_figures.py
```

The script may reuse the crop/inset/metric overlay logic from:

```text
scripts/production/comparisons/us_recon_four_dataset_p100k_v1/make_paper_2d_rendering_figures.py
```

Recommended figure sizes:

```text
T11-A: figsize = (10.5, 7.2), dpi = 300
T11-B: figsize = (10.5, 10.8), dpi = 300
```

### Objective Metric Figure Style Plan

Use different objective-metric figure types for different experimental
questions:

| Figure | Experiment | Chosen plot type | Reason |
| --- | --- | --- | --- |
| T12 | Easy-Pose rho sensitivity | Response curve / line plot | Rho is an ordered scalar spacing factor, so continuity and trend direction matter. This should be the only standard ablation figure that uses a conventional line curve. |
| T14 | Renderer parameter sensitivity | Tornado / range-dot sensitivity plot | Renderer sweeps are one-at-a-time across several unrelated parameters; range-dot rows show which parameter matters most without five repetitive mini line charts. |
| T15 | Responsibility DC parameter sensitivity | Heatmaps + compact dot panel | DC thresholds interact in pairs, so two-dimensional heatmaps communicate stable regions better than independent curves. |

### T12 Easy-Pose Rho Sensitivity Curve

T12 is the paper-facing numeric sensitivity figure for Easy-Pose spacing. It
uses the already completed T07 rho sweep and should be plotted as a line chart,
not as an image panel figure.

Motivation and paper role:

```text
T12 asks whether Easy-Pose depends on a brittle spacing scale. Since rho is an
ordered scalar, a response curve is the clearest way to show whether the default
rho=1.00 lies in a stable region or whether the method only works after hidden
per-dataset tuning.
```

Purpose:

```text
Show how Easy-Pose quality changes as the canonical sweep spacing scale rho
varies. This supports the claim that the final rho=1.00 setting is stable on
the real representative dataset, while also exposing dataset-specific tuning
behavior.
```

Standard figure scope:

| Role | Dataset | Sweep | Status |
| --- | --- | --- | --- |
| Synthetic | `synthetic_liver` | `l1` | completed |
| Real | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | completed |

Available rho values:

```text
0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50
```

Recommended plot:

```text
one canvas
x-axis = rho
left y-axis = PSNR
two PSNR lines = synthetic_liver/l1 and TUS-QC/010_LH...
optional second panel or right y-axis = LPIPS
mark rho=1.00 with a vertical dashed line
mark each dataset's best-PSNR point with a hollow marker
```

Preferred layout:

```text
rows = 1
cols = 2
panel A = PSNR vs rho
panel B = LPIPS vs rho
shared x-axis = rho
same legend = Synthetic, Real
```

Do not include SSIM in the main curve unless space allows. SSIM should remain
in the table because its dynamic range is small and it clutters the sensitivity
plot. If a single-panel figure is required, use PSNR only.

Data points for the standard figure:

| Dataset | rho | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| synthetic_liver/l1 | 0.60 | 30.022 | 0.683 | 0.273 |
| synthetic_liver/l1 | 0.75 | 29.936 | 0.681 | 0.260 |
| synthetic_liver/l1 | 0.90 | 29.826 | 0.679 | 0.250 |
| synthetic_liver/l1 | 1.00 | 29.747 | 0.677 | 0.244 |
| synthetic_liver/l1 | 1.10 | 29.677 | 0.675 | 0.241 |
| synthetic_liver/l1 | 1.25 | 29.554 | 0.672 | 0.234 |
| synthetic_liver/l1 | 1.50 | 29.304 | 0.667 | 0.230 |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | 0.60 | 32.771 | 0.865 | 0.153 |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | 0.75 | 32.884 | 0.867 | 0.147 |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | 0.90 | 32.950 | 0.868 | 0.143 |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | 1.00 | 33.001 | 0.868 | 0.141 |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | 1.10 | 33.037 | 0.869 | 0.141 |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | 1.25 | 33.068 | 0.869 | 0.140 |
| tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370 | 1.50 | 33.058 | 0.869 | 0.141 |

Expected interpretation:

```text
Synthetic_liver/l1 favors a smaller spacing scale for PSNR/SSIM, while TUS-QC
is stable around rho=1.00 and has its best PSNR near rho=1.25. The real
representative dataset gains only about 0.07 dB from tuning away from rho=1.00,
so the default rho=1.00 is defensible for paper-facing production use.
```

Writing guidance:

```text
When writing T12, emphasize stability rather than the absolute best rho. The
curve should be described as showing a broad usable region around the production
default on TUS-QC. The synthetic curve can be described as dataset-dependent
metric preference: smaller rho improves PSNR, while larger rho improves LPIPS.
This supports a conservative default instead of aggressive per-dataset tuning.
```

Generated output:

```text
docs/production/ablation_final_outputs/figures/paper_easy_pose_rho_sensitivity.pdf
docs/production/ablation_final_outputs/figures/paper_easy_pose_rho_sensitivity.png
docs/production/ablation_final_outputs/manifests/paper_easy_pose_rho_sensitivity_manifest.json
```

### T13 Easy-Pose 3D Geometry And Reconstruction Visualization

T13 is an optional discussion/mechanism figure for Easy-Pose. It should not be a
cross-method 3D qualitative comparison. The earlier 3D comparison attempts were
rejected because standard 3DGS/SuperSplat-style display models are not fair
across methods with different primitive semantics:

Current execution decision:

```text
Do not run T13 in the next supplemental ablation launch.
Keep T13 as a documented optional discussion figure only.
No checkpoint-regeneration or 3D export tasks should be scheduled unless the
user explicitly re-enables this figure later.
```

```text
Reference note = docs/production/comparison_final_outputs/3d_visualization_ablation_notes.md
Decision there = do not include multi-method 3D comparison figures
Allowed scope = carefully scoped Ours-only or native-renderer demonstration
```

T13 is allowed because it compares Ours against Ours:

```text
Ours trained with file poses
vs
Ours trained with Easy-Pose canonical sweep
```

Both sides share the same Gaussian parameter semantics, export path, visual
calibration, and viewer bias. The figure therefore avoids the fairness problem
that made cross-method 3D visualization unsuitable.

Preferred dataset:

| Role | Dataset | Sweep | Reason |
| --- | --- | --- | --- |
| Primary | `synthetic_liver` | `l1` | Has true/file pose geometry, so pose-overlay claims can refer to GT/file pose. |
| Optional supplement | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` | Real-sweep reference; describe file pose as tracked reference, not GT. |

Purpose:

```text
Show that the Easy-Pose canonical sweep geometry is close to the file/GT sweep
geometry on a regular sweep, and that the reconstructed Ours Gaussian volume
does not collapse when per-frame 6DoF poses are replaced by the canonical prior.
The intended discussion point is that small residual pose mismatch can be
absorbed by the Gaussian representation under the validated Easy-Pose scope.
```

Recommended panel design:

| Panel | Content | Notes |
| --- | --- | --- |
| A | File-pose Ours 3D reconstruction | Same camera, crop, brightness, and opacity mapping as panel B. |
| B | Easy-Pose Ours 3D reconstruction | Same Ours-only export/display convention as panel A. |
| C | Pose geometry overlay | Draw file/GT scan planes and centerline against Easy-Pose canonical scan planes and centerline. |
| D | Native 2D render comparison | File-pose render, Easy-Pose render, and GT slice for one representative frame. |

Pose overlay design:

```text
file/GT pose planes = white or blue wireframes
Easy-Pose canonical planes = yellow wireframes
draw every Nth scan plane, not every frame
draw file/GT and canonical sweep centerlines
mark first and last frame
use the same coordinate normalization as the 3D reconstruction panels
```

Implementation route:

```text
Reuse the SuperSplat/PLY route or the Ours-specific export route only for Ours
vs Ours. Do not mix methods in this figure.
If SuperSplat is used, explicitly document that both compared rows are Ours and
therefore share the same attribute semantics.
```

Checkpoint policy:

```text
Ours ablation runs still must not retain checkpoint payloads.
For T13, any needed checkpoint should be treated as a transient artifact:
  1. train or rerun the file-pose and Easy-Pose rows;
  2. immediately export the 3D visualization artifacts;
  3. write a manifest with the source config, checkpoint path, and export files;
  4. delete checkpoint payloads after successful export.
```

Required rerun/export tasks:

| Dataset | Variant | Status |
| --- | --- | --- |
| synthetic_liver / l1 | file pose | needs visualization export because checkpoints were deleted |
| synthetic_liver / l1 | Easy-Pose canonical sweep | needs visualization export because checkpoints were deleted |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | file pose | optional supplement |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | Easy-Pose canonical sweep | optional supplement |

Planned output:

```text
docs/production/ablation_final_outputs/figures/paper_easy_pose_3d_geometry.pdf
docs/production/ablation_final_outputs/figures/paper_easy_pose_3d_geometry.png
docs/production/ablation_final_outputs/manifests/paper_easy_pose_3d_geometry_manifest.json
```

Expected paper usage:

```text
Use T13 after the Easy-Pose metric and internal-ablation discussion. The figure
is qualitative mechanism evidence, not a replacement for T01/T04/T12 metrics.
Phrase the claim as pose-requirement reduction with stable reconstruction under
regular sweeps, not as a universal 3D fidelity claim.
```

### T14 Local-Global Renderer Parameter Sensitivity Figure

T14 is the paper-facing numeric sensitivity figure for local-global renderer
parameters. It should not use another set of line charts. Use a tornado /
range-dot sensitivity plot so the figure emphasizes which renderer parameters
matter most and whether the production default lies in a stable range.

Motivation and paper role:

```text
T14 tests whether the local-global renderer benefit is robust to reasonable
hyperparameter changes. Because the swept parameters are not one shared ordered
axis, a tornado/range-dot plot is preferred over many small line charts.
```

Purpose:

```text
Show that the local-global renderer is not supported by a fragile single
hyperparameter point. The main sensitivity axis is global attenuation strength;
secondary panels show that beta, min_transmission, quadrature_points, and
candidate_k_sigma have reasonable stable regions around the production default.
```

Standard figure scope:

| Role | Dataset | Sweep |
| --- | --- | --- |
| Synthetic | `synthetic_liver` | `l1` |
| Real | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` |

Fixed settings for one-at-a-time sweeps:

```text
renderer = local-global enabled
formula = integral unless the parameter being swept is not compatible
DC = full responsibility DC
pose = file pose
loss = 0.8 L1 + 0.2 SSIM
metrics recorded = PSNR, SSIM, LPIPS
main figure x-axis = delta PSNR relative to the production default
SSIM/LPIPS = retained in result tables or manifest, not encoded in the main figure
```

Production defaults:

```text
ray_attenuation_strength = 0.55
ray_attenuation_beta = 0.80
ray_attenuation_min_transmission = 0.35
ray_attenuation_quadrature_points = 3
ray_attenuation_candidate_k_sigma = 2.0
```

Figure layout:

```text
main canvas = one figure
panel A = Synthetic tornado sensitivity
panel B = Real tornado sensitivity
panel C = default-vs-best summary strip / legend
each parameter is one horizontal row
x-axis = delta PSNR relative to production default
points = tested parameter values
thin horizontal segment = full observed range for that parameter
default value = vertical zero line and filled star marker
best value = hollow marker with label
```

Preferred layout:

```text
1 row x 3 columns
panel A and panel B are the two main panels
panel C lists the production default, best value, and max absolute delta for each parameter
sort parameter rows by absolute PSNR range on the Real dataset
use one color per parameter family, not one color per dataset
```

Sweep values:

| Row | Parameter | Values | Default |
| --- | --- | --- | --- |
| 1 | `ray_attenuation_strength` | `0.00, 0.20, 0.35, 0.55, 0.75, 1.00` | `0.55` |
| 2 | `ray_attenuation_beta` | `0.00, 0.40, 0.60, 0.80, 1.00` | `0.80` |
| 3 | `ray_attenuation_min_transmission` | `0.20, 0.35, 0.50, 0.65, 0.80` | `0.35` |
| 4 | `ray_attenuation_quadrature_points` | `1, 2, 3` | `3` |
| 5 | `ray_attenuation_candidate_k_sigma` | `-1.0, 1.5, 2.0, 3.0` | `2.0` |

Experimental size:

```text
one-at-a-time unique configs = 19
datasets per config = 2
total training runs = 38
```

Counting rule:

```text
The shared production default config appears in all panels but should be run
once and reused across panels. The 19 unique configs count removes repeated
default rows across beta, min_transmission, quadrature_points, and
candidate_k_sigma sweeps.
```

Expected interpretation:

```text
Strength should have the widest range and therefore appear as the dominant
local/global composition parameter. The production value 0.55 should remain
close to the best Real setting and within a stable plateau. The shorter ranges
for beta, min_transmission, quadrature_points, and candidate_k_sigma should
support that these defaults are defensible and not isolated failure-prone
choices.
```

Writing guidance:

```text
When writing T14, sort the discussion by observed real-data range. If q=3 and
beta/default settings sit at or near zero delta while alternative values degrade
PSNR, state that the production renderer is not a fragile point estimate. If a
nearby value such as strength=0.75 or candidate_k_sigma=3.0 is marginally better
on one row, describe it as a flat plateau rather than a reason to retune the
paper configuration. The key claim is robustness and mechanism ranking, not
that every default is mathematically optimal.
```

Generated output:

```text
docs/production/ablation_final_outputs/figures/paper_renderer_parameter_sensitivity.pdf
docs/production/ablation_final_outputs/figures/paper_renderer_parameter_sensitivity.png
docs/production/ablation_final_outputs/manifests/paper_renderer_parameter_sensitivity_manifest.json
```

### T15 Responsibility DC Parameter Sensitivity Figure

T15 is the paper-facing numeric sensitivity figure for responsibility density
control. It should not be another line-chart sweep. Use heatmaps and a compact
dot panel so the figure reads as a mechanism/threshold analysis rather than a
collection of one-dimensional tuning curves.

Motivation and paper role:

```text
T15 tests whether responsibility DC depends on a brittle threshold combination.
Because DC uses interacting selectivity, budget, and pruning-protection
thresholds, heatmaps are more faithful than independent one-dimensional curves.
```

Purpose:

```text
Show that responsibility DC is not dependent on one brittle threshold setting.
The main sensitivity axis is responsibility densification selectivity; secondary
panels cover densification budget, low-contribution pruning, attenuation-aware
prune protection, and opacity pruning. The figure supports the claim that DC is
a stable topology-control mechanism rather than a hidden loss or renderer tweak.
```

Standard figure scope:

| Role | Dataset | Sweep |
| --- | --- | --- |
| Synthetic | `synthetic_liver` | `l1` |
| Real | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370` |

Fixed settings:

```text
renderer = full local-global renderer
pose = file pose
DC = responsibility DC enabled
scale activation = bounded_sigmoid
loss = 0.8 L1 + 0.2 SSIM
metrics recorded = PSNR, SSIM, LPIPS
main figure color/value = delta PSNR relative to the production default
SSIM/LPIPS = retained in result tables or manifest, optionally shown as tiny text in the cell tooltip/source table but not encoded in the main figure
```

Production defaults:

```text
density_control.strategy = contribution_residual
density_control.timing.start_iter = 1000
density_control.timing.end_iter = 25000
density_control.timing.freq = 2000
density_control.thresholds.densify_grad = 6e-6
density_control.thresholds.split_scale = 0.6
density_control.thresholds.prune_opacity = 0.0001
density_control.thresholds.prune_intensity = 0.01
density_control.responsibility.score_quantile = 0.98
density_control.responsibility.max_fraction = 0.05
density_control.responsibility.low_contribution_quantile = 0.10
density_control.responsibility.protect_attenuation_enabled = true
density_control.responsibility.protect_attenuation_quantile = 0.80
```

Figure layout:

```text
main canvas = one figure
panel A = densification policy heatmap, Synthetic
panel B = densification policy heatmap, Real
panel C = opacity pruning dot panel, Synthetic + Real
panel D = pruning protection heatmap, Synthetic
panel E = pruning protection heatmap, Real
panel F = shared colorbar / legend / Gaussian-count note
mark the production default cell with a bold outline and star marker
```

Preferred layout:

```text
2 rows x 3 columns
top row = A, B, C
bottom row = D, E, F
use one shared diverging color scale for all heatmaps: delta PSNR vs production default
use white/black cell text only for the production cell and the best cell to avoid clutter
```

Sweep values:

| Panel | Parameter(s) | Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| A/B | `responsibility.score_quantile` x `responsibility.max_fraction` | score: `0.94, 0.96, 0.98, 0.99`; fraction: `0.02, 0.05, 0.08, 0.12` | `0.98` x `0.05` | Jointly shows selectivity and per-event densification budget. |
| C | `thresholds.prune_opacity` | `0.00001, 0.00005, 0.0001, 0.0005, 0.001` | `0.0001` | Dot panel on log x-axis; marker shape denotes dataset and marker size can encode final Gaussian count or prune count. |
| D/E | `responsibility.low_contribution_quantile` x `responsibility.protect_attenuation_quantile` | low contribution: `0.02, 0.05, 0.10, 0.15, 0.20`; protect attenuation: `0.60, 0.70, 0.80, 0.90` | `0.10` x `0.80` | Jointly shows pruning aggressiveness and attenuation-aware protection. |

Experimental size:

```text
densification heatmap configs = 4 x 4 = 16
pruning protection heatmap configs = 5 x 4 = 20
opacity dot-panel configs = 5
unique configs after sharing the production default = 39
datasets per config = 2
total training runs = 78
```

Counting rule:

```text
The shared production default config appears in the densification heatmap, the
pruning-protection heatmap, and the opacity dot panel, but should be run once
and reused across all panels. The 39 unique configs count removes the repeated
default entries across the three panel groups.
```

Parameters deliberately not included in the main paper figure:

| Parameter | Reason |
| --- | --- |
| `density_control.timing.start_iter/end_iter/freq` | Schedule choices are engineering/runtime policy; they can be logged in supplement if needed but would distract from DC mechanism thresholds. |
| `thresholds.densify_grad` | Used mainly by gradient fallback and non-responsibility DC; production responsibility selection is driven by `score_quantile` and `max_fraction`. |
| `thresholds.split_scale` | Clone/split morphology threshold; useful for debugging Gaussian count dynamics but less directly tied to the proposed responsibility score. |
| `thresholds.prune_intensity` | Current production comparisons mainly calibrate opacity and responsibility contribution/protection; intensity pruning is less interpretable for paper-facing threshold sensitivity. |
| `responsibility.min_score` | Safety floor only; normally dominated by quantile and top-fraction thresholding. |
| `responsibility.mix_mode`, `union_gradient_fraction`, `union_gradient_min_factor` | Alternative hybrid strategy. Keep fixed to the production responsibility-only mode for fairness and clarity. |

Expected interpretation:

```text
The score_quantile x max_fraction heatmaps should show a stable densification
plateau around the production default rather than a single isolated lucky point.
The low_contribution_quantile x protect_attenuation_quantile heatmaps should
show whether attenuation-aware protection prevents harmful pruning while keeping
quality close to the best region. The opacity dot panel provides a compact
sanity check that raw weak-opacity pruning is not the dominant hidden factor.
```

Writing guidance:

```text
When writing T15, keep the conclusion aligned with the DC claim. The heatmaps
should be read as threshold-stability evidence: most settings remain within a
small delta-PSNR band, and the production default is close to the stable region.
If the best cell is not exactly the default, say that the margin is small and
does not justify per-dataset threshold tuning. The point is that responsibility
DC is not a hidden metric-tuning knob.
```

Generated output:

```text
docs/production/ablation_final_outputs/figures/paper_dc_parameter_sensitivity.pdf
docs/production/ablation_final_outputs/figures/paper_dc_parameter_sensitivity.png
docs/production/ablation_final_outputs/manifests/paper_dc_parameter_sensitivity_manifest.json
```

## 10. Expanded Paper Plan Accounting

This section records the expanded Ours ablation launch that completed the
paper-facing tables, parameter-sensitivity figures, and multi-sweep validation.
It supersedes the old "116/116 completed" wording from the historical rerun.

### 10.1 Main Table Completion Runs

These runs completed the paper-facing module and module-internal tables.

| Group | Completed rows | Dataset multiplier | New training tasks | Notes |
| --- | ---: | ---: | ---: | --- |
| T01 three-module joint table | 3 combinations: b/d/f | 2 datasets | 6 | Required for the full 8-row LG/DC/EP factorial. |
| T02 renderer internal table | 2 modes: simple, integral q=1 | 2 datasets | 4 | Required for renderer-internal paper table and T11-B. |
| T03 DC internal table | 2 synthetic rows: Resp-D, Resp-P | 1 dataset | 2 | Real rows already exist from T05. |
| T04 Easy-Pose internal table | 2 pose-prior modes: step, axis | 2 datasets | 4 | Required for Easy-Pose internal table and T11-B. |
| **Main-table subtotal** |  |  | **16** | Completed in expanded paper plan job. |

T11 visual figures:

| Figure | Extra training needed | Dependency |
| --- | ---: | --- |
| T11-A joint visual | 0 | Generated from completed T01 additive-path rows. |
| T11-B internal visual | 0 beyond main-table runs | Generated from completed T02/T03/T04 source rows. |

### 10.2 Parameter Sensitivity Runs

T12 was reused from the completed rho sweep. T14 and T15 were completed in the
expanded paper plan and then rendered into final objective-metric figures.

| Figure | Design | Unique configs | Dataset multiplier | New training tasks when reusing existing default | Self-contained training tasks |
| --- | --- | ---: | ---: | ---: | ---: |
| T12 Easy-Pose rho | response curve | already completed | 2 standard datasets extracted from T07 | 0 | 0 |
| T14 renderer sensitivity | tornado / range-dot | 19 | 2 | 36 | 38 |
| T15 DC sensitivity | heatmaps + dot panel | 39 | 2 | 76 | 78 |
| **Parameter subtotal** |  |  |  | **112** | **116** |

Counting rule:

```text
T14 and T15 both include the same production-default full Ours file-pose row.
That row already exists for synthetic_liver/l1 and TUS-QC/010_LH... from the
completed GS-strict rerun, so the recommended launch reuses it and runs only
non-default sensitivity configs.

If a fully self-contained sensitivity output package is required, regenerate
the shared default row once, not once per figure.
```

### 10.3 Optional Visualization Export Runs

T13 requires 3D export artifacts, not just metrics. Because checkpoints were
deleted by policy, these rows need transient rerun/export if T13 is included in
the paper.

Current status: **deferred / do not run now**.

| Figure | Required rows | New transient train/export tasks | Notes |
| --- | --- | ---: | --- |
| T13 Easy-Pose 3D geometry | synthetic_liver/l1 file pose + Easy-Pose | 2 | Required only if the 3D discussion figure is used. Delete checkpoints after export. |
| T13 optional real supplement | TUS-QC file pose + Easy-Pose | 2 optional | Use tracked-reference wording, not GT-pose wording. |

### 10.4 Recommended Launch Totals

| Scope | New training tasks |
| --- | ---: |
| Main paper tables and T11 dependencies | 16 |
| Parameter sensitivity figures using existing default row | 112 |
| **Recommended quantitative total** | **128** |
| Add shared default regeneration for a self-contained sensitivity package | +2 |
| **Self-contained quantitative total** | **130** |
| T13 3D export runs | 0 now; deferred |

Completed execution order:

1. Ran the 16 main-table completion tasks.
2. Generated/updated T01-T04 tables and T11-A/T11-B visuals.
3. Ran T14 renderer sensitivity non-default configs.
4. Ran T15 DC sensitivity non-default configs.
5. Generated T12/T14/T15 objective-metric figures.
6. Kept T13 deferred; run/export it only if the paper later needs the 3D
   Easy-Pose discussion figure.

### 10.5 Multi-Sweep Mean Validation

Status: **completed**.

Execution:

```text
Slurm job = 66969
Node = gpu3
GPU request = 3 RTX 4090 GPUs
Internal parallelism = 6 workers, 2 workers per GPU
New training runs = 60 / 60 completed
Worker failures = 0
Checkpoint payload files retained = 0
Output root = outputs/production/ablations/ours/multisweep_mean_validation_20260702
Summary = outputs/production/ablations/ours/multisweep_mean_validation_20260702/summary.tsv
Analysis note = outputs/production/ablations/ours/multisweep_mean_validation_20260702/notes/analysis.md
```

Purpose:

```text
The detailed paper-facing ablation tables remain on the fixed representative
synthetic + real sweeps for readability. To avoid the impression that the main
module conclusions depend on one cherry-picked sweep, run a small multi-sweep
validation for the performance-bearing ablations and describe the mean trend in
text. Do not create a new large paper table unless later requested.
```

Motivation and paper role:

```text
The main tables are intentionally compact, but compact tables can look
under-supported if they use only one sweep per dataset. The multi-sweep
validation checks whether the core trends persist across five synthetic and
five TUS-QC sweeps, plus a small TDSC stress check, without expanding the paper
with another large table.
```

Writing guidance:

```text
Use this section in prose after the main ablation tables. The intended sentence
is that the representative-sweep conclusions were also checked by averaging key
ablation paths over multiple sweeps. Report only the mean trend: LG remains the
dominant real-data contributor, Easy-Pose is stable and beneficial on TUS-QC,
DC remains near-neutral, synthetic_liver shows a PSNR/perceptual tradeoff, and
TDSC supports the renderer benefit but not a universal full-stack monotonic
gain. This protects the paper from overclaiming while demonstrating that the
main conclusions are not cherry-picked.
```

Selected sweeps:

| Role | Dataset | Sweeps | Reason |
| --- | --- | --- | --- |
| Synthetic | `synthetic_liver` | `l1`, `l2`, `l3`, `r1`, `r2` | Five regular synthetic sweeps covering both liver sides while staying inside the already validated Easy-Pose positive scope. |
| Real | `tus_rec_2024_qc` | `010_LH_Per_L_DtP_qc_0005_0370`, `010_RH_Per_L_DtP_qc_0005_0574`, `011_LH_Per_S_PtD_qc_0008_0447`, `010_LH_Per_C_DtP_qc_0140_0514`, `011_RH_Per_L_DtP_qc_0007_0584` | Five QC real sweeps with different subject/side/direction patterns; the first is the standard representative sweep already used in T01-T04. |
| TDSC stress check | `tdsc_abus_2023` | `validation_100`, `validation_101`, `validation_102` | Small independent real-dataset check for the key module path, because TDSC was one of the harder comparison settings for Ours. |

Report style:

```text
No standalone table by default.
In the ablation text, state that the key trends were repeated over 5 synthetic
and 5 TUS-QC real sweeps, with an additional 3-sweep TDSC stress check for the
key module path. Summarize the mean direction and approximate mean/std for
PSNR/SSIM/LPIPS where useful.
Detailed per-sweep values should stay in outputs/ tables or supplement.
```

Experiments to include:

| Group | Variants | Why multi-sweep mean is useful |
| --- | --- | --- |
| T01 key module path | Base, +LG, +LG+DC, +LG+DC+EP | Validates the main module story beyond one representative sweep. Use only the main additive path, not all 8 factorial combinations. Run it on 5 synthetic + 5 TUS-QC sweeps, plus 3 TDSC stress-check sweeps. |
| T02 renderer internal | local-only, simple global attenuation, integral q=1, integral q=3 | Renderer is the strongest performance-bearing module, so its internal ranking should be averaged across several sweeps. |
| T04 Easy-Pose internal | file pose, canonical + file step norm, canonical + file axis projection, canonical + uniform calibrated spacing | Easy-Pose is a key method claim and should show stable behavior across several regular/near-regular sweeps. |

Experiments not included:

| Excluded group | Reason |
| --- | --- |
| T03 DC internal | DC is claimed as a mechanism-oriented, performance-neutral component. Existing single-sweep internal table plus T05/T09 supplement are enough unless reviewers specifically question DC robustness. |
| T14 renderer parameter sensitivity | Fine-grained parameter sensitivity is a mechanism/stability figure, not a broad performance claim. Multi-sweep expansion would be expensive and hard to present cleanly. |
| T15 DC parameter sensitivity | DC thresholds already show very small ranges on representative sweeps; multi-sweep expansion has low information gain. |
| T12 rho sensitivity | Already has multiple sweeps from T07; no extra multi-sweep validation is needed. |

Reuse-aware task count:

| Group | Total rows | Existing reusable rows | New training tasks |
| --- | ---: | ---: | ---: |
| T01 key path, synthetic + TUS-QC | 4 variants x 10 sweeps = 40 | 24 | 16 |
| T02 renderer internal, synthetic + TUS-QC | 4 variants x 10 sweeps = 40 | 16 | 24 |
| T04 Easy-Pose internal, synthetic + TUS-QC | 4 variants x 10 sweeps = 40 | 24 | 16 |
| TDSC key path stress check | 4 variants x 3 sweeps = 12 | 8 | 4 |
| **Total** | **132** | **72** | **60** |

Reuse details:

```text
T01:
  Base and +LG exist only for synthetic_liver/l1 and the standard TUS sweep.
  Need Base/+LG for synthetic_liver/l2, synthetic_liver/l3,
  synthetic_liver/r1, synthetic_liver/r2,
  TUS 010_RH_Per_L_DtP_qc_0005_0574,
  TUS 011_LH_Per_S_PtD_qc_0008_0447,
  TUS 010_LH_Per_C_DtP_qc_0140_0514, and
  TUS 011_RH_Per_L_DtP_qc_0007_0584.
  +LG+DC and +LG+DC+EP can reuse existing T06 file/canonical rows for all
  selected positive-scope sweeps.

T02:
  local-only, simple, and integral q=1 need the eight non-standard selected
  sweeps listed above.
  integral q=3 can reuse existing T06 file-pose full Ours rows for all selected
  positive-scope sweeps.

T04:
  file and uniform canonical can reuse existing T06 file/canonical rows for all
  selected positive-scope sweeps.
  step-norm and axis-projection need the eight non-standard selected sweeps.

TDSC stress check:
  +LG+DC and +LG+DC+EP can reuse existing T06 file/canonical rows for
  validation_100, validation_101, and validation_102.
  Base and +LG exist for validation_100 from the historical T02 factorial.
  Need Base/+LG for validation_101 and validation_102 only.
```

Generated result files:

```text
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/multisweep_per_sweep_metrics.tsv
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/multisweep_mean_std.tsv
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/t01_key_path_synthetic_mean_std.md
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/t01_key_path_tus_qc_mean_std.md
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/t02_renderer_internal_synthetic_mean_std.md
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/t02_renderer_internal_tus_qc_mean_std.md
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/t04_easy_pose_internal_synthetic_mean_std.md
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/t04_easy_pose_internal_tus_qc_mean_std.md
outputs/production/ablations/ours/multisweep_mean_validation_20260702/tables/tdsc_key_path_mean_std.md
```

Mean/std summary:

| Group | Scope | Main mean result |
| --- | --- | --- |
| T01 key path | synthetic_liver, 5 sweeps | Base 29.569 PSNR; +LG 29.475; +LG+DC 29.473; +LG+DC+EP 29.467. Synthetic remains a PSNR tradeoff, while LPIPS improves from 0.275 to about 0.238-0.249 after adding LG. |
| T01 key path | TUS-QC, 5 sweeps | Base 33.450 PSNR; +LG 33.889; +LG+DC 33.886; +LG+DC+EP 34.235. Real TUS strongly supports LG and Easy-Pose; DC is near-neutral. |
| T02 renderer internal | synthetic_liver, 5 sweeps | local-only has the highest PSNR mean (29.570), but renderer variants improve LPIPS strongly (0.276 to about 0.238-0.239). |
| T02 renderer internal | TUS-QC, 5 sweeps | integral q=3 is best: 33.886 PSNR vs 33.452 local-only, with LPIPS 0.143 vs 0.165. |
| T04 Easy-Pose internal | synthetic_liver, 5 sweeps | uniform canonical is essentially tied with file pose in PSNR (-0.006 dB) with a small LPIPS cost (+0.010). |
| T04 Easy-Pose internal | TUS-QC, 5 sweeps | uniform canonical improves PSNR over file pose by +0.348 dB and keeps LPIPS essentially tied. |
| TDSC key path stress check | TDSC, 3 sweeps | +LG improves PSNR over Base by +0.347 dB; +DC and +EP are near-neutral to slightly negative after +LG. |

Final textual claim:

```text
We report detailed ablation tables on one representative synthetic and one
representative real sweep for compactness. For the main performance-bearing
components, we additionally repeated the key module path, renderer-internal
variants, and Easy-Pose variants over five synthetic and five TUS-QC real
sweeps, and ran a three-sweep TDSC stress check for the key module path. The
averaged results show that local-global rendering is the dominant real-data
quality contributor, Easy-Pose retains quality and improves TUS-QC under
regular sweeps, and responsibility DC remains near-neutral. Synthetic_liver is
best described as a PSNR/perceptual tradeoff: local-only can keep slightly
higher PSNR, while local-global rendering improves LPIPS. TDSC confirms the
renderer benefit but not a monotonic full-stack gain, so claims should be
phrased as dataset-dependent outside the main TUS-QC scope.
```

Observed runtime envelope:

```text
new training runs = 60
summary rows = 61 including header
workers = 6
wall time = approximately 80-100 min as estimated
```

## 11. Pass / Fail Criteria

### Renderer

Strong pass:

```text
Local-global improves PSNR or L1 on at least 3/4 core datasets and does not
materially degrade SSIM/GMSD. LPIPS/DISTS are reported but not the sole pass
criterion because current GS comparisons show Ours is not optimized for all
perceptual metrics.
```

### Responsibility DC

Strong pass:

```text
Responsibility DC is slightly better than gradient DC on at least part of the
representative scope and does not materially damage PSNR/L1/SSIM/GMSD on the
four-dataset factorial.
```

Acceptable pass:

```text
Responsibility DC is performance-neutral relative to gradient DC, has weak
positive rows or sub-component evidence, and has a clear allocation or
pruning-protection rationale. It is not required to be a large final-quality
booster under the fair comparison loss.
```

### Easy-Pose

Strong retention:

```text
canonical_sweep PSNR gap >= -0.3 dB relative to file_pose
```

Acceptable retention:

```text
canonical_sweep PSNR gap >= -0.6 dB relative to file_pose
```

### Seed Stability

Strong pass:

```text
The full method's mean improvement over the ablated baseline exceeds one
standard deviation on the representative datasets.
```

## 12. Historical Execution Order

Completed order for the 2026-06-27 historical GS-strict rerun:

1. Updated Ours ablation base alias and runner to inherit the latest comparison
   Ours config.
2. Ran Stage 0 smoke with no checkpoint retention.
3. Ran T02 and derived T01/T03/T04 from new T02 where possible.
4. Ran T05 to validate responsibility sub-components under bounded scale and
   fair loss.
5. Ran T06 easy-pose pairs.
6. Ran T07 spacing tolerance because the user requested all non-absolutely
   reusable experiments to rerun.
7. Ran T08 negative control.
8. Ran T09 seed stability.
9. Generated T10 mechanism/efficiency evidence from retained logs, summaries,
   per-frame metrics, and selected visuals.
10. Filled registries and committed the completed package.

## 13. Output Reset Log

| Date | Output path | Action | Reason |
| --- | --- | --- | --- |
| 2026-06-27 | `outputs/production/ablations/ours/stage0_dry_run_1m30k_20260622` | cleared | old protocol / smoke no longer current |
| 2026-06-27 | `outputs/production/ablations/ours/t01_module_ladder_derived_20260623` | cleared | derived from old protocol |
| 2026-06-27 | `outputs/production/ablations/ours/t02_renderer_dc_factorial_1m30k_20260622` | cleared | must rerun under latest Ours config |
| 2026-06-27 | `outputs/production/ablations/ours/t03_renderer_component_derived_20260623` | cleared | derived from old T02 |
| 2026-06-27 | `outputs/production/ablations/ours/t04_dc_component_derived_20260623` | cleared | derived from old T02 |
| 2026-06-27 | `outputs/production/ablations/ours/t05_dc_subcomponents_1m30k_20260623` | cleared | must rerun under latest Ours config |
| 2026-06-27 | `outputs/production/ablations/ours/t06_easy_pose_pairs_1m30k_20260623` | cleared | must rerun under latest Ours config |
| 2026-06-27 | `outputs/production/ablations/ours/t07_spacing_tolerance_reuse_20260623` | cleared | not absolutely reusable under user standard |
| 2026-06-27 | `outputs/production/ablations/ours/t08_easy_pose_negative_control_1m30k_20260623` | cleared | must rerun if reported |
| 2026-06-27 | `outputs/production/ablations/ours/t09_seed_stability_1m30k_20260623` | cleared | seed stability is config-specific |
| 2026-06-27 | `outputs/production/ablations/ours/t10_efficiency_mechanisms_20260623` | cleared | old mechanism artifacts tied to old runs |

## 14. Decision Log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-06-22 | Original Ours production ablation lifecycle created. | Needed paper-facing module and mechanism evidence. |
| 2026-06-23 | Old T01-T10 production ablations were completed. | Historical package under older loss/checkpoint/metric protocol. |
| 2026-06-27 | Old production ablation numeric records invalidated for current paper-facing use. | Latest comparison Ours config changed to bounded scale, fair `0.8 L1 + 0.2 SSIM`, final 30k checkpoint, six metrics, and dataset-specific corridor overrides. |
| 2026-06-27 | All non-absolutely-reusable ablation experiments marked for rerun. | User standard: rerun anything not absolutely safe to reuse. |
| 2026-06-27 | Old `outputs/production/ablations/ours` stage outputs cleared. | Prevent accidental reuse and prepare clean rerun output root. |
| 2026-06-27 | Checkpoint retention disabled for future Ours ablation work by this agent. | Save storage; detailed result records replace checkpoint retention. |
| 2026-07-02 | Main paper-facing ablation standard fixed to `synthetic_liver/l1` plus TUS-QC `010_LH_Per_L_DtP_qc_0005_0370`. | Keep all standard tables comparable with six metric columns: Syn/Real PSNR, SSIM, LPIPS. |
| 2026-07-02 | Expanded module ablation plan finalized as T01-T04 plus T11 visual support. | T01 covers joint LG/DC/EP on/off; T02-T04 open each module internally; T11 provides matched qualitative evidence. |
| 2026-07-02 | Parameter sensitivity finalized as three objective-metric figures T12/T14/T15. | Easy-Pose uses a rho response curve, renderer uses tornado/range-dot sensitivity, and DC uses heatmaps plus a dot panel. |
| 2026-07-02 | Historical 116-row rerun reclassified as valid source evidence, not the complete final paper package. | The expanded paper plan introduces missing combinations and sensitivity sweeps beyond the completed rerun. |
| 2026-07-05 | Multi-sweep mean validation completed. | Job 66969 ran 60/60 new rows over five synthetic sweeps, five TUS-QC sweeps, and a three-sweep TDSC stress check; mean trends support LG and Easy-Pose on TUS-QC, show synthetic PSNR/perceptual tradeoff, and show TDSC benefits mainly from LG. |
| 2026-06-27 | Full GS-strict rerun completed in Slurm job 66353. | 116/116 rows completed, worker failures 0, checkpoint payload files retained 0. |
| 2026-06-27 | Local-global renderer retained as a strong positive component. | T03 improves PSNR/L1 on 3/4 core datasets and improves GMSD in all four factorial averages. |
| 2026-06-27 | Responsibility DC retained as performance-neutral with weak positive evidence. | T04 is slightly positive on synthetic_liver and TUS-QC, near-neutral elsewhere; T05 supports retaining full DC on TUS-QC. |
| 2026-06-27 | Easy-pose canonical sweep retained for positive-scope datasets. | T06 minimum canonical-file PSNR gap is -0.1050 dB, above the -0.3 dB strong-retention threshold. |
| 2026-06-27 | Easy-pose scope boundary retained. | T08 spine negative control confirms canonical sweep should not be claimed for the spine topology case. |
| 2026-06-27 | Spacing/corridor tuning noted as a modest follow-up, not a main explanation for comparison gaps. | T07 shows dataset-tunable rho choices, but TUS/TDSC improvements over rho 1.00 are only about 0.07-0.11 dB on real rows. |
| 2026-06-27 | Seed stability passes. | T09 full-vs-base gains exceed pooled seed std on both TUS-QC and TDSC representative rows. |
