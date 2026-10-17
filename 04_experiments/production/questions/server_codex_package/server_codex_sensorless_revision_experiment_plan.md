# Server-Side Codex Execution Plan: Sensorless 3D Ultrasound Reconstruction

## Mission

Bring the implementation, experiments, and manuscript claims into strict agreement for a paper whose primary task is:

> **Sensorless 3D ultrasound reconstruction with Gaussian Splatting**

The final paper is intended to make three contributions:

1. **Calibration-Guided Geometry Parameterization**
2. **Physics-Aware Local-Global Gaussian Renderer**
3. **Contribution-Weighted Density Control**

The current repository review identified two critical problems:

- The current production `ours` row uses `pose_mode=file`, so its main results are not sensorless.
- The current pruning protection uses high \(\tau_g\), although high \(\tau_g\) means weak attenuation under the implemented transmission formula.

This task must resolve those problems, run all required experiments, and produce implementation-grounded evidence for the final paper.

Do not merely edit documentation. Modify code/configs where required, run the experiments, analyze the results, and return reproducible artifacts.

---

# 1. Non-Negotiable Experimental Definitions

## 1.1 Sensorless reconstruction

A run is considered **sensorless** only when reconstruction and Gaussian initialization do not use frame-wise tracked poses.

The sensorless method may use:

- B-mode frames
- frame ordering
- fixed image-plane physical metadata
- nominal dataset/protocol spacing \(d_0\)
- fixed spacing scale \(\rho\)
- fixed canonical sweep direction \(\mathbf u\)
- fixed reference index \(i_0\)

The sensorless method must not use during initialization or reconstruction:

- `poses.npy` frame-wise transforms
- first/last tracked pose
- tracked trajectory length
- pose-network predictions trained with tracker supervision
- tracked poses for per-sequence tuning
- tracked poses for selecting \(\rho\), axis, or sign at test time

Tracked/file poses may be used only for:

- a clearly labeled tracked-pose upper bound
- diagnostic comparison
- benchmark preparation if explicitly disclosed
- evaluation that is clearly separated from reconstruction input

## 1.2 Canonical motion scope

The current canonical geometry model is scoped to:

- one regular sweep
- one fixed canonical direction
- fixed inter-frame spacing
- identity inter-frame rotation
- translation-dominant, near-parallel acquisition

Do not claim arbitrary freehand 6-DoF support.

## 1.3 Main result policy

The main paper result must use `pose_mode=canonical_sweep`.

The existing `pose_mode=file` result should be relabeled as:

> **Tracked-pose upper bound**

It must not be presented as the main sensorless result.

---

# 2. Repository Audit Before Modification

Before changing code, create a machine-readable audit report containing:

- current git commit
- current branch
- uncommitted changes
- relevant production config paths
- current pose modes
- current dataset-level \(d_0\) values
- current \(\rho\), axis, and reference-frame values
- renderer hyperparameters
- density-control hyperparameters
- current result directories and available checkpoints

Save as:

```text
artifacts/sensorless_revision/repository_audit.md
artifacts/sensorless_revision/repository_audit.json
```

Also copy the exact configs used by every new experiment into the result directory.

---

# 3. Required Code Corrections

## 3.1 Create an explicit sensorless production configuration

Create a production-grade canonical-sweep configuration derived from the existing Ours production config.

Requirements:

```yaml
pose_mode: canonical_sweep
canonical_sweep_spacing_mode: uniform
canonical_sweep_delta_scale: 1.0
canonical_sweep_ref_frame: 0
pose_optimization:
  enabled: false
```

Use an explicitly configured canonical axis for every dataset.

Do not silently inherit `pose_mode=file`.

Create clearly named configs, for example:

```text
configs_hydra/experiment/production/sensorless/ours_canonical.yaml
configs_hydra/experiment/production/sensorless/ours_canonical_<dataset>.yaml
```

At runtime, print and save:

- resolved pose mode
- resolved \(d_0\)
- resolved \(\rho\)
- resolved axis
- resolved reference frame
- whether file poses are used for initialization
- whether file poses are used for training
- whether file poses are used for testing

Add a hard assertion for sensorless configs:

```text
No frame-wise file pose may reach SliceMeta used by initialization,
training, or held-out rendering.
```

It is acceptable for file poses to be loaded only for dataset length if the generated canonical poses fully replace them before any geometry is constructed. Log this explicitly.

---

## 3.2 Correct transmission-aware pruning

The renderer uses:

\[
T_{igp}
=
\tau_g+(1-\tau_g)\exp(-\lambda_{\mathrm{ray}}\rho_{igp}),
\]

where low \(\tau_g\) produces stronger attenuation.

The existing rule protects high \(\tau_g\), which protects weakly attenuating Gaussians. Replace or supplement it with a protection score that reflects actual path modulation.

### Required implementations

Implement and compare at least these pruning-protection variants.

#### Variant P0: No transmission protection

```text
protection = false
```

#### Variant P1: Existing high-\(\tau\) protection

Keep the current implementation unchanged as a diagnostic baseline:

```text
tau >= high_quantile(tau)
```

Do not use it as the proposed final method unless experiments clearly justify it.

#### Variant P2: Low-\(\tau\) protection

Protect Gaussians with strong attenuation potential:

```text
tau <= low_quantile(tau)
```

Use a configurable quantile.

#### Variant P3: Actual path-effect protection — preferred

Implement a per-Gaussian score based on actual rendered path effect.

Preferred definitions, in descending order:

\[
a_g
=
\sum_{i,p}
-\log T_{igp},
\]

or

\[
a_g
=
\sum_{i,p}
(1-T_{igp}),
\]

or, if necessary:

\[
a_g
=
(1-\tau_g)
\sum_{i,p}\rho_{igp}.
\]

Requirements:

- score must depend on whether rays actually encounter the Gaussian
- score must be accumulated from diagnostic rendering
- score must be detached from gradient computation
- accumulation schedule must be logged
- use high quantile of \(a_g\) as protection
- expose score statistics in logs

The intended pruning rule is:

\[
\text{prune}_g
=
\text{weak-local}_g
\land
\text{low-contribution}_g
\land
\neg \text{path-protected}_g.
\]

Preserve optional independent scale pruning if already used.

### Required unit tests

Create tests verifying:

1. For fixed overlap, lower \(\tau_g\) yields lower \(T_{igp}\).
2. Existing P1 protects weak attenuators.
3. P2 protects strong attenuation-potential Gaussians.
4. P3 assigns larger effect to a Gaussian traversed by many relevant rays.
5. A Gaussian with weak local response but large global path effect survives pruning under P3.
6. A Gaussian with weak local response and negligible path effect is pruned.

Save test outputs.

---

## 3.3 Make density-control statistics explicit and reproducible

The current contribution score is:

\[
\bar r_g
=
\frac{
\sum_p c_{gp}r_p
}{
\sum_p c_{gp}+\epsilon
},
\]

\[
s_g
=
\bar r_g
\log\left(1+\sum_p c_{gp}\right).
\]

Keep this exact formula unless experiments motivate a change.

Rename logging/documentation terms consistently:

- `local_response_contribution`
- `residual_weighted_contribution`
- `contribution_score`
- `path_effect_score`

Avoid calling the quantity probabilistic responsibility or true marginal contribution.

For every density-control event, save:

- selected diagnostic frame
- contribution distribution
- residual-weighted contribution distribution
- final densification score distribution
- selected clone/split indices
- pruning reasons
- path-protection reasons
- Gaussian count before and after
- mean/min/max \(\alpha_g\)
- mean/min/max \(\tau_g\)
- mean/min/max path-effect score

Create optional diagnostic images showing:

- reconstruction
- residual map
- local contribution map
- Gaussian densification locations
- pruned Gaussian locations

---

# 4. Nominal Spacing and Pose-Information Leakage Audit

For each dataset, produce a table with:

| Dataset | \(d_0\) value | Unit | Source | Uses pose analysis? | Available in real deployment? | Final paper wording |
|---|---:|---|---|---|---|---|

Distinguish:

- acquisition/device-provided spacing
- synthetic known spacing
- dataset-provided spacing
- manually specified spacing
- value estimated from tracked poses during benchmark preparation

For TUS and spine, explicitly identify whether the current \(d_0\) was estimated from ground-truth pose steps.

## Required alternative spacing protocols

Run sensorless experiments under at least:

### S0: Current nominal spacing

Use the repository’s current fixed \(d_0\).

### S1: Rounded protocol spacing

Use a rounded, sequence-independent dataset-level value that could plausibly be specified without per-sequence tracking.

### S2: Perturbed spacing sensitivity

Use:

\[
\rho\in
\{0.8,0.9,1.0,1.1,1.2\}.
\]

If resources permit, add:

\[
\{0.7,1.3\}.
\]

Do not tune \(\rho\) per test sequence.

Report whether a single dataset-level \(\rho=1\) remains competitive.

The main result must use a fixed predeclared \(\rho\), preferably \(1.0\).

---

# 5. Required Experimental Matrix

Use the same datasets and train/test protocols as the existing production comparison unless a protocol flaw is identified.

Use at least three random seeds for the final essential rows when computationally feasible. If full three-seed runs are too expensive, use:

- one seed for broad screening
- three seeds for the final selected configurations
- clearly report which rows are single-seed

## 5.1 Geometry experiment

Run:

| ID | Pose source | Renderer | Density control | Purpose |
|---|---|---|---|---|
| G0 | file poses | proposed renderer | proposed DC | tracked-pose upper bound |
| G1 | canonical sweep | proposed renderer | proposed DC | main sensorless method |
| G2 | canonical sweep, wrong/perturbed spacing | proposed renderer | proposed DC | spacing sensitivity |
| G3 | canonical sweep, axis/sign variations | proposed renderer | proposed DC | coordinate/protocol sensitivity |
| G4 | canonical sweep, no global renderer | local-only renderer | proposed DC | geometry under simpler renderer |
| G5 | canonical sweep, fixed/no densification | proposed renderer | no/protocol baseline DC | geometry without adaptive capacity |

If there are existing sensorless pose-estimation baselines, run them using their intended pose estimates. Otherwise clearly state that no learned pose baseline is available.

### Axis/sign experiment

For each dataset, test at least:

- configured canonical axis
- opposite sign
- one incorrect orthogonal axis

Determine whether sign is merely a coordinate convention and whether wrong axis materially harms reconstruction.

---

## 5.2 Renderer ablation

Under the same canonical sensorless geometry, run:

| ID | Local branch | Global branch | Composition |
|---|---|---|---|
| R0 | local only | off | \(L\) |
| R1 | local | current integral transmission | \(L(1-\beta+\beta T)\) |
| R2 | local | global branch with \(\beta=1\) | \(LT\) |
| R3 | local | global branch, reduced \(\beta\) | sensitivity |
| R4 | local | global branch, varied ray strength | sensitivity |
| R5 | alternative overlap approximation if available | global | implementation comparison |

Required \(\beta\) sensitivity:

\[
\beta\in\{0,0.4,0.6,0.8,1.0\}.
\]

Required strength sensitivity around the current value:

\[
\lambda_{\mathrm{ray}}
\in
\{0.25,0.40,0.55,0.70,1.00\}.
\]

Use a broad screening subset first, then confirm the selected setting on all datasets.

Report:

- image metrics
- runtime
- memory
- number of ray candidates
- average transmission
- fraction of pixels at min-transmission clamp
- gradient stability
- NaN/failure rate

---

## 5.3 Density-control ablation

Under canonical geometry and selected renderer, run:

| ID | Densification | Pruning protection |
|---|---|---|
| D0 | no density control | none |
| D1 | standard/legacy gradient DC | legacy |
| D2 | contribution-weighted densification | none |
| D3 | contribution-weighted densification | current high-\(\tau\) P1 |
| D4 | contribution-weighted densification | low-\(\tau\) P2 |
| D5 | contribution-weighted densification | path-effect P3 |

If compute permits, also run:

| ID | Description |
|---|---|
| D6 | contribution score without `log1p(contribution)` |
| D7 | contribution score with multi-frame accumulation |
| D8 | contribution score plus gradient fallback |

### Multi-frame diagnostic extension

Because current density control uses one diagnostic frame per event, implement an optional multi-frame accumulation experiment:

- collect statistics over \(K\in\{1,4,8\}\) frames
- aggregate contribution and residual-contribution before densification/pruning
- keep compute budget documented
- test whether multi-frame statistics improve stability or coverage

Do not make this the final method unless it produces a meaningful improvement relative to cost.

Report:

- final metrics
- Gaussian count trajectory
- number cloned
- number split
- number pruned
- contribution score histograms
- path-effect protection counts
- runtime overhead
- memory overhead
- failure cases

---

# 6. Evaluation Requirements

## 6.1 Main image metrics

Use the existing production metrics, including as applicable:

- PSNR
- L1 or MAE
- SSIM
- LPIPS
- DISTS
- GMSD

Report per sequence, per dataset, and overall mean/std.

Do not report only aggregate averages.

## 6.2 Geometry and scale diagnostics

The current image metrics do not prove physical-scale correctness.

Add geometry diagnostics where ground-truth poses are available, but use them only for evaluation.

For canonical generated poses, compare against file poses using:

- inter-frame translation magnitude error
- total sweep-length error
- direction/angular error
- relative translation error after removing global translation
- relative rotation magnitude in ground truth
- deviation from fixed-orientation assumption

Do not use aligned pose metrics to tune each test sequence.

Clearly distinguish:

- raw physical-scale diagnostics
- rigid-aligned diagnostics
- similarity-aligned diagnostics

Similarity alignment must not be used as evidence that spacing is physically correct.

## 6.3 Scope diagnostics

Quantify how well each dataset satisfies the assumed motion model:

- mean and max inter-frame rotation
- fraction of motion explained by dominant translation axis
- spacing coefficient of variation
- curvature or lateral deviation from best-fit line
- sweep monotonicity

Create a table:

| Dataset/sequence | Mean rotation step | Dominant-axis ratio | Spacing CV | Line-fit residual | Canonical suitability |
|---|---:|---:|---:|---:|---|

This table is necessary to justify the scoped regular-sweep assumption.

## 6.4 Computational metrics

Report:

- total training time
- iteration time
- peak GPU memory
- initial Gaussian count
- final Gaussian count
- renderer forward/backward time
- density-control overhead
- ray candidate statistics

Use the same hardware where possible.

---

# 7. Dataset and Split Integrity

For every experiment, record:

- dataset version
- subject/sweep IDs
- train/test split
- resized image resolution
- number of train frames
- number of test frames
- nominal spacing
- pose mode
- random seed
- git commit
- resolved config

Verify that no test sequence is used to choose:

- \(d_0\)
- \(\rho\)
- axis/sign
- \(\beta\)
- ray strength
- density thresholds
- pruning thresholds

If historical tuning has already inspected test sequences, disclose it and create a cleaner validation protocol if possible.

---

# 8. Required Statistical Analysis

For the final key comparisons:

- sensorless full method vs tracked-pose upper bound
- local-only vs local-global renderer
- contribution DC vs legacy/no DC
- P3 protection vs P1/P2/no protection

Report:

- mean
- standard deviation
- paired per-sequence difference
- bootstrap 95% confidence interval or paired statistical test
- number of sequences improved/degraded
- largest failure cases

Do not overinterpret tiny metric differences.

Define a practically meaningful improvement threshold before final selection where possible.

---

# 9. Failure Analysis

Identify and save examples for:

- irregular sweep spacing
- substantial probe rotation
- incorrect canonical direction
- incorrect nominal spacing
- regions with zero existing Gaussian contribution
- excessive attenuation collapse
- transmission clamp saturation
- over-densification
- over-pruning
- canonical geometry worse than file-pose geometry

For each failure case, save:

- input frames
- target
- reconstruction
- residual map
- local-only output
- transmission map
- final output
- Gaussian count trajectory
- relevant config

Produce:

```text
artifacts/sensorless_revision/failure_analysis.md
```

---

# 10. Decision Rules

Use the following decision logic after experiments.

## 10.1 Can the paper remain primarily sensorless?

Yes only if:

- main results use canonical geometry
- no frame-wise poses are used during reconstruction
- performance is competitive enough to support the task claim
- spacing and axis are fixed without test-pose tuning
- the regular-sweep scope is documented and empirically justified

If canonical performance collapses across most datasets, report this plainly. Do not silently fall back to file poses while retaining a sensorless title.

## 10.2 Which pruning protection becomes final?

Prefer P3 if it:

- matches the intended path-effect interpretation
- improves or preserves image metrics
- prevents harmful pruning
- has acceptable overhead

Use P2 if P3 is too expensive and P2 performs comparably.

Do not retain P1 merely because it is the old implementation.

## 10.3 Can the renderer be called physics-aware?

Use **Physics-Aware Local-Global Gaussian Renderer** as the contribution name only if the paper body consistently describes it as:

> acoustic-path-aware, physics-inspired, bounded transmission modulation

Do not claim full physical ultrasound simulation.

## 10.4 Can Contribution-Weighted Density Control remain the name?

Yes, provided the paper defines contribution as:

> local response contribution

and states the exact score.

---

# 11. Required Output Artifacts

Create this directory structure:

```text
artifacts/sensorless_revision/
├── repository_audit.md
├── repository_audit.json
├── code_changes.md
├── experiment_manifest.csv
├── experiment_manifest.json
├── results_summary.md
├── results_per_sequence.csv
├── geometry_diagnostics.csv
├── runtime_memory.csv
├── density_control_stats.csv
├── pruning_protection_ablation.csv
├── spacing_sensitivity.csv
├── renderer_ablation.csv
├── failure_analysis.md
├── manuscript_facts.md
├── recommended_final_config.yaml
├── configs/
├── logs/
├── plots/
├── diagnostic_images/
└── tests/
```

## `experiment_manifest.csv`

Each row must contain:

- experiment ID
- dataset
- sequence
- seed
- git commit
- config path
- pose mode
- \(d_0\)
- \(\rho\)
- axis
- renderer variant
- density-control variant
- pruning protection variant
- status
- checkpoint path
- result path

## `manuscript_facts.md`

Write concise verified facts that the paper can safely claim:

### Calibration-Guided Geometry Parameterization
- exact inputs
- exact missing inputs
- exact equation
- fixed vs optimized parameters
- scope
- limitations
- source of nominal spacing

### Physics-Aware Local-Global Gaussian Renderer
- exact Gaussian attributes
- exact local formula
- exact global formula
- exact composition
- safe physical interpretation

### Contribution-Weighted Density Control
- exact diagnostic schedule
- exact score
- exact split/clone rule
- exact pruning rule
- selected path-protection rule

Also list claims that must not appear.

---

# 12. Required Final Report

Return a final report with the following sections.

## A. Executive conclusion

State plainly:

- whether the main task is now genuinely sensorless
- whether canonical geometry is competitive
- which datasets satisfy the motion assumption
- whether the renderer improves results
- whether contribution-weighted DC improves results
- which pruning protection is correct and effective

## B. Code modifications

For every change:

- file path
- function/class
- rationale
- before/after behavior
- tests

## C. Experiment results

Provide tables for:

- tracked upper bound vs sensorless main method
- spacing sensitivity
- axis/sign sensitivity
- renderer ablation
- density-control ablation
- pruning protection ablation
- multi-frame diagnostic experiment
- runtime and memory

## D. Final recommended method

Specify one exact final production config.

## E. Manuscript consequences

State:

- what the paper may claim
- what must be weakened
- what must be removed
- whether multi-segment geometry should remain excluded
- whether file-pose results should be upper bound only

## F. Unresolved risks

List any remaining concerns, including:

- pose-derived nominal spacing
- test-set tuning
- poor performance on irregular sweeps
- image metrics not proving absolute geometry
- baseline fairness

---

# 13. Execution Priorities

If compute is limited, execute in this order:

1. Sensorless configuration validation and leakage assertions
2. One-sequence smoke tests for G0/G1
3. Fix and unit-test P2/P3 pruning protection
4. Broad single-seed screening:
   - G0/G1
   - R0/R1
   - D0/D2/D3/D4/D5
5. Spacing sensitivity around \(\rho=1\)
6. Final configuration selection
7. Full-dataset final runs
8. Three-seed confirmation for essential rows
9. Geometry/scope diagnostics
10. Failure analysis and final report

Do not skip the tracked-pose versus canonical comparison.

---

# 14. Stop Conditions and Error Handling

Stop and report immediately if:

- a supposedly sensorless run uses file poses in initialization or rendering
- canonical pose generation is silently overridden
- test metrics require unavailable tracked poses without being labeled
- result directories mix file-pose and canonical runs
- a code change causes inconsistent checkpoints
- P3 gradients or diagnostics corrupt training
- numerical instability produces NaNs

Do not overwrite existing production results.

Use new experiment names and output directories.

---

# 15. Final Instruction

The goal is not to defend the existing paper draft. The goal is to determine the strongest method and the strongest defensible sensorless claim supported by the actual implementation and experiments.

Prefer negative but reliable conclusions over positive conclusions obtained through pose leakage, per-test tuning, or mislabeled tracked-pose results.
