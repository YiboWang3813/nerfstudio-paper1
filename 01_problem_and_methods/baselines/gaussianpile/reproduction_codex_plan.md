# GaussianPile Codex Reproduction Plan

Legacy note: this document preserves the original reproduction plan and may mention old `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/gaussianpile/` and should be launched with Hydra `experiment=...` aliases.

Date: 2026-05-27

This document is the concrete implementation and experiment plan for reproducing GaussianPile in `omniultra`.
It is based on `docs/methods/baselines/gaussianpile/reproduction_plan.md`, the repository development rules in
`docs/operations/development_skills.md`, `docs/operations/git_workflow_skills.md`, and
`docs/operations/scope_skills.md`, plus the existing `methods/ours` and `ops/ours_renderer` implementation patterns.

The user has explicitly authorized integration changes to shared entry points such as `run.py` and
`configs/base/models.py` for GaussianPile registration. The implementation should still keep those shared edits
minimal and limited to GaussianPile import/config registration.

## 1. Acceptance Target

The reproduction is considered accepted when all of the following are true:

1. GaussianPile runs through the main config-driven framework.
2. Runtime method code is implemented under `methods/gaussianpile`.
3. The GaussianPile CUDA custom renderer is implemented under `ops/gaussianplie_renderer`.
4. `methods/gaussianpile/render.py` calls the CUDA renderer from `ops/gaussianplie_renderer`.
5. The synthetic liver experiment runs with:
   - dataset: `synthetic_liver`
   - initial Gaussians: `1_000_000`
   - training iterations: `30_000`
   - density control: enabled
   - slice forward model: GaussianPile focus-aware additive rasterization
6. A 1M-Gaussian speed benchmark reaches more than `100` train iterations per second.
7. Tests cover reference math, CUDA forward, additive accumulation, focus limits, and backward gradients.
8. Slurm scripts, logs, checkpoints, summaries, and result reports follow `docs/operations/development_skills.md`,
   `docs/operations/slurm_skills.md`, and `docs/operations/pipeline_skills.md`.
9. Final delivery includes `git diff --name-only` and `git status --short` boundary checks.

Note: the requested renderer directory is spelled `ops/gaussianplie_renderer` rather than
`ops/gaussianpile_renderer`. Keep this spelling consistently in paths, package names, rebuild scripts, tests, and
Slurm commands unless the user later requests a rename.

## 2. Allowed Edit Scope

GaussianPile method-owned files:

```text
methods/gaussianpile/**
ops/gaussianplie_renderer/**
tests/methods/gaussianpile/**
tests/ops/gaussianplie_renderer/**
config_files/_base_/models/gaussianpile.yaml
config_files/experiments/gaussianpile/**
scripts/methods/baselines/gaussianpile/**
docs/methods/baselines/gaussianpile/*.md
docs/methods/baselines/gaussianpile/*.md
```

Authorized integration files:

```text
run.py
configs/base/models.py
```

The integration edits should only register GaussianPile state/callback/config modules. Do not use this authorization
as a reason to alter unrelated methods, engine semantics, scene loading, generic Gaussian containers, or shared CUDA
helpers.

Default do-not-touch areas unless a later task explicitly expands scope:

```text
engine/**
scene/**
gaussian/**
utils/**
ops/common/**
methods/<other_method>/**
ops/<other_renderer>/**
README.md
```

## 3. Existing Repository References

Use `methods/ours` as the primary method skeleton reference:

```text
methods/ours/configs.py
methods/ours/state.py
methods/ours/gaussian_container.py
methods/ours/gaussian_solver.py
methods/ours/density_control.py
methods/ours/gs_only_callbacks.py
methods/ours/render.py
```

Use `ops/ours_renderer` as the primary custom CUDA renderer layout reference:

```text
ops/ours_renderer/setup.py
ops/ours_renderer/rebuild.sh
ops/ours_renderer/include/**
ops/ours_renderer/src/**
ops/ours_renderer/ours_renderer/api.py
ops/ours_renderer/ours_renderer/autograd.py
ops/ours_renderer/ours_renderer/types.py
ops/ours_renderer/ours_renderer/us_focus_reference.py
ops/ours_renderer/tests/**
```

Use these existing configs as experiment/config style references:

```text
config_files/_base_/scene/liver.yaml
config_files/_base_/models/ours.yaml
config_files/experiments/us_focus/liver_1m_10k_dc.yaml
config_files/experiments/ultragauss/liver_1m_10k_dc.yaml
```

Use these Slurm patterns as references, but fix log/checkpoint paths for GaussianPile:

```text
scripts/methods/ours/local_global_renderer/slurm/**
scripts/methods/ours/renderer_baseline/slurm/run_10k_dc_ablation.sh
```

## 4. Method Math to Implement

GaussianPile uses world-space 3D Gaussians and slice-camera rendering. It does not use SH, RGB view-dependent color,
depth sorting, alpha compositing, or front-to-back transmittance.

Per Gaussian parameters:

```text
mu       : (N, 3), world-space center in mm
scale    : (N, 3), positive anisotropic scale in mm
quat     : (N, 4), unit quaternion, wxyz convention
alpha    : (N,), intensity / opacity coefficient
```

Covariance:

```text
S = diag(scale)
R = quat_to_matrix(quat)
Sigma = R S S^T R^T
```

For each slice pose, transform to camera/slice coordinates:

```text
mu_c = R_c mu + t
Sigma_c = R_c Sigma R_c^T
```

Inject finite slice thickness / focus precision:

```text
Sigma_e_inv = Sigma_c_inv + e3 e3^T / sigma_z^2
Sigma_e = inverse(Sigma_e_inv)
mu_e = Sigma_e Sigma_c_inv mu_c
```

Opacity modulation:

```text
opacity_r = exp(-0.5 * (mu_c^T Sigma_c_inv mu_c - mu_e^T Sigma_e_inv mu_e))
```

Screen-space marginal:

```text
mu_2d = mu_e[:2]
Sigma_2d = Sigma_e[:2, :2]
alpha_tilde = alpha * opacity_r / sqrt(clamp(det(Sigma_2d), det_eps))
```

Additive rasterization:

```text
I(p) = sum_i alpha_tilde_i * exp(-0.5 * d_i^T Sigma_2d_i^-1 d_i)
d_i = p - mu_2d_i
```

Critical implementation details:

1. The determinant term is part of the differentiable model. Backward must not treat it as a constant.
2. `sigma_z` defaults to `DatasetMeta.delta_elevational_mm` when not explicitly configured.
3. Clamp determinant and covariance diagonals with small epsilons to avoid singular footprints.
4. Renderer output is a single-channel slice image compatible with the existing `LossComputer` and `MetricComputer`.
5. The returned depth image from `methods/gaussianpile/render.py` can be zeros because GaussianPile is additive volume
   accumulation, not depth compositing.

## 5. Implementation Phases

### Phase 1: Method Skeleton

Create the method package:

```text
methods/gaussianpile/
  __init__.py
  configs.py
  state.py
  render.py
  gaussian_container.py
  gaussian_solver.py
  density_control.py
  offline_callbacks.py
```

`configs.py` should define:

```text
ModelConfig
RenderConfig
GaussInitConfig
GaussContainerConfig
GaussianPileSliceRenderConfig
OfflineConfig
```

Minimum config fields:

```text
model.name = "gaussianpile"
model.gauss_init.num_points = 1_000_000
model.render.enable_input_checks = false
model.slice_render.tile_size_px = 16
model.slice_render.radius_sigma = 3.0
model.slice_render.sigma_z_mm = null
model.slice_render.det_eps = 1.0e-6
model.slice_render.cov_eps = 1.0e-6
model.slice_render.opacity_cull_threshold = 1.0e-6
model.gauss_container.lr.means = 0.0006
model.gauss_container.lr.scales = 0.002
model.gauss_container.lr.quats = 0.001
model.gauss_container.lr.opacities = 0.02
model.gauss_container.lr_final_mult = 0.1
model.offline.density_control.start_iter = 500
model.offline.density_control.end_iter = 25000
model.offline.density_control.freq = 500
model.offline.density_control.densify_grad = 0.00005
model.offline.density_control.prune_opacity = 0.02
```

`state.py` should register:

```text
@register_state("GaussianPileOfflineState")
class GaussianPileOfflineState(State):
    scene
    gauss_container
    gauss_solver
    dc
```

`offline_callbacks.py` should register:

```text
GaussianPileOfflineSetupCallback
GaussianPileOfflineTrainCallback
GaussianPileOfflineTestCallback
GaussianPileOfflineSaveCallback
GaussianPileOfflineFinishCallback
```

The training loop should follow `methods/ours/gs_only_callbacks.py`:

1. Build scene through `SceneFactory.build_scene`.
2. Set `scene.sampler.num_init` from `model.gauss_init.num_points`.
3. Initialize Gaussian container from `scene.build_init_gaussians()`.
4. Render one sampled train slice per iteration.
5. Compute loss through existing `LossComputer`.
6. Backprop, optimizer step, density control step, scalar logging, test, save.
7. Save checkpoints through existing `io_manager`.

### Phase 2: Gaussian Container

Implement `GaussianPileGaussContainer` with:

```text
means
scales
quats
opacities
```

Activation choices:

```text
means      : identity
scales     : bounded sigmoid by default for stable liver runs, with min/max in mm
quats      : normalize with eps
opacities  : sigmoid * opacity_max
```

Initialization:

1. Use `scene.build_init_gaussians()` for world-mm `means`, `scales`, `quats`.
2. Initialize opacity around configured value, e.g. `0.1`.
3. Do not add SH coefficients.
4. Do not add method-unrelated intensity channels unless a compatibility issue requires it. If an `intensity` property is
   added for compatibility, document that GaussianPile's physical alpha is `opacity * intensity` or choose one
   canonical alpha before training.

Required methods:

```text
init_from_scene(scene_info)
ensure_gaussianpile_renderer(dataset_meta, model_cfg)
append_gaussians(...)
remove_gaussians(mask)
```

### Phase 3: Solver

Implement `GaussianPileGaussSolver` following `OursGaussSolver` patterns:

1. Adam optimizer with one parameter group per Gaussian attribute.
2. Per-attribute learning rates from the GaussianPile config.
3. Exponential LR decay to `0.1 * initial_lr` over the training horizon.
4. Proper optimizer state resize after densify/prune.
5. Public methods:

```text
setup()
zero_grad()
step()
step_lr(iteration)
replace_tensor_to_optimizer(...)
cat_tensors_to_optimizer(...)
prune_optimizer(...)
```

### Phase 4: Density Control

Implement `GaussianPileOfflineDensityController` with:

1. Accumulated position-gradient statistics.
2. Start/stop/frequency gates:
   - start: `500`
   - stop: `25000`
   - freq: configurable, default `500`
3. Densify high-gradient Gaussians:
   - threshold: `5.0e-5`
   - split large Gaussians
   - clone small Gaussians
4. Prune low-opacity Gaussians:
   - threshold: `0.02`
5. Enforce optional `n_max` to avoid runaway memory use.
6. After every prune/densify operation, verify container tensor lengths, optimizer state lengths, and statistics buffer
   lengths match.

The initial version may reuse the local structure of `methods/ours/density_control.py`, but GaussianPile thresholds
must match the plan and final config.

### Phase 5: PyTorch Reference Renderer

Before CUDA kernels, implement a small differentiable reference in:

```text
ops/gaussianplie_renderer/gaussianplie_renderer/reference.py
```

Required functions:

```text
quat_to_matrix_wxyz(quats)
build_covariance(scales, quats, cov_eps)
world_to_camera(means, covariances, pose)
focus_project(means_c, covs_c, opacities, sigma_z, det_eps, cov_eps)
render_reference(means, scales, quats, opacities, pose, grid, config)
```

The reference can be slow because it is only for tests and tiny cases. It should use torch autograd so CUDA backward can
be checked with finite differences and reference gradients.

### Phase 6: CUDA Renderer Package

Create:

```text
ops/gaussianplie_renderer/
  README.md
  setup.py
  rebuild.sh
  include/gaussianplie_renderer/
  src/
  gaussianplie_renderer/
    __init__.py
    api.py
    autograd.py
    check.py
    types.py
    reference.py
  tests/
```

Python API should expose:

```text
GaussianPlieRenderConfig
GaussianPlieRenderer
render_gaussianpile(...)
```

`GaussianPlieRenderConfig` should include:

```text
H
W
lateral_spacing_mm_per_pix
axial_spacing_mm_per_pix
axial_near_mm
tile_size_px
radius_sigma
sigma_z_mm
det_eps
cov_eps
opacity_cull_threshold
enable_input_checks
```

CUDA forward steps:

1. Validate contiguous CUDA tensors and dtypes.
2. Convert Gaussian parameters to covariance.
3. Transform world Gaussians to current slice camera frame.
4. Apply focus reparameterization.
5. Compute `mu_2d`, `Sigma_2d_inv`, footprint radius, and `alpha_tilde`.
6. Cull Gaussians with tiny `alpha_tilde`, invalid covariance, or out-of-image footprint.
7. Bin Gaussians into tiles.
8. Launch additive raster kernel over tiles/pixels.
9. Return image and optional compact diagnostics.

CUDA backward steps:

1. Accumulate gradients from pixels to `alpha_tilde`, `mu_2d`, and conic/covariance parameters.
2. Backprop through determinant normalization.
3. Backprop through focus reparameterization.
4. Backprop through camera transform.
5. Backprop through covariance factorization to scales and quats.
6. Backprop alpha to raw opacity through PyTorch container activation.

The first implementation can prioritize correctness and then optimize mapper/raster kernels to pass the 100+ it/s
target.

### Phase 7: `methods/gaussianpile/render.py`

Implement render as the method boundary:

```text
def render(
    model_cfg,
    dataset_meta,
    slice_meta,
    container,
    is_jitter=False,
    return_intermediates=False,
    pose_override=None,
):
    ...
```

Behavior:

1. Ignore `is_jitter` unless GaussianPile later adds method-specific jitter.
2. Use `pose_override` when provided; otherwise use `slice_meta.pose`.
3. Move pose to container device/dtype.
4. Build or reuse the `GaussianPlieRenderer` from the container.
5. Use dataset metadata to build slice grid geometry. Match the `ours_renderer.grid_geometry` convention where possible.
6. Call the CUDA renderer with `means`, `scales`, `quats`, `opacities`, and pose.
7. Return:

```text
pred_image
torch.zeros_like(pred_image)
diagnostics_dict
```

### Phase 8: Framework Integration

Authorized minimal integration changes:

`configs/base/models.py`:

```python
from methods.gaussianpile.configs import ModelConfig as GaussianPileModelConfig

MODEL_CONFIG_REGISTRY = {
    ...
    "gaussianpile": GaussianPileModelConfig,
}
```

`run.py`:

```python
import methods.gaussianpile.state  # noqa: F401
import methods.gaussianpile.offline_callbacks  # noqa: F401
```

No other shared behavior should be changed for initial integration.

## 6. Config Files

Create model defaults:

```text
config_files/_base_/models/gaussianpile.yaml
```

Minimum contents:

```yaml
model:
  name: "gaussianpile"
  render:
    enable_input_checks: false
  gauss_init:
    num_points: 1000000
  slice_render:
    tile_size_px: 16
    radius_sigma: 3.0
    sigma_z_mm: null
    det_eps: 1.0e-6
    cov_eps: 1.0e-6
    opacity_cull_threshold: 1.0e-6
  gauss_container:
    lr:
      means: 0.0006
      scales: 0.002
      quats: 0.001
      opacities: 0.02
    lr_final_mult: 0.1
    init_val:
      opacities: 0.1
    constraints:
      scale_activation: "bounded_sigmoid"
      scale_min_mm: 0.05
      scale_max_mm: 3.0
      opacity_max: 1.0
  offline:
    density_control:
      enable:
        densify: true
        prune: true
      timing:
        start_iter: 500
        end_iter: 25000
        freq: 500
      thresholds:
        densify_grad: 5.0e-5
        split_scale: 0.6
        prune_opacity: 0.02
      n_max: 2000000
```

Create experiments:

```text
config_files/experiments/gaussianpile/liver_smoke_100k_500.yaml
config_files/experiments/gaussianpile/liver_1m_300_speed.yaml
config_files/experiments/gaussianpile/liver_1m_30k_dc.yaml
```

`liver_smoke_100k_500.yaml`:

1. Synthetic liver scene.
2. 100k initial Gaussians.
3. 500 iterations.
4. Density control can be disabled for first import/render smoke, then enabled in a second smoke.
5. Test/save intervals small enough to verify metrics and checkpoint.

`liver_1m_300_speed.yaml`:

1. Synthetic liver scene.
2. 1M initial Gaussians.
3. 300 iterations.
4. Disable test and save.
5. Enable density control only if measuring full training-path speed; also keep a no-DC speed variant if renderer profiling
   needs isolation.

`liver_1m_30k_dc.yaml`:

1. Synthetic liver scene.
2. 1M initial Gaussians.
3. 30k iterations.
4. Density control enabled from 500 to 25000.
5. Loss should reproduce paper intent: `L1 + 0.2 * SSIM` using repository loss configuration. If the existing config
   expresses SSIM as a similarity term rather than D-SSIM, verify sign/scale before final run.
6. Metrics: `psnr`, `l1`, `ssim`, `lpips`.
7. Output root: `checkpoints/gaussianpile`.
8. Unique exp tag: `gaussianpile_liver_1m_30000_dc`.

## 7. Tests

Method tests:

```text
tests/methods/gaussianpile/test_config_build.py
tests/methods/gaussianpile/test_container_init.py
tests/methods/gaussianpile/test_density_control_shapes.py
```

Renderer tests:

```text
tests/ops/gaussianplie_renderer/test_reference_math.py
tests/ops/gaussianplie_renderer/test_focus_limits.py
tests/ops/gaussianplie_renderer/test_additive_property.py
tests/ops/gaussianplie_renderer/test_forward_reference.py
tests/ops/gaussianplie_renderer/test_backward_finite_difference.py
```

Required checks:

1. Identity camera transform:
   - `mu_c == mu`
   - `Sigma_c == Sigma`
2. Focus all-in limit:
   - large `sigma_z` gives `Sigma_e ~= Sigma_c`
   - large `sigma_z` gives `mu_e ~= mu_c`
   - large `sigma_z` gives `opacity_r ~= 1`
3. Thin-slice limit:
   - smaller `sigma_z` suppresses Gaussians with larger absolute `z_c`
4. Projection:
   - `Sigma_2d` equals the xy marginal of `Sigma_e`
   - determinant is positive after clamps
5. Additive renderer:
   - two identical Gaussians produce exactly twice the single-Gaussian image within tolerance
6. CUDA forward:
   - CUDA output matches PyTorch reference on tiny scenes
7. CUDA backward:
   - gradients match finite differences for means, scales, quats, and opacities on tiny scenes
8. Computer integration:
   - one rendered image can pass through `LossComputer` and produce a scalar loss with backward
9. Density control:
   - after prune/densify, all Gaussian tensors, optimizer states, and stats buffers have the same length

## 8. Slurm Scripts

Create:

```text
scripts/methods/baselines/gaussianpile/slurm/01_reference_tests.sh
scripts/methods/baselines/gaussianpile/slurm/02_rebuild_renderer.sh
scripts/methods/baselines/gaussianpile/slurm/03_forward_cuda.sh
scripts/methods/baselines/gaussianpile/slurm/04_backward_cuda.sh
scripts/methods/baselines/gaussianpile/slurm/05_liver_smoke.sh
scripts/methods/baselines/gaussianpile/slurm/06_speed_1m_300.sh
scripts/methods/baselines/gaussianpile/slurm/07_liver_1m_30k_dc.sh
```

Each script must:

1. Write stdout/stderr to `logs/gaussianpile/%x_%j.out` and `logs/gaussianpile/%x_%j.err`.
2. Use `PROJECT_DIR=/home/boywang25/ultrasound-scene-reconstruction/omniultra` by default.
3. Activate `conda activate gaussian_splatting`.
4. Load `module load cuda/11.8.0` for renderer build and GPU tests.
5. Echo job id, host, date, config path, output tag, and overrides.

Before submitting:

```bash
mkdir -p logs/gaussianpile
```

Recommended execution order:

```bash
sbatch scripts/methods/baselines/gaussianpile/slurm/01_reference_tests.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/02_rebuild_renderer.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/03_forward_cuda.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/04_backward_cuda.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/05_liver_smoke.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/06_speed_1m_300.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/07_liver_1m_30k_dc.sh
```

If the queue is constrained, submit serially or use `afterany` dependencies. Do not spam pending jobs.

## 9. Speed Plan

The speed gate is more than `100` train iterations per second with 1M initial Gaussians.

Use `config_files/experiments/gaussianpile/liver_1m_300_speed.yaml` first:

1. Disable test interval.
2. Disable save interval.
3. Run exactly 300 train iterations.
4. Use `/usr/bin/time` or parse train progress speed from logs.
5. Write a small speed summary to:

```text
checkpoints/gaussianpile/gaussianpile_liver_1m_300_speed.time
```

If speed is below target:

1. Verify the CUDA extension was rebuilt and imported from `ops/gaussianplie_renderer`.
2. Confirm the renderer is not falling back to the PyTorch reference.
3. Profile tile binning and raster kernels separately.
4. Reduce diagnostics in the training loop.
5. Check culling thresholds and footprint radius to avoid excessive per-pixel work.
6. Avoid depth sorting and alpha compositing entirely.
7. Keep additive tile accumulation memory coalesced and avoid atomics where tile-local pixel ownership can be used.
8. Compare against `ours_renderer` speed scripts for approximate environment baseline.

Do not start the final 30k acceptance run until the 1M speed benchmark is at or above target.

## 10. Final 30k Experiment

Primary config:

```text
config_files/experiments/gaussianpile/liver_1m_30k_dc.yaml
```

Expected settings:

```text
dataset / scene: synthetic_liver via config_files/_base_/scene/liver.yaml
split: holdout_interval
initial Gaussians: 1_000_000
iterations: 30_000
density control: enabled
density start: 500
density stop: 25_000
loss: L1 + SSIM-style term with paper lambda 0.2 where repository semantics allow
metrics: PSNR, L1, SSIM, LPIPS
output root: checkpoints/gaussianpile
exp_tag: gaussianpile_liver_1m_30000_dc
```

After completion, collect:

```text
Slurm job id
Slurm script path
stdout log path
stderr log path
config path
checkpoint directory
logs/scalars.tsv
final PSNR
final L1
final SSIM
final LPIPS
final Gaussian count
speed / it/s
visualization path
```

Write or verify:

```text
checkpoints/gaussianpile/<experiment_name>/run_manifest.yaml
```

The manifest must include:

```yaml
method_name: gaussianpile
experiment_name: gaussianpile_liver_1m_30000_dc
experiment_namespace: gaussianpile
config: config_files/experiments/gaussianpile/liver_1m_30k_dc.yaml
git_commit: <commit or dirty-worktree note>
slurm_job_id: <job id>
slurm_script: scripts/methods/baselines/gaussianpile/slurm/07_liver_1m_30k_dc.sh
dataset: synthetic_liver
split: holdout_interval
seed: <configured seed or inherited default>
resolution_or_scale: <dataset H/W and spacing>
train_steps: 30000
final_metrics:
  psnr: <value>
  ssim: <value>
  lpips: <value>
  l1: <value>
  speed_it_s: <value>
  final_gaussian_count: <value>
logs:
  stdout: logs/gaussianpile/<job>.out
  stderr: logs/gaussianpile/<job>.err
notes: <known caveats>
```

## 11. Result Report

Create:

```text
docs/methods/baselines/gaussianpile/reproduction_results.md
```

The report should include:

1. Plan source:
   - `docs/methods/baselines/gaussianpile/reproduction_plan.md`
   - this Codex plan
2. Implementation scope:
   - method files
   - renderer files
   - config files
   - scripts
   - tests
   - integration files
3. Renderer math:
   - focus reparameterization
   - determinant normalization
   - additive rasterization
4. Validation:
   - reference tests
   - CUDA forward tests
   - CUDA backward tests
   - smoke run
   - speed run
5. Final experiment:
   - job id
   - checkpoint path
   - scalars path
   - visualization path
   - final metrics
6. Acceptance gate status:
   - 1M initial Gaussians
   - 30k iterations
   - DC enabled
   - 100+ it/s
7. Known issues:
   - compression not implemented unless separately requested
   - voxelizer not implemented unless separately requested
   - any numerical limitations in CUDA backward
8. Remaining risks and next steps.

## 12. Delivery Checklist

Before final delivery, run the minimum relevant checks:

```bash
pytest -q tests/methods/gaussianpile
pytest -q tests/ops/gaussianplie_renderer
```

For CUDA changes, run through Slurm after rebuilding:

```bash
sbatch scripts/methods/baselines/gaussianpile/slurm/02_rebuild_renderer.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/03_forward_cuda.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/04_backward_cuda.sh
```

For experiment validation:

```bash
sbatch scripts/methods/baselines/gaussianpile/slurm/05_liver_smoke.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/06_speed_1m_300.sh
sbatch scripts/methods/baselines/gaussianpile/slurm/07_liver_1m_30k_dc.sh
```

Final local checks:

```bash
git diff --name-only
git status --short
git diff --check
```

Final response must report:

1. Changed files.
2. Main implementation summary.
3. Tests and Slurm jobs run.
4. Final metrics and speed.
5. Checkpoint, scalars, logs, and report paths.
6. Boundary check result.
7. Running or pending Slurm jobs, if any.
8. Remaining risks.
