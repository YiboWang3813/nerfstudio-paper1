# NeRF-US No-Prior Baseline Reproduction Plan

Legacy note: this document preserves the original reproduction plan and may quote upstream `--config` commands or old project `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/` and should be launched with `python run.py experiment=preproduction/baseline_reproduction/nerfus/<alias>`.

日期：2026-06-04

目标：在 `omniultra` 中实现一个 **NeRF-US no-prior baseline**，用于 baseline 对比。该 baseline 只保留 NeRF-US / Ultra-NeRF 共有的 ultrasound-specific NeRF 表示和渲染路径，**明确不实现、不训练、不加载 diffusion prior**。

本文基于：

- 原始论文：`docs/literature/papers/nerfus_dagli24a.pdf`
- 作者开源代码：`/home/boywang25/ultrasound-scene-reconstruction/third_parties/nerf-us`
- 作者仓库 commit：`e6e4ead081389c0b5d2069dfddd6c419943ac131`
- 本项目参考实现：`methods/ultranerf/**`

## 0. Scope Decision

本计划不做完整 NeRF-US。原因：

1. 作者仓库没有提供 fine-tuned NeRF-US diffusion checkpoint。
2. 仓库只提供了 Nerfbusters base checkpoint 的下载入口，代码会下载 `pt/nbusters.ckpt` 作为 adaptation 起点，但不是论文中 synthetic-knee LoRA fine-tune 后的 prior。
3. 作者 config 中引用的 `outputs/diffusion/cubes_shapenet/ddpm-fulldata/checkpoints/synkn.ckpt` 不在仓库内。
4. 训练 diffusion 需要额外 synthetic knee / phantom voxel cube 数据、旧 PyTorch / Nerfstudio / Lightning 环境和 Slurm GPU 时间。
5. 本项目当前需求是 baseline 对比，不值得为这个 baseline 投入独立 diffusion training / adapter 工程。

因此后续实现只做：

```text
3D point -> MLP -> [alpha, beta, rho_b, rho_s, phi] -> ultrasound renderer -> image loss
```

这在方法结构上等价于 Ultra-NeRF 的一个 tuned variant。报告、配置和结果命名必须明确标注：

```text
nerfus_no_prior
NeRF-US without diffusion prior
```

不要把该结果称为完整 NeRF-US。

## 1. Paper Summary

NeRF-US 的完整论文方法包含两部分：

1. Ultra-NeRF 风格 ultrasound-specific NeRF renderer。
2. 3D diffusion geometry prior，用于训练期约束 `rho_b` 和 `rho_s`。

完整 NeRF-US 相对 Ultra-NeRF 的真正新增点是第 2 点。删除 diffusion 后，剩余主体与 Ultra-NeRF 高度一致：

- MLP 输入 3D point。
- 输出 ultrasound parameters。
- 沿 ultrasound scan-line 做 attenuation / reflection / scattering 渲染。
- 用 rendered B-mode image 和 ground truth frame 做 photometric supervision。

所以本计划的 baseline 价值是：

- 作为 “NeRF-US paper hyperparams but no diffusion prior” 的参考点。
- 检查 NeRF-US 中非 diffusion 部分是否能在本项目数据和训练框架下稳定运行。
- 与 `methods/ultranerf` 做公平对照，避免把没有公开权重的 diffusion 部分变成过重工程。

## 2. What We Keep

### 2.1 NeRF Field

论文 NeRF field：

```text
F_theta(q) -> [alpha(q), beta(q), rho_b(q), rho_s(q), phi(q)]
```

参数含义：

- `alpha`: attenuation。
- `beta`: reflectance。
- `rho_b`: border probability。
- `rho_s`: scattering density。
- `phi`: scattering intensity。

论文 MLP 设置：

- 输入：3D point `q=(x,y,z)`。
- 不使用 view direction branch。
- 使用 NeRF positional encoding。
- 8 layers。
- 256 hidden units。
- ReLU。
- 第 5 层 skip connection。
- 输出 5 个 ultrasound parameters。

输出 activation：

- `alpha`: positive continuous value，论文使用 absolute value。
- `beta`, `rho_b`, `rho_s`, `phi`: sigmoid 到 `[0, 1]`。

本项目 `methods/ultranerf/field.py` 已经有同语义实现。`methods/nerfus` 可以复制该风格，但默认宽度改成 `256`，以匹配 NeRF-US 论文。

### 2.2 Ultrasound Rendering

保留 ultrasound-specific rendering。沿每条 scan-line，field 输出参数通过 attenuation、reflection、boundary mask、scattering map 和 PSF 合成 B-mode intensity。

推荐实现沿用 `methods/ultranerf/render.py` 的形式：

```text
reflection_transmission = exclusive_cumprod(1 - beta * border_indicator)
attenuation_transmission = exp(exclusive_cumsum(-alpha * dist * attenuation_scale))
transmission = reflection_transmission * attenuation_transmission

reflected = transmission * beta * PSF(border_indicator)
scatter_map = scatter_indicator * phi
backscatter = transmission * PSF(scatter_map)

pred = reflected + backscatter
```

保留 renderer config：

```text
boundary_sampling_mode: expectation | bernoulli | relaxed_bernoulli
scatter_sampling_mode: expectation | bernoulli | relaxed_bernoulli
eval_render_mode: expectation | bernoulli | mc_average
psf_kernel_size
psf_sigma_lateral
psf_sigma_axial
attenuation_scale
output_gain
```

首版训练建议使用 `expectation`，降低 stochastic rendering 方差；eval 可以使用 `mc_average` 与 UltraNeRF 当前风格对齐。

### 2.3 Loss

完整论文 loss 是：

```text
photometric_loss + lambda_rho_b * diffusion_border_loss + lambda_rho_s * diffusion_scatter_loss
```

本 baseline 删除 diffusion 后只保留：

```text
photometric_loss
```

NeRF-US 论文附录说加入 prior 后不使用 SSIM loss，photometric MSE 足够。由于本计划没有 prior，baseline 可保留两套 alias：

1. `nerfus/smoke`：MSE only，贴近 NeRF-US paper no-prior setting。
2. `nerfus/ultranerf_loss` 或 override：MSE+SSIM，用于和当前 `ultranerf` 配置做更接近的公平对照。

最终报告必须写清楚 loss 配置，避免把 loss 差异误认为方法差异。

## 3. What We Drop

明确不实现：

- `methods/nerfus/prior.py`
- diffusion checkpoint adapter
- `NerfusTrainer` checkpoint loading
- 3D UNet / DDPM scheduler migration
- LoRA fine-tuning
- synthetic knee voxel cube generation
- diffusion-guided `rho_b` / `rho_s` loss
- `prior.enabled=true`
- prior ablation

明确不运行：

```bash
python nerfus/run.py --config config/synthetic-knee.yaml --name synthetic-knee-experiment --pt
```

明确不依赖：

```text
pt/nbusters.ckpt
outputs/diffusion/**/synkn.ckpt
```

作者仓库中 diffusion 相关代码只作为论文考古记录，不进入本项目实现路线。

## 4. Author Code Audit

作者 repo 中与 no-prior baseline 相关的可参考内容：

- `nerfus/nerf/nerfus_config.py`
  - field 默认 `hidden_dim=256`
  - `hidden_layers=8`
  - `num_samples=64`
  - `probe_depth=0.14`
  - `probe_width=0.08`
  - `train_num_rays_per_batch=4096`
- `nerfus/nerf/nerfus_pipeline.py`
  - 有 border / scatter loss multiplier config，但本计划不使用。

不应照搬：

- `nerfus_pipeline.py` 的 simplified `intensity = sigmoid(features[..., 0])` 路径。它没有完整实现论文 ultrasound rendering。
- Nerfstudio `VanillaPipeline` / `DataParser` 层级。
- 旧 diffusion CLI `--config`。
- 对 `encoded_positions` reshape 成 diffusion cube 的逻辑。

本项目实现应以 `methods/ultranerf` 为主参考，而不是作者 Nerfstudio 插件。

## 5. Experimental Settings From Paper

论文数据集：

1. `Ultrasound in the Wild`
   - human knee ultrasound。
   - Butterfly iQ+ handheld ultrasound device。
   - 10 casual sweeps。
   - 30 FPS。
   - 每个 sweep 至少 85 frames。
   - COLMAP registration。
   - every 8th frame as test。

2. `phantom dataset`
   - 来自 Ultra-NeRF / Wysocki et al. 2024。
   - human spine phantom。
   - 非 in-the-wild capture。

论文 metrics：

- PSNR。
- SSIM / MS-SSIM。
- LPIPS。

论文主结果中完整 NeRF-US：

```text
Ultrasound in the Wild:
NeRF-US    PSNR 27.37 +/- 1.83, SSIM 0.8412 +/- 0.13, LPIPS 0.2325 +/- 0.14
Ultra-NeRF PSNR 24.14 +/- 2.54, SSIM 0.8312 +/- 0.15, LPIPS 0.2573 +/- 0.16

phantom:
NeRF-US    PSNR 29.24 +/- 2.13, SSIM 0.5306 +/- 0.05, LPIPS 0.1963 +/- 0.13
Ultra-NeRF PSNR 28.12 +/- 2.35, SSIM 0.5314 +/- 0.09, LPIPS 0.2182 +/- 0.10
```

注意：这些数字包含 diffusion prior，不应作为 no-prior baseline 的验收目标。

论文 NeRF hyperparameters：

```text
iterations: 300K
batch_size: 2^12 rays
optimizer: RAdam
betas: beta1=0.9, beta2=0.999
eps: 1e-8
lr: 5e-4 -> 5e-5 exponential decay
MLP: 8 layers, 256 hidden units
PE frequencies: NeRF positional encoding
```

本 baseline 优先复用这些 NeRF 超参，不复用 diffusion 超参。

## 6. OmniUltra Implementation Plan

### Phase 0: Method Scaffold

新增 method-owned 文件：

```text
methods/nerfus/
  __init__.py
  configs.py
  field.py
  render.py
  solver.py
  state.py
  offline_callbacks.py
configs_hydra/model/nerfus.yaml
configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/smoke.yaml
tests/methods/nerfus/
docs/methods/baselines/nerfus/
```

不新增：

```text
methods/nerfus/prior.py
methods/nerfus/diffusion.py
ops/nerfus_renderer/**
```

边界：

- 普通 method agent 默认可修改 `methods/nerfus/**`、`configs_hydra/model/nerfus.yaml`、`configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/**`、`tests/methods/nerfus/**`、`scripts/methods/baselines/nerfus/**`、`docs/methods/baselines/nerfus/**`。
- 如果需要 `run.py`、`configs/base/models.py` 注册，作为最小 integration step 处理。
- 不修改 `methods/ultranerf/**`，只参考其风格。

### Phase 1: Field / Renderer

实现：

1. `NerfusField`
   - 复制 `UltraNeRFField` 结构。
   - 默认 `depth=8`、`width=256`、`skip_layer=4`、`pe_num_freqs=10`。
   - 输出 keys: `alpha`, `beta`, `rho_b`, `rho_s`, `phi`, `raw`。

2. `render_slice`
   - 复制 `methods/ultranerf/render.py` 的 ultrasound renderer。
   - 保留 expectation / bernoulli / relaxed sampling mode。
   - 保留 PSF。
   - 保留 debug maps。

3. `NerfusSolver`
   - 默认 `optimizer=radam`。
   - `lr=5e-4`。
   - exponential decay target `5e-5` over production iterations。
   - 支持 `adam` fallback，方便和 UltraNeRF 对照。

### Phase 2: Offline Training Callback

参考 `methods/ultranerf/offline_callbacks.py`：

1. `NerfusOfflineSetupCallback`
   - build scene。
   - build field。
   - build solver。
   - setup checkpoint items。

2. `NerfusOfflineTrainCallback`
   - random patch sampling。
   - render patch。
   - photometric loss。
   - optimizer step / lr step。
   - train render stats。

3. `NerfusOfflineTestCallback`
   - full image eval。
   - metrics。
   - save compose / debug maps。

4. `NerfusOfflineFinishCallback`
   - write manifest。
   - close scalar manager。

Loss policy:

- Default `mse` only for `nerfus/smoke` and paper-hyperparam runs.
- Optional MSE+SSIM only for controlled comparison with current `ultranerf` setup.

### Phase 3: Hydra Configs

`configs_hydra/model/nerfus.yaml`:

```yaml
# @package _global_

model:
  name: nerfus
  field:
    depth: 8
    width: 256
    pe_num_freqs: 10
    include_input: true
    skip_layer: 4
    use_viewdirs: false
    alpha_activation: abs
    alpha_bias_init: 0.0
  ray:
    coordinate_unit: mm
    coordinate_scale: 100.0
    train_patch_height: 128
    train_patch_width: 128
    eval_chunk_size: 65536
  render:
    boundary_sampling_mode: expectation
    scatter_sampling_mode: expectation
    eval_render_mode: mc_average
    eval_mc_samples: 4
    eval_seed: 20260604
    eval_save_stochastic: true
    enable_psf: true
    psf_kernel_size: 7
    psf_sigma_lateral: 2.0
    psf_sigma_axial: 1.0
    attenuation_scale: 0.01
    eps: 1.0e-6
    output_mode: clamp
    output_gain: 1.0
    relaxed_temperature: 0.1
  solver:
    optimizer: radam
    lr: 5.0e-4
    lr_final: 5.0e-5
    lr_decay_steps: 300000
```

`configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/smoke.yaml`:

```yaml
# @package _global_

defaults:
- /engine: debug_offline
- /scene: liver
- /model: nerfus
- /loss: default
- /metric: default
- /system: default
- _self_

system:
  exp_tag: nerfus_no_prior_smoke

engine:
  state: NerfusOfflineState
  iterations: 500
  needed_callbacks:
  - NerfusOfflineSetupCallback
  - NerfusOfflineTrainCallback
  - NerfusOfflineTestCallback
  - NerfusOfflineSaveCallback
  - NerfusOfflineFinishCallback

loss:
  needed_losses:
  - mse
  lambdas:
    mse: 1.0

metric:
  needed_metrics:
  - psnr
  - l1
  - ssim
  - lpips

experiment:
  family: nerfus
  name: smoke
  description: NeRF-US no-prior smoke baseline.
```

Production aliases can be added only after smoke works:

```text
configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/fullres_30k_no_prior.yaml
configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/fullres_100k_no_prior.yaml
```

Do not create `fullres_300k` until 30k / 100k behavior is worth scaling.

### Phase 4: Tests

Add focused tests:

```text
tests/methods/nerfus/test_field.py
tests/methods/nerfus/test_render.py
tests/methods/nerfus/test_config_smoke.py
```

Minimum assertions:

- field output keys and shapes are correct。
- `alpha` finite and non-negative。
- sigmoid outputs are in `[0, 1]`。
- renderer output shape matches target patch。
- renderer outputs finite values。
- expectation mode is differentiable。
- fixed-seed stochastic eval is reproducible。
- Hydra config can build model dataclass and dry-run。

### Phase 5: Slurm Scripts

Only after CPU dry-run and tests pass, add:

```text
scripts/methods/baselines/nerfus/slurm/run_nerfus.sh
scripts/methods/baselines/nerfus/slurm/submit_no_prior_baseline.sh
```

Runner must use Hydra:

```bash
python run.py experiment="${EXPERIMENT}" ${OVERRIDES}
```

No old commands:

```bash
python run.py --config ...
CONFIG=config_files/...
python nerfus/run.py --config ...
ns-train nerfus ...
```

## 7. Baseline Comparison Design

Recommended comparison:

```text
ultranerf_current
nerfus_no_prior_mse
nerfus_no_prior_ultranerf_loss
```

Controls:

- Same dataset。
- Same split。
- Same frame resolution。
- Same iteration budget。
- Same eval protocol。
- Same metrics。
- Record field width, optimizer, lr, loss.

This lets us answer:

1. Does NeRF-US paper MLP width / optimizer / MSE-only setup matter without diffusion?
2. Is any observed gain just UltraNeRF hyperparameter tuning?
3. Is this baseline worth keeping in production comparisons?

Suggested output:

```text
outputs/production/comparisons/nerfus_no_prior_vs_ultranerf/
outputs/production/comparisons/nerfus_no_prior_vs_ultranerf/tables/metrics.tsv
docs/methods/baselines/nerfus/no_prior_comparison.md
```

## 8. Validation Gates

Config:

```bash
python run.py experiment=preproduction/baseline_reproduction/nerfus/smoke task=dry_run system.device=cpu engine.iterations=1
```

Tests:

```bash
PYTHONPATH=. pytest -q tests/methods/nerfus
```

Slurm scripts:

```bash
bash -n scripts/methods/baselines/nerfus/slurm/run_nerfus.sh
bash -n scripts/methods/baselines/nerfus/slurm/submit_no_prior_baseline.sh
```

Regression checks:

```bash
rg "config_files|--config|legacy_config" \
  --glob '!docs/legacy/**' \
  --glob '!git history**' \
  --glob '!docs/methods/**'
```

GPU training / eval:

- Use Slurm.
- Use `gaussian_splatting` unless explicitly documenting a separate env.
- Report job id, output root, Hydra run dir, logs, final metrics.

## 9. Risks

1. Method identity
   - Without diffusion, this is not complete NeRF-US.
   - All configs and reports must say `no_prior`.

2. Fairness
   - Differences against UltraNeRF may come from width, optimizer, lr, loss, sampling mode, not a fundamentally new method.
   - Keep controlled aliases.

3. Compute
   - Width 256 may be slower than current UltraNeRF width 128.
   - Start with 128 patch smoke, then scale.

4. Dataset mismatch
   - Paper's gains are on in-the-wild knee data.
   - Existing project data may be synthetic liver / phantom-like, so absolute paper metrics are not expected.

5. Author code mismatch
   - Author open-source NeRF pipeline has simplified rendering and missing adapted diffusion checkpoint.
   - Treat it as reference only, not canonical implementation.

## 10. Immediate Next Tasks

1. Scaffold `methods/nerfus` without `prior.py` or diffusion code.
2. Copy UltraNeRF-style field / renderer and set NeRF-US paper hyperparams.
3. Add `configs_hydra/model/nerfus.yaml` and `configs_hydra/experiment/preproduction/baseline_reproduction/nerfus/smoke.yaml`.
4. Add focused unit tests.
5. Run CPU dry-run and pytest.
6. If smoke passes, add Slurm runner and run no-prior baseline.
7. Compare against current UltraNeRF under matched dataset / split / iteration / metrics.
