# Ultra-NeRF Reproduction Landing Plan

Legacy note: this document preserves the original reproduction plan and may mention old `config_files/...` paths or `python run.py --config ...` commands. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/ultranerf/` and should be launched with `python run.py experiment=preproduction/baseline_reproduction/ultranerf/<alias>`.

日期：2026-05-27

目标：在 `omniultra` 中复现 Ultra-NeRF，用 NeRF/INR 结构在 `synthetic_liver` 数据集上做超声图像重建。最终验收实验使用原图分辨率 `512 x 256`，`images.npy` 共 200 帧，按项目现有 `holdout_interval` 规则划分 160 张训练、40 张测试，训练 `30000` iterations，目标测试指标为 `PSNR > 20`，`SSIM ~= 0.5`，并保存可复查的渲染图、指标和 manifest。

## 1. 已读材料与关键结论

### 1.1 项目内计划

已阅读：

- `docs/operations/development_skills.md`
- `docs/operations/git_workflow_skills.md`
- `docs/operations/scope_skills.md`
- `docs/methods/baselines/ultranerf/reproduction_plan.md`

关键结论：

1. Ultra-NeRF 是隐式 MLP field，不使用 Gaussian container、density control 或 CUDA renderer。
2. MLP 输入为 3D query point，输出 5 个物理参数：
   - `alpha`: attenuation coefficient，非负。
   - `beta`: reflection coefficient，`sigmoid` 到 `[0, 1]`。
   - `rho_b`: border probability，`sigmoid` 到 `[0, 1]`。
   - `rho_s`: scattering density，`sigmoid` 到 `[0, 1]`。
   - `phi`: scattering amplitude，`sigmoid` 到 `[0, 1]`。
3. 论文正文强调 tissue property 是 isotropic，首版不应使用 viewing direction branch。方向只用于生成 scan-line rays，不作为 MLP 输入。
4. 训练第一版应使用 deterministic expectation：
   - `G = rho_b`
   - `H = rho_s`
   这样 renderer 完全可微，避免 Bernoulli 采样造成高方差或不可导问题。
5. loss 和 metric 应接入项目现有 `engine.computer.LossComputer` 和 `MetricComputer`，不要在 UltraNeRF renderer 里写死 loss。

### 1.2 现有 `methods/ultranerf` 状态

当前已有文件：

- `methods/ultranerf/args.py`
- `methods/ultranerf/nerf_model.py`
- `methods/ultranerf/render.py`
- `methods/ultranerf/offline_runner.py`
- `methods/ultranerf/__init__.py`

问题：

1. `offline_runner.py` 引用 `common.engine.base_runner`、`common.scene.factory`、`common.utils.metric_utils`，这些不是当前 `omniultra` 框架路径，无法直接跑。
2. 当前 `NeuralRadianceField` 默认 `use_viewdirs=True`，与计划和论文中的 isotropic field 不一致。
3. 当前 renderer 已有物理公式雏形，但默认使用 Gumbel/Bernoulli hard sample。验收首版应改成 expectation mode。
4. 当前 UltraNeRF 尚未注册到：
   - `configs/base/models.py`
   - `run.py`
   - `config_files/_base_/models/ultranerf.yaml`
   - `config_files/experiments/ultranerf/*.yaml`
5. 因此现有代码只能作为公式和结构参考，需要按当前 engine callback/state 风格重建 method 内实现。

### 1.3 官方 TensorFlow 实现结论

参考路径：

- `/home/boywang25/ultrasound-scene-reconstruction/third_parties/ultra-nerf`

关键文件：

- `run_ultra_nerf.py`
- `run_nerf_helpers.py`
- `load_us.py`
- `conf_us.txt`

可复用设置：

1. 官方 synthetic liver 几何：
   - depth `140 mm`
   - width `80 mm`
   - image `H=512, W=256`
   - linear probe
2. 官方 ray generation `get_rays_us_linear`：
   - 每列一个 scan-line origin。
   - lateral origin 位于 `[-W/2, W/2] * spacing_width`。
   - ray direction 为 probe pose 旋转后的深度方向。
3. 官方默认训练参数：
   - `netdepth=8`
   - `netwidth=128`
   - `N_samples=64`
   - `N_rand=32*32*4=4096`
   - `lrate=1e-4`
   - `lrate_decay=250`
   - `loss=ssim`
   - `ssim_lambda=0.75`
   - README 示例使用 `n_iters=200000`
4. 官方 renderer：
   - `alpha = abs(raw[..., 0])`
   - `beta = sigmoid(raw[..., 1])`
   - `rho_b = sigmoid(raw[..., 2])`
   - `rho_s = sigmoid(raw[..., 3])`
   - `phi = sigmoid(raw[..., 4])`
   - `attenuation_transmission = cumprod(exp(-alpha * dists), exclusive=True)`
   - `reflection_transmission = cumprod(1 - beta * border_indicator, exclusive=True)`
   - `b = transmission * PSF(scatterers * phi)`
   - `r = transmission * beta * PSF(border_indicator)`
   - final image `intensity_map = b + r`
5. 官方 PSF 是简单 `7 x 7` 2D Gaussian kernel，`g_size=3`。

### 1.4 第三方 PyTorch 实现结论

参考路径：

- `/home/boywang25/ultrasound-scene-reconstruction/third_parties/ultra-nerf-pytorch`

关键文件：

- `nerf/model.py`
- `render/render.py`
- `render/ray.py`
- `render/device.py`
- `train_model.py`

可复用设置：

1. `SyentheticLiverProbe` 使用：
   - `near=0`
   - `far=0.14`
   - `width=0.08`
   - `h=512`
   - `w=256`
2. PyTorch 复现也使用 8 层 MLP、width 256、skip connection、5 维输出。
3. `RayBundleLinear` 的点采样方式与项目 `LinearSliceMeta` 的概念匹配，可以迁移成当前 repo 内纯函数，而不是照搬外部类层级。
4. PyTorch 复现训练是 epoch over frames，不完全符合当前 engine 的 iteration 语义；在 `omniultra` 内应使用 `engine.iterations=30000`。

## 2. Agent 边界

本任务是 UltraNeRF method agent，默认允许修改：

- `methods/ultranerf/**`
- `tests/methods/ultranerf/**`
- `config_files/_base_/models/ultranerf.yaml`
- `config_files/experiments/ultranerf/**`
- `scripts/methods/baselines/ultranerf/**`
- `docs/methods/baselines/ultranerf/*.md`
- `docs/methods/baselines/ultranerf/*.md`

不涉及 CUDA 自定义算子，因此不新增 `ops/ultranerf_renderer/**`。

本任务已获得用户明确授权，可以直接做 UltraNeRF 所需的最小 integration 修改：

- `run.py`
- `configs/base/models.py`

原因：

1. UltraNeRF 当前未注册到主框架。
2. 要用 `python run.py --config ...` 正式训练，必须在 `configs/base/models.py` 注册 `UltraNeRFConfig`，并在 `run.py` import `methods.ultranerf.state` 和 `methods.ultranerf.offline_callbacks` 完成 state/callback 注册。

计划执行时，不需要在修改上述两个共享注册文件前暂停询问；按第 7 节执行最小共享文件改动即可。授权范围仅限 UltraNeRF 注册和 callback/state import，不包含修改 engine、scene、gaussian、utils、ops/common 或其他 method 的核心实现。

## 3. 验收实验定义

### 3.1 数据和 split

使用项目现有 `synthetic_liver` 数据：

- base scene config: `config_files/_base_/scene/liver.yaml`
- dataset root: `/home/boywang25/ultrasound-scene-reconstruction/data`
- dataset name: `synthetic_liver`
- sweep: `l1`
- image file: `images.npy`
- pose type: `relative`
- pose file: `relative_poses.npy`
- full resolution: `512 x 256`
- total frames: `200`

使用与 `ours` 相同 holdout split：

```yaml
scene:
  loading:
    test_mode: "holdout_interval"
    holdout_trim_each: 10
    holdout_num_samples: 40
```

当前 `Scene._build_holdout_indices` 会在 `[10, 189]` 上等间隔取 40 张测试帧，其余 160 张为训练帧。

### 3.2 训练目标

正式验收配置：

- experiment namespace: `ultranerf`
- config: `config_files/experiments/ultranerf/liver_fullres_30k_holdout.yaml`
- output tag: `ultranerf_liver_fullres_30k_holdout`
- iterations: `30000`
- train frames: `160`
- test frames: `40`
- resolution: `512 x 256`
- near/far: `0 mm / 140 mm`
- width: `80 mm`
- samples per ray: 第一版使用 one sample per pixel depth，即完整 image grid `512 x 256`；若显存不足，用 random ray batch 训练、full image eval。
- metrics: `psnr`, `l1`, `ssim`, `lpips`
- expected threshold: final test `psnr > 20`, `ssim ~= 0.5`

### 3.3 推荐输出目录

训练输出：

```text
checkpoints/ultranerf/ultranerf_liver_fullres_30k_holdout/
```

日志输出：

```text
logs/ultranerf/
```

报告输出：

```text
docs/methods/baselines/ultranerf/reproduction_results.md
```

## 4. 实现阶段计划

### Phase 0: 建立基线审计

目标：确认当前 UltraNeRF prototype 的不可运行点和可复用点，避免盲改。

步骤：

1. 检查当前文件列表：

   ```bash
   rg --files methods/ultranerf
   rg --files config_files/experiments/ultranerf config_files/_base_/models scripts/methods/baselines/ultranerf tests/methods/ultranerf
   ```

2. 记录现有 `methods/ultranerf` 中可保留的逻辑：
   - positional encoding。
   - 8-layer MLP。
   - 5-parameter output activation。
   - PSF convolution。
   - attenuation/reflection/backscatter composition。

3. 删除或替换不适配当前 framework 的 `offline_runner.py` 依赖路径，不继续维护 `common.*` runner。

交付：

- `methods/ultranerf` 结构调整清单。
- 确认不会修改其他 method。

### Phase 1: Method 包重构

目标：把 UltraNeRF 改成当前 engine 可接入的 method 包。

建议文件结构：

```text
methods/ultranerf/
  __init__.py
  configs.py
  state.py
  field.py
  ray_sampler.py
  render.py
  solver.py
  offline_callbacks.py
```

职责：

1. `configs.py`
   - 定义 `ModelConfig`，`name="ultranerf"`。
   - 子配置建议：
     - `FieldConfig`
     - `RayConfig`
     - `RenderConfig`
     - `SolverConfig`
     - `OfflineConfig`
   - 所有 YAML 字段必须在 dataclass 里有默认值。

2. `state.py`
   - 注册 `UltraNeRFOfflineState`。
   - 持有：
     - `scene`
     - `field`
     - `solver`
     - `loss_computer`
     - `metric_computer`

3. `field.py`
   - 实现 `UltraNeRFField`。
   - 输入 shape: `(..., 3)`。
   - 输出 dict 或 named tensor：
     - `alpha`
     - `beta`
     - `rho_b`
     - `rho_s`
     - `phi`
   - 默认：
     - depth `8`
     - width `128`，先对齐官方 TF；如 30k 质量不足，再开 ablation 到 `256`。
     - PE `L=10`
     - skip layer `4`
     - no direction branch。
   - activation：
     - 首版用 `softplus(raw_alpha)` 或 `abs(raw_alpha)` 做成 config。
     - `beta/rho_b/rho_s/phi = sigmoid(raw)`。

4. `ray_sampler.py`
   - 从 `SliceMeta` 和 `DatasetMeta` 生成 linear probe query grid。
   - 使用项目 meta：
     - `near_mm=0`
     - `far_mm=140`
     - `width_mm=80`
     - `H=512`
     - `W=256`
   - 输出：
     - `points_world`: `(H, W, 3)` 或 sampled `(N, 3)`。
     - `depths_mm`: `(H, W)` 或 sampled `(N,)`。
     - `image_indices`: sampled pixel 坐标，用于从 GT image gather target。
   - 必须确认坐标 convention 与 `LinearSliceMeta.initialize_from_image_pose` 一致。优先复用 `slice_meta` 中已有 grid/pose 字段；如果没有现成字段，再实现官方 `get_rays_us_linear` 等价逻辑。

5. `render.py`
   - 实现纯 PyTorch renderer。
   - 支持两种训练路径：
     - `render_full_slice`: full image `(512, 256)`，用于 eval 和 debug。
     - `render_sampled_pixels`: random rays/pixels，降低训练显存和时间。
   - 首版 sampling mode：
     - `boundary_sampling_mode="expectation"`
     - `scatter_sampling_mode="expectation"`
   - 可选后续模式：
     - `bernoulli`
     - `relaxed_bernoulli`
   - PSF：
     - `enable_psf=true`
     - `kernel_size=7`
     - `sigma_lateral=1.0`
     - `sigma_axial=0.5` 或官方 TF 单 sigma variant。
   - 输出：
     - `pred_image` 或 sampled `pred_pixels`。
     - `render_info`，包含 `alpha/beta/rho_b/rho_s/phi/transmission/r/b` 的统计或 debug map。

6. `solver.py`
   - Adam optimizer。
   - 默认 `lr=1e-4`，对齐官方。
   - exponential decay：
     - `lr_decay_steps=250000` 或配置成 `250 * 1000`。
     - 30k 训练中衰减较轻。
   - 保存/加载 field 和 optimizer state。

7. `offline_callbacks.py`
   - 参考 `methods/ultragauss/offline_callbacks.py` 和 `methods/ultragray/offline_callbacks.py`。
   - 注册：
     - `UltraNeRFOfflineSetupCallback`
     - `UltraNeRFOfflineTrainCallback`
     - `UltraNeRFOfflineTestCallback`
     - `UltraNeRFOfflineSaveCallback`
     - `UltraNeRFOfflineFinishCallback`
   - train：
     - `datum = state.scene.next_train_datum()`
     - sampled pixel/ray forward。
     - 写 `state.inputs["gt_image"]` 或 sampled target。
     - 写 `state.outputs["pred_image"]` 或 sampled pred。
     - 使用 `LossComputer`。
   - test：
     - 遍历 `state.scene.next_test_datum()`。
     - full image render。
     - 使用 `MetricComputer`。
     - 保存可视化图和 diff。

验收：

```bash
python - <<'PY'
from methods.ultranerf.configs import ModelConfig
from methods.ultranerf.field import UltraNeRFField
print(ModelConfig())
print(UltraNeRFField() )
PY
```

### Phase 2: Ray sampler 与坐标验证

目标：确保 `512 x 256` 图像中每个像素能映射到正确世界坐标，且 train/test 与 `ours` 使用同一帧。

步骤：

1. 读取 `scene/meta/dataset_meta/synthetic_liver.py`，确认 meta：
   - `H=512`
   - `W=256`
   - `near_mm=0`
   - `far_mm=140`
   - `width_mm=80`
   - `delta_lateral_mm=0.3125`
   - `delta_axial_mm=0.2734`
2. 写 `build_linear_probe_grid(meta, pose)`：
   - lateral: `linspace(-width/2, width/2, W)` 或按项目 `LinearSliceMeta` 现有定义。
   - axial/depth: `linspace(near, far, H)`。
   - local point: `(lateral, depth, 0)` 或项目 slice convention 对应坐标。
   - world point: `pose @ local_homo`。
3. 与 `LinearSliceMeta` 已有字段做一致性检查。如果 slice meta 已包含预计算 grid，优先直接用它，避免重复定义坐标。
4. 单元测试：
   - identity pose 下，横向坐标范围约 `[-40, 40] mm`。
   - depth 坐标范围约 `[0, 140] mm`。
   - full grid shape 为 `(512, 256, 3)`。
   - holdout split 下 `len(scene.train_indices)==160`，`len(scene.test_indices)==40`。

验收测试：

```bash
pytest -q -p no:cacheprovider tests/methods/ultranerf/test_ray_sampler.py
```

### Phase 3: Renderer 首版

目标：先跑通稳定、可微、能出非空图的 ultrasound renderer。

首版公式：

```text
alpha = positive(raw0)
beta = sigmoid(raw1)
G = rho_b = sigmoid(raw2)
H = rho_s = sigmoid(raw3)
phi = sigmoid(raw4)

attenuation_step = exp(-alpha * delta_depth * attenuation_scale)
attenuation_transmission = exclusive_cumprod(attenuation_step, depth_dim)
reflection_step = clamp(1 - beta * G, min=eps)
reflection_transmission = exclusive_cumprod(reflection_step, depth_dim)
transmission = attenuation_transmission * reflection_transmission

scatter_map = H * phi
psf_scatter = conv2d(scatter_map, psf_kernel)
psf_border = conv2d(G, psf_kernel)

B = transmission * psf_scatter
R = transmission * beta * psf_border
pred = normalize_or_clamp(B + R)
```

实现细节：

1. 使用 exclusive cumprod，第一行 transmission 为 1。
2. `delta_depth` 使用 mm 或 normalized coordinate 必须一致。推荐内部坐标先用 mm，然后通过 `coordinate_scale_mm=100.0` 或 scene bbox normalization 将输入归一化到 MLP 友好范围。
3. intensity normalization 做成 config：
   - `output_activation="none|clamp|scale_by_max|log_compression"`
   - 首版推荐 `clamp` 到 `[0, 1]` 前先检查是否导致梯度饱和。
4. 保留 debug stats：
   - `pred_min/max/mean`
   - `alpha_mean`
   - `beta_mean`
   - `rho_b_mean`
   - `rho_s_mean`
   - `phi_mean`
   - `transmission_min/max/mean`

单元测试：

1. field 输出 shape/range。
2. `alpha` 增大时深部 transmission 不增。
3. `beta` 和 `rho_b` 增大时 reflection transmission 不增。
4. PSF on/off 不改变输出 shape。
5. expectation renderer 可 `loss.backward()`。

验收测试：

```bash
pytest -q -p no:cacheprovider tests/methods/ultranerf/test_field.py tests/methods/ultranerf/test_render.py
```

### Phase 4: Loss/metric 接入

目标：使用现有 computer，不在 UltraNeRF 内写私有 loss。

训练 loss 推荐：

```yaml
loss:
  needed_losses: ["l1", "ssim"]
  lambdas:
    l1: 0.25
    ssim: 0.75
```

理由：

1. 官方 TF 使用 `loss=ssim` 时为 `ssim_lambda=0.75`，`l2_weight=0.25`。
2. 项目现有 `loss.yaml` 支持 `l1` 和 `ssim`，不一定已有 L2/MSE loss；首版用 L1+SSIM 更稳。
3. 若 PSNR 不足，再追加 `l2` loss 的 integration/公共 computer 修改，或改用现有 L1 权重做 ablation。

metric：

```yaml
metric:
  needed_metrics: ["psnr", "l1", "ssim", "lpips"]
```

训练 sampled pixels 时的注意事项：

1. 如果 `LossComputer` 只接受 full image，则 Phase 4 首版用 full image training，但配合 `render_chunk_size` 分块减少显存。
2. 如果要 sampled rays 训练，需要确保 loss computer 支持 sampled tensors，或在 method 内只对 sampled pred/target 计算简单 L1 后逐步接入 full image loss。这个属于风险项，优先选择不改公共 computer 的路径。

### Phase 5: 配置文件

新增：

```text
config_files/_base_/models/ultranerf.yaml
config_files/experiments/ultranerf/liver_smoke_512x256_500.yaml
config_files/experiments/ultranerf/liver_fullres_30k_holdout.yaml
```

`config_files/_base_/models/ultranerf.yaml` 建议默认：

```yaml
model:
  name: "ultranerf"
  field:
    depth: 8
    width: 128
    pe_num_freqs: 10
    include_input: true
    skip_layer: 4
    use_viewdirs: false
    alpha_activation: "abs"
  ray:
    coordinate_unit: "mm"
    coordinate_scale: 100.0
    train_num_rays: 4096
    full_image_eval: true
  render:
    boundary_sampling_mode: "expectation"
    scatter_sampling_mode: "expectation"
    enable_psf: true
    psf_kernel_size: 7
    psf_sigma_lateral: 1.0
    psf_sigma_axial: 0.5
    attenuation_scale: 1.0
    eps: 1.0e-6
    output_mode: "clamp"
  solver:
    optimizer: "adam"
    lr: 1.0e-4
    lr_decay_steps: 250000
    lr_decay_rate: 0.1
  offline:
    diagnostics:
      train_step: 100
      save_debug_maps: true
      eval_train_count: 4
```

Smoke config：

```yaml
includes:
  - "config_files/_base_/engine/debug_offline.yaml"
  - "config_files/_base_/scene/liver.yaml"
  - "config_files/_base_/models/ultranerf.yaml"
  - "config_files/_base_/loss.yaml"
  - "config_files/_base_/metric.yaml"
  - "config_files/_base_/system.yaml"

system:
  exp_tag: "ultranerf_liver_smoke_512x256_500"

scene:
  loading:
    test_mode: "holdout_interval"
    holdout_trim_each: 10
    holdout_num_samples: 40

engine:
  state: "UltraNeRFOfflineState"
  iterations: 500
  needed_callbacks:
    - UltraNeRFOfflineSetupCallback
    - UltraNeRFOfflineTrainCallback
    - UltraNeRFOfflineTestCallback
    - UltraNeRFOfflineSaveCallback
    - UltraNeRFOfflineFinishCallback
  interval:
    train:
      step: 10
      print: 50
    test: 250
    save: 500

loss:
  needed_losses: ["l1", "ssim"]
  lambdas:
    l1: 0.25
    ssim: 0.75

metric:
  needed_metrics: ["psnr", "l1", "ssim", "lpips"]
```

正式 config 在 smoke 基础上改：

```yaml
system:
  exp_tag: "ultranerf_liver_fullres_30k_holdout"

engine:
  iterations: 30000
  interval:
    train:
      step: 20
      print: 100
    test: 2500
    save: 5000
```

### Phase 6: Method 内测试

新增测试目录：

```text
tests/methods/ultranerf/
```

测试文件：

```text
tests/methods/ultranerf/test_field.py
tests/methods/ultranerf/test_ray_sampler.py
tests/methods/ultranerf/test_render.py
tests/methods/ultranerf/test_config_smoke.py
```

最小测试内容：

1. `test_field.py`
   - 输入 `(2, 3)` 和 `(4, 8, 3)`。
   - 输出 shape 正确。
   - `alpha >= 0`。
   - 其他 4 个输出在 `[0, 1]`。
   - PE 维度为 `3 + 3*10*2 = 63` when include input。

2. `test_ray_sampler.py`
   - synthetic liver meta 下 full grid shape。
   - identity pose 下 lateral/depth range。
   - sampled rays 返回 target pixel indices。

3. `test_render.py`
   - 构造常量 field output。
   - renderer 输出非 NaN。
   - backward 可执行。
   - PSF on/off shape 一致。

4. `test_config_smoke.py`
   - config dataclass import。
   - smoke YAML 字段都能在 dataclass 找到。
   - callback/state 注册名匹配。

命令：

```bash
pytest -q -p no:cacheprovider tests/methods/ultranerf
```

### Phase 7: Integration 接入

目标：让 UltraNeRF 可以通过统一入口运行。

已授权修改的共享注册文件：

- `configs/base/models.py`
- `run.py`

原因：

- `configs/base/models.py` 需要注册 `methods.ultranerf.configs.ModelConfig`。
- `run.py` 需要 import `methods.ultranerf.state` 和 `methods.ultranerf.offline_callbacks` 以触发 registry。

最小改动方案：

1. `configs/base/models.py`
   - 添加：

     ```python
     from methods.ultranerf.configs import ModelConfig as UltraNeRFModelConfig
     ```

   - registry 添加：

     ```python
     "ultranerf": UltraNeRFModelConfig,
     ```

2. `run.py`
   - 添加：

     ```python
     import methods.ultranerf.state  # noqa: F401
     import methods.ultranerf.offline_callbacks  # noqa: F401
     ```

可能影响：

- 只影响 config registry 和 callback import。
- 不改变已有方法运行逻辑。

验收：

```bash
python run.py --config config_files/experiments/ultranerf/liver_smoke_512x256_500.yaml --task train
```

用户已在 2026-05-27 明确授权本任务修改 `run.py` 和 `configs/base/models.py`，因此开发时不要在 Phase 6 后暂停询问，直接完成 Phase 7 的最小 integration 接入。

### Phase 8: Slurm 脚本

新增：

```text
scripts/methods/baselines/ultranerf/slurm/run_ultranerf_smoke.sh
scripts/methods/baselines/ultranerf/slurm/run_ultranerf_fullres_30k.sh
scripts/methods/baselines/ultranerf/slurm/eval_ultranerf_fullres_30k.sh
```

通用要求：

1. stdout/stderr：

   ```text
   logs/ultranerf/%x_%j.out
   logs/ultranerf/%x_%j.err
   ```

2. checkpoint：

   ```text
   checkpoints/ultranerf/<exp_tag>/
   ```

3. 环境：

   ```bash
   source ~/.bashrc
   conda activate gaussian_splatting
   ```

4. 不需要 `module load cuda/11.8.0` 编译扩展，但可以保留 CUDA module load 用于 GPU runtime 环境一致性。

提交前：

```bash
mkdir -p logs/ultranerf
sbatch scripts/methods/baselines/ultranerf/slurm/run_ultranerf_smoke.sh
```

正式：

```bash
sbatch scripts/methods/baselines/ultranerf/slurm/run_ultranerf_fullres_30k.sh
```

### Phase 9: Smoke 实验

目标：先确认能训练、能 eval、能保存图和指标。

配置：

- `config_files/experiments/ultranerf/liver_smoke_512x256_500.yaml`
- `iterations=500`
- `test_interval=250`
- `save=500`

检查点：

1. `.out` 中应打印：
   - train/test split `train=160, test=40`
   - field config。
   - ray config。
   - loss。
2. `.err` 无 traceback。
3. `checkpoints/ultranerf/ultranerf_liver_smoke_512x256_500/logs/scalars.tsv` 存在。
4. `vis/` 中有 test pred/gt/diff。
5. pred 不是全黑、全白或 NaN。

如果 smoke fails：

1. NaN：检查 `reflection_step` clamp、cumprod、output normalization。
2. 全黑：检查 sigmoid 初始化、PSF、attenuation scale 是否过强。
3. 图像转置：对比官方 TF 代码中 `tf.transpose(images[img_i])`，确认项目 image convention 是 `(H, W)` 且不应再转置。
4. 显存不足：降低 `train_num_rays`，eval 使用 chunked full render。

### Phase 10: 30k 正式训练

目标：满足验收指标。

配置：

- `config_files/experiments/ultranerf/liver_fullres_30k_holdout.yaml`

默认参数：

- `iterations=30000`
- `field.depth=8`
- `field.width=128`
- `pe_num_freqs=10`
- `train_num_rays=4096`
- `lr=1e-4`
- `loss l1:ssim = 0.25:0.75`
- `test_interval=2500`
- `save_interval=5000`

监控：

```bash
squeue -u boywang25
tail -n 80 logs/ultranerf/<job_name>_<job_id>.out
tail -n 80 logs/ultranerf/<job_name>_<job_id>.err
```

每次 test 记录：

- `test/psnr`
- `test/ssim`
- `test/l1`
- `test/lpips`
- train loss。
- render stats。
- sample render images。

完成后保存 manifest：

```text
checkpoints/ultranerf/ultranerf_liver_fullres_30k_holdout/run_manifest.yaml
```

manifest 至少包含：

```yaml
method_name: ultranerf
experiment_name: ultranerf_liver_fullres_30k_holdout
config: config_files/experiments/ultranerf/liver_fullres_30k_holdout.yaml
dataset: synthetic_liver
sweep: l1
split: holdout_interval
train_frames: 160
test_frames: 40
resolution: 512x256
train_steps: 30000
seed: 42
slurm_job_id: <job id>
stdout: logs/ultranerf/<job>.out
stderr: logs/ultranerf/<job>.err
final_metrics:
  psnr: <value>
  ssim: <value>
  l1: <value>
  lpips: <value>
notes: <known caveats>
```

### Phase 11: 指标不足时的 ablation 顺序

如果 30k 后 `PSNR <= 20` 或 `SSIM < 0.45`，按以下顺序做最小 ablation，每次只改一个因素。

1. Loss 权重：
   - baseline: `l1=0.25, ssim=0.75`
   - try A: `l1=0.5, ssim=0.5`
   - try B: `l1=0.1, ssim=0.9`

2. Field width：
   - baseline: `128`
   - try: `256`

3. Coordinate normalization：
   - baseline: `coordinate_scale=100.0`
   - try A: normalize scene bbox to `[-1, 1]`
   - try B: use meters as official TF (`mm * 0.001`)

4. Output intensity:
   - baseline: clamp。
   - try A: min-max scale per rendered image during loss disabled for eval。
   - try B: `softplus` final intensity then clamp。
   - try C: optional log compression。

5. Renderer stochasticity：
   - keep expectation unless visual result is too smooth。
   - only after deterministic version is stable, try Bernoulli/relaxed Bernoulli。

6. Training rays:
   - baseline: `4096` sampled pixels/iter。
   - try: `8192` if GPU memory allows。
   - try: full image every N iterations for loss refinement。

7. PSF:
   - baseline: `7x7`, lat `1.0`, axial `0.5`。
   - try official-like isotropic kernel。
   - try disable PSF for diagnosis only。

每个 ablation 写独立 config，命名：

```text
config_files/experiments/ultranerf/ablation_<factor>_<value>.yaml
```

不要覆盖正式 baseline config。

## 5. 验收 Gates

### 5.1 文档和边界

完成前必须运行：

```bash
git diff --name-only
git status --short
```

确认改动只包含 UltraNeRF method 允许范围，除非已获得 integration 授权。

### 5.2 Python 代码

必需：

```bash
pytest -q -p no:cacheprovider tests/methods/ultranerf
```

如果 integration 已完成：

```bash
python run.py --config config_files/experiments/ultranerf/liver_smoke_512x256_500.yaml --task train
```

### 5.3 正式实验

必需：

1. Slurm job id。
2. stdout/stderr 路径。
3. checkpoint 路径。
4. `scalars.tsv` 或 summary metrics。
5. final test PSNR/SSIM/L1/LPIPS。
6. `run_manifest.yaml`。
7. 至少一组 test pred/gt/diff 图。

验收阈值：

```text
test/psnr > 20
test/ssim around 0.5
```

## 6. 风险与处理

1. 当前 UltraNeRF 未注册。
   - 处理：本任务已获授权，Phase 7 直接最小修改 `run.py` 和 `configs/base/models.py`。

2. Full image training 太慢或显存不足。
   - 处理：训练用 sampled pixels/rays，eval 用 full image chunked render。

3. LossComputer 不支持 sampled tensors。
   - 处理：首版 full image loss；或 method 内训练 sampled L1、eval 用 full metrics。若要通用 sampled loss，另开 integration 修改 computer。

4. 坐标系转置导致图像质量差。
   - 处理：保存第 0 步 pred/gt/diff，比较官方 `get_rays_us_linear` 和 `LinearSliceMeta`；确认是否需要 transpose。

5. Deterministic expectation 结果过平滑。
   - 处理：先达到可见结构和 PSNR，再尝试 Bernoulli/relaxed Bernoulli。

6. 官方训练 200k iter，30k 可能不足。
   - 处理：优先优化坐标归一化、loss、width 和 sampled rays；如果仍不足，在报告中说明 30k 限制，并给出 60k/100k 后续建议。

## 7. 最终交付格式

代码与配置：

- `methods/ultranerf/**`
- `tests/methods/ultranerf/**`
- `config_files/_base_/models/ultranerf.yaml`
- `config_files/experiments/ultranerf/*.yaml`
- `scripts/methods/baselines/ultranerf/slurm/*.sh`

本任务已获授权的 integration 文件：

- `configs/base/models.py`
- `run.py`

结果报告：

```text
docs/methods/baselines/ultranerf/reproduction_results.md
```

最终汇报必须包含：

1. 改动范围。
2. 训练 config。
3. Slurm job id。
4. checkpoint 和日志路径。
5. final metrics。
6. 代表性 render 路径。
7. 是否满足 `PSNR > 20` 和 `SSIM ~= 0.5`。
8. 未解决风险和下一步建议。
