# UltraGS Baseline 复现构建计划（Codex）

日期：2026-06-02

## 1. 任务定位

本任务处在**生产前 / 探索复现阶段**，目标不是马上产出论文级正式对比，而是把论文作者开源的 UltraGS baseline 可靠地接入 `omniultra`，先在项目已有 `synthetic_liver` 数据集上跑通、验证关键模块，再进入 production alias、Slurm pipeline 和正式结果报告。

命名必须避开项目里已有且不再使用的 `ultrags` 方法。所有新增实现统一使用：

```text
methods/ultrags_baseline/**
ops/ultrags_baseline_renderer/**
configs_hydra/model/ultrags_baseline.yaml
configs_hydra/experiment/preproduction/baseline_reproduction/ultrags_baseline/**
scripts/methods/baselines/ultrags_baseline/**
tests/methods/ultrags_baseline/**
tests/ops/ultrags_baseline_renderer/**
docs/methods/baselines/ultrags_baseline/*.md
```

按 `docs/operations` 边界，这属于 `Method Agent` 任务。真正接入统一运行框架时会涉及共享注册点：

```text
run.py
configs/base/models.py
```

这两处属于 integration 范围。实现阶段如果需要修改，必须显式按 integration 步骤处理，只做注册导入和 model config registry，不改公共框架语义。

## 2. 已完成的源码获取

作者源码已递归拉取到：

```text
/home/boywang25/ultrasound-scene-reconstruction/third_parties/UltraGS
```

当前 commit：

```text
de46f9a667c2d4b5a1d99ae88ea6b4b3514a6f54
```

主要源码结构：

```text
UltraGS/
  train.py
  render.py
  metrics.py
  arguments/__init__.py
  scene/
    __init__.py
    cameras.py
    dataset_readers.py
    gaussian_model.py
  gaussian_renderer/__init__.py
  submodules/
    simple-knn/
    ultrasound-rasterization/
```

注意：仓库中没有 `.gitmodules` 内容，`git submodule status --recursive` 没有输出，但 `submodules/simple-knn` 和 `submodules/ultrasound-rasterization` 源码已经存在。

## 3. 作者代码核验结论

### 3.1 输入格式

README 要求每个 case 是 NeRF-style 格式：

```text
case_name/
  images/
  transforms_train.json
  points3d.ply   # optional
```

如果没有 `points3d.ply`，作者代码在 `scene/dataset_readers.py` 中随机初始化 `100000` 个点。

本项目当前 `synthetic_liver` 使用统一 Scene 格式：

```text
data/synthetic_liver/<sweep>/
  images.npy
  relative_poses.npy
  test_images.npy
  test_poses.npy
```

因此接入本项目时优先走 `SceneFactory` / `SingleSweepContainer` / `SliceMeta`，不要为 synthetic_liver 先反向生成作者格式。后续拿到作者 UltraGS 数据集后，再补一个 `transforms_train.json + points3d.ply` 的导入/转换脚本。

### 3.2 训练循环

作者 `train.py` 的核心设置：

```text
optimizer: Adam
default iterations: 30000
position_lr_init: 1.6e-4
position_lr_final: 1.6e-6
feature_lr: 2.5e-3
opacity_lr: 5e-2
scaling_lr: 5e-3
rotation_lr: 1e-3
fovx_lr / fovy_lr: 1e-3
lambda_dssim: 0.2
densify_from_iter: 500
densify_until_iter: 30000
densification_interval: 100
opacity_reset_interval: 3000
densify_grad_threshold: 2e-4
opacity_cull: 5e-2
```

论文/整理文档写的是 5000 iterations，作者代码默认是 30000。复现阶段应保留两个入口：

```text
smoke / debug: 低迭代、关闭或延迟 densify
paper_5k: 5000 iterations，按论文主设置
code_released_30k: 30000 iterations，用于和作者 released code 对齐
```

### 3.3 2D Gaussian 表示

作者 `GaussianModel` 使用：

```text
_xyz:          (N, 3)
_features_dc: (N, 1, 3)
_features_rest: (N, 8, 3) when sh_degree=2
_scaling:     (N, 2)
_rotation:    (N, 4)
_opacity:     (N, 1)
_fovx/_fovy:  (N, 1)
```

`_scaling` 是 2D disk scale；`_rotation` 是 3D quaternion；CUDA rasterizer 通过 `compute_transmat` 构建 2D Gaussian disk 到 image plane 的 projective transform。

本项目中应实现为：

```text
UltraGSBaselineContainer
  means
  scales_2d
  quats
  opacities
  sh_dc
  sh_rest
  fov_x
  fov_y
```

其中 `fov_x/fov_y` 先按 released code 保留参数与 optimizer group，DAR 版本再决定是否作为全局参数、逐帧参数或每 Gaussian 参数。

### 3.4 PD Rendering 与 SH degree

论文整理文档说低阶 SH 为 `l=1`，但作者代码默认：

```text
ModelParams.sh_degree = 2
```

CUDA `forward.cu::computeColorFromSH` 中：

```text
deg > 0: 标准一阶 SH 项
deg > 1: ultrasound attenuation / reflection / scattering 项
```

也就是说 released code 的 PD Rendering 实际藏在 `sh_degree=2` 的额外 5 个 SH-like 系数里，而不是独立 Python 公式：

```text
attenuation: exp(-depth)
reflection: - (x*x + y*y + z*z)
scattering: x*y, y*z, x*z cross terms
```

最终 renderer 输出 RGB 后，在 Python 侧转灰度：

```python
rendered = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
```

复现策略：

1. 第一阶段忠实迁移 released code 的 CUDA SH/PD 行为。
2. 第二阶段再做 paper-aligned 检查：是否需要显式 `I_att/I_refl/I_scat`、Softplus 正约束、`alpha=1`、`beta/Gamma` 参数化。
3. 所有偏离 released code 的 paper-aligned 改动都必须通过 ablation alias 独立命名。

### 3.5 DAR / FoV 风险

作者代码中 `_fovx/_fovy` 会被初始化、保存、加载、densify、prune，并加入 Adam：

```text
fovx_lr = 1e-3
fovy_lr = 1e-3
```

但 `gaussian_renderer/__init__.py` 中真正使用 FoV offset 的两行被注释：

```python
# tanfovx = math.tan((viewpoint_camera.FoVx + pc._fovx[0][0]) * 0.5)
# tanfovy = math.tan((viewpoint_camera.FoVy + pc._fovy[0][0]) * 0.5)
tanfovx = math.tan((viewpoint_camera.FoVx) * 0.5)
tanfovy = math.tan((viewpoint_camera.FoVy) * 0.5)
```

这与论文中的 DAR 描述冲突。复现阶段必须拆成两个模式：

```text
dar_mode: "released_disabled"
  完全复现作者当前代码，FoV 参数存在但不影响 render。

dar_mode: "global_offset"
  使用全局 theta_x/theta_y 或 FoV offset，按论文进行 FoV 优化。

dar_mode: "per_gaussian_offset"
  若确认作者意图是每 Gaussian FoV offset，再作为实验分支处理。
```

`synthetic_liver` 的第一轮 smoke 使用 `released_disabled`，避免一开始把 FoV 自由度和数据格式适配混在一起。

### 3.6 Train/test split 风险

作者 `readUSInfo` 当前逻辑：

```python
train_cam_infos = [info for idx, info in enumerate(cam_infos) if idx % 1 == 0]
test_cam_infos = [info for idx, info in enumerate(cam_infos) if idx % 8 == 0]
```

这会把 test frames 也放进 train。论文整理文档要求每 8 帧 test，其余训练。项目内复现必须使用真正 holdout：

```text
test:  frame_index % 8 == 0
train: frame_index % 8 != 0
```

`synthetic_liver` 当前已有 `interpolated_file` 和 `holdout_interval` 两种测试模式。UltraGS baseline 第一阶段建议新增/使用一个明确的 modulo-8 holdout 配置或在 method callback 内通过 Scene indices 固定该 split。

### 3.7 预处理风险

PDF 提到 anisotropic diffusion + CLAHE，作者 README 只说要先用 COLMAP / colmap2nerf 准备 pose，源码中没有找到 CLAHE 或 anisotropic diffusion 实现。现阶段：

1. `synthetic_liver` 不跑 COLMAP，直接使用已有 pose。
2. 不把 CLAHE/diffusion 混进核心方法代码。
3. 后续作者真实数据集接入时，把预处理放到：

```text
scripts/methods/baselines/ultrags_baseline/preprocess/**
```

并在报告中注明参数。

## 4. 本项目内部对齐方案

### 4.1 运行框架

本项目使用：

```text
run.py -> Hydra -> setup_experiment -> Engine -> State + Callbacks
```

UltraGS baseline 应复用现有 pattern：

```text
methods/ultragauss/offline_callbacks.py
methods/ultragray/offline_callbacks.py
methods/ultragauss/gaussian_container.py
methods/ultragray/gaussian_container.py
methods/ultragauss/gaussian_solver.py
```

不要把作者 `train.py` 作为生产入口直接复制进根目录。可以在探索期保留只读对照，但项目内运行必须是：

```bash
python run.py experiment=preproduction/baseline_reproduction/ultrags_baseline/smoke
```

### 4.2 数据适配

第一阶段数据来源：

```text
configs_hydra/scene/liver.yaml
scene.dataset_name=synthetic_liver
scene.loading.sweep_name=l1
scene.loading.pose_type=relative
```

`SingleSweepContainer` 已经给每帧 `SliceMeta`：

```text
image
pose
pixel_pos
ray_origins
ray_dirs
probe_origin
vec_lateral / vec_axial / vec_elevational
```

UltraGS rasterizer需要 virtual camera：

```text
image_height / image_width
world_view_transform
full_proj_transform
FoVx / FoVy
camera_center
```

计划新增 method-owned adapter：

```text
methods/ultrags_baseline/camera_adapter.py
```

职责：

1. 从 `SliceMeta` 的 probe pose 和 `DatasetMeta` 构造 virtual pinhole camera。
2. 将 probe 坐标系的 lateral/axial/elevational 轴映射到作者 rasterizer 的 camera convention。
3. 用 config 控制 virtual FoV 初值、near/far、axis convention。
4. 提供 debug 函数导出 camera matrix、camera center、FoV、image size。

第一版不追求 COLMAP sensorless 初始化，只用 synthetic_liver 已有 pose 验证 renderer 和训练链路。等你补 UltraGS 数据集后，再进入 COLMAP/NeRF transforms 的数据接入。

### 4.3 初始化

第一阶段直接复用 `Scene.build_init_gaussians()`：

```text
means:      sampled pixel_pos
colors:     sampled pixel intensity -> grayscale RGB/SH init
scales_2d:  config init, e.g. [0.3, 0.3]
quats:      config init or derived from slice axes
opacity:    inverse_sigmoid(0.1)
fov_x/y:    0.0 or config theta init
```

后续作者数据集路径：

```text
points3d.ply -> BasicPointCloud -> container init
```

可作为单独 loader，不要污染 `SceneFactory`。

### 4.4 Loss / metric

作者训练 loss：

```text
(1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
lambda_dssim = 0.2
```

本项目可复用 `LossComputer` 和 `MetricComputer`。`preproduction/baseline_reproduction/ultrags_baseline/smoke.yaml` 应显式设置：

```text
loss.needed_losses: [l1, ssim]
loss.lambdas.l1: 0.8
loss.lambdas.ssim: 0.2
metric.needed_metrics: [psnr, l1, ssim]
```

MSE 可在后续 production report 前加入 metric registry 或用已有 metric key 确认是否已经支持。

## 5. 实施阶段

### Phase A：源码对照与最小骨架

目标：建立 importable method 包，不写 CUDA 逻辑。

新增文件：

```text
methods/ultrags_baseline/__init__.py
methods/ultrags_baseline/configs.py
methods/ultrags_baseline/state.py
methods/ultrags_baseline/gaussian_container.py
methods/ultrags_baseline/gaussian_solver.py
methods/ultrags_baseline/render.py
methods/ultrags_baseline/camera_adapter.py
methods/ultrags_baseline/offline_callbacks.py
configs_hydra/model/ultrags_baseline.yaml
configs_hydra/experiment/preproduction/baseline_reproduction/ultrags_baseline/smoke.yaml
tests/methods/ultrags_baseline/test_config_and_container.py
```

验收：

```bash
python -m py_compile methods/ultrags_baseline/*.py
python run.py experiment=preproduction/baseline_reproduction/ultrags_baseline/smoke task=dry_run system.device=cpu engine.iterations=1
pytest -q tests/methods/ultrags_baseline/test_config_and_container.py
```

需要 integration 修改：

```text
run.py                         # import callbacks/state
configs/base/models.py         # register "ultrags_baseline"
```

这一步只做注册，不修改其他 method。

### Phase B：Renderer 包迁移

目标：把作者 `submodules/ultrasound-rasterization` 迁移为 method-owned ops。

新增：

```text
ops/ultrags_baseline_renderer/
  setup.py
  rebuild.sh
  include/ or cuda_rasterizer/
  src/
  ultrags_baseline_renderer/
    __init__.py
    api.py
    autograd.py
    types.py
tests/ops/ultrags_baseline_renderer/
  test_import.py
  test_forward_shapes.py
```

迁移原则：

1. Python 包名改为 `ultrags_baseline_renderer`。
2. Extension 名改为 `ultrags_baseline_renderer._C`。
3. 不修改 `ops/common/**`。
4. 保留作者 license/copyright 注释。
5. `rebuild.sh` 使用本项目推荐方式：

```bash
find . -name "*.so" -delete
rm -rf build *.egg-info
python -m pip install -e . --no-build-isolation -v
```

6. CUDA 编译和测试必须通过 Slurm，在 `gaussian_splatting` 环境加载 `cuda/11.8.0`。

验收：

```bash
bash -n ops/ultrags_baseline_renderer/rebuild.sh
# Slurm compute node:
bash ops/ultrags_baseline_renderer/rebuild.sh
pytest -q -p no:cacheprovider tests/ops/ultrags_baseline_renderer
```

### Phase C：released-code parity smoke

目标：在 `synthetic_liver/l1` 上跑通最小训练，不启用 paper-aligned 修正。

推荐 alias：

```text
configs_hydra/experiment/preproduction/baseline_reproduction/ultrags_baseline/smoke.yaml
```

设置：

```text
engine.iterations: 200 或 500
scene.loading.test_mode: holdout_interval 或 modulo8_holdout
scene.sampler.num_init: 50000 - 100000
model.render.dar_mode: released_disabled
model.render.sh_degree: 2
model.offline.density_control.enable.densify: false
model.offline.density_control.enable.prune: false
```

训练 callback 对齐作者逻辑：

```text
每步随机/轮询一帧
render -> grayscale image
loss -> backward
optimizer.step
interval test/save/log
```

验收：

```bash
python run.py experiment=preproduction/baseline_reproduction/ultrags_baseline/smoke task=dry_run system.device=cpu engine.iterations=1
# Slurm:
python run.py experiment=preproduction/baseline_reproduction/ultrags_baseline/smoke system.output_root=outputs/preproduction/baseline_reproduction/ultrags_baseline/smoke
```

成功标准：

1. 无 NaN/Inf。
2. renderer forward/backward 可跑。
3. `logs/scalars.tsv` 有 train loss。
4. test 输出可保存 `vis/item_*`。
5. PSNR/SSIM 不要求接近论文，只要求训练链路合理下降。

### Phase D：density control 与 checkpoint

目标：迁移作者 clone/split/prune/opacity reset。

新增/实现：

```text
methods/ultrags_baseline/density_control.py
```

对齐作者：

```text
densify_from_iter: 500
densify_until_iter: configurable
densification_interval: 100
opacity_reset_interval: 3000
densify_grad_threshold: 2e-4
opacity_cull: 5e-2
percent_dense: 0.01
```

需要注意：

1. 官方使用 `means2D.grad` 统计 densification；项目 wrapper 必须保留 `screenspace_points` 并 retain grad。
2. `fov_x/fov_y`、SH、scale、quat、opacity 都要在 clone/split/prune 时同步更新 optimizer state。
3. checkpoint items 应包括 container、solver、density controller state。

验收：

```text
1000-1500 iteration smoke，densify 至少触发一次。
num_gaussians 变化合理。
checkpoint save/resume 可恢复。
```

### Phase E：DAR 复现分支

目标：解决论文 DAR 与 released code 冲突。

实现配置：

```text
model.render.dar_mode: released_disabled | global_offset | per_gaussian_offset
model.render.fov_x_init
model.render.fov_y_init
model.render.fov_x_min
model.render.fov_x_max
model.render.fov_y_min
model.render.fov_y_max
model.gauss_container.lr.fov_x
model.gauss_container.lr.fov_y
```

优先实现顺序：

1. `released_disabled`：复现作者当前代码。
2. `global_offset`：用全局 trainable `theta_x/theta_y`，按论文公式 bounded 到 FoV 范围。
3. 只有在有证据表明作者意图是每 Gaussian FoV 时，再做 `per_gaussian_offset`。

验收：

```text
记录 fov_x/fov_y 曲线。
对比 w/o DAR 和 DAR smoke 输出。
确认 FoV 不退化到边界。
```

### Phase F：paper_5k synthetic_liver

目标：从 smoke 进入可报告的探索复现实验，但仍不是正式生产对比。

推荐 alias：

```text
configs_hydra/experiment/preproduction/baseline_reproduction/ultrags_baseline/paper_5k_synthetic_liver.yaml
```

设置：

```text
engine.iterations: 5000
scene.loading.sweep_name: l1
split: frame_index % 8 == 0 test, others train
sh_degree: 2 for released code parity
position lr: 1.6e-4
fov lr: 1.0e-3 if DAR enabled
loss: 0.8 L1 + 0.2 SSIM
metrics: PSNR, SSIM, MSE/L1
```

输出：

```text
outputs/preproduction/baseline_reproduction/ultrags_baseline/paper_5k/
docs/methods/baselines/ultrags_baseline/synthetic_liver_report.md
```

报告必须记录：

1. source commit。
2. alias 和 overrides。
3. DAR mode。
4. split。
5. run dir。
6. final metrics。
7. renderer build job id。
8. released code 与 paper mismatch 的处理决定。

## 6. Slurm 计划

新增脚本：

```text
scripts/methods/baselines/ultrags_baseline/slurm/run_smoke.sh
scripts/methods/baselines/ultrags_baseline/slurm/run_paper_5k.sh
scripts/methods/baselines/ultrags_baseline/slurm/submit_synthetic_liver.sh
```

脚本要求：

1. stdout/stderr 写到 `logs/ultrags_baseline/`。
2. `EXPERIMENT` 传 Hydra alias，不使用 `CONFIG`。
3. 支持 `OVERRIDES`。
4. 编译/训练/评估均在计算节点。
5. CUDA 相关任务加载 `cuda/11.8.0`。
6. conda env 使用 `gaussian_splatting`。

最小验证：

```bash
mkdir -p logs/ultrags_baseline
bash -n scripts/methods/baselines/ultrags_baseline/slurm/run_smoke.sh
bash -n scripts/methods/baselines/ultrags_baseline/slurm/run_paper_5k.sh
```

## 7. 测试计划

### Method tests

```text
tests/methods/ultrags_baseline/test_config_and_container.py
  - Hydra config fields compose。
  - container 初始化 N、shape、dtype、device 正确。
  - activated opacity/scale finite。

tests/methods/ultrags_baseline/test_camera_adapter.py
  - synthetic_liver SliceMeta -> virtual camera shape 正确。
  - view/projection matrix finite。
  - FoV bounded。

tests/methods/ultrags_baseline/test_density_control.py
  - clone/split/prune 后所有参数 N 一致。
  - optimizer state 跟随参数变化。
```

### Ops tests

```text
tests/ops/ultrags_baseline_renderer/test_import.py
  - import ultrags_baseline_renderer
  - required _C symbols exist

tests/ops/ultrags_baseline_renderer/test_forward_shapes.py
  - small N/H/W forward returns image/radii/allmap
  - backward returns finite gradients for means, opacity, scale, rotation, SH
```

CUDA tests 走 Slurm。

## 8. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| 作者 released code 与论文 DAR 不一致 | 直接影响论文 claim 复现 | 先做 `released_disabled` parity，再独立实现 `global_offset` DAR |
| 作者 train/test split 泄漏 | PSNR/SSIM 虚高 | 项目内强制 true holdout，报告注明 |
| 官方默认 30k，论文写 5k | 指标不可比 | 同时保留 `paper_5k` 和 `code_released_30k` |
| PD Rendering 不是显式公式 | 消融难做 | 先迁移 CUDA SH degree 2 实现，再封装 ablation flags |
| synthetic_liver 有真实/合成 pose，不是 sensorless COLMAP | 与论文数据条件不同 | 当前只作为工程 smoke；作者数据集到位后再做 sensorless reproduction |
| virtual pinhole 与 linear probe geometry 坐标约定不同 | 可能出现投影错位 | `camera_adapter` 加矩阵单测和可视化 debug |
| CUDA extension 编译环境 | 登录节点不能 heavy build | Slurm + `gaussian_splatting` + `cuda/11.8.0` |
| 许可证 | 作者和 3DGS/2DGS 非商业研究许可 | 保留 license/copyright，不把第三方源码混入无归属公共目录 |

## 9. Promotion Gate

只有满足以下条件，`ultrags_baseline` 才能从生产前进入生产期：

1. `smoke` alias dry-run 通过。
2. renderer 在 Slurm 上 rebuild + ops smoke 通过。
3. synthetic_liver smoke 训练无 NaN，loss/metric 可记录。
4. true holdout split 明确写入 alias 或报告。
5. released-code parity 和 paper-aligned DAR 的差异已经记录。
6. `paper_5k_synthetic_liver` 至少完成一次，并有 `docs/methods/baselines/ultrags_baseline/*.md` 报告。
7. 无 checkpoint、logs、outputs、build、egg-info 进入 git。

进入生产期后再做：

```text
正式 multi-sweep synthetic_liver 对比
DAR / PD Rendering 消融
作者 UltraGS 数据集复现
多 seed 或最终 eval-only
FPS benchmark
```

## 10. 预计首个可交付里程碑

第一批实现建议只交付到 Phase C：

```text
1. method/config/state/callback skeleton
2. renderer package 可 import / forward
3. preproduction/baseline_reproduction/ultrags_baseline/smoke dry_run 通过
4. Slurm smoke 训练跑通 200-500 iter
5. 初始复现报告记录 released code mismatch
```

这能最快确认工程接入是否正确，同时避免在 DAR、COLMAP、作者数据集缺失和 production 对比之间过早耦合。
