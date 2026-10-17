# UltraGauss 复现与调试记录

Legacy note: this report preserves historical reproduction evidence and may mention old `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/ultragauss/` and should be launched with Hydra `experiment=...` aliases.

本文记录 UltraGauss baseline 的核心实现结论、调试过程、实验结果和当前停点。开发分支已 merge 回 main，当前复现入口以 main 下的配置、脚本和 checkpoint 路径为准。目标不是继续把对比方法刷到最强，而是保证它作为论文对比方法在设计上对齐、数值上稳定、结果可信。

## 当前结论

当前代码已经大部分对齐 `docs/literature/ultragauss/UltraGauss: Ultrafast Gaussian Reconstruction of 3D Ultrasound Volumes.md` 中的复现设计。

已经对齐的核心设计：

- 用 explicit 3D Gaussian 表示体数据，每个 Gaussian 包含 `means / precision_tril / opacities / intensities`。
- 用 triangular precision factor 参数化 `Lambda = L L^T`，其中 diagonal raw 参数平方后加 `diag_beta`，off-diagonal 直接学习。
- 渲染时按 probe slice 上的 3D Mahalanobis distance 计算 Gaussian-pixel 响应，而不是 camera 3DGS 的投影 alpha compositing。
- 像素输出为 Gaussian intensity 的 alpha 加权平均：
  `sum(alpha_i * intensity_i) / (sum(alpha_i) + bg_alpha)`。
- CUDA renderer 支持 world/probe geometry、chi-square culling、fixed background alpha/intensity、forward/backward。
- 外围训练接入现有 engine/loss/metric 体系，支持 offline train/test/save、holdout evaluation、render diagnostics、density control。
- 实验配置已放到 `config_files/experiments/ultragauss/`，没有沿用 `us_focus` 命名。

当前可以作为 UltraGauss baseline 定版的提交：

- `a79da7d Tune UltraGauss precision learning rate`

当前推荐配置：

- `config_files/experiments/ultragauss/liver_1m_10k_dc.yaml`
- synthetic liver
- 1M initial Gaussians
- 10k optimization steps
- density control on
- `model.gauss_container.lr.precision_tril = 0.035`

最终 10k 指标：

| run | iter | PSNR | L1 | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| current default, `precision_lr=0.035` | 2500 | 26.9110 | 0.03198 | 0.54835 | 0.29446 |
| current default, `precision_lr=0.035` | 5000 | 26.8620 | 0.03229 | 0.53495 | 0.23627 |
| current default, `precision_lr=0.035` | 7500 | 26.7248 | 0.03291 | 0.52405 | 0.22177 |
| current default, `precision_lr=0.035` | 10000 | 26.7170 | 0.03306 | 0.51848 | 0.21140 |

这个结果作为对比方法比较合适：最终 PSNR 约 `26.72`，LPIPS 约 `0.21`，表现稳定但没有被继续调到过强。

## 实现范围

主要修改集中在以下路径：

- `ops/ultragauss_renderer/**`
- `methods/ultragauss/**`
- `tests/methods/ultragauss/**`
- `ops/ultragauss_renderer/tests/**`
- `config_files/experiments/ultragauss/**`

没有为了 UltraGauss baseline 大幅改动共享框架。共享的 scene/loss/metric/engine 主要作为已有基础设施使用。

## 核心实现模块

### CUDA renderer

目录：

- `ops/ultragauss_renderer/src/mapper/**`
- `ops/ultragauss_renderer/src/shader/**`
- `ops/ultragauss_renderer/tests/test_forward_reference.py`

主要功能：

- 构造 `L` 和 `Lambda = L L^T`。
- 根据 slice geometry 把 Gaussian 和 pixel 放到一致坐标系中计算响应。
- 用 chi-square threshold `7.815` 做有效支持域截断。
- 对每个 pixel 累积：
  - alpha sum
  - alpha-weighted intensity sum
  - optional depth/intermediates
- 输出 grayscale prediction、alpha image、depth image 和中间统计。

关键验证：

- `test_ultragauss_forward_matches_dense_reference_identity_probe`
  - CUDA forward 与 dense PyTorch reference 对齐。
- `test_ultragauss_backward_matches_dense_reference_identity_probe`
  - CUDA backward 的 `means / precision_tril / opacities / intensities` 梯度与 dense reference 对齐。

### Gaussian container

文件：

- `methods/ultragauss/gaussian_container.py`

实现要点：

- 注册四类参数：
  - `means`
  - `precision_tril`
  - `opacities`
  - `intensities`
- `opacities` 和 `intensities` 用 sigmoid 激活。
- 初始化时优先使用 scene sampler 给出的 `scales/quats`，转换为 UltraGauss 的 raw triangular precision。
- 如果没有 scene scales，则回退到 raw diagonal/off-diagonal 随机初始化。

这一步很关键。早期版本用随机 precision 初始化，导致 geometry scale 与 scene 初始化不匹配，训练初期和 density control 都不稳定。改成从 scene scales 初始化后，渲染尺度、bbox 和训练收敛都明显更合理。

### Solver 与 LR scheduler

文件：

- `methods/ultragauss/gaussian_solver.py`
- `tests/methods/ultragauss/test_solver_scheduler.py`

早期 bug：

- 只给 `means` 注册了 LR scheduler。
- `precision_tril / opacities / intensities` 没有按训练步数 decay，后期容易继续剧烈变化，导致指标退化。

修复：

- 遍历 container 的所有 specs，为所有参数组注册 exponential LR scheduler。

效果：

- 10k 训练从明显退化变为稳定可用。
- `precision_tril` 后续仍需要单独调 LR，因为它直接控制 support/coverage 和视觉平滑程度。

### Density control

文件：

- `methods/ultragauss/density_control.py`

已实现内容：

- 在指定 iteration window 内累计 mean gradient norm。
- 按 gradient threshold 选择 clone/split。
- split 时根据 UltraGauss 采样公式：
  `y = mu + L^{-T} z, z ~ N(0, I)`
- 按 opacity/intensity/scale 做 prune。
- 支持 `n_max` 防止 Gaussian 数量失控。

关键修复：

- 原先 scale proxy 只用 `1 / diag(L)`，忽略 off-diagonal。
- 这与 renderer 中用 covariance marginal scale 做 culling 的逻辑不一致。
- 现在 `_approx_scale_from_precision_tril` 计算的是 `sqrt(diag(inv(L L^T)))`，即 covariance diagonal 对应的 marginal scale。
- 新增测试 `test_ultragauss_scale_proxy_matches_covariance_diagonal`，用显式 inverse reference 校验。

影响：

- 数学上更一致。
- 直接用原 LR 会降低 coverage，最终 PSNR/LPIPS 反而变差。
- 调低 `precision_tril` LR 后，coverage 恢复，LPIPS 明显改善。

### Offline callbacks 与 diagnostics

文件：

- `methods/ultragauss/offline_callbacks.py`
- `methods/ultragauss/render.py`

已接入：

- UltraGauss setup/train/test/save/finish callback。
- 每步渲染当前 training slice。
- holdout test interval。
- render stats：
  - predicted image mean/std/min/max
  - ground truth image mean/std/min/max
  - alpha coverage
  - pixel-gaussian pair count
  - opacity/intensity stats
  - max scale stats

这些 diagnostics 对调试非常重要。很多问题不是直接 NaN，而是表现为：

- pred mean 偏高或偏低。
- alpha coverage 下降。
- pixel-gaussian pair 数明显变少。
- scale mean 持续漂移。
- LPIPS 变好但 PSNR 变差。

## 关键 bug 与修复记录

### 1. Renderer culling math

现象：

- 早期 culling/bbox 的尺度判断和 dense reference 不完全一致。
- pixel-gaussian pair 数和 coverage 统计异常。

处理：

- 对齐 covariance diagonal / chi-square radius 的计算。
- 用 reference math 测试验证 bbox half width 应来自 covariance diagonal。

相关提交：

- `544ae77 Fix UltraGauss renderer culling math`

### 2. CUDA buffer 稳定性

现象：

- 渲染中间 buffer 在大规模 1M Gaussian 实验下需要更稳的容量和边界处理。

处理：

- 稳定 CUDA buffer/中间输出逻辑。

相关提交：

- `6de9684 Stabilize UltraGauss renderer CUDA buffers`

### 3. Scene scale 初始化

现象：

- 随机 precision 初始化时，Gaussian 支持域与已有 scene sampler 的尺度先验不匹配。
- 初期 loss 和 coverage 不理想，后续 density control 也更难稳定。

处理：

- 用 scene sampler 产生的 scales/quats 初始化 UltraGauss precision。
- 将 covariance/scale 表示转换为 triangular precision raw 参数。

相关提交：

- `57ef7f6 Initialize UltraGauss precision from scene scales`

### 4. 全参数 LR scheduler

现象：

- 只 decay `means` 学习率时，`precision_tril/opacities/intensities` 后期仍然更新过猛。
- 10k 指标相比 2500/5000 明显退化。

处理：

- 给所有参数组注册 scheduler。

相关提交：

- `b79b14d Schedule all UltraGauss parameter learning rates`

### 5. Precision gradient backward bug

现象：

- CUDA backward 与 dense PyTorch reference 对不上。
- 问题集中在 `precision_tril` 的 off-diagonal 梯度。

根因：

- `l12` 的梯度公式多算了本应属于 `l23` 路径的一项。

修复：

- 修正 `ops/ultragauss_renderer/src/shader/backward.cu` 中 `dl12` 的表达式。
- 增加 CUDA backward reference test。

验证：

- Slurm GPU pytest：
  - `ops/ultragauss_renderer/tests/test_forward_reference.py`
  - `2 passed`

相关提交：

- `f1ed9e3 Fix UltraGauss precision gradient`

### 6. Covariance scale proxy

现象：

- density control 和 scale wall 用的 `gauss_scales` 只看 precision Cholesky diagonal。
- 当 off-diagonal 学起来后，这个 scale proxy 会低估或误判真实 covariance marginal scale。

处理：

- 改为闭式计算 `sqrt(diag(inv(L L^T)))`。
- 不用通用 matrix inverse，而是用 triangular inverse 的元素公式。

相关提交：

- `3b47dae Use covariance scales for UltraGauss density control`

## 实验过程与指标

所有关键实验均使用：

- dataset: `config_files/_base_/scene/liver.yaml`
- initial Gaussians: `1,000,000`
- iterations: `10,000`
- holdout interval evaluation
- config root: `config_files/experiments/ultragauss/liver_1m_10k_dc.yaml`

### 早期 scale-init baseline

tag:

- `ultragauss_liver_1m_10000_dc_scaleinit`

结果：

| iter | PSNR | L1 | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | 22.9862 | 0.05152 | 0.37646 | 0.51891 |
| 10000 | 17.2886 | 0.11279 | 0.13777 | 0.47593 |

结论：

- 单独做 scene scale 初始化还不够。
- 因为只有 `means` scheduler，其他参数后期更新过猛，整体退化。

### 全参数 LR scheduler 后 baseline

tag:

- `ultragauss_liver_1m_10000_dc_lrd`

结果：

| iter | PSNR | L1 | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | 26.6807 | 0.03273 | 0.53224 | 0.36525 |
| 5000 | 26.8059 | 0.03260 | 0.52275 | 0.29402 |
| 7500 | 26.6515 | 0.03335 | 0.50999 | 0.26985 |
| 10000 | 26.5244 | 0.03395 | 0.49924 | 0.26128 |

结论：

- 训练进入可用范围。
- 仍有明显后期 PSNR/SSIM 退化。

### Loss weight 试探

tag:

- `ultragauss_liver_1m_10000_dc_l1w`

override:

- `loss.lambdas.l1=0.8`
- `loss.lambdas.ssim=0.2`

结果：

| iter | PSNR | L1 | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | 26.3819 | - | - | 0.4278 |
| 10000 | 26.1730 | 0.0355 | 0.4720 | 0.3260 |

结论：

- 更高 L1 权重没有改善 holdout。
- 放弃这条方向。

### Precision gradient 修复后

tag:

- `ultragauss_liver_1m_10000_dc_gradfix`

结果：

| iter | PSNR | L1 | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | 26.8091 | 0.03218 | 0.54264 | 0.35512 |
| 5000 | 26.8982 | 0.03213 | 0.53253 | 0.27822 |
| 7500 | 26.7793 | 0.03277 | 0.52087 | 0.25311 |
| 10000 | 26.7237 | 0.03312 | 0.51226 | 0.24195 |

结论：

- 这是第一个数学上正确且主指标不错的版本。
- 相比 scheduler baseline，最终 PSNR 和 LPIPS 均有改善。

### Covariance scale proxy 直接启用

tag:

- `ultragauss_liver_1m_10000_dc_covscale`

结果：

| iter | PSNR | L1 | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | 26.8312 | 0.03216 | 0.54259 | 0.35675 |
| 5000 | 26.8424 | 0.03245 | 0.52680 | 0.28601 |
| 7500 | 26.6384 | 0.03339 | 0.51147 | 0.26510 |
| 10000 | 26.5335 | 0.03393 | 0.50055 | 0.26093 |

结论：

- 实现更正确，但原 `precision_tril lr=0.05` 下 coverage 下降。
- 不能直接作为最终配置，需要 retune precision LR。

### Covariance scale + precision LR 0.02

tag:

- `ultragauss_liver_1m_10000_dc_covscale_plr002`

结果：

| iter | PSNR | L1 | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | 26.8790 | 0.03211 | 0.54980 | 0.26589 |
| 5000 | 26.7198 | 0.03272 | 0.53501 | 0.22361 |
| 7500 | 26.6186 | 0.03321 | 0.52681 | 0.21296 |
| 10000 | 26.6425 | 0.03319 | 0.52361 | 0.20229 |

结论：

- LPIPS 最好，最终约 `0.2023`。
- PSNR 稍低。
- 说明降低 precision LR 能改善感知指标和结构，但牺牲像素误差。

### 当前默认：Covariance scale + precision LR 0.035

tag:

- `ultragauss_liver_1m_10000_dc_covscale_plr0035`

结果：

| iter | PSNR | L1 | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | 26.9110 | 0.03198 | 0.54835 | 0.29446 |
| 5000 | 26.8620 | 0.03229 | 0.53495 | 0.23627 |
| 7500 | 26.7248 | 0.03291 | 0.52405 | 0.22177 |
| 10000 | 26.7170 | 0.03306 | 0.51848 | 0.21140 |

结论：

- 这是当前最合适的折中点。
- 2500 PSNR 是所有试验中最高。
- 5000 也保持较高 PSNR，同时 LPIPS 明显优于 `precision_lr=0.05`。
- 10000 最终 PSNR 基本持平 gradfix baseline，LPIPS 明显更好。
- 作为 baseline 不宜继续刷太强，建议停在这里。

## 当前仍未完全复刻的点

以下点不是明显 bug，而是当前实现与论文/潜在官方代码可能不同的地方：

1. `intensities` 初始化仍是常数初始化后优化。
   - 文档中 UltraGauss 可以自然从像素 intensity 先验初始化。
   - 当前 scene sampler 没有在方法边界内直接返回每个 sampled Gaussian 对应的 source intensity。
   - 要做这件事可能需要改 shared `scene/**` 或 sampler 输出接口，超出当前 baseline 必要范围。

2. Density control 是按文档精神实现，不保证与原作者代码逐行一致。
   - clone/split/prune 的时机、阈值、上限都是根据当前框架和 liver 1M/10k 任务调过。

3. Loss 配方不是论文官方精确配置。
   - 当前使用已有 `l1 + ssim + scale_wall`。
   - 论文没有给出足够完整的训练 loss 细节。

4. 目前只系统验证了 synthetic liver 这一组主配置。
   - 作为对比方法足够。
   - 若未来要跨数据集报告，需要再跑相同配置或轻微 retune。

## 建议停点

建议把 UltraGauss baseline 暂时定版在当前提交：

- `a79da7d Tune UltraGauss precision learning rate`

理由：

- 算子 forward/backward 已有 dense reference 测试。
- 主要数学 bug 已修。
- 1M/10k liver 配置已稳定跑完。
- 指标合理，已经能作为可信 baseline。
- 继续调参大概率会让对比方法过强，反而不符合当前项目定位。

后续除非发现以下情况，否则不建议继续大规模调 UltraGauss：

- CUDA reference test 失败。
- 主方法比较时发现 UltraGauss baseline 明显异常低。
- 需要在新数据集上补充 UltraGauss 对比。
- 审稿/汇报需要更严格解释某个实现细节。

## 复现实验命令参考

当前默认配置运行：

```bash
python run.py --config config_files/experiments/ultragauss/liver_1m_10k_dc.yaml
```

Slurm 示例：

```bash
mkdir -p logs/ultragauss
sbatch scripts/methods/baselines/ultragauss/slurm/03_liver_1m_10k_dc.sh
```

本地 CPU 侧参考测试：

```bash
pytest -q -p no:cacheprovider tests/methods/ultragauss/test_reference_math.py tests/methods/ultragauss/test_solver_scheduler.py ops/ultragauss_renderer/tests/test_forward_reference.py
```

GPU reference test 需要在 Slurm/GPU 节点上跑：

```bash
sbatch scripts/methods/baselines/ultragauss/slurm/01_renderer_reference.sh
```
