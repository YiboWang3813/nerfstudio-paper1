# ultrags_baseline 完整生产前验证报告

日期：2026-06-02

本文档整合 `ultrags_baseline` 三轮 production-pre 验证结果，覆盖方法落地、CUDA renderer 构建、项目数据入口、10k 训练调参、关键问题定位、最终候选配置和生产状态判断。

生产前实验输出和 Slurm 日志已按项目规则清理；本文保留必要指标、配置结论和提交记录。当前报告不是 UltraGS 原论文数据集的最终复现报告，而是 `synthetic_liver/l1` 下的方法层生产候选验证。

## 入口与范围

- 方法包：`methods/ultrags_baseline`
- CUDA 算子：`ops/ultrags_baseline_renderer`
- 主要配置：
  - `experiment=preproduction/baseline_reproduction/ultrags_baseline/main_10k_synthetic_liver_1m_physical_camera_dc`
  - `experiment=preproduction/baseline_reproduction/ultrags_baseline/main_10k_synthetic_liver_300k_physical_camera_dc`
- Slurm 入口：
  - `scripts/methods/baselines/ultrags_baseline/slurm/run_rebuild_renderer.sh`
  - `scripts/methods/baselines/ultrags_baseline/slurm/run_main_10k.sh`
- 数据：`synthetic_liver`, sweep `l1`
- 测试划分：`holdout_interval`, 16 test frames
- 指标：`l1`, `psnr`, `ssim`, `lpips`
- 外部参考源码：`third_parties/UltraGS`, commit `de46f9a667c2d4b5a1d99ae88ea6b4b3514a6f54`

## 数据与初始化结论

`ultrags_baseline` 不依赖 COLMAP 输出也能在本项目内完成方法层复现。项目已有的 `Scene.build_init_gaussians()` 可以提供：

- 世界坐标系下的采样点 `means`
- per-pixel intensity，用于初始化 SH DC
- 初始 scale / quaternion
- 项目数据规定下的 train/test split

因此，缺少 COLMAP 点云不是阻断点。真正影响 `synthetic_liver` 表现的关键是 UltraGS 虚拟相机和项目物理切片坐标之间的对齐方式。

## 代码变更摘要

已验证并提交的关键实现包括：

- 新增 `methods/ultrags_baseline` 方法包和 `ops/ultrags_baseline_renderer` CUDA renderer。
- 在 `run.py`、`configs/base/models.py`、Hydra experiment/model configs 中注册 `ultrags_baseline`。
- 将项目 sampled world-pixel 初始化接入 UltraGS-style Gaussian container。
- 为 density control 补齐 clone/split/prune、opacity reset 和 n-max 裁剪流程。
- 修正 split densification：追加 split children 后删除被 split 的原 Gaussian，匹配官方 UltraGS 行为。
- 新增 `model.render.camera_mode=slice_plane_physical`：
  - lateral -> image x
  - axial -> image y
  - elevational -> camera depth
  - FoV 由 `DatasetMeta.width_mm`, `near_mm`, `far_mm` 和虚拟相机距离推导
- 增加 opacity cap 实验配置、splitfix 配置、1M/300k physical-camera 候选配置。
- 增加 renderer/reference、container/config、density control 和 physical camera 单元测试。

相关提交：

- `dc32551 Add ultrags baseline method scaffold`
- `5c53259 Fix ultrags baseline renderer preflight`
- `856325b Add ultrags baseline preflight report`
- `84412e4 Enable ultrags baseline density preflight`
- `935caef Tune ultrags baseline density preflight`
- `5cad70f Add ultrags baseline low opacity preflight`
- `eb6bf8f Add ultrags baseline small scale preflight`
- `cc26bdf Report ultrags baseline 10k preflight`
- `6375c59 Add ultrags baseline opacity cap preflights`
- `5e11890 Match ultrags split densification pruning`
- `50968a9 Add physical slice camera for ultrags baseline`
- `cb8ffae Add ultrags physical camera density candidate`
- `3450a8e Report ultrags physical camera preflight`

## CUDA 构建与基础验证

| job | 结果 | 说明 |
| --- | --- | --- |
| 60584 | 失败后修复 | extension 可安装，import smoke 通过；GPU forward test 因 identity projection 造成异常 footprint 和 OOM。 |
| 60585 | 通过 | 默认 `TORCH_CUDA_ARCH_LIST` 改为 `7.5;8.0;8.6`，测试改用有效透视投影；`tests/ops/ultrags_baseline_renderer` 为 `2 passed in 2.42s`。 |

最终本地验证命令：

```bash
PYTHONPATH=. pytest -q -p no:cacheprovider tests/methods/ultrags_baseline tests/ops/ultrags_baseline_renderer
```

结果：`6 passed, 1 skipped`。

## 早期冒烟实验

公共设置：`seed=42`，20 iter，`l1 + ssim`，density control 暂停，`active_sh_degree=0`，测试间隔 20，保存关闭。

| exp | job | 初始化点数 | FoV | train total | train L1 | test L1 | test PSNR | alpha coverage | pred mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `preflight_baseline_20` | 60587 | 2,000 | 60/60 | 0.3310 | 0.1382 | 0.1660 | 14.2819 | 0.1915 | 0.0037 |
| `preflight_fov45_20` | 60594 | 2,000 | 45/45 | 0.3321 | 0.1389 | 0.1660 | 14.2851 | 0.1777 | 0.0029 |
| `preflight_init5000_20` | 60595 | 5,000 | 60/60 | 0.3222 | 0.1336 | 0.1631 | 14.4181 | 0.3933 | 0.0099 |

结论：这些 20 iter 实验只证明训练闭环可用，不能判断方法质量。2k/5k 点数过低，输出明显偏暗、偏稀疏。提高初始化点数会提升 coverage，但仍不足以构成有效复现实验。

## 10k 调参过程

### Released-camera 路径

| job | run | init points | 关键设置 | final/latest eval | Gaussian count | 结论 |
| --- | --- | ---: | --- | --- | ---: | --- |
| 60600 | `main10k_100k_dc_l1` | 100k | opacity reset enabled, opacity cull 0.05 | 10k: PSNR 23.0289, SSIM 0.5012, LPIPS 0.7444, L1 0.0467 | 7,044 | 指标较高但高斯塌缩，scale 爆到 64.02，不可作为复现主配置。 |
| 60602 | `main10k_1m_tuned_dc_l1` | 1M | opacity 0.1, reset off, scale bound | 10k: PSNR 21.4842, SSIM 0.4775, LPIPS 0.7441, L1 0.0582 | 965,800 | 保持 1M 级别，但 alpha 过饱和。 |
| 60603 | `main10k_1m_low_opacity_l1` | 1M | init opacity 0.01, opacity lr 0.02 | 10k: PSNR 21.5599, SSIM 0.4824, LPIPS 0.7368, L1 0.0569 | 1,000,000 | 比默认 1M 略好，仍明显低于可生产预期。 |
| 60606 | `main10k_1m_small_scale_l1` | 1M | init 2D scale 0.1, init opacity 0.05 | 2.5k: PSNR 18.7204, SSIM 0.3511, LPIPS 0.7381, L1 0.0858 | 1,000,000 | 小 scale 明显欠拟合，取消。 |
| 60623 | `main10k_1m_low_opacity_splitfix_l1` | 1M | split-prune fix, released camera | 10k: PSNR 21.1985, SSIM 0.4741, LPIPS 0.7365, L1 0.0595 | 1,000,000 | DC 更忠实且 scale 稳定，但相机不匹配仍限制指标。 |
| 60624 | `main10k_300k_low_opacity_splitfix_l1` | 300k | split-prune fix, released camera | 10k: PSNR 20.9191, SSIM 0.4674, LPIPS 0.7519, L1 0.0611 | 300,000 | 同样受 released-camera 限制。 |

Released-camera 路径的诊断：

- 原始 60 度透视相机并不匹配 `synthetic_liver` 的物理切片平面。
- 100k run 的高 PSNR 来自 aggressive pruning 后的退化表示，不符合 1M/10k 生产实验要求。
- 单纯增加高斯数量不能解决问题；1M 下 alpha 很快饱和。
- 修正 split-prune 后 scale 更稳定，但 PSNR 未恢复，说明核心不在 density control 本身。

### Opacity cap 路径

| job | run | init points | 关键设置 | 10k eval | Gaussian count | 结论 |
| --- | --- | ---: | --- | --- | ---: | --- |
| 60615 | `main10k_1m_opacity_cap_l1` | 1M | opacity cap 0.2 | PSNR 20.5467, SSIM 0.4583, LPIPS 0.7497, L1 0.0643 | 1,000,000 | opacity 有界但欠拟合。 |
| 60616 | `main10k_300k_opacity_cap_l1` | 300k | opacity cap 0.2 | PSNR 20.7111, SSIM 0.4674, LPIPS 0.7645, L1 0.0641 | 300,000 | 与 1M cap 同样欠拟合。 |

Opacity cap 不是正确主修复。它能压住 opacity，但在 UltraGS 的 2D alpha compositing 下削弱了亮度学习。

## 关键修复：physical slice camera

项目中的 `synthetic_liver` 图像不是任意透视相机图像，而是由 `SliceMeta.pixel_pos` 定义的物理切片平面。released UltraGS 相机假设输入图像已经与透视相机 image plane 对齐；这个假设在当前项目数据入口下不成立。

`slice_plane_physical` 让 virtual camera 与项目数据一致：

- camera x 轴对应 slice lateral 方向
- camera y 轴对应 slice axial 方向
- camera z 轴对应 slice elevational 方向
- camera center 放在 slice plane 外侧
- FoV 由物理宽度和深度范围推导，而不是固定 60 度

这个改动后，1M PSNR 从约 21.2 dB 提升到 24.36 dB，LPIPS 从约 0.74 降到 0.50 左右。

## 最终生产候选实验

| job | run | init points | key setting | 10k eval | Gaussian count | status |
| --- | --- | ---: | --- | --- | ---: | --- |
| 60629 | `main10k_1m_physical_camera_l1` | 1M | physical slice camera | PSNR 24.3613, SSIM 0.5351, LPIPS 0.4981, L1 0.0427 | 1,000,000 | 主生产候选 |
| 60634 | `main10k_300k_physical_camera_l1` | 300k | physical slice camera | PSNR 24.3031, SSIM 0.5347, LPIPS 0.5194, L1 0.0430 | 300,000 | 轻量生产候选 |
| 60640 | `main10k_1m_physical_camera_resume_5000` | 1M | resume check from 5k | N/A | 1,000,000 | 成功加载 container 和 solver |

1M physical-camera 的完整测试曲线：

| iter | L1 | PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2,500 | 0.0439 | 24.1957 | 0.5335 | 0.5378 |
| 5,000 | 0.0434 | 24.2419 | 0.5332 | 0.5033 |
| 7,500 | 0.0429 | 24.3387 | 0.5349 | 0.5020 |
| 10,000 | 0.0427 | 24.3613 | 0.5351 | 0.4981 |

300k physical-camera 的完整测试曲线：

| iter | L1 | PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: |
| 2,500 | 0.0442 | 24.1472 | 0.5328 | 0.5514 |
| 5,000 | 0.0438 | 24.1768 | 0.5306 | 0.5128 |
| 7,500 | 0.0431 | 24.2859 | 0.5344 | 0.5219 |
| 10,000 | 0.0430 | 24.3031 | 0.5347 | 0.5194 |

## 推荐生产配置

主保真配置：

```bash
python run.py experiment=preproduction/baseline_reproduction/ultrags_baseline/main_10k_synthetic_liver_1m_physical_camera_dc
```

轻量配置：

```bash
python run.py experiment=preproduction/baseline_reproduction/ultrags_baseline/main_10k_synthetic_liver_300k_physical_camera_dc
```

建议默认选择：

- 需要最大保真时用 1M physical-camera。
- 需要更低训练/显存成本时用 300k physical-camera。它的 10k PSNR 只比 1M 低 0.0582 dB，SSIM 低 0.0004，但 LPIPS 略差 0.0213。

## 生产状态判断

当前状态：`synthetic_liver` 上达到 production-candidate，可以作为项目内生产配置进入下一阶段。

已满足：

- 方法、renderer、Hydra 配置、Slurm 入口均已落地。
- CUDA renderer 构建和参考测试通过。
- 项目数据入口符合规则，不依赖 COLMAP。
- 1M/10k 和 300k/10k 均完整完成，指标正常。
- checkpoint 保存逻辑跑通。
- 5k resume 加载 container/solver 成功。
- 生产前实验结果已写入本报告，workbench 输出和 Slurm 日志已清理。

仍需注意：

- 本报告不覆盖 UltraGS 原论文数据集。用户当前要求暂不接入该数据集，因此这不是阻断项。
- 如果后续接入 UltraGS 数据集，需要复验 `slice_plane_physical` 是否仍适用；对非项目物理切片数据可能需要回到 released-camera 或新增 dataset-specific camera adapter。
- 当前最佳配置是 `synthetic_liver` 上的生产候选，不应再使用 released 60-degree camera 作为项目默认。

## 最终结论

`ultrags_baseline` 的低指标问题已经定位并修复。缺少 COLMAP 输出不会阻断方法层复现；项目世界坐标像素点采样可以提供足够初始化信息。真正关键是把 UltraGS virtual camera 对齐到项目 `SliceMeta` 的物理切片平面。

在 `synthetic_liver/l1` 上，最终 1M physical-camera 配置达到：

- PSNR 24.3613
- SSIM 0.5351
- LPIPS 0.4981
- L1 0.0427
- Gaussian count 1,000,000

300k physical-camera 配置也达到接近质量，可作为轻量生产候选。
