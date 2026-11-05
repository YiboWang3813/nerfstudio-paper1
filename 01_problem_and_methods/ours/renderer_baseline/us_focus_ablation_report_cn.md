# us_focus 渲染器消融实验报告

Legacy note: 本报告保留早期 us_focus renderer baseline 的历史消融记录，因此正文中仍会出现旧 `config_files/...` 路径。当前可运行入口已经迁移到 `configs_hydra/experiment/preproduction/ideas/ours/renderer_baseline/` 和 production comparison alias，运行时应使用 Hydra `experiment=...`。

日期：2026-05-22；更新：2026-05-26

本文档汇总截至 Phase 22 为止，`ops/ours_renderer` 中 `us_focus` 路径的全部已完成消融实验。除特别说明外，实验均使用 synthetic liver holdout、10k 训练迭代、开启 density control，并使用 `l1 + ssim + scale_wall` 损失组合。

原始结果汇总表：

```text
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/us_focus_ablation_results.tsv
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/us_focus_pose_prior_results.tsv
```

## 总结

目前 `us_focus` CUDA 渲染器在已测试 liver 设置上优于旧的 `probe3d` 路径，已经满足最初目标：100 万初始高斯、开启 DC、10k iter、PSNR 超过 25、LPIPS 约 0.2，并且速度进入 130-150 it/s 区间。

| Profile | 配置 / 设置 | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1M fast default | `liver_1m_10k_dc_fast.yaml` | 28.1984 +/- 0.0004 | 0.1929 +/- 0.0001 | 0.0277 | 0.5904 | 697,124 +/- 99 | 134-135 it/s |
| 1M fast-fidelity | `liver_1m_10k_dc_fast_fidelity.yaml`, `l1=0.65, ssim=0.35` | 28.2666 +/- 0.0013 | 0.1927 +/- 0.0005 | 0.0275 | 0.5910 | 697,509 +/- 130 | 约 134 it/s |
| 1M perceptual | `liver_1m_10k_dc_perceptual.yaml` | 27.9686 +/- 0.0043 | 0.1825 +/- 0.0002 | 0.0284 | 0.5845 | 724,994 +/- 174 | 约 129 it/s |
| x2 liver fast | `liver_x2_750k_10k_dc_fast.yaml` | 30.6180 +/- 0.0024 | 0.1086 +/- 0.0001 | 0.0208 | 0.7014 | 578,085 +/- 8 | 约 164 it/s |

当前推荐：

- 若优先 PSNR/L1 fidelity，使用 `config_files/experiments/us_focus/liver_1m_10k_dc_fast_fidelity.yaml`。它相对 Phase 10 fast default 平均 PSNR 提高约 0.068，同时基本保持 LPIPS 和速度。
- 保留 `config_files/experiments/us_focus/liver_1m_10k_dc_fast.yaml` 作为历史 speed/fidelity baseline。
- 若优先 LPIPS/感知质量，使用 `config_files/experiments/us_focus/liver_1m_10k_dc_perceptual.yaml`。
- x2 liver 设置使用 `config_files/experiments/us_focus/liver_x2_750k_10k_dc_fast.yaml`。
- `scale_wall` 继续使用 50；更强 wall 权重没有带来足够稳定的指标收益。
- `us_focus_det_gamma` 继续使用 0.25；更低或更高的已测值都会明显恶化 LPIPS。
- soft lateral focus 不应删除，但应作为 perceptual mode，而不是 reconstruction-fidelity 默认项。
- A-line 深度自适应 elevation thickness 应作为 fidelity-biased focus prior 来表述。它在 PSNR/SSIM 上最好，但 constant/no-slope 变体在 LPIPS 上有优势。
- Phase 18 的 `l2/l3/r1/r2/r3` 多 sweep 验证支持上述分工：fast_fidelity 在所有 5 个新 sweep 上都提升 PSNR，perceptual 在所有 5 个新 sweep 上都降低 LPIPS，但 PSNR 损失明显。
- Phase 22 的大噪声真实训练结果支持“PEM pose 只能作为 noisy prior”的设定，但暂不支持直接声称当前 joint pose optimization 已能在真实 GS 训练中修正大幅 PEM 误差。3deg/0.5mm noisy pose 会让 r2 30k PSNR 从 GT pose 的 27.289 降到约 24.57；当前多种 pose 开启时机和学习率都未恢复 pose，且指标未超过 noisy-fixed。
- Phase 24 用 r2 相邻帧步长 5%-10% 的小误差模拟更真实的 PEM prior。固定 5%/10% noisy prior 几乎无损，PSNR 分别为 27.2894/27.2824，LPIPS 为 0.1881/0.1887；这支持“PEM 提供接近真实轨迹的 noisy prior，GS 建图对小 prior error 稳健”的叙事。10% 噪声下 15k 后打开 pose 梯度的 PSNR 约 27.27，图像指标只轻微下降；但多个 pose LR/prior 组合都会把旋转误差从 0.0063 deg 推到 0.067-0.080 deg，说明当前 joint pose refinement 不应直接宣称稳定校正 GT pose。
- Phase 25 证明了更合理的受约束 refinement 版本：冻结 rotation、只优化 translation，并使用 15k-17k pose-only correction 窗口后固定 refined pose。该设置把 r2 10% prior 的 translation error 从 `0.02813 mm` 降到 `0.02608 mm`，rotation 保持 `0.00630 deg` 不漂移，PSNR `27.2897` 基本不受影响。这可以支撑“GS 阶段能在受约束设置下改善 noisy pose prior 且不损害重建”的叙事。
- Phase 26 用更容易观察的 25%-50% translation-only prior error 做受约束 refinement stress test。50% fixed prior 的 translation error 为 `0.14065 mm`、PSNR `27.1744`；15k-20k pose-only、`lr_trans=3e-4`、`lambda_prior_trans=0.001` 后，translation error 降到 `0.12347 mm`，PSNR 升到 `27.1912`，rotation 仍固定。这证明受约束 GS refinement 能稳定改善中等 translation prior error，但改善幅度约 `12.2%`，更适合写成“鲁棒补偿/可选 refinement”，不宜写成大幅 pose 校正主贡献。
- Phase 27 正式接入 `methods/fima` 的 adjacent-pair PEM，并把导出的链式 pose prior 喂给 Ours。当前 5k PEM prior 在 r2 上的绝对轨迹误差为 `1.0838 mm / 4.1424 deg`，远大于此前假设的 5%-10% 小误差；Ours fixed prior 仍能达到 `27.2618 PSNR / 0.1978 LPIPS`，但 translation-only residual refine 没有改善 pose，translation error 从 `1.0865 mm` 轻微变差到 `1.0892 mm`。这说明当前 FiMA prior 质量还不足以支撑“GS refine 真实 PEM pose”的强叙事，下一步应优先提升/筛选 PEM checkpoint 和降低旋转链式漂移。

## 创新点状态

| 模块 / 设计点 | 当前结论 | 证据 |
| --- | --- | --- |
| `us_focus` vs `probe3d` | 保留 `us_focus` 作为默认路径 | Phase 3：`us_focus` PSNR 28.2198、LPIPS 0.1945；`probe3d` PSNR 28.0131、LPIPS 0.3343。 |
| 深度自适应 elevation thickness | 保留在 fidelity profile 中 | Phase 5：depth-adaptive PSNR 28.2242、SSIM 0.5905，优于 constant/no-slope 的 PSNR/SSIM。 |
| constant/no-slope elevation | 仅作为 perceptual variant | Phase 5：LPIPS 可到约 0.183，但 PSNR/SSIM 下降。 |
| lateral focus | 作为可选 perceptual mode | soft lateral 能把 1M perceptual profile 的 LPIPS 提到约 0.182，但相对 fast default 损失约 0.23 PSNR。 |
| determinant normalization gamma | 固定 `0.25` | Phase 1 gamma sweep 中 `0.0/0.125/0.5` 的 LPIPS 都明显更差。 |
| tile mapper | 使用 `tile_size_px=8` | Phase 4：tile8 保持质量且有较高速度；tile32 降到 16.10 it/s。 |
| strong prune + attribute LR | 保留在 fast profiles 中 | Phase 8/9：减少最终高斯数，并把速度推入 130-150 it/s 目标区间。 |
| L1-heavy loss ratio | 提升为 fast-fidelity profile | Phase 17：`l1=0.65, ssim=0.35` 平均 PSNR 28.2666、LPIPS 0.1927。 |
| 多 sweep 一致性 | 支持 profile 分工 | Phase 18：fast_fidelity 在 5 个新 sweep 上平均 PSNR 27.7455，高于 fast 的 27.6706；perceptual 平均 LPIPS 0.1881，低于 fast 的 0.1964。 |
| pose 梯度 | 梯度链路成立；受约束 synthetic refine 成立；真实 FiMA prior 尚未 refine 成功 | Phase 19-21：6DoF pose 有限差分通过；受控 pose-only recovery 可收敛；joint 受控分支有非零 pose 梯度。Phase 22/24：自由 6DoF joint refinement 容易产生 rotation 漂移。Phase 25/26：translation-only + pose-only window 可降低 translation error，同时保持 rotation 不漂移和图像质量基本不变。Phase 27：FiMA 5k prior 旋转链式漂移过大，Ours residual refine 未降低 pose error。 |

## Phase 1：初始参数扫描

这些是早期基于原始 1M 10k DC liver 设置的参数扫描，尚未经过后续 tile/optimizer 调优。

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ele_slope_0` | `us_focus_sigma_ele_1=0.0` | 25.3473 | 0.2322 | 0.0356 | 0.5416 | 1,008,588 |
| `ele_slope_0005` | `us_focus_sigma_ele_1=0.005` | 25.5448 | 0.2308 | 0.0346 | 0.5537 | 977,260 |
| `ele_slope_002` | `us_focus_sigma_ele_1=0.02` | 26.0829 | 0.2413 | 0.0326 | 0.5691 | 891,291 |
| `lateral_off` | `us_focus_use_lateral_focus=false` | 28.1461 | 0.1915 | 0.0279 | 0.5874 | 864,706 |
| `det_gamma_0` | `us_focus_det_gamma=0.0` | 25.2334 | 0.3257 | 0.0347 | 0.5714 | 1,028,776 |
| `det_gamma_0125` | `us_focus_det_gamma=0.125` | 25.4874 | 0.2606 | 0.0342 | 0.5665 | 1,006,469 |
| `det_gamma_05` | `us_focus_det_gamma=0.5` | 25.8997 | 0.2706 | 0.0342 | 0.5542 | 796,898 |
| `init_500k` | `num_points=500000` | 25.4787 | 0.2674 | 0.0341 | 0.5650 | 526,177 |
| `init_1500k` | `num_points=1500000` | 25.7622 | 0.2253 | 0.0339 | 0.5589 | 1,304,946 |

结论：

- 关闭 lateral focus 是第一轮最关键的质量提升。
- `us_focus_det_gamma=0.25` 应继续作为默认归一化设置。`0.0/0.125/0.5` 都显著恶化 LPIPS。

## Phase 2：focus 参数交互

除显式开启 lateral focus 的实验外，本组均以 lateral-off baseline 为基础。

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phase2_lat_off_base` | none | 28.1459 | 0.1919 | 0.0279 | 0.5874 | 865,035 |
| `phase2_lat_off_ele1_zero` | `us_focus_sigma_ele_1=0.0` | 27.8875 | 0.1866 | 0.0289 | 0.5687 | 948,294 |
| `phase2_lat_off_ele1_low` | `us_focus_sigma_ele_1=0.005` | 28.0616 | 0.1844 | 0.0282 | 0.5814 | 904,732 |
| `phase2_lat_off_ele1_high` | `us_focus_sigma_ele_1=0.02` | 28.2201 | 0.2083 | 0.0276 | 0.5929 | 797,917 |
| `phase2_lat_off_ele1_vhigh` | `us_focus_sigma_ele_1=0.03` | 28.2344 | 0.2226 | 0.0276 | 0.5945 | 753,210 |
| `phase2_lat_off_ele_wide` | `sigma_ele_0=1.0, min=0.8, max=5.0` | 28.2233 | 0.1952 | 0.0277 | 0.5906 | 840,211 |
| `phase2_lat_off_ele_narrow` | `sigma_ele_0=0.6, min=0.3, max=2.0` | 28.0221 | 0.1886 | 0.0282 | 0.5828 | 888,449 |
| `phase2_lat_off_depth_const_wide` | constant `sigma_ele=1.2` | 28.1698 | 0.1836 | 0.0279 | 0.5839 | 893,894 |
| `phase2_lat_on_default` | lateral on default | 25.7580 | 0.2331 | 0.0337 | 0.5611 | 944,696 |
| `phase2_lat_on_soft` | soft lateral | 27.8972 | 0.1772 | 0.0286 | 0.5801 | 915,928 |
| `phase2_lat_on_wide` | wide lateral | 28.0661 | 0.1803 | 0.0281 | 0.5853 | 890,193 |
| `phase2_lat_on_flat` | flat lateral `sigma_lat=16.0` | 28.1453 | 0.1927 | 0.0279 | 0.5874 | 865,172 |

结论：

- lateral off + depth-adaptive elevation 是 PSNR/SSIM 最优方向。
- soft/wide lateral focus 能改善 LPIPS，但会损失 PSNR，因此更适合作为 perceptual mode。

## Phase 3：`us_focus` 与旧 `probe3d` 对比

`plane_cond2d` 因数值不稳定和梯度爆炸已废弃，本组未对比。

| Ablation | Shader | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 观测训练速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `compare_focus_ele_wide_rebuild` | `us_focus` | 28.2198 | 0.1945 | 0.0277 | 0.5905 | 840,281 | warmup 后约 124-129 it/s |
| `compare_probe3d_rebuild` | `probe3d` | 28.0131 | 0.3343 | 0.0286 | 0.5783 | 949,525 | warmup 后约 100-103 it/s |

结论：

- `us_focus` 是更强默认路径，LPIPS 和 SSIM 明显优于 `probe3d`。
- 速度瓶颈不只是坐标系转换次数；mapper/workspace/backward 代价和最终高斯数量更关键。

## Phase 4：渲染速度超参数

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase3_tile8` | `tile_size_px=8` | 28.2242 | 0.1956 | 0.0277 | 0.5905 | 840,004 | 117.30 it/s |
| `phase3_tile32` | `tile_size_px=32` | 28.2217 | 0.1954 | 0.0277 | 0.5904 | 840,477 | 16.10 it/s |
| `phase3_window3` | `num_pixels_lateral=3, num_pixels_axial=3` | 28.2234 | 0.1946 | 0.0277 | 0.5906 | 840,518 | 106.80 it/s |
| `phase3_window7` | `num_pixels_lateral=7, num_pixels_axial=7` | 28.2217 | 0.1950 | 0.0277 | 0.5905 | 840,149 | 106.68 it/s |
| `phase3_ksigma25` | `k_sigma=2.5` | 28.2058 | 0.1928 | 0.0277 | 0.5899 | 851,463 | 115.76 it/s |
| `phase3_ksigma35` | `k_sigma=3.5` | 28.2254 | 0.1956 | 0.0276 | 0.5906 | 838,618 | 99.51 it/s |

结论：

- `tile_size_px=8` 是最佳速度/质量默认值。
- `tile_size_px=32` 虽保持质量，但速度崩到 16.10 it/s。
- `k_sigma=2.5` 在单独测试时 PSNR 略低，但后续与 strong prune 和 attribute LR 组合后成为 fast profile 的关键设置。

## Phase 5：tile8 下的创新点稳定性

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase3_tile8` | depth-adaptive elevation, lateral off | 28.2242 | 0.1956 | 0.0277 | 0.5905 | 840,004 | 117.30 it/s |
| `phase5_tile8_const_wide` | constant `sigma_ele=1.2` | 28.1697 | 0.1834 | 0.0279 | 0.5838 | 894,270 | 111.16 it/s |
| `phase5_tile8_ele1_zero` | `us_focus_sigma_ele_1=0.0` | 28.0623 | 0.1827 | 0.0283 | 0.5780 | 921,956 | 110.29 it/s |
| `phase5_tile8_lat_soft` | soft lateral focus on | 27.9611 | 0.1817 | 0.0285 | 0.5833 | 890,695 | 119.20 it/s |

结论：

- depth-adaptive thickness 仍是 fidelity 默认项。
- constant/no-slope thickness 和 soft lateral 是 perceptual 替代项，不应替代 canonical PSNR/SSIM profile。

## Phase 6：初始高斯数量和 density-control budget

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase3_tile8` | 1M default DC reference | 28.2242 | 0.1956 | 0.0277 | 0.5905 | 840,004 | 117.30 it/s |
| `phase6_tile8_init750k` | `num_points=750000` | 28.2597 | 0.2106 | 0.0276 | 0.5921 | 671,321 | 125.35 it/s |
| `phase6_tile8_init1250k` | `num_points=1250000` | 28.1952 | 0.1865 | 0.0277 | 0.5893 | 991,804 | 107.08 it/s |
| `phase6_tile8_strong_prune` | stronger prune/DC | 28.2224 | 0.1953 | 0.0277 | 0.5906 | 789,351 | 118.49 it/s |

结论：

- 750k 提升速度和 PSNR/SSIM，但 LPIPS 变差。
- 1250k 改善 LPIPS，但速度较慢。
- strong prune 是最保守有效的默认改进。

## Phase 7：750k fast perceptual variants

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase6_tile8_init750k` | depth-adaptive default | 28.2597 | 0.2106 | 0.0276 | 0.5921 | 671,321 | 125.35 it/s |
| `phase7_750k_ksigma25` | `k_sigma=2.5` | 28.2449 | 0.2088 | 0.0276 | 0.5915 | 679,668 | 134.21 it/s |
| `phase7_750k_const_wide` | constant `sigma_ele=1.2` | 28.2122 | 0.2056 | 0.0278 | 0.5860 | 704,253 | 121.68 it/s |
| `phase7_750k_lat_soft` | soft lateral focus on | 27.9890 | 0.1920 | 0.0284 | 0.5851 | 712,959 | 130.51 it/s |

结论：

- `750k + soft lateral` 是 lower-budget perceptual 最佳组合：PSNR 接近 28，LPIPS 低于 0.2，速度 130.5 it/s。

## Phase 8：strong-prune default 上的优化器学习率

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase6_tile8_strong_prune` | default LR reference | 28.2224 | 0.1953 | 0.0277 | 0.5906 | 789,351 | 118.49 it/s |
| `phase8_sp_means_high` | `lr.means=2.4e-4` | 28.2193 | 0.1925 | 0.0277 | 0.5902 | 794,303 | 119.23 it/s |
| `phase8_sp_means_low` | `lr.means=1.0e-4` | 28.2315 | 0.1979 | 0.0276 | 0.5909 | 785,133 | 120.27 it/s |
| `phase8_sp_attr_high` | `lr.opacities=8e-2, lr.intensities=2e-3` | 28.2182 | 0.1940 | 0.0276 | 0.5913 | 684,130 | 127.36 it/s |

结论：

- 提高 opacity/intensity 学习率是最有效的优化器变化。它基本保持质量，同时最终高斯数减少约 105k，速度明显提高。

## Phase 9：strong-prune + attr-high 后续组合

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase8_sp_attr_high` | reference | 28.2182 | 0.1940 | 0.0276 | 0.5913 | 684,130 | 127.36 it/s |
| `phase9_sp_attr_means_high` | `lr.means=2.4e-4` | 28.2135 | 0.1920 | 0.0277 | 0.5909 | 687,706 | 127.03 it/s |
| `phase9_sp_attr_ksigma25` | `k_sigma=2.5` | 28.1989 | 0.1928 | 0.0277 | 0.5906 | 697,616 | 134.90 it/s |
| `phase9_sp_attr_lat_soft` | soft lateral focus on | 27.9686 | 0.1822 | 0.0284 | 0.5845 | 724,899 | 129.45 it/s |

结论：

- `strong_prune + attr_high + k_sigma=2.5` 第一次在 1M 初始高斯下达到目标速度区间，同时保持 PSNR > 28、LPIPS < 0.2。
- soft lateral 在 1M family 里 LPIPS 最好，但损失约 0.25 PSNR。

## Phase 10：formal fast config 三 seed 复现

配置：`config_files/experiments/us_focus/liver_1m_10k_dc_fast.yaml`

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase10_fast_seed42` | 42 | 28.1981 | 0.1928 | 0.0277 | 0.5904 | 696,994 | 134.98 it/s |
| `phase10_fast_seed43` | 43 | 28.1981 | 0.1928 | 0.0277 | 0.5903 | 697,235 | 134.88 it/s |
| `phase10_fast_seed44_retry` | 44 | 28.1989 | 0.1930 | 0.0277 | 0.5904 | 697,142 | 134.43 it/s |

均值：

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 28.1984 | 0.0004 |
| LPIPS | 0.1929 | 0.0001 |
| L1 | 0.0277 | 0.0000 |
| SSIM | 0.5904 | 0.0000 |
| 最终高斯数 | 697,124 | 99 |

结论：

- 该配置达到原始目标：1M 初始高斯、开启 DC、PSNR > 25、LPIPS < 0.2、速度 130-150 it/s。

## Phase 11：liver x2 resolution transfer

配置：`config_files/experiments/us_focus/liver_x2_750k_10k_dc_fast.yaml`

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase11_x2_fast_seed42` | 42 | 30.6188 | 0.1086 | 0.0208 | 0.7014 | 578,088 | 164.24 it/s |
| `phase11_x2_fast_seed43` | 43 | 30.6205 | 0.1088 | 0.0208 | 0.7015 | 578,073 | 163.44 it/s |
| `phase11_x2_fast_seed44` | 44 | 30.6148 | 0.1084 | 0.0208 | 0.7012 | 578,093 | 163.71 it/s |

均值：

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 30.6180 | 0.0024 |
| LPIPS | 0.1086 | 0.0001 |
| L1 | 0.0208 | 0.0000 |
| SSIM | 0.7014 | 0.0001 |
| 最终高斯数 | 578,085 | 8 |

结论：

- fast profile 可以干净迁移到 x2 liver 设置，且质量和速度都更好。
- 这还不是多 scene/anatomy 验证。

## Phase 12：formal perceptual config 三 seed 复现

配置：`config_files/experiments/us_focus/liver_1m_10k_dc_perceptual.yaml`

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase12_perceptual_seed42_env` | 42 | 27.9742 | 0.1822 | 0.0284 | 0.5844 | 724,759 | 128.85 it/s |
| `phase12_perceptual_seed43_env` | 43 | 27.9636 | 0.1827 | 0.0284 | 0.5845 | 725,047 | 128.86 it/s |
| `phase12_perceptual_seed44_env` | 44 | 27.9681 | 0.1825 | 0.0284 | 0.5845 | 725,176 | 129.15 it/s |

均值：

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 27.9686 | 0.0043 |
| LPIPS | 0.1825 | 0.0002 |
| L1 | 0.0284 | 0.0000 |
| SSIM | 0.5845 | 0.0000 |
| 最终高斯数 | 724,994 | 174 |

结论：

- perceptual profile 相比 fast default 的 LPIPS 改善约 0.010，但损失约 0.23 PSNR，速度慢约 6 it/s。

## Phase 13：scale-wall loss 权重扫描

本组使用 formal fast config 和 seed 42。

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase10_fast_seed42` | reference, `scale_wall=50` | 28.1981 | 0.1928 | 0.0277 | 0.5904 | 696,994 | 134.98 it/s |
| `phase13_fast_scale_wall25` | `scale_wall=25` | 28.2032 | 0.1924 | 0.0277 | 0.5906 | 697,404 | 132.46 it/s |
| `phase13_fast_scale_wall75` | `scale_wall=75` | 28.1985 | 0.1922 | 0.0277 | 0.5905 | 697,331 | 133.60 it/s |
| `phase13_fast_scale_wall100` | `scale_wall=100` | 28.1984 | 0.1927 | 0.0277 | 0.5905 | 697,758 | 135.05 it/s |

结论：

- 已测权重都稳定。
- `scale_wall=75` 的单 seed LPIPS 最好，但相对默认只好约 0.0006，不足以替换默认值。
- 默认继续用 `scale_wall=50`；除非视觉检查明确显示更强 wall regularization 能减轻高方向波纹。

## Phase 14：L1/SSIM loss ratio 扫描

本组使用 formal fast config 和 seed 42。

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase10_fast_seed42` | `l1=0.5, ssim=0.5` | 28.1981 | 0.1928 | 0.0277 | 0.5904 | 696,994 | 134.98 it/s |
| `phase14_fast_l1_06_ssim_04` | `l1=0.6, ssim=0.4` | 28.2408 | 0.1927 | 0.0276 | 0.5906 | 697,538 | 134.46 it/s |
| `phase14_fast_l1_04_ssim_06` | `l1=0.4, ssim=0.6` | 28.1598 | 0.1935 | 0.0278 | 0.5903 | 697,380 | 135.05 it/s |
| `phase14_fast_l1_07_ssim_03` | `l1=0.7, ssim=0.3` | 28.2952 | 0.1934 | 0.0274 | 0.5913 | 697,342 | 134.71 it/s |

结论：

- L1-heavy ratio 会改善 PSNR/L1，速度基本不变。
- SSIM-heavy `0.4/0.6` 更差。
- `0.7/0.3` 单 seed PSNR 最高，但 LPIPS 变差，因此只适合 PSNR-biased profile。

## Phase 15：fast-fidelity `0.6/0.4` 三 seed 复现

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase14_fast_l1_06_ssim_04` | 42 | 28.2408 | 0.1927 | 0.0276 | 0.5906 | 697,538 | 134.46 it/s |
| `phase15_fast_l1_06_ssim_04_seed43` | 43 | 28.2417 | 0.1926 | 0.0276 | 0.5907 | 697,671 | 134.85 it/s |
| `phase15_fast_l1_06_ssim_04_seed44` | 44 | 28.2467 | 0.1930 | 0.0276 | 0.5908 | 697,532 | 135.10 it/s |

均值：

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 28.2431 | 0.0026 |
| LPIPS | 0.1928 | 0.0001 |
| L1 | 0.0276 | 0.0000 |
| SSIM | 0.5907 | 0.0001 |
| 最终高斯数 | 697,580 | 64 |

结论：

- `l1=0.6, ssim=0.4` 稳定提升 PSNR，但后续 Phase 17 的 `0.65/0.35` 更好。

## Phase 16：细化 loss ratio 和 wall 权重

| Ablation | Override | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase14_fast_l1_06_ssim_04` | `l1=0.6, ssim=0.4` reference | 28.2408 | 0.1927 | 0.0276 | 0.5906 | 697,538 | 134.46 it/s |
| `phase16_fast_l1_065_ssim_035` | `l1=0.65, ssim=0.35` | 28.2669 | 0.1927 | 0.0275 | 0.5911 | 697,433 | 134.38 it/s |
| `phase16_fast_l1_06_ssim_04_scale_wall75` | `l1=0.6, ssim=0.4, scale_wall=75` | 28.2433 | 0.1932 | 0.0276 | 0.5909 | 698,017 | 135.11 it/s |

结论：

- `l1=0.65, ssim=0.35` 是更好的 fast-fidelity 候选。
- `0.6/0.4 + scale_wall=75` 没有改善标量指标。

## Phase 17：fast-fidelity `0.65/0.35` 三 seed 复现

已提升配置：`config_files/experiments/us_focus/liver_1m_10k_dc_fast_fidelity.yaml`

| Ablation | Seed | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase16_fast_l1_065_ssim_035` | 42 | 28.2669 | 0.1927 | 0.0275 | 0.5911 | 697,433 | 134.38 it/s |
| `phase17_fast_l1_065_ssim_035_seed43` | 43 | 28.2679 | 0.1921 | 0.0275 | 0.5909 | 697,403 | 135.10 it/s |
| `phase17_fast_l1_065_ssim_035_seed44` | 44 | 28.2649 | 0.1932 | 0.0275 | 0.5909 | 697,692 | 132.40 it/s |

均值：

| Metric | Mean | Std |
| --- | ---: | ---: |
| PSNR | 28.2666 | 0.0013 |
| LPIPS | 0.1927 | 0.0005 |
| L1 | 0.0275 | 0.0000 |
| SSIM | 0.5910 | 0.0001 |
| 最终高斯数 | 697,509 | 130 |

结论：

- 这是当前最佳 1M fast-fidelity profile。
- 相比 Phase 10 fast default，平均 PSNR 提升约 0.068，SSIM 提升约 0.0006，同时 LPIPS 维持在约 0.193，速度仍在目标区间。

## Phase 18：多 sweep 一致性验证

本组使用 `synthetic_liver` 中除 `l1` 之外的 5 个 sweep：`l2/l3/r1/r2/r3`。这些 sweep 的 DatasetMeta 相同，但图像内容不同。每个 sweep 先跑 seed42，对比三个 profile：

- `fast`：`config_files/experiments/us_focus/liver_1m_10k_dc_fast.yaml`
- `fast_fidelity`：`config_files/experiments/us_focus/liver_1m_10k_dc_fast_fidelity.yaml`
- `perceptual`：`config_files/experiments/us_focus/liver_1m_10k_dc_perceptual.yaml`

| Sweep | Profile | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | 端到端速度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `l2` | fast | 27.9518 | 0.1981 | 0.0286 | 0.5750 | 702,633 | 129.69 it/s |
| `l2` | fast_fidelity | 28.0266 | 0.1985 | 0.0284 | 0.5755 | 702,449 | 135.18 it/s |
| `l2` | perceptual | 27.6524 | 0.1887 | 0.0294 | 0.5686 | 731,128 | 128.70 it/s |
| `l3` | fast | 28.0885 | 0.2036 | 0.0280 | 0.5815 | 663,847 | 138.21 it/s |
| `l3` | fast_fidelity | 28.1639 | 0.2034 | 0.0278 | 0.5821 | 664,196 | 137.78 it/s |
| `l3` | perceptual | 27.5074 | 0.1981 | 0.0294 | 0.5750 | 684,396 | 130.45 it/s |
| `r1` | fast | 27.9362 | 0.1928 | 0.0289 | 0.5694 | 695,184 | 135.29 it/s |
| `r1` | fast_fidelity | 28.0060 | 0.1929 | 0.0287 | 0.5700 | 696,251 | 135.50 it/s |
| `r1` | perceptual | 27.6981 | 0.1835 | 0.0297 | 0.5632 | 722,502 | 130.04 it/s |
| `r2` | fast | 27.2742 | 0.1952 | 0.0312 | 0.5402 | 714,093 | 135.40 it/s |
| `r2` | fast_fidelity | 27.3502 | 0.1958 | 0.0309 | 0.5409 | 714,427 | 135.48 it/s |
| `r2` | perceptual | 27.0238 | 0.1861 | 0.0320 | 0.5332 | 743,521 | 128.96 it/s |
| `r3` | fast | 27.1024 | 0.1923 | 0.0318 | 0.5320 | 684,370 | 137.99 it/s |
| `r3` | fast_fidelity | 27.1810 | 0.1930 | 0.0316 | 0.5326 | 684,177 | 138.45 it/s |
| `r3` | perceptual | 26.8839 | 0.1840 | 0.0326 | 0.5261 | 703,797 | 133.10 it/s |

跨 sweep seed42 均值：

| Profile | PSNR | LPIPS | L1 | SSIM | 最终高斯数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast | 27.6706 | 0.1964 | 0.0297 | 0.5597 | 692,025 |
| fast_fidelity | 27.7455 | 0.1967 | 0.0295 | 0.5602 | 692,300 |
| perceptual | 27.3531 | 0.1881 | 0.0306 | 0.5532 | 717,069 |

结论：

- fast_fidelity 在 `l2/l3/r1/r2/r3` 全部 5 个 sweep 上都提升 PSNR，平均提升约 0.075；L1/SSIM 也有小幅改善。这个结果支持把 `l1=0.65, ssim=0.35` 作为跨内容更稳的 fidelity profile。
- fast_fidelity 的 LPIPS 与 fast 基本持平，平均略差 0.0003。因此它不是 perceptual 改善项，而是明确的 fidelity 改善项。
- perceptual 在全部 5 个 sweep 上都降低 LPIPS，平均从 0.1964 降到 0.1881，但平均 PSNR 从 27.6706 降到 27.3531。这个结果支持将 soft lateral 作为 perceptual mode，而不是默认 reconstruction-fidelity mode。
- `r2/r3` 明显比 `l2/l3/r1` 更难，PSNR 和 SSIM 都显著更低。后续 20k/30k 长训练应优先覆盖 `r2/r3`，因为它们更能暴露收敛和伪影问题。

## 剩余风险和下一步实验

- 当前结果已经覆盖 synthetic liver 的 `l1/l2/l3/r1/r2/r3` 多 sweep，但仍缺少另一个 scene/anatomy 或 real-data split。
- 用户已观察到图像高方向仍有一定波纹。标量指标目前不支持直接提高 `scale_wall`，但仍需要视觉 side-by-side 检查。
- lateral focus 不应删除。它对 perceptual profile 有价值，但 canonical reconstruction-fidelity profile 应默认关闭。
- depth-adaptive elevation thickness 应保留在主要创新点中，但论文表述要准确：它在本 liver 设置中提升 fidelity metrics，而 LPIPS 可能偏好 constant/no-slope 替代项。
- Phase 19-21 已验证来自图像 loss 的梯度能够通过 `us_focus` CUDA backward 传给 probe pose，并且在小型受控问题中能支持 pose-only 和 joint pose+Gaussian 优化。下一步应把这个能力接入真实训练框架的 pose 参数化、学习率和正则项。
- Phase 22 显示，真实训练中如果直接放开 pose，Gaussian 容易吸收 pose prior 误差，图像梯度不一定把 pose 拉向 GT。后续需要引入更明确的 pose 监督/约束实验，例如 PEM prior schedule、train-frame-only pose metrics、pose regularizer 分量拆解、冻结 Gaussian warmup 后短窗 pose-only refinement，或跨帧 pose smooth/trajectory prior。
- 下一组自然实验是在困难 sweep `r3` 上复验 30k noisy pose-prior，以及在 `r2` 上做更结构化的 pose correction 方案，而不是继续盲目增大 pose_lr。

## Phase 19：6DoF pose 梯度有限差分验证

本组验证的是真正用于 pose optimization 叙事的链路：`image loss -> CUDA d_pose -> differentiable 6DoF pose params`。与旧测试只检查 4x4 pose 矩阵条目不同，这里使用 `[rx, ry, rz, tx, ty, tz]` 参数化生成 probe-to-world pose。

Slurm 作业：

```text
59250 / logs/ablations/usf_pose19_59250.out
```

| 参数 | Analytic grad | Finite difference | Abs err | Rel err |
| --- | ---: | ---: | ---: | ---: |
| `rx` | 4.0946e-03 | 4.0184e-03 | 7.6222e-05 | 0.0190 |
| `ry` | -2.1171e-03 | -2.2688e-03 | 1.5168e-04 | 0.0669 |
| `rz` | 2.7248e-02 | 2.7391e-02 | 1.4255e-04 | 0.0052 |
| `tx` | -3.4163e-03 | -3.4163e-03 | 7.3574e-08 | 0.0000 |
| `ty` | -1.7881e-03 | -1.7878e-03 | 3.1618e-07 | 0.0002 |
| `tz` | 6.3141e-04 | 6.3205e-04 | 6.3924e-07 | 0.0010 |

结论：

- 6DoF pose 参数的最大绝对误差为 `1.52e-4`，最大相对误差为 `0.0669`，通过阈值。
- 初始化扰动为旋转误差 `2.1236 deg`、平移误差 `0.0998 mm`。
- 归一化梯度下降搜索中，`step=0.02` 将 loss 从 `3.5882e-4` 降到 `1.3664e-4`，说明该梯度提供有效下降方向。

## Phase 20：冻结 Gaussian 的 pose-only recovery

本组冻结 Gaussian 位置、尺度、旋转、opacity 和 intensity，仅优化 6DoF pose，从 small/medium 两档扰动恢复到目标图像。

Slurm 作业：

```text
59251 / logs/ablations/usf_pose20_59251.out
```

| Case | Init loss | Final loss | Init PSNR | Final PSNR | Init rot err | Final rot err | Init trans err | Final trans err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 6.7405e-04 | 1.8424e-14 | 31.7131 | 120.0000 | 3.0188 deg | 0.0000 deg | 0.2490 mm | 0.0000 mm |
| medium | 2.5276e-03 | 1.5929e-14 | 25.9729 | 120.0000 | 5.7083 deg | 0.0000 deg | 0.5177 mm | 0.0000 mm |

结论：

- 在受控 synthetic renderer 问题中，图像梯度可以稳定驱动 pose-only optimization。
- small 和 medium 扰动均收敛到近零 loss、近零 rotation/translation error。

## Phase 21：joint pose + Gaussian 受控优化

本组模拟后续联合训练叙事：从同一个错误 pose 和轻微扰动 Gaussian 开始，对比固定错误 pose 的 Gaussian-only、固定 Gaussian 的 pose-only、以及 joint pose+Gaussian 优化。

Slurm 作业：

```text
59252 / logs/ablations/usf_pose21_59252.out
```

| Branch | Init loss | Final loss | Ratio | Init PSNR | Final PSNR | Init rot err | Final rot err | Init trans err | Final trans err | First pose grad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian-only, fixed wrong pose | 2.3762e-03 | 6.4364e-06 | 0.002709 | 26.2411 | 51.9136 | 4.5166 deg | 4.5166 deg | 0.4162 mm | 0.4162 mm | N/A |
| Pose-only, fixed Gaussian | 2.3762e-03 | 8.5738e-04 | 0.360812 | 26.2411 | 30.6683 | 4.5166 deg | 1.2361 deg | 0.4162 mm | 0.1048 mm | 1.6962e-02 |
| Joint pose + Gaussian | 2.3762e-03 | 2.6270e-07 | 0.000111 | 26.2411 | 65.8055 | 4.5166 deg | 2.1648 deg | 0.4162 mm | 0.1229 mm | 1.6962e-02 |

结论：

- joint 分支中 pose 梯度非零且有限，证明 CUDA `us_focus` backward 可以和 Gaussian 属性优化共存。
- joint 优化将图像 loss 降低约 9000 倍，明显优于 pose-only 的最终图像误差。
- Gaussian-only 在固定错误 pose 下也能拟合图像，但 pose error 不会变化；这提示真实联合训练必须加入合理 pose 参数化、学习率和正则，否则 Gaussian 可能吸收一部分 pose 误差。

## Phase 22：真实 r2 noisy PEM pose-prior 训练

本组把 Phase 19-21 的受控 pose 梯度能力接入真实 `synthetic_liver/r2` 训练。目标是验证用户提出的叙事：PEM 提供有噪 pose prior，前半程按 prior 建图，后半程打开 pose 梯度联合优化 GS 和 pose。

公共设置：

- 配置：`config_files/experiments/us_focus/liver_1m_30k_dc_fast_fidelity_pose_prior.yaml`
- sweep：`r2`
- 迭代：30k
- 初始高斯：1M
- 损失：`l1 + ssim + scale_wall`，开启 density control
- noisy prior：local probe-frame noise，`rot=3.0 deg`，`trans=0.5 mm`，`seed=20260526`
- 结果汇总：

```text
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/us_focus_pose_prior_results.tsv
```

有效 Slurm 作业：

```text
59256, 59264, 59265, 59266, 59267, 59274, 59275, 59273
```

无效/排除作业：

```text
59257
```

`59257` 使用了早期错误的 left-multiplied noise，导致 nominal `0.5 mm` 平移噪声实际统计为 `1.317 mm`。实现已修正为 local/right-multiplied noise；最终结论只采用修正后的 local-noise 结果。

| Ablation | Pose 设置 | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | Pose rot err | Pose trans err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase22_r2_gt_pose_30k` | GT pose fixed | 27.2890 | 0.1884 | 0.0311 | 0.5392 | 846,475 | 0.0052 deg | 0.0000 mm |
| `phase22_r2_noisy_fixed_30k_local` | noisy prior fixed | 24.5674 | 0.2526 | 0.0412 | 0.4635 | 865,887 | 2.9985 deg | 0.5000 mm |
| `phase22_r2_noisy_joint0_30k_local` | trainable from 0k, lr=3e-5, lambda=0.1 | 24.3879 | 0.2542 | 0.0418 | 0.4615 | 863,100 | 3.0100 deg | 0.4935 mm |
| `phase22_r2_noisy_delay10k_30k_local` | trainable from 10k, lr=3e-5, lambda=0.1 | 24.4476 | 0.2533 | 0.0416 | 0.4627 | 866,098 | 3.0573 deg | 0.4954 mm |
| `phase22_r2_noisy_delay15k_30k_local` | trainable from 15k, lr=3e-5, lambda=0.1 | 24.4857 | 0.2532 | 0.0415 | 0.4628 | 864,932 | 3.0499 deg | 0.4965 mm |
| `phase22_r2_delay15_lr1e4_lam01` | trainable from 15k, lr=1e-4, lambda=0.1 | 24.4859 | 0.2564 | 0.0415 | 0.4637 | 863,408 | 3.0505 deg | 0.4959 mm |
| `phase22_r2_delay15_lr1e4_lam001` | trainable from 15k, lr=1e-4, lambda=0.01 | 24.4786 | 0.2554 | 0.0415 | 0.4635 | 862,918 | 3.0661 deg | 0.4954 mm |
| `phase22_r2_delay15_lr3e4_lam001` | trainable from 15k, lr=3e-4, lambda=0.01 | 24.5047 | 0.2823 | 0.0413 | 0.4682 | 852,995 | 3.0449 deg | 0.4968 mm |

结论：

- 有噪 PEM prior 明显伤害重建质量。相对 GT pose fixed，local noisy fixed 的 PSNR 下降约 `2.72 dB`，LPIPS 从 `0.1884` 恶化到 `0.2526`。
- 当前 joint pose optimization 没有修正 pose。`0k/10k/15k` 三种开启时机均未超过 noisy-fixed，pose rotation error 最终都在约 `3.01-3.06 deg`。
- 提高 pose 学习率或降低 prior weight 也没有解决问题。`lr=1e-4` 基本不变；`lr=3e-4` 的 PSNR 略高到 `24.5047`，但 LPIPS 明显恶化到 `0.2823`。
- 这说明 CUDA backward 和 pose 参数梯度是成立的，但真实 GS 训练下图像 loss 对 pose 的监督不够可辨，Gaussian 属性和 geometry 会吸收一部分 pose 误差。
- 论文/叙事上可以保留“PEM pose 是 noisy prior，GS 框架具备 pose-gradient refinement 能力”，但不能把当前 Phase 22 结果写成“真实 joint training 已成功校正 PEM pose”。下一步应设计更强的 pose correction 实验，而不是继续无结构地调大学习率。

## Phase 23：GT pose 相邻帧运动尺度统计

本组不是训练实验，而是校准 noisy PEM prior 的尺度。脚本：

```text
scripts/eval/analyze_pose_motion_scale.py
```

命令：

```bash
python scripts/eval/analyze_pose_motion_scale.py
```

该脚本读取 `synthetic_liver` 的 `relative_poses.npy`，统计相邻帧 probe pose 的旋转和平移。结论是 Phase 22 使用的 `rot=3.0 deg, trans=0.5 mm` 不是中等 PEM 误差，而是远大于真实相邻帧运动尺度的 stress test。

| Sweep | Split | N | Trans mean | Trans p95 | Rot mean | Rot p95 | `0.5mm / trans_p95` | `3deg / rot_p95` | 5% trans | 5% rot | 10% trans | 10% rot |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `l1` | train | 200 | 0.2685 | 0.2687 | 0.0071 | 0.0405 | 1.861x | 74.16x | 0.0134 | 0.0020 | 0.0269 | 0.0040 |
| `l2` | train | 200 | 0.2685 | 0.2687 | 0.0260 | 0.1047 | 1.861x | 28.66x | 0.0134 | 0.0052 | 0.0269 | 0.0105 |
| `l3` | train | 200 | 0.2494 | 0.2496 | 0.0428 | 0.1013 | 2.003x | 29.63x | 0.0125 | 0.0051 | 0.0250 | 0.0101 |
| `r1` | train | 200 | 0.2839 | 0.2855 | 0.0064 | 0.0280 | 1.752x | 107.23x | 0.0143 | 0.0014 | 0.0285 | 0.0028 |
| `r2` | train | 200 | 0.2812 | 0.2813 | 0.0024 | 0.0280 | 1.778x | 107.23x | 0.0141 | 0.0014 | 0.0281 | 0.0028 |
| `r3` | train | 200 | 0.2522 | 0.2523 | 0.0021 | 0.0280 | 1.982x | 107.23x | 0.0126 | 0.0014 | 0.0252 | 0.0028 |

对 Phase 22 的修正解释：

- `0.5 mm` 平移噪声在 `r2` 上约等于 `1.78` 个相邻帧 p95 步长，不是“小误差”。
- `3 deg` 旋转噪声在 `r2/r3` 上约等于 `107x` 相邻帧 p95 旋转，更不符合近线性平扫假设。
- 如果要模拟 PEM 约 5%-10% 的 pose prior error，更合理的 `r2` train 噪声量级大约是 `0.014-0.028 mm` 平移和 `0.0014-0.0028 deg` 旋转。这个尺度很小，可能对 GS 重建指标影响有限，但可以作为“审稿人质疑”场景的严谨实验。
- 更合适的论文叙事不是“GS 必须大幅校正 PEM pose”，而是“PEM 提供接近真实轨迹的 pose prior；GS 阶段具备可微 pose refinement 能力，并在轻微 prior error 下验证鲁棒性/不退化”。如果要证明校正能力，应单独设计可辨识的 pose-only 或 semi-frozen refinement 实验，而不是让全量 Gaussian 与 pose 同时自由优化。

## Phase 24：r2 小误差 PEM prior 合理性验证

本组按 Phase 23 统计出的真实相邻帧运动尺度，重新设置 PEM prior 噪声。目标不是证明 GS 必须大幅校正 pose，而是验证更合理的 PEM prior 误差下：

- 固定 noisy prior 建图是否几乎不退化。
- 15k 后打开 pose 梯度是否至少不明显伤害重建。
- 当前真实 joint training 是否能把 pose 拉回 GT。

公共设置：

- 配置：`config_files/experiments/us_focus/liver_1m_30k_dc_fast_fidelity_pose_prior.yaml`
- sweep：`r2`
- 迭代：30k
- 初始高斯：1M
- 损失：`l1 + ssim + scale_wall`，开启 density control
- noise：local probe-frame fixed-magnitude perturbation，`seed=20260526`
- 结果汇总：

```text
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/us_focus_pose_prior_results.tsv
```

Slurm 作业：

```text
59283, 59284, 59285, 59288, 59289
```

对照基线：

- `phase22_r2_gt_pose_30k`：PSNR `27.2890`，LPIPS `0.1884`，L1 `0.0311`，SSIM `0.5392`。
- `phase22_r2_noisy_fixed_30k_local` 大噪声 stress test：`3 deg / 0.5 mm`，PSNR `24.5674`，LPIPS `0.2526`。

| Ablation | Pose 设置 | Nominal noise | 实测 pose err | PSNR | LPIPS | L1 | SSIM | 最终高斯数 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `phase24_r2_noise5_fixed_30k` | fixed noisy prior | `0.001399 deg / 0.014065 mm` | `0.00543 deg / 0.01407 mm` | 27.2894 | 0.1881 | 0.0311 | 0.5392 | 845,822 |
| `phase24_r2_noise10_fixed_30k` | fixed noisy prior | `0.002798 deg / 0.028130 mm` | `0.00630 deg / 0.02813 mm` | 27.2824 | 0.1887 | 0.0311 | 0.5390 | 846,951 |
| `phase24_r2_noise10_delay15_30k` | 0-15k fixed，15-30k trainable，lr=3e-5，lambda=0.1 | `0.002798 deg / 0.028130 mm` | `0.07970 deg / 0.02590 mm` | 27.2721 | 0.1891 | 0.0311 | 0.5383 | 847,453 |
| `phase24_r2_noise10_delay15_lr1e5_lam1_30k` | 0-15k fixed，15-30k trainable，lr=1e-5，lambda=1.0 | `0.002798 deg / 0.028130 mm` | `0.07655 deg / 0.02708 mm` | 27.2717 | 0.1881 | 0.0311 | 0.5383 | 846,263 |
| `phase24_r2_noise10_delay15_lr3e6_lam1_30k` | 0-15k fixed，15-30k trainable，lr=3e-6，lambda=1.0 | `0.002798 deg / 0.028130 mm` | `0.06737 deg / 0.02742 mm` | 27.2763 | 0.1886 | 0.0311 | 0.5386 | 847,550 |

结论：

- 小误差 fixed PEM prior 几乎无损。5% 噪声下 PSNR 与 GT fixed 基本持平，10% 噪声下 PSNR 仅下降约 `0.0066 dB`，LPIPS 只从 `0.1884` 到 `0.1887`。这证明如果 PEM prior 只是相邻步长 5%-10% 量级误差，GS 建图对该 prior 是稳健的。
- 这解决了“PEM prior 小误差是否还有必要处理”的叙事悖论：实验上小误差不需要靠 GS 大幅纠正也能重建好，但方法上保留 pose prior / pose-gradient refinement 可以回应审稿人对 PEM 误差传播的质疑。
- 10% 噪声下 15k 后打开 pose 梯度并未造成明显图像崩坏。三组 delay15 的 PSNR 都在 `27.2717-27.2763`，LPIPS 在 `0.1881-0.1891`，相对 10% fixed 的损失很小。因此它可以暂时作为“可选后半程 refinement 不明显伤害图像”的证据。
- 但这些 joint refinement 都没有把 pose 严格拉回 GT。平移误差从 `0.02813 mm` 小幅降到 `0.02590-0.02742 mm`，而旋转误差从 `0.00630 deg` 增到 `0.06737-0.07970 deg`。这说明真实 GS joint training 中，图像 loss 可能通过轻微旋转漂移换取类似重建误差；当前结果不能写成“GS 后半程稳定优化出更准 pose”。
- 单纯降低统一 pose LR 或增大 prior weight 不足以解决旋转漂移。`lr=1e-5, lambda=1.0` 和 `lr=3e-6, lambda=1.0` 只略微降低旋转漂移，但没有改变趋势。
- 后续 pose 叙事如果要更强，应做代码级受约束 refinement：rotation/translation 分组 LR、只优化 translation、冻结 rotation、或加入跨帧 trajectory smooth prior。当前最稳妥的论文表述是“PEM 提供接近真实轨迹的 pose prior；小误差 prior 下 GS 重建稳健；算子支持 pose 梯度，受控实验验证梯度有效，真实 joint refinement 作为可选机制仍需约束。”

## Phase 25：受约束 translation-only pose refinement

本组针对 Phase 24 暴露出的 rotation 漂移问题，加入更符合 r2 近线性平扫先验的约束：

- rotation residual 冻结，只优化 translation residual。
- rotation 和 translation 支持独立 optimizer 参数组。
- 支持 pose-only 时间窗：窗口内只更新 pose，不更新 Gaussian；窗口后固定 refined pose，继续训练 GS。
- 支持 residual smooth loss；本组用于验证 smooth 项是否对当前噪声模型有益。

代码提交：

```text
ed50a50 Add constrained pose refinement controls
fd86b4d Add pose refinement end iteration
4eb0118 Keep refined pose after optimization window
```

公共设置：

- 配置：`config_files/experiments/us_focus/liver_1m_30k_dc_fast_fidelity_pose_prior.yaml`
- sweep：`r2`
- 迭代：30k
- noisy prior：10% r2 相邻步长，`0.002798 deg / 0.028130 mm`
- rotation：`optimize_rotation=false`
- translation：`optimize_translation=true`
- 对照：10% fixed prior 为 PSNR `27.2824`、LPIPS `0.1887`、rot `0.00630 deg`、trans `0.02813 mm`

有效 Slurm 作业：

```text
59300, 59304, 59305
```

无效/排除作业：

```text
59301, 59302
```

`59301/59302` 使用了早期 `end_iter` 实现。该实现会在 pose 优化窗口后回退到原 prior，丢掉 learned delta。已在 `4eb0118` 修正为：`end_iter` 后停止 pose optimizer step 和 pose loss，但继续使用 refined pose 渲染。最终结论只采用修复后的 `59304/59305`。

| Ablation | Pose 设置 | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | Pose rot err | Pose trans err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase24_r2_noise10_fixed_30k` | fixed 10% noisy prior | 27.2824 | 0.1887 | 0.0311 | 0.5390 | 846,951 | 0.00630 deg | 0.02813 mm |
| `phase25_r2_transonly_joint15_30k` | 15k 后 translation-only joint，lr_trans=3e-5，lambda_trans=0.1 | 27.2810 | 0.1881 | 0.0311 | 0.5385 | 846,559 | 0.00630 deg | 0.02618 mm |
| `phase25b_r2_transonly_poseonly15_17_30k` | 15k-17k translation-only pose-only，之后固定 refined pose | 27.2897 | 0.1899 | 0.0311 | 0.5389 | 831,349 | 0.00630 deg | 0.02608 mm |
| `phase25b_r2_transonly_poseonly15_17_smooth_30k` | 同上，加 `lambda_smooth_trans=1.0` | 27.2885 | 0.1890 | 0.0311 | 0.5391 | 831,765 | 0.00630 deg | 0.02759 mm |

结论：

- 受约束 translation-only refinement 成立。相对 10% fixed prior，`phase25_r2_transonly_joint15_30k` 把 translation error 从 `0.02813 mm` 降到 `0.02618 mm`，rotation 保持 `0.00630 deg`，PSNR/LPIPS 基本不变。
- pose-only window 版本最适合论文叙事。`phase25b_r2_transonly_poseonly15_17_30k` 在 15k-17k 只更新 translation pose，之后固定 refined pose 继续训练 GS，最终 translation error 降到 `0.02608 mm`，PSNR `27.2897` 基本不受影响。这说明 GS 图像梯度确实可以在受约束设置下改善 noisy pose prior。
- smooth 项在当前随机逐帧噪声模型下不占优。`lambda_smooth_trans=1.0` 后 translation error 只降到 `0.02759 mm`。这不说明轨迹平滑先验无效，而说明当前 synthetic noise 是逐帧随机 residual；如果要验证 smooth prior，应改用更 PEM-like 的低频/平滑 pose error。
- 最终可写叙事应避免“自由 6DoF joint optimization 自动修准 pose”。更稳妥且已有实验支持的版本是：“PEM 提供强 pose prior；GS 采用可微渲染，在后半程用受约束 translation-only/pose-only refinement 修正小 residual；该过程改善 pose prior，且重建质量基本不下降。”

## Phase 26：中等 translation-only prior error stress test

本组回应“10% 相邻步长误差修正幅度太小，是否足够写文章”的问题。实验不再加入 rotation 噪声，只测试 translation-only prior error，并把误差放大到 r2 相邻步长的 25% 和 50%。这不是模拟最真实 PEM 误差，而是一个更可观察的 stress test：如果 GS 阶段真的能从图像梯度中获得 pose 方向，应当能在该设置下降低 translation error，同时不引入 rotation drift。

公共设置：

- 配置：`config_files/experiments/us_focus/liver_1m_30k_dc_fast_fidelity_pose_prior.yaml`
- sweep：`r2`
- 迭代：30k
- rotation noise：`0.0 deg`
- rotation：`optimize_rotation=false`
- translation：`optimize_translation=true`
- 25% translation noise：`0.070325 mm`
- 50% translation noise：`0.140650 mm`

有效 Slurm 作业：

```text
59307, 59308, 59309, 59310, 59311
```

| Ablation | Pose 设置 | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | Pose rot err | Pose trans err | Prior trans err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase26_r2_trans25_fixed_30k` | fixed 25% translation prior | 27.2521 | 0.1894 | 0.0312 | 0.5373 | 846,193 | 0.00517 deg | 0.07033 mm | 0.07033 mm |
| `phase26_r2_trans25_poseonly15_17_30k` | 15k-17k translation-only pose-only，lr_trans=1e-4，lambda_trans=0.01 | 27.2592 | 0.1896 | 0.0312 | 0.5372 | 832,790 | 0.00517 deg | 0.06556 mm | 0.07033 mm |
| `phase26_r2_trans50_fixed_30k` | fixed 50% translation prior | 27.1744 | 0.1923 | 0.0315 | 0.5324 | 845,022 | 0.00517 deg | 0.14065 mm | 0.14065 mm |
| `phase26_r2_trans50_poseonly15_17_30k` | 15k-17k translation-only pose-only，lr_trans=1e-4，lambda_trans=0.01 | 27.1773 | 0.1921 | 0.0315 | 0.5327 | 830,745 | 0.00517 deg | 0.13200 mm | 0.14065 mm |
| `phase26_r2_trans50_poseonly15_20_lr3e4_lam001_30k` | 15k-20k translation-only pose-only，lr_trans=3e-4，lambda_trans=0.001 | 27.1912 | 0.1926 | 0.0315 | 0.5330 | 813,932 | 0.00517 deg | 0.12347 mm | 0.14065 mm |

结论：

- 25% translation-only prior error 下，受约束 refinement 把 translation error 从 `0.07033 mm` 降到 `0.06556 mm`，约 `6.8%` 相对改善；PSNR 从 `27.2521` 到 `27.2592`，变化很小。
- 50% translation-only prior error 下，15k-17k 保守窗口把 translation error 从 `0.14065 mm` 降到 `0.13200 mm`，约 `6.2%` 相对改善；延长到 15k-20k 并提高 `lr_trans` 后降到 `0.12347 mm`，约 `12.2%` 相对改善。
- 更强窗口的 PSNR `27.1912` 高于 50% fixed prior 的 `27.1744`，LPIPS `0.1926` 与 fixed 的 `0.1923` 基本持平，rotation 保持 `0.00517 deg` 不漂移。这说明受约束 GS pose refinement 方向正确且稳定。
- 但该改善仍不是大幅校正。论文中不应写成“GS 显著恢复 PEM pose”，更合适的表述是“PEM pose prior 已较准；在中等 translation residual 下，GS 的可微渲染可提供受约束的 residual correction，并保持重建质量不退化”。
- 如果要进一步强化 pose 叙事，下一步应做更结构化的 PEM-like 低频 translation drift，而不是逐帧随机噪声；低频漂移更符合真实 PEM systematic bias，也更可能被图像一致性和 smooth prior 共同校正。

## Phase 27：FiMA PEM prior 接入 Ours

本组开始正式把 `methods/fima` 的姿态估计模型上线到 Ours。使用 adjacent-pair PEM：输入相邻两帧图像，输出 `P_i -> P_{i+1}` 的 6D 相对变换；导出时从第 0 帧开始链式相乘，得到全 sweep 的 `4x4` pose prior，再作为 `model.pose_optimization.prior_poses_abspath` 喂给 Ours。

相关代码提交：

```text
2148210 Add external PEM pose prior support
469aabd Add FIMA to Ours pose prior pipeline
d75e06b Fix FIMA pose prior pipeline resume
abbf2db Apply pose manager during Ours evaluation
129b740 Allow distinct Ours phase tags in FIMA pipeline
```

新增脚本与结果文件：

```text
scripts/eval/eval_pose_prior_against_gt.py
scripts/slurm/pose_experiments/slurm_phase27_fima_to_ours_runner.sh
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/fima/phase27_fima_prior_errors.tsv
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours/us_focus_pose_prior_results.tsv
```

公共设置：

- PEM 配置：`Hydra FiMA paper-edge alias with matching OCC override`
- Ours 配置：`config_files/experiments/us_focus/liver_1m_30k_dc_fast_fidelity_pose_prior.yaml`
- sweep：`r2`
- FIMA 训练：5000 iter，adjacent-pair，链式导出
- Ours 训练：30k iter，1M 初始高斯，开启 density control
- prior 文件：`/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/fima/phase27_fima_priors/phase27_fima_r2_adj_5000_chain_poses.npy`

### PEM prior 误差

| 指标 | 数值 |
| --- | ---: |
| 帧数 | 200 |
| absolute translation mean | 1.083815 mm |
| absolute translation p95 | 3.134193 mm |
| absolute rotation mean | 4.142418 deg |
| absolute rotation p95 | 8.746876 deg |
| adjacent step translation absolute-diff mean | 0.000586 mm |
| adjacent step rotation absolute-diff mean | 0.044392 deg |
| GT adjacent step translation p95 | 0.281299 mm |
| GT adjacent step rotation p95 | 0.023883 deg |
| absolute translation mean / GT step p95 | 3.85x |
| absolute rotation mean / GT step p95 | 173.45x |

结论：

- 这个 PEM prior 不是 5%-10% 小误差。绝对 translation 误差约等于 `3.85x` 相邻帧 p95 步长，绝对 rotation 误差约等于 `173x` 相邻帧 p95 旋转。
- step translation diff 很小，说明平移步长学得接近；主要问题是 rotation step 偏差和链式累积漂移。
- 因此当前 Phase 27 更像“真实 PEM prior 首次接入与误差诊断”，不应作为最终论文中的 PEM quality 数字。

### Ours fixed/refine 结果

`phase27` 首轮发现 evaluation 没有在 test 渲染路径中应用 residual pose；`abbf2db` 修正后，用 `phase27b` 重新跑 fixed/refine。fixed prior 因为数据容器在 setup 时已被 prior 替换，结果与修正前基本一致；refine 结果以 `phase27b` 为准。

| Ablation | Pose 设置 | PSNR | LPIPS | L1 | SSIM | 最终高斯数 | Pose rot err | Pose trans err | Prior rot err | Prior trans err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase27b_r2_fima_prior_fixed_30k` | fixed FiMA prior | 27.2618 | 0.1978 | 0.03118 | 0.54008 | 826,821 | 4.12736 deg | 1.08650 mm | 4.12736 deg | 1.08650 mm |
| `phase27b_r2_fima_prior_transonly_poseonly15_20_30k` | 15k-20k translation-only pose-only residual，之后固定 refined pose | 27.2678 | 0.1994 | 0.03116 | 0.54030 | 789,701 | 4.12736 deg | 1.08923 mm | 4.12736 deg | 1.08650 mm |

结论：

- Ours 当前已经按 residual 方式调制 PEM prior，而不是直接预测 new pose 替代 prior。实现上 `current_pose = prior @ delta`，符合“GS 预测 residual 修正 prior”的设计。
- 对当前 FiMA 5k prior，fixed 建图仍能达到 `27.2618 PSNR / 0.1978 LPIPS`，说明 Ours 对相当差的 prior 也有一定重建鲁棒性。
- translation-only residual refine 没有改善 pose。最终 translation error 从 `1.08650 mm` 轻微变差到 `1.08923 mm`，rotation 因冻结保持 `4.12736 deg`。这说明当 PEM prior 的主问题是 rotation 链式漂移时，仅 translation-only GS residual 无法解决。
- refine 分支 PSNR 略高但 LPIPS 略差，且 pose error 没有改善，因此不能用它证明“GS 把真实 PEM prior 优化得更准”。更合理的解释是 pose-only 窗口和 Gaussian density/control 轨迹带来轻微图像指标差异。

### 对论文叙事的影响

- “PEM 作为 noisy pose prior”这一叙事是成立且必要的；Phase 27 给出了真实 PEM prior 的误差诊断，而不是继续假设 5%-10%。
- 当前 FiMA 5000 iter checkpoint 的 rotation 漂移太大，不能直接支撑“GS refine 真实 PEM pose”。
- GS 中 pose refine 应继续保留 residual 形式：`P_refined = P_prior @ DeltaP`。这比直接产生 new pose 更符合 prior-conditioned optimization，也更容易用小 LR、smooth 和 motion prior 约束。
- 下一步优先级应放在 PEM：保存/选择更好的 checkpoint、降低 rotation drift、增加链式或轨迹级损失、跨 sweep 训练验证。只有 PEM prior 进入小误差或可控低频漂移区间后，再评价 GS residual refine 是否能进一步降低 pose error。
