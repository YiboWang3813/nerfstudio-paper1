# us_focus 生产前补充消融报告

日期：2026-05-31

本文档记录 `ours/us_focus` 在正式生产配置前的补充消融。实验阶段定义为 production preflight ablation，目标是补齐既有报告中尚未充分覆盖的安全边界，而不是重新探索方法方向。

## 实验入口

- Production alias：`experiment=preproduction/ideas/ours/renderer_baseline/main_10k`
- Legacy alias：`experiment=preproduction/ideas/ours/renderer_baseline/fast_fidelity`
- 正式输出根：`outputs/production/ablations/ours/renderer_baseline/production_preflight_fixed`
- 正式汇总表：`outputs/production/ablations/ours/renderer_baseline/production_preflight_fixed/summary.tsv`
- Runner：`scripts/methods/ours/renderer_baseline/slurm/run_production_preflight.sh`
- Submitter：`scripts/methods/ours/renderer_baseline/slurm/submit_production_preflight.sh`
- 数据：`synthetic_liver/l1`
- 公共设置：`seed=42`，10k iter，开启 density control，`l1 + ssim + scale_wall`

## 重要修复

正式消融前发现并修复了一个 CUDA wrapper 参数长度 bug：Python 和 device 侧都使用 14 个 `us_focus_params`，但 C++ wrapper 原先只分配/拷贝 12 个，导致 `inflate_lat/ax` 越界读。越界读会污染 footprint 半径，造成 tile candidate 数暴涨、OOM 或异常慢。修复后 1M 高斯、`k_sigma=2.0` 可在 24GB 4090 上稳定完成 10k，单作业耗时约 1 分 38 秒。

相关提交：`b3573f4 Fix us_focus CUDA parameter vector length`

## 完成状态

- Stage A-F 共 38 条实验全部完成，Slurm 状态均为 `COMPLETED 0:0`。
- 单条 10k 作业耗时约 1:32-2:05；3M 初始高斯作业耗时 1:57。
- 旧输出根中早期失败来自上述 CUDA 越界读和旧 worktree extension，正式结论只使用 `_fixed` 输出根。

## 结论摘要

当前生产默认总体是合理的。最需要锁定的风险点是：

- `tile_size_px=4` 明确不可用：PSNR 只有 23.30，LPIPS 0.507，且高斯反增到 102.8 万。
- `opacity_max=0.1` 虽然 PSNR 最高，但 LPIPS 变差到 0.218，并保留 93.6 万高斯；`opacity_max=0.4` 又过度稀疏，最终只剩 53.3 万且质量下降。生产应继续用 `0.2`。
- elevation 相关参数存在真实 tradeoff：更宽或更陡会提高 PSNR，但通常损害 LPIPS 并减少高斯；更窄/更浅会改善 LPIPS 或保留更多点，但 PSNR/SSIM 下降。
- scale bound、prune、densify 阈值整体很稳，不是当前主要风险源。
- 初始高斯数从 1M 提到 1.5M/2M/3M 会持续降低 LPIPS，但 PSNR 略降、最终点数增加。3M 可运行，不是显存不可行项；是否上生产取决于是否更重视 LPIPS 和显存/存储预算。

## 推荐生产配置

短期保守生产配置建议使用 `preproduction/ideas/ours/renderer_baseline/main_10k`：

| 参数 | 建议 | 理由 |
| --- | --- | --- |
| `k_sigma` | 2.5 | 平衡 PSNR/LPIPS/高斯数；2.0 LPIPS 更好但 PSNR 低，3.0 PSNR 高但 LPIPS 变差 |
| `tile_size_px` | 8 | 8 和 16 质量相近；4 明确失败；保留当前 8 便于与既有速度优化一致 |
| `us_focus_sigma_ele_1` | 0.01 | 0.005 LPIPS 好但 PSNR 低；0.015/0.02 PSNR 高但 LPIPS 明显变差 |
| elevation corridor | current | wide 有 PSNR 收益但 LPIPS 变差；narrow 不推荐 |
| `scale_min/max` | 0.05 / 3.0 | 该组差异极小，无需改默认 |
| `opacity_max` | 0.2 | 0.1/0.4 都有明显副作用 |
| DC end | 8000 for 10k | 继续 DC 到 12k/16k/24k 在 10k 配置下无收益 |
| prune | current 或 strong | strong 稍省点且质量不降，可进入多 sweep 复验 |
| densify_grad | 1.5e-5 或 2e-5 | 差异很小，2e-5 略省点且指标正常 |
| init count | 1M 默认，2M/3M 作为 LPIPS 优先配置 | 2M/3M LPIPS 更低，但最终点数和训练成本更高 |

## Stage A: k_sigma

| exp | PSNR | LPIPS | L1 | SSIM | final gauss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k_sigma=2.0` | 28.199 | 0.182 | 0.02776 | 0.587 | 742,419 |
| `k_sigma=2.25` | 28.240 | 0.189 | 0.02761 | 0.590 | 714,411 |
| `k_sigma=2.5` ref | 28.264 | 0.193 | 0.02753 | 0.591 | 697,220 |
| `k_sigma=2.75` | 28.280 | 0.194 | 0.02748 | 0.592 | 687,908 |
| `k_sigma=3.0` | 28.285 | 0.195 | 0.02746 | 0.592 | 683,485 |
| `k_sigma=3.5` | 28.284 | 0.194 | 0.02746 | 0.592 | 681,928 |

结论：增大 `k_sigma` 到 2.75-3.5 对 PSNR/SSIM 有小幅收益，但 LPIPS 变差。生产默认 2.5 仍是平衡点；若论文图像更看重 LPIPS，可复验 2.0。

## Stage B: tile size

| exp | PSNR | LPIPS | L1 | SSIM | final gauss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tile_size_px=4` | 23.297 | 0.507 | 0.04939 | 0.374 | 1,028,225 |
| `tile_size_px=8` ref | 28.268 | 0.193 | 0.02752 | 0.591 | 697,226 |
| `tile_size_px=16` | 28.264 | 0.193 | 0.02753 | 0.591 | 697,256 |

结论：`tile_size_px=4` 不应进入生产。8 和 16 基本等价，继续用 8。

## Stage C: elevation focus

| exp | PSNR | LPIPS | L1 | SSIM | final gauss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sigma_ele_1=0.005` | 28.201 | 0.183 | 0.02775 | 0.586 | 755,826 |
| `sigma_ele_1=0.01` ref | 28.266 | 0.193 | 0.02752 | 0.591 | 697,383 |
| `sigma_ele_1=0.015` | 28.303 | 0.203 | 0.02741 | 0.593 | 647,069 |
| `sigma_ele_1=0.02` | 28.316 | 0.212 | 0.02738 | 0.595 | 604,714 |
| narrow | 28.168 | 0.188 | 0.02778 | 0.588 | 735,302 |
| current ref | 28.266 | 0.193 | 0.02752 | 0.591 | 697,295 |
| wide | 28.332 | 0.198 | 0.02736 | 0.593 | 659,766 |

结论：这组证明 elevation culling/focus 不是随便设的。更强 focus 会提高 PSNR/SSIM，但 LPIPS 变差且 DC 会删除更多高斯；更弱 focus 会改善 LPIPS 但降低 PSNR。当前值是合理折中。

## Stage D: representation bounds

| exp | PSNR | LPIPS | L1 | SSIM | final gauss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scale_max=2.0` | 28.266 | 0.192 | 0.02752 | 0.591 | 697,792 |
| `scale_max=3.0` ref | 28.272 | 0.193 | 0.02751 | 0.591 | 697,200 |
| `scale_max=4.0` | 28.264 | 0.193 | 0.02752 | 0.591 | 697,042 |
| `scale_min=0.03` | 28.264 | 0.193 | 0.02753 | 0.591 | 696,218 |
| `scale_min=0.05` ref | 28.265 | 0.193 | 0.02752 | 0.591 | 697,376 |
| `scale_min=0.08` | 28.266 | 0.192 | 0.02752 | 0.591 | 699,224 |
| `opacity_max=0.1` | 28.380 | 0.218 | 0.02726 | 0.594 | 936,095 |
| `opacity_max=0.2` ref | 28.268 | 0.193 | 0.02751 | 0.591 | 697,826 |
| `opacity_max=0.4` | 28.179 | 0.204 | 0.02773 | 0.590 | 532,689 |

结论：scale bound 基本不敏感；opacity bound 很敏感，`0.2` 是生产安全点。

## Stage E: density control

| exp | PSNR | LPIPS | L1 | SSIM | final gauss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dc_end=8000` ref | 28.265 | 0.193 | 0.02753 | 0.591 | 697,359 |
| `dc_end=12000` | 28.242 | 0.195 | 0.02755 | 0.591 | 689,410 |
| `dc_end=16000` | 28.238 | 0.195 | 0.02755 | 0.591 | 689,094 |
| `dc_end=24000` | 28.237 | 0.195 | 0.02755 | 0.591 | 689,301 |
| prune light | 28.265 | 0.193 | 0.02752 | 0.591 | 720,535 |
| prune current ref | 28.265 | 0.193 | 0.02752 | 0.591 | 697,347 |
| prune strong | 28.269 | 0.193 | 0.02751 | 0.591 | 682,063 |
| `densify_grad=1e-5` | 28.272 | 0.193 | 0.02751 | 0.591 | 699,789 |
| `densify_grad=1.5e-5` ref | 28.261 | 0.193 | 0.02753 | 0.591 | 696,854 |
| `densify_grad=2e-5` | 28.267 | 0.193 | 0.02751 | 0.591 | 696,449 |

结论：DC 参数比预期稳定。10k 配置下 `dc_end=8000` 最稳；strong prune 可以作为节省点数的候选，但需要多 sweep 复验。

## Stage F: init count

| exp | PSNR | LPIPS | L1 | SSIM | final gauss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `num_points=1.0M` ref | 28.264 | 0.193 | 0.02753 | 0.591 | 697,220 |
| `num_points=1.5M` | 28.234 | 0.181 | 0.02762 | 0.589 | 846,597 |
| `num_points=2.0M` | 28.206 | 0.176 | 0.02770 | 0.588 | 925,407 |
| `num_points=3.0M` | 28.176 | 0.172 | 0.02779 | 0.587 | 991,349 |

结论：3M 初始高斯在修复后可运行，并且最终会被 DC 压到约 99.1 万，不会无限膨胀。更大的初始化显著改善 LPIPS，但 PSNR/SSIM 略降。若生产目标强调感知指标，2M/3M 值得在 r2/r3 和 30k 下复验；若目标强调速度和简洁，1M 仍是默认。

## 生产前剩余建议

1. 在 `r2/r3` 或另一个 sweep 上复验三个候选：当前默认、`k_sigma=2.0`、`num_points=2M`。
2. 对 `prune strong` 做一次多 sweep 复验；若仍不伤质量，可用于控制最终高斯数量。
3. 如果生产训练固定为 30k，需要单独复验 `dc_end=8000` 是否仍最优；本报告只证明 10k 下继续 DC 没收益。
4. 保留 `tile_size_px=4` 为禁用配置，不再用于正式实验。
