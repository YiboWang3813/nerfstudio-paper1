# Ray 衰减线速度优化验证记录（2026-06-17）

## 背景

UltraG-Ray 报告了较高的 inference rendering speed，但其速度数字不能直接和 ours 的训练时 renderer forward+backward 对比：

- UltraG-Ray 的约 680 FPS 来自 spine phantom inference，论文说明该场景最终约 90k Gaussians。
- 其 porcine muscle 场景达到 500k Gaussians 时约 95 FPS。
- ours 当前 profile 是训练相关的 forward+backward，TUS-REC-2024 单 sweep 10k checkpoint 约 205k Gaussians，并且包含 local tile echo path 与 global ray attenuation path。

不过 UltraG-Ray 的速度叙事仍然给出三个有价值的方向：

1. 更强的 Gaussian/candidate count 控制。
2. 更窄的 per-ray relevant set culling。
3. 利用 depth-sorted ray candidates 做 cumulative transmission，避免每个像素重复扫描同一条 ray 前缀。

本轮在 `dev/ours-ray-prefix-speed-20260617` 分支验证这些点。

## 改动

### Ray-bin 索引审计

检查发现 `kernel_ray_csr_attenuation` 和 backward kernel 内部使用像素 `x` 查询 `ray_start/ray_end`。集成路径当前构建 ray CSR 时实际使用 per-pixel ray bin，因此主实验没有受影响；但这对非 per-pixel bin 是潜在越界/错读风险。已改成：

```text
ray_bin = clamp(x / tile_size_px, 0, Tw - 1)
```

当前主路径仍传 `tile_size_px=1`，所以该修复是防御性正确性修复。

### Slice-depth Prefix Forward

对 `ray_attenuation_formula=slice_depth`，单个 Gaussian 对某个 lateral pixel column 的响应不依赖当前深度 `y`，只通过 depth order 决定是否进入当前像素前缀。因此 forward 可以改为 per-column prefix scan：

```text
for x:
  T = 1
  j = ray_start[x]
  for y in [0, H):
    while depth[j] <= y + margin:
      T *= T_j(x)
      j += 1
    modulation[y, x] = 1 - beta + beta * clamp(T, minT, 1)
```

这与旧逐像素前缀重复扫描在数学上等价。Backward 暂时保持原逐像素 VJP，因此训练梯度路径不变，只减少 forward ray modulation 的重复工作。

### Candidate k-sigma

新增配置：

```yaml
model.slice_render.ray_attenuation_candidate_k_sigma: -1.0
```

- `-1.0`: 沿用 local renderer 的 `k_sigma`，保持默认行为。
- `>0`: 只用于 ray-CSR candidate construction，不改变 local tile mapper 的候选范围。

该参数用于验证 UltraG-Ray 风格的更窄 relevant set culling 是否能减少 ray candidates。

## 验证

数据集和 checkpoint：

- dataset: `tus_rec_2024`
- checkpoint: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_simple_slice_depth_s0p30_q1_10k_20260617/stage_e_slice_depth_ray_s0p30_q1_10000`
- GPU: `gpu3`, NVIDIA GeForce RTX 4090
- profile: warmup 5, repeats 20, frames 8

### Prefix Forward

| Variant | cand k | ray cand | ray modulation ms | fwd ms | bwd ms | total ms | profile L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slice-depth before prefix | -1.0 | 71324 | 0.475 | 2.976 | 1.092 | 4.068 | 0.015612 |
| slice-depth prefix | -1.0 | 71324 | 0.307 | 2.796 | 1.118 | 3.914 | 0.015612 |

效果：

- ray modulation 降低约 35.4%。
- forward+backward total 降低约 3.8%。
- profile L1 不变，说明 prefix forward 与旧公式输出一致。

### Candidate Culling

| Variant | cand k | ray cand | ray modulation ms | total ms | profile L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix default | -1.0 | 71324 | 0.307 | 3.914 | 0.015612 |
| prefix cull | 2.0 | 46672 | 0.230 | 3.907 | 0.015620 |
| prefix cull | 1.5 | 26845 | 0.170 | 3.756-3.912 | 0.015671 |

解释：

- `candidate_k_sigma=2.0` 把 ray candidates 降低约 34.6%，profile L1 只增加约 `8e-6`，属于很稳的裁剪点。
- `candidate_k_sigma=1.5` 进一步把 candidates 降低约 62.4%，但 profile L1 已有可见上升，属于偏激进设置。
- total ms 对 candidate k 的响应不如 modulation ms 明显，因为 backward 仍是旧逐像素 VJP，且 profile 有 event 噪声。

### 2k 训练 Sanity

| Variant | cand k | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix default | -1.0 | 0.019486 | 29.4450 | 0.8060 | 0.2538 |
| prefix cull | 2.0 | 0.019447 | 29.4636 | 0.8062 | 0.2523 |

`candidate_k_sigma=2.0` 的 2k sanity 没有显示训练不稳定，指标还略优于 default prefix 运行。该结果不能替代 10k 结论，但足够说明候选裁剪不是立即失败。

### 10k 验证

| Method | Formula | cand k | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| local-only / single-strength | none | - | 0.015284 | 31.4357 | 0.8393 | 0.2194 |
| local-global ray | integral | -1.0 | 0.014746 | 31.7286 | 0.8447 | 0.2052 |
| local-global ray | slice_depth | -1.0 | 0.014728 | 31.7365 | 0.8447 | 0.2063 |
| local-global ray | slice_depth | 2.0 | 0.014754 | 31.7252 | 0.8445 | 0.2058 |

`candidate_k_sigma=2.0` 的 10k 质量结论：

- 相对 local-only，L1 降低约 3.47%，PSNR 提升约 0.29 dB，SSIM 提升约 0.0052，LPIPS 降低约 6.18%。
- 相对默认 `slice_depth`，L1/PSNR/SSIM 略低，但差距很小；LPIPS 略好。
- 相对 integral，整体基本持平，LPIPS 略差于 integral 但优于默认 `slice_depth`。

训练后的 `candidate_k_sigma=2.0` checkpoint profile：

| Variant | cand k | ray cand | ray modulation ms | fwd ms | bwd ms | total ms | profile L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local-only profile path | - | 0 | 0.000 | 2.305 | 0.642 | 2.947 | 0.036899 |
| prefix cull | 2.0 | 46692 | 0.233 | 2.753 | 1.164 | 3.918 | 0.015604 |

相对最初未 prefix 的简化衰减 profile（`4.120 ms`），最终 `prefix + cand k=2.0` 为 `3.918 ms`，总 renderer 时间下降约 4.9%。相对同 checkpoint 的 local-only profile path，仍慢约 33%。因此该优化降低了 ray 分支开销，但还没有把 local-global renderer 变成近似零成本模块。

### Prefix Backward

在 `slice_depth` 衰减公式下，单个候选 Gaussian 对同一 lateral column 的响应不随像素深度 `y` 改变，只通过 `depth[j] <= y + margin` 决定从哪一行开始进入透射率前缀。因此 backward 也可以从逐像素 candidate scan 改成 per-column reverse accumulation：

```text
for x:
  suffix = 0
  y_cursor = H - 1
  for candidate j from far to near:
    add all pixels y affected by j into suffix += dL/dT_y * T_y
    dL/dT_j = suffix / T_j
    run one Gaussian VJP for candidate j
```

该路径只作用于 `ray_attenuation_formula=slice_depth`；integral 公式继续使用旧逐像素 backward。

同一 10k `candidate_k_sigma=2.0` checkpoint 的 profile 对比：

| Variant | fwd ms | bwd ms | total ms | profile L1 |
| --- | ---: | ---: | ---: | ---: |
| before prefix backward | 2.753 | 1.164 | 3.918 | 0.015604 |
| after prefix backward | 2.784 | 1.079 | 3.862 | 0.015604 |

效果：

- backward 降低约 7.3%。
- forward+backward total 降低约 1.4%。
- profile L1 不变，说明该优化没有改变 forward 输出。
- total 收益有限，说明当前速度瓶颈已经更多落在 forward ray-CSR 构建、D2H total-candidate sync、ray modulation 与 local/global 固定开销上。

2k sanity 训练对比：

| Variant | cand k | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| before prefix backward | 2.0 | 0.019447 | 29.4636 | 0.8062 | 0.2523 |
| after prefix backward | 2.0 | 0.019429 | 29.4633 | 0.8063 | 0.2545 |

训练指标基本等价，说明新 backward 没有破坏 2k 收敛。

### Fixed-Capacity Ray CSR

旧 ray-CSR 构建路径需要先在 GPU 上 scan 每个 Gaussian 触达的 ray 数量，再把 `offsets[n_gauss]` 拷回 CPU，用 total candidate 数量分配 `keys/values` tensor。这个 D2H copy 会同步当前 CUDA stream，profile 中约为 `0.232 ms`。

尝试了两类无 D2H 路径：

1. **Per-ray fixed capacity**：每个 ray column 预留 K 个槽位。
2. **Fixed total capacity**：整个 slice 预留固定 total candidate 槽位。

容量诊断：

- checkpoint: `stage_e_slice_depth_ray_s0p30_ck2p0_q1_10000`
- frames: all 365 train-container frames
- `ray_candidate_k_sigma=2.0`
- max total candidates: `62594`
- max candidates per ray: `289`

因此：

- per-ray K=288 不安全，最大帧需要 289；
- per-ray K=292/300/320 安全，但需要排序 `384*K` 个槽位，容量在 112k 以上；
- fixed-total 65536 覆盖全 sweep 最大 total candidates，排序容量更接近真实 total。

Profile 对比：

| Variant | capacity | overflow | fwd ms | bwd ms | total ms | D2H ms | emit ms | sort ms | profile L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact CSR | dynamic | - | 2.784 | 1.079 | 3.862 | 0.232 | 0.008 | 0.076 | 0.015604 |
| per-ray fixed | 292/ray = 112128 | 0 | 2.783 | 1.071 | 3.854 | 0.000 | 0.025 | 0.073 | 0.015604 |
| fixed total | 65536 | 0 | 2.744 | 1.061 | 3.805 | 0.000 | 0.024 | 0.069 | 0.015604 |

Fixed-total 65536 是当前最好结果：

- 去掉 D2H sync；
- 无 overflow；
- profile L1 不变；
- forward 降低约 1.4%；
- forward+backward total 降低约 1.5%。

2k sanity：

| Variant | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| before fixed total | 0.019429 | 29.4633 | 0.8063 | 0.2545 |
| fixed total 65536 | 0.019495 | 29.4380 | 0.8061 | 0.2535 |

2k 结果在正常随机波动范围内，说明 fixed-total 路径没有明显破坏训练。

## 当前判断

1. Prefix forward 值得保留。它不改变 slice-depth 衰减公式，稳定减少 forward ray modulation 的重复扫描。
2. `ray_attenuation_candidate_k_sigma=2.0` 可以作为当前速度配置候选；它 10k 质量基本持平，但不是绝对最优精度配置。
3. 默认追求最高重建质量时仍可使用 `candidate_k_sigma=-1.0`；默认追求速度/质量平衡时可使用 `candidate_k_sigma=2.0`。
4. Prefix backward 值得保留，但收益偏小。
5. Fixed-total capacity CSR 值得保留为速度实验开关；它对当前 TUS-REC sweep 有稳定小收益，但容量依赖数据和 Gaussian 数量，不应无条件设为全局默认。
6. 继续加速时，应优先考虑：
   - 更强的 Gaussian count / contribution-aware pruning；
   - 更细的 candidate k 取值扫描；
   - 训练前期关闭或弱化 ray branch 的 schedule；
   - 针对 fixed-total 路径做容量自适应或 cached-capacity 设计。

## 输出路径

- prefix profile: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_slice_depth_10k_speed_20260617`
- `ck=2.0` profile: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_slice_depth_ck2p0_speed_20260617`
- `ck=1.5` profile: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_slice_depth_ck1p5_speed_20260617`
- prefix default 2k sanity: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_slice_depth_s0p30_q1_2k_20260617`
- `ck=2.0` 2k sanity: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_slice_depth_ck2p0_s0p30_q1_2k_20260617`
- `ck=2.0` 10k validation: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_slice_depth_ck2p0_s0p30_q1_10k_20260617`
- `ck=2.0` trained checkpoint profile: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_slice_depth_ck2p0_10k_trained_speed_20260617`
- prefix backward profile: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_backward_ck2p0_speed_20260617`
- prefix backward 2k sanity: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_prefix_backward_ck2p0_s0p30_q1_2k_20260617`
- fixed-capacity capacity scan: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_fixedcap_capacity_scan_20260617`
- fixed-total 65536 profile: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_fixedtotal_65536_speed_20260617`
- fixed-total 65536 2k sanity: `outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_fixedtotal_65536_ck2p0_s0p30_q1_2k_20260617`
