# 简化切片深度衰减公式验证记录（2026-06-17）

## 目的

上一版 local-global renderer 的全局衰减线采用接近 Ultragray 的 ray integral 形式：对每个像素射线上的候选高斯计算沿轴向局部窗口的积分响应，再与高斯的 attenuation 属性共同形成 transmission。该设计有效，但公式和实现偏复杂，且与 Ultragray 的叙事距离不够远。

本轮验证一个更贴合 offline one-sweep 超声切片重建的简化公式：保留“local tile 强度线 + global ray 衰减线”的双路径结构，但把 ray 衰减响应从沿射线积分改成 probe/slice 坐标下的切片深度包络响应。目标是：

1. 精度仍能维持或超过当前 integral 衰减版本。
2. ray 衰减核心计算更轻，速度至少快于当前 integral 方案。
3. 数学叙事从“沿射线积分”转向“切片内横向/层厚响应调制”，和 Ultragray 拉开距离。

## 实现

新增配置项：

```yaml
model.slice_render.ray_attenuation_formula: integral | slice_depth
```

`integral` 保持原公式；`slice_depth` 使用新的简化响应：

```text
sigma = sqrt((sx^2 + sy^2 + sz^2) / 3)
response = sigma * exp(-0.5 * ((mu_lat - x_lat)^2 + mu_ele^2) / sigma^2)
psi = ray_strength * response
T_i = tau_i + (1 - tau_i) * exp(-psi)
T_pixel = clamp(prod_i T_i, min_transmission, 1)
modulation = 1 - beta + beta * T_pixel
```

其中 `mu_lat` 和 `mu_ele` 来自世界坐标到 probe 坐标的变换，`x_lat` 是当前像素的横向坐标。这个公式不再做 ray window quadrature，也不依赖高斯 quaternion；因此 `slice_depth` 分支的 ray attenuation 梯度回传到 attenuation、means、scales，quaternion 仍由 local tile 强度线提供监督。

代码改动范围：

- `methods/ours/configs.py`
- `methods/ours/render.py`
- `methods/ours/gaussian_container.py`
- `ops/ours_renderer/ours_renderer/types.py`
- `ops/ours_renderer/ours_renderer/autograd.py`
- `ops/ours_renderer/include/shader/ray_attenuation.cuh`
- `ops/ours_renderer/include/shader/wrapper_shader.cuh`
- `ops/ours_renderer/src/shader/ray_attenuation.cu`
- `ops/ours_renderer/src/shader/wrapper_shader.cu`
- 相关诊断、profile、Slurm sweep 脚本

## 实验设置

数据集和协议：

- dataset: `tus_rec_2024`
- experiment: `preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_echo_alpha_ray_csr_tail_2k`
- one sweep offline reconstruction
- train/test holdout 维持当前 ours 协议
- GPU: `gpu3`, NVIDIA GeForce RTX 4090

核心超参：

- `ray_attenuation_beta=0.60`
- `ray_attenuation_strength=0.30`
- `ray_attenuation_min_transmission=0.55`
- `ray_attenuation_formula=slice_depth`
- `ray_attenuation_quadrature_points=1`
- 训练 10k iterations

主要输出：

- 10k 训练：`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_simple_slice_depth_s0p30_q1_10k_20260617`
- 10k speed profile：`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_simple_slice_depth_s0p30_q1_10k_speed_20260617`
- integral 10k 复测 profile：`outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_q3_10k_speed_reprofile_20260617`

## 结果

### 10k 精度对比

| Method | Formula | qpts | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| local-only / single-strength | none | - | 0.015284 | 31.4357 | 0.8393 | 0.2194 |
| local-global ray | integral | 3 | 0.014746 | 31.7286 | 0.8447 | 0.2052 |
| local-global ray | slice_depth | 1 | 0.014728 | 31.7365 | 0.8447 | 0.2063 |

相对 local-only，`slice_depth` 版本：

- L1 下降约 3.64%。
- PSNR 提升约 0.30 dB。
- SSIM 提升约 0.0054。
- LPIPS 下降约 5.97%。

相对 integral 版本，`slice_depth` 版本：

- L1 略好：`0.014746 -> 0.014728`。
- PSNR 略好：`31.7286 -> 31.7365`。
- SSIM 基本持平。
- LPIPS 略差：`0.2052 -> 0.2063`。

结论：简化公式不是精度倒退方案。它能保留 local-global 衰减带来的主要收益，并且在 L1/PSNR 上略优于 integral 版本。

### 速度对比

当前代码下，同一 profile 脚本复测：

| Method | Formula | qpts | fwd ms | bwd ms | total ms | ray total ms | ray cand |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local-global ray | integral | 3 | 3.132 | 1.092 | 4.224 | 0.841 | 71872 |
| local-global ray | slice_depth | 1 | 3.009 | 1.111 | 4.120 | 0.719 | 71324 |

速度结论：

- ray 衰减核心 `ray total` 约快 14.5%。
- renderer `fwd+bwd total` 约快 2.5%。
- 端到端 renderer 加速不大，因为 local mapping、local raster、autograd glue 和 backward 仍占主要比例。

注意：profile 内部 event 有少量噪声，尤其 `ray total` 的个别 qpts 读数会异常；因此论文/报告中更适合使用保守表述：简化公式降低了 ray attenuation 核心开销，并带来小幅整体 renderer 加速。

## 判断

该 idea 可以继续保留，理由如下：

1. 精度有效：10k 上不弱于 integral，且明显优于 local-only。
2. 速度方向正确：核心衰减核更快，整体 renderer 也小幅更快。
3. 叙事更独立：不再强调 Ultragray 式 ray integral，而是利用 ours 已有的探头坐标系变换，在切片坐标下定义全局传播调制。

该 idea 当前还不是“显著加速”方案。它更适合作为 local-global renderer 的默认候选公式，后续若要进一步加速，应继续从 ray 候选数量、CSR traversal 和 local/global 融合调度上优化。

## 后续建议

1. 把 `slice_depth` 作为新的默认消融候选，与 integral 并列保留。
2. 后续多数据集验证时优先跑：
   - local-only
   - local-global integral
   - local-global slice_depth
3. 如果 slice_depth 在其他数据集也稳定，可以在论文中主推：
   - local tile path: local echo response
   - global ray path: slice-depth propagation modulation
   - 二者共享 probe/world coordinate transform，但在候选选择和物理职责上分离。
