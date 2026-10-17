# Ours Ray-CSR Attenuation Stage C

Date: 2026-06-16
Branch: `dev/ours-ray-csr-attenuation-stage-c-20260616`

## Motivation

Stage B 的 separated attenuation 在概念上更干净，但旧实现复用了 local tile CSR 做全局传播：

- 每个 Gaussian 先被展开到二维 local tile footprint。
- 每个目标 tile 再沿 axial prefix 扫描上游 tile。
- 同一个 Gaussian 的 probe-frame stats 被反复重算，候选也混合了 local response footprint 和 propagation path。

这导致两个问题：速度约比纯 local 慢一个数量级，且候选逻辑不是一个清晰的 ray-level propagation model。

Stage C 改为保留 Ours local tile renderer 作为局部响应分支，同时新增 lateral ray CSR 作为全局 attenuation 分支：

- local branch: 原 `OursTileGaussMapperCUDA + RenderOursSliceForwardCUDA_Impl`。
- global branch: 每个 Gaussian 只展开到 lateral tile ray range，按 axial depth 排序。
- 每个输出 tile 只扫描自身 lateral ray 上深度不超过当前 tile 的候选。

## Implementation

新增配置：

```yaml
model:
  slice_render:
    ray_attenuation_mapper_kind: tile_prefix  # tile_prefix | ray_csr
```

新增 production-style alias:

```text
configs_hydra/experiment/preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_separated_attenuation_ray_csr_tail_2k.yaml
```

核心代码变更：

- `ops/ours_renderer/src/shader/ray_attenuation.cu`
  - 新增 ray-CSR mapper: count, emit, radix sort, ray range identify。
  - 新增 `RayCsrAttenuationCUDA`。
  - 新增 `RayCsrAttenuationBackwardCUDA`。
- `ops/ours_renderer/src/shader/wrapper_shader.cu`
  - `ProjectedSliceForwardRayTileCUDA` 按 mapper kind 选择旧 `tile_prefix` 或新 `ray_csr`。
- Python autograd/config 已接入 mapper kind。

当前 backward 为最小验证版：反传时重建 ray CSR，避免改变 forward 保存张量协议。后续如果 ray-CSR 被保留，可以把 ray CSR 存入 ctx 以进一步减小 backward 开销。

## Verification

Build:

```text
bash -lc 'source ~/.bashrc && conda activate gaussian_splatting && module load cuda/11.8.0 && bash ops/ours_renderer/rebuild.sh'
```

Result:

```text
ours_renderer smoke check passed.
```

All experiments below ran on Slurm node `gpu3`, confirmed device:

```text
NVIDIA GeForce RTX 4090
```

Output root:

```text
outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_csr_stage_c_20260616
```

## 200-iter Speed Smoke

Same TUS-REC-2024 QC sweep, 300k init Gaussians, `us_focus`, 200 iterations.

| Run | Mapper | Test PSNR | Test L1 | Test SSIM | Test LPIPS | Tqdm final speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_local_200` | local only | 17.3151 | 0.0820 | 0.4411 | 0.4372 | 33.39 it/s |
| `tile_prefix_tail_200` | old tile-prefix attenuation | 17.3056 | 0.0821 | 0.4410 | 0.4355 | 16.86 it/s |
| `ours_ray_csr_c_63783` | new ray-CSR attenuation | 17.3095 | 0.0821 | 0.4410 | 0.4344 | 32.17 it/s |

Conclusion:

- 旧 tile-prefix 路线确实是明显速度瓶颈。
- 新 ray-CSR 路线基本恢复到 pure local 速度级别。
- 200 iter 指标差异太小，只能用于 smoke 和速度判断。

## 2k Effect Check

Same TUS-REC-2024 QC sweep, 2k iterations.

| Run | Mapper | Test PSNR | Test L1 | Test SSIM | Test LPIPS | Tqdm final speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_local_2k` | local only | 27.8497 | 0.0242 | 0.8098 | 0.2641 | 95.73 it/s |
| `ray_csr_tail_2k` | new ray-CSR attenuation | 27.8447 | 0.0242 | 0.8097 | 0.2646 | 87.12 it/s |

Conclusion:

- 新 ray-CSR 在 2k 下约比 pure local 慢 9%，而不是旧路线的严重拖慢。
- 当前 Stage B 的 beta/strength/tail 参数迁移到 ray-CSR 后，质量收益仍然是 neutral/slightly negative：
  - PSNR: -0.0050
  - SSIM: -0.0001
  - LPIPS: +0.0005
- 因此 Stage C 解决了“实现低效且候选不合理”的问题，但还没有证明 attenuation 分支本身能稳定改善重建质量。

## Interpretation

这次结果把问题拆清楚了：

1. 旧 attenuation 失败里确实有工程实现因素。旧 tile-prefix global branch 太慢，且候选结构不符合传播建模。
2. 换成 ray-CSR 后，速度已不再是主要阻碍。
3. 但现有 attenuation 参数没有自动转化为质量收益，说明下一步不能再用旧 Stage B 超参直接下结论。

## Next Steps

1. 诊断 `modulation` 分布：
   - mean/min/max/std；
   - 多少像素接近 1；
   - 多少像素被 `min_transmission` clamp；
   - train/test frame 上 attenuation map 是否有结构。
2. 在新 ray-CSR 路线做小 sweep：
   - `ray_attenuation_beta`: 0.1, 0.2, 0.35
   - `ray_attenuation_strength`: 0.04, 0.08, 0.16, 0.32
   - `ray_attenuation_min_transmission`: 0.6, 0.75, 0.8, 0.9
3. 如 modulation 仍几乎恒定，优先检查 attenuation 属性学习是否被当前 loss/初始化压住，而不是继续扩大 10k 正式实验。
4. 若 sweep 出现稳定正收益，再把 ray-CSR backward 保存 CSR，减少 10k/30k 训练开销。
