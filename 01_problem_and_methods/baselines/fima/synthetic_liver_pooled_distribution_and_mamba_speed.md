# Synthetic Liver Pooled FiMA Feasibility and Mamba Speed

Date: 2026-05-26

## Conclusion

`synthetic_liver` 的六个 sweep (`l1/l2/l3/r1/r2/r3`) 满足先做 pooled FiMA train/test 的条件。

理由：

- 六个 sweep 的图像尺寸、pose convention、frame count 一致。
- 每个 sweep 都能按 GS 使用的 `holdout_interval(trim_each=10, holdout_num_samples=40)` 划分为 `160 train / 40 test`。
- 图像均值、前景比例、相邻帧 L1 差异都在同一量级。
- 相邻帧平移 p95 在 `0.2496-0.2855 mm`，变化约 `1.14x`。
- 相邻帧旋转 p95 的相对变化较大，但绝对值很小，最大也只有 `0.1047 deg`。

因此下一步可以做：

```text
FiMA train = l1/l2/l3/r1/r2/r3 的 train frames 混合
FiMA test  = l1/l2/l3/r1/r2/r3 的 test frames 混合
```

不做 leave-one-sweep-out，也不做每个 sweep 单独训练。

## Important Split Caveat

当前 GS 的 holdout 是均匀插帧式：

```text
N=200, holdout_num_samples=40
```

这意味着 test frames 大约每 4-5 帧出现一次。若严格要求 FiMA sequence window 在全局 frame id 上连续且不能跨 test frame，则 `window_size=32` 无法构造任何训练窗口。

当前 speed benchmark 使用的是 train-only ordered subsequence window：窗口内只包含 train frames，但 frame id 不一定全局连续。这和现有 `FiMASequenceSampler` 的 train-pool 行为更接近，能用于估计模型速度；但正式 pooled FiMA 训练前，需要明确 window 语义：

- 若坚持 adjacent global edge supervision，必须使用更短窗口或改 split；
- 若保持 `window_size=32`，则需要让模型预测 train-subsequence 内相邻样本的相对变换，或改成 mask-supervised edges。

## Distribution Summary

统计使用原始 `images.npy + relative_poses.npy`；同时记录 `images_x2.npy` 形状。

| sweep | image shape | x2 shape | N/train/test | img mean | img std | fg>0.05 | adj L1 | step t mean/p95/max mm | step r mean/p95/max deg | path mm | bbox xyz mm | train-test mean diff | train-test fg diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| l1 | (200, 512, 256) | (200, 256, 128) | 200/160/40 | 43.38121 | 25.17612 | 0.9719 | 8.48918 | 0.26848/0.26867/0.26873 | 0.00711/0.04045/0.05595 | 53.428 | [0.0, 6.113, 51.786] | 0.30077 | 0.00025 |
| l2 | (200, 512, 256) | (200, 256, 128) | 200/160/40 | 44.56402 | 25.43907 | 0.9720 | 8.74032 | 0.26846/0.26867/0.26872 | 0.02597/0.10468/0.11191 | 53.423 | [0.814, 4.904, 51.83] | 0.25816 | 0.00025 |
| l3 | (200, 512, 256) | (200, 256, 128) | 200/160/40 | 44.11186 | 25.12957 | 0.9712 | 8.79309 | 0.24936/0.24958/0.24963 | 0.04276/0.10125/0.23739 | 49.623 | [0.956, 5.238, 47.891] | 0.29046 | 0.00041 |
| r1 | (200, 512, 256) | (200, 256, 128) | 200/160/40 | 45.35027 | 24.85184 | 0.9721 | 8.93155 | 0.28393/0.28545/0.35102 | 0.00636/0.02798/0.45284 | 56.502 | [0.0, 10.729, 54.288] | 0.10361 | 0.00021 |
| r2 | (200, 512, 256) | (200, 256, 128) | 200/160/40 | 48.41332 | 26.82385 | 0.9720 | 9.61170 | 0.28118/0.28130/0.28132 | 0.00237/0.02798/0.03956 | 55.955 | [0.0, 13.53, 53.683] | 0.19809 | 0.00016 |
| r3 | (200, 512, 256) | (200, 256, 128) | 200/160/40 | 49.55347 | 27.00079 | 0.9720 | 9.96884 | 0.25218/0.25231/0.25232 | 0.00211/0.02798/0.02798 | 50.184 | [0.0, 15.021, 47.361] | 0.19635 | 0.00018 |

Aggregate ranges:

| statistic | min | max | ratio | cv |
|---|---:|---:|---:|---:|
| image_mean | 43.381214 | 49.553467 | 1.142 | 0.050 |
| foreground_ratio | 0.971220 | 0.972078 | 1.001 | 0.000 |
| adjacent_image_l1 | 8.489184 | 9.968839 | 1.174 | 0.058 |
| step_translation_p95_mm | 0.249578 | 0.285452 | 1.144 | 0.050 |
| step_rotation_p95_deg | 0.027976 | 0.104678 | 3.742 | 0.621 |
| path_length_mm | 49.622910 | 56.501923 | 1.139 | 0.049 |

Pooled split totals:

```text
train frames = 960
test frames  = 240
```

## Mamba-FiMA Speed Benchmark

Environment:

- GPU: NVIDIA GeForce RTX 4090
- `mamba_ssm`: 1.1.1
- `causal_conv1d`: 1.1.0
- PyTorch: 1.12.1
- `torch.version.cuda`: 11.6
- loaded CUDA module: 11.8

Benchmark config:

- six-sweep pooled train-only windows
- `windows=774`
- `window_size=32`
- `batch_size=1`
- `d_model=160`
- `num_mixer_layers=6`
- forward + backward + AdamW step

| image input | it/s | ms/iter | estimated 40k time | peak GPU memory |
|---|---:|---:|---:|---:|
| `images_x2`, 256x128 | 65.779 | 15.202 | 10.13 min | 3.229 GB |
| downsampled 128x64 | 67.666 | 14.779 | 9.85 min | 3.088 GB |
| downsampled 64x32 | 68.261 | 14.650 | 9.77 min | 3.056 GB |
| original `images.npy`, 512x256 | 63.973 | 15.632 | 10.42 min | 12.856 GB |

## Speed Decision

当前 Mamba-FiMA 速度已经满足目标：即使按 40k iterations 估算，也约 10 分钟，不需要为了速度继续下采样。

推荐正式训练优先使用 `images_x2.npy`：

- 速度几乎不比 `512x256` 慢；
- 显存从约 `12.9 GB` 降到约 `3.2 GB`；
- 更适合后续加 uncertainty head、batch size sweep 或并行实验。

若后续发现收敛需要远超 40k iterations，例如 200k iterations，按当前速度估算也约 50-52 分钟，仍然在 1 小时目标附近。
