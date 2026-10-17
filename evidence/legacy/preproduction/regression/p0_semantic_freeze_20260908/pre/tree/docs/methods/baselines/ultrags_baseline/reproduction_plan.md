# UltraGS 复现文档

本文档根据论文 PDF《UltraGS: Real-Time Physically-Decoupled Gaussian Splatting for Ultrasound Novel View Synthesis》整理，重点提取 Methods、实验设置、数据集、训练超参数、评估指标与复现注意事项。论文元数据中给出的代码与数据地址为：<https://github.com/Bean-Young/UltraGS>。本文只基于 PDF 内容整理，未对 GitHub 代码做额外核验。

## 1. 复现目标

UltraGS 的目标是在无外部追踪设备的 freehand/sensorless B-mode ultrasound 场景中做 novel view synthesis。它把 3D Gaussian Splatting 改造成适合超声成像的版本，核心目标包括：

- 从无传感器的超声帧序列中恢复全局一致的 3D 表示；
- 通过 Dynamic Aperture Rectification（DAR）校正超声有效视场和 SfM 初始标定误差；
- 通过 Physically-Decoupled Rendering（PD Rendering）用轻量一阶声学近似建模超声强度；
- 在单张 NVIDIA RTX 3090 上达到实时合成，论文报告速度为 64.69 fps；
- 在 Wild、Phantom、Clinical 三类数据集上达到或超过 NeRF、TensoRF、Ultra-NeRF、3DGS、SuGaR。

## 2. 方法总览

UltraGS pipeline 包含两个核心模块：

1. **Depth-aware Gaussian primitive + DAR**
   - 使用 2D Gaussian disks，而不是标准 3DGS 的体积 3D Gaussians；
   - 将 COLMAP 输出视为 stochastic global anchors，只作为粗初始化；
   - 通过可学习 FoV 参数动态修正超声视场、焦距和初始 SfM 漂移；
   - 用 perspective-correct wave-splat intersection 显式恢复 metric depth。

2. **PD Rendering**
   - 使用低阶 SH（论文为一阶 `l = 1`，4 个参数）表示基础声学响应；
   - 把最终 B-mode intensity 分解为基础响应、深度衰减、镜面反射、体散射；
   - 使用 Softplus 保证声学权重为正，避免非物理能量增益；
   - 目标不是完整模拟声波传播，而是在实时约束下保留足够的一阶物理先验。

## 3. 数据准备

### 3.1 论文使用的数据集

| 数据集 | 内容 | 采集方式 | 用途 |
|---|---|---|---|
| Ultrasound in the Wild Dataset | 10 个膝关节样本 | Butterfly iQ+，30 FPS，handheld longitudinal suprapatellar sweeps | 评估真实手持运动、运动伪影和轨迹变化下的鲁棒性 |
| Phantom Dataset | 9 个腰椎 phantom scans | KUKA 机械臂，包含 tilted 和 perpendicular sweeps | 在可控几何约束和遮挡下评估 |
| Clinical Ultrasound Examination Dataset | 6 个临床病例 | Canon i900，伦理批准 SL2024-KY-29-01 | 评估真实临床扫描变异、遮挡和无约束探头运动 |

Clinical Dataset 进一步包含：

- 3 个 wrist joint/carpal tunnel 病例：volar scanning；
- 3 个 kidney 病例：dual-plane strategy，包括 transverse 和 longitudinal scans。

### 3.2 数据划分

论文沿用既有协议：**每隔 8 帧选 1 帧作为 test set**，其余帧用于训练。

复现时建议：

1. 将每个 scan 序列拆成单帧 B-mode 图像。
2. 保持帧顺序，按 `frame_index % 8 == 0` 选测试帧。
3. 对所有 baseline 使用相同划分。
4. 对 speed/FPS 评估使用同一输出分辨率和同一 GPU。

## 4. COLMAP / SfM 初始化

UltraGS 是 sensorless 方法，不依赖外部 probe tracker。论文使用 COLMAP 做初始化，但强调：

- COLMAP 不是用于获得绝对真实相机位姿；
- COLMAP 输出只作为 coarse coordinate baseline 或 stochastic global anchors；
- 后续由 Gaussian 参数、FoV 参数和 DAR 共同修正 optical-to-acoustic mismatch。

### 4.1 超声图像预处理

由于超声图像低对比度、speckle noise 强、可重复特征少，论文在 COLMAP 前做了两个预处理：

1. **Anisotropic diffusion filtering**
   - 目的：抑制 speckle noise，同时保留结构边缘；
   - 作用：稳定 keypoint detection。

2. **CLAHE（contrast-limited adaptive histogram equalization）**
   - 目的：增强低信噪比区域的局部对比度；
   - 作用：提升跨视角 feature matching 鲁棒性。

论文未给出 anisotropic diffusion 和 CLAHE 的具体参数。复现时需要从开源代码确认，或在补充实验中固定一组参数并报告。

### 4.2 Virtual pinhole proxy

论文明确说明：超声成像本质上是 planar slicing，不是光学 pinhole projection。UltraGS 使用 virtual pinhole model 只是为了提供可微的坐标参数化和优化脚手架。

复现要点：

- 将超声帧序列作为虚拟相机序列输入 SfM；
- 得到的相机位姿/内参只作为初始化；
- 不应把初始位姿当作真实探头位姿；
- 后续依靠可学习 FoV 与 Gaussian 几何自校正。

## 5. Gaussian 表示

### 5.1 从 3D Gaussian 改为 2D Gaussian disk

标准 3DGS 用 anisotropic 3D Gaussians 表示场景，但论文认为这会在超声场景中造成：

- view inconsistency；
- depth map offset；
- tissue-probe interface 定位不准；
- 结构错位，不利于临床诊断。

UltraGS 改用 **2D Gaussian disks** 表示超声平面内的组织界面和结构表面。每个 primitive 包含：

- 中心点：`p_k`；
- 局部正交切向量：`t_u`, `t_v`；
- 2D covariance matrix：`Σ ∈ R^{2x2}`；
- 可学习 appearance/acoustic parameters；
- 参与 splatting 的透明/半透明累积属性。

任意 disk surface 上的 3D 点：

```text
P(u, v) = p_k + u * t_u + v * t_v
```

椭圆分布：

```text
G(p) = exp(-1/2 * [u, v] * Σ^{-1} * [u, v]^T)
```

高密度半透明 2D disks 堆叠后形成 quasi-volumetric representation，更适合模拟连续组织中的声强变化。

## 6. Perspective-Correct Wave-Splat Intersection

UltraGS 为每个像素显式计算与 Gaussian disk 的交点，从而恢复 metric depth。

对图像坐标 `(x, y)`，像素射线被表示为两个正交平面的交：

```text
h_x = (-1, 0, 0, x)^T
h_y = (0, -1, 0, y)^T
```

设：

- `H` 为 local-to-world transform；
- `W` 为 world-to-camera matrix；
- `M = (W H)^{-1}`。

计算交点局部坐标 `(u, v)` 后得到世界坐标：

```text
X_w = [p_k + u * t_u + v * t_v, 1]
```

再变换到相机坐标：

```text
X_c = W * X_w = [X_c, Y_c, z, 1]
```

其中 `z` 即最终 metric depth。该步骤是复现 UltraGS 几何一致性的关键，不能简单沿用标准 3DGS 的 projection/accumulation。

## 7. Dynamic Aperture Rectification（DAR）

超声有效 FoV 会随 beam focusing、depth-dependent imaging ranges 和无约束探头运动变化。UltraGS 为横向和纵向视场引入两个可学习参数：

```text
theta_x, theta_y
```

FoV 参数化为：

```text
FOV_x = 2 * arctan(exp(theta_x))
FOV_y = 2 * arctan(exp(theta_y))
```

内参矩阵：

```text
K = [[f_x, 0, W_img / 2],
     [0, f_y, H_img / 2],
     [0, 0, 1]]

f_x,y = dim_x,y / (2 * tan(FOV_x,y / 2))
```

复现要点：

- `theta_x, theta_y` 与 Gaussian 参数一起优化；
- 初始 FoV 应根据 probe-specific specifications 设置；
- 论文说 FoV 参数被约束在 bounded ranges，但未给出具体上下界；
- DAR 的作用是修正初始 pose drift 和 intrinsic misalignment；
- 消融中去掉 DAR 会导致 geometric distortions 和 structural ghosting。

## 8. PD Rendering

标准 3DGS 通常用高阶 SH（如 `l = 3`）建模光学 view-dependent appearance。UltraGS 认为临床超声视角受限，且强度主要由 wave-matter interactions 决定，因此改用一阶 SH：

```text
l = 1
SH 参数数 = 4
base acoustic response = c(d, k)
```

PD Rendering 将最终超声 intensity 拆为四部分：

```text
I_final = c(d, k) + w_att * I_att + w_refl * I_refl + w_scat * I_scat
```

### 8.1 Depth-dependent attenuation

论文从 Beer-Lambert law 出发：

```text
I(z) = I_0 * exp(-alpha * z)
```

B-mode 通常经过 log compression：

```text
ln(I(z)) = ln(I_0) - alpha * z
```

忽略常数项后，显示域衰减与深度线性相关：

```text
I_att = [-alpha * z, 0, 0]^T
```

论文实现中：

```text
alpha = 1
```

解释：在 log-compressed B-mode 中，attenuation coefficient 和 display gain 不可分辨，固定 `alpha` 相当于归一化选择。

### 8.2 Specular reflection

组织边界处 acoustic impedance mismatch 会产生镜面反射。论文使用二次项：

```text
I_refl = beta * c ⊙ c
```

其中：

- `beta` 是可学习 reflection coefficient；
- `⊙` 是 Hadamard product。

### 8.3 Volumetric scattering

亚波长组织结构产生 diffuse scattering。论文用可学习散射矩阵：

```text
Gamma ∈ R^{3x3}
diag(Gamma) = 0
I_scat = (Gamma * c) ⊙ c
```

零对角约束用于表示 inter-channel coupling。

### 8.4 正约束

论文要求：

```text
w_att, w_refl, w_scat > 0
```

实现方式：

```text
Softplus
```

这保证能量衰减和反射项保持物理合理，防止优化出非物理的 energy gain。

## 9. 训练设置

论文明确给出的训练超参数如下：

| 项目 | 设置 |
|---|---|
| optimizer | Adam |
| total iterations | 5000 |
| Gaussian position learning rate | `1.6e-4` |
| FoV parameter learning rate | `1.0e-3` |
| attenuation coefficient `alpha` | `1` |
| SH order | `l = 1` |
| GPU | NVIDIA RTX 3090 |

论文解释不对称学习率的原因：

- FoV 参数用更高学习率，以便早期快速几何校正；
- Gaussian position 用较低学习率，保持空间 primitive 收敛稳定。

### 9.1 建议训练流程

1. 准备每个 scan 的 B-mode frame sequence。
2. 做 anisotropic diffusion + CLAHE，生成 COLMAP 输入图像。
3. 运行 COLMAP，获得初始 camera poses/intrinsics/sparse anchors。
4. 用 COLMAP 输出初始化 UltraGS 坐标系。
5. 将 probe-specific FoV 转换成 `theta_x, theta_y` 初值。
6. 初始化 2D Gaussian disks。
7. 训练 5000 iterations：
   - 更新 Gaussian positions；
   - 更新 2D covariance / disk parameters；
   - 更新 acoustic response；
   - 更新 FoV 参数；
   - 用 PD Rendering 渲染训练视角；
   - 与 GT B-mode image 计算重建损失。
8. 每隔固定 iteration 在 test frames 上评估 PSNR/SSIM/MSE。
9. 训练结束后测 FPS。

注意：论文没有明确给出损失函数形式、opacity/scale/rotation 学习率、densification/pruning 策略、batch/ray sampling 设置、图像分辨率。严格复现时必须从开源代码或作者补充材料确认。

## 10. 实验设置

### 10.1 Baselines

论文比较方法：

- NeRF；
- TensoRF；
- Ultra-NeRF；
- 3DGS；
- SuGaR；
- UltraGS / Ours。

### 10.2 Metrics

论文使用：

- PSNR，越高越好；
- SSIM，越高越好；
- MSE，越低越好；
- inference speed / FPS，越高越好。

所有实验在 NVIDIA RTX 3090 GPU 上进行。

### 10.3 Test split

所有数据集每隔 8 帧选测试帧。复现时务必对所有 baseline 固定相同测试帧，否则 PSNR/SSIM 不可比较。

## 11. 论文报告的主要结果

### 11.1 Wild Dataset 与 Phantom Dataset

| Method | Wild PSNR | Wild SSIM | Wild MSE | Phantom PSNR | Phantom SSIM | Phantom MSE | Speed fps |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeRF | 20.176 | 0.6834 | 0.0066 | 20.359 | 0.6301 | 0.0058 | 0.28 |
| TensoRF | 24.058 | 0.7528 | 0.0051 | 27.260 | 0.7178 | 0.0029 | 1.61 |
| Ultra-NeRF | 19.140 | 0.6218 | 0.0109 | 25.115 | 0.6814 | 0.0042 | 0.43 |
| 3DGS | 22.327 | 0.7745 | 0.0057 | 27.115 | 0.7142 | 0.0031 | 52.56 |
| SuGaR | 21.392 | 0.6291 | 0.0159 | 28.899 | 0.8843 | 0.0024 | 9.81 |
| UltraGS | 25.454 | 0.7969 | 0.0043 | 29.550 | 0.8966 | 0.0020 | 64.69 |

### 11.2 Clinical Dataset

| Method | Case1 PSNR | Case1 SSIM | Case2 PSNR | Case2 SSIM | Case3 PSNR | Case3 SSIM | Case4 PSNR | Case4 SSIM | Case5 PSNR | Case5 SSIM | Case6 PSNR | Case6 SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NeRF | 18.282 | 0.3388 | 21.880 | 0.5363 | 17.329 | 0.3835 | 23.119 | 0.5985 | 21.634 | 0.5980 | 25.376 | 0.6660 |
| TensoRF | 17.206 | 0.3593 | 22.787 | 0.6378 | 19.999 | 0.5124 | 22.681 | 0.5692 | 22.252 | 0.6816 | 26.158 | 0.7389 |
| Ultra-NeRF | 17.661 | 0.2404 | 18.194 | 0.3994 | 19.030 | 0.3335 | 19.053 | 0.4331 | 20.648 | 0.4971 | 22.450 | 0.7030 |
| 3DGS | 17.280 | 0.4270 | 20.367 | 0.5510 | 18.524 | 0.4420 | 21.807 | 0.6673 | 22.110 | 0.6773 | 25.927 | 0.7333 |
| SuGaR | 18.170 | 0.4601 | 23.831 | 0.7085 | 20.902 | 0.5128 | 23.541 | 0.6381 | 20.135 | 0.5923 | 27.264 | 0.7721 |
| UltraGS | 18.846 | 0.4978 | 24.644 | 0.7380 | 21.570 | 0.5493 | 24.030 | 0.6719 | 22.723 | 0.6911 | 28.181 | 0.7888 |

## 12. 消融实验

论文做了三类消融和鲁棒性分析。

### 12.1 数据稀疏性

观察：

- 单视角 acquisition 会产生明显 structural blur；
- 4 个 sparse views 已能得到较清晰结构；
- 1000 iterations 仍偏 hazy；
- 3000 iterations 已能恢复 sharp anatomical boundaries；
- 5000 iterations 为最终报告设置。

复现建议：

- 固定同一病例，分别使用 1 view、4 views、完整训练集；
- 分别保存 1000、3000、5000 iteration 的渲染图；
- 用 qualitative figure 和 PSNR/SSIM 同时报告。

### 12.2 PD Rendering 物理项消融

| Variant | Wild PSNR | Wild SSIM | Phantom PSNR | Phantom SSIM |
|---|---:|---:|---:|---:|
| w/o `I_att` | 24.115 | 0.7732 | 28.230 | 0.8641 |
| w/o `I_refl` & `I_scat` | 24.380 | 0.7801 | 28.512 | 0.8715 |
| w/o DAR | 23.422 | 0.7520 | 27.105 | 0.8500 |
| w/o PD Rendering | 23.887 | 0.7610 | 27.451 | 0.8320 |
| Full model | 25.454 | 0.7969 | 29.550 | 0.8966 |

论文定性结论：

- 去掉 attenuation：深度相关对比度下降，深部组织能量衰减建模失败；
- 去掉 reflection/scattering：图像变平，缺少 speckle texture 和高对比组织边界；
- 去掉 DAR：几何畸变和 ghosting 明显；
- 用标准高阶 SH 替代 PD Rendering：出现强度不一致和体积伪影。

## 13. 最小可复现实验清单

为了先跑通 UltraGS，建议按以下最小实验执行：

1. 选择 Wild Dataset 中 1 个 scan 或 Clinical Dataset 中 1 个病例。
2. 抽帧并按每 8 帧划分 test set。
3. 对训练帧做 anisotropic diffusion 和 CLAHE。
4. 运行 COLMAP，得到初始相机和点云。
5. 初始化 2D Gaussian disks。
6. 设置：
   - Adam；
   - 5000 iterations；
   - Gaussian position lr `1.6e-4`；
   - FoV lr `1.0e-3`；
   - SH order `l = 1`；
   - attenuation `alpha = 1`；
   - `w_att, w_refl, w_scat` 经 Softplus 保证为正。
7. 开启 DAR，联合优化 Gaussian 与 FoV。
8. 用 PD Rendering 输出 B-mode 图像。
9. 在 test frames 上计算 PSNR、SSIM、MSE。
10. 渲染固定数量 test views，统计 FPS。

## 14. 复现时需要从代码或作者处补齐的信息

PDF 中没有完整披露以下实现细节，严格复现前应优先查代码：

- anisotropic diffusion 的迭代次数、conductance、time step；
- CLAHE 的 clip limit、tile/grid size；
- COLMAP 参数，包括 feature extractor、matcher、相机模型、是否固定内参；
- probe-specific FoV 初值与 bounded range；
- 2D Gaussian 初始化数量、尺度、opacity、covariance 初始化；
- densification、clone、split、prune 策略；
- opacity、covariance、SH/acoustic coefficients 的学习率；
- loss function，例如是否使用 L1、SSIM、DSSIM 或二者混合；
- batch size / image sampling 策略；
- 图像 resize/crop/padding 规则；
- FPS 统计方式，包括 warm-up、渲染分辨率、是否同步 CUDA；
- baseline 的训练步数和调参策略。

## 15. 预期复现现象

如果实现正确，应观察到：

- UltraGS 在 Wild Dataset 上明显优于标准 3DGS 的 PSNR，并保持更高 FPS；
- Phantom Dataset 上 UltraGS PSNR 接近或超过 `29.55`，SSIM 接近或超过 `0.8966`；
- Clinical Case 6 上可达到论文最高报告值附近，即 PSNR `28.181`、SSIM `0.7888`；
- 去掉 DAR 后结构错位和 ghosting 明显；
- 去掉 PD Rendering 后 intensity inconsistency 和 volumetric artifacts 增加；
- 3000 iterations 左右应出现接近最终结果的清晰边界，5000 iterations 收敛到最终质量；
- NeRF/Ultra-NeRF 类方法更容易出现 volumetric blur；
- 标准 3DGS 容易出现 depth offset 或 center-offset artifacts。

## 16. 复现优先级建议

优先复现顺序：

1. 跑通数据划分和 COLMAP 初始化；
2. 在标准 3DGS 或 2DGS 框架上替换为 2D Gaussian disks；
3. 实现 DAR，并验证 FoV 参数会稳定收敛；
4. 实现一阶 SH base acoustic response；
5. 加入 `I_att`，确认深度对比度变化合理；
6. 加入 `I_refl` 和 `I_scat`；
7. 跑 full model 与 `w/o DAR`、`w/o PD Rendering`；
8. 最后复现全部 baseline 表格。

最关键的两个复现风险是 **COLMAP 初始化质量** 和 **DAR 的 FoV 约束范围**。如果初始 anchors 太差且 FoV 无边界，训练可能会走向退化焦距；如果 FoV 约束过紧，又无法修正 ultrasound aperture mismatch。
