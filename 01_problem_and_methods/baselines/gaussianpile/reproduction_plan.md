# Method

## 0) 复现目标与实现边界

GaussianPile 是一个 **slice-based volumetric reconstruction / compression** 方法。它把 3D Gaussian Splatting 改造成适合有限切片厚度成像的体数据表示，核心是 **focus-aware forward model**：

- 每个 3D Gaussian 仍在 world 坐标中表示体数据。
- 每张输入 slice 被看作一个 virtual camera / imaging plane。
- 渲染时先把 Gaussian 变到当前 slice 的 camera 坐标系。
- 用成像系统的 elevational / axial sensitivity $h(-z_c)$ 调制 Gaussian，得到 **Focus Gaussian**。
- 再取 Focus Gaussian 在 $z_c=0$ 成像平面上的 2D marginal distribution。
- 所有 Gaussian 对像素的贡献 **additive accumulation**，不做 alpha compositing 或遮挡排序。

这篇是 Gaussian 方法，适合落到你的 `gaussian container / gaussian solver / density control / ops/{method}_renderer / computer` 架构中。但 loss/metric 仍由仓库现有 `computer` 基础设施负责，renderer 只输出 rendered slice 和必要 diagnostics。

## 1) Gaussian 表示

每个 Gaussian primitive 在 world 坐标中表示为：

$$
g(\mathbf{x})
=
\alpha
\exp\left(
-\frac12(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)
\right).
$$

可学习参数：

$$
\mathcal{G}=(\mu,s,q,\alpha).
$$

- $\mu\in\mathbb{R}^3$：Gaussian center。
- $s\in\mathbb{R}^3$：anisotropic scale。
- $q$：unit quaternion，表示 spatial orientation。
- $\alpha$：intensity coefficient / opacity。

协方差沿用标准 3DGS 风格：

$$
S=\operatorname{diag}(\mathrm{mod}\cdot s),
$$

$$
R=\operatorname{quatToMatrix}(q),
$$

$$
\Sigma=RSS^TR^T.
$$

GaussianPile 不使用 standard 3DGS 的 spherical harmonics，因为 slice-based volumetric imaging 的 intensity 不是 BRDF / view-dependent color。去掉 SH 是方法的一部分，也减少每个 primitive 的参数量。

## 2) 坐标系与 slice camera

每次训练/渲染随机选择一张 slice，对应该 slice 的 virtual camera。对当前 slice，使用 view matrix $[R_c|t]$ 将 world-space Gaussian 变换到 camera-space / slice-space：

$$
\mu_c=R_c\mu+t,
$$

$$
\Sigma_c=R_c\Sigma R_c^T.
$$

这里的 camera 坐标系约定：

- $x_c,y_c$：slice plane 内坐标。
- $z_c$：垂直 slice plane 的 elevational / axial 坐标。
- 成像平面为 $z_c=0$。

关键实现点：

- 不是把 slice 内每个像素变回 world 后直接算原 Gaussian。
- 论文方法先在 camera-space 中加入 focus/sensitivity 模型，得到 Focus Gaussian，再投到 2D。
- 对不同 slice，同一个 world Gaussian 会因为不同 $R_c,t$ 得到不同 $\mu_c,\Sigma_c$，从而在不同成像平面产生不同 contribution。

## 3) Focus-aware physical model

论文假设 imaging system 的 3D PSF 是空间不变 Gaussian：

$$
psf(\mathbf{x}_c)
\propto
\exp\left(
-\frac12
\left(
\frac{x_c^2}{\sigma_x^2}
+
\frac{y_c^2}{\sigma_y^2}
+
\frac{z_c^2}{\sigma_z^2}
\right)
\right).
$$

其中 $\sigma_z$ 反映 elevational / axial direction 的 focus ability，也就是有限切片厚度。

定义轴向 sensitivity map：

$$
h(-z_c)=
\exp\left(
-\frac{z_c^2}{2\sigma_z^2}
\right).
$$

解释：

- $\sigma_z\to\infty$：all-in-focus，轴向几乎不衰减，接近原始 3DGS / line-integral 类方法。
- $\sigma_z\to0$：zero-thickness，只允许精确位于 slice plane 的 primitive 贡献。
- 有限 $\sigma_z$：有限厚度 Focus Gaussian，适合 ultrasound、light-sheet microscopy 等 slice-based imaging。

$\sigma_z$ 可由系统参数估计或标定；若未知，论文建议根据扫描步长 $\delta_z$ 取：

$$
\sigma_z\approx\delta_z.
$$

补充材料给出估计式：

$$
\sigma_z\approx0.27\cdot\frac{\lambda}{NA_{ele}},
$$

optical system 可近似：

$$
\sigma_z\approx\frac{\lambda}{NA^2},
$$

ultrasound system 可近似：

$$
\sigma_z\approx k\frac{\lambda F}{D_{ele}},
$$

其中 $F$ 是 focal length，$D_{ele}$ 是 elevational effective aperture。

## 4) Gaussian Piling forward model

### 4.1 Convolution interpretation

单个 camera-space Gaussian：

$$
g_c\sim\mathcal{G}(\mu_c,\Sigma_c).
$$

rendered slice intensity 来自 $g_c$ 与 sensitivity $h$ 的轴向卷积，并在 $z_c=0$ 取值：

$$
I(x_c,y_c)
=
\left[
h(-z_c)*g_c(\mathbf{x}_c)
\right]_{z_c=0}.
$$

积分形式：

$$
I(x_c,y_c,z_c=0)
=
\int_{-\infty}^{\infty}
h(t)\,g_c(x_c,y_c,t)\,dt.
$$

因为两个 Gaussian 的乘积仍是 Gaussian，所以可以解析得到 Focus Gaussian。

### 4.2 Axial re-parameterization

把有限切片厚度编码为 camera-space inverse covariance 的轴向注入：

$$
\Sigma_e^{-1}
=
\Sigma_c^{-1}
+
\frac{e_3e_3^T}{\sigma_z^2},
$$

$$
\mu_e
=
\Sigma_e\Sigma_c^{-1}\mu_c,
$$

其中：

$$
e_3=[0,0,1]^T.
$$

$f\sim\mathcal{G}(\mu_e,\Sigma_e)$ 称为 **Focus Gaussian**。

直觉：

- $\Sigma_e^{-1}$ 增加了 $z_c$ 方向 precision。
- $\Sigma_e$ 在轴向支撑上收缩。
- lateral structure 由 $\Sigma_c$ 的 $x,y$ 分量保留。
- $\mu_e$ 会被重新定位到 focus-aware effective mean。

### 4.3 Opacity modulation

乘积 Gaussian 中剩余常数项被吸收到 opacity modulation：

$$
opacity_r
=
\exp\left(
-\frac12
\left(
\mu_c^T\Sigma_c^{-1}\mu_c
-
\mu_e^T\Sigma_e^{-1}\mu_e
\right)
\right).
$$

物理含义：

- 离当前 slice focus zone 越远，$opacity_r$ 越小。
- 该项抑制离焦 Gaussian 的跨切片 ghosting / floating artifacts。
- 可根据 $opacity_r$ 或最终 $\tilde{\alpha}$ 做 culling 以减小计算量。

### 4.4 Screen-space marginal projection

Focus Gaussian 在成像平面上的 2D Gaussian 取 $\Sigma_e,\mu_e$ 的 $x,y$ marginal：

$$
\Sigma_{2d}
=
\begin{bmatrix}
\Sigma_e[0,0] & \Sigma_e[1,0]\\
\Sigma_e[1,0] & \Sigma_e[1,1]
\end{bmatrix},
$$

$$
\mu_{2d}
=
[\mu_e[0],\mu_e[1]]^T.
$$

最终 intensity coefficient：

$$
\tilde{\alpha}
=
\alpha\cdot opacity_r\ /\sqrt{\det(\Sigma_{2d})}.
$$

$1/\sqrt{\det(\Sigma_{2d})}$ 来自对 $z_c$ 方向解析积分后的归一化系数，用来在 footprint 变大/变小时保持总能量一致。实现时需要对 $\det(\Sigma_{2d})$ 做 clamp / epsilon，避免数值爆炸。

### 4.5 Additive rasterization

对像素 $p$：

$$
I(p)
=
\sum_{i\in\mathcal{N}(p)}
\tilde{\alpha}_i
\exp\left(
-\frac12
d_i^T\Sigma_{2d,i}^{-1}d_i
\right),
$$

其中：

- $\mathcal{N}(p)$：screen-space footprint 与 pixel $p$ 重叠的 Gaussian 集合。
- $d_i=p-\mu_{2d,i}$。

slice imaging 的贡献是强度积分，天然相加；因此 GaussianPile 不做：

- depth sorting。
- alpha compositing。
- front-to-back transmittance。
- camera RGB-style SH color。

renderer 只需要 additive accumulation。

## 5) Backward / differentiability

论文实现了 CUDA backward。梯度链路：

1. pixel loss gradient $\partial L/\partial I(p)$。
2. additive rasterization 回传到 $\tilde{\alpha},\mu_{2d},\Sigma_{2d}^{-1}$。
3. screen-space projection 回传到 $\mu_e,\Sigma_e$。
4. focus re-parameterization 回传到 $\mu_c,\Sigma_c$。
5. camera transform 回传到 world-space $\mu,\Sigma$。
6. covariance factorization 回传到 $s,q$，同时 $\tilde{\alpha}$ 回传到 $\alpha$。

补充材料中特别强调 determinant term 的梯度：

$$
\tilde{\alpha}
=
\alpha\cdot opacity_r\cdot [\det(\Sigma_{2d})]^{-1/2},
$$

因此 $\alpha$ 和 covariance gradients 是耦合的。实现 CUDA backward 时不能只把 determinant 当成常数。

如果先做 PyTorch prototype，可以依赖 autograd 验证公式；正式版本应在 `ops/gaussianpile_renderer` 中实现 forward/backward。

## 6) Optimization and adaptive control

论文训练流程遵循 3DGS loop，但有 GaussianPile-specific 修改：

- 每轮 iteration 随机选择一个 virtual camera / slice 渲染。
- 优化 rendered slice 与 target slice 的 photometric loss。
- 使用 densify-and-prune 调整 Gaussian 分布。

论文给出的实现参数：

| 参数 | 论文值 |
|---|---:|
| optimizer | Adam |
| training iterations | `30k` |
| initial Gaussians $M$ | `1000k` |
| position lr | `0.0006` |
| opacity lr | `0.02` |
| scale lr | `0.002` |
| rotation lr | `0.001` |
| lr schedule | exponential decay to 0.1 initial value |
| opacity minimal threshold $\tau$ | `0.02` |
| adaptive control start | `500` |
| adaptive control stop | `25000` |
| gradient threshold | `0.00005` |
| paper loss weight $\lambda$ | `0.2` |

loss/metric 由现有 `computer` 负责。若要复现论文 loss，在 computer 配置中使用：

$$
L=L_1+\lambda L_{ssim},
\qquad
\lambda=0.2.
$$

不要在 renderer 或 Gaussian container 中实现 loss。

## 7) Prune and densification

论文描述的 adaptive control 依据：

- tile-wise focus radii。
- Gaussian opacity scores。
- gradient threshold。

目标：

- refine Gaussian point cloud distribution。
- 保证结构覆盖。
- 防止 axial overgrowth。
- 剔除对当前 slice-based volume 无贡献的 primitive。

建议落地策略：

- prune：删除最终 opacity / focus opacity 低于 $\tau=0.02$ 的 Gaussians。
- prune：删除 focus radius 过大或长期离焦的 Gaussians。
- densify：对高梯度 Gaussian split/clone。
- densify/prune 仅在 500 到 25000 iterations 之间启用。
- 所有阈值 config 化，因为论文没有给出 tile-wise focus radius 的完整公式和 split/clone 细节。

## 8) Quantization and compression

这是方法的一部分，但可作为训练后可选 pipeline 实现。

流程：

1. 对所有 Gaussians 按 Morton / Z-order 排序，暴露空间局部相关性。
2. position：归一化到 scene bounding box 后量化，典型每轴 14 bit。
3. opacity：映射到 $[0,1]$，12 bit 量化。
4. scale：先 log-transform，再 12 bit 量化。
5. quaternion：限制到 positive half-sphere，再每分量 12 bit 量化。
6. 每个 attribute stream 做 delta encoding。
7. 使用 LZMA entropy compression。

注意：

- 训练和渲染可先不实现 compression。
- 如果目标是复现论文 compression ratio，需要实现该 pipeline。
- quantization 不应改变 renderer 数学；它是 checkpoint/export 后处理。

## 9) Differentiable voxelization

为 3D 评估和体可视化，论文集成 differentiable voxelizer：

$$
V=\mathcal{V}(\mathcal{G}_3),
$$

其中 $\mathcal{G}_3$ 是 world-space original Gaussians，即 **Sec. 4.2 focus re-parameterization 之前的 Gaussians**。

体素强度：

$$
\sigma(\mathbf{x})
=
\sum_{i=1}^{M}
g_i^3(\mathbf{x};\mu_i,\Sigma_i).
$$

实现：

- 将 3D space 划成 uniform $8^3$ tiles。
- 每个 tile 只保留 $3\sigma$ ellipsoid 与 tile 相交的 Gaussians。
- tile 内对体素累加 world-space Gaussian contribution。
- forward/backward 均可 CUDA 加速。

voxelizer 是可选模块：

- 若只需要 2D slice reconstruction/rendering，可先不实现。
- 若需要 3D metric、volume visualization 或体正则，再实现。
- loss/metric 仍交给现有 `computer`。

## 10) 代码模块落盘建议

### 10.1 Gaussian container

container 保存 per-Gaussian raw 参数和 activation：

- `xyz_raw` / `mu`。
- `scale_raw`。
- `rotation_raw` / `quat_raw`。
- `opacity_raw`。

不需要 SH。

container 应提供：

- `get_xyz()`。
- `get_scaling()`。
- `get_rotation()` / `get_quaternion()`。
- `get_covariance()`。
- `get_opacity()`。
- `append_gaussians(...)` / `remove_gaussians(mask)`。

### 10.2 Gaussian solver

solver 负责：

- Adam parameter groups。
- position / opacity / scale / rotation 的独立学习率。
- lr exponential decay 到初值 0.1。
- density control 后 resize optimizer state。
- 接收 computer scalar loss 做 backward/step。

### 10.3 Density control

density control 持有 container 和 solver：

- 500 到 25000 iterations 启用。
- 使用 gradient threshold `0.00005`。
- 使用 opacity threshold `0.02`。
- 维护 tile-wise focus radius / opacity / gradient statistics。
- prune / split / clone 后同步 container tensor、optimizer state、statistics buffer。

### 10.4 `ops/gaussianpile_renderer`

这是核心自定义算子，应实现：

1. world Gaussian 到 camera-space：$\mu_c,\Sigma_c$。
2. focus re-parameterization：$\Sigma_e,\mu_e$。
3. opacity modulation：$opacity_r$。
4. screen-space projection：$\mu_{2d},\Sigma_{2d}$。
5. coefficient：$\tilde{\alpha}$。
6. tile/binning/culling。
7. additive rasterization。
8. CUDA backward。

实现约束：

- 不做 SH。
- 不做 alpha compositing。
- 不做 depth sorting。
- 不复用 RGB camera 3DGS renderer 的可见性模型。
- 需要支持每个 slice 的不同 camera/view matrix。
- loss/metric 不在 op 内计算。

### 10.5 Optional ops

- `ops/gaussianpile_voxelizer`：world-space Gaussian voxelization。
- `compression/gaussianpile_quantizer`：Morton sort + quantization + delta + LZMA。

### 10.6 模块责任边界速查

| 内容 | 建议落盘位置 |
|---|---|
| $\mu,s,q,\alpha$ raw 参数 | `gaussian container` |
| covariance activation | `gaussian container` |
| Adam + lr schedule | `gaussian solver` |
| prune/densify | `density control` |
| world-to-camera transform | `ops/gaussianpile_renderer` |
| focus Gaussian | `ops/gaussianpile_renderer` |
| additive rasterization | `ops/gaussianpile_renderer` |
| voxelization | `ops/gaussianpile_voxelizer` |
| compression | export/compression module |
| loss/metric | existing `computer` |

# Method-related Settings

## 1) 必需输入

每个训练样本至少包含：

- `image`: 单通道 slice image。
- `pose/view_matrix`: 当前 slice 的 world-to-camera matrix $[R_c|t]$。
- `pixel_grid`: slice plane 的 $x_c,y_c$ 坐标或可生成它的 spacing/origin。
- `slice_spacing` 或 `sigma_z`: 若无系统 PSF 标定，使用 $\sigma_z\approx\delta_z$。
- `mask/valid_region`: 可选，由 data pipeline 或 computer 处理。

## 2) 默认方法参数

| 参数 | 论文值 | 落盘位置 |
|---|---:|---|
| initial Gaussians $M$ | `1000k` | container init |
| training iterations | `30k` | training config |
| optimizer | Adam | solver |
| position lr | `0.0006` | solver |
| opacity lr | `0.02` | solver |
| scale lr | `0.002` | solver |
| rotation lr | `0.001` | solver |
| lr final multiplier | `0.1` | solver |
| adaptive start | `500` | density control |
| adaptive stop | `25000` | density control |
| gradient threshold | `0.00005` | density control |
| opacity threshold $\tau$ | `0.02` | density control / renderer culling |
| $\sigma_z$ fallback | $\delta_z$ | renderer config |
| paper loss $\lambda$ | `0.2` | computer config |
| voxel tile size | $8^3$ | voxelizer |
| voxel cull support | $3\sigma$ ellipsoid | voxelizer |

## 3) 论文未公开但代码必须配置的参数

- initial position distribution / random perturbation。
- scale initialization。
- opacity initialization。
- quaternion initialization。
- tile-wise focus radius formula。
- split / clone decision rule。
- max/min Gaussian count。
- screen-space radius cutoff。
- determinant clamp epsilon。
- covariance positive floor。
- exact D-SSIM implementation in computer。
- quantization bounds and bit packing format。

## 4) 最小验证协议

1. **Camera transform test**：identity view 下 $\mu_c=\mu,\Sigma_c=\Sigma$。
2. **Focus limit test**：$\sigma_z\to\infty$ 时 $\Sigma_e\to\Sigma_c,\mu_e\to\mu_c,opacity_r\to1$。
3. **Thin-slice test**：$\sigma_z$ 变小时，离 $z_c=0$ 越远的 Gaussian $opacity_r$ 越小。
4. **Projection test**：$\Sigma_{2d}$ 是 $\Sigma_e$ 的 $x,y$ marginal，且 positive definite。
5. **Additive renderer test**：两个相同 Gaussian 的输出等于单个输出的 2 倍。
6. **Computer integration test**：renderer 输出可被现有 computer 接收并产生可反传 scalar loss。
7. **Density control test**：prune/densify 后 container tensor、optimizer state、statistics buffer 长度一致。
8. **Voxelizer test**：world-space Gaussian voxelization 不使用 focus re-parameterization。
9. **Compression roundtrip test**：quantize/dequantize 后 render 误差在可接受范围内。

## 5) 推荐保存的复现产物

- `config.yaml`: Gaussian init、$\sigma_z$、renderer epsilon、adaptive control、solver lr、computer 配置名。
- checkpoint: raw $\mu,s,q,\alpha$ 与 optimizer state。
- renderer logs: active Gaussians、culled Gaussians、tile count、render time、GPU memory。
- validation renders: 输入 slice 重渲染、未见 slice 重渲染、absolute difference map。
- optional voxel volume: world Gaussian voxelization result。
- optional compressed package: quantized streams + metadata + LZMA output。
