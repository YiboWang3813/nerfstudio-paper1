# Method

## 0) 复现目标与实现边界

UltraG-Ray 是一个 **Gaussian field + ultrasound ray casting forward model**。它和 UltraGauss 的关键区别是：

- UltraGauss：probe-plane intersection，在当前 probe plane 上计算 Gaussian-pixel 3D Mahalanobis contribution。
- UltraG-Ray：ray casting，沿超声 scan line/ray 建模 acoustic attenuation，再在 ray 上的 3D pixel location 计算 echo intensity。

因此 UltraG-Ray 的 renderer 不能复用 camera 3DGS projection renderer，也不是 UltraGauss 的纯 plane-intersection renderer。它需要对每个像素构造一条从探头表面出发的 acoustic ray：

$$
r(z)=o+zd,
$$

其中 $o$ 是该 scan line 在 transducer surface 上的起点，$d$ 是局部 surface normal / beam direction，$z$ 是沿 beam 的深度。像素 $p$ 对应 world 坐标中的 3D 点：

$$
p=o+z_p d.
$$

最终 B-mode 像素被分解为：

$$
B(p)=T(p)\cdot E(p),
$$

其中 $T(p)$ 是从探头到 $p$ 的累计传输/衰减项，$E(p)$ 是 $p$ 处由 Gaussian field 产生的回波强度。

你的代码仓库已经有 loss/metric 的 `computer` 基础设施，因此 UltraG-Ray 方法实现只需要把 renderer 输出接入现有 computer，不需要在方法模块里重新实现 loss 或 metric。论文中没有完全公开所有 refinement 阈值和工程细节；凡是论文未明示但代码必须决定的内容，下面均标为 implementation config 或待查官方代码。

## 1) Gaussian 表示

论文中每个 Gaussian $G_i$ 包含：

$$
G_i=\{\mu_i,\Sigma_i,\tau_i,I_i\}.
$$

- $\mu_i\in\mathbb{R}^3$：Gaussian mean，world/local scene 坐标。
- $\Sigma_i\in\mathbb{R}^{3\times3}$：anisotropic covariance。
- $\tau_i$：learnable transmittance value，用于 attenuation。
- $I_i$：learnable echo intensity，使用低阶 spherical harmonics 表达方向相关 backscattering。

协方差不直接优化矩阵本身，而通过 scale vector $s_i$ 与 quaternion $q_i$ 参数化：

$$
\Sigma_i = R(q_i)S(s_i)^2R(q_i)^T,
$$

其中 $S(s_i)=\operatorname{diag}(s_{ix},s_{iy},s_{iz})$，$R(q_i)$ 是四元数归一化后得到的旋转矩阵。该参数化保证 $\Sigma_i$ 正定，并允许 anisotropic Gaussian shape。

与 UltraGauss 的 triangular precision 不同，UltraG-Ray 沿用标准 3DGS 风格的 **scale + quaternion** covariance 参数化。服务器复现时不要把 UltraGauss 的 $L L^T$ precision 参数化直接搬到 UltraG-Ray，除非明确做方法改造。

## 2) Ray 与坐标约定

### 2.1 坐标系

建议显式区分：

- **World/local scene 坐标**：Gaussian $\mu_i,\Sigma_i$ 所在坐标系。
- **Probe/transducer 坐标**：由当前 ultrasound frame pose 定义，负责生成 rays。
- **Pixel/image 坐标**：2D ultrasound image grid，每个像素映射到一条 ray 和深度 $z_p$。
- **Gaussian canonical 坐标**：每个 Gaussian 自己的标准正态坐标，用于 ray-Gaussian integral。

UltraG-Ray 的核心计算不是把 Gaussian 投影到 2D plane 后 splat，而是：

1. 由当前 probe pose 和像素位置生成 world-space ray $(o,d)$。
2. 对每个相关 Gaussian，把 ray 变换到该 Gaussian 的 canonical space。
3. 用 canonical-space line integral 计算该 Gaussian 对 transmission 的影响。
4. 在 world-space 3D pixel location $p=o+z_p d$ 处计算 Gaussian echo weight。

### 2.2 Ray 到 Gaussian canonical space

对第 $i$ 个 Gaussian，令其旋转和尺度为 $R_i,S_i$。论文 Eq.6 将 world-space ray 变换到 Gaussian canonical space：

$$
o_g=S_i^{-1}R_i^T(o-\mu_i),
$$

$$
d_g=S_i^{-1}R_i^Td,
$$

$$
\ell_g=\|d_g\|\ell.
$$

其中 $\ell$ 是 world-space ray segment length。实现时通常对每个像素取 $\ell=z_p$，即从 transducer surface 到当前 pixel depth 的路径长度。若只计算局部 Gaussian-ray overlap，也应将 closest-approach interval 裁剪到 $[0,z_p]$，避免位于当前像素之后的 Gaussian 影响当前像素的 transmission。

## 3) Gaussian culling / relevant set selection

论文在 ray casting 前先筛选不可能贡献当前图像的 Gaussians：

1. 将每个 Gaussian 按当前 transducer pose 投影到 **far plane**，far plane 对应 ultrasound imaging depth。
2. 使用由 transducer pose 和 lateral width 定义的 orthographic camera model。
3. 对每个 potential ray endpoint，在 far plane 上定义一个小矩形区域。
4. 判断 Gaussian projected footprint 是否与该矩形相交。
5. 只有相交的 Gaussians 进入 ray casting module。

注意论文特别说明：矩形的 vertical extent 不设为 transducer elevational height，而是设为小值，用来近似 2D ultrasound imaging plane。

工程含义：

- 这一步不是最终物理模型，只是 acceleration/culling。
- 真正的 echo 和 attenuation 仍然用 3D ray-Gaussian equations。
- `ops/ultrag_ray_renderer` 应提供 per-pixel 或 per-tile relevant Gaussian selection，避免所有 ray 对所有 Gaussian 暴力计算。

## 4) Transmission / attenuation model

### 4.1 Ray-Gaussian overlap integral

在 canonical space 中，第 $i$ 个 Gaussian 对当前 ray segment 的 overlap 定义为：

$$
\psi_i
=
\int_0^\ell
\exp\left(
-\frac12\|o_g+t d_g\|^2
\right)dt.
$$

该积分表示 scan line 与 Gaussian 的重叠程度：ray 越接近 $\mu_i$，$\psi_i$ 越大。

论文使用 **3-point Gauss-Legendre quadrature** 近似该积分。具体说法是：在 ray 与 Gaussian closest approach 附近取一个短区间进行积分，因为 integrand 的质量主要集中在那里。实现时建议：

- 先求 canonical ray 上 closest approach 参数：

$$
t^\star=-\frac{o_g^Td_g}{d_g^Td_g+\epsilon}.
$$

- 将积分窗口中心放在 $t^\star$ 附近。
- 窗口必须裁剪到当前 pixel path 的有效区间。
- 用 3 个 Gauss-Legendre nodes/weights 做固定成本近似。

论文没有公开短区间 half-width 的具体数值；这是 `ops/ultrag_ray_renderer` 的 implementation config，需查官方代码或做消融。

### 4.2 Per-Gaussian transmittance

论文 Eq.8 将 overlap integral 映射为 per-Gaussian transmission factor：

$$
T_i=\tau_i+(1-\tau_i)\exp(-\psi_i).
$$

累计 transmission：

$$
T(z)=\prod_{i=1}^{n}T_i.
$$

其中 $n$ 是当前 ray / pixel 相关的 Gaussian 数量。直觉：

- $\tau_i$ 是 learnable base transmittance。
- $\exp(-\psi_i)$ 是由 ray-Gaussian overlap 引起的 attenuation。
- $\psi_i$ 越大，$\exp(-\psi_i)$ 越小，$T_i$ 越接近 $\tau_i$。
- 多个 Gaussian 的 attenuation 以乘法累积，类似 Beer-Lambert exponential decay 的离散近似。

实现建议在 log-domain 中累积以提高数值稳定性：

$$
\log T(z)=\sum_i\log(T_i+\epsilon),
\qquad
T(z)=\exp(\log T(z)).
$$

如果某个 Gaussian 的 closest approach 在当前像素之后，不应影响当前像素 $p$ 的 forward-path transmission。

## 5) Echo intensity model

### 5.1 3D Mahalanobis echo weight

UltraG-Ray 的 echo 不使用 2D splat footprint，而是在 world-space 3D pixel location $p$ 处直接计算 3D Mahalanobis weight：

$$
w_i(p)=
\exp\left(
-\frac12
(p-\mu_i)^T\Sigma_i^{-1}(p-\mu_i)
\right).
$$

这里的 $p=o+z_p d$ 是 ray 上对应当前像素深度的 3D 点。该式与 UltraGauss 的“3D Gaussian evaluation”相似，但 UltraG-Ray 的 $p$ 来自 acoustic ray，而不是固定 probe plane 上的 $z=0$ 像素点。

### 5.2 View-dependent echo amplitude

每个 Gaussian 的 echo amplitude $I_i(d)$ 用 degree-1 spherical harmonics 表示，方向输入为 ray direction $d$。由于 B-mode 是单通道，SH 输出是 scalar intensity，不是 RGB：

$$
I_i(d)=\sum_{b=1}^{B}\beta_{i,b}Y_b(d).
$$

论文训练从 SH degree 0 开始，1000 iterations 后提升到 degree 1。

### 5.3 Soft coverage

为避免低覆盖区域产生不稳定归一化，论文定义：

$$
S(p)=\sum_{i=1}^{n}w_i(p),
$$

$$
\gamma(p)=1-\exp(-S(p)).
$$

最终 echo：

$$
E(p)=
(1-\gamma(p))I_{bkg}
+
\gamma(p)
\frac{
\sum_{i=1}^{n}I_i(d)w_i(p)
}{
S(p)+\epsilon
}.
$$

论文将 $I_{bkg}$ 设为 black。若强度归一化到 $[0,1]$，可先取 $I_{bkg}=0$。

## 6) Final B-mode synthesis

最终输出：

$$
B(p)=T(p)\cdot E(p).
$$

renderer 最少应返回：

- `pred`: $B(p)$，单通道 B-mode prediction。
- 可选 `echo_map`: $E(p)$。
- 可选 `transmittance_map`: $T(p)$。
- 可选 diagnostics：per-frame relevant Gaussian 数、ray-Gaussian pairs、render time。

这些 diagnostics 用于 debug/logging，不应进入 loss，除非你的现有 `computer` 已经支持对应辅助监督。

## 7) Initialization and optimization strategy

### 7.1 初始化

论文初始化：

- Gaussian means $\mu_i$：在 local scene 内随机初始化，不使用 SfM/COLMAP。
- transmittance：$\tau_i=0.99$。
- scale：isotropic scale $0.5\ \mathrm{mm}$。
- SH：从 degree 0 开始，仅 DC band。

### 7.2 优化器与学习率

论文使用 Adam，参数组学习率：

| 参数组 | 初始学习率 |
|---|---:|
| Gaussian means $\mu_i$ | $1\times10^{-4}$ |
| Gaussian scales $s_i$ | $5\times10^{-3}$ |
| quaternions $q_i$ | $5\times10^{-3}$ |
| transmittance $\tau_i$ | $5\times10^{-4}$ |
| SH degree-0 / DC coefficients | $5\times10^{-3}$ |
| SH degree-1 coefficients | $1\times10^{-5}$ |

Learning-rate scheduler：全部学习率在训练结束前平滑衰减到初值的 10%。

训练长度与 batch：

- 训练总步数：30k iterations/epochs。论文文字称 epochs，但 refinement 按 iteration 间隔触发；实现中按 training steps 处理更清晰。
- Batch size：8。

### 7.3 Loss 接入

论文 Eq.10 写出：

$$
L=(1-\lambda_{ssim})L_1+\lambda_{ssim}L_{SSIM}+\lambda_{scale}\mathbb{E}[\exp(S)].
$$

论文设置：

$$
\lambda_{ssim}=0.5,\qquad \lambda_{scale}=10^{-3}.
$$

但你的代码仓库已有完整 loss/metric `computer` 基础设施，因此 UltraG-Ray 方法模块不要重新实现 loss。正确边界是：

- renderer 输出 `pred`，必要时输出 `echo_map/transmittance_map/diagnostics`。
- training loop 将 `pred/target/mask` 交给现有 computer。
- solver 接收 computer 返回的 scalar loss 做 `backward()` 和 optimizer step。
- 如果想加入论文的 scale regularizer，也应作为 computer 的一个可配置 loss term，而不是写进 renderer/op。

### 7.4 Ultrasound-specific refinement

论文使用 3DGS-inspired 但 ultrasound-adapted 的 refinement：

- refinement interval：$r=2500$ iterations。
- importance scores 在同一个 2500-iteration interval 内累计。
- refinement 从 1k iterations 后开始。
- refinement 到 20k iterations 停止。
- 总训练到 30k iterations。
- scale prune：

$$
s<s_{\min}=5\times10^{-5}
\quad\text{or}\quad
s>s_{\max}=5\ \mathrm{mm}.
$$

- Gaussian 上限：

$$
N_{\max}=500K.
$$

策略：

- 低贡献或尺度异常 Gaussians prune。
- 使用累计梯度/importance score 决定是否修改 Gaussian。
- small-scale high-importance Gaussian：duplicate。
- large-scale high-importance Gaussian：split into two nearby Gaussians。

由于 $s_i$ 是 3D scale vector，论文没有说明 prune 时对三个轴使用 `min/max/mean` 还是 `any/all` 判定。代码中应将 `scale_prune_reduction` 作为 config，例如先用 `max(s_i)>s_max` 删除过大 Gaussian，用 `min(s_i)<s_min` 删除退化 Gaussian，并在日志中记录。

附录 A 的 optimization dynamics 说明：随机初始化后，第一轮 refinement 常出现 Gaussian 数量骤降，因为大量无贡献 Gaussian 被 scale regularization 缩小并 prune；后续 split/duplicate 再增加数量。refinement 最后 10k steps 关闭，避免新加 Gaussian 尚未优化稳定就进入最终模型。

## 8) Out-of-plane sampling (OPS)

真实 ultrasound 不是无限薄 ray，而是具有 elevational beam width 的薄体积。UltraG-Ray 用 OPS 近似这一点：

- 对所有 ray origins 加 out-of-plane offset。
- offset 从 cosine-weighted distribution 采样。
- 最大幅度：

$$
\delta=2\ \mathrm{mm}.
$$

作用：

- 近似 elevational slice thickness。
- 给 observed plane 外提供额外约束。
- 防止 Gaussians 塌缩到单个 2D frame/slice 上。

实现上，OPS 应在 `ops/ultrag_ray_renderer` 生成 ray origins 时完成，或者在 dataloader/render call 前生成 perturbed origins。它不应改变 Gaussian container 中的 scene coordinates。

## 9) 代码模块落盘建议

若服务器代码沿用 `gaussian container / gaussian solver / density control / ops/{method}_renderer / computer` 架构，UltraG-Ray 建议按下面边界落盘。

### 9.1 Gaussian container

container 保存 per-Gaussian raw 参数和 activation：

- `xyz_raw` 或 `mu`：$\mu_i$。
- `scale_raw`：scale raw 参数；activation 后得到正尺度 $s_i$。若仓库已有 scale activation，沿用现有实现。
- `quat_raw`：quaternion raw 参数；activation 时 normalize 得到 $q_i$ 和 $R_i$。
- `transmittance_raw`：$\tau_i$ raw 参数；activation 后约束到合理范围，通常可用 sigmoid 到 $[0,1]$。
- `sh_dc`：degree-0 SH coefficient。
- `sh_rest`：degree-1 SH coefficients，1000 iterations 后启用或解冻。

container 应提供：

- `get_xyz()`。
- `get_scaling()`。
- `get_rotation()` / `get_quaternion()`。
- `get_covariance()` 或 `get_covariance_inv()`。
- `get_transmittance()`。
- `get_sh_coeffs(active_degree)`。
- `append_gaussians(...)` / `remove_gaussians(mask)`，供 density control 修改数量。

注意：ray origins、ray directions、OPS offset、probe pose 都不属于 container；它们是 renderer 输入。

### 9.2 Gaussian solver

solver 注册 Adam 参数组和 scheduler：

- means lr：$1\times10^{-4}$。
- scales lr：$5\times10^{-3}$。
- quaternions lr：$5\times10^{-3}$。
- transmittance lr：$5\times10^{-4}$。
- SH DC lr：$5\times10^{-3}$。
- SH degree-1 lr：$1\times10^{-5}$。

solver 负责：

- 所有学习率平滑衰减到初值的 10%。
- 1000 iterations 后启用 SH degree 1。
- density control prune/split/duplicate 后同步 resize Adam state。
- 接收 computer 的 scalar loss 进行 backward/step。

solver 不负责 ray-Gaussian integral、culling、OPS 或 B-mode synthesis。

### 9.3 Density control

density control 持有 container 和 solver，负责 refinement：

- 每 2500 iterations 触发。
- 1k 到 20k iterations 之间启用。
- 维护累计 importance/gradient statistics。
- 根据 scale 和 importance 执行 prune/duplicate/split。
- enforce $N\le500K$。
- prune 条件至少包含 $s<5\times10^{-5}$ 或 $s>5$ mm。

duplicate/split 后必须：

- 更新 container tensors。
- 更新 solver optimizer state。
- 为新 Gaussian 初始化 scale/quaternion/transmittance/SH 参数。
- 清理或重置 importance statistics。

### 9.4 `ops/ultrag_ray_renderer`

这是 UltraG-Ray 最核心的自定义算子位置。建议实现：

1. `ray_generation`: 由 pose、image grid、spacing、origin 生成 world-space ray origins $o$、directions $d$ 和 pixel depths $z_p$。
2. `ops_sampling`: 对 ray origins 加 cosine-weighted out-of-plane perturbation，最大 $\delta=2$ mm。
3. `gaussian_culling`: 将 Gaussians 投影到 far plane，用 rectangular ray endpoint regions 筛选 relevant Gaussians。
4. `ray_gaussian_integral`: 对 relevant ray-Gaussian pairs 计算 Eq.6-7 的 canonical-space overlap $\psi_i$。
5. `transmittance_kernel`: 计算 $T_i=\tau_i+(1-\tau_i)\exp(-\psi_i)$ 并沿 ray 累积 $T(p)$。
6. `echo_kernel`: 在 $p=o+z_p d$ 处计算 3D Mahalanobis weight $w_i(p)$、soft coverage $\gamma(p)$ 和 echo $E(p)$。
7. `compose_kernel`: 输出 $B(p)=T(p)E(p)$，并可选输出 echo/transmittance maps。

实现约束：

- 不做 2D Gaussian splatting。
- 不做 camera alpha compositing。
- 不把 UltraGauss 的 probe-plane $z=0$ intersection 直接复用为主模型。
- transmission 只应累计当前 pixel path 上的相关 Gaussian。
- echo weight 使用 world-space 3D point $p$ 与 world-space Gaussian 的 Mahalanobis distance。
- loss/metric 不在 op 内计算，交给现有 computer。

### 9.5 模块责任边界速查

| 内容 | 建议落盘位置 |
|---|---|
| $\mu,s,q,\tau,SH$ raw 参数 | `gaussian container` |
| scale/quaternion/transmittance/SH activation | `gaussian container` |
| Adam parameter groups | `gaussian solver` |
| lr decay to 10% | `gaussian solver` |
| SH degree schedule | `gaussian solver` 或 training scheduler |
| prune/duplicate/split | `density control` |
| importance accumulation | `density control`，renderer 可返回必要 stats |
| ray generation from probe pose | `ops/ultrag_ray_renderer` |
| OPS ray-origin perturbation | `ops/ultrag_ray_renderer` |
| far-plane culling | `ops/ultrag_ray_renderer` |
| ray-Gaussian integral | `ops/ultrag_ray_renderer` |
| transmittance/echo/B-mode composition | `ops/ultrag_ray_renderer` |
| loss/metric 计算 | 现有 `computer` 基础设施 |

# Method-related Settings

## 1) 必需输入

每个训练样本至少包含：

- `image`: 单通道 B-mode ultrasound image，建议归一化到仓库 computer 期望的 range。
- `pose`: 当前帧 probe/transducer pose，用于生成 world-space rays。
- `probe_geometry`: lateral width、imaging depth、spacing/origin、beam direction convention 等。
- `mask/valid_region`: 可选；若图像存在无效区域，由数据管线或 computer 处理。

UltraG-Ray 假设 pose 已知。若 pose 由外部模块估计，renderer 只消费 pose，不负责 pose learning。

## 2) 默认方法参数

| 参数 | 论文值 | 落盘位置 |
|---|---:|---|
| total training steps | `30k` | training config |
| batch size | `8` | dataloader/training config |
| transmittance init | `0.99` | container init |
| isotropic scale init | `0.5 mm` | container init |
| SH start degree | `0` | container/solver |
| SH degree-1 start | `1000 iterations` | solver/training scheduler |
| means lr | `1e-4` | solver |
| scales lr | `5e-3` | solver |
| quaternions lr | `5e-3` | solver |
| transmittance lr | `5e-4` | solver |
| SH DC lr | `5e-3` | solver |
| SH degree-1 lr | `1e-5` | solver |
| lr final multiplier | `0.1` | solver scheduler |
| refinement interval | `2500` | density control |
| refinement start | `1000` | density control |
| refinement stop | `20000` | density control |
| max Gaussians | `500k` | density control |
| prune min scale | `5e-5` | density control |
| prune max scale | `5 mm` | density control |
| OPS max offset | `2 mm` | renderer config |
| quadrature | 3-point Gauss-Legendre | renderer config |

## 3) 论文未公开但代码必须配置的参数

- `quadrature_window_radius`: closest approach 周围积分短区间大小。
- `transmittance_activation`: $\tau_i$ raw 到 $[0,1]$ 的映射方式。
- `scale_activation`: raw scale 到正尺度的映射方式。
- `importance_threshold`: refinement 触发 duplicate/split 的阈值。
- `split_scale_threshold`: high-importance Gaussian 选择 duplicate 还是 split 的尺度阈值。
- `scale_prune_reduction`: 对 3D scale vector 执行 prune 时使用 `min/max/mean/any/all` 中哪种规则。
- `far_plane_rect_size`: 每个 ray endpoint 在 far plane 上用于 culling 的矩形大小。
- `far_plane_vertical_extent`: 论文说取小值近似 2D imaging plane，但未给具体数值。
- `epsilon`: 所有除法/log/归一化稳定项。

这些都应进入 config，并在日志中保存。

## 4) 最小验证协议

1. **Ray generation test**：identity pose 下，中心列/行 ray direction 与 probe convention 一致。
2. **Canonical transform test**：各向同性 Gaussian、identity rotation 时，Eq.6 与手写 $(o-\mu)/s,d/s$ 一致。
3. **Integral test**：3-point quadrature 输出随 ray-Gaussian 距离增大而单调减小。
4. **Transmission test**：$\psi_i$ 增大时 $T_i$ 下降并下界接近 $\tau_i$。
5. **Echo test**：$p=\mu_i$ 时 3D Mahalanobis weight 最大。
6. **Renderer smoke test**：固定少量 Gaussians，输出 `echo_map/transmittance_map/pred` 形状、范围和梯度正确。
7. **Computer integration test**：renderer 输出能被现有 computer 接收并产生可反传 scalar loss。
8. **Density control test**：prune/split/duplicate 后 container tensor、solver optimizer state、importance buffer 长度一致。

## 5) 推荐保存的复现产物

- `config.yaml`: ray/probe geometry、Gaussian 参数、refinement 参数、OPS、quadrature、solver lr、computer 配置名。
- checkpoint: container raw 参数 $\mu,s,q,\tau,SH$ 与 solver optimizer state。
- renderer logs: relevant Gaussians、ray-Gaussian pairs、render time、GPU memory。
- debug maps: `echo_map`、`transmittance_map`、`pred`。
- validation renders: 输入视角重渲染、未见视角重渲染、absolute difference map。
