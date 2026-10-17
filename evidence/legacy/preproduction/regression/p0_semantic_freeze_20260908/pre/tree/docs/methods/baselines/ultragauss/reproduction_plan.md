# Method

## 0) 复现目标与实现边界

这份说明把 UltraGauss 视为一个可训练的显式 3D 表示：

- 输入：若干 2D 超声切片/视频帧 $I_k\in[0,1]^{H\times W}$ 及其对应 6-DoF probe pose。
- 输出：由 $N$ 个 3D anisotropic Gaussians 表示的连续体数据，可在任意 probe plane 上重渲染 2D 超声图像。
- 核心差异：UltraGauss 不是 camera 3DGS 的“投影 + 深度排序 + alpha compositing”，而是“把每个 Gaussian 变到 probe 坐标系，在 probe plane 上取 3D Gaussian 截面响应，并在平面内做加权平均”。

论文没有公开写出完整训练 loss、densification/sparsification 的所有阈值、$\beta/\epsilon/\alpha_{BG}/c_{BG}$ 的具体数值。你的代码仓库已有 loss/metric 的 `computer` 基础设施，因此 UltraGauss 复现只需要把 renderer 输出接入现有 computer，不需要在方法模块里重新实现 loss 或 metric。下面凡是论文未明示但复现必须决定的地方，均标为“实现假设/待查代码”，不要把这些当作论文原文。

## 1) 坐标、输入与参数约定

### 1.1 坐标系

建议实现时显式区分三个坐标系：

- **World/acquisition volume 坐标**：Gaussian 参数 $\mu_i,\Sigma_i,\Lambda_i$ 所在坐标系。
- **Probe 坐标**：当前训练/渲染切片的局部坐标系，probe plane 为 $z=0$。
- **Pixel/grid 坐标**：2D 图像像素 $(u,v)$，需要映射到 probe plane 上的连续坐标 $x=[x_1,x_2,0]^T$。

论文采用的计算顺序是 **probe-space rendering**，不是直接在 world 坐标中把“世界像素点”和 Gaussian 做距离计算。更具体地说：Gaussian 参数先由 world/acquisition volume 坐标变换到当前 probe 坐标；像素也先由 2D grid 映射为 probe plane 上的 3D 点 $x^p=[x_1,x_2,0]^T$；随后在 probe 坐标中计算 3D Mahalanobis distance 作为 Gaussian-pixel contribution。

每张训练图像 $I_k$ 需要一个 world-to-probe 刚体变换 $W_k$。若 $W_k$ 是 $4\times4$ 齐次矩阵，记其旋转/线性部分为 $R_k\in\mathbb{R}^{3\times3}$、平移为 $t_k$：

$$
\mu_{i,k}^{p} = R_k\mu_i+t_k,\qquad
\Sigma_{i,k}^{p}=R_k\Sigma_iR_k^T.
$$

论文用齐次坐标写作：

$$
\mu_i^{3D}=\hbar^{-1}(W\hbar(\mu_i)),\qquad
\Sigma_i^{3D}=W\Sigma_iW^T.
$$

实现中不要把 $4\times4$ 齐次矩阵直接乘到 $3\times3$ covariance 上；covariance 只接受线性部分 $R_k$。若内部主要保存 precision $\Lambda_i=\Sigma_i^{-1}$，则 probe 坐标下：

$$
\Lambda_{i,k}^{p}=(\Sigma_{i,k}^{p})^{-1}
=R_k^{-T}\Lambda_iR_k^{-1}.
$$

对刚体旋转 $R_k^{-1}=R_k^T$，所以上式可写为 $\Lambda_{i,k}^{p}=R_k\Lambda_iR_k^T$。

因此当前帧中第 $i$ 个 Gaussian 对像素 $x^p$ 的距离应写为：

$$
d_i^2(x^p)=
(x^p-\mu_{i,k}^{p})^T
\Lambda_{i,k}^{p}
(x^p-\mu_{i,k}^{p}).
$$

数学上也可以把像素点反变换到 world 坐标 $x^w=W_k^{-1}x^p$，再用 world-space Gaussian 计算：

$$
d_i^2(x^w)=
(x^w-\mu_i)^T\Lambda_i(x^w-\mu_i).
$$

只要刚体变换和 covariance/precision 变换完全一致，这两种写法等价。但论文的 Eq.8-9、bbox culling、probe-plane intersection 和 CUDA load balancing 都围绕 probe 坐标展开：probe plane 固定为 $z=0$，是否相交只需检查 bbox 的 $z$ 范围，2D rasterization window 直接来自 probe-space 的 $x,y$ 边界。因此服务器端复现应优先采用 probe-space 实现。

### 1.2 Gaussian 参数

每个 Gaussian $G_i$ 至少包含：

$$
\theta_i=\{\mu_i,\ L_i,\ c_i,\ \alpha_i\}.
$$

- $\mu_i\in\mathbb{R}^3$：world 坐标中的中心。
- $L_i$：下三角 precision factor 的 6 个可学习自由度。
- $c_i\in[0,1]$：单通道灰度/散射强度。
- $\alpha_i\in[0,1)$：该 Gaussian 的最大响应/重要性权重。

实现建议把 $c_i,\alpha_i$ 存成 unconstrained logits，再用 sigmoid 映射到 $[0,1]$。论文明确说 $\alpha_i=0.731$ 并由 sigmoid 约束，因此若 raw 参数是 logit，初始化 raw $\alpha_i\approx 1.0$。

## 2) 前向成像模型

### 2.1 Probe-plane intersection rendering

给定 probe plane 上的像素位置 $x=[x_1,x_2,0]^T$，第 $i$ 个 Gaussian 在该像素的响应为：

$$
\hat{\alpha}_i(x)
= \alpha_i\exp\left(
-\frac{1}{2}
(x-\mu_i^p)^T\Lambda_i^p(x-\mu_i^p)
\right).
$$

这里的 Mahalanobis distance 是 **3D distance**，不是把 Gaussian 投影成 2D Gaussian 后再算 2D distance。这一点是 UltraGauss 与 camera 3DGS 的核心差别。

像素处总响应与灰度输出：

$$
\hat{\alpha}(x)=\sum_{i=1}^{N}\hat{\alpha}_i(x)+\alpha_{BG},
$$

$$
\hat{c}(x)
=
\frac{
\sum_{i=1}^{N}\hat{\alpha}_i(x)c_i+\alpha_{BG}c_{BG}
}{
\sum_{i=1}^{N}\hat{\alpha}_i(x)+\alpha_{BG}
}.
$$

实现时维护两个 pixel accumulators：

$$
A(x)=\sum_i \hat{\alpha}_i(x),\qquad
C(x)=\sum_i \hat{\alpha}_i(x)c_i,
$$

最后：

$$
\hat{I}(x)=\frac{C(x)+\alpha_{BG}c_{BG}}{A(x)+\alpha_{BG}}.
$$

$\alpha_{BG}$ 用来避免除零，并提供弱背景。论文说明了它的作用，但没有给数值；复现时应把它做成 config，并记录实验值。

### 2.2 为什么保留 $c_i$ 与 $\alpha_i$ 两个参数

附录 G.1 对比了两种灰度建模：

单参数方案：

$$
\hat{\rho}_i(x)=\rho_i
\exp\left(-\frac12(x-\mu_i^p)^T\Lambda_i^p(x-\mu_i^p)\right),
\qquad
\hat{c}(x)=\frac{1}{N}\sum_i\hat{\rho}_i(x).
$$

UltraGauss 方案：

$$
\hat{c}(x)=
\frac{\sum_i\hat{\alpha}_i(x)c_i}{\sum_i\hat{\alpha}_i(x)}.
$$

后者是 $\hat{\alpha}_i$ 加权平均，$\alpha_i$ 控制 Gaussian 对该像素的贡献强度，$c_i$ 控制贡献的灰度值。论文附录 G.1 的消融显示，分开学习 $c_i,\alpha_i$ 明显优于单参数方案。因此复现不要把二者合并。

## 3) 三角 precision 参数化

UltraGauss 不直接学习 covariance $\Sigma_i$，而学习 precision $\Lambda_i=\Sigma_i^{-1}$。论文先讨论过对称矩阵 $M$ 的参数化，但最终采用下三角形式：

$$
\Lambda_i = L_iL_i^T,
$$

$$
L_i=
\begin{bmatrix}
\ell_{11}^2+\beta & 0 & 0\\
\ell_{12} & \ell_{22}^2+\beta & 0\\
\ell_{13} & \ell_{23} & \ell_{33}^2+\beta
\end{bmatrix}.
$$

正文 4.6 又写成：

$$
\Lambda_i=L_iL_i^T+\epsilon I.
$$

复现时可以同时保留 $\beta$ 与 $\epsilon$：

- $\beta>0$ 保证 $L_i$ 对角严格正。
- $\epsilon I$ 作为额外 jitter，避免极端数值问题。

若 $L_i$ 对角为正，则 $L_i$ 可逆，$\Lambda_i=L_iL_i^T$ 正定，$\Sigma_i=\Lambda_i^{-1}$ 存在。需要 covariance 时不要做通用矩阵逆，优先使用 triangular solve：

$$
\Sigma_i=(L_i^{-1})^TL_i^{-1}.
$$

Gaussian 重采样时可直接采样：

$$
y=\mu_i+L_i^{-T}z,\qquad z\sim\mathcal{N}(0,I_3).
$$

实现要点：

- raw 参数为 $(\ell_{11},\ell_{12},\ell_{13},\ell_{22},\ell_{23},\ell_{33})$。
- 构造 $L$ 时只对三个 diagonal raw 参数平方并加 $\beta$。
- off-diagonal 不做约束。
- 若只需要 $\operatorname{diag}(\Sigma_i)$，可计算 $L_i^{-T}$ 每列的平方范数，无需完整 $\Sigma_i$。

论文附录 A 给出的理论 FLOPs：

- 3DGS 式 quaternion+scale：每迭代计算 $\Sigma^{-1}$ 约 $108N$ FLOPs。
- UltraGauss triangular precision：每迭代计算 $\Sigma^{-1}$ 约 $20N$ FLOPs。
- 需要 densification 方差时，实践中只需额外约 $20N$ FLOPs；若必须完整 $\Sigma$，约 $25N$ FLOPs。

实测加速：

- $\Sigma^{-1}$ 计算 forward 约 $1.4\times$，backward 约 $1.2\times$。
- densification 相关 $\Sigma$ 计算 forward 约 $1.25\times$，backward 约 $6.65\times$。

## 4) Rasterization bounds 与 CUDA 渲染

直接对所有 pixel-Gaussian pair 计算响应复杂度太高。UltraGauss 只在每个 Gaussian 的主要概率质量范围内做累加。

### 4.1 椭球截断

设 $q_p=\chi^2_{3,p}$。论文采用 $p=95\%$，对应：

$$
q_p=7.815.
$$

Gaussian 的有效区域定义为：

$$
(x-\mu_i^p)^T\Lambda_i^p(x-\mu_i^p)\le q_p.
$$

论文 Eq.14 给出 cofactor 形式的 3D axis-aligned bounding box。为了降低实现出错概率，服务器端实现建议使用等价且更直接的 bound：

$$
r_j=\sqrt{q_p\cdot(\Sigma_i^p)_{jj}},\qquad j\in\{1,2,3\},
$$

$$
b_{\min}=\mu_i^p-r,\qquad b_{\max}=\mu_i^p+r.
$$

这是椭球 $\delta^T\Lambda\delta\le q_p$ 沿坐标轴的精确 axis-aligned half-width，因为：

$$
\max_{\delta^T\Lambda\delta\le q_p}\delta_j^2
=q_p(\Lambda^{-1})_{jj}
=q_p\Sigma_{jj}.
$$

若为了完全贴近论文 Eq.14 使用 cofactor 公式，务必单元测试它与上式输出一致；否则很容易把 covariance/precision 的 cofactor 和 determinant 写反。

### 4.2 两阶段 GPU 流程

每个训练/渲染 frame 的 CUDA forward 可拆成：

**Phase 1: Gaussian-level culling**

对所有 Gaussians 并行：

1. 将 $\mu_i,\Lambda_i$ 或 $\Sigma_i$ 变到当前 probe 坐标。
2. 计算 3D bbox $(b_{\min},b_{\max})$。
3. 若 bbox 在 probe 法向上不穿过 $z=0$，拒绝：

$$
b_{\min,3}>0\quad\text{or}\quad b_{\max,3}<0.
$$

4. 对 accepted Gaussians 记录 2D bbox：

$$
b_{\min,1:2}\le (x_1,x_2)\le b_{\max,1:2}.
$$

5. 将 accepted mask 做 buffer compaction，得到紧凑 Gaussian 列表。论文引用 NVIDIA Thrust。

**Phase 2: Pixel-level rasterization**

对 compacted Gaussians 并行：

1. 将 2D bbox clip 到图像边界。
2. 遍历 bbox 内像素，把像素坐标映射到 probe plane 连续坐标 $x=[x_1,x_2,0]^T$。
3. 计算 $d^2=(x-\mu_i^p)^T\Lambda_i^p(x-\mu_i^p)$。
4. 可选但建议：若 $d^2>q_p$，跳过该像素，避免 bbox 角落的多余贡献。
5. 计算 $\hat{\alpha}_i(x)=\alpha_i\exp(-0.5d^2)$。
6. 原子累加：

$$
\operatorname{atomicAdd}(A[x],\hat{\alpha}_i(x)),
\qquad
\operatorname{atomicAdd}(C[x],\hat{\alpha}_i(x)c_i).
$$

7. 后处理 kernel 计算 $\hat{I}(x)$。

反向传播需要对 $\mu_i,L_i,c_i,\alpha_i$ 回传梯度。若先用 PyTorch 原型验证数学，可用稠密 pairwise 版本；正式复现应实现 custom CUDA forward/backward，否则 $N=300K/2M$ 不现实。

### 4.3 $p$ 的消融结论

附录 G.3 扫描了 $p$。趋势符合预期：

- $p$ 越大，椭球越大，每轮 atomic add 越多，速度越慢。
- $p$ 越大，每个 pixel 聚合的 Gaussians 越多，质量可能更高。
- 论文最终选 $p=95\%$ 作为速度与质量折中。

复现默认值使用 $p=0.95$，并把每轮 accumulated Gaussian-pixel pairs 作为日志指标。

## 5) Acoustic shadowing

UltraGauss 的主渲染不做 depth compositing。阴影作为轻量 Beer-Lambert attenuation 后处理加入：

$$
T(x)=\exp\left(-\int_0^x\alpha(s)\,ds\right)
\approx
\exp\left(-\sum_{j=0}^{N_s}\hat{\alpha}_j\delta_j\right).
$$

其中 $\hat{\alpha}_j$ 是 Eq.6 的 opacity/absorption map，$\delta_j$ 是相邻采样间距；论文设为 pixel spacing，即：

$$
\delta_j=\frac{1}{\mathrm{ImageHeight}}.
$$

实现流程：

1. 先渲染 $A(x)$ 与 $\hat{I}(x)$。
2. 若为 curvilinear/cone 图像，将 $A(x)$ 双线性 warp 到 beam-aligned square grid。
3. 沿 beam/depth 方向做 cumulative sum：

$$
S(u,v)=\sum_{j\le v}A(u,j)\delta.
$$

4. 得到 attenuation：

$$
T(u,v)=\exp(-S(u,v)).
$$

5. unwarp 回原图坐标。
6. 最终输出：

$$
I_{\text{shadow}}(x)=\hat{I}(x)\cdot T(x).
$$

阴影模块要求知道 probe geometry/beam direction。若先复现正交 slice 重建或你的数据暂时没有明确 beam geometry，可先关闭 shadow；若复现 curvilinear/linear probe 的真实阴影效果，应实现 warp-integrate-unwarp。

## 6) 优化与训练配置

### 6.1 初始化

论文给出的初始化：

- $\mu_i$：在 acquisition volume bounds 内均匀随机采样；不使用 COLMAP，因为 ultrasound volume 不是 surface-only reconstruction。
- $c_i=0.5$。
- $\alpha_i=0.731$，并用 sigmoid 约束到 $[0,1)$。
- $L_{ij}\sim U[4,5)$；在论文的归一化坐标下，对应较小初始 marginal variance，约 $(1.7-4.3)\times10^{-3}$。
- 加 small jitter $\epsilon$ 保证 $\Lambda_i\succ0$。

### 6.2 学习率

论文给出的 optimizer 与学习率：

- Optimizer：Adam。
- 除 $\mu_i$ 之外，所有参数学习率为 $0.05$。
- $\mu_i$ 使用指数衰减 schedule，初始学习率 $1.6\times10^{-4}$。
- 单卡 NVIDIA RTX-A4000。

论文正文 4.6 写 $N\in\{100K,200K,2M\}$，但实验图表和附录反复使用 $N\in\{100K,300K,2M\}$。复现实验应以图表中的 $300K$ 为准，并在报告里注明正文疑似笔误。

### 6.3 Loss 与 metric 接入

论文未明确写出训练 loss。由于当前代码仓库已经有完整的 loss/metric `computer` 基础设施，UltraGauss 方法实现不应新写 loss 或 metric，只需要遵守现有 computer 的输入/输出约定。

renderer/training step 应向 computer 提供：

- `pred`: UltraGauss 渲染出的单通道图像 $\hat{I}$，形状、dtype、range 与仓库现有 renderer 输出保持一致。
- `target`: 对应输入/监督图像 $I$。
- 可选 `mask/valid_region`: 若你的数据有扇形有效区域、padding 或 crop 区域，由数据管线或 computer 处理。
- 可选 renderer diagnostics：`opacity_accumulation`、`accepted_gaussians`、`atomic_pairs` 等只作为日志，不参与 loss，除非现有 computer 已支持。

注意边界：

- loss/metric 的选择、组合、mask 规则、日志格式由现有 computer 决定。
- `ops/ultragauss_renderer` 不应依赖 SSIM/PSNR/LPIPS，也不应内置 loss。
- `gaussian solver` 只接收 computer 产生的 scalar loss 做 `backward()` 和 optimizer step。
- 如果将来需要对齐官方代码的 loss，只在 computer 配置层切换，不修改 renderer/container 的数学实现。

### 6.4 Densification 与 sparsification

论文说 heuristics follow Kerbl et al. 2023 and Ye et al. 2024，并为 plane-intersection rasterization 做小改动，但未给完整阈值。可复现实现的工程约定：

- 定期检查 Gaussian 对训练像素的贡献和梯度。
- 对 underfit/高梯度区域 split/clone Gaussians。
- 对低贡献或过小 $\alpha_i$ 的 Gaussians prune。
- 重采样使用：

$$
y=\mu_i+L_i^{-T}z.
$$

- 对 split 后的 Gaussian 重新分配 $c_i,\alpha_i,L_i$，并确保 $\Lambda_i$ 仍正定。

这部分是复现风险最高的未明示细节。建议先实现固定 $N$ 版本，确认 renderer 和优化能工作；再实现 adaptive densification/sparsification，并用附录 G.2 的定性现象验证 speckle 与清晰度变化。

## 7) 代码模块落盘建议

若服务器代码已经按 `gaussian container / gaussian solver / density control / ops/{method}_renderer` 拆分，UltraGauss 相关逻辑建议按下面边界落盘。核心原则：**Gaussian 椭球状态归 container；优化器与学习率归 solver；数量变化归 density control；所有 per-frame probe-space 渲染和 CUDA kernel 归 renderer/op。**

### 7.1 Gaussian container

`gaussian container` 应保存 UltraGauss 所需的全部 per-Gaussian 可学习属性，并提供 raw-to-physical activation 接口。

建议字段：

- `xyz_raw` 或 `mu`：Gaussian center $\mu_i\in\mathbb{R}^3$，通常物理形态和 raw 形态一致。
- `l_raw`：6 维 lower-triangular precision raw 参数 $(\ell_{11},\ell_{12},\ell_{13},\ell_{22},\ell_{23},\ell_{33})$。
- `color_raw`：灰度 $c_i$ 的 logit/raw 参数。
- `opacity_raw`：$\alpha_i$ 的 logit/raw 参数。
- 可选 `background_alpha_raw/background_color_raw`：若背景作为全局可学习参数实现，也可放在 container；若作为 config 常量，则不用放入 per-Gaussian container。

建议 activation：

$$
c_i=\sigma(\text{color\_raw}_i),
\qquad
\alpha_i=\sigma(\text{opacity\_raw}_i).
$$

$$
L_i=
\begin{bmatrix}
\ell_{11}^2+\beta & 0 & 0\\
\ell_{12} & \ell_{22}^2+\beta & 0\\
\ell_{13} & \ell_{23} & \ell_{33}^2+\beta
\end{bmatrix},
\qquad
\Lambda_i=L_iL_i^T+\epsilon I.
$$

container 应提供至少这些方法/属性：

- `get_xyz()`：返回 $\mu_i$。
- `get_l_matrix()`：由 `l_raw` 构造 $L_i$。
- `get_precision()`：返回 $\Lambda_i$。
- `get_covariance()`：仅在 bbox/densification 需要时通过 triangular solve 得到 $\Sigma_i=\Lambda_i^{-1}$。
- `get_color()`、`get_opacity()`：返回 sigmoid 后的物理值。
- `append_gaussians(...)` / `remove_gaussians(mask)`：供 density control resize 参数。

注意：probe pose 不应写入 container。container 存的是 scene/world 表示；每帧 world-to-probe 变换属于 renderer forward 的输入。

### 7.2 Gaussian solver

`gaussian solver` 负责把 container 中的 raw tensors 注册进 optimizer 和 learning-rate scheduler，并在 density control 改变 Gaussian 数量后维护 optimizer state。

UltraGauss 至少需要参数组：

- `xyz_raw/mu`：学习率初始 $1.6\times10^{-4}$，指数衰减。
- `l_raw`：学习率 $0.05$。
- `color_raw`：学习率 $0.05$。
- `opacity_raw`：学习率 $0.05$。
- 若背景可学习：背景参数使用单独 parameter group，学习率可先设 $0.05$ 或更小，并标为实现假设。

solver 应负责：

- 初始化 Adam。
- 为 $\mu_i$ 注册 exponential decay scheduler。
- 其他参数维持常数学习率 $0.05$，除非官方代码另有设置。
- 在 `append/prune/split` 后同步扩展或裁剪 optimizer state，例如 Adam 的 `exp_avg`、`exp_avg_sq`。
- 暴露 `resize_optimizer_state(keep_mask/new_tensors)` 一类工具给 density control 调用。

不要把 probe-space covariance、bbox 或 pixel accumulation 放入 solver；这些是 renderer 的 forward/backward 逻辑。

### 7.3 Density control

`density control` 应只负责 Gaussian 数量变化策略，并持有 container 与 solver 来完成实际增删和 optimizer state resize。

UltraGauss 对应逻辑：

- 定期统计每个 Gaussian 的贡献、opacity、梯度或 screen/probe-plane footprint。
- prune：删除低贡献、低 $\alpha_i$、长期不被任何 probe plane bbox 接收的 Gaussians。
- clone/split：对高梯度或 underfit 区域的 Gaussians 增殖。
- resample：使用论文 Eq.12：

$$
y=\mu_i+L_i^{-T}z,\qquad z\sim\mathcal{N}(0,I_3).
$$

- split/clone 后需要为新 Gaussian 初始化 `l_raw/color_raw/opacity_raw`，并保持 $\Lambda_i\succ0$。

论文没有给完整阈值，因此 density control 的阈值应全部 config 化，并在实验日志中记录。建议第一阶段先关闭 density control，固定 $N$ 验证 renderer；第二阶段再打开 prune/split。

### 7.4 `ops/ultragauss_renderer`

`ops/{method_name}_renderer` 是 UltraGauss 与常规 3DGS 差异最大的地方，应实现 probe-plane intersection renderer，而不是复用 camera projection renderer。

输入建议：

- container activation 后的 `mu`、`L` 或 `precision`、`color`、`opacity`。
- 当前 batch/frame 的 world-to-probe pose `W` 或 `(R,t)`。
- pixel-to-probe-plane grid，或图像尺寸、spacing、origin，由 op 内部生成 $x^p=[x_1,x_2,0]^T$。
- renderer config：`p=0.95`、`alpha_bg`、`color_bg`、shadow 开关、probe geometry。

CUDA forward 建议拆 kernel：

1. `transform_and_bound_kernel`：把 Gaussian 变到 probe space，计算 probe-space $\mu_i^p,\Lambda_i^p/\Sigma_i^p$ 和 bbox。
2. `cull_kernel`：根据 $b_{\min,3},b_{\max,3}$ 判断是否与 probe plane 相交。
3. `compact`：压缩 accepted Gaussian 列表，可用 Thrust/CUB。
4. `rasterize_kernel`：遍历 accepted Gaussians 的 probe-space 2D bbox，计算 3D Mahalanobis distance，atomic add 到 $A(x)$ 和 $C(x)$。
5. `normalize_kernel`：计算 $\hat{I}(x)=(C+\alpha_{BG}c_{BG})/(A+\alpha_{BG})$。
6. 可选 `shadow_kernel`：warp-integrate-unwarp，得到 $I_{\text{shadow}}=\hat{I}\cdot T$。

CUDA backward 需要回传到：

- $\mu_i$。
- `l_raw`，经由 $L\rightarrow\Lambda\rightarrow d^2\rightarrow\hat{\alpha}$ 链式求导。
- `color_raw`。
- `opacity_raw`。
- 若可学习，背景参数。

关键实现约束：

- renderer 内部的主坐标应是 probe 坐标。
- 像素不是 camera ray，也不做深度排序。
- 不使用 alpha compositing transmittance 作为主成像模型。
- bbox/culling 使用 probe-space $z=0$ 平面。
- 若用 world-space 等价形式实现，仍要保证输出与 probe-space dense reference 在小 $N$ 测试中一致；但为了贴合论文和 CUDA 负载均衡，推荐 probe-space。

### 7.5 模块责任边界速查

| 内容 | 建议落盘位置 |
|---|---|
| $\mu_i,c_i,\alpha_i,L_i$ raw 参数 | `gaussian container` |
| sigmoid、triangular precision activation | `gaussian container` |
| $\Sigma/\Lambda$ 的 world-space 物理属性接口 | `gaussian container` |
| Adam parameter groups | `gaussian solver` |
| $\mu_i$ 指数衰减学习率 | `gaussian solver` |
| optimizer state resize | `gaussian solver` |
| prune/split/clone/resample 策略 | `density control` |
| Eq.12 采样新 Gaussian 位置 | `density control` 调 container 的 $L^{-T}$ |
| world-to-probe transform | `ops/ultragauss_renderer` |
| probe-space Mahalanobis distance | `ops/ultragauss_renderer` |
| bbox、culling、compaction、atomic accumulation | `ops/ultragauss_renderer` |
| shadow warp-integrate-unwarp | `ops/ultragauss_renderer` 或其子 op |
| loss/metric 计算 | 现有 `computer` 基础设施 |

## 8) 服务器端实现 checklist

最小可行复现顺序：

1. 实现任意 2D ultrasound slice/frame + world-to-probe pose 的数据接口，统一 intensity 到 $[0,1]$。
2. 用 PyTorch 写一个小 $N$ 的稠密 renderer，验证 Eq.6-8、坐标变换和梯度。
3. 实现 triangular precision 参数化，单元测试 $\Lambda\succ0$、$\Sigma=\Lambda^{-1}$、bbox half-width。
4. 实现 CUDA forward：culling、compaction、bbox rasterization、atomic accumulators。
5. 实现 CUDA backward 或用 custom autograd 接入。
6. 先固定 $N=100K$，关闭 shadow/densification，在你自己的一个小体数据或少量带 pose 的 2D slices 上跑通重建。
7. 加入 $p=95\%$ bbox、background，并把 renderer 输出接入现有 computer/logging。
8. 加入 densification/sparsification。
9. 加入 shadow 模块。
10. 扩展到 $N=300K/2M$，再根据你的数据规模决定是否需要完整 ablation。

必要单元测试：

- $L$ 构造后所有 batch 的 Cholesky/特征值检查通过。
- $\max_{\delta^T\Lambda\delta\le q}\delta_j=\sqrt{q\Sigma_{jj}}$ 的 bbox 数值测试通过。
- 同一 Gaussian 在 identity pose 下，probe plane $z=0$ 的响应与手写 Mahalanobis 公式一致。
- bbox culling 不应漏掉 $d^2\le q_p$ 的 pixel。
- background 分母永不为 0。
- shadow 开关关闭时输出完全等于主 renderer。

# Method-related Settings

这一节只保留和方法实现强相关的配置，不再复述论文里与当前复现无关的实验协议。服务器端复现时优先保证下面这些 method-level 设定正确。

## 1) 必需输入

每个训练样本至少包含：

- `image`: 单通道 2D ultrasound frame/slice，建议归一化到 $[0,1]$。
- `pose`: 当前帧的 world-to-probe 变换 $W_k$ 或 `(R_k,t_k)`。
- `pixel_grid` 或足够生成 grid 的元数据：`H, W, spacing, origin`，用于把像素 $(u,v)$ 映射到 probe plane 上的 $x^p=[x_1,x_2,0]^T$。

如果你的数据来自已有 3D volume，可以自己采样任意方向 slice 并生成 ground-truth pose；如果来自 freehand video，则 pose 可以来自外部 pose estimator。UltraGauss 方法本身假设 pose 已知或已估计，不在本文方法中联合优化 pose。

## 2) 默认方法参数

| 参数 | 论文/建议值 | 落盘位置 | 备注 |
|---|---:|---|---|
| Gaussian 数量 $N$ | `100K`, `300K`, `2M` | config + container init | 正文 4.6 写过 `200K`，但图表/附录使用 `300K`；复现优先按 `300K` |
| probability threshold $p$ | `0.95` | renderer config | 对应 $\chi^2_{3,0.95}=7.815$ |
| $c_i$ 初始化 | `0.5` | container init | 若 raw 为 logit，则 raw 初始化为 `logit(0.5)=0` |
| $\alpha_i$ 初始化 | `0.731` | container init | 若 raw 为 logit，则 raw 初始化约 `1.0` |
| $L_{ij}$ 初始化 | $U[4,5)$ | container init | 三个 diagonal raw 之后会平方并加 $\beta$ |
| $\mu_i$ 初始化 | acquisition bounds 内均匀采样 | container init | 不使用 COLMAP surface points |
| optimizer | Adam | solver | 论文明确 |
| $\mu_i$ 初始 lr | $1.6\times10^{-4}$ | solver param group | 指数衰减 |
| 其他参数 lr | `0.05` | solver param group | 包括 $L_i,c_i,\alpha_i$ |
| shadow | 可选 | renderer config | 有 probe geometry/beam direction 时再打开 |

## 3) 论文未公开但代码必须配置的参数

这些参数不要写成“论文设定”，只能写成 implementation config：

- `beta`: $L$ 对角线正值偏置，保证 diagonal $>0$。
- `epsilon`: $\Lambda=L L^T+\epsilon I$ 的 jitter。
- `alpha_bg`: background opacity，避免分母为 0。
- `color_bg`: background grayscale。
- `densify_interval/prune_interval`: 论文未给完整 densification/sparsification 阈值。
- `opacity_prune_threshold`: 低贡献 Gaussian 删除阈值。
- `grad_split_threshold`: 高梯度 split/clone 阈值。
- `max_gaussians/min_gaussians`: 防止 density control 失控。

建议所有未公开参数都进入 `config.yaml`，并在实验日志中保存。

## 4) 最小验证协议

不需要复现论文数据集，也可以验证方法是否正确：

1. **Dense reference renderer**：小 $N$ 下用纯 PyTorch 全量 pixel-Gaussian pair 渲染一帧。
2. **CUDA renderer 对齐**：同样输入下，CUDA bbox renderer 输出应与 dense reference 接近。
3. **Probe-space 变换测试**：identity pose 下，$z=0$ 平面响应与手写 Mahalanobis 公式一致；随机刚体 pose 下，probe-space 和 world-space 等价公式输出一致。
4. **bbox 测试**：所有满足 $d^2\le q_p$ 的点都应落在 bbox 内。
5. **优化 smoke test**：固定 $N=100K$，关闭 shadow 和 density control，在少量带 pose 的 slices 上通过现有 computer 得到可反传的 scalar loss，训练后 loss 下降，并能在未训练方向上渲染出结构。
6. **density control 测试**：打开 prune/split 后，container tensor 和 Adam state 长度始终一致。
7. **shadow 测试**：关闭 shadow 时输出等于主 renderer；开启时 attenuation map 沿 beam 方向累计衰减。

## 5) 推荐保存的复现产物

- `config.yaml`: $N,p,\beta,\epsilon,\alpha_{BG},c_{BG}$、learning rates、shadow/density control 开关，以及所接入的 computer 配置名。
- `poses.*`: 每张输入帧的 world-to-probe pose。
- checkpoint: container raw 参数 $\mu,L,c,\alpha$ 与 solver optimizer state。
- renderer logs: accepted Gaussians/frame、Gaussian-pixel atomic adds/frame、render time、GPU memory。
- validation renders: 输入视角重渲染、未见视角重渲染、absolute difference map。
