# Method

## 0) 复现目标与实现边界

Ultra-NeRF 是一个 **physics-enhanced implicit neural representation (INR)**，用于从 pose-annotated 2D ultrasound sweeps 学习一个连续 3D ultrasound scene，并从 novel probe views 合成 B-mode 图像。

它和前两篇 Gaussian 方法的核心差异：

- UltraGauss / UltraG-Ray：显式 Gaussian field，参数直接附着在 Gaussians 上。
- Ultra-NeRF：隐式 MLP field，输入 3D query point $q=(x,y,z)$，输出局部 tissue / rendering parameters。

因此这篇的复现重点不是 Gaussian container 或 density control，而是：

- 一个 MLP implicit field。
- 一个 ultrasound-specific ray rendering module。
- 一个 pose/image grid 到 ultrasound scan-line rays 的采样器。
- 与现有 loss/metric `computer` 的接口。

你的代码仓库已经有 loss/metric 的 `computer` 基础设施，因此 Ultra-NeRF 方法实现只需要把 renderer 输出接入现有 computer，不需要在方法模块里重新实现 loss 或 metric。论文给出了 loss 形式，但代码落地时应把它作为 computer 的可选配置，而不是写入 renderer。

## 1) Implicit field 表示

Ultra-NeRF 用 MLP 表示一个连续 3D volume function：

$$
F_\Theta:\mathbb{R}^3\rightarrow\mathbb{R}^5.
$$

对 ray 上 query point：

$$
q=r(t)=o+td,
$$

MLP 输出：

$$
\theta(q)=
\left[
\alpha(q),\ \beta(q),\ \rho_b(q),\ \rho_s(q),\ \phi(q)
\right].
$$

五个参数含义：

- $\alpha$：attenuation coefficient，补偿 TGC 后的有效衰减参数。
- $\beta$：reflectance / reflection coefficient。
- $\rho_b$：border probability，表示 query point 是否为 tissue boundary 的概率。
- $\rho_s$：scattering density，表示 query point 是否为 scattering point 的概率。
- $\phi$：scattering intensity / amplitude。

论文强调这些参数是 **isotropic physical tissue properties**，因此网络输入不包含 viewing direction。view-dependent B-mode appearance 由 rendering module 和 ray direction 间接产生，而不是由 MLP 直接接收方向编码。

## 2) Network architecture

论文 Appendix A.1 给出的网络设定：

- MLP 使用 8-layer architecture，参考 DeepSDF 的容量/效率折中。
- hidden layers 使用 ReLU。
- 第 5 层加入 skip connection：把 positional-encoded input 拼接到第 5 层 activation。
- 不使用 NeRF 原版的 viewing-direction branch。
- 输出 5 个参数。

输出 activation：

- $\beta,\rho_b,\rho_s,\phi$ 使用 sigmoid，约束到 $[0,1]$。
- $\alpha$ 可取任意正值，论文使用 absolute value activation。

实现建议：

- 若仓库已有 MLP/INR 基类，新增 `UltraNeRFField`，不要塞入 Gaussian container。
- 将 raw network output 拆成 5 个 named tensors，便于 renderer 和 debug map 使用。
- $\alpha=|\alpha_{\text{raw}}|$ 在 0 附近不可导点可能带来数值问题；若仓库更偏向稳定实现，可用 `softplus` 作为 implementation variant，但必须标注这不是论文原始 activation。

## 3) Positional encoding

论文 Appendix A.2 使用 NeRF 风格 positional encoding：

$$
\gamma(p)=
(\sin(2^0\pi p),\cos(2^0\pi p),\ldots,
\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)).
$$

默认 embedding size：

$$
L=10.
$$

对 3D 坐标 $q=(x,y,z)$，应对每个坐标维度分别编码，再拼接。若包含原始坐标本身，需按仓库已有 PE 约定处理；论文公式只写了 sinusoidal terms。

论文消融描述：不使用 positional encoding 时，模型无法有效捕捉 ultrasound 数据的结构。因此 PE 是必需模块。

## 4) Ray 与 query point 采样

论文中每条 ray 对应一条 ultrasound scan-line：

$$
r(t)=o+td.
$$

- $o$：image plane 顶部 / transducer side 的 scan-line origin。
- $d$：沿 scan-line 指向组织内部的方向。
- $t$：沿 ray 的深度。

每个像素对应 ray 上一个 query position。实现中需要由当前 frame pose、image geometry 和 pixel grid 生成：

- ray origins $o$。
- ray directions $d$。
- sample depths $t$ 或直接 3D query points $q$。

和 natural-image NeRF 不同，Ultra-NeRF 的 ray 是 ultrasound scan-line，不是 pinhole camera ray。实现时应显式定义 probe coordinate convention，例如 image height 对应 depth axis，image width 对应 lateral axis，然后用 pose 变换到 world/local scene 坐标。

## 5) Ultrasound volume rendering

Ultra-NeRF rendering 基于 Salehi et al. 的 ray-tracing ultrasound simulation。对每条 scan-line $r$，距离 $t$ 处的 echo energy：

$$
E(r,t)=R(r,t)+B(r,t).
$$

其中：

- $R(r,t)$：reflected energy，主要来自 tissue boundary / interface。
- $B(r,t)$：backscattered energy，来自 scattering medium。

### 5.1 Reflected energy

论文 Eq.4：

$$
R(r,t)=
|I(r,t)\cdot \beta(r,t)|
\cdot
PSF(r)\otimes G(r',t').
$$

含义：

- $I(r,t)$：到达距离 $t$ 时剩余 acoustic energy。
- $\beta(r,t)$：反射系数。
- $PSF(r)$：预定义 2D point-spread function。
- $G(r,t)$：boundary indicator；boundary 处为 1，否则为 0。
- $\otimes$：卷积。

$G(r,t)$ 由 Bernoulli distribution 采样：

$$
G(r,t)\sim\operatorname{Bernoulli}(\rho_b(r,t)).
$$

该 probabilistic boundary 设计表示网络对 tissue border / ray-boundary interaction 的不确定性。

### 5.2 Remaining energy

论文 Eq.5 将剩余能量 $I(r,t)$ 建模为 reflection loss 与 attenuation loss 的组合，并假设初始强度 $I_0=1$：

$$
I(r,t)
=
I_0
\prod_{n=0}^{t-1}
\mathcal{L}_{ref}(r,n)
\cdot
\exp\left(
-
\int_0^{t-1}
\alpha(r,n)\, f\, dt
\right).
$$

其中 $\mathcal{L}_{ref}$ 是由 $\beta$ 和 boundary indicator $G$ 控制的反射能量损失项。PDF 文本抽取的公式括号不够可靠；从物理和上下文看，代码中应实现为 **boundary-gated reflection loss**，即非边界点不应导致反射损失，边界点按 $\beta$ 减少 remaining energy。一个合理形式是：

$$
\mathcal{L}_{ref}(r,n)=1-\beta(r,n)G(r,n).
$$

attenuation 部分：

- $\alpha$ 是 compensated attenuation coefficient。
- $f$ 是 ultrasound frequency / scaling factor。
- TGC 未知，因此论文将 TGC-compensated attenuation 简化为单个可学习参数 $\alpha$，它只对应 physical attenuation up to unknown scale。

实现时建议使用 cumulative log-space 计算：

$$
\log I(r,t)
=
\sum_{n<t}\log(\mathcal{L}_{ref}(r,n)+\epsilon)
-
\sum_{n<t}\alpha(r,n) f\Delta t.
$$

### 5.3 Backscattered energy

论文 Eq.6：

$$
B(r,t)=I(r,t)\cdot PSF(r)\otimes T(r',t').
$$

scattering map：

$$
T(r,t)=H(r,t)\cdot \phi(r,t).
$$

其中：

$$
H(r,t)\sim\operatorname{Bernoulli}(\rho_s(r,t)).
$$

$H$ 表示 query point 是否为 scattering point；$\phi$ 表示 scattering point amplitude。论文描述 $\phi$ 对应一个均值为 $\phi$、单位方差的 normal sampling 的强度控制项。工程上如果要保持可微和稳定，可以先使用 deterministic expectation：

$$
\mathbb{E}[T(r,t)]=\rho_s(r,t)\phi(r,t),
$$

再根据需要引入 stochastic sampling。

### 5.4 PSF convolution

论文公式中 reflected 和 backscattered energy 都经过 $PSF(r)$ 卷积。实现上有两种选择：

- renderer 内部对 per-ray/per-sample map 做固定 PSF convolution。
- 先实现无 PSF 或简化 PSF 的版本跑通，再将 PSF 作为 renderer config 加入。

论文没有在正文给出 PSF kernel 的具体数值，因此 PSF shape/size 是 implementation config 或待查官方代码项。

## 6) Differentiability of rendering

Appendix C 指出：由于 $G,H$ 来自 Bernoulli sampling，rendering module 不是完全可微。即使如此，loss 仍可回传到其他连续变量，但 sampled discrete variables 本身不参与网络权重更新。

论文提到一种可微替代：

$$
G,H \sim \operatorname{RelaxedBernoulli}(\rho, T_{temp}),
$$

即用 low-temperature Relaxed Bernoulli 替代原始 Bernoulli。不过作者初步实验发现，Relaxed Bernoulli 对生成 B-mode 没有明显改变。

复现建议：

1. 第一版优先用 deterministic expectation：

$$
G=\rho_b,\qquad H=\rho_s,
$$

这样完全可微，便于集成现有训练框架。

2. 若要贴近论文 stochastic rendering，再加入 Bernoulli / Relaxed Bernoulli 开关。

3. 将 `boundary_sampling_mode` 和 `scatter_sampling_mode` 做成 config：`expectation | bernoulli | relaxed_bernoulli`。

## 7) Loss 接入

论文 Appendix A.3 给出训练 loss：

$$
L
=
\lambda\,SSIM(I',I)
+
(1-\lambda)
\sum_{i\in I}
(I'(i)-I(i))^2.
$$

论文设置：

$$
\lambda=0.9.
$$

注意原文把 SSIM 写在 loss 里，但不同代码库可能使用 $1-SSIM$、negative SSIM 或现成 SSIM loss。你的仓库已有 `computer` 基础设施，所以 Ultra-NeRF 方法模块不应实现这个 loss。正确边界：

- renderer 输出 `pred` 和可选 parameter maps。
- training loop 将 `pred/target/mask` 交给现有 computer。
- solver 接收 computer 产生的 scalar loss 做 `backward()` 与 optimizer step。
- 若要复现论文 loss，只在 computer 配置层设置 SSIM+L2，$\lambda=0.9$。

## 8) Parameter maps 与 debug outputs

Ultra-NeRF 的一个重要复现检查点是 parameter decomposition。renderer/field 可以保存：

- `attenuation_map`: $\alpha$。
- `reflectance_map`: $\beta$。
- `border_probability_map`: $\rho_b$。
- `scattering_density_map`: $\rho_s$。
- `scattering_intensity_map`: $\phi$。
- `remaining_energy_map`: $I(r,t)$。
- `reflected_energy_map`: $R(r,t)$。
- `backscattered_energy_map`: $B(r,t)$。
- `pred`: final B-mode echo $E(r,t)$ 或按实现命名的 rendered image。

这些 map 主要用于 debug/visualization，不直接放进 loss，除非现有 computer 已支持辅助监督。

## 9) 代码模块落盘建议

Ultra-NeRF 不是 Gaussian 方法，因此不要强行使用 `gaussian container` 或 `density control`。建议按你的仓库架构映射为：

### 9.1 Field / model container

如果仓库有通用 model/container 层，新增 `UltraNeRFField` 或 `implicit field container`：

- MLP 权重。
- positional encoding 配置 $L=10$。
- output activation。
- forward: `query_points -> alpha, beta, rho_b, rho_s, phi`。

如果只能复用已有 container 抽象，建议命名上明确为 `field_container`，不要叫 Gaussian container。

### 9.2 Solver

solver 负责：

- 注册 MLP 参数到 optimizer。
- 学习率、scheduler、checkpoint。
- 接收 computer scalar loss 反传。

论文没有在 PDF 中给出 optimizer、learning rate、batch size、training steps 等明确值。它们应视为 implementation config 或待查官方代码。不要从 UltraG-Ray/UltraGauss 迁移学习率。

### 9.3 Renderer / ops

建议新增：

`ops/ultra_nerf_renderer` 或 `renderers/ultra_nerf_renderer`

职责：

1. 由 pose + image geometry 生成 ultrasound scan-line rays。
2. 沿 ray 采样 query points。
3. 调用 `UltraNeRFField` 得到 5 个 physical parameters。
4. 计算 boundary/scattering indicators 或 expectation。
5. 计算 remaining energy $I(r,t)$。
6. 计算 reflected energy $R(r,t)$。
7. 计算 backscattered energy $B(r,t)$。
8. 应用 PSF convolution。
9. 输出 rendered B-mode 与 parameter maps。

该 renderer 可先用 PyTorch 实现，确认公式和梯度；若性能成为瓶颈，再把 ray sampling、cumulative energy、PSF convolution 写成 CUDA op。

### 9.4 Computer

loss/metric 保持现有基础设施：

- `pred/target/mask` 进入 computer。
- SSIM/L2 等组合由 computer 配置。
- metric 不放进 renderer 或 field。

### 9.5 模块责任边界速查

| 内容 | 建议落盘位置 |
|---|---|
| MLP weights | `UltraNeRFField` / model container |
| positional encoding | `UltraNeRFField` |
| output activation | `UltraNeRFField` |
| ray generation | `ops/ultra_nerf_renderer` |
| ray sampling | `ops/ultra_nerf_renderer` |
| Bernoulli / expectation rendering mode | `ops/ultra_nerf_renderer` config |
| remaining energy accumulation | `ops/ultra_nerf_renderer` |
| PSF convolution | `ops/ultra_nerf_renderer` or submodule |
| optimizer/scheduler/checkpoint | solver |
| loss/metric | existing `computer` |

# Method-related Settings

## 1) 必需输入

每个训练样本至少包含：

- `image`: 单通道 B-mode ultrasound image。
- `pose`: 当前帧 probe pose，用于生成 scan-line rays。
- `probe_geometry`: image spacing、origin、depth range、lateral range、scan-line direction convention。
- `mask/valid_region`: 可选，由 data pipeline 或 computer 使用。

Ultra-NeRF 假设有 pose-annotated sweeps。pose learning 不是本文方法的一部分。

## 2) 默认方法参数

| 参数 | 论文值 | 落盘位置 |
|---|---:|---|
| MLP layers | `8` | field config |
| activation | ReLU | field config |
| skip connection | concat input to 5th layer activation | field config |
| output dim | `5` | field config |
| PE size $L$ | `10` | field config |
| direction input to MLP | no | field config |
| $\beta,\rho_b,\rho_s,\phi$ activation | sigmoid | field config |
| $\alpha$ activation | abs | field config |
| paper loss lambda | `0.9` | computer config if reproducing paper loss |

## 3) 论文未公开但代码必须配置的参数

- MLP hidden width。
- optimizer type。
- learning rate。
- training steps / epochs。
- batch size。
- samples per ray / ray sampling resolution.
- near/far depth range.
- ultrasound frequency/scaling factor $f$。
- depth step $\Delta t$。
- PSF kernel shape/size。
- `boundary_sampling_mode`: expectation / Bernoulli / Relaxed Bernoulli。
- `scatter_sampling_mode`: expectation / Bernoulli / Relaxed Bernoulli。
- Relaxed Bernoulli temperature。
- intensity normalization / log compression handling。

这些参数应进入 config，并优先查官方代码或已有仓库习惯。

## 4) 最小验证协议

1. **Field shape test**：输入 batch query points，输出 5 个 map，shape 和 activation range 正确。
2. **PE test**：$L=10$ 时维度符合预期，且不含方向输入。
3. **Ray test**：identity pose 下 scan-line ray 与图像 depth axis 一致。
4. **Energy accumulation test**：$\alpha$ 或 $\beta$ 增大时，深部 remaining energy 不增。
5. **Rendering mode test**：expectation mode 完全可微；Bernoulli mode 下 discrete sample 不应被误认为可微。
6. **PSF test**：关闭 PSF 时 renderer 仍可跑通；开启 PSF 后图像 shape 不变。
7. **Computer integration test**：renderer 输出可被现有 computer 接收并产生可反传 scalar loss。
8. **Debug map test**：保存 $\alpha,\beta,\rho_b,\rho_s,\phi$ 等 parameter maps，用于确认网络没有退化到全黑或常数输出。

## 5) 推荐保存的复现产物

- `config.yaml`: field architecture、PE、ray sampling、PSF、rendering mode、computer 配置名。
- checkpoint: MLP weights 与 optimizer state。
- renderer logs: rays/frame、samples/ray、render time、memory。
- parameter maps: $\alpha,\beta,\rho_b,\rho_s,\phi,I,R,B$。
- validation renders: 输入视角重渲染、未见视角重渲染、absolute difference map。
