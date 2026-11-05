# Pose-Uncertainty-Aware US-GS Reconstruction 详细落地方案

面向服务器端 Codex / Cursor / 研发 agent 使用。

本文档目标：把当前“FiMA/PEM pose prior + `us_focus` Gaussian reconstruction”的简单串联，升级为一个真正有框架创新的方向：

> **Pose-Uncertainty-Aware Gaussian Splatting for Freehand Ultrasound Reconstruction**

核心思想：

- 不再继续押注 “GS photometric loss 反向优化 pose”；
- 不再满足于 “pose estimator 输出 pose，GS 消费 pose” 的弱串联；
- 改为让 pose estimator 输出 **pose prior + pose uncertainty**；
- GS reconstruction 显式利用 pose uncertainty，避免错误 pose prior 污染 3D Gaussian representation；
- 将 learned pose prior 从一个 deterministic camera parameter，提升为 Bayesian / probabilistic prior。

---

## 0. 当前背景与路线调整

### 0.1 已经验证失败或不适合作主线的方向

当前实验已经说明：

1. **Online reconstruction 不适合作主线**  
   online 局部性太强，通常比 offline 精度低，而且速度更慢。

2. **GS 阶段自由优化 pose 不适合作主线**  
   在真实训练中，Gaussian attributes 和 geometry 会吸收 pose error，导致 image loss 不一定把 pose 拉向 GT。  
   小 pose error 时 fixed prior 几乎无损；大 pose error 时 GS refine 也修不动；受约束 translation-only refine 只有小幅收益。

3. **简单串联 pose estimator + GS reconstructor 不够强**  
   如果方法只是：
   ```text
   image sequence -> FiMA/PEM pose prior -> GS reconstruction
   ```
   这个故事太像 pipeline engineering，创新性不足。

### 0.2 新核心假设

新的核心假设不是：

> GS can correct pose.

而是：

> Learned pose priors are uncertain; the reconstruction model should consume both the pose estimate and its uncertainty.

也就是说：

$$
q_\theta(T_i\mid I_{1:N})
=
\mathcal N(\hat T_i, \Sigma_i)
$$

其中：

- $\hat T_i$：FiMA/PEM 网络预测的 pose prior；
- $\Sigma_i$：pose prior uncertainty；
- GS 训练时不再把 $\hat T_i$ 当作 GT pose，而是把它当成一个 noisy probabilistic prior。

---

## 1. 推荐论文主线

建议将方法主线定义为：

> **Relocalization-guided, pose-uncertainty-aware ultrasound Gaussian reconstruction.**

更具体：

1. FiMA/PEM 网络从超声图像序列预测 pose prior；
2. 同时预测或估计 pose uncertainty；
3. `us_focus` Gaussian renderer 在 pose prior 下进行重建；
4. pose uncertainty 用于：
   - frame-level loss weighting；
   - outlier frame downweighting；
   - pose uncertainty marginalization；
   - Gaussian footprint inflation；
   - reconstruction robustness analysis。

最终叙事：

> Instead of treating learned pose as deterministic ground truth or trying to optimize it away with ambiguous photometric gradients, we model pose as an uncertain prior and propagate this uncertainty into Gaussian reconstruction.

---

## 2. 与贝叶斯不确定性的联系

### 2.1 Aleatoric vs Epistemic uncertainty

可以借用 Bayesian deep learning 里的标准表述：

- **Aleatoric uncertainty**：数据本身噪声 / 输入相关噪声。  
  在这里可以对应：
  - 超声图像模糊；
  - speckle noise；
  - poor acoustic contact；
  - 局部图像不可辨识；
  - 某些 slice 本身难以定位。

- **Epistemic uncertainty**：模型知识不足。  
  在这里可以对应：
  - pose estimator 对某些 out-of-distribution 图像不确定；
  - 训练数据不足；
  - sweep / anatomy domain shift。

Kendall & Gal 2017 提出在视觉任务中联合建模 aleatoric 和 epistemic uncertainty；Gal & Ghahramani 的 MC Dropout 将 dropout 解释为 Bayesian approximation，用于估计模型不确定性。

本项目第一阶段不需要实现复杂 Bayesian NN，可以先用 **heteroscedastic pose regression** 预测输入相关的不确定性。

---

## 3. 总体概率建模

### 3.1 Deterministic pose pipeline

原始 deterministic pipeline 是：

$$
\hat T_i = f_\theta(I_{1:N})_i
$$

$$
\mathcal G^*
=
\arg\min_\mathcal G
\sum_i
\mathcal L_{\mathrm{img}}
\left(
R_{\mathrm{US}}(\mathcal G,\hat T_i),I_i
\right)
$$

问题：

- $\hat T_i$ 被当作硬 pose；
- pose 错误会直接污染 GS reconstruction；
- GS photometric loss 不一定能修正 pose；
- bad pose frame 没有被显式降权。

### 3.2 Probabilistic pose prior

新建模：

$$
q_\theta(T_i\mid I_{1:N})
=
\mathcal N(\hat T_i,\Sigma_i)
$$

其中：

$$
\hat T_i
=
\mathrm{Exp}(\hat \xi_i)
$$

或如果是 adjacent transform：

$$
\hat T_{i\to i+1}
=
\mathrm{Exp}(\hat \xi_{i\to i+1})
$$

$\Sigma_i$ 可以是：

1. scalar uncertainty；
2. translation / rotation 两个 uncertainty；
3. 6D diagonal covariance；
4. full $6\times 6$ covariance。

第一版强烈建议使用：

$$
\sigma_{\mathrm{trans},i},\quad \sigma_{\mathrm{rot},i}
$$

两个 scalar。

---

## 4. 方法模块设计

建议分三层实现：

```text
Level 1: uncertainty-weighted GS loss
Level 2: pose-uncertainty inflated us_focus footprint
Level 3: Monte Carlo pose marginalization
```

不要一开始全做。按层推进。

---

# Part A：Uncertainty-Weighted GS Reconstruction

## 5. Frame-level uncertainty weighting

### 5.1 目标

避免低质量 pose prior 对 GS reconstruction 造成过强约束。

原始 loss：

$$
\mathcal L_{\mathrm{rec}}
=
\sum_i
\mathcal L_i
$$

其中：

$$
\mathcal L_i
=
\mathcal L_{\mathrm{img}}
\left(
R_{\mathrm{US}}(\mathcal G,\hat T_i),I_i
\right)
$$

新 loss：

$$
\mathcal L_{\mathrm{rec}}
=
\sum_i
w_i
\mathcal L_i
$$

其中：

$$
w_i = \frac{1}{\sigma_i^2+\epsilon}
$$

或者：

$$
w_i=\exp(-\alpha\sigma_i)
$$

推荐第一版使用 bounded exponential：

$$
w_i
=
w_{\min}
+
(1-w_{\min})
\exp(-\alpha u_i)
$$

其中：

- $u_i$：uncertainty score；
- $w_{\min}$：最小权重，防止某些帧完全无梯度；
- $\alpha$：权重衰减强度。

默认：

```yaml
pose_uncertainty:
  use_weighting: true
  weight_mode: exp
  alpha: 1.0
  min_weight: 0.1
  max_weight: 1.0
```

### 5.2 为什么不用直接 $1/\sigma^2$

直接：

$$
w_i=\frac{1}{\sigma_i^2+\epsilon}
$$

容易在 $\sigma_i$ 很小时权重爆炸，导致训练不稳定。

因此建议实际实现用：

$$
w_i
=
\operatorname{clamp}
\left(
\frac{1}{\sigma_i^2+\epsilon},
w_{\min},
w_{\max}
\right)
$$

或 exponential mapping。

---

## 6. FiMA/PEM uncertainty head

### 6.1 原始 pose head

假设当前 FiMA/PEM 输出 adjacent transform vector：

$$
\hat \xi_i\in\mathbb R^6
$$

其中：

$$
\hat \xi_i=
[\hat r_x,\hat r_y,\hat r_z,\hat t_x,\hat t_y,\hat t_z]
$$

### 6.2 新增 uncertainty head

新增输出：

#### Option A：两个 scalar

$$
\log\sigma_{\mathrm{rot},i}^2,\quad
\log\sigma_{\mathrm{trans},i}^2
$$

#### Option B：6D diagonal

$$
\log \boldsymbol{\sigma}_i^2\in\mathbb R^6
$$

第一版建议 Option A。

网络输出维度从：

```text
6
```

改为：

```text
8
```

前 6 维是 pose mean，后 2 维是 log variance。

---

## 7. Heteroscedastic pose regression loss

### 7.1 Two-scalar version

将 rotation 和 translation 分开：

$$
e_{\mathrm{rot},i}
=
\|\xi_{\mathrm{rot},i}-\hat\xi_{\mathrm{rot},i}\|_2^2
$$

$$
e_{\mathrm{trans},i}
=
\|\xi_{\mathrm{trans},i}-\hat\xi_{\mathrm{trans},i}\|_2^2
$$

令：

$$
s_{\mathrm{rot},i}=\log\sigma_{\mathrm{rot},i}^2
$$

$$
s_{\mathrm{trans},i}=\log\sigma_{\mathrm{trans},i}^2
$$

NLL loss：

$$
\mathcal L_{\mathrm{pose},i}
=
\frac{1}{2}\exp(-s_{\mathrm{rot},i})e_{\mathrm{rot},i}
+
\frac{1}{2}s_{\mathrm{rot},i}
+
\frac{1}{2}\exp(-s_{\mathrm{trans},i})e_{\mathrm{trans},i}
+
\frac{1}{2}s_{\mathrm{trans},i}
$$

实际实现建议加 clamp：

```python
logvar = logvar.clamp(min=-8.0, max=4.0)
```

避免 $\sigma$ 过小或过大。

### 7.2 6D diagonal version

$$
\mathcal L_{\mathrm{pose},i}
=
\frac12
\sum_{d=1}^6
\left[
\exp(-s_{i,d})
(\xi_{i,d}-\hat\xi_{i,d})^2
+
s_{i,d}
\right]
$$

第一版不建议直接用 6D，因为 calibration 更难，且 rotation/translation 单位差异大。

---

## 8. 从 pose uncertainty 到 frame weight

### 8.1 简单标量 uncertainty score

由两个 log variance 合成：

$$
u_i
=
\beta_r s_{\mathrm{rot},i}
+
\beta_t s_{\mathrm{trans},i}
$$

或者转成标准差：

$$
u_i
=
\beta_r \sigma_{\mathrm{rot},i}
+
\beta_t \sigma_{\mathrm{trans},i}
$$

第一版推荐用 normalized scalar：

```python
u = 0.5 * normalize(logvar_rot) + 0.5 * normalize(logvar_trans)
```

再映射权重：

```python
weight = min_weight + (max_weight - min_weight) * torch.exp(-alpha * u)
```

### 8.2 Calibration 后再用于 GS

FiMA/PEM uncertainty 训练完后，必须验证：

1. uncertainty-error correlation；
2. high uncertainty frames 是否对应更大 pose error；
3. uncertainty 能否检测 corrupted pose / outlier frame。

不能直接相信网络输出的 uncertainty。

---

## 9. 第一阶段实验：Synthetic corruption + oracle uncertainty

这是最重要的 proof-of-concept。

### 9.1 目的

验证：

> 当 pose prior 中混入坏帧时，uncertainty-aware GS 能保护 reconstruction，不被错误 pose 污染。

### 9.2 数据构造

从已有 FiMA prior 或 GT pose 出发：

```text
80% frames: clean or realistic FiMA pose
20% frames: corrupted pose
```

corruption 类型：

```text
translation corruption: 0.2 mm / 0.5 mm / 1.0 mm
rotation corruption: 0.1 deg / 0.5 deg / 1.0 deg
low-frequency drift
random frame outlier
```

### 9.3 Oracle uncertainty

直接给 corrupted frames 高 uncertainty：

```python
u_i = 0.0 for clean frames
u_i = 1.0 for corrupted frames
```

对应权重：

```python
w_i = 1.0 for clean frames
w_i = 0.1 for corrupted frames
```

### 9.4 对比方法

| Method | Description |
|---|---|
| Clean GT / clean FiMA | upper bound |
| Uniform corrupted prior | 所有帧等权 |
| Oracle downweight | corrupted frame 低权重 |
| Oracle hard filter | 直接删除 corrupted frame |
| Predicted uncertainty weighting | 后续版本 |

### 9.5 预期结果

如果这个方向成立：

```text
Uniform corrupted prior: PSNR/LPIPS 明显变差
Oracle downweight: 明显恢复
Oracle hard filter: 接近上界
```

只有这个实验成立，才继续训练 uncertainty head。

### 9.6 Synthetic liver 的 FiMA 训练划分约束

FiMA/PEM 是姿态估计模型，不是场景表征模型。因此在 `synthetic_liver`
上不要按单个 sweep 专门训练一个 FiMA 模型，否则会得到过于乐观的
sweep-specific prior。

后续先只做以下两件事，不做 leave-one-sweep-out，也不做每个 sweep 单独训练：

1. **六个 sweep 数据分布统计**
   - sweep 集合：`l1/l2/l3/r1/r2/r3`。
   - 统计图像分布：强度均值/方差、直方图、前景比例、相邻帧 L1/SSIM。
   - 统计运动分布：相邻帧平移范数、旋转角、p50/p95/max、轨迹长度、轨迹 bbox。
   - 统计 split 一致性：每个 sweep 均使用与 GS 一致的 `holdout_interval`
     train/test 划分，FiMA 不允许看到 GS 的 test frames。
   - 输出报告用于判断这六个 sweep 是否可以作为同一 synthetic_liver
     distribution pooled training。

2. **若分布统计支持 pooled training，则只做 pooled FiMA**
   - FiMA 训练集：六个 sweep 的 train frames 混合。
   - FiMA 测试集：六个 sweep 的 test frames 混合。
   - FiMA window 不能跨 sweep，且不能跨该 sweep 的 holdout test frame。
   - GS 仍然每个 sweep 单独训练和测试；FiMA 只提供该 sweep 的 pose prior
     与 uncertainty。

这两项先记录为计划，不在本阶段执行。当前阶段优先解决 FiMA 实际使用
`mamba_ssm` 的环境和代码路径。

---

# Part B：Pose-Uncertainty-Inflated us_focus Footprint

## 10. 为什么需要 footprint inflation

Frame-level weighting 只是在 loss 层面降低坏帧影响，但它没有改变单个 pixel 的 Gaussian contribution。

更进一步，可以把 pose uncertainty 注入到 renderer：

$$
\Sigma_{2d}^{\mathrm{eff}}
=
\Sigma_{2d}
+
\Sigma_{\mathrm{pose},2d}
$$

直觉：

> 如果 pose prior 不确定，那么该 slice 对 3D geometry 的约束不应该是尖锐的，而应该是更宽、更软的。

这比直接修 pose 更合理。

---

## 11. 简化版 footprint inflation

### 11.1 有效 2D covariance

原始 `us_focus`：

$$
w_{g,p}
=
o_g r_{g,i}
(\det\Sigma_{2d,g}+\epsilon)^{-\gamma}
\exp
\left(
-\frac12
d^T\Sigma_{2d,g}^{-1}d
\right)
$$

修改为：

$$
\Sigma_{2d,g}^{\mathrm{eff}}
=
\Sigma_{2d,g}
+
\begin{bmatrix}
\sigma_{\mathrm{pose,lat},i}^2 & 0\\
0 & \sigma_{\mathrm{pose,ax},i}^2
\end{bmatrix}
$$

然后：

$$
w_{g,p}
=
o_g r_{g,i}
(\det\Sigma_{2d,g}^{\mathrm{eff}}+\epsilon)^{-\gamma}
\exp
\left(
-\frac12
d^T
(\Sigma_{2d,g}^{\mathrm{eff}})^{-1}
d
\right)
$$

### 11.2 从 pose uncertainty 到 lat/ax inflation

第一版简单映射：

$$
\sigma_{\mathrm{pose,lat}}
=
k_t\sigma_{\mathrm{trans}}
+
k_r z_{\mathrm{ax}}\sigma_{\mathrm{rot}}
$$

$$
\sigma_{\mathrm{pose,ax}}
=
k_t\sigma_{\mathrm{trans}}
$$

其中：

- translation uncertainty 直接带来平移误差；
- rotation uncertainty 在深度 $z_{\mathrm{ax}}$ 处转化为 lateral uncertainty；
- $k_t,k_r$ 是超参数。

默认：

```yaml
pose_uncertainty:
  use_footprint_inflation: true
  inflation_mode: diag
  k_trans: 1.0
  k_rot: 1.0
  inflation_min_mm: 0.0
  inflation_max_mm: 1.0
```

### 11.3 更严格的 Jacobian 版本

理论上：

$$
\Sigma_{\mathrm{pose},2d}
=
J_T\Sigma_TJ_T^T
$$

其中：

$$
J_T=
\frac{\partial \mu_{2d}}{\partial \xi_T}
$$

但第一版不要做完整 Jacobian。先做 diagonal approximation。

---

## 12. CUDA 修改点：footprint inflation

### 12.1 us_focus_params 增加参数

当前已有：

```text
us_focus_params[0]  sigma_ele_0
...
us_focus_params[11] use_lateral_focus
```

建议新增或单独建 `pose_uncertainty_params`：

```text
pose_uncertainty_params[0] use_frame_weighting
pose_uncertainty_params[1] use_footprint_inflation
pose_uncertainty_params[2] det_gamma_uncertain
pose_uncertainty_params[3] k_trans
pose_uncertainty_params[4] k_rot
pose_uncertainty_params[5] inflation_min_mm
pose_uncertainty_params[6] inflation_max_mm
pose_uncertainty_params[7] weight_min
pose_uncertainty_params[8] weight_max
pose_uncertainty_params[9] weight_alpha
```

### 12.2 新增 per-frame uncertainty tensor

Python -> CUDA 传入：

```python
pose_uncertainty: Tensor
```

shape 第一版：

```text
[num_frames, 2]
```

其中：

```text
[:, 0] = sigma_rot
[:, 1] = sigma_trans
```

如果 renderer 一次只渲染一张 slice，则也可以传：

```text
[2]
```

对应当前 slice。

### 12.3 CUDA eval 修改

在 `us_focus_eval_at_pixel` 内，在构造 `S2_00, S2_01, S2_11` 后：

```cpp
float sigma_rot = pose_uncertainty[0];
float sigma_trans = pose_uncertainty[1];

float depth = fmaxf(mu_ax - near_mm, 0.f);

float infl_lat = k_trans * sigma_trans + k_rot * depth * sigma_rot;
float infl_ax  = k_trans * sigma_trans;

infl_lat = clamp(infl_lat, inflation_min_mm, inflation_max_mm);
infl_ax  = clamp(infl_ax,  inflation_min_mm, inflation_max_mm);

S2_00 += infl_lat * infl_lat;
S2_11 += infl_ax  * infl_ax;
```

注意：这会影响：

- 2D inverse；
- determinant normalization；
- tile mapper footprint。

### 12.4 Mapper 也必须使用 inflated footprint

如果 shader 使用：

$$
\Sigma_{2d}^{eff}
$$

mapper 也必须用同样的 covariance 估计 tile radius：

$$
r_{\mathrm{lat}}
=
k_\sigma\sqrt{\Sigma_{2d}^{eff}[0,0]}
$$

$$
r_{\mathrm{ax}}
=
k_\sigma\sqrt{\Sigma_{2d}^{eff}[1,1]}
$$

否则会漏贡献。

---

# Part C：Monte Carlo Pose Marginalization

## 13. MC pose marginalization

### 13.1 理论目标

目标函数：

$$
\mathcal L
=
\sum_i
\mathbb E_{T_i\sim q_\theta(T_i)}
\left[
\mathcal L_{\mathrm{img}}
(
R_{\mathrm{US}}(\mathcal G,T_i),I_i
)
\right]
$$

用 Monte Carlo 近似：

$$
\mathcal L_i
\approx
\frac{1}{K}
\sum_{k=1}^K
\mathcal L_{\mathrm{img}}
(
R_{\mathrm{US}}(\mathcal G,T_i^{(k)}),I_i
)
$$

其中：

$$
T_i^{(k)}
=
\hat T_i\exp(\delta\xi_i^{(k)})
$$

$$
\delta\xi_i^{(k)}
\sim \mathcal N(0,\Sigma_i)
$$

### 13.2 实现建议

第一版不要在 CUDA 内实现 MC。

直接在 Python training loop 中：

```python
loss = 0
for k in range(K):
    pose_k = perturb_pose(pose_prior, pose_uncertainty)
    pred_k = renderer(..., pose=pose_k)
    loss += image_loss(pred_k, gt)
loss = loss / K
```

默认：

```yaml
pose_uncertainty:
  use_mc_marginalization: false
  mc_samples: 2
```

只有在小规模 ablation 中打开。

### 13.3 作用

MC marginalization 是理论上最干净的 Bayesian baseline，但计算成本较高。它可以作为：

- upper bound；
- appendix；
- 验证 footprint inflation 是否能近似 MC。

---

# Part D：FiMA/PEM Uncertainty 实现计划

## 14. 修改 FiMA 网络

### 14.1 输出 head

当前 paper-style FiMA 输出 adjacent edge：

```python
pred_edge = model(images)
# shape: [B, N-1, 6]
```

修改为：

```python
pred_mean, pred_logvar = model(images)
# pred_mean: [B, N-1, 6]
# pred_logvar: [B, N-1, 2] or [B, N-1, 6]
```

第一版：

```python
pred_logvar_rot   = pred_logvar[..., 0]
pred_logvar_trans = pred_logvar[..., 1]
```

### 14.2 初始化

为了避免 logvar 崩，初始化 logvar head bias：

```python
logvar_rot_bias = log(initial_rot_sigma ** 2)
logvar_trans_bias = log(initial_trans_sigma ** 2)
```

根据 normalized target 空间设置。

如果 target 已标准化，初始：

```python
initial_sigma = 1.0
logvar_bias = 0.0
```

### 14.3 Loss 实现

```python
def heteroscedastic_pose_loss(pred_mean, pred_logvar, target):
    rot_err = ((pred_mean[..., :3] - target[..., :3]) ** 2).sum(dim=-1)
    trans_err = ((pred_mean[..., 3:] - target[..., 3:]) ** 2).sum(dim=-1)

    logvar_rot = pred_logvar[..., 0].clamp(-8.0, 4.0)
    logvar_trans = pred_logvar[..., 1].clamp(-8.0, 4.0)

    loss_rot = 0.5 * torch.exp(-logvar_rot) * rot_err + 0.5 * logvar_rot
    loss_trans = 0.5 * torch.exp(-logvar_trans) * trans_err + 0.5 * logvar_trans

    return (loss_rot + loss_trans).mean()
```

### 14.4 输出到 chain pose

FiMA chain pose 仍然使用 `pred_mean`。

Uncertainty chain 可以简单从 adjacent edge uncertainty 聚合到 frame uncertainty：

$$
\sigma_{T_i}^2
=
\sum_{j<i}
\sigma_{j\to j+1}^2
$$

也就是：

```python
frame_var = edge_var.cumsum(dim=sequence_dim)
frame_sigma = torch.sqrt(frame_var + eps)
```

注意：这会导致后面帧 uncertainty 变大，符合 chaining error accumulation 直觉。

---

## 15. Uncertainty calibration 评估

### 15.1 相关性

计算：

```python
pose_error_i
uncertainty_i
```

指标：

```text
Pearson correlation
Spearman correlation
```

期望：

```text
uncertainty 越高，pose error 越大
```

### 15.2 Bad-pose detection AUC

定义 bad pose：

```text
translation error > threshold
or rotation error > threshold
```

用 uncertainty score 做 binary classifier，算：

```text
AUROC
AUPRC
```

### 15.3 Calibration curve

按 uncertainty 分 bin：

```text
low / medium / high uncertainty
```

每个 bin 统计：

```text
mean predicted sigma
mean actual error
```

如果 predicted sigma 和 actual error 单调相关，就可以用。

---

# Part E：实验设计

## 16. 实验组 1：Proof-of-concept with oracle uncertainty

### 16.1 目标

先不训练 uncertainty head，只验证 uncertainty-aware GS reconstruction 的上限。

### 16.2 设置

对已有 sweep，例如 `r2`：

```text
pose = FiMA prior 或 GT pose
corrupt_ratio = 0.1 / 0.2 / 0.3
corrupt_trans = 0.2 / 0.5 / 1.0 mm
corrupt_rot = 0.1 / 0.5 / 1.0 deg
```

### 16.3 方法对比

| Method | Description |
|---|---|
| clean_pose | 无 corruption，上界 |
| corrupted_uniform | corrupted pose，所有帧等权 |
| corrupted_oracle_weight | corrupted frame 降权 |
| corrupted_oracle_filter | corrupted frame 删除 |
| corrupted_inflation | corrupted frame footprint inflation |
| corrupted_weight+inflation | 二者结合 |

### 16.4 期望结论

如果成功：

```text
corrupted_uniform 明显差
oracle_weight 恢复一部分
oracle_filter 接近 clean 上界
inflation 比 uniform 稳定
weight+inflation 最稳
```

---

## 17. 实验组 2：Predicted uncertainty from FiMA

### 17.1 目标

验证 FiMA uncertainty 是否有用。

### 17.2 训练

训练 heteroscedastic FiMA：

```text
image-only sequence edge FiMA
output: mean + logvar
loss: heteroscedastic NLL
```

### 17.3 评估

输出：

```text
pose mean error
uncertainty-error Spearman
bad pose AUROC
calibration curve
```

### 17.4 接入 GS

对比：

| Method | Description |
|---|---|
| FiMA fixed prior | 当前 baseline |
| FiMA + predicted uncertainty weighting | 用 predicted uncertainty 加权 |
| FiMA + predicted footprint inflation | 用 predicted uncertainty inflate footprint |
| FiMA + oracle uncertainty | upper bound |

---

## 18. 实验组 3：MC marginalization baseline

### 18.1 目标

验证真正 Bayesian pose marginalization 是否有帮助。

### 18.2 设置

只在小规模跑：

```text
K = 2 / 4
subset frames
short training
```

### 18.3 对比

| Method | Description |
|---|---|
| deterministic pose | 固定 mean pose |
| MC marginalization K=2 | 2 samples |
| MC marginalization K=4 | 4 samples |
| footprint inflation | 近似方案 |

如果 footprint inflation 接近 MC marginalization，就可以把它作为高效近似。

---

## 19. 实验组 4：Realistic PEM-like uncertainty

### 19.1 不要只用逐帧随机噪声

更真实的 pose prior error：

1. low-frequency drift；
2. calibration bias；
3. local tracking failure；
4. occasional outlier frame；
5. smooth random walk。

### 19.2 推荐 corruption 类型

#### Translation random walk

$$
\delta t_i=\delta t_{i-1}+\epsilon_i
$$

#### Sinusoidal drift

$$
\delta t_i=A\sin(\omega i+\phi)
$$

#### Calibration bias

$$
T_i^{prior}=T_i^{gt}T_{\mathrm{bias}}
$$

#### Outlier frames

随机选 10% frames：

$$
T_i^{prior}=T_i^{gt}\Delta T_{\mathrm{large}}
$$

### 19.3 目的

证明 uncertainty-aware reconstruction 对真实 tracking failure 更稳。

---

# Part F：代码落地计划

## 20. Stage 0：冻结当前稳定 renderer

当前 `us_focus fast_fidelity` 作为默认 renderer。不要继续大改：

```yaml
shader_weight_mode: us_focus
compositing_mode: additive
mapper_kind: tile
us_focus_det_gamma: 0.25
```

保留：

```text
probe3d baseline
plane_cond2d ablation
us_focus fast_fidelity
us_focus perceptual
```

---

## 21. Stage 1：Oracle uncertainty weighting

### 21.1 目标

不动 FiMA，不动 CUDA，只在 training loop 里给每帧加 weight。

### 21.2 实现

在 dataset 或 batch 中添加：

```python
frame_weight
```

loss：

```python
loss_img = image_loss(pred, gt, reduction="none")
loss_img = loss_img.mean(dim=[1,2,3])  # per-frame
loss = (frame_weight * loss_img).sum() / (frame_weight.sum() + 1e-6)
```

### 21.3 配置

```yaml
pose_uncertainty:
  enabled: true
  source: oracle
  weighting: true
  min_weight: 0.1
  corrupt_ratio: 0.2
  corrupt_weight: 0.1
```

---

## 22. Stage 2：Pose corruption generator

新增脚本：

```text
scripts/methods/ours/pose_uncertainty_prior/generate_pose_prior_corruption.py
```

支持：

```bash
python scripts/methods/ours/pose_uncertainty_prior/generate_pose_prior_corruption.py   --pose_path ...   --out_pose_path ...   --out_uncertainty_path ...   --corrupt_ratio 0.2   --corrupt_trans_mm 0.5   --corrupt_rot_deg 0.5   --mode random_outlier
```

输出：

```text
corrupted_poses.npy
oracle_uncertainty.npy
oracle_frame_weight.npy
corruption_mask.npy
```

---

## 23. Stage 3：FiMA uncertainty head

修改 FiMA：

1. 输出 mean + logvar；
2. 实现 heteroscedastic NLL；
3. 保存 frame uncertainty；
4. 做 calibration evaluation。

---

## 24. Stage 4：Uncertainty calibration script

新增：

```text
scripts/eval/evaluate_pose_uncertainty_calibration.py
```

输出：

```text
Spearman correlation
Pearson correlation
AUROC bad-pose detection
calibration bins
plots if possible
```

---

## 25. Stage 5：Predicted uncertainty weighting

让 GS training 读取 FiMA uncertainty 文件，并将其映射到 frame loss weight。

---

## 26. Stage 6：Footprint inflation forward

修改 `us_focus` CUDA：

1. pass current slice uncertainty；
2. inflate $\Sigma_{2d}$；
3. mapper 同步使用 inflated footprint；
4. uncertainty=0 时与原输出 allclose。

---

## 27. Stage 7：MC marginalization

Python training loop 实现 optional MC pose sampling。

---

# Part G：配置草案

## 28. Oracle weighting config

```yaml
pose_uncertainty:
  enabled: true
  source: oracle
  use_frame_weighting: true
  use_footprint_inflation: false
  use_mc_marginalization: false

  corrupt_ratio: 0.2
  corrupt_trans_mm: 0.5
  corrupt_rot_deg: 0.5
  corrupt_weight: 0.1

  weight_mode: exp
  weight_alpha: 1.0
  min_weight: 0.1
  max_weight: 1.0
```

## 29. Predicted uncertainty config

```yaml
pose_uncertainty:
  enabled: true
  source: predicted
  uncertainty_path: outputs/preproduction/baseline_reproduction/fima/frame_pose_uncertainty.npy

  use_frame_weighting: true
  use_footprint_inflation: false

  normalize_uncertainty: true
  weight_mode: exp
  weight_alpha: 1.0
  min_weight: 0.1
  max_weight: 1.0
```

## 30. Footprint inflation config

```yaml
pose_uncertainty:
  enabled: true
  source: predicted
  uncertainty_path: outputs/preproduction/baseline_reproduction/fima/frame_pose_uncertainty.npy

  use_frame_weighting: true
  use_footprint_inflation: true

  inflation_mode: diag
  k_trans: 1.0
  k_rot: 1.0
  inflation_min_mm: 0.0
  inflation_max_mm: 1.0
```

## 31. MC marginalization config

```yaml
pose_uncertainty:
  enabled: true
  source: predicted
  use_mc_marginalization: true
  mc_samples: 2
  mc_only_for_high_uncertainty: true
  high_uncertainty_percentile: 80
```

---

# Part H：评估指标

## 32. Reconstruction metrics

继续使用：

```text
PSNR
LPIPS
L1
SSIM
```

## 33. Pose prior metrics

```text
translation error
rotation error
step error
chain drift
```

## 34. Uncertainty metrics

```text
Spearman(error, uncertainty)
Pearson(error, uncertainty)
AUROC for bad pose detection
AUPRC for bad pose detection
calibration bins
```

## 35. Robustness metrics

在 corrupted pose prior 下：

```text
quality drop from clean
recovery ratio
```

定义：

$$
\mathrm{recovery}
=
\frac{
M_{\mathrm{ours}}-M_{\mathrm{corrupted}}
}{
M_{\mathrm{clean}}-M_{\mathrm{corrupted}}
}
$$

对 PSNR 越大越好。  
对 LPIPS 越小越好，需要反向定义。

---

# Part I：Codex 任务拆分

## 36. Task 1：Oracle uncertainty weighting

**目标**：不改 renderer，不改 FiMA，只加 frame weight 支持。

要求：

1. dataset / training batch 支持 `frame_weight`；
2. loss 支持 per-frame weighting；
3. synthetic corruption 生成 oracle weights；
4. 运行 corrupted_uniform vs oracle_weight。

不要做：

- FiMA uncertainty head；
- CUDA footprint inflation；
- MC marginalization。

## 37. Task 2：Pose corruption generator

新增脚本：

```text
scripts/methods/ours/pose_uncertainty_prior/generate_pose_prior_corruption.py
```

支持生成：

1. random outlier；
2. translation bias；
3. rotation bias；
4. smooth drift；
5. sinusoidal drift。

## 38. Task 3：FiMA uncertainty head

修改 FiMA：

1. 输出 mean + logvar；
2. 实现 heteroscedastic NLL；
3. 保存 frame uncertainty；
4. 做 calibration evaluation。

## 39. Task 4：Uncertainty calibration script

新增：

```text
scripts/eval/evaluate_pose_uncertainty_calibration.py
```

输出：

```text
Spearman correlation
Pearson correlation
AUROC bad-pose detection
calibration bins
plots if possible
```

## 40. Task 5：Predicted uncertainty weighting

让 GS training 读取 FiMA uncertainty 文件，并将其映射到 frame loss weight。

## 41. Task 6：Footprint inflation forward

修改 `us_focus` CUDA：

1. pass current slice uncertainty；
2. inflate $\Sigma_{2d}$；
3. mapper 同步使用 inflated footprint；
4. uncertainty=0 时与原输出 allclose。

## 42. Task 7：MC marginalization

Python training loop 实现 optional MC pose sampling。

---

# Part J：论文叙事建议

## 43. 不要再强调 GS pose optimization

不要写：

> GS optimizes pose.

应该写：

> GS consumes learned pose priors probabilistically.

或：

> Pose uncertainty is propagated into differentiable Gaussian reconstruction.

## 44. 方法贡献建议

### Contribution 1

提出 `us_focus` renderer：

> A probe-conditioned ultrasound Gaussian renderer with focus-aware additive accumulation.

### Contribution 2

提出 pose uncertainty-aware reconstruction：

> A probabilistic reconstruction objective that treats relocalization output as uncertain pose prior rather than deterministic camera pose.

### Contribution 3

提出 uncertainty coupling：

> Frame-level uncertainty weighting and pose-uncertainty-inflated Gaussian footprint reduce the effect of unreliable pose priors.

### Contribution 4

系统性分析：

> Direct GS pose optimization is ill-posed in ultrasound because Gaussian attributes can absorb pose error; uncertainty-aware consumption is more reliable than unconstrained pose refinement.

这个负结果可以作为一个 strong insight，而不是失败。

---

# Part K：关键风险

## 45. 风险 1：uncertainty 不校准

如果 FiMA 输出的 uncertainty 和真实 error 不相关，predicted uncertainty weighting 不会有效。

应对：

1. 先用 oracle uncertainty 验证上界；
2. 用 calibration loss；
3. 用 proxy uncertainty；
4. 用 ensemble / MC dropout 作为对照。

## 46. 风险 2：downweight 后训练帧太少

如果 bad frames 被低权重，GS 可能 coverage 不够。

应对：

1. 设置 `min_weight=0.1`；
2. 不要完全丢帧；
3. density control 保持；
4. frame sampling 时保证覆盖。

## 47. 风险 3：footprint inflation 让图像变糊

过大的 inflation 会降低 PSNR。

应对：

1. 限制 `inflation_max_mm`；
2. 只对 high uncertainty frames 开启；
3. gamma 调低；
4. 与 frame weighting 组合。

## 48. 风险 4：MC marginalization 太慢

应对：

1. 只作为 ablation；
2. 只对 high uncertainty frames；
3. K=2；
4. 小分辨率 / 短训练。

---

# Part L：最小可行实验路线

建议下一步严格按这个顺序：

## Stage A：Oracle corruption proof

1. 生成 corrupted pose prior；
2. uniform corrupted GS；
3. oracle weighted GS；
4. oracle filtered GS；
5. 看是否能恢复 PSNR/LPIPS。

如果这一步失败，说明 uncertainty-aware reconstruction 方向需要重想。

## Stage B：Predicted uncertainty proof

1. 训练 FiMA uncertainty head；
2. 验证 uncertainty-error correlation；
3. 用 predicted uncertainty 做 GS weighting；
4. 和 oracle 上界比较。

## Stage C：Footprint inflation

1. uncertainty=0 allclose；
2. corrupted prior + inflation；
3. inflation vs weighting；
4. weighting + inflation。

## Stage D：Bayesian MC ablation

1. K=2；
2. 小规模；
3. 和 footprint inflation 比较。

---

## 49. 最终一句话

本方向的核心不是让 GS 去“纠正” pose，而是让 GS 在重建时知道 pose 是不确定的：

\[
	extbf{Relocalization gives a distribution, not a camera.}
\]

这可以把当前弱串联升级为真正的联合框架：

\[
	ext{Bayesian pose prior}
+
	ext{uncertainty-aware ultrasound Gaussian reconstruction}
\]

它更符合已有实验事实，也比继续追求 GS pose refinement 更有期刊潜力。
