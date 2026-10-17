本地源码路径：/home/boywang25/ultrasound-scene-reconstruction/third_parties/FreehandUSRecon

# Method

## 0) 逐小节映射（对应论文结构）

- `2 Data Materials`（监督信号定义）-> 本笔记 `1) 相对位姿监督与均值标签策略`
- `3 Ultrasound Volume Reconstruction` -> 本笔记 `2) DCL-Net：3D 上下文骨干网络`
- `3.1 Attention Module` -> 本笔记 `3) 自注意力模块（面向斑纹相关区域）`
- `3.2 Case-Wise Correlation Loss` -> 本笔记 `4) MSE + case-wise correlation 损失`

## 1) 相对位姿监督与均值标签策略（Sec.2–3）

论文面向 **sensorless freehand 3D TRUS 重建**：训练阶段用 **电磁跟踪（EM）** 提供每帧位姿，网络学习目标是从 **无跟踪** 的连续 2D 帧序列估计相邻帧之间的相对变换。

设相邻帧 $I_i$、$I_{i+1}$ 对应的齐次变换为 $M_i$、$M_{i+1}$，相对变换为：

$$
M'_i = M_{i+1} M_i^{-1}
$$

将 $M'_i$ 分解为 6-DoF：

$$
\theta_i = \{t_x, t_y, t_z, \alpha_x, \alpha_y, \alpha_z\}_i
$$

（平移单位为 **mm**，旋转单位为 **度**。）

网络输入为连续 **$N$** 帧（每帧空间尺寸 $H\times W$），堆叠成形状为 **$N \times H \times W$** 的“伪体积”。相邻帧之间理论上可对应 $\{\theta_i \mid i=1,\ldots,N-1\}$，但论文 **不直接用逐帧 $\theta_i$ 作为回归标签**，而是用均值向量平滑小幅抖动噪声：

$$
\bar{\theta} = \frac{1}{N-1}\sum_{i=1}^{N-1}\theta_i \tag{1}
$$

**测试阶段**：沿视频以窗口长度 $N$ **滑动**；两相邻帧之间的运动取 **所有覆盖该边的 batch 上估计均值运动的平均**（文中表述为对 batches 平均）。

## 2) DCL-Net：3D 上下文骨干网络（Sec.3）

整体建立在 **3D ResNeXt** 之上：包含 **3D 残差块**、skip connection，以及 **多路径（cardinality）** 以提取判别特征。

**为何用 3D 卷积而非 2D**：在本任务里 channel 维对应 **时间/帧序**；3D 卷积更有利于刻画 **连续帧之间特征的细微位移**，从而把 **斑纹相关的对应关系** 与 **相对位姿** 联系起来。

## 3) 自注意力模块（Sec.3.1）

在最后一个残差块输出的特征图上接入 **self-attention**，生成 **attention map**，对 **斑纹丰富、利于帧间相关的区域** 赋予更高权重；动机来自传统 decorrelation 思路——关注斑纹信息有助于提升重建/位姿估计。

## 4) MSE + case-wise correlation 损失（Sec.3.2）

**总损失** = **MSE（对所有自由度回归 $\bar{\theta}$）** + **case-wise correlation loss**。

动机：单独 MSE 容易使估计 **过度平滑**，网络更易 **记住“平均扫查风格/平均轨迹”**，对真实速度/朝向变化不敏感（文中引用相关讨论）。

**Case-wise correlation loss** 基于 Pearson 相关：从一个 TRUS 视频中随机采样 **$K$** 段、每段 **$N$** 帧；对 **每个自由度** 分别计算估计均值与 GT 均值的相关性，并聚合为：

$$
\mathcal{L}_{\text{corr}} = 1 - \frac{1}{6}\sum_{d=1}^{6}
\frac{\mathrm{Cov}\left(\bar{\theta}^{\mathrm{GT}}_{d},\bar{\theta}^{\mathrm{Out}}_{d}\right)}
{\sigma\left(\bar{\theta}^{\mathrm{GT}}_{d}\right)\sigma\left(\bar{\theta}^{\mathrm{Out}}_{d}\right)} \tag{2}
$$

其中 $\mathrm{Cov}$ 为协方差，$\sigma$ 为标准差。

---

# Experiment

## 1) 逐小节映射（对应论文结构）

- `2 Data Materials` -> 本笔记 `2) 数据集、采集协议与 GT 构造`
- `4 Experiments and Results`（开头段落）-> 本笔记 `3) 训练实现细节与推理耗时`
- `4 Experiments and Results`（评估指标段）-> 本笔记 `4) 评估指标`
- `4.1 Parameter Setting` -> 本笔记 `5) 输入帧数 $N$ 消融与注意力可视化`
- `4.2 Reconstruction Performance and Discussions` -> 本笔记 `6) 主对比、相关损失消融与体积可视化`

## 2) 数据集、采集协议与 GT 构造（Sec.2 + Sec.4）

### 规模与划分

- **640** 条 TRUS 视频（**640 例患者**，各不相同）
- 划分：**500 / 70 / 70** = 训练 / 验证 / 测试

### 采集与标注来源

- 数据来自 **NIH IRB 批准的临床试验**
- 设备：**Philips iU22**
- 姿态 GT：**EM tracking**（每帧对应位置与朝向，转齐次矩阵 $M=[R,T;0,1]$）

### 临床扫查协议（文中描述）

- **端射（end-firing）经直肠探头**
- 医师沿前列腺 **从 base 到 apex** 进行相对稳定轴向扫查（sweep）

## 3) 训练实现细节与推理耗时（Sec.4）

### 优化与预算

- **Adam**，初始学习率 **$5\times 10^{-5}$**
- **每 5 个 epoch** 学习率乘以 **0.9**
- 训练 **300 epochs**
- **batch size $K = 20$**（文中符号：每个 batch 采样 **$K$** 段视频片段）
- 框架：**PyTorch**

### 预处理

- 前列腺 ROI 在每帧中占比相对较小：在不超过成像视野前提下 **crop**，再 **resize 到 $224\times 224$**（适配 ResNeXt 设计）

### 默认输入帧数与耗时（文中报告）

- 训练/测试默认取 **$N=5$** 帧作为片段输入时：
  - 训练总时长约 **4 小时**
  - 测试：对 **100 帧** 视频估计全部变换矩阵约 **2.58 s**

## 4) 评估指标（Sec.4）

1. **Distance error（mm）**：对整个视频中 **所有对应帧角点** 的平均距离误差；反映速度与朝向变化在全局轨迹上的差异。
2. **Final drift（mm）**：用 EM tracking 与 DCL-Net 估计运动分别变换片段末端帧中心点，两者中心点距离（定义参照 Prevost et al.）。

## 5) 输入帧数 $N$ 消融与注意力可视化（Sec.4.1）

### $N$ 的选择

- 改变连续输入帧数 $N$，总体误差呈 **先降后升**
- **$N=5$ 或 $N=6$** 接近最优且性能相近
- **paired t-test**：$N=5$ **显著优于** $N=2$（$p<0.05$）
- 解释：更长上下文带来更稳定的轨迹估计

### 注意力可视化（Fig.3）

- 对比 US 图像、相邻帧 **speckle correlation map**（参照 Chang et al.）与 **绕 Y 轴旋转 $\alpha_y$** 的 attention map
- 文中结论：attention 在 **correlation 较亮（更大 elevational separation）** 的区域更强，符合 TRUS 扫查协议下 probe tip 附近运动更小等现象

## 6) 主对比、相关损失消融与体积可视化（Sec.4.2）

### 对比方法（Table 1）

- **Linear motion**：用训练集 **平均运动向量** 固定施加到所有测试序列
- **Decorrelation**：Chang et al. 的斑纹 decorrelation 流程
- **2D CNN**：Prevost et al.（两帧 + optical flow 栈输入的 CNN 回归）
- **3D CNN (NS2)**：**Vanilla ResNeXt**，但输入 **仅两帧**（作为“非上下文”3D 卷积对照）

### 主结果（Table 1，文中汇总）

DCL-Net 在 **distance error** 与 **final drift** 的 Min/Median/Max/Average 上 **整体优于** 上述基线；文中报告 **paired t-test** 显著性 **$p<0.05$**。

（文中同时强调：即便平均 distance error 仍在 **~10 mm** 量级，这属于 **真实临床数据** 上的最佳报告水平之一，任务本身难度大。）

### Case-wise correlation loss 消融（Fig.4）

- **仅 MSE**：预测曲线更接近“均值/线性运动”，对 GT 速度变化不敏感；测试集相关系数均值约 **$0.09 \pm 0.03$**
- **MSE + correlation loss**：对 GT 变化更敏感；相关系数均值约 **$0.21 \pm 0.09$**（paired t-test，$\alpha=0.05$ 显著更好）

### 体积重建可视化（Fig.5）

展示 **好/中/差** 四类测试案例；与 **2D-CNN** 相比，DCL-Net 轨迹估计 **更平滑**，对探头速度变化更敏感、振动噪声更少。

## 7) 给你后续复现用的 checklist（浓缩）

- **数据**：640 videos；500/70/70；EM GT；crop + **224×224**
- **网络**：3D ResNeXt + **self-attention**；标签用 **$\bar{\theta}$（Eq.1）**
- **损失**：**MSE + $\mathcal{L}_{\text{corr}}$（Eq.2）**；采样 **$K$** 段、每段 **$N$** 帧用于 correlation 项
- **训练**：Adam **$5\times 10^{-5}$**；**×0.9 / 5 epochs**；**300 epochs**；**$K=20$**
- **默认 $N$**：**5**（与“100 帧 2.58s”报告一致）
- **评测**：distance error + final drift；主对比含 Linear / Decorrelation / 2D CNN / 3D CNN(NS2)
