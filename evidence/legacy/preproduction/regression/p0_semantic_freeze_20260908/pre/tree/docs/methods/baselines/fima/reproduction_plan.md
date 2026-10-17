# Method（方法）

## 0）逐小节映射（对应论文 MICCAI 2024 原文结构）

- `2 Methods`（整体流水线 Fig.2）→ 本笔记 `1) FiMA 总体问题设定与输入输出`
- `2.1 ReMamba with Multi-directional State Space Model`（Fig.3）→ 本笔记 `2) ReMamba：多尺度 + 多方向 SSM 编码`
- `2.2 Adaptive Fusion Strategy`（Fig.4）→ 本笔记 `3) 自适应多 IMU 融合`
- `2.3 Online Alignment Strategy`（Fig.2(C)）→ 本笔记 `4) 在线对齐与损失汇总`

论文方法缩写为 **FiMA**（Fine-grained Context and Multi-modal Alignment）：利用状态空间模型（SSM）在长序列上的依赖管理能力，在**细粒度特征**上联合建模时空信息，并结合**多 IMU** 模态与**测试阶段在线对齐**提升自由扫描三维超声重建中的相邻帧相对变换估计精度。

---

## 1）FiMA 总体问题设定与输入输出

### 任务定义

给定长度为 $N$ 的 B-mode 扫描序列 $\mathcal{I}=\{I_i\mid i=1,2,\ldots,N\}$，以及对应的 **$M$ 个 IMU** 在相邻帧间的测量 $\mathcal{U}=\{U_i\mid i=1,2,\ldots,N-1\}$，FiMA 估计相邻帧之间的变换参数：

$$
\boldsymbol{\theta}=\{\boldsymbol{\theta}_i\mid i=1,2,\ldots,N-1\},\quad
\boldsymbol{\theta}_i=\big(\mathbf{t}_i,\boldsymbol{\phi}_i\big),
$$

其中：

- $\mathbf{t}_i=(t_x,t_y,t_z)_i$：三轴平移；
- $\boldsymbol{\phi}_i=(\phi_x,\phi_y,\phi_z)_i$：三轴旋转角（Euler angles）。

每个时刻的 IMU 集合写为 $U_i=\{U_i^j\mid j=1,\ldots,M\}$，且单个 IMU $j$ 提供：

- 三轴角度 $\boldsymbol{\Phi}_i^j=(\Phi_x,\Phi_y,\Phi_z)_i^j$；
- 三轴加速度 $\mathbf{A}_i^j=(A_x,A_y,A_z)_i^j$。

文中说明：对 $\boldsymbol{\Phi}_i$ 与 $\mathbf{A}_i$ 的**预处理流程**遵循文献 [7]（MoNet 相关工作）中的做法。

### 三大模块（与 Fig.2 对应）

1. **ReMamba（A）**：对图像序列做多尺度时空编码；
2. **Adaptive Fusion（B）**：将图像时空特征与多 IMU 时序信息自适应融合；
3. **Online Alignment（C）**：在测试阶段用 IMU 衍生伪标签做多模态特征对齐。

---

## 2）ReMamba：多尺度 + 多方向 SSM 编码

### 2.1 SSM / Mamba 预备知识与动机

连续时间 SSM 将输入序列 $x(t)$ 映射到输出 $y(t)$，隐含状态 $h(t)$ 满足：

$$
h'(t)=Ah(t)+Bx(t),\quad y(t)=Ch(t) \tag{1}
$$

其中 $A,B,C$ 为可学习矩阵。连续形式对离散图像序列不够直接；**Mamba** 将 SSM 离散化，并使 SSM 参数**随输入变化**，从而具备选择性传播/遗忘能力，有利于把长程上下文压缩到小状态维度。

**设计动机**：既往 CNN/RNN 类方法多在**粗粒度**特征上抽取时间依赖，时空耦合不足；而在**细粒度**特征上建模长程依赖难度大。ReMamba 通过 **SSM-F（细粒度）+ SSM-C（粗粒度）** 与 **多方向扫描（row-major / col-major / forward / backward）** 统一看待时空序列。

### 2.2 整体骨架

ReMamba 包含：**stem**、若干 **residual block**、以及多个 **ReMamba block**（内含多方向 SSM）。每个 ReMamba block 接收**两个不同尺度**的特征，输出**同形状**的两路特征（与多尺度设计一致）。

### 2.3 ReMamba Block 内的 SSM-F（细粒度）

目标：在多尺度特征上捕获**多样化的细粒度时空依赖**。

数据流要点（对应 Fig.3(A)(B)）：

1. 输入来自前一 residual block；经 **LayerNorm**；
2. **主分支**与**门控分支**分别线性投影到高维，得到 $x$ 与 $z$；
3. 对 $x$：依次经过 **1D 卷积** → **SiLU** → **SSM**；
   - 在 **行优先（row-major）** 方向扫描一次（Fig.3(A)）；
   - 在 **列优先（col-major）** 方向再扫描一次（Fig.3(B)）；
4. 对 $z$：经 SiLU 后，与两个方向的 SSM 输出分别**逐元素相乘**，两路乘积**相加**，再线性投影回原维度；
5. 与输入残差相加，得到**细粒度时序信息**输出。

**直觉**：不同 flatten/扫描方向对应不同“上下文邻域”，双向（行/列）有助于同时捕获沿帧内空间结构演化与序列方向上的依赖。

### 2.4 ReMamba Block 内的 SSM-C（粗粒度）

目标：感知**整体运动模式**，互补 SSM-F 的局部细粒度依赖。

数据流要点（对应 Fig.3(C)(D)）：

1. 对前一 residual block 输出做 **average pooling**，并与**前一 SSM-C 输出**相加作为 SSM-C 输入；
2. **无需 flatten**（与 SSM-F 不同）；
3. 使用 **forward（Fig.3(C)）** 与 **backward（Fig.3(D)）** 两个方向的 SSM；
4. 得到**粗粒度时序信息**输出。

---

## 3）自适应多 IMU 融合（Fusion Module）

### 3.1 动机

多 IMU 提供图像之外的刚体/运动状态信息，但：

- 扫描运动复杂多样；
- 各 IMU 特性、安装状态与噪声不同，**时序提示未必一致**。

因此需要**自适应**判断：在每个时刻，各 IMU 提供的时序信息是否可信、应占多大权重。

### 3.2 融合机制（Fig.4）

核心假设：图像侧抽取的时空特征与 IMU 侧时序特征**并非独立**，存在相关性；网络应结合图像时空上下文，评估每个 IMU 当前是否给出“合理”的运动线索。

步骤概要：

1. 将每个 IMU 的 **加速度/角度** 映射到与图像特征对齐的两个空间：
   - 其一表征该 IMU 的 **运动特性（motion characteristics）**；
   - 其二表征 **细节时序信息（detailed temporal information）**；
2. 计算图像时空特征与各 IMU **运动特性表征**之间的 **affinity（亲和度）**；
3. 用该亲和度作为权重，对各路 IMU 的**细节时序信息**进行加权融合；
4. 对加权后的 IMU 时序表示，使用 **两个 multi-head attention**：
   - 其一融合 **图像时空特征**；
   - 其二融合 **IMU 时序特征**；
   得到 **多模态融合特征**；
5. 再接 **带 skip 的 SSM-C** 与线性层，解码预测变换 $\hat{\boldsymbol{\theta}}$。

---

## 4）在线对齐与损失汇总

### 4.1 监督重建损失 $L_{\mathrm{sup}}$（训练阶段）

训练时使用 **MAE + Pearson 相关**形式的复合损失：

$$
L_{\mathrm{sup}}=\lVert \hat{\boldsymbol{\theta}}-\boldsymbol{\theta}\rVert_1+\left(1-\frac{\mathrm{Cov}(\hat{\boldsymbol{\theta}},\boldsymbol{\theta})}{\sigma(\hat{\boldsymbol{\theta}})\,\sigma(\boldsymbol{\theta})}\right) \tag{2}
$$

其中 $\boldsymbol{\theta}$ 为真值变换参数，$\mathrm{Cov}$ 为协方差，$\sigma$ 为标准差。第二项鼓励预测与真值在统计意义上线性相关。

### 4.2 多模态对齐损失 $L_{\mathrm{align}}$（互信息下界）

令 $F_I$ 为 ReMamba 输出的图像特征，$F_A$ 为 Fusion Module 中与 **IMU 加速度**相关的加权时序特征。对齐阶段最大化二者互信息；直接优化互信息困难，采用文献 [10]（对比预测编码 CPC）中的**下界**，对长度为 $N$ 的序列：

$$
L_{\mathrm{align}}=
-\frac{1}{N-1}\sum_{i=1}^{N-1}
\log\frac{\exp(F_{I,i}\cdot F_{A,i}/\tau)}{\sum_{j=1}^{N-1}\exp(F_{I,i}\cdot F_{A,j}/\tau)} \tag{3}
$$

其中 $\tau$ 为温度系数，文中取 **$\tau=0.1$**。

### 4.3 基于 IMU 角度的先验相关损失 $L_{\mathrm{prior}}$

IMU 在多数情况下角度测量较可靠。文中使用 Fusion Module 中与 **IMU 角度**对应的融合权重作为先验，对多个 IMU 角度测量做加权平均得到 $\bar{\boldsymbol{\Phi}}$，再约束估计欧拉角 $\hat{\boldsymbol{\phi}}$：

$$
L_{\mathrm{prior}}=
1-\frac{\mathrm{Cov}(\hat{\boldsymbol{\phi}},\bar{\boldsymbol{\Phi}})}{\sigma(\hat{\boldsymbol{\phi}})\,\sigma(\bar{\boldsymbol{\Phi}})} \tag{4}
$$

### 4.4 总目标与测试阶段“在线对齐”配置要点

- 文中表述为：使用上述损失函数的**和**优化 FiMA。
- **Online alignment** 强调在 **测试阶段**仍通过 IMU 伪标签与特征对齐机制稳定泛化；实验实现部分给出测试阶段额外优化配置：**60 个 epoch、学习率 $2\times10^{-6}$**（见 Experiment 笔记）。

---

# Experiment（实验设置）

## 0）逐小节映射（对应论文结构）

- `3 Experiments` → `Materials and Implementation` → 本笔记 `1) 数据与伦理`
- 同一小节 → `Materials and Implementation`（训练/测试超参）→ 本笔记 `2) 训练与测试协议`
- `Quantitative and Qualitative Analysis` → 本笔记 `3) 指标与对比方法`
- Table 1 / Fig.5 → 本笔记 `4) 主要定量结论与定性说明`

---

## 1）数据与伦理

### 1.1 数据集构建（沿用文献 [8] 的设置）

两类采集部位、共 **50 名志愿者**：

| 数据集 | 扫描数量 | 扫描手法 | 平均序列长度（文中报告） |
|--------|----------|----------|---------------------------|
| **Arm（手臂血管）** | 583 条 | linear、curved、loop、sector 等多种 | **386.39 mm** |
| **Carotid（颈动脉）** | 432 条 | linear、loop、sector | **241.25 mm** |

### 1.2 图像规格与 IMU 布置

- 图像尺寸：**$248\times260$** 像素；
- 像素间距：**$0.15\times0.15\ \mathrm{mm}^2$**；
- 多个 IMU 的**空间布置**与文中 **Fig.1** 一致（与采集硬件一体化配置）。

### 1.3 划分方式与增强

- **Arm**：按志愿者划分 train / val / test = **375 / 104 / 104** 条扫描；
- **Carotid**：**276 / 78 / 78** 条扫描；
- 每条扫描做 **随机增强**，细节遵循文献 **[8]**（OSCNet 数据集与增强协议）。

### 1.4 伦理

数据采集与应用经 **本地 IRB** 批准，符合伦理规范。

---

## 2）训练与测试协议（Materials and Implementation）

### 2.1 优化器与学习率（训练阶段）

- 优化器：**Adam**；
- **Epochs**：**200**；
- **Batch size**：**1**；
- 初始学习率：**$2\times10^{-4}$**；
- 防过拟合：**每 30 个 epoch 学习率减半**。

### 2.2 测试阶段在线对齐相关超参

- **迭代 epoch**：**60**；
- **学习率**：**$2\times10^{-6}$**。

（与 Method 中 Online Alignment 对应：测试时仍以较小学习率做若干轮适配。）

---

## 3）指标与对比方法

### 3.1 评价指标（沿用文献 [8]）

全部指标均为 **越小越好**（文中 Table 1 用 ↓ 标示）：

| 缩写 | 含义 |
|------|------|
| **FDR** | Final drift rate（终点漂移率，%） |
| **ADR** | Average drift rate（平均漂移率，%） |
| **MD** | Maximum drift（最大漂移，mm） |
| **SD** | Sum of drift（漂移总和，mm） |
| **HD** | Symmetric Hausdorff distance（对称 Hausdorff 距离，mm） |
| **MEA** | Mean error of angle（角度平均误差，deg） |

### 3.2 对比方法

包含传统/深度学习基线与传感器配置差异：

- **CNN-OF*** [11]、**ResNet*** [3]、**DC2-Net*** [2]、**RecON*** [6]（*表示 **无需传感器**）；
- **MoNet** [7]（单 IMU）、**OSCNet** [8]（多 IMU + 在线自洽，强相关前置工作）。

### 3.3 消融命名（Table 1 表注）

- **ReM′ / ReM**：分别将主干中的块替换为 **Mamba block** vs **ReMamba block**；
- **F′ / F**：多 IMU 映射方式——**F′** 为加速度/角度直接映射到高维再拼接；**F** 为本文 **自适应融合**；
- **FiMA**：完整模型（ReMamba + 自适应融合 + 在线对齐）。

---

## 4）主要定量结论与定性说明

### 4.1 定量（Table 1 摘要）

- **ReMamba（ReM）** 在多项指标上优于单 IMU 的 **MoNet**，说明即便在本文的多 IMU设定以外对比意义上，**多方向 SSM 时空建模**仍带来增益。
- **ReM+F** 一致优于 **ReM+F′**，验证 **自适应融合** 相对简单拼接的有效性。
- **FiMA** 在 Arm / Carotid 上取得最优结果；相对 **OSCNet**，**FDR/ADR** 改善约为：
  - Arm：**25.57% / 21.96%**；
  - Carotid：**39.24% / 36.43%**。

### 4.2 定性（Fig.5）

展示代表性三维重建结果：探头轨迹、重建表面与分割结果显示 **FiMA** 与真值更吻合，血管等结构连续性更好，便于后续三维解剖分析。

---

## 5）复现 checklist（从论文可直接落地的配置项）

- [ ] 输入序列长度 $N$ 按数据集扫描长度变化；相邻帧监督维度与 $N-1$ 对齐。
- [ ] IMU 数量 $M$、布置与预处理遵循 [7]/硬件 Fig.1。
- [ ] 图像分辨率 **$248\times260$**，间距 **0.15 mm**。
- [ ] 划分：**Arm 375/104/104**，**Carotid 276/78/78**；增强遵循 [8]。
- [ ] 训练：**Adam**，**200 epochs**，**batch=1**，**lr=$2\times10^{-4}$**，**每 30 epoch 减半**。
- [ ] 对齐损失温度 **$\tau=0.1$**；测试阶段对齐：**60 epochs**，**lr=$2\times10^{-6}$**。
- [ ] 损失：**式 (2)(3)(4) 求和**（具体权重若开源代码有异，以官方实现为准）。

---

## 6）说明（关于 PDF 抽取与符号）

本文档 Method / Experiment 叙述依据 Yan 等 MICCAI 2024（LNCS 15007）正文 **Section 2–3** 与图表编号 **Fig.2–5、Table 1**；公式编号与原文一致为 **(1)–(4)**。若需像素级重建网络细节（如体素尺寸、插值 compounding），需结合引用文献 **[7][8]** 中的采集与重建管线补充——本短文侧重 FiMA 提出的 **SSM 时空编码 + 多 IMU 自适应融合 + 测试对齐** 三部分。

---

## 7）从说明书到代码：可实现性评估与待补信息（实现缺口清单）

下面回答「若只凭本说明书能否实现网络」：**可以搭出方法级别的管线草图，但无法在不参考 Fig.2–4、[7][8] 或开源实现的前提下，唯一还原作者同款结构与张量形状**。建议把下列条目逐项对照原图/附录/代码补进笔记。

### 7.1 已相对明确、可直接用于实现的条目

- **任务形式**：序列输入 → 相邻帧 $N-1$ 组变换 $\hat{\boldsymbol{\theta}}_i$（6 DoF：平移 + 欧拉角）。
- **损失组件**：式 (2) 监督项；式 (3) InfoNCE/CPC 型对齐（$\tau=0.1$）；式 (4) 角度 Pearson 相关形式先验。
- **训练超参**：Adam；200 epoch；batch=1；lr $2\times10^{-4}$；每 30 epoch 减半。
- **测试阶段微调**：60 epoch；lr $2\times10^{-6}$（需与 7.4 一起澄清「优化哪些参数」）。
- **数据规格与划分**：248×260；0.15 mm；Arm/Carotid 划分比例；增强指向 [8]。

### 7.2 ReMamba 侧含糊点（缺了就无法对齐论文实现）

| 缺口 | 为何关键 | 建议补全来源 |
|------|-----------|----------------|
| **Stem / residual 的具体拓扑** | 通道数、下采样次数、是否多尺度 FPN 式分支决定「两路不同尺度特征」的张量形状 | Fig.2(A)、若有附录表 |
| **「两个尺度」的定义** | 未写明是双分辨率分支、还是同一分辨率两路处理 | 原图 |
| **SSM-F 的 1D 序列构造** | 「三维图像序列特征展平为一维」：是 $(T,H,W)$ 沿何种顺序拼接、序列长度是否为 $T\!\cdot\!H\!\cdot\!W$ 或按帧分开多次扫描，会直接影响 Mamba 有效感受野 | Fig.3 + 实现 |
| **行优先 / 列优先作用在哪个网格** | 单帧特征图上的空间扫描，还是时空 flatten 后的伪 2D 网格 | Fig.3 |
| **1D Conv 的 kernel、padding、dilation** | 决定局部混合范围 | 代码或附录 |
| **Mamba/SSM 内部宽度** | $d_{\mathrm{model}}$、$d_{\mathrm{state}}$、expand、层数等短文未给 | 开源或对照 VMamba/Mamba 默认再调参 |
| **SSM-C 的「forward/backward」轴向** | 是沿时间 $T$、沿帧序号、还是别的维度 | Fig.3(C)(D) |
| **Block 堆叠深度** | residual block / ReMamba block 各多少个 | 原图或附录 |

### 7.3 Fusion 模块含糊点

| 缺口 | 为何关键 | 建议补全来源 |
|------|-----------|----------------|
| **IMU 映射两路的网络结构** | MLP 层数、隐藏维度、accel 与 angle 是否共享 | Fig.4 |
| **Affinity 的具体公式** | 点积、双线性、注意力打分；是否在 IMU 维做 softmax | Fig.4 |
| **两个 Multi-Head Attention 的角色** | 自注意力还是交叉注意力；Q/K/V 各来自哪一路特征；head 数与维度 | Fig.4 |
| **输出到 $\hat{\boldsymbol{\theta}}$ 的 SSM-C** | 与 ReMamba 内 SSM-C 是否同构、输入序列长度 | 原图 |
| **IMU 数量 $M$** | 正文未写死；依赖 Fig.1 / [8] 硬件 | Fig.1、[8] |

### 7.4 损失与训练/测试流程含糊点（最易实现错）

- **$L_{\mathrm{sup}}$ 与 $L_{\mathrm{align}}$、$L_{\mathrm{prior}}$ 是否同时在训练期反传**：正文写训练期用式 (2)，对齐写在测试语境；但又写「损失求和」——**是否训练期就加 (3)(4)、抑或仅在测试 60 epoch 微调**，说明书单独无法定论，**必须以开源或作者邮件/附录为准**。
- **多损失是否有权重系数**：若简单相加，各项量级差异可能导致训练不稳；文中未给 $\lambda$。
- **$F_{I,i}$、$F_{A,i}$ 的向量定义**：全局池化后的固定维向量、还是保留空间维再平均；维度 $d$ 未给。
- **式 (2)(4) 中 Pearson 项在 batch=1 时如何估计方差/协方差**：通常需在**序列维**或**通道分组**上统计，否则 batch=1 时标量不稳定；**具体沿哪一维聚合**未写明。
- **测试 60 epoch**：是冻结主干只训对齐头，还是全网络微调；优化器是否仍为 Adam；是否仍用 GT（若有）——短篇未写。

### 7.5 监督与坐标系（实现易错）

- **$\boldsymbol{\theta}$ 真值来源**：与 [8] 相同的外部追踪/标定流程；本笔记未展开。
- **欧拉角顺序与单位**（如 ZYX vs XYZ；度 vs 弧度）；平移单位是否与 0.15 mm 像素间距一致。
- **IMU 时间对齐**：与帧 $i\to i+1$ 的同步、插值、[7] 的预处理（滤波、重力去除、坐标变换）细节。

### 7.6 小结（直接回答你的两个问题）

1. **「拿到这份说明书能否实现这份网络？」**  
   **能实现「FiMA 思路」的可运行原型**（Mamba 式编码 + 多 IMU 融合 + CPC 对齐 + 测试微调），但**不能保证与论文 Table 1 完全一致的架构与性能**，除非补全上表中的形状、模块与损失调度细节（最好对照 **Fig.2–4** 与 **[7][8]**，优先找 **官方代码**）。

2. **「有没有含糊、需要补充的地方？」**  
   **有，且集中在：张量布局与多尺度定义、Fusion 中注意力与 affinity 细节、Mamba 超参、三种损失的启用阶段与权重、batch=1 时相关损失的统计方式、以及 $M$ 与坐标系/真值管线。**  

若你希望把某一节补成「可照着写 PyTorch 的模块规格」（例如固定通道表 + 序列长度定义），需要基于 Fig.2–4 逐项量化；可以把插图导出或标注关键维度后发我继续写在同一份 `.md` 里。
