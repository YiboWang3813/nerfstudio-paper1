# 我的理解

就是搞了一个 $h(z_c)$ 来调制原来的高斯效应。因为 $h(z_c)$ 是和物理分辨率以及像素在相机坐标系下的$z$ 坐标有关的。引入了它，就可以卖这是物理驱动的。Focus 也体现在这个地方因为像素的空间位置也影响了高斯的加权贡献，那对于不同的切片位置（本质上就是不同的 $z_c$）就会产生不同的效应。

总之就是像素（某个成像平面，就是一个又一个slice）的 $z_c$ 影响了像素-高斯权重贡献的公式，不管是衰减了不透明度（近似实现了elevatioanl的截断筛选），还是把 $\Sigma_c^{-1}$ 变成 $\Sigma_e^{-1}$ 以及 $\mu_c$ 变成 $\mu_e$。

补充理解：
$slice_i$ 结合 $pose_i$ 可以产生唯一的成像平面 $z_c^i$ 。如果你站在全局角度上看，不同的slice就是不同的$z_c^i$ 。如果你站在 $slice_i$ 的角度上看，在它内部 $z_c$ 都是0.
成像平面是死的，高斯是活的，高斯的世界坐标要先通过投影转换到相机坐标系中。这里的投影矩阵是从 $pose_i$ 来的，所以这就天然地产生了对于不同的成像平面，相同的高斯也会变得不同。

# Method（第 4 节）

GaussianPile 通过**物理可知的聚焦（focus）模型**，把 3D Gaussian Splatting（3DGS）与基于切片的体数据重建统一起来（见图 2）。本节先介绍聚焦感知物理模型（4.1），再给出可微 CUDA 实现的前向渲染（4.2），最后说明优化与压缩管线（4.3）。

---

## 4.1 聚焦感知建模（Focus-aware Modeling）

核心思想是把**聚焦成像的物理模型**嵌入渲染过程。分析在**相机坐标系**下进行，相关量用下标 $c$ 区分。

假设成像系统的点扩散函数（PSF）$\mathrm{psf}(\mathbf{x}_c)$ 为**空间不变**的三维高斯：

$$
\mathrm{psf}(\mathbf{x}_c) \propto \exp\left(-\frac{1}{2}\left(\frac{x_c^2}{\sigma_x^2}+\frac{y_c^2}{\sigma_y^2}+\frac{z_c^2}{\sigma_z^2}\right)\right),
$$

其中 $\sigma_x,\sigma_y,\sigma_z$ 分别对应系统在 $x,y,z$ 方向的空间分辨率。**$\sigma_z$ 尤其重要**：它反映系统在**轴向（elevational）**方向的聚焦能力。

据此定义**灵敏度图（sensitivity map）** $h$ 为系统沿轴向的**反转冲激响应**：

$$
h(-z_c)=\exp\left(-\frac{z_c^2}{2\sigma_z^2}\right).
$$

该定义使高斯原语在后续推导中仍保持高斯形式，从而**全程可微**且便于高效计算。

对实际系统，$\sigma_z$ 可由理论/数值估计（见补充材料）、或实验标定得到。若难以获得，可假设扫描步长 $\delta_z$ 满足 Nyquist，此时取 **$\sigma_z \approx \delta_z$**；文中消融（表 4）表明该选择在 2D 与 3D 指标上均较优。

---

## 4.2 高斯堆叠（Gaussian Piling）

对单个 3D 高斯原语 $g_c \sim \mathcal{N}(\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c)$，渲染强度 $I$ 来自原语与灵敏度图 $h(z)$ 的卷积，并在成像平面（$z_c=0$）处取值：

$$
I(x_c,y_c)=\left[h(-z_c)*g_c(\mathbf{x}_c)\right]_{z_c=0}.
$$

写成积分形式即对轴向积分；完整展开见补充材料。为高效实现，将上述过程**现象学地**分解为以下步骤。

### 轴向重参数化（Axial re-parameterization）

将**有限切片厚度**编码为：在相机坐标下向逆协方差注入轴向分辨率项：

$$
\boldsymbol{\Sigma}_e^{-1}=\boldsymbol{\Sigma}_c^{-1}+\frac{\mathbf{e}_3\mathbf{e}_3^\top}{\sigma_z^2},\quad \boldsymbol{\mu}_e=\boldsymbol{\Sigma}_e\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c,
$$

其中 $\mathbf{e}_3=[0,0,1]^\top$。该操作**收缩轴向支撑**而不破坏横向结构，得到有效高斯 $f\sim\mathcal{N}(\boldsymbol{\mu}_e,\boldsymbol{\Sigma}_e)$，称为 **Focus Gaussian**。$\sigma_z$ 在此起到**调和 2D 切片保真与 3D 体一致性**的作用。

### 不透明度调制（Opacity modulation）

为抑制离焦原语、稳定薄切片重建，根据轴向注入引起的**马氏距离变化**调制不透明度：

$$
\mathrm{opacity}_r=\exp\left(-\frac{1}{2}\left(\boldsymbol{\mu}_c^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c-\boldsymbol{\mu}_e^\top\boldsymbol{\Sigma}_e^{-1}\boldsymbol{\mu}_e\right)\right).
$$

物理含义：离切片焦区较远的高斯受到更强的透明衰减，从而**压低跨切片的“鬼影”/漂浮伪影**。随后剔除低于小阈值的不透明度以减负。

### 屏幕空间投影（Screen-space projection）

对 $(x_c,y_c)$ 平面取 Focus Gaussian 协方差 $\boldsymbol{\Sigma}_e$ 的**边缘分布**，得到成像平面上的 2D 高斯参数：

$$
\boldsymbol{\Sigma}_{2d}=\begin{bmatrix}\Sigma_e[0,0]&\Sigma_e[1,0]\\\Sigma_e[1,0]&\Sigma_e[1,1]\end{bmatrix},\quad \boldsymbol{\mu}_{2d}=[\mu_e[0],\mu_e[1]]^\top.
$$

最终渲染强度中的 $\tilde{\alpha}$ 用积分归一化系数缩放：

$$
\tilde{\alpha}=\alpha\cdot \mathrm{opacity}_r/\sqrt{\det(\boldsymbol{\Sigma}_{2d})},
$$

**行列式项**使亮度随 footprint 大小自适应，在椭圆伸缩时保持总能量一致。

### 加性光栅化（Additive rasterization）

切片成像中，像素强度来自沿投影线对原语不透明度的积分，贡献**自然叠加而无遮挡**。因此采用**加性累积**：

$$
I(p)=\sum_{i\in\mathcal{N}(p)}\tilde{\alpha}_i\exp\left(-\frac{1}{2}\mathbf{d}_i^\top\boldsymbol{\Sigma}_{2d,i}^{-1}\mathbf{d}_i\right),
$$

其中 $\mathcal{N}(p)$ 为与像素 $p$ 重叠的高斯集合，$\mathbf{d}_i$ 为从 2D 中心 $\boldsymbol{\mu}_{2d}$ 指向 $p$ 的向量。梯度沿加性结构回传：$\partial I/\partial\tilde{\alpha}_i=\exp(\cdot)$，便于 CUDA 加速优化。

### 设计动机小结

1. **3D–2D 一致性**：体素强度与 3D 高斯协方差耦合；同一高斯被多张切片观测时共享协方差，**约束 3D 连续性**，避免标准 3DGS 中“2D 看似合理、3D 坍塌”的体不一致（文中有与图 5 对照）。
2. **聚焦模型下的切片**：渲染切片可视为由 $\boldsymbol{\Sigma}_e$ 塑造的有效 3D 高斯混合的**边缘分布**。这种以协方差驱动的强度表示**去掉**标准 3DGS 中为拟合 BRDF 而引入的**球谐（SH）系数**，约减少每原语 **40%** 参数量，并改善几何保真。

---

## 4.3 优化（Optimization）

训练与保存流程总体遵循标准 3DGS 循环，但有几处关键改动（优化过程见图 3）。

### 反向传播

在 CUDA 中实现聚焦管线的**完整梯度**。给定像素上的上游梯度，链式法则依次穿过：(i) 屏幕空间投影到 $(\boldsymbol{\mu}_e,\boldsymbol{\Sigma}_e)$；(ii) 轴向重参数化到相机空间 $(\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c)$；(iii) 相机变换到世界空间可学习参数 $(\boldsymbol{\mu},\mathbf{s},\mathbf{q},\alpha)$。对 $\tilde{\alpha}=\alpha\cdot\mathrm{opacity}_r/\sqrt{\det(\boldsymbol{\Sigma}_{2d})}$，**行列式项**使 $\alpha$ 与协方差的梯度耦合；完整推导见补充材料。两阶段均在 CUDA 上加速，保持实时性。

### 切片采样

每轮迭代**随机选一个虚拟相机（即一张成像切片）**渲染，以保证对整个 3D 图像栈的覆盖。

### 损失函数

用光度损失监督渲染的切片投影：**L1** $\mathcal{L}_1$ 与 **D-SSIM** $\mathcal{L}_{\mathrm{ssim}}$ 的加权和：

$$
\mathcal{L}=\mathcal{L}_1+\lambda\mathcal{L}_{\mathrm{ssim}},
$$

$\lambda$ 平衡像素级与感知保真。

### 剪枝与致密化（Prune and densification）

训练中采用基于** tile 内聚焦半径**与**高斯不透明度分数**的致密化–剪枝策略，逐步细化点云分布，保证结构覆盖并抑制轴向过度膨胀。收敛后得到可微点云表示，在重建输入切片的同时沿扫描轴改善表观分辨率；同一套 CUDA 核适用于文中所测多种切片成像模态（如 ABUS、光片显微镜等）。

### 量化与压缩

相对体素栅格，GaussianPile 用稀疏 Focus Gaussian 表示场景，参数**结构化且空间相关**。据此设计**定制量化 + 熵编码**：先用 **Morton（Z 序）**排序以暴露局部相关性；位置归一化到场景包围盒后按轴量化（典型每轴 **14 bit**）；不透明度映射到 **[0,1]** 的 **12 bit**；尺度先取对数再 **12 bit** 量化以保持乘性精度；四元数限制在正半球后每分量 **12 bit**。各属性流做**差分编码**并用 **LZMA** 熵压缩。该聚焦感知编码利用各向异性稀疏与参数平滑变化，相对体素存储约 **16×** 压缩，同时保持 2D 渲染保真与体一致性。

### 可微体素化（Differentiable Voxelization）

为评估 3D 重建质量并支持体可视化，集成可微体素化器 $\mathcal{V}$：由世界坐标 3D 高斯集合 $\mathcal{G}_3=\{g_i^3\}$ 构造强度体 $\mathbf{V}\in\mathbb{R}^{X\times Y\times Z}$，$\mathbf{V}=\mathcal{V}(\mathcal{G}_3)$。体素化将空间划分为均匀 **$8^3$** tile，按 tile 剔除高斯，仅保留 $3\sigma$ 椭球与 tile 相交者；tile 内体素强度为**世界坐标原高斯**（4.2 节重参数化之前）的贡献聚合：

$$
\sigma(\mathbf{x})=\sum_{i=1}^{M} g_i^3(\mathbf{x};\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i).
$$

前向与反向均在 CUDA 上加速，查询可达 **>100 FPS**，便于 3D 结构评估及训练期可选的体正则。

---

*说明：公式编号与原文一致；部分排版在 PDF 文本提取中可能缺字，以上按论文语义整理。*

---

# 补充材料：完整前向与反向过程（PDF 第 12–14 页，§8–§9）

以下摘自论文 **Supplementary Material** 的 **§8 Derivation of Forward rendering process** 与 **§9 Derivation of Differentiable Backward Pass**，公式编号沿用原文 **(13)–(54)**。文末用表格与正文 **§4 Method** 逐项对照。

## 与正文 §4 的总览对照

| 正文 | 补充材料 |
|------|----------|
| **§4.1** 聚焦 PSF、灵敏度 $h(-z_c)$、$\sigma_z$ 来源 | **§8** 式 (16)–(17) 与 **式 (18)–(20)**（$\sigma_z$ 与 NA、超声等）；正文 §4.1 未写 (18)–(20)，此处补全 |
| **§4.2** 卷积 → Focus Gaussian → $\tilde{\alpha}$ → 加性和 | **§8** 式 (13)–(15) 世界到相机，(21)–(37) **完整解析推导**（正文只给“现象学”分步） |
| **§4.3** 反向链、$\tilde{\alpha}$ 与 $\det\boldsymbol{\Sigma}_{2d}$ 耦合 | **§9** 式 (38)–(54) **显式梯度公式** |

---

## §8 前向渲染过程的解析推导

### 8.1 从世界系原语到相机系

成像目标由 3D 高斯原语组成，每个原语在**世界坐标** $\mathbf{x}=[x,y,z]^\top$ 下为：

$$
g(\mathbf{x})=\alpha\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right). \tag{13}
$$

为减少冗余，可学习参数记为 $\mathcal{G}=(\boldsymbol{\mu},\mathbf{s},\mathbf{q},\alpha)$：位置 $\boldsymbol{\mu}$、各向异性尺度 $\mathbf{s}\in\mathbb{R}^3$、单位四元数 $\mathbf{q}$、强度/不透明度 $\alpha$。CUDA 光栅化前将 $(\mathbf{s},\mathbf{q})$ 变为协方差（与经典 3DGS 一致）：对角缩放 $\mathbf{S}=\mathrm{diag}(\mathrm{mod}\cdot\mathbf{s})$（与正文/经典 3DGS 中由尺度向量构造对角缩放矩阵同角色；符号以 PDF 为准），$\mathbf{R}=\mathrm{quatToMatrix}(\mathbf{q})$，世界系协方差

$$
\boldsymbol{\Sigma}=\mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top.
$$

对每个虚拟切片，用其视图矩阵 $[\mathbf{R}_c\,|\,\mathbf{t}]$ 将中心与协方差变到**相机坐标**（聚焦成像平面）：

$$
\boldsymbol{\mu}_c=\mathbf{R}_c\boldsymbol{\mu}+\mathbf{t}, \tag{14}
$$

$$
\boldsymbol{\Sigma}_c=\mathbf{R}_c\boldsymbol{\Sigma}\mathbf{R}_c^\top. \tag{15}
$$

### 8.2 PSF、灵敏度与卷积成像模型

假设 PSF **空间不变**且为归一化 3D 高斯：

$$
\mathrm{psf}(\mathbf{x}_c)=\frac{1}{(2\pi)^{3/2}\sigma_x\sigma_y\sigma_z}\exp\left(-\frac{1}{2}\left(\frac{x_c^2}{\sigma_x^2}+\frac{y_c^2}{\sigma_y^2}+\frac{z_c^2}{\sigma_z^2}\right)\right). \tag{16}
$$

灵敏度图（轴向反转冲激响应）：

$$
h(-z_c)=\exp\left(-\frac{z_c^2}{2\sigma_z^2}\right). \tag{17}
$$

**$\sigma_z$ 与仪器参数（正文 §4.1 未展开）**：一般与 elevational 数值孔径 $\mathrm{NA}_{\mathrm{ele}}$、波长 $\lambda$ 有关，例如

$$
\sigma_z \approx 0.27\cdot\frac{\lambda}{\mathrm{NA}_{\mathrm{ele}}}. \tag{18}
$$

线光学系统（数值孔径 $\mathrm{NA}$）可写为 $\sigma_z\approx \lambda/\mathrm{NA}^2$（式 19）；超声近似为 $\sigma_z \approx k\lambda F/D_{\mathrm{ele}}$（式 20），其中 $F$ 为焦距，$D_{\mathrm{ele}}$ 为 elevational 有效孔径。

**单原语**在 $z_c=0$ 处的像为相机系高斯 $g_c$ 与 $h(-z)$ 的卷积取值：

$$
I(x_c,y_c)=\left[h(-z)*g_c(\mathbf{x}_c)\right]_{z_c=0}, \tag{21}
$$

$$
I(x_c,y_c,z_c=0)=\int_{-\infty}^{\infty} h(t)\,g_c(x_c,y_c,t)\,\mathrm{d}t. \tag{22}
$$

因 $g_e(\mathbf{x})=h(z_c)\,g_c(\mathbf{x})$ 为两高斯之积，仍为高斯形式，即 **Focus Gaussian**。

### 8.3 指数合并 → $\boldsymbol{\Sigma}_e,\boldsymbol{\mu}_e,\mathrm{opacity}_r$

$g_c$ 在相机系下的指数项为

$$
-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}_c^{-1}(\mathbf{x}-\boldsymbol{\mu}_c). \tag{23}
$$

灵敏度 $h$ 的指数可写成对 $\mathbf{x}$ 的二次型：

$$
-\frac{z_c^2}{2\sigma_z^2}=-\frac{1}{2\sigma_z^2}\mathbf{x}^\top(\mathbf{e}_3\mathbf{e}_3^\top)\mathbf{x},\quad \mathbf{e}_3=[0,0,1]^\top. \tag{24}
$$

总指数为

$$
-\frac{1}{2}\left[(\mathbf{x}-\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}_c^{-1}(\mathbf{x}-\boldsymbol{\mu}_c)+\frac{1}{\sigma_z^2}\mathbf{x}^\top(\mathbf{e}_3\mathbf{e}_3^\top)\mathbf{x}\right]. \tag{25}
$$

展开式 (25) 中第一项： 

$$
(\mathbf{x}-\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}_c^{-1}(\mathbf{x}-\boldsymbol{\mu}_c)=\mathbf{x}^\top\boldsymbol{\Sigma}_c^{-1}\mathbf{x}-2\mathbf{x}^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c+\boldsymbol{\mu}_c^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c. \tag{26}
$$

故完整指数为

$$
-\frac{1}{2}\left[\mathbf{x}^\top\left(\boldsymbol{\Sigma}_c^{-1}+\frac{\mathbf{e}_3\mathbf{e}_3^\top}{\sigma_z^2}\right)\mathbf{x}-2\mathbf{x}^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c+\boldsymbol{\mu}_c^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c\right]. \tag{27}
$$

与标准高斯形式 $-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_e)^\top\boldsymbol{\Sigma}_e^{-1}(\mathbf{x}-\boldsymbol{\mu}_e)$ 对比，得

$$
\boldsymbol{\Sigma}_e^{-1}=\boldsymbol{\Sigma}_c^{-1}+\frac{\mathbf{e}_3\mathbf{e}_3^\top}{\sigma_z^2}, \tag{28}
$$

$$
\boldsymbol{\Sigma}_e^{-1}\boldsymbol{\mu}_e=\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c. \tag{29}
$$

$$
\boldsymbol{\mu}_e=\boldsymbol{\Sigma}_e\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c. \tag{30}
$$

常数项 $\boldsymbol{\mu}_c^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c-\boldsymbol{\mu}_e^\top\boldsymbol{\Sigma}_e^{-1}\boldsymbol{\mu}_e$ 并入 **不透明度调制**（与正文 §4.2 一致）：

$$
\mathrm{opacity}_r=\exp\left(-\frac{1}{2}\left(\boldsymbol{\mu}_c^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c-\boldsymbol{\mu}_e^\top\boldsymbol{\Sigma}_e^{-1}\boldsymbol{\mu}_e\right)\right). \tag{31}
$$

### 8.4 对 $z_c$ 积分、2D 边缘分布与 $\tilde{\alpha}$

先将式 (22) 写回 Focus Gaussian 形式（对 $\mathbf{x}=(x_c,y_c,z_c)^\top$ 保留 $z_c$ 维以便积分）：

$$
I(x_c,y_c)=\alpha\cdot\mathrm{opacity}_r\int_{-\infty}^{\infty}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_e)^\top\boldsymbol{\Sigma}_e^{-1}(\mathbf{x}-\boldsymbol{\mu}_e)\right)\mathrm{d}z_c. \tag{32}
$$

被积式为 **三维**高斯密度形状；对 $z_c$ 积分得到 $(x_c,y_c)$ 上的 **二维边缘高斯**，其均值与协方差由 $\boldsymbol{\mu}_e,\boldsymbol{\Sigma}_e$ 的 $x,y$ 分量给出。积分常数并入系数后即为（式 33；与式 (36) 一致，系数为 $1/\sqrt{\det\boldsymbol{\Sigma}_{2d}}$）：

$$
I(x_c,y_c)=\frac{\alpha\cdot\mathrm{opacity}_r}{\sqrt{|\boldsymbol{\Sigma}_{2d}|}}\exp\left(-\frac{1}{2}([x_c,y_c]^\top-\boldsymbol{\mu}_{2d})^\top\boldsymbol{\Sigma}_{2d}^{-1}([x_c,y_c]^\top-\boldsymbol{\mu}_{2d})\right), \tag{33}
$$

其中 $|\boldsymbol{\Sigma}_{2d}|=\det\boldsymbol{\Sigma}_{2d}$，且

$$
\boldsymbol{\Sigma}_{2d}=\begin{bmatrix}\Sigma_e[0,0]&\Sigma_e[1,0]\\\Sigma_e[1,0]&\Sigma_e[1,1]\end{bmatrix},\quad \boldsymbol{\mu}_{2d}=[\mu_e[0],\mu_e[1]]^\top. \tag{34}
$$

像素 $\mathbf{p}$ 处令 $\mathbf{d}$ 为从 $\boldsymbol{\mu}_{2d}$ 指向 $\mathbf{p}$ 的 2D 偏移，则

$$
I(\mathbf{p})=\tilde{\alpha}\exp\left(-\frac{1}{2}\mathbf{d}^\top\boldsymbol{\Sigma}_{2d}^{-1}\mathbf{d}\right), \tag{35}
$$

$$
\tilde{\alpha}=\alpha\cdot\mathrm{opacity}_r\cdot\frac{1}{\sqrt{\det(\boldsymbol{\Sigma}_{2d})}}. \tag{36}
$$

多原语时由成像模型**线性叠加**（与正文 §4.2 加性光栅化一致）：

$$
I(\mathbf{p})=\sum_{i\in\mathcal{N}(\mathbf{p})}\tilde{\alpha}_i\exp\left(-\frac{1}{2}\mathbf{d}_i^\top\boldsymbol{\Sigma}_{2d,i}^{-1}\mathbf{d}_i\right). \tag{37}
$$

---

## §9 可微反向传播

### 9.1 前向量回顾（供反传衔接）

$$
\boldsymbol{\mu}_{2d}=[\mu_e[0],\mu_e[1]]^\top, \tag{38}
$$

$$
\boldsymbol{\Sigma}_{2d}^{-1}=\frac{1}{\det(\boldsymbol{\Sigma}_{2d})}\begin{bmatrix}\Sigma_e[1,1]&-\Sigma_e[1,0]\\-\Sigma_e[1,0]&\Sigma_e[0,0]\end{bmatrix}, \tag{39}
$$

$\boldsymbol{\Sigma}_{2d}$ 同式 (40) 与 (34)。渲染强度

$$
\tilde{\alpha}=\alpha\cdot\mathrm{opacity}_r\cdot\frac{1}{\sqrt{\det(\boldsymbol{\Sigma}_{2d})}}. \tag{41}
$$

以上把 2D 渲染与 3D 原语参数衔接起来。

### 9.2 反向公式（相机系内 → 世界系）

**对 2D 均值的梯度**（$\mu_e$ 的 $z$ 分量无来自像素的直接路径，置 0）：

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\mu}_e}=\begin{bmatrix}\mathrm{d}\mathcal{L}/\mathrm{d}\mu_{2d}[0]\\\mathrm{d}\mathcal{L}/\mathrm{d}\mu_{2d}[1]\\0\end{bmatrix}. \tag{42}
$$

**由 $\tilde{\alpha}$ 回传**（对应正文 §4.3 中“$\tilde{\alpha}$ 与行列式耦合”）：

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\alpha}=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\tilde{\alpha}}\cdot\frac{\mathrm{opacity}_r}{\sqrt{\det(\boldsymbol{\Sigma}_{2d})}}, \tag{43}
$$

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\,\mathrm{opacity}_r}=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\tilde{\alpha}}\cdot\frac{\alpha}{\sqrt{\det(\boldsymbol{\Sigma}_{2d})}}, \tag{44}
$$

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\,\det(\boldsymbol{\Sigma}_{2d})}=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\tilde{\alpha}}\cdot\left(-\frac{1}{2}\cdot\frac{\alpha\cdot\mathrm{opacity}_r}{(\det(\boldsymbol{\Sigma}_{2d}))^{3/2}}\right). \tag{45}
$$

**$\boldsymbol{\Sigma}_{2d}^{-1}$ 到 $\boldsymbol{\Sigma}_{2d}$**（利用 $\boldsymbol{\Sigma}_{2d}\boldsymbol{\Sigma}_{2d}^{-1}=\mathbf{I}$）：

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_{2d}}=-\boldsymbol{\Sigma}_{2d}\cdot\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_{2d}^{-1}}\cdot\boldsymbol{\Sigma}_{2d}+\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\,\det(\boldsymbol{\Sigma}_{2d})}\cdot\det(\boldsymbol{\Sigma}_{2d})\cdot\boldsymbol{\Sigma}_{2d}^{-1}. \tag{46}
$$

**铺回 $\boldsymbol{\Sigma}_e$**（仅 $2\times2$ 块非零）：

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_e}=\begin{bmatrix}
\mathrm{d}\mathcal{L}/\mathrm{d}\Sigma_{2d}[0,0]&\mathrm{d}\mathcal{L}/\mathrm{d}\Sigma_{2d}[1,0]&0\\
\mathrm{d}\mathcal{L}/\mathrm{d}\Sigma_{2d}[1,0]&\mathrm{d}\mathcal{L}/\mathrm{d}\Sigma_{2d}[1,1]&0\\
0&0&0
\end{bmatrix}. \tag{47}
$$

记 $q=\boldsymbol{\mu}_c^\top\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c-\boldsymbol{\mu}_e^\top\boldsymbol{\Sigma}_e^{-1}\boldsymbol{\mu}_e$（与 $\mathrm{opacity}_r=\exp(-q/2)$ 一致），**从 $(\boldsymbol{\mu}_e,\boldsymbol{\Sigma}_e)$ 到 $(\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c)$**：

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}q}=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\,\mathrm{opacity}_r}\cdot\left(-\frac{1}{2}\mathrm{opacity}_r\right), \tag{48}
$$

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\mu}_c}=\left(\boldsymbol{\Sigma}_e\boldsymbol{\Sigma}_c^{-1}\right)^\top\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\mu}_e}+\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}q}\cdot2\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c, \tag{49}
$$

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_c^{-1}}=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}q}\cdot\boldsymbol{\mu}_c\boldsymbol{\mu}_c^\top+\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_e^{-1}}, \tag{50}
$$

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_e^{-1}}=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}q}\cdot(-\boldsymbol{\mu}_e\boldsymbol{\mu}_e^\top)-\boldsymbol{\Sigma}_e\cdot\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_e}\cdot\boldsymbol{\Sigma}_e. \tag{51}
$$

**$\boldsymbol{\Sigma}_c^{-1}\to\boldsymbol{\Sigma}_c$**：

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_c}=-\boldsymbol{\Sigma}_c^{-1}\cdot\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_c^{-1}}\cdot\boldsymbol{\Sigma}_c^{-1}. \tag{52}
$$

**回到世界坐标**（文中记 $\tilde{\boldsymbol{\mu}},\tilde{\boldsymbol{\Sigma}}$ 对应世界系量；与式 (14)(15) 的前向互逆）：

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\tilde{\boldsymbol{\mu}}}=\mathbf{R}_c^\top\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\mu}_c}. \tag{53}
$$

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\tilde{\boldsymbol{\Sigma}}}=\mathbf{R}_c^\top\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\Sigma}_c}\mathbf{R}_c. \tag{54}
$$

此后可再经 $\boldsymbol{\Sigma}$ 对 $(\mathbf{s},\mathbf{q})$ 的构造链回传（与标准 3DGS 相同，补充 §9 未逐行展开）。

---

## 正文 §4 与补充 §8–§9 的细对照

- **§4.1 的 PSF / $h(-z_c)$**：与补充 **(16)(17)** 同构；补充额外给出 **(18)–(20)** 的 $\sigma_z$ 物理估计，正文只强调 Nyquist $\sigma_z\approx\delta_z$ 与实验标定。
- **§4.2 的“轴向重参数化 + opacity + 2D 投影 + 加性”**：在补充 **§8.3–8.4** 中由 **(21)–(37)** 从卷积一步推到底；正文 **式 (6)–(10)** 是同一套量的**实现等价分步**，无新物理，仅省略 (13)–(15) 的世界系与 (28)–(30) 的配平细节。
- **§4.3 的反向**：正文一句“$\tilde{\alpha}$ 行列式耦合 $\alpha$ 与 $\boldsymbol{\Sigma}_e$”；补充 **(43)–(46)** 给出 **$\alpha$、$\mathrm{opacity}_r$、$\det(\boldsymbol{\Sigma}_{2d})$** 的显式标量链；(47)–(52) 把梯度铺回 **$\boldsymbol{\Sigma}_e\to(\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c^{-1})$**，**(53)(54)** 对齐正文“(iii) 相机到世界”。

*注：式 (33) 原文排版在 PDF 中与式 (36) 应一致为 $\propto 1/\sqrt{\det\boldsymbol{\Sigma}_{2d}}$；此处按 (36)(41) 统一为 $\sqrt{\det}$ 写法。*
