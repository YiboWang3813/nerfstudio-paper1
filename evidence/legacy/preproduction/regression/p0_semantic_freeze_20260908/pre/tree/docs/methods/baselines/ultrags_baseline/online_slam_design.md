# Online Ultrasound GS 实现方案设计

## 1. 输入或轻微修改的输入

### 1.1 纯GT输入
一次 sweep 中的图像序列记为：

\[
\mathcal{D}=\{\mathbf{I}_i\}_{i=0}^{N-1}
\]

其中 \(\mathbf{I}_i\) 为第 \(i\) 帧 B-mode（或经预处理后的）观测。后续所有位姿矩阵 $\mathbf{P}_i$ 均为 \(4\times4\) 齐次变换，**语义统一为 `probe->world`**：将探头坐标系下的点变换到世界（或场景固定）坐标系。旋转矩阵 \(\mathbf{R}_i\) 的三列按超声约定与成像几何对齐：

\[
\mathbf{P}_i=
\begin{bmatrix}
\mathbf{R}_i & \mathbf{t}_i\\
\mathbf{0}^\top & 1
\end{bmatrix},\quad
\mathbf{R}_i=[\mathbf{r}_{x,i},\mathbf{r}_{y,i},\mathbf{r}_{z,i}]
\]

- \(\mathbf{r}_{x,i}\)：\(x\)-lateral（侧向）
- \(\mathbf{r}_{y,i}\)：\(y\)-axial（轴向，沿声束）
- \(\mathbf{r}_{z,i}\)：\(z\)-elevational（探头外法向/切片堆叠方向）
- \(x\)-lateral 与 \(y\)-axial 张成**成像平面**。

设传感器提供的 GT 绝对位姿序列为 \(\{\mathbf{P}_i^{abs}\}_{i=0}^{N-1}\)，其中 \(\mathbf{P}_i^{abs}\in \mathrm{SE}(3)\) 表示第 \(i\) 帧 `probe->world` 变换。为统一后续噪声注入、PEM 链式累计与窗口优化的参考系，对该序列做**首帧重基准（re-basing）**：\[
\mathbf{P}_i^{rel}:=\left(\mathbf{P}_0^{abs}\right)^{-1}\mathbf{P}_i^{abs},\quad i=0,\ldots,N-1.
\]
由定义立即得到
\[
\mathbf{P}_0^{rel}=\mathbf{I}_{4\times4},
\]
且对任意 \(i>0\)，\(\mathbf{P}_i^{rel}\) 表示「第 \(i\) 帧相对于第 0 帧坐标系」的累计刚体变换。后文若无特别说明，统一记
\[
\mathbf{P}_i \equiv \mathbf{P}_i^{rel},
\]
即默认使用重基准后的相对位姿。

### 1.2 GT+Noise 粗姿态（无位姿估计模型时）

在暂不接入真实 $\mathcal{M}_p$、或做**可控消融**时，在 GT 位姿 \(\mathbf{P}_i\) 上施加随机扰动，得到粗姿态 \(\mathbf{P}_i^{noise}\)，用于模拟「有偏估计」：

\[
\mathbf{P}_i^{noise}=\Delta\mathbf{T}_i\mathbf{P}_i
\]

\[
\Delta\mathbf{T}_i=
\begin{bmatrix}
\exp([\delta\boldsymbol{\theta}_i]_\times) & \delta\mathbf{t}_i\\
\mathbf{0}^\top & 1
\end{bmatrix},\quad
\delta\boldsymbol{\theta}_i\sim\mathcal{N}(0,\sigma_r^2\mathbf{I}),\quad
\delta\mathbf{t}_i\sim\mathcal{N}(0,\sigma_t^2\mathbf{I})
\]

其中 \(\sigma_r,\sigma_t\) 对应角度（弧度）与平移（与 \(\mathbf{t}_i\) 同单位）的标准差，实验中取多档以观察 refinement 的增益随噪声强度的变化。

## 2. 组件

设计原则：**面向对象、业务内聚、低耦合**。窗口调度与帧级预算归 **Local Window Manager**；高斯生命周期与「哪些参数在何时更新」归 **Gaussian State Manager**；数值目标（含帧加权、正则）归 **统一损失**；**Pipeline** 只做阶段编排与对象协作，不写具体公式。

**三种运行模式（共用同一套 LWM/GSM/窗口 Pipeline，仅输入与优化事件不同）**：

| 模式 | 粗姿态 \(\mathbf{P}_i^{\mathrm{coarse}}\) | PEM | PRM \(\Phi_p\) | 位姿相关损失 §3.4–3.5 |
|------|---------------------------------------------|-----|----------------|------------------------|
| **A. GT** | \(\mathbf{P}_i\)（真值） | 不调用 | 不调用 | 关闭 |
| **B. GT+Noise** | \(\mathbf{P}_i^{\mathrm{noise}}\)（§1.2） | 不调用 | **调用** | 开启（锚定 \(\mathbf{P}^{\mathrm{noise}}\)） |
| **C. PEM** | \(\mathbf{P}_i^{\mathrm{pred}}\)（§2.1，链式） | **调用** | **调用** | 开启（锚定 PEM 边） |

### 2.1 Pose Estimation Model (PEM)

**仅在模式 C 使用。** 位姿估计模型 PEM \(\mathcal{M}_p\) 以**相邻两帧**为输入，离线训练后在线推理相邻帧相对边：

\[
\mathbf{T}_{i-1\rightarrow i}^{\mathrm{pred}}=\mathcal{M}_p(\mathbf{I}_{i-1},\mathbf{I}_i)
\]

约定 \(\mathbf{P}_0^{\mathrm{pred}}=\mathbf{I}_{4\times4}\)。**与「边到即修」一致**：第 \(i\) 帧绝对位姿由**当前边**左乘到**上一帧已精化**的绝对位姿上（与 `ultrags_online_slam_new.md` 组三一致），而非乘在「未修正的纯链式累积」上：

\[
\mathbf{P}_i^{\mathrm{pred}}=\mathbf{T}_{i-1\rightarrow i}^{\mathrm{pred}}\mathbf{P}_{i-1}^{\mathrm{refined}}
\]

**模式 C** 中 \(\mathbf{P}_0^{\mathrm{pred}}=\mathbf{I}\)；\(\mathbf{P}_0^{\mathrm{refined}}\) 初值由 \(\Phi_p\) 与归一化约定共同确定（常与 \(\mathbf{I}\) 一致）。若某步尚未得到 \(\mathbf{P}_{i-1}^{\mathrm{refined}}\)，链式中可暂用 \(\mathbf{P}_{i-1}^{\mathrm{pred}}\) 的缓存。链式会放大漂移；\(\Phi_p\) 与窗口损失用于抑制。

### 2.2 Pose Refinement Model (PRM)

**仅在模式 B、C 使用。** 粗姿态统记为 \(\mathbf{P}_i^{\mathrm{coarse}}\in\{\mathbf{P}_i^{\mathrm{noise}},\mathbf{P}_i^{\mathrm{pred}}\}\)（模式 A 下 \(\mathbf{P}_i^{\mathrm{coarse}}\equiv\mathbf{P}_i\)，不进入 \(\Phi_p\)）。

PRM \(\Phi_p\) 将粗姿态映射为精化姿态（实现上为**浅层 MLP**，输入可为 \(\mathbf{P}_i^{\mathrm{coarse}}\) 的 \(\mathfrak{se}(3)\) 或矩阵元素等参数化）：

\[
\mathbf{P}_i^{\mathrm{coarse}}\xrightarrow{\Phi_p}\mathbf{P}_i^{\mathrm{refined}}
\]

**梯度来源（与主高斯联合训练并行）**：

1. **主重建路径**：用 \(\mathbf{P}_i^{\mathrm{refined}}\) 与当前窗口内活跃高斯做常规 \(\tilde{\mathcal{L}}_{\mathrm{rec}}\)（§3.1），\(\Phi_p\) 经可微渲染反传。
2. **Pose 辅助路径（可选，减轻本帧高斯与位姿争抢解释 \(\mathbf{I}_i\)）**：记 \(\mathcal{G}_{\mathrm{exclude}\,i}\subset\mathcal{G}\) 为**排除**「主归属为第 \(i\) 帧」的高斯后的子集（由 GSM 查询）。构造
   \[
   \mathbf{I}_i^{\mathrm{aux}}=\mathrm{Render}\!\left(\mathbf{P}_i^{\mathrm{refined}},\mathcal{G}_{\mathrm{exclude}\,i}\right),
   \]
   与 \(\mathbf{I}_i\) 计算与 §3.1 同形式的 \(\mathcal{L}_{\mathrm{rec}}^{\mathrm{pose\text{-}aux}}(i)\)（见 §3.1）。**端到端训练 \(\Phi_p\) 时应对 \(\mathbf{P}_i^{\mathrm{refined}}\) 反传**；若仅用 **GT \(\mathbf{P}_i\)** 与 \(\mathcal{G}_{\mathrm{exclude}\,i}\) 渲染，则该支路**不**对 \(\Phi_p\) 反传，可作分析或 teacher 对照，与主路径区分。

相邻帧的**相对位姿**由粗、精两套绝对位姿分别诱导，用于 §3.4–3.5：

\[
\mathbf{T}_{i-1\rightarrow i}^{\mathrm{coarse}}
:=
\mathbf{P}_i^{\mathrm{coarse}}\left(\mathbf{P}_{i-1}^{\mathrm{coarse}}\right)^{-1},\quad
\mathbf{T}_{i-1\rightarrow i}^{\mathrm{refined}}
:=
\mathbf{P}_i^{\mathrm{refined}}\left(\mathbf{P}_{i-1}^{\mathrm{refined}}\right)^{-1}
\]

**模式 B**：\(\mathbf{P}_{i-1}^{\mathrm{coarse}}\) 与 \(\mathbf{P}_i^{\mathrm{coarse}}\) 均由 §1.2 独立噪声得到，粗边一致。**模式 C**：\(\mathbf{P}_i^{\mathrm{coarse}}\) 来自 PEM 链式（上式 \(\mathbf{P}^{\mathrm{pred}}\)），\(\mathbf{P}_{i-1}^{\mathrm{coarse}}\) 在优化语境下取与当前缓存一致的粗姿态定义（通常即上一帧 PEM 输出或上一帧 refine 前快照，由实现固定一种约定，保证 \(\mathcal{L}_{\mathrm{prior}}\) 锚定对象稳定）。

### 2.3 Online Gaussian Window Manager（统一窗口 + 高斯语义状态）

实现上将原 **Local Window Manager（LWM）** 与 **Gaussian State Manager（GSM）** 合并为单一组件 `OnlineGaussianWindowManager`（代码 `online_gaussian_window_manager.py`），避免帧驱动循环中重复传递 `active_mask` 与双对象同步。文档上仍按逻辑分区说明其职责。

**窗口与预算（原 LWM 语义）**：无学习参数；**不**直接改高斯参数数值，只维护 per-Gaussian 布尔 `active_mask`、`last_seen_frame`、可选 `slice_membership`、帧级预算 \(c_i\) 与权重 \(w_i\)。

- **窗口定义**：\(\mathcal{W}_t=\{\max(0,t-W+1),\ldots,t\}\)。
- **帧级预算与权重**：\(w_i \propto (c_i+\epsilon)^{-\alpha}\)（与 §3.1 对接）；每步优化后更新 \(c_i\)。
- **`settle`**：由 `last_seen_frame` 与当前 \(t,W\) 更新 `active_mask`，并**同步**下方生命周期（见下）。

**归属与生命周期（原 GSM 语义）**：维护 `owner_frame_id`、离散 `lifecycle`（New / Window-active / Marginalized）；根据 `active_mask` 将滑出窗口的高斯标为 Marginalized。**PRM 辅助路径**（§2.2）通过 `get_exclude_indices(frame_id)` 得到 \(\mathcal{G}_{\mathrm{exclude}\,i}\)。

**与渲染/优化器**：通过 `apply_grad_mask`（如 `freeze_strategy=grad_mask`）对非 active 高斯梯度清零，实现「低反传/阻尼」效果；有效学习率日程仍由优化器侧管理。

**设计要点**：Marginalized **不是** Frozen（全零梯度）；合并后仍保持「窗口决定谁反传、归属决定 exclude 集」的语义不变。

### 2.5 Gaussian Merge Manager (GMM)

**职责**：持有「全局高斯容器 + 优化器句柄」的视图（可为薄封装），负责**去冗余与压缩**：在空间邻近且统计相似的高斯上执行 merge/clone 抑制，控制全局点数、显存与前向时延。GMM 不负责窗口调度、不负责损失权重，避免与 LWM/GSM 职责重叠。

**与 Local Window 的关系**：Local Window 通过 `active_mask` 控制「本步谁参与反向」，解决的是**算得动**；它不会删除或合并重复椭球。GMM 解决的是**表示复杂度与重复度**。二者互补，不可互相替代。

**是否需要保留**：

| 情形 | 建议 |
|------|------|
| 长序列、密采样、每帧播种/分裂频繁 | **保留并定期触发**，优先控制点数线性增长与视锥重叠冗余。 |
| 短 sweep、点数受限、资源充足 | 可**降频或关闭在线 merge**，仅保留末尾 compaction。 |

**接口设计（低耦合）**：

1. `collect_candidates(global_pool, meta) -> candidate_pairs_or_clusters`
2. `merge(candidates, constraints) -> remap_info`（返回 old->new 索引映射与存活 mask）
3. `sync_state(remap_info, ogwm) -> None`（统一同步 `active_mask`、`last_seen_frame`、归属与生命周期索引）
4. `post_merge_finetune(k_steps)`（可选，小步渲染一致性微调）

其中 `meta` 只读 GSM/LWM 提供的标签（如 `frame_id`、`last_seen_frame`、共视信息）；GMM 不改 LWM 的 \(w_i,c_i\) 逻辑。

**合并策略（实现要点）**：

1. **触发时机**：每 \(N\) 帧、点数超阈值、显存超阈值，或 sweep 结束后离线 compaction；与 checkpoint/可视化周期对齐。
2. **候选筛选**：\(\|\mu_a-\mu_b\|\) 小于尺度自适应阈值，且尺度/旋转/外观（强度、opacity）相似；可叠加「曾共视」约束减少误并。
3. **合并算子**：对候选 \((a,b)\) 生成新高斯，约束外观能量与协方差体积不过度漂移（守恒或上界）；必要时对局部做少量微调步。
4. **拓扑一致性**：每次 merge 后必须调用 `sync_state`，保证索引变化在 OGWM 中原子生效；若在线启用 densify/prune，复用同一 remap 通道。

**验收指标**：在重建质量（PSNR/SSIM）不明显下降前提下，全局点数显著下降，单帧渲染耗时或显存峰值可测下降。建议记录 `merge_before/after_points`、`ms_per_frame`、`max_mem_mb` 三组指标用于回归。

## 3. 统一损失函数

总损失由**加权图像重建**、**高斯尺度与 elevational 约束**、**可选时序几何一致性**、**相对位姿先验**与**时序平滑**（一阶/二阶）加权组成。

在线 Local Window 下，当前处理到第 \(t\) 帧时 \(\mathcal{W}_t=\{\max(0,t-W+1),\ldots,t\}\)。**模式 A（GT）**：\(\mathbf{P}_i^{\mathrm{coarse}}=\mathbf{P}_i^{\mathrm{refined}}=\mathbf{P}_i\)，不优化位姿，通常 \(\lambda_{\mathrm{prior}}=\lambda_{\mathrm{smooth}}=\lambda_{\mathrm{acc}}=0\)，\(\lambda_{\mathrm{pose\text{-}aux}}=0\)。**模式 B/C**：优化 \(\mathbf{P}_i^{\mathrm{refined}}\)（经 \(\Phi_p\)），开启 §3.4–3.5（权重按实验）；\(\mathcal{L}_{\mathrm{temp}}\)、\(\mathcal{L}_{z\text{-floor}}\) 仍按超声薄片按需开启。

### 3.1 重建损失（含 LWM 帧加权与可选 Pose 辅助）

**主重建**：\(\mathbf{I}_i^{\mathrm{rendered}}=\mathrm{Render}(\mathbf{P}_i^{\mathrm{refined}},\mathcal{G}_{\ast,i})\)，其中 \(\mathcal{G}_{\ast,i}\) 为当前 step 策略下对帧 \(i\) 参与光栅化的高斯子集（如 \(\mathcal{W}_t\) 内 active 等，由 GSM 决定）。单帧 \(i\) 的重建项与离线一致：

\[
\mathcal{L}_{\mathrm{rec}}(i)
=\lambda_{l1}\|\mathbf{I}_i-\mathbf{I}_i^{\mathrm{rendered}}\|_1
+\lambda_{ssim}\left(1-\mathrm{SSIM}(\mathbf{I}_i,\mathbf{I}_i^{\mathrm{rendered}})\right)
\]

一步 Local BA 中，对采样到的帧集合 \(\mathcal{S}\subseteq\mathcal{W}_t\)（可单帧或多帧 mini-batch），由 LWM 提供 \(w_i\)：

\[
\tilde{\mathcal{L}}_{\mathrm{rec}}
=\sum_{i\in\mathcal{S}} w_i\,\mathcal{L}_{\mathrm{rec}}(i),\quad
w_i \propto \big(c_i+\epsilon\big)^{-\alpha}
\]

\(c_i\) 在 LWM 中维护；归一化 \(\sum_{i\in\mathcal{S}} w_i\) 可按 batch 或固定常数，由实现约定。

**Pose 辅助重建（仅模式 B/C，可选）**：在 \(\mathcal{G}_{\mathrm{exclude}\,i}\) 上渲染 \(\mathbf{I}_i^{\mathrm{aux}}=\mathrm{Render}(\mathbf{P}_i^{\mathrm{refined}},\mathcal{G}_{\mathrm{exclude}\,i})\)，与 \(\mathbf{I}_i\) 同形式得 \(\mathcal{L}_{\mathrm{rec}}^{\mathrm{pose\text{-}aux}}(i)\)。总和中可写 \(\sum_{i\in\mathcal{S}'} \lambda_{\mathrm{pose\text{-}aux}} w_i' \mathcal{L}_{\mathrm{rec}}^{\mathrm{pose\text{-}aux}}(i)\)，\(\mathcal{S}'\) 可与 \(\mathcal{S}\) 相同或为其子集（例如每步只对当前帧 \(t\) 开一条辅助支路）。**模式 A** 下恒为 0。

### 3.2 高斯尺度约束与 elevational 下界

\(\mathrm{ScaleWall}(\{\mathbf{s}_j\})\) 惩罚过大的高斯尺度，抑制椭球无界膨胀：

\[
\mathcal{L}_{scale}=\mathrm{ScaleWall}(\{\mathbf{s}_j\})
\]

针对超声切片**极易退化为面**的问题，对 elevational 方向尺度 \(\sigma_{z,j}\)（或协方差特征对应项）增加**下界惩罚**（与实现中参数化一致，示例为软 hinge）：

\[
\mathcal{L}_{z\text{-floor}}
=\sum_{j\in\mathcal{J}_{\mathrm{active}}}
\left[\log\epsilon_z-\log\sigma_{z,j}\right]_+^2
\]

其中 \(\mathcal{J}_{\mathrm{active}}\) 为当前 step 参与反传的高斯索引集，\(\epsilon_z\) 为最小可接受厚度尺度。**本项属于损失层**；GSM 只决定哪些 \(j\) 属于 active，不计算 \(\mathcal{L}_{z\text{-floor}}\)。

### 3.3 时序几何一致性（可选）

对跨帧对应关系 \(\mathcal{M}\)（由最近邻、小块 warp 或索引传播得到）惩罚尺度与旋转的跳跃，减轻「每帧各训各的」薄片感：

\[
\mathcal{L}_{\mathrm{temp}}
=\sum_{(j,j')\in\mathcal{M}}
\left(
\|\mathbf{s}_j-\mathbf{s}_{j'}\|_2^2
+
\|\mathbf{R}_j-\mathbf{R}_{j'}\|_F^2
\right)
\]

权重 \(\lambda_{\mathrm{temp}}\) 按数据调节；GT Pose 下该项**仅约束几何**，与位姿无关。

### 3.4 Pose prior（相对位姿、锚定粗运动）

希望 refined 的相邻相对变换不要偏离 coarse 给出的边太远。在 \(\mathrm{SE}(3)\) 上，定义**粗边到精边**的残差：

\[
\Delta\mathbf{T}_i
=\left(\mathbf{T}_{i-1\rightarrow i}^{coarse}\right)^{-1}
\mathbf{T}_{i-1\rightarrow i}^{refined},\quad
\mathbf{e}_i=\log(\Delta\mathbf{T}_i)\in\mathbb{R}^6
\]

\(\log(\cdot)\) 取李群到李代数的对数映射（6 维：3 旋转 + 3 平移）。对窗口内除最旧帧外的边求和，并对 \(\|\mathbf{e}_i\|_2\) 施加鲁棒损失 \(\rho(\cdot)\)（如 Huber），减轻外点边对整体的拉扯：

\[
\mathcal{L}_{prior}
=\sum_{i\in\mathcal{W}_t\setminus\{t-W+1\}}
\rho(\|\mathbf{e}_i\|_2)
\]

该项在 **GT Pose 组** 中通常关闭（无可优化位姿或退化为平凡项）；在 **Noise/PEM 组** 中用于锚定粗运动。

### 3.5 平滑项（一阶与二阶）

**一阶平滑** \(\mathcal{L}_{smooth}\)：惩罚 refined 绝对位姿序列相邻帧相对变换的幅值（在 \(\mathfrak{se}(3)\) 范数下），抑制高频抖动。

\[
\mathcal{L}_{smooth}
=\sum_{i\in\mathcal{W}_t\setminus\{t-W+1\}}
\left\|\log\!\left(\mathbf{P}_i^{refined}\left(\mathbf{P}_{i-1}^{refined}\right)^{-1}\right)\right\|_2^2
\]

**二阶平滑** \(\mathcal{L}_{acc}\)：惩罚相邻**相对运动**之间的变化（类比加速度），使轨迹更平滑；求和下标从 \(t-W+3\) 到 \(t\)，保证用到连续三帧的相对边。

\[
\mathcal{L}_{acc}
=\sum_{i=t-W+3}^{t}
\left\|
\log\!\left(\mathbf{P}_i^{refined}\left(\mathbf{P}_{i-1}^{refined}\right)^{-1}\right)
-\log\!\left(\mathbf{P}_{i-1}^{refined}\left(\mathbf{P}_{i-2}^{refined}\right)^{-1}\right)
\right\|_2^2
\]

GT Pose 下 \(\lambda_{\mathrm{smooth}}=\lambda_{\mathrm{acc}}=0\)。

### 3.6 总损失

\[
\mathcal{L}
=\tilde{\mathcal{L}}_{\mathrm{rec}}
+\lambda_{\mathrm{pose\text{-}aux}}\sum_{i\in\mathcal{S}'} w_i\,\mathcal{L}_{\mathrm{rec}}^{\mathrm{pose\text{-}aux}}(i)
+\lambda_{scale}\mathcal{L}_{scale}
+\lambda_{z}\mathcal{L}_{z\text{-floor}}
+\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}
+\lambda_{prior}\mathcal{L}_{prior}
+\lambda_{smooth}\mathcal{L}_{smooth}
+\lambda_{acc}\mathcal{L}_{acc}
\]

**权重习惯**：\(\lambda_{scale}\) 通常始终开启；\(\lambda_z,\lambda_{\mathrm{temp}}\) 针对超声薄片与跨帧不连续；\(\lambda_{\mathrm{pose\text{-}aux}}\) 仅在模式 B/C 且启用 \(\mathcal{G}_{\mathrm{exclude}\,i}\) 辅助时非零；\(\lambda_{prior},\lambda_{smooth},\lambda_{acc}\) 在**模式 A** 为零，在 **B/C** 按 `ultrags_online_slam_new.md` 网格搜索（常取 \(\eta_{\mathrm{pose}}<\eta_{\mathrm{gs}}\) 的 Joint 日程）。Pairwise 子图仅含 \((i-1,i)\) 时，求和范围退化为单条边。

## 4. 完整多条件 Pipeline（Local Window × 三种位姿模式）

**共用约定**：窗口大小 \(W\ge1\)；全局高斯池 \(\mathcal{G}\)；LWM 维护 \(\mathcal{W}_t\) 与 \(\{c_i\}\)；GSM 标注生命周期与 \(\mathcal{G}_{\mathrm{exclude}\,i}\)；每步主损失为 §3.6；**渲染位姿**统一为 \(\mathbf{P}_i^{\mathrm{refined}}\)（模式 A 下 \(\mathbf{P}_i^{\mathrm{refined}}\equiv\mathbf{P}_i\)）。

**问题背景（与模式无关）**：单 slice 过薄易退化为面；对策仍是 **LWM**、**GSM**、**\(\mathcal{L}_{z\text{-floor}},\mathcal{L}_{\mathrm{temp}}\)** 与法向膨胀式播种，见 §2–3。

### 4.0 三种模式与额外优化事件

| 步骤 | 模式 A（GT） | 模式 B（GT+Noise） | 模式 C（PEM） |
|------|--------------|-------------------|---------------|
| 粗姿态来源 | \(\mathbf{P}_i\) | §1.2 得 \(\mathbf{P}_i^{\mathrm{noise}}\) | §2.1：\(\mathbf{T}^{\mathrm{pred}}=\mathcal{M}_p(\mathbf{I}_{i-1},\mathbf{I}_i)\)，\(\mathbf{P}_i^{\mathrm{pred}}=\mathbf{T}_{i-1\rightarrow i}^{\mathrm{pred}}\mathbf{P}_{i-1}^{\mathrm{refined}}\) |
| PEM | 不调 | 不调 | **每新帧 \(t\ge1\) 调一次** |
| \(\Phi_p\) | 不调 | **调** | **调** |
| 位姿损失 §3.4–3.5 | 关 | 开 | 开 |
| \(\mathcal{L}^{\mathrm{pose\text{-}aux}}\) | 关 | 可选开 | 可选开 |

**额外事件（模式 B/C，与 `ultrags_online_slam_new.md` 对齐）**：在每一新帧 \(t\ge1\) 完成播种、且已得到 \(\mathbf{P}_t^{\mathrm{coarse}}\) 之后、进入该帧的 **Local BA** 之前，可插入 **\(K_{\mathrm{pair}}\ge0\)** 步 **Pairwise 子图**优化（仅边 \((t-1,t)\) 上的 \(\mathbf{P}_t^{\mathrm{refined}}\) 与/或高斯，损失为 \(\mathcal{L}_{\mathrm{rec}}+\lambda_{\mathrm{prior}}\mathcal{L}_{\mathrm{prior}}^{(t-1,t)}+\cdots\)），即「**边到即修**」。\(K_{\mathrm{pair}}=0\) 时退化为直接进入窗口 Local BA。**模式 A**：无 Pairwise、无 \(\Phi_p\)。

### 4.1 初始化（\(t=0\)）

1. **模式 B**：按 §1.2 生成 \(\mathbf{P}_0^{\mathrm{noise}}\)，令 \(\mathbf{P}_0^{\mathrm{coarse}}\leftarrow\mathbf{P}_0^{\mathrm{noise}}\)，\(\mathbf{P}_0^{\mathrm{refined}}=\Phi_p(\mathbf{P}_0^{\mathrm{coarse}})\)。**模式 C**：\(\mathbf{P}_0^{\mathrm{pred}}=\mathbf{I}\)，\(\mathbf{P}_0^{\mathrm{refined}}=\mathbf{I}\)（与 §2.1 一致）。**模式 A**：\(\mathbf{P}_0^{\mathrm{refined}}=\mathbf{P}_0\)。
2. 从 \(\mathbf{I}_0\) 采样 \(\mathcal{G}_{\mathrm{seed}}\) 并入 \(\mathcal{G}\)；GSM：**New**。
3. 用 \(\mathbf{P}_0^{\mathrm{refined}}\) 渲染，计算 §3.6（模式 A 无位姿项；B/C 若仅训高斯可先冻结 \(\Phi_p\) 若干步，由实现决定）。
4. 重复 **\(N_G\)** 步或直至 \(\mathcal{L}\) 低于阈值；LWM 更新 \(c_0\)。

### 4.2 Warm-up（\(1\le t \le W-1\)）

对每一新帧 \(t\)，\(\mathcal{W}_t=\{0,\ldots,t\}\)；**边际化尚未发生**（直至 \(t\ge W\)）。

1. **位姿（仅 B/C）**：
   - **B**：\(\mathbf{P}_t^{\mathrm{coarse}}\leftarrow\mathbf{P}_t^{\mathrm{noise}}\)（§1.2），\(\mathbf{P}_t^{\mathrm{refined}}=\Phi_p(\mathbf{P}_t^{\mathrm{coarse}})\)。
   - **C**：\(\mathbf{T}_{t-1\rightarrow t}^{\mathrm{pred}}=\mathcal{M}_p(\mathbf{I}_{t-1},\mathbf{I}_t)\)，\(\mathbf{P}_t^{\mathrm{pred}}=\mathbf{T}_{t-1\rightarrow t}^{\mathrm{pred}}\mathbf{P}_{t-1}^{\mathrm{refined}}\)，\(\mathbf{P}_t^{\mathrm{coarse}}\leftarrow\mathbf{P}_t^{\mathrm{pred}}\)，再经 \(\Phi_p\) 得 \(\mathbf{P}_t^{\mathrm{refined}}\)。
   - **A**：\(\mathbf{P}_t^{\mathrm{refined}}=\mathbf{P}_t\)。
2. **可选**：若 \(K_{\mathrm{pair}}>0\) 且 `mode ∈ {B,C}`，在 \((t-1,t)\) 上执行 \(K_{\mathrm{pair}}\) 步 Pairwise（`new.md` 组二 2.2 / 组三 3.2）。
3. **播种**（同前）：\(\mathbf{I}_t\) 采样 + 可选沿 \(\mathbf{r}_{z,t}\) 自 \(t-1\) 复制；GSM：**New**，余 **Window-active**。
4. **Local BA**：**\(K_{\mathrm{warm}}\)** 步。每步：LWM 给出 \(\mathcal{S},\{w_i\}\)；渲染用 \(\{\mathbf{P}_i^{\mathrm{refined}}\}_{i\in\mathcal{S}}\)；若开启 \(\mathcal{L}^{\mathrm{pose\text{-}aux}}\)，GSM 提供 \(\mathcal{G}_{\mathrm{exclude}\,i}\)；反传更新 \(\mathcal{G}\) 与（若未冻结）\(\Phi_p\)；更新 \(c_i\)。

### 4.3 稳态滑动（\(t\ge W\)）

\(\mathcal{W}_t=\{t-W+1,\ldots,t\}\)；**首次**边际化 `frame_id` \(0\) 出现在 \(t=W\)。

每进入新一帧 \(t\ge W\)：

1. **位姿子流程**：同 §4.2 步 1–2（B/C 的 PEM+\(\Phi_p\) / Noise+\(\Phi_p\)；A 仅用 \(\mathbf{P}_t\)）。
2. **播种**；**边际化** `frame_id` \(t-W\)（GSM → **Marginalized**）。
3. **Local BA**：**\(K_{\mathrm{step}}\)** 步，流程同 §4.2 步 4。
4. **GMM（可选）**：同前（§2.5）。

### 4.4 Sweep 尾部（可选）

LWM **replay** 可与 **模式 B/C** 下对 \(\Phi_p\) 的 **额外低 LR 扫步** 叠加（由实验配置），仅对 \(c_i\) 偏低帧增加 \(\tilde{\mathcal{L}}_{\mathrm{rec}}\)。

### 4.5 协作关系小结

| 对象 | 模式 A | 模式 B | 模式 C |
|------|--------|--------|--------|
| **OGWM**（统一 LWM+GSM） | 同 | 需提供 \(\mathcal{G}_{\mathrm{exclude}\,i}\) 若开 pose-aux | 同左 |
| **损失 §3** | 无位姿项 | 全项 + 可选 \(\mathcal{L}^{\mathrm{pose\text{-}aux}}\) | 同 B |
| **PEM** | — | — | 每帧前边推理 + 链式到 \(\mathbf{P}_t^{\mathrm{pred}}\) |
| **\(\Phi_p\)** | — | 自 \(\mathbf{P}^{\mathrm{noise}}\) 洗姿态 | 自 \(\mathbf{P}^{\mathrm{pred}}\) 洗姿态 |
| **GMM** | 可选 | 可选 | 可选 |

## 5. 代码实现对照（双向检查版）

本节以当前 `methods/ultrags/` 的 Python 代码为准，逐项对照组件、接口与 Pipeline，确保文档与实现一一对应。**模式 C 的 Pose Estimation Model 仍留空占位，不在本节展开代码实现。**

### 5.1 组件与类映射（当前真实代码）

| 组件 | 代码位置 | 核心类/函数 | 当前实现状态 |
|------|----------|-------------|-------------|
| Scene 数据流 | `scene/scene.py`, `scene/factory.py`, `scene/scheduler/online_scheduler.py` | `Scene`, `SceneFactory`, `OnlineScheduler` | 已实现 Online 顺序流与增量采样 |
| 高斯容器 | `methods/ultrags/gauss_container.py` | `UltraGSGaussContainer` | 已实现参数注册与初始化 |
| 高斯求解器 | `methods/ultrags/gauss_solver.py` | `UltraGSGaussSolver` | 已实现优化器与学习率调度 |
| 渲染 | `methods/ultrags/render.py` | `render(...)` | 已支持 `gauss_indices` 子集渲染与全局索引回传 |
| OGWM（LWM+GSM 合并） | `methods/ultrags/online_gaussian_window_manager.py`；`local_window_manager`/`gaussian_state_manager` 为兼容别名 | `OnlineGaussianWindowManager` | 窗口、budget、owner、lifecycle、exclude、grad mask 合一 |
| UltraGS 运行时 | `methods/ultrags/training_runtimes.py`, `methods/ultrags/state.py` | `TrainingRunTimes`（image/pose/gauss + 在线帧 scratch） | Loss/Metric 从 `state.runtimes` 读取；`UltraGSOnlineState.model_cfg` 供插件读配置 |
| GMM | `methods/ultrags/gaussian_merge_manager.py` | `GaussianMergeManager`, `MergeRemapInfo` | 已实现 MVP 合并与 remap 同步 |
| Loss 计算 | `methods/ultrags/computers.py` + `utils/loss/pose_loss.py` | `UltraGSOnlineLossComputer` + `pose_prior/smooth/acc/pose_aux_rec` | 已支持在线 pose 项真实分发与注册 |
| PoseRefiner | `methods/ultrags/pose_refiner.py` | `HighDimEncoder`, `PoseRefiner` | 已实现 HighDimEncoder + Encoder/Decoder MLP |
| Online 状态总线 | `methods/ultrags/state.py` | `UltraGSOnlineState` | `ogwm` / `gmm` / `pose_carryover` / `runtimes` |
| Online 主回调 | `methods/ultrags/online_callbacks.py` | `UltraGSOnlineSetupCallback`, `UltraGSOnlineStreamCallback` | 已实现 A/B 主干与 C 占位 |

### 5.2 配置接口（`config.py`）与代码读取对应

`methods/ultrags/config.py` 的 `OnlineConfig` 已采用嵌套 dataclass，主回调读取关系如下：

- `online.mode` -> `online_callbacks.py` / 插件 `mode = str(online_cfg.mode).upper()`
- `online.iters_per_frame / warmup_opt_iters / seed_every_frames / eval_every_frames / save_every_frames / pose_refine_steps_per_frame` -> Online 循环与事件频率
- `online.lwm.*` + `online.gsm.freeze_strategy` -> `OnlineGaussianWindowManager` 初始化与 `settle` / `apply_grad_mask`
- `online.gauss_lr_max_steps` -> `UltraGSGaussSolver` 指数 LR 的 `max_steps`（0 则用 `engine.iterations`）
- `online.gmm.merge_every_frames / merge_dist_thresh_mm / merge_opacity_thresh` -> GMM 初始化与触发
- `online.noise_pose.rot_std_deg / trans_std_mm` -> case B 噪声姿态采样
- `online.pose_refiner.*` -> `PoseRefiner` 网络结构与学习率

### 5.3 关键类与函数接口（当前签名）

1. **OGWM (`online_gaussian_window_manager.py`)**
   - `initialize(num_points, warmup_last_frame)`
   - `settle(cur_frame_id, window_size_frames)`（内部同步 lifecycle）
   - `update_seen(seen_gauss_indices, frame_id)`
   - `on_concat(n_new, frame_id)` / `on_prune` / `on_remap`
   - `build_exclude_mask` / `get_exclude_indices`
   - `apply_grad_mask(container, active_mask=None)`
   - `get_frame_weight` / `update_frame_budget`
   - 属性：`active_mask`, `last_seen_frame`, `owner_frame_id`, `lifecycle`, `slice_membership`

2. **GMM (`gaussian_merge_manager.py`)**
   - `collect_candidates(global_pool, meta)`
   - `merge(candidates, solver, container, constraints) -> MergeRemapInfo`
   - `sync_state(remap_info, ogwm)`
   - `post_merge_finetune(k_steps)`

3. **Render (`render.py`)**
   - `render(cfg, dataset_meta, slice_meta, container, active_global_indices=None, is_training=False, return_intermediates=False)`
   - 返回：`pred_image, depth_image, intermediates`
   - 当 `gauss_indices` 非空时，`intermediates` 额外含 `pixel_gauss_indices_global`

4. **PoseRefiner (`pose_refiner.py`)**
   - `HighDimEncoder.forward(x)`：sin-cos 高频编码
   - `PoseRefiner.forward(coarse_pose)`：`coarse -> high_dim -> encoder_mlp -> decoder_mlp -> delta -> refined`
   - 支持输入 `(...,4,4)` 或 `(...,pose_dim)`

### 5.4 Online Pipeline（按 `callbacks.py` 实现顺序）

#### 5.4.1 Setup (`UltraGSOnlineSetupCallback.on_setup`)

1. 创建 `Scene`
2. 初始化 `UltraGSGaussContainer` 与 `UltraGSGaussSolver`
3. 创建 `UltraGSOnlineLossComputer` + `UltraGSOfflineMetricComputer`
4. 创建 `LocalWindowManager` / `GaussianStateManager`；按需创建 `UltraGSOnlineDensityController`
5. 根据 `online.mode` 初始化 pose 相关组件：
   - A：不启用 PoseRefiner
   - B：启用 `PoseRefiner` 与 `pose_refiner_optimizer`
   - C：占位（不启用 PEM/PRM 真实前向）
6. 可选 warmup 优化（在 warmup 帧上循环）

#### 5.4.2 Stream Loop (`UltraGSOnlineStreamCallback.on_train`)

每帧执行：

1. `scene.next_stream_datum()` 取帧；`_resolve_stream_new_gaussians` 按 ``seed_every_frames`` 与 ``num_new`` 生成待 concat 高斯
2. 管道内 `run_concat_seed`：`concat` + `gsm.on_concat`（及 LWM 步进）
3. 模式分支（POSE_WASH 等）：A/B/C 见 `online_callbacks`
4. 每帧多步 `gs_optim_one_step`：渲染、loss、反传、`gsm` membership 更新、可选 DC
5. `atom_merge`：按 ``density_control.merge_every_frames`` 由 DC 执行合并
6. 日志、评估、保存

### 5.5 设计-实现差异与潜在问题（双向检查结论）

1. **模式 C 未实现真实 PEM**
   - 当前仅保留占位分支；没有 `PoseEstimationModel` 的前向与参数管理。
   - 这是已知留空，不影响模式 A/B 跑通。

2. **GMM 为 MVP 合并策略**
   - 当前候选筛选为简化近邻策略；
   - 尚未加入尺度/旋转/共视联合判据及合并后局部微调。

### 5.6 当前可视为“已对齐”的范围

- **能跑通**：A（GT）与 B（GT+Noise+PoseRefiner 监督）主流程；
- **接口齐全**：OGWM/GMM/Render 子集索引/OnlineLossComputer/`TrainingRunTimes`/PoseRefiner；
- **未实现且明确留空**：模式 C（PEM 真实代码）。
