# Online Ultrasound GS 更新规格

本文合并 `docs/methods/baselines/ultrags_baseline/online_slam_design.md` 中 **§1 输入**、**§2.1 PEM（公式，实现搁置）**、**§2.2 PRM**、**§3 损失** 与 `docs/methods/baselines/ultrags_baseline/online_without_pose_workflow.md` 的 **窗格预算、GSM（membership / 子集）与 Pipeline 真值**。
**流程顺序**以 `docs/methods/baselines/ultrags_baseline/online_without_pose_workflow.md` 为准；**训练强度**由 **帧出现次数预算**（\(C_i^{W_j}\) 与 \(K^{\mathrm{ind}}/K^{\mathrm{joint}}\) 的步数）控制，**主重建不对帧乘 \(w_i\)**。

**实现范围（当前迭代）**：**必须实现 Gaussian State Manager（GSM）**，且 **仅**按 `docs/methods/baselines/ultrags_baseline/online_without_pose_workflow.md` §0：**`membership`**（`concat` 时写入出生帧 + 每步 GS 后按 `render` 回传的 **`seen_gauss_global_indices`** 更新）、**可选**由 membership 筛 \(\mathcal{G}_{\mathrm{sub}}\)（超参 **`sub_locality_l`**）；**不**引入与 workflow 无关的 **`last_seen_frame` / `active_mask` / `settle` 边际化** 等 OGWM 遗留语义——**高斯侧状态以 `membership` 为主**。**不**接入 **PEM**（`pose_estimator = None`）。**流式取数**：`scene/scene.py` 的 **`next_stream_datum()`** 只返回 `slice_meta` 与帧索引，**不在 Scene 内做像素播种**。**流式播种**：`online_callbacks._resolve_stream_new_gaussians` 按 **`model.online.seed_every_frames`** 与 **`scene.sampler.num_new`** 调用 **`UnifiedGaussianSampler`**（每帧少量点；**增密主要靠 DC**）。**密度与合并**：**`density_control.py`** 中的 **`UltraGSOnlineDensityController`**（clone/split/prune + 可选近邻合并），配置见 **`model.online.density_control`**（已无独立 `gmm` 模块）。

---

## 1. 符号定义

### 1.1 数据与位姿（slam §1）

一次 sweep 图像序列 \(\mathcal{D}=\{\mathbf{I}_i\}_{i=0}^{N-1}\)。\(\mathbf{P}_i\in\mathbb{R}^{4\times4}\) 为齐次变换，语义 **`probe→world`**：

\[
\mathbf{P}_i=
\begin{bmatrix}
\mathbf{R}_i & \mathbf{t}_i\\
\mathbf{0}^\top & 1
\end{bmatrix},\quad
\mathbf{R}_i=[\mathbf{r}_{x,i},\mathbf{r}_{y,i},\mathbf{r}_{z,i}]
\]

\(\mathbf{r}_{x,i}\)：lateral；\(\mathbf{r}_{y,i}\)：axial（声束）；\(\mathbf{r}_{z,i}\)：elevational；\(x\)–\(y\) 张成成像平面。

GT 绝对位姿 \(\{\mathbf{P}_i^{abs}\}\) 经**首帧重基准**：\(\mathbf{P}_i^{rel}:=(\mathbf{P}_0^{abs})^{-1}\mathbf{P}_i^{abs}\)，后文默认 \(\mathbf{P}_i\equiv\mathbf{P}_i^{rel}\)，\(\mathbf{P}_0=\mathbf{I}\)。

**GT+Noise（模式 B，slam §1.2）**：\(\mathbf{P}_i^{noise}=\Delta\mathbf{T}_i\mathbf{P}_i\)，\(\Delta\mathbf{T}_i\) 由小旋转 \(\delta\boldsymbol{\theta}_i\) 与小平移 \(\delta\mathbf{t}_i\) 构成（方差 \(\sigma_r^2,\sigma_t^2\)）。

### 1.2 窗格与预算（workflow + `docs/methods/baselines/ultrags_baseline/online_local_window_budget.md`）

| 符号 | 含义 |
|------|------|
| \(N\) | 帧数；\(i\in\{0,\ldots,N-1\}\) |
| \(T\) | 与 Offline 对齐的总优化步预算；\(S=T/N\) |
| \(W\) | Local window 长度；\(M=N-W+1\) 个窗 |
| \(W_j\) | 第 \(j\) 个窗，\(j=0,\ldots,M-1\) |
| \(\mathcal{L}_j\) | \(\mathcal{L}_j=\{j,j+1,\ldots,j+W-1\}\) |
| \(t_{\mathrm{new}}(j)\) | 窗内最右帧，\(t_{\mathrm{new}}(j)=j+W-1\) |
| \(C_i^{W_j}\) | 帧 \(i\) 在窗 \(W_j\) 内的 GS 步数预算（\(i\notin\mathcal{L}_j\) 时为 0） |
| \(C_i^{\mathrm{all}}\) | \(\sum_j C_i^{W_j}\) |
| \(r^{\mathrm{front}},r^{\mathrm{rear}}\) | **cfg 超参数**，单窗内独立/联合比例，\(r^{\mathrm{front}}+r^{\mathrm{rear}}=1\)；与 \(C_i^{W_j}\) 组合得 \(K^{\mathrm{ind}}/K^{\mathrm{joint}}\)（见 §2.1） |
| \(\mathcal{G}\) | 全局高斯池 \(G_{\mathrm{global}}\)（`GaussContainer`） |
| \(N_m\) | `membership` 槽数（cfg，常 3 或 5，见 workflow §0） |
| \(l\) | 子集局部性超参；用于 \(\mathcal{G}_{\mathrm{sub}}\)（workflow §0 末段） |

\(C_i^{W_j}\) 分段式与 `docs/methods/baselines/ultrags_baseline/online_local_window_budget.md` 一致；\(W=1\) 时 \(C_i^{W_j}=S\,\mathbf{1}_{\{i=j\}}\)。

### 1.3 配置项（cfg，与 Pipeline / LWM 对接）

| 配置键（建议） | 含义 |
|----------------|------|
| `total_opt_steps` 或等价 | 对应 \(T\) |
| `window_size_frames` | 对应 \(W\) |
| `ratio_front` | \(r^{\mathrm{front}}\) |
| `ratio_rear` | \(r^{\mathrm{rear}}\) |
| `num_pose_refinement` | Pose 清洗 `POSE_WASH` 内 **仅训练 \(\Phi_p\)** 的迭代步数（原叙事 \(N_p\)） |
| `seed_every_frames` | 每隔多少 **stream 帧** 做一次像素 concat 播种（`0` = 仅首帧池空时播种） |
| `scene.sampler.num_new`（及 `scales_init` 等） | 单次像素混合采样条数（与 `seed_every_frames` 配合） |
| `membership_k` | 对应 \(N_m\)（GSM）；`0` 时无 `membership` 张量 |
| `sub_locality_l` | 对应 \(l\)；**`0`** 时等价全量 \(\mathcal{G}\)，`render` 不传子集索引 |
| `model.online.density_control.*` | Online DC：`enable` / `timing` / `thresholds`；合并：`merge_after_dc`、`merge_every_frames`、`merge_dist_thresh_mm`、`merge_opacity_thresh`、`merge_max_pairs` |

**说明**：\(r^{\mathrm{front}},r^{\mathrm{rear}}\) 由 cfg **直接读取**，与 LWM 输出的 \(C_i^{W_j}\) 按 §2.1 公式组合为整数步数 \(K_i^{\mathrm{ind}},K_i^{\mathrm{joint}}\)（取整规则实现自定）。

### 1.4 运行模式（slam §2 表）

| 模式 | \(\mathbf{P}_i^{\mathrm{coarse}}\) | PEM | \(\Phi_p\) | §3.4–3.5 |
|------|-----------------------------------|-----|------------|----------|
| **A** | \(\mathbf{P}_i\)（GT） | 否 | 否 | 关 |
| **B** | \(\mathbf{P}_i^{noise}\)（§1.1） | 否 | 是 | 开 |
| **C** | \(\mathbf{P}_i^{\mathrm{pred}}\)（PEM，**当前不实现**） | 是（**占位 None**） | 是 | 开 |

### 1.5 粗精位姿与相对边（slam §2.2）

\(\mathbf{P}_i^{\mathrm{coarse}}\xrightarrow{\Phi_p}\mathbf{P}_i^{\mathrm{refined}}\)。渲染统一用 \(\mathbf{P}_i^{\mathrm{refined}}\)（模式 A 下等于 \(\mathbf{P}_i\)）。

\[
\mathbf{T}_{i-1\rightarrow i}^{\mathrm{coarse}}
=\mathbf{P}_i^{\mathrm{coarse}}\bigl(\mathbf{P}_{i-1}^{\mathrm{coarse}}\bigr)^{-1},\quad
\mathbf{T}_{i-1\rightarrow i}^{\mathrm{refined}}
=\mathbf{P}_i^{\mathrm{refined}}\bigl(\mathbf{P}_{i-1}^{\mathrm{refined}}\bigr)^{-1}.
\]

### 1.6 GSM 相关符号（`docs/methods/baselines/ultrags_baseline/online_without_pose_workflow.md` §0）

- **`membership`**：形状 \((G, N_m)\)，行对应高斯；空槽为 \(-1\)。**出生**：`[id_{\mathrm{born}},-1,\ldots,-1]`。
- **共视更新**：每一步 GS / render 后，用 **`intermediates["seen_gauss_global_indices"]`**（与 §2.2 规则 3 同路径）更新该行（FIFO 满则移位），使高斯–帧从属与 workflow 例一致（如 `[0,1,-1,\ldots]`）。
- **集合语义（叙事）**：\(G_i\) 表示 membership 仅含 \(i\) 的高斯；\(G_{ijk}\) 表示与多帧有关；均为 **分析/日志** 用语，实现上以 `membership` 张量为准。
- **子集 \(\mathcal{G}_{\mathrm{sub}}\)**（启用 \(l\) 时）：最右帧为 \(i\) 时，保留 membership 与时间区间 \([i-W+1-l+1,\,i]\) 有交的高斯（闭区间，与 workflow 举例一致）；子集为**索引视图**，参数仍在全局 \(\mathcal{G}\)，梯度写回全局行。不启用 \(l\) 时可视作 \(\mathcal{G}_{\mathrm{sub}}=\mathcal{G}\)（全量）。

---

## 2. 组件设计

### 2.1 Local Window Manager（LWM）

**数学**

预算表 \(\mathcal{C}_{i,j}=C_i^{W_j}\)。记 **cfg** 中 \(r^{\mathrm{front}},r^{\mathrm{rear}}\)（与下标 \(i\) 无关，全局标量）。定义：

\[
K_i^{\mathrm{ind}}(j)=
\begin{cases}
C_i^{W_j}\, r^{\mathrm{front}}, & j=0,\; i\in\mathcal{L}_0,\\
C_i^{W_j}\, r^{\mathrm{front}}, & j\ge 1,\; i=t_{\mathrm{new}}(j),\\
0, & j\ge 1,\; i\ne t_{\mathrm{new}}(j),
\end{cases}
\]

\[
K_i^{\mathrm{joint}}(j)=
\begin{cases}
C_i^{W_j}\, r^{\mathrm{rear}}, & j=0,\; i\in\mathcal{L}_0,\\
C_i^{W_j}, & j\ge 1,\; i\in\{j,\ldots,j+W-2\},\\
C_i^{W_j}\, r^{\mathrm{rear}}, & j\ge 1,\; i=t_{\mathrm{new}}(j).
\end{cases}
\]

**代码 / 接口**

- `build_budget_table(N,T,W)`；`K_ind(i,j,r_front,r_rear)`、`K_joint(i,j,r_front,r_rear)`；维护当前 **窗索引 `j`** / 与流式帧对齐（仅 **预算与步数调度**，**不**维护任何高斯张量或 `active_mask`）。
- **去掉** `loss * get_frame_weight` 与 EMA 帧权训练强度。

---

### 2.2 Gaussian State Manager（GSM）——**仅** `docs/methods/baselines/ultrags_baseline/online_without_pose_workflow.md` §0

**职责**：回答 workflow 里的核心问题——**高斯如何与帧关联**、**如何从 \(G_{\mathrm{global}}\) 取参与计算的高斯（含后期 \(\mathcal{G}_{\mathrm{sub}}\)）**。实现上 **只依赖 `membership`**（及由它派生的子集查询），**不**与 LWM「共享」`last_seen`、`active_mask` 等；二者分工：**LWM = 窗格与步数预算；GSM = membership +（可选）子集**。

**规则（workflow §0 原文）**

1. **`membership`**，形状 \((G, N_m)\)，空槽 \(-1\)。
2. **出生**：新采样高斯写 `[id_{\mathrm{born}},-1,\ldots,-1]`。
3. **每步优化 / 每次 render 后**：用 `render(..., return_intermediates=True)` 中的 **`seen_gauss_global_indices`**（经 `online_callbacks._update_membership_from_intermediates` → `update_membership_from_render`）更新对应行（FIFO 等规则与 workflow 例一致），**保证每步结束后 membership 与几何从属一致**。
4. **\(\mathcal{G}_{\mathrm{sub}}\)**（workflow 后期、启用 \(l\)）：按 §1.6，用 **membership 与时间区间** 筛出子集；**子集是视图，参数仍在 \(G_{\mathrm{global}}\)**，写回全局。未启用 \(l\) 则全程用全量 \(\mathcal{G}\)。
5. **POSE_WASH（workflow）**：「冻结 \(G_{\mathrm{global}}\)」→ 实现为 **对全部高斯参数不反传 / 不 step GS**，仅训练 \(\Phi_p\)；**不**用 `active_mask` 表达冻结。

**代码 / 接口（建议）**

- `membership` 张量随 `GaussContainer` 点数变长/剪枝而 resize。
- `update_membership_from_render(frame_id, seen_gauss_indices)`：每 `GS_OPTIM` 步后由 `online_callbacks._update_membership_from_intermediates` 调用（与 workflow「每个 iter 后刷新」一致）。
- `gauss_indices_for_sub(i_right, W, l)` → 供 `render(..., active_global_indices=...)`；无 \(l\) 时传全量或 `None`。
- **可选**：若 slam 中 Pose 辅助需「排除本帧主高斯」，**由 `membership` 规则派生**（例如首槽为 \(i\) 的点集），**不**另建独立 `owner` 张量——若实现侧暂不做辅助支路，可仅保留主路径 + 全冻结 POSE_WASH。

---

### 2.3 Pose Estimation Model（PEM）——**本阶段不接入**

**数学**（保留）：\(\mathbf{T}_{i-1\rightarrow i}^{\mathrm{pred}}=\mathcal{M}_p(\mathbf{I}_{i-1},\mathbf{I}_i)\)，\(\mathbf{P}_i^{\mathrm{pred}}=\mathbf{T}_{i-1\rightarrow i}^{\mathrm{pred}}\mathbf{P}_{i-1}^{\mathrm{refined}}\)。

**代码**：`pose_estimator = None`；模式 C **暂不跑通**，或 coarse 用 GT/占位策略由工程另行约定。

---

### 2.4 Pose Refinement Model（PRM，模式 B/C）

**数学**

\(\mathbf{P}_i^{\mathrm{coarse}}\in\{\mathbf{P}_i^{\mathrm{noise}},\mathbf{P}_i^{\mathrm{pred}}\}\)；\(\Phi_p\) 为浅层 MLP。主路径见 slam §2.2；**Pose 辅助**（若开启）的 exclude 集 **由 `membership` 派生**，**不**依赖独立 `owner` 管线（见 §2.2）。

**代码**

- `PoseRefiner`、`slice_meta.replace_pose(...)`。
- **`POSE_WASH`**：冻结 GS，仅 \(\Phi_p\) 反传，迭代 **`num_pose_refinement`** 次（cfg）。

---

### 2.5 Density Controller（DC）——Online / Offline

**职责**：在全局池 \(\mathcal{G}\) 上做 **means 梯度统计**、**clone / split**、**弱 prune**（opacity/intensity 阈值）；Online 下还可选 **近邻合并**（体素+轴排序候选、贪心融合 loser→winner 参数、`solver.prune` 后 **`gsm.on_remap(keep_mask)`**）。

**代码**：`methods/ultrags/density_control.py` — **`UltraGSOfflineDensityController`**（Offline 按迭代 `iteration`）；**`UltraGSOnlineDensityController`**（Online 按全局 GS 步 `state.iteration`：`should_accumulate_grad_stats` / `should_densify_prune` / `prune_and_densify`）。合并核心为 **`online_pair_merge_and_fuse`**；**`merge_after_dc`** 时在 **`prune_and_densify`** 末尾调用；**`atom_merge`** 在帧末按 **`merge_every_frames`** 调 **`run_periodic_pair_merge`**（同一套合并核）。

**配置**：**`model.online.density_control`**（与 Offline 的 `model.offline.density_control` 分离）。**已移除**：独立 YAML 键 `gmm` 与文件 `gaussian_merge_manager.py`。

---

## 3. 损失函数

**帧权**：**不**使用 \(\sum_i w_i\mathcal{L}_{\mathrm{rec}}(i)\) 的 \(w_i\) 调控训练量。

**每步语义（与实现约定）**：**每一次** `GS_OPTIM` 子步（含独立段与联合段）均：

1. 用**当前步选定的帧 \(i\)** 及其 \(\mathbf{P}_i^{\mathrm{refined}}\) 做主渲染，计算 \(\mathcal{L}_{\mathrm{rec}}(i)\) 及 §3.2–3.3 等与该步相关的几何项；
2. **同时**计算当前窗口内可用的 **位姿相关项** \(\mathcal{L}_{\mathrm{prior}},\mathcal{L}_{\mathrm{smooth}},\mathcal{L}_{\mathrm{acc}}\)（模式 B/C 且对应 \(\lambda\neq0\)）——即 **每一 solver 步都带齐已启用的 pose 损失**，而非仅在段末算一次。

（Pose 辅助项 \(\mathcal{L}_{\mathrm{rec}}^{\mathrm{pose\text{-}aux}}\) 仍按 slam，可选。）

### 3.1 重建与 Pose 辅助

\[
\mathcal{L}_{\mathrm{rec}}(i)
=\lambda_{l1}\|\mathbf{I}_i-\mathbf{I}_i^{\mathrm{rendered}}\|_1
+\lambda_{ssim}\bigl(1-\mathrm{SSIM}(\mathbf{I}_i,\mathbf{I}_i^{\mathrm{rendered}})\bigr).
\]

\(\mathbf{I}_i^{\mathrm{rendered}}=\mathrm{Render}(\mathbf{P}_i^{\mathrm{refined}},\mathcal{G}_{\ast})\)。\(\mathcal{G}_{\ast}\) 由 **GSM** 给出：全量 \(G_{\mathrm{global}}\) 或 **\(\mathcal{G}_{\mathrm{sub}}\)**（§1.6，由 `membership` + \(l\) 决定），**不**与 `active_mask` 求交。

### 3.2–3.6 尺度、\(z\)-floor、时序、Pose prior、平滑、总损失

公式与 `docs/methods/baselines/ultrags_baseline/online_slam_design.md` §3.2–§3.6 相同。\(\mathcal{L}_{\mathrm{prior}},\mathcal{L}_{\mathrm{smooth}},\mathcal{L}_{\mathrm{acc}}\) 的求和范围取**当前优化窗口**（\(\mathcal{W}_t\) 或 \(\mathcal{L}_j\)，与 Pipeline 当前窗一致）。

总损失 \(\mathcal{L}\) 为 §3.1–§3.5 各项加权和（§3.6），**每步 `backward` 前**按当前帧与当前窗组装的张量计算。

---

## 4. Pipeline（窗口本位 + `docs/methods/baselines/ultrags_baseline/online_without_pose_workflow.md`）

### 4.1 窗口本位：首窗 \(W_0\) 与「窗尚未满」

**严格按 workflow §0（如 \(W=3\)）**：**不必**等 \(\{0,1,\ldots,W-1\}\) 帧全部到齐再开始。流式时间线为：

1. **仅帧 0 到达**：`CONCAT_SEED(0)` → 对帧 0 做 **独立段** \(K_0^{\mathrm{ind}}(0)\) 次 `GS_OPTIM`（无联合；此时全局只有帧 0 相关高斯）。首帧 **跳过 `POSE_WASH`**。
2. **帧 1 到达**：`POSE_WASH`（`num_pose_refinement`）→ `CONCAT_SEED(1)` → 帧 1 独立段 \(K_1^{\mathrm{ind}}(0)\) 步 …
3. **直至帧 \(W-1\) 到达并完成其独立段**后，\(\mathcal{L}_0=\{0,\ldots,W-1\}\) 在数据上齐全，再进入 **\(W_0\) 联合段**（§4.2）。

也即在 **首窗填充期**，逻辑窗格始终是 **\(W_0\)**，预算仍用 **\(C_i^{W_0}\)** 与 §2.1 的 \(K^{\mathrm{ind}}(0)\)、\(K^{\mathrm{joint}}(0)\)；**与 workflow「一帧一帧」叙事一致**，而不是「等满窗再一次性跑 \(W_0\)」。

**后续窗 \(j\ge1\)**：每推进一次，仅对新进入的最右帧做 **独立段**，再对 \(\mathcal{L}_j\) 做 **联合段**（workflow §0 后半）。

### 4.2 原子（更新）

| 原子 | 含义 |
|------|------|
| `LWM_ADVANCE` | 更新窗索引 `j` / 与流式帧对齐（仅预算语义） |
| `STREAM_DATUM` | `scene.next_stream_datum()` 取下一帧 `slice_meta`；**不**在 Scene 内播种 |
| `RESOLVE_SEED` | `_resolve_stream_new_gaussians`：按 `seed_every_frames` + `num_new` 生成待 `concat` 字典（无则跳过） |
| `GSM_PREP_RENDER` | 按当前步决定 `active_global_indices`：全量或 \(\mathcal{G}_{\mathrm{sub}}\)（§1.6，`gsm.gauss_indices_for_sub`） |
| `POSE_WASH` | **冻结 \(G_{\mathrm{global}}\)**（GS 不反传或不 step）→ **`num_pose_refinement`** 步仅 \(\Phi_p\) → 解冻；后期窗若用 \(\mathcal{G}_{\mathrm{sub}}\)，冻结仍作用于**当前参与集上的参数行**（与 workflow） |
| `CONCAT_SEED` | `run_concat_seed`：`concat` + **`gsm.on_concat`**（membership 出生帧） |
| `GS_OPTIM` | 选一帧 \(i\) → `render(..., active_global_indices=...)` → **§3 全量损失** → `backward` → **`solver.step(active_indices)`**；可选 **`dc.update_grad_stats` / `dc.prune_and_densify`** |
| `UPDATE_MEMBERSHIP` | 每 `GS_OPTIM` 步后，用 `intermediates["seen_gauss_global_indices"]` 更新 **membership** |
| `DC_MERGE_FRAME`（可选） | **`atom_merge`**：`merge_every_frames>0` 时 **`dc.run_periodic_pair_merge`** |

**已搁置**：`PEM_PREDICT`（`pose_estimator is None`）。**合并**：已由 **DC** 实现（`merge_after_dc` / `merge_every_frames`），非独立 GMM 插件。

### 4.3 联合段：一步一帧一 solver

联合段共有 \(\sum_i K_i^{\mathrm{joint}}(j)\) **子步**。**每一子步相互独立**：从优化池 \(\{(i,\text{剩余步数})\}\) 中按随机或轮询 **选一个帧 \(i\)**，用该帧当前 **\(\mathbf{P}_i^{\mathrm{refined}}\)** 与其它状态渲染；**因每步主渲染帧与位姿上下文可能变化，每步必须单独 `backward` 与 `solver.step()` 一次**，**不**在同一 `backward` 内对多帧混合渲染。

### 4.4 每窗内顺序（摘要）

- **独立段**：`POSE_WASH`（除 \(i=0\) 与模式 A）→ `CONCAT_SEED` → \(K_i^{\mathrm{ind}}\) 次 `GS_OPTIM`（每步 §3 全量含 pose）。
- **联合段**：\(\sum_i K_i^{\mathrm{joint}}\) 次子步，每子步 §4.3。
- **模式 A**：无 `POSE_WASH`；\(\mathbf{P}_i^{\mathrm{refined}}=\mathbf{P}_i\)。
- **模式 C（PEM 未接时）**：不调用 PEM；粗姿态来源需工程占位。

### 4.5 代码落地（当前实现）

- `local_window_manager.py`（预算表 + `k_ind/k_joint`）
- `gaussian_state_manager.py`（`membership` + `gauss_indices_for_sub` + `update_membership_from_render`）
- `online_callbacks.py`（`UltraGSOnlineStreamCallback` 流式循环、`next_stream_datum`、`_resolve_stream_new_gaussians`、首窗/后续窗 pipeline、`run_concat_seed`、`gs_optim_one_step`、`atom_merge`、日志与评测）
- `density_control.py`（`UltraGSOnlineDensityController` / `prune_and_densify`、`online_pair_merge_and_fuse`）
- `scene/scene.py`（`next_stream_datum()`：仅数据与 bbox，**无**流式像素播种）
- `render.py`（`active_global_indices` 子集渲染 + `intermediates["seen_gauss_global_indices"]`）
- `gauss_solver.py` / `gauss_container.py`（Online 子集 `step` 等）
- `engine/computer.py`（通用 `LossComputer/MetricComputer`，读取 `state.inputs/state.outputs`）
- `state.py`（`UltraGSOnlineState`：`stream_*` 运行时、`stream_prev_*` 已移除；**无**独立 `gmm` 字段）
- **PEM** `None`；**合并** 见 **`density_control`**（`merge_after_dc` / `merge_every_frames`）

---

## 5. 文献对照

| 来源 | 本文对应 |
|------|----------|
| `docs/methods/baselines/ultrags_baseline/online_without_pose_workflow.md` | §4 窗口本位时间线、独立/联合、首帧跳过 pose |
| `docs/methods/baselines/ultrags_baseline/online_slam_design.md` §1、§2.1–2.2、§3 | §1、§2.3–2.4、§3；slam §2.5 式「合并」在实现中归 **§2.5 DC**（`density_control.py`） |
| `docs/methods/baselines/ultrags_baseline/online_local_window_budget.md` | \(C_i^{W_j}\) |
