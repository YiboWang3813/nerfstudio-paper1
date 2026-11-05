# Ours Renderer：Tile Mapper + Alpha 合成（`plane_cond2d`）数学推导

**文档与代码同步**：`plane_cond2d_device.cuh` / `plane_cond2d.cu` / `core_tile.cu`（2026-05，含 Schur AABB、`plane_cond2d_plane_weight`、禁止 `fmax(q,0)` 再 `exp`）。

本文档描述当前 CUDA 实现中 **`mapper_kind = tile`**、**`compositing_mode = alpha`**、**`shader_weight_mode = plane_cond2d`** 的完整计算链，从世界坐标毫米到像素灰度，含前向与反向。符号与索引顺序与下列源文件一致：

| 模块 | 文件 |
|------|------|
| Tile 映射 | `src/mapper/core_tile.cu` |
| 前向着色 | `src/shader/forward.cu`, `include/shader/plane_cond2d_device.cuh` |
| 反向（plane_cond2d） | `src/shader/plane_cond2d.cu` |
| 公共几何 | `include/shader/shared.cuh`, `include/shader/slice_grid_geometry.cuh` |
| Python 入口 | `ours_renderer/autograd.py`, `src/shader/wrapper_shader.cu` |

**对照模式**：`shader_weight_mode = probe3d` 时，片元权重由 `ours_eval_gauss_contrib_at_pixel`（`shared.cuh`）给出，协方差构造与 `plane_cond2d` 不同；附录 A 简述差异。本文主体以 **`plane_cond2d` 严格切片截面模式** 为准（`debug_offline_liver_ours_plane_cond2d.yaml`）。

> **建模说明（当前实现）**
> `plane_cond2d` 在 Schur 条件平面二次型 \(q_{\mathrm{cond}}\) 之外，加入 **elev 衰减** \(\exp(-\tfrac{1}{2}\mu_z^2/S_{zz})\)，使
> \(\alpha = \mathrm{clamp}\big(o\,\exp(-\tfrac{1}{2}(q_{\mathrm{cond}}+\mu_z^2/S_{zz})),\,0,\,1\big)\) 等价于 3D 高斯密度在探头切片平面 \(z{=}0\) 处的取值（差一个归一化常数）。
> 若仅需条件投影 \(\exp(-\tfrac{1}{2}q_{\mathrm{cond}})\) 而无 elev 项，需改回旧公式（不推荐用于 slice 重建）。

---

## 0. 总览：两阶段前向 + 两阶段反向

### 0.1 前向

1. **Tile Gauss Mapper**（与 shader 无关）：为每个高斯 \(g\) 计算所覆盖的 tile 矩形、深度键，生成每个 tile 的有序高斯列表 CSR。
2. **Tile CSR Shader + Alpha 合成**：对每个像素 \(p\)，遍历其所在 tile 的高斯列表（前向到后），计算 \(\alpha_{g,p}, I_g\)，按 front-to-back alpha 合成得到 \(C_p\)。

### 0.2 反向（`plane_cond2d`）

1. **Pass-1 — Alpha 反传重放**（`kernel_plane_cond2d_alpha_back_tile_csr`）：逐像素**复现**前向合成顺序，将 \(\partial L/\partial C_p\) 分发为对每个 \((g,p)\) 的 \(\partial L/\partial \alpha_{g,p}\)、\(\partial L/\partial I_g\)，再经 `ours_plane_cond2d_bwd_wire` **原子累加**到 per-Gaussian workspace。
2. **Pass-2 — Finalize**（`kernel_plane_cond2d_finalize_gauss`）：读取 workspace，经 Schur / 2×2 逆 / 世界协方差链式法则，写出 \(\partial L/\partial \boldsymbol{\mu}_g, \mathbf{s}_g, \mathbf{q}_g, o_g, \iota_g, \mathbf{T}\)。

### 0.3 调用链（`plane_cond2d` + tile + alpha）

```text
Python render_projected_slice
  → ProjectedSliceForwardCUDA (wrapper_shader.cu)
       → OursTileGaussMapperCUDA (core_tile.cu)     # shader_weight_mode 传入 mapper
       → kernel_ours_tile_csr_forward (forward.cu)
            → ours_plane_cond2d_eval_at_pixel (plane_cond2d_device.cuh)
  → ProjectedSliceBackwardCUDA
       → RenderOursSliceBackwardPlaneCond2D_Impl (plane_cond2d.cu)
            → kernel_plane_cond2d_alpha_back_tile_csr   # Pass-1 重放
            → kernel_plane_cond2d_finalize_gauss        # Pass-2
```

**`plane_cond2d_device` 命名空间（`plane_cond2d_device.cuh`）**

| 函数 | 用途 |
|------|------|
| `schur_conditional_plane` | \(\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_2,\mathbf{I}_2\)；逆失败 → `false` |
| `plane_cond2d_schur_aabb_radii_mm` | Schur 轴对齐盒半宽（mapper + shader 共用） |
| `plane_cond2d_plane_weight` | \(q\ge 0\) 时 \(\exp(-q/2)\)，否则 0 |
| `plane_cond2d_mapper_footprint_mm` | tile/pixel mapper 中心与 mm 半径 |
| `ours_plane_cond2d_eval_at_pixel` | 片元 \(\alpha,\iota\) |
| `inv2_sym_bwd` | \(\bar{\mathbf{I}}_2\to\bar{\boldsymbol{\Sigma}}_2\)（`inv_det` 饱和时置 0） |

> **不可微分/离散步骤**（反向为 0 或子梯度）：tile 矩形硬裁剪、elev 走廊剔除、Schur AABB（`plane_cond2d`）/椭球 AABB（`probe3d`）、\(q<0\) 时平面权重为 0、\(\alpha < 1/255\) 跳过、\(T<10^{-4}\) 提前终止、Schur/行列式失败分支、`fmax(scale,10^{-6})`、四元数有限差分、梯度 `sanitize` 截断。

---

## 1. 符号与数据布局

### 1.1 高斯参数（每个 \(g\)）

| 符号 | 代码 | 形状 | 说明 |
|------|------|------|------|
| \(\boldsymbol{\mu}^w_g\) | `gauss_means[g*3+0:3]` | \(\mathbb{R}^3\) | 世界坐标中心，**mm** |
| \(\mathbf{s}_g\) | `gauss_scales[g*3+k]` | \(\mathbb{R}^3_{+}\) | 三轴 scale；代码用 \(s_k=\max(s_k,10^{-6})\) |
| \(\mathbf{q}_g\) | `gauss_quats[g*4+…]` | \((w,x,y,z)\) | **仓库约定**：`q[0]=w, q[1]=x, q[2]=y, q[3]=z` |
| \(o_g\) | `gauss_opacities[g]` | 标量 | 不透明度乘子 |
| \(\iota_g\) | `gauss_intensities[g]` | 标量 | 强度（合成时线性加权） |

### 1.2 位姿 `pose_probe_to_world`（4×4，行主序展平）

代码按**列向量**读旋转列与平移：

\[
\mathbf{T} = \begin{bmatrix}
\mathbf{r}_{\mathrm{lat}} & \mathbf{r}_{\mathrm{ax}} & \mathbf{r}_{\mathrm{ele}} & \mathbf{t} \\
0 & 0 & 0 & 1
\end{bmatrix},
\quad
\begin{aligned}
\mathbf{r}_{\mathrm{lat}} &= (\texttt{pose[0]},\texttt{pose[4]},\texttt{pose[8]}),\\
\mathbf{r}_{\mathrm{ax}}  &= (\texttt{pose[1]},\texttt{pose[5]},\texttt{pose[9]}),\\
\mathbf{r}_{\mathrm{ele}} &= (\texttt{pose[2]},\texttt{pose[6]},\texttt{pose[10]}),\\
\mathbf{t} &= (\texttt{pose[3]},\texttt{pose[7]},\texttt{pose[11]}).
\end{aligned}
\]

归一化探头轴（代码 `ours_normalize3`）：

\[
\hat{\mathbf{a}}_{\mathrm{lat}} = \frac{\mathbf{r}_{\mathrm{lat}}}{\|\mathbf{r}_{\mathrm{lat}}\|},\quad
\hat{\mathbf{a}}_{\mathrm{ax}} = \frac{\mathbf{r}_{\mathrm{ax}}}{\|\mathbf{r}_{\mathrm{ax}}\|},\quad
\hat{\mathbf{a}}_{\mathrm{ele}} = \frac{\mathbf{r}_{\mathrm{ele}}}{\|\mathbf{r}_{\mathrm{ele}}\|}.
\]

世界点相对原点的探头坐标（**未**再减 elev，elev 在 \(z\) 分量）：

\[
\mathbf{d} = \boldsymbol{\mu}^w_g - \mathbf{t},\quad
\mu_{\mathrm{lat}} = \mathbf{d}\!\cdot\!\hat{\mathbf{a}}_{\mathrm{lat}},\;
\mu_{\mathrm{ax}} = \mathbf{d}\!\cdot\!\hat{\mathbf{a}}_{\mathrm{ax}},\;
\mu_z = \mathbf{d}\!\cdot\!\hat{\mathbf{a}}_{\mathrm{ele}}.
\]

### 1.3 像素与切片网格

- 像素索引：`col = x`，`row = y`，线性下标 `p = y * W + x`。
- **Tile**：`tile_size_px = T_s`，`tx = x / T_s`，`ty = y / T_s`，`tile_id = ty * T_w + tx`。
- **平面坐标（mm）** — `sg_linear_pixel_to_plane_mm` / `slice_grid_pack`：

\[
\xi_{\mathrm{lat}} = (col - w_{\mathrm{off}})\,\Delta_{\mathrm{lat}},\qquad
\xi_{\mathrm{ax}} = row\,\Delta_{\mathrm{ax}} + z_{\mathrm{near}},
\]

其中 \(w_{\mathrm{off}}=\texttt{pack[SG\_LAT\_COL\_OFFSET]}\)（`build_slice_grid_pack`：若 `center_lateral` 则为 \((W-1)/2\)，否则 0），\(\Delta_{\mathrm{lat}}=\texttt{SG\_DELTA\_LAT\_MM}\)，\(\Delta_{\mathrm{ax}}=\texttt{SG\_DELTA\_AX\_MM}\)，\(z_{\mathrm{near}}=\texttt{SG\_NEAR\_MM}\)。

**`slice_grid_pack` 布局**（`SLICE_GRID_PACK_LEN=16`，见 `slice_grid_geometry.cuh` / `grid_geometry.py`）：

| 下标 | 宏 | 含义 |
|------|-----|------|
| 0 | `SG_PROBE_TYPE` | 0=linear，1=convex（`plane_cond2d` 仅 linear） |
| 1–2 | `SG_H`, `SG_W` | 图像高宽 |
| 3–5 | `SG_DELTA_LAT_MM` 等 | spacing / near |
| 6 | `SG_LAT_COL_OFFSET` | \(w_{\mathrm{off}}\) |
| 7 | `SG_CENTER_LATERAL` | 标志位（仅 Python 侧） |

> **Mapper vs shader 横向原点**：tile mapper 用 `u_pix = μ/Δ_lat + (W-1)/2`（硬编码）；`plane_cond2d` shader 用 `pack[SG_LAT_COL_OFFSET]`。默认 `center_lateral=True` 时二者一致；若关闭居中，须保证 pack 与 mapper 同步（当前 mapper 未读 pack）。

> **与 probe3d 的坐标差**：`probe3d` 在 shader 内用 \(x_c=(col-\frac{W-1}{2})\Delta_{\mathrm{lat}}\)（硬编码），`plane_cond2d` 用 pack 的 \(w_{\mathrm{off}}\)；默认 liver 配置下二者相同。

### 1.4 旋转矩阵 \(\mathbf{R}_g\)

`ours_quat_to_rot`：先对 \(\mathbf{q}_g\) 归一化，再按 Hamilton 四元数 \((w,x,y,z)\) 填 **行主序** 9 元数组 `Rg[i*3+j] = (\mathbf{R}_g)_{ij}`。

世界系协方差（代码 `mat3_sigma_world`）：

\[
\boldsymbol{\Sigma}^w_g = \mathbf{R}_g \,\mathrm{diag}(s_0^2,s_1^2,s_2^2)\,\mathbf{R}_g^{\mathsf T}.
\]

实现等价于列向量 \(\mathbf{c}_k\)（`rot_col(Rg,k)` = \(\mathbf{R}_g\) 第 \(k\) 列）：

\[
\boldsymbol{\Sigma}^w_g = s_0^2 \mathbf{c}_0\mathbf{c}_0^{\mathsf T} + s_1^2 \mathbf{c}_1\mathbf{c}_1^{\mathsf T} + s_2^2 \mathbf{c}_2\mathbf{c}_2^{\mathsf T}.
\]

### 1.5 探头系 \(3\times3\) 协方差 \(\mathbf{S}_c\)

`build_sigma_c_probe`：令 \(\mathbf{A}=[\hat{\mathbf{a}}_{\mathrm{lat}},\hat{\mathbf{a}}_{\mathrm{ax}},\hat{\mathbf{a}}_{\mathrm{ele}}]^{\mathsf T}\)（行为主轴），则

\[
\mathbf{S}_c = \mathbf{A}\,\boldsymbol{\Sigma}^w_g\,\mathbf{A}^{\mathsf T},
\]

按行主序存入 `Sc[9]`，索引 \((i,j)\in\{\mathrm{lat},\mathrm{ax},\mathrm{ele}\}^2\)：

\[
\begin{aligned}
S_{00}&=S_{\mathrm{lat,lat}}, & S_{01}&=S_{\mathrm{lat,ax}}, & S_{02}&=S_{\mathrm{lat,ele}},\\
S_{11}&=S_{\mathrm{ax,ax}}, & S_{12}&=S_{\mathrm{ax,ele}}, & S_{22}&=S_{\mathrm{ele,ele}}.
\end{aligned}
\]

**对角 jitter**（与 probe3d 一致，`kProbeCovJitter=10^{-6}`）：

\[
S_{00}\leftarrow S_{00}+\varepsilon,\quad S_{11}\leftarrow S_{11}+\varepsilon,\quad S_{22}\leftarrow S_{22}+\varepsilon,\quad \varepsilon=10^{-6}.
\]

---

## 2. 阶段 I：Tile Gauss Mapper（前向，不可微）

实现：`kernel_ours_count_tile_rects` → emit keys → CUB sort → `kernel_ours_identify_tile_ranges`。

对每个高斯 \(g\)：

### 2.1 Elev 走廊剔除

用 probe3d 风格投影长度（**非** \(\boldsymbol{\Sigma}^w\)）：

\[
e_{k,0} = s_0\, (\mathbf{R}_g)_{0k}^{\mathrm{proj,lat}},\quad \ldots
\]

（代码中 `ex0,ex1,ex2` 为 scale 乘旋转列在 \(\hat{\mathbf{a}}_{\mathrm{lat}}\) 上的投影，与 `shared.cuh` 相同。）

\[
r_{\mathrm{ele}} = k_\sigma \sqrt{e_0^2+e_1^2+e_2^2},\quad
\text{若 } |\mu_z| \ge h_{\mathrm{elev}} + h_{\mathrm{slab}} + r_{\mathrm{ele}} \Rightarrow \text{该高斯不映射任何 tile}.
\]

### 2.2 像素空间中心与半径

**`probe3d` / 默认 mapper**：中心 \((\mu_{\mathrm{lat}},\mu_{\mathrm{ax}})\)，半径 `ours_lat_ax_radii_mm`（椭球 L2 投影 × \(k_\sigma\)）。

**`plane_cond2d` mapper**（`plane_cond2d_mapper_footprint_mm`，与 shader 共用 Schur）：

1. 与 §3 相同链路得 \(\boldsymbol{\mu}_2,\sigma^2_{00},\sigma^2_{01},\sigma^2_{11}\)（Schur 失败 → 该高斯 footprint 失败，tile 矩形为空）。
2. 椭球半径 \(r^{\mathrm{ell}}_{\mathrm{lat}}, r^{\mathrm{ell}}_{\mathrm{ax}}\)（`ours_lat_ax_radii_mm`）。
3. Schur 盒半宽（`plane_cond2d_schur_aabb_radii_mm`）：

\[
r^{\mathrm{schur}}_{\mathrm{lat}} = k_\sigma\sqrt{\sigma^2_{00}+|\sigma^2_{01}|},\quad
r^{\mathrm{schur}}_{\mathrm{ax}} = k_\sigma\sqrt{\sigma^2_{11}+|\sigma^2_{01}|}.
\]

4. 中心 \(\mu^{\mathrm{map}}=\boldsymbol{\mu}_2\)，mm 半径（**比 shader AABB 更保守**，含无条件中心偏移）：

\[
r_{\mathrm{lat}} = \max\!\big(r^{\mathrm{ell}}_{\mathrm{lat}}+|\mu_{\mathrm{lat}}-\mu_{2,\mathrm{lat}}|,\; r^{\mathrm{schur}}_{\mathrm{lat}}\big),\quad
r_{\mathrm{ax}} \text{ 同理}.
\]

**像素索引**（tile mapper，`core_tile.cu`）：

\[
u_{\mathrm{pix}} = \frac{\mu^{\mathrm{map}}_{\mathrm{lat}}}{\Delta_{\mathrm{lat}}} + \frac{W-1}{2},\qquad
v_{\mathrm{pix}} = \frac{\mu^{\mathrm{map}}_{\mathrm{ax}} - z_{\mathrm{near}}}{\Delta_{\mathrm{ax}}}.
\]

\[
r^{\mathrm{pix}}_{\mathrm{lat}} = \max\!\left(1,\frac{r_{\mathrm{lat}}}{\Delta_{\mathrm{lat}}}\right),\quad
r^{\mathrm{pix}}_{\mathrm{ax}} = \max\!\left(1,\frac{r_{\mathrm{ax}}}{\Delta_{\mathrm{ax}}}\right).
\]

Tile 索引范围：

\[
\begin{aligned}
t_x^{\min} &= \left\lfloor \frac{u_{\mathrm{pix}} - r^{\mathrm{pix}}_{\mathrm{lat}}}{T_s} \right\rfloor,\quad
t_x^{\max} = \left\lceil \frac{u_{\mathrm{pix}} + r^{\mathrm{pix}}_{\mathrm{lat}}}{T_s} \right\rceil,\\
t_y^{\min,\max} &\text{ 同理用 } v_{\mathrm{pix}}, r^{\mathrm{pix}}_{\mathrm{ax}}.
\end{aligned}
\]

裁剪到 \([0,T_w)\times[0,T_h)\)。矩形为空则 `tiles_touched_count=0`。

### 2.3 排序键与 CSR

对每个 \((g,\text{tile})\) 实例发键：

\[
\text{key} = (\texttt{tile\_id} \ll 32) \;|\; \texttt{depth\_key},
\quad
\texttt{depth\_key} =
\begin{cases}
\left\lfloor (v_{\mathrm{pix}} + 500)\times 1000 \right\rfloor & \texttt{sort\_by\_v}=\text{true},\\
0 & \text{否则}.
\end{cases}
\]

按 key 升序排序后，每个 tile 得到 CSR：`tile_range_start[tid]`, `tile_range_end[tid]`，列表 `sorted_gauss_ids[]` 为 **从近到远（\(v\) 小到大）** 的高斯 id。

**输出**（`ProjectedSliceForwardCUDA`）：`gauss_rects[g]=(min_tx,min_ty,max_tx,max_ty)`，CSR 写入 `csr0=sorted_gauss_ids`, `csr1=tile_range_start`, `csr2=tile_range_end`。

---

## 3. 阶段 II-A：`plane_cond2d` 片元权重（前向）

函数：`ours_plane_cond2d_eval_at_pixel(col,row,…)`（`plane_cond2d_device.cuh`）。

**单像素流程**：

```text
slice_grid_pack → (ξ_lat, ξ_ax)
  → Sc, Schur → μ2, Σ2, I2
  → Schur AABB 剔除?
  → q = δ^T I2 δ  →  plane_cond2d_plane_weight(q)
  → × exp(-μ_z²/(2 S_zz))  →  × o_g  →  clamp α∈[0,1]
```

### 3.1 Schur 补（条件在 elev 上的 2D 高斯）

在 \(\mathbf{S}_c\) 上划块（lat/ax = “\(p\)”，ele = “\(z\)”）：

\[
\mathbf{S}_{pp}=\begin{bmatrix}S_{00}&S_{01}\\S_{01}&S_{11}\end{bmatrix},\quad
\mathbf{S}_{pz}=\begin{bmatrix}S_{\mathrm{lat,ele}}\\S_{\mathrm{ax,ele}}\end{bmatrix}
=\begin{bmatrix}\texttt{Sc[2]}\\\texttt{Sc[5]}\end{bmatrix},\quad
S_{zz}=S_{\mathrm{ele,ele}}=\texttt{Sc[8]}.
\]

**条件均值**（代码 `schur_conditional_plane`）：

\[
\boldsymbol{\mu}_{2} =
\begin{bmatrix}\mu_{2,\mathrm{lat}}\\ \mu_{2,\mathrm{ax}}\end{bmatrix}
=
\begin{bmatrix}\mu_{\mathrm{lat}}\\ \mu_{\mathrm{ax}}\end{bmatrix}
- \frac{\mu_z}{S_{zz}}\,\mathbf{S}_{pz}.
\]

**条件协方差**：

\[
\boldsymbol{\Sigma}_{2} =
\begin{bmatrix}\sigma^2_{00}&\sigma^2_{01}\\\sigma^2_{01}&\sigma^2_{11}\end{bmatrix}
=
\mathbf{S}_{pp} - \frac{1}{S_{zz}}\mathbf{S}_{pz}\mathbf{S}_{pz}^{\mathsf T}.
\]

求逆 \(\boldsymbol{\Sigma}_2^{-1}\)（`invert_sigma2_sym`）：\(\det_2 = \sigma^2_{00}\sigma^2_{11}-(\sigma^2_{01})^2\)。若 \(\det_2 < 10^{-12}\) 或非有限 → **返回** \(\alpha=0, I=0\)。否则

\[
\mathrm{inv\_det} = \min\!\left(\frac{1}{\det_2},\,10^{8}\right),\quad
\boldsymbol{\Sigma}_2^{-1} = \mathrm{inv\_det}\begin{bmatrix}\sigma^2_{11}&-\sigma^2_{01}\\-\sigma^2_{01}&\sigma^2_{00}\end{bmatrix}.
\]

记 \(\mathbf{I}_2 = \boldsymbol{\Sigma}_2^{-1}\)，元素 \(i_{00},i_{01},i_{11}\)。

### 3.2 Schur AABB 剔除（在 \(\mu_2\) 上）

调用 `plane_cond2d_schur_aabb_radii_mm`（与 mapper 中 \(r^{\mathrm{schur}}\) 相同，**不含** \(|\mu-\mu_2|\) pad）：

\[
r_{\mathrm{lat}} = k_\sigma\sqrt{\sigma^2_{00}+|\sigma^2_{01}|},\quad
r_{\mathrm{ax}} = k_\sigma\sqrt{\sigma^2_{11}+|\sigma^2_{01}|}.
\]

若 \(|\xi_{\mathrm{lat}}-\mu_{2,\mathrm{lat}}|>r_{\mathrm{lat}}\) 或 \(|\xi_{\mathrm{ax}}-\mu_{2,\mathrm{ax}}|>r_{\mathrm{ax}}\) → 直接 `return`（\(\alpha=0\)）。硬边界；反向不穿过 AABB。

### 3.3 二次型与 \(\alpha\)

\[
\boldsymbol{\delta} =
\begin{bmatrix}\xi_{\mathrm{lat}}-\mu_{2,\mathrm{lat}}\\ \xi_{\mathrm{ax}}-\mu_{2,\mathrm{ax}}\end{bmatrix},\quad
q = \boldsymbol{\delta}^{\mathsf T}\mathbf{I}_2\boldsymbol{\delta}.
\]

**平面权重**（`plane_cond2d_plane_weight`，代码实现）：

\[
\mathrm{base}_{\mathrm{plane}} =
\begin{cases}
\exp(-\tfrac{1}{2} q), & q\ge 0 \;\wedge\; \text{finite}\\
0, & q<0 \;\vee\; \text{非有限}
\end{cases}
\]

> **实现陷阱（已修复）**：勿写 `q_+=\max(q,0)` 再 `exp(-q_+/2)`。当 \(q<0\) 时会得到 \(\exp(0)=1\)，在去掉 AABB 后会产生**大片不透明多边形伪影**。

\[
q_{\mathrm{elev}} = \frac{\mu_z^2}{S_{zz}},\quad
\mathrm{base}_{\mathrm{elev}} = \exp(-\tfrac{1}{2} q_{\mathrm{elev}}),
\]
\[
\mathrm{base} = \mathrm{base}_{\mathrm{plane}}\cdot \mathrm{base}_{\mathrm{elev}},\quad
\alpha_{g,p} = \mathrm{clamp}(o_g \cdot \mathrm{base},\,0,\,1),\quad
I_g = \iota_g.
\]

若 \(\alpha_{g,p} < 1/255\)（`OURS_ALPHA_MIN`），前向合成中 **跳过** 该层。

---

## 4. 阶段 II-B：Alpha 合成（前向）

`kernel_ours_tile_csr_forward`，`compositing_mode = 1`（alpha）。

对像素 \(p\)，令 tile 列表 \(\mathcal{L}_p = \{g_{p,1},\ldots,g_{p,K}\}\)（CSR 顺序，近→远）。初始化 \(T\leftarrow 1\)，\(C\leftarrow 0\)。

对每个列表中、且 \(\alpha_{g,p}\ge 1/255\) 的层：

\[
C \leftarrow C + I_g\,\alpha_{g,p}\,T,\qquad
T \leftarrow T\,(1-\alpha_{g,p}),
\]

若 \(T < 10^{-4}\) 则 **break**。

最终像素值（训练配置 `fixed_bg_color = b`）：

\[
\boxed{C_p = C + T\,b.}
\]

---

## 5. 反向 Pass-1：Alpha 合成链（与顺序重放）

`kernel_plane_cond2d_alpha_back_tile_csr`：对每个像素，**再次前向**收集最多 `OURS_MAX_ALPHA_LAYERS=256` 层的 \((\alpha,I,T,g)\)，再 **从远到近** 反传。

记 \(\bar{C}_p = \partial L/\partial C_p\)（`grad_pixel_colors[p]`）。初始化 \(\bar{T}\leftarrow \bar{C}_p \cdot b\)（背景项）。

对层 \(k=K,\ldots,1\)（存储的 `ca[kk], cI[kk], cT[kk]=T_before_k`）：

\[
\begin{aligned}
\overline{I_g} &= \bar{C}_p \cdot \alpha_{g,p} \cdot T_{\mathrm{before}},\\
\overline{\alpha_{g,p}} &= \bar{C}_p \cdot I_g \cdot T_{\mathrm{before}} + \bar{T}\cdot(-T_{\mathrm{before}}),\\
\bar{T} &\leftarrow \bar{C}_p \cdot I_g \cdot \alpha_{g,p} + \bar{T}\cdot(1-\alpha_{g,p}).
\end{aligned}
\]

这与标准 front-to-back alpha 合成导数一致（`backward.cu` 中 probe3d 同式）。

---

## 6. 反向 Pass-1b：`ours_plane_cond2d_bwd_wire`（累加 workspace）

对每层调用 `ours_plane_cond2d_bwd_fill` 重算 \(\boldsymbol{\mu}_2, \mathbf{I}_2, o_g, I_g\)（与前向相同分支；Schur 失败则 **跳过** 该层 wire）。

记 \(\bar{\alpha}=\overline{\alpha_{g,p}}\)，\(\bar{I}=\overline{I_g}\)，\(\boldsymbol{\delta}\) 同前向，\(\mathrm{base}_{\mathrm{plane}}=0\) 若 \(q<0\) 否则 \(\exp(-\tfrac{1}{2}q)\)，\(\mathrm{base}_{\mathrm{elev}}=\exp(-\tfrac{1}{2}\mu_z^2/S_{zz})\)，\(\mathrm{base}=\mathrm{base}_{\mathrm{plane}}\mathrm{base}_{\mathrm{elev}}\)。

**对 opacity / intensity**（`PLANE_COND2D_WSP_G_OPACITY` / `G_INTENSITY`）：

\[
\overline{o_g} \mathrel{+}= \bar{\alpha}\cdot \mathrm{base},\qquad
\overline{\iota_g} \mathrel{+}= \bar{I}.
\]

**平面内链**（代码 `if (quad_raw > 0.f)`；\(q=0\) 时仅 elev/opacity 有梯度；\(\bar{\mathrm{base}}_{\mathrm{tot}}=\bar{\alpha}\cdot o_g\)）：

\[
\overline{q} = \bar{\mathrm{base}}_{\mathrm{tot}}\cdot \mathrm{base}_{\mathrm{elev}}\cdot\left(-\tfrac{1}{2}\mathrm{base}_{\mathrm{plane}}\right),\quad
\boxed{\bar{\boldsymbol{\mu}}_2 = -2\bar{q}\,\mathbf{I}_2\boldsymbol{\delta}},\quad
\bar{\mathbf{I}}_2 \text{ 累加项同 } \overline{q}\,\delta_i\delta_j \text{（见代码）}.
\]

**elev 链**：

\[
\bar{\mu}_z \mathrel{+}= \bar{\mathrm{base}}_{\mathrm{tot}}\cdot \mathrm{base}_{\mathrm{elev}}\cdot\left(-\frac{\mu_z}{S_{zz}}\right),\quad
\bar{S}_{zz} \mathrel{+}= \bar{\mathrm{base}}_{\mathrm{tot}}\cdot \mathrm{base}_{\mathrm{elev}}\cdot\frac{q_{\mathrm{elev}}}{S_{zz}}.
\]

Workspace 布局（`PLANE_COND2D_WSP_STRIDE=14`）：

| 下标 | 宏 | 含义 |
|------|-----|------|
| 0 | `G_MU2_LAT` | \(\bar{\mu}_{2,\mathrm{lat}}\) |
| 1 | `G_MU2_AX` | \(\bar{\mu}_{2,\mathrm{ax}}\) |
| 2 | `G_MU_Z` | elev 对 \(\mu_z\) 的直接累加 |
| 3,4,6 | `G_I00,I01,I11` | \(\bar{\mathbf{I}}_2\) |
| 5 | `G_SZZ` | elev 对 \(S_{zz}\) 的直接累加 |
| 7–11 | — | 保留未用 |
| 12 | `G_OPACITY` | \(\bar{o}_g\) |
| 13 | `G_INTENSITY` | \(\bar{\iota}_g\) |

> **子梯度**：\(q\le 0\) 时平面内梯度为 0；`invert_sigma2_inv_det_clamped` 为真时 `inv2_sym_bwd` 输出 0；\(\alpha\) 在 1 处 clamp 未做饱和子梯度（实现上仅前向 `fminf`）。

> **未反传**：Schur/probe3d AABB 硬边界、\(q<0\) 平面分支、\(\alpha<1/255\) 截断、tile 列表 membership、\(T\) 早停 — 对 \(\boldsymbol{\mu},\mathbf{s},\mathbf{q}\) 梯度为 0。

---

## 7. 反向 Pass-2：`kernel_plane_cond2d_finalize_gauss`

仅当 `gauss_rects[g]` 非空时处理。读取

\[
\bar{\boldsymbol{\mu}}_2 = (\texttt{wrow[0]},\texttt{wrow[1]}),\;
\bar{\mathbf{I}}_2 = (\texttt{wrow[3]},\texttt{wrow[4]},\texttt{wrow[6]}),\;
\bar{o}_g=\texttt{wrow[12]},\;\bar{\iota}_g=\texttt{wrow[13]}.
\]

### 7.1 \(\bar{\mathbf{I}}_2 \to \bar{\boldsymbol{\Sigma}}_2\)（`inv2_sym_bwd`）

对 \(\mathbf{S}_2=\boldsymbol{\Sigma}_2\)，\(\mathbf{I}_2=\mathbf{S}_2^{-1}\)，代码实现

\[
\bar{\mathbf{S}}_2 = -\mathbf{I}_2\,\bar{\mathbf{I}}_2\,\mathbf{I}_2
\]

（对称 2×2，非对角元在 `gb` 上乘以 2 以匹配存储），得到 \(\bar{\sigma}^2_{00},\bar{\sigma}^2_{01},\bar{\sigma}^2_{11}\)。

### 7.2 Schur 反传（`schur_bwd`）

记 \(s_{zz}=S_{22}\)，\(s_{pz,\mathrm{lat}}=S_{02}\)，\(s_{pz,\mathrm{ax}}=S_{05}\)，\(inv_{zz}=1/s_{zz}\)。

由

\[
\mu_{2,\mathrm{lat}} = \mu_{\mathrm{lat}} - s_{pz,\mathrm{lat}}\,inv_{zz}\,\mu_z,\quad
\sigma^2_{00} = S_{00} - inv_{zz}\,s_{pz,\mathrm{lat}}^2,\;\ldots
\]

Finalize 先将 `wrow[2]`,`wrow[5]` 并入 Schur 输入，再 `schur_bwd`。代码累积 \(\bar{\boldsymbol{\mu}}_{\mathrm{probe}}=(\bar{\mu}_{\mathrm{lat}},\bar{\mu}_{\mathrm{ax}},\bar{\mu}_z)\) 与 \(\bar{\mathbf{S}}_c\)（9 元）。与解析式一致，例如：

\[
\bar{\mu}_{\mathrm{lat}} \mathrel{+}= \bar{\mu}_{2,\mathrm{lat}},\quad
\bar{\mu}_z \mathrel{+}= \bar{\mu}_{2,\mathrm{lat}}(-s_{pz,\mathrm{lat}} inv_{zz}) + \bar{\mu}_{2,\mathrm{ax}}(-s_{pz,\mathrm{ax}} inv_{zz}),
\]

\[
\bar{S}_{00}\mathrel{+}=\bar{\sigma}^2_{00},\;\bar{S}_{01}\mathrel{+}=\bar{\sigma}^2_{01},\;\bar{S}_{11}\mathrel{+}=\bar{\sigma}^2_{11},
\]

以及对 \(S_{02},S_{05},S_{22}\) 的链式项（见 `plane_cond2d.cu` `schur_bwd`）。

### 7.3 \(\bar{\mathbf{S}}_c \to \bar{\boldsymbol{\Sigma}}^w\)（`plane_cond2d_sc_to_world_grads`）

\[
\bar{\boldsymbol{\Sigma}}^w = \sum_{i,j\in\{\mathrm{lat,ax,ele}\}} \bar{S}_{ij}\,\hat{\mathbf{a}}_i\hat{\mathbf{a}}_j^{\mathsf T}.
\]

### 7.4 \(\bar{\boldsymbol{\Sigma}}^w \to \bar{\mathbf{s}}, \bar{\mathbf{R}}_g\)（`sigma_w_to_scale_rot_grads`）

对 \(k=0,1,2\)，\(\mathbf{c}_k\) 为 \(\mathbf{R}_g\) 第 \(k\) 列：

\[
\bar{s}_k = 2 s_k\, \mathbf{c}_k^{\mathsf T}\bar{\boldsymbol{\Sigma}}^w \mathbf{c}_k,\qquad
\overline{\mathbf{R}}_{:,k} \mathrel{+}= 2 s_k^2\, \bar{\boldsymbol{\Sigma}}^w \mathbf{c}_k.
\]

### 7.5 \(\bar{\mathbf{R}}_g \to \bar{\mathbf{q}}_g\)

`grad_quat_finite_diff`：对 \(w,x,y,z\) 四维中心差分，\(\varepsilon=10^{-3}\)，近似 \(\bar{\mathbf{q}}\)（**非**解析旋转导数）。

### 7.6 \(\bar{\boldsymbol{\mu}}_{\mathrm{probe}} \to \bar{\boldsymbol{\mu}}^w, \bar{\mathbf{t}}\)

\[
\bar{\boldsymbol{\mu}}^w = \bar{\mu}_{\mathrm{lat}}\hat{\mathbf{a}}_{\mathrm{lat}} + \bar{\mu}_{\mathrm{ax}}\hat{\mathbf{a}}_{\mathrm{ax}} + \bar{\mu}_z\hat{\mathbf{a}}_{\mathrm{ele}},\quad
\bar{\mathbf{t}} = -\bar{\boldsymbol{\mu}}^w \;\;(\texttt{atomicAdd}~\texttt{grad\_pose[3,7,11]}).
\]

### 7.7 \(\bar{\mathbf{S}}_c \to \bar{\hat{\mathbf{a}}}_i\)（`plane_cond2d_sc_to_axis_grads`）

\[
\overline{\hat{\mathbf{a}}}_i \mathrel{+}= 2\sum_j \bar{S}_{ij}\,\boldsymbol{\Sigma}^w_g\,\hat{\mathbf{a}}_j.
\]

与 \(\bar{\boldsymbol{\mu}}_{\mathrm{probe}}\) 对轴的导数合并后，经 `normalize3_backward` 传到 \(\mathbf{r}_{\mathrm{lat}},\mathbf{r}_{\mathrm{ax}},\mathbf{r}_{\mathrm{ele}}\)（`grad_pose` 旋转列）。

### 7.8 梯度截断

所有写出梯度经 `plane_cond2d_sanitize_grad`：非有限 → 0；\(|\cdot|>10^3\) → \(\mathrm{sign}\times 10^3\)。

Schur 在 finalize 再次失败时：仅写入 \(\bar{o}_g,\bar{\iota}_g\)，其余为 0。

---

## 8. 端到端前向公式（单像素、plane_cond2d + tile + alpha）

合并 §2–§4，对像素 \((col,row)\)：

\[
\boxed{
C_p = \sum_{k=1}^{K} \iota_{g_k}\,\alpha_{g_k,p}\, T_{k-1} + T_K\, b,
}
\]

其中 \(T_0=1\)，\(T_k = T_{k-1}(1-\alpha_{g_k,p})\)，\(\alpha_{g,p}\) 由 §3.2–3.3 的 Schur AABB + \(q\ge 0\) 的平面权重与 elev 衰减给出，\(g_k\) 为 tile CSR 中第 \(k\) 个高斯且通过所有剔除。

---

## 9. Python / Autograd 接口对应

`render_projected_slice` → `_ProjectedSliceCudaFunction`（`autograd.py`，`meta` 长度 16）：

| `meta[i]` | 字段 | liver `plane_cond2d` 典型值 |
|-----------|------|---------------------------|
| 0,1 | H, W | 数据集 |
| 2–4 | \(\Delta_{\mathrm{lat}},\Delta_{\mathrm{ax}}, z_{\mathrm{near}}\) | scene meta |
| 5 | `tile_size_px` | 16 |
| 6–7 | elev 走廊 / slab | 0.228225 / 0 |
| 8 | `k_sigma` | 2.0 |
| 9 | `mapper_sort_by_v` | 1 |
| 10 | mapper：0=tile，1=pixel | 0 |
| 11–12 | pixel mapper 桶数 | 默认 0 |
| 13 | compositing：0=add，1=alpha | 1 |
| 14 | `fixed_bg_color` | 0 |
| 15 | shader：0=probe3d，1=plane_cond2d | 1 |

- 另传入 `slice_grid_geometry`（`build_slice_grid_pack`）供 `plane_cond2d` shader。
- 前向返回 `(colors_hw, gauss_rects, csr0, csr1, csr2)`。
- 反向：`ProjectedSliceBackwardCUDA` → `RenderOursSliceBackwardPlaneCond2D_Impl`（`plane_cond2d.cu`）；workspace 大小 `n_gauss × PLANE_COND2D_WSP_STRIDE`。

---

## 附录 A：`probe3d` 片元权重（`shader_weight_mode=0`）

函数 `ours_eval_gauss_contrib_at_pixel`：

**平面坐标**（无 `slice_grid_pack`）：

\[
x_c = (col - \tfrac{W-1}{2})\Delta_{\mathrm{lat}},\quad
y_c = row\,\Delta_{\mathrm{ax}} + z_{\mathrm{near}},\quad
z_c = 0.
\]

**协方差**（对角化投影，**不是** \(\mathbf{R}\mathrm{diag}(s^2)\mathbf{R}^{\mathsf T}\)）：

\[
c_{ab} = \sum_{k=0}^{2} e_{k,a}\,e_{k,b},\quad a,b\in\{\mathrm{lat,ax,ele}\},
\]

\(e_{k,a}\) 为代码中 `ex0..ez2`；再 \(c_{00},c_{11},c_{22}\mathrel{+}=10^{-6}\)，求 \(\mathbf{C}^{-1}\)，

\[
\boldsymbol{\Delta} = (x_c-\mu_{\mathrm{lat}},\, y_c-\mu_{\mathrm{ax}},\, -\mu_z)^{\mathsf T},\quad
\alpha = o_g \exp(-\tfrac{1}{2}\boldsymbol{\Delta}^{\mathsf T}\mathbf{C}^{-1}\boldsymbol{\Delta}).
\]

AABB 在 **无条件** \((\mu_{\mathrm{lat}},\mu_{\mathrm{ax}})\) 上。

**反向**：`ours_alpha_atomic_bwd_wire` 累加 `wrow[3..11]` 为 \(\bar{\mathbf{C}}^{-1}\) 9 元；finalize 用 probe3d 的 `ex,ey,ez` 链到 scale/rotation（与 plane_cond2d finalize **不同**）。

---

## 附录 B：`compositing_mode = additive`（非当前 liver 实验）

前向：\(C_p = \sum_g \iota_g \alpha_{g,p}\)（无 \(T\)）。

反向：`kernel_plane_cond2d_additive_bwd` 在 `gauss_rects` 覆盖的像素上直接累加，**无** alpha 重放；公式为 \(\partial(\iota\alpha)/\partial\cdot\) 的 product rule。

---

## 附录 C：常量一览（与代码一致）

| 常量 | 值 | 位置 |
|------|-----|------|
| `OURS_ALPHA_MIN` | \(1/255\) | `shared.cuh` |
| `kProbeCovJitter` | \(10^{-6}\) | `plane_cond2d_device::kProbeCovJitter` |
| `kDetMin` | \(10^{-12}\) | `invert_sigma2_sym` |
| `kMaxInvDet` | \(10^{8}\) | `inv_det = min(1/det, kMaxInvDet)` |
| `kGradClamp` | \(10^{3}\) | `plane_cond2d_sanitize_grad` |
| scale 下限 | \(10^{-6}\) | `fmaxf(sptr[k],1e-6f)` |
| alpha 早停 | \(T<10^{-4}\) | forward/back replay |
| 最大重放层数 | 256 | `OURS_MAX_ALPHA_LAYERS` |
| `PLANE_COND2D_WSP_STRIDE` | 14 | workspace floats / gaussian |
| `SHADER_WEIGHT_PLANE_COND2D` | 1 | `slice_grid_geometry.cuh` |
| 四元数 FD 步长 | \(10^{-3}\) | `grad_quat_finite_diff` |

---

## 附录 D：不可微 / 启发式行为（排查训练用）

1. **离散剔除**：tile 矩形、elev 走廊、Schur/probe3d AABB、\(q<0\)、\(\alpha<1/255\)、\(T<10^{-4}\) 早停 → 相关参数梯度为 0。
2. **\(q<0\)**：平面权重为 0，平面内梯度为 0（勿用 \(\max(q,0)\) 再 \(\exp\)，否则 \(q<0\) 时权重为 1）。
3. **`inv_det` clamp**：\(1/\det_2 > 10^8\) 时 \(\bar{\mathbf{S}}_2\) 置 0。
4. **\(\alpha\le 1\) clamp**：前向 `fminf(alpha,1)`，反向未单独处理饱和。
5. **四元数**：`grad_quat_finite_diff`，\(\varepsilon=10^{-3}\)。
6. **Pose**：旋转列为独立归一化向量，非 SO(3) 最小参数化。
7. **Opacity**：建议在 \((0,1]\) 初始化；渲染器对 \(\alpha\) 做 \([0,1]\) clamp。

---

## 附录 E：渲染伪影与实现对照（排查用）

| 现象 | 可能原因 | 代码位置 |
|------|----------|----------|
| 16×16 栅格块 | tile 硬边界 + Schur AABB 在 \(k_\sigma{=}2\) 处截断 | `core_tile.cu`, `plane_cond2d_schur_aabb_radii_mm` |
| 大片不透明多边形 | 曾用 `fmax(q,0)` 且去掉 AABB，\(q<0\) 时 \(\alpha\approx o\) | 已改为 `plane_cond2d_plane_weight` |
| mapper 漏高斯 | footprint 小于 shader 支撑 | mapper 用 `max(r^ell+pad, r^schur)`，shader 仅 `r^schur` |
| 训练 scale 膨胀 | `scale_wall` 与窄支撑对抗；或 Schur/pose 梯度异常 | 配置 `scale_wall`；查 `sanitize` |

**验证脚本**：`tests/ops/ours_renderer/validate_plane_cond2d.py`（CPU 参考与 GPU 对齐，含 \(q<0\to 0\) 语义）。

---

*文档版本：与 `plane_cond2d_device.cuh`（Schur AABB + `plane_cond2d_plane_weight`）及 tile alpha 路径同步；改 CUDA 时请同步更新 §0.3、§2.2、§3、附录 E。*
