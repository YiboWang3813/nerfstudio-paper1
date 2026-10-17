# 切片着色算子（s2s_shader）：数学与实现对照

本文档给出切片着色算子的 **前向 / 反向** 显式公式，并与仓库内 `s2s_shader/__init__.py`、`src/forward.cu`、`src/backward.cu`、`src/bindings.cpp`、`include/forward.cuh`、`include/backward.cuh`、`include/rasterize_points.cuh` 的张量一一对应，格式与 `ultrags_shader/development.md` 对齐，便于在保持本算子「切片强度渲染」核心逻辑的同时，将协方差逆矩阵 `sigma_inv` 的构造与梯度逻辑与 `ultrags_shader` 共享。

**记号**：$G$ 为高斯个数；$P = H \times W$，像素展平索引 $p \in \{0,\ldots,P-1\}$；像素 $p$ 按固定顺序遍历若干高斯，第 $k$ 个高斯下标为 $g_{p,k}$（由 `pixel_gauss_indices` 与 `pixel_gauss_offsets` / `pixel_gauss_counts` 给出）。

---

## 1. 参数与张量定义（LaTeX ↔ 代码）

### 1.1 每个高斯 $g$ 的可学习量（Python / Autograd 侧）

结合 `s2s_shader/__init__.py` 中 `_ShadeFunction.forward` 的签名，本算子在 PyTorch 端暴露的高斯参数为：

* **中心**：$\boldsymbol{\mu}_g \in \mathbb{R}^3$ → `gauss_means` $(G,3)$  
* **各向异性尺度**：$\mathbf{s}_g = (s_{g,x}, s_{g,y}, s_{g,z})^\top$ → `gauss_scales` $(G,3)$  
* **单位四元数**：$\mathbf{q}_g \in \mathbb{R}^4$ → `gauss_quats` $(G,4)$  
* **不透明度**：$o_g$ → `gauss_opacities` $(G,)$  
* **标量强度**：$I_g$ → `gauss_intensities` $(G,)$  

Python 端不做任何「打包」或预构造协方差的工作；C++ / CUDA 内部直接使用这些张量，执行旋转矩阵和精度矩阵的构造。

### 1.2 像素与稀疏映射

* **像素世界坐标**：$\mathbf{x}_p \in \mathbb{R}^3$ → `pixel_positions` $(P,3)$  
* **像素 $p$ 对应的高斯列表：**
  * 起始下标 $b_p$ → `pixel_gauss_offsets[p]`  
  * 个数 $n_p$ → `pixel_gauss_counts[p]`  
  * 第 $k$ 个高斯下标 $g_{p,k} =$ `pixel_gauss_indices[b_p + k]`  

### 1.3 前向中间量与输出

| 数学符号 | 代码 | Shape |
| :--- | :--- | :--- |
| $\boldsymbol{\Sigma}_g^{-1}$（行主序展平） | `gauss_sigma_invs` | $(G,9)$ |
| 像素强度 $I_p$ | `pixel_intensities` | $(P,)$ |
其中 $\mathrm{nnz} = \sum_p n_p$ 为所有像素-高斯关联对的总数。  
为节省显存并简化接口，本算子的前向 **不再缓存** 每对 $(p,k)$ 的马氏指数 $e_{p,k}$ 和偏移向量 $\mathbf{d}_{p,k}$，而是在反向像素核中按需重新计算。

---

## 2. 前向传播（Forward）

前向逻辑在 Python 侧由 `s2s_shader/__init__.py` 中的 `_ShadeFunction.forward` 封装，对应 C++ 函数为 `_C.forward`，其签名为：

```text
_C.forward(
    gauss_means, gauss_scales, gauss_quats, gauss_opacities, gauss_intensities,
    pixel_positions,
    pixel_gauss_offsets, pixel_gauss_counts, pixel_gauss_indices,
) -> (pixel_colors, gauss_Rs, gauss_sigma_invs)
```

核心前向关系（无打包、直接使用这些张量）为：

1. **协方差逆与旋转矩阵（CUDA 内部完成）**

   对每个高斯 $g$，由 `gauss_scales[g]` 与 `gauss_quats[g]` 调用 `covariance.cuh` 中与 `ultrags_shader` 共用的函数，在 CUDA 内部得到：

   \[
   \mathbf{R}_g = \mathrm{Rot}(\mathbf{q}_g),\qquad
   \boldsymbol{\Sigma}^{-1}_g = \mathrm{CovInv}(\mathbf{s}_g, \mathbf{R}_g),
   \]

   并写入 `gauss_Rs[g]` 与 `gauss_sigma_invs[g]`。

2. **像素强度累积（无 Alpha 混合，仅线性叠加）**

   对像素 $p$ 与其关联的高斯 $g_{p,k}$：

   - 像素-高斯偏移：
     \[
     \mathbf{d}_{p,k} = \mathbf{x}_p - \boldsymbol{\mu}_{g_{p,k}}.
     \]
   - 马氏指数：
     \[
     e_{p,k} = -\frac{1}{2}\,\mathbf{d}_{p,k}^\top \boldsymbol{\Sigma}^{-1}_{g_{p,k}} \mathbf{d}_{p,k}.
     \]
   - 权重与贡献：
     \[
     \alpha_{p,k} = o_{g_{p,k}} \exp(e_{p,k}),\qquad
     I_p = \sum_{k=1}^{n_p} \alpha_{p,k} I_{g_{p,k}}.
     \]

   CUDA 实现可在像素 kernel 中直接使用 `gauss_means`、`gauss_sigma_invs`、`gauss_opacities`、`gauss_intensities`、`pixel_positions` 以及 `pixel_gauss_*` 三个索引张量实现上述公式，对应 Python 端返回的 `pixel_colors`（再在模块中 reshape 为图像）。

---

## 3. 反向传播（Backward）

反向逻辑在 Python 侧由 `_ShadeFunction.backward` 封装，在 CUDA 侧由 `kernel_render_pixel_backward`（`src/backward.cu`）实现。上游给定损失对像素强度的梯度 $\dfrac{\partial L}{\partial I_p}$（`grad_pixel_intensities[p]`）。

### 3.1 对强度与不透明度的梯度

根据前向
\[
I_p = \sum_{k} \alpha_{p,k} I_{g_{p,k}},
\qquad
\alpha_{p,k} = o_{g_{p,k}} \exp(e_{p,k}),
\]
有
\[
\frac{\partial L}{\partial I_{g}} = \sum_{p,k:\,g_{p,k}=g} \frac{\partial L}{\partial I_p}\,\alpha_{p,k},
\qquad
\frac{\partial L}{\partial o_g} = \sum_{p,k:\,g_{p,k}=g} \frac{\partial L}{\partial I_p}\,I_g \exp(e_{p,k}).
\]

对应实现：

* `alpha_gp = opacity * exp(e_gp)`（`src/backward.cu` L38–L40）  
* `atomicAdd(grad_intensities[g], grad_Ip * alpha_gp)`（L42）  
* `atomicAdd(grad_opacities[g], grad_Ip * intensity * exp_e)`（L45）。

### 3.2 对中心 $\boldsymbol{\mu}_g$ 的梯度

由
\[
e_{p,k} = -\frac{1}{2}\mathbf{d}_{p,k}^\top \boldsymbol{\Sigma}^{-1} \mathbf{d}_{p,k},
\qquad
\mathbf{d}_{p,k} = \mathbf{x}_p - \boldsymbol{\mu}_g,
\]
得到
\[
\frac{\partial e_{p,k}}{\partial \mathbf{d}_{p,k}} = -\boldsymbol{\Sigma}^{-1}\mathbf{d}_{p,k},
\qquad
\frac{\partial \mathbf{d}_{p,k}}{\partial \boldsymbol{\mu}_g} = -\mathbf{I},
\]
从而
\[
\frac{\partial e_{p,k}}{\partial \boldsymbol{\mu}_g}
= \boldsymbol{\Sigma}^{-1}\mathbf{d}_{p,k}.
\]

又有
\[
\frac{\partial L}{\partial e_{p,k}}
= \frac{\partial L}{\partial I_p}\,\frac{\partial I_p}{\partial e_{p,k}}
= \frac{\partial L}{\partial I_p}\,o_g I_g \exp(e_{p,k}),
\]
令
\[
C_{p,k} := \frac{\partial L}{\partial e_{p,k}}
= \frac{\partial L}{\partial I_p}\,o_g I_g \exp(e_{p,k}),
\]
则
\[
\frac{\partial L}{\partial \boldsymbol{\mu}_g}
= \sum_{p,k:\,g_{p,k}=g} C_{p,k}\,\boldsymbol{\Sigma}^{-1}\mathbf{d}_{p,k}.
\]

对应实现：

* `C_gp = grad_Ip * opacity * intensity * exp_e`（L48）即上式中的 $C_{p,k}$；  
* `Sd = S * d`（L60–L62，对应 $\boldsymbol{\Sigma}^{-1}\mathbf{d}$）；  
* `atomicAdd(grad_means[g*3 + i], C_gp * Sd_i)`（L64–L66）。

### 3.3 对精度矩阵 $\boldsymbol{\Sigma}^{-1}_g$ 的梯度

由
\[
e_{p,k} = -\frac{1}{2}\sum_{i,j} \left[\boldsymbol{\Sigma}^{-1}\right]_{ij} d_i d_j,
\]
可得
\[
\frac{\partial e_{p,k}}{\partial \left[\boldsymbol{\Sigma}^{-1}\right]_{ij}}
= -\frac{1}{2} d_i d_j.
\]

结合 $C_{p,k} = \dfrac{\partial L}{\partial e_{p,k}}$：
\[
\frac{\partial L}{\partial \left[\boldsymbol{\Sigma}^{-1}\right]_{ij}}
= \sum_{p,k:\,g_{p,k}=g} C_{p,k}
\left(-\frac{1}{2} d_i d_j\right).
\]

对应实现：

* `coeff = -0.5f * C_gp`（L71）；  
* 对所有 $(i,j)$ 逐项 `atomicAdd`（L73–L81），直接累加到 `grad_sigma_invs[g * 9 + ...]`。

### 3.4 向 `scale` 与 `quat` 的回传（与 `ultrags_shader` 保持一致）

本文件中仅构造到 $\dfrac{\partial L}{\partial \boldsymbol{\Sigma}^{-1}_g}$。若要进一步回传到 `gauss_scales` 与 `gauss_quats`，应与 `ultrags_shader/development.md` §3.3–3.4 完全一致：

1. 通过一个「精度矩阵 → 旋转 + 尺度」的 kernel（类似 `transform_grads_sigma_to_R_S`）将 $\dfrac{\partial L}{\partial \boldsymbol{\Sigma}^{-1}_g}$ 映射到 $\dfrac{\partial L}{\partial \mathbf{R}_g}$ 与 $\dfrac{\partial L}{\partial \mathbf{s}_g}$；  
2. 再通过「旋转矩阵 → 四元数」的 kernel（类似 `transform_grads_R_to_Quat`）得到 $\dfrac{\partial L}{\partial \mathbf{q}_g}$；  
3. Python 端按 `ultrags_shader/__init__.py` 中的方式将打包梯度拆成 `dL_dGaussScales`、`dL_dGaussQuats`。

---

## 4. 调用链与文件映射（给 agent 的索引）

| 阶段 | 函数 / Kernel | 主要文件 |
| :--- | :--- | :--- |
| PyTorch Autograd 封装 | `_C.forward` / `_C.backward` | `s2s_shader/__init__.py` |
| C++ ↔ CUDA 包装 | `RasterizeGaussiansForwardCUDA` / `RasterizeGaussiansBackwardCUDA` | `src/bindings.cpp`, `include/rasterize_points.cuh` |
| 前向 kernel | `kernel_render_pixel_forward` | `src/forward.cu`, `include/forward.cuh` |
| 反向 kernel | `kernel_render_pixel_backward` | `src/backward.cu`, `include/backward.cuh` |

---

## 5. 实现要点与与 ultrags_shader 的对齐建议

1. **核心差异**：本算子只输出标量强度图 `pixel_intensities`，不做 Alpha 混合，也不输出深度图；`ultrags_shader` 则实现了带透射率的 result-first alpha blending 与 elevational 深度。  
2. **sigma_inv 逻辑复用**：数学上，本算子的 $\boldsymbol{\Sigma}^{-1}_g$ 与 `ultrags_shader` 完全一致，均由 `gauss_scales` 与 `gauss_quats` 经旋转矩阵与协方差构造得到。重构 CUDA 时应直接 reuse / include `ultrags_shader` 的协方差构造与梯度转换函数。  
3. **稀疏映射一致性**：`pixel_gauss_offsets` / `pixel_gauss_counts` / `pixel_gauss_indices` 在前、后向使用完全一致，任何截断或重采样策略必须同时作用在两侧。  
4. **数值稳定性**：在复用 `ultrags_shader` 的协方差逻辑时，应保持相同的尺度下界、epsilon 防护与矩阵对称化策略，以免两算子的正则化表现不一致。  
5. **扩展路径**：若未来需要为切片算子引入透射率或深度输出，可以直接参考 `ultrags_shader` 的三 kernel 结构（预处理 → Alpha 累积 → 与背景合成），在本算子中增加额外输出和中间量，而保留当前 `kernel_render_pixel_forward` 作为「简单强度分支」。