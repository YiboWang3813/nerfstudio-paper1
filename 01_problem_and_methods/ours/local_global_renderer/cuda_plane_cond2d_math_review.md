# CUDA 算子数学设计文档审查建议

本文整理针对 `plane_cond2d + tile mapper + alpha compositing` 这一路径的数学审查意见。重点关注：从世界坐标系 3D 高斯椭球出发，经 pose 投影到探头坐标系，再通过条件高斯映射到成像平面的 slice-based volume reconstruction 过程是否严格成立。

---

## 1. 总体结论

当前文档作为 **CUDA 实现说明** 基本完整，但作为 **严格的 slice-based 3D Gaussian volume reconstruction 数学推导**，仍有若干关键不严谨之处。

主要问题包括：

1. `plane_cond2d` 使用的是条件高斯的平面内分布，但缺少 3D 高斯在 elev 方向到成像平面的距离衰减项。
2. Tile mapper 使用无条件中心 `(mu_lat, mu_ax)`，而 shader 使用条件均值 `mu_2`，二者不一致，可能导致贡献被漏算。
3. 文档中的 `delta` / `mu_2` 反向链式法则存在明显错误。
4. 一些数值 clamp、hard branch、early stop 使当前反向并非严格的连续前向导数。
5. pose 的旋转列独立归一化不是严格的 SE(3) 参数化，若优化 pose，需要额外注意。

因此，当前算子更准确地说是一个工程上可用的 **conditional-plane Gaussian shader heuristic**，而不是严格的 3D 高斯切片体渲染公式。

---

## 2. 正确且可以保留的部分

### 2.1 世界坐标到探头坐标的均值变换

设世界系高斯中心为：

$$
\boldsymbol{\mu}^w
$$

pose 的平移为：

$$
\mathbf{t}
$$

探头三个单位轴为：

$$
\hat{\mathbf{a}}_{\mathrm{lat}},\quad
\hat{\mathbf{a}}_{\mathrm{ax}},\quad
\hat{\mathbf{a}}_{\mathrm{ele}}
$$

定义：

$$
\mathbf{A}
=
\begin{bmatrix}
\hat{\mathbf{a}}_{\mathrm{lat}}^T \\
\hat{\mathbf{a}}_{\mathrm{ax}}^T \\
\hat{\mathbf{a}}_{\mathrm{ele}}^T
\end{bmatrix}
$$

则探头坐标系下的高斯中心为：

$$
\boldsymbol{\mu}^c
=
\mathbf{A}(\boldsymbol{\mu}^w-\mathbf{t})
$$

即：

$$
\mu_{\mathrm{lat}}
=
(\boldsymbol{\mu}^w-\mathbf{t})^T\hat{\mathbf{a}}_{\mathrm{lat}}
$$

$$
\mu_{\mathrm{ax}}
=
(\boldsymbol{\mu}^w-\mathbf{t})^T\hat{\mathbf{a}}_{\mathrm{ax}}
$$

$$
\mu_z
=
(\boldsymbol{\mu}^w-\mathbf{t})^T\hat{\mathbf{a}}_{\mathrm{ele}}
$$

这部分数学上是合理的。

---

### 2.2 世界协方差到探头协方差的变换

若世界坐标系下 3D 高斯协方差为：

$$
\boldsymbol{\Sigma}^w
=
\mathbf{R}
\operatorname{diag}(s_0^2,s_1^2,s_2^2)
\mathbf{R}^T
$$

则探头坐标系下协方差为：

$$
\mathbf{S}_c
=
\mathbf{A}\boldsymbol{\Sigma}^w\mathbf{A}^T
$$

这也是标准的协方差坐标变换。

---

### 2.3 Schur 条件高斯公式

将探头系高斯拆成平面内坐标和 elev 坐标：

$$
\begin{bmatrix}
\mathbf{p}\\
z
\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
\boldsymbol{\mu}_p\\
\mu_z
\end{bmatrix},
\begin{bmatrix}
\mathbf{S}_{pp} & \mathbf{S}_{pz}\\
\mathbf{S}_{zp} & S_{zz}
\end{bmatrix}
\right)
$$

其中：

$$
\mathbf{p}
=
\begin{bmatrix}
\mathrm{lat}\\
\mathrm{ax}
\end{bmatrix}
$$

当条件为成像平面：

$$
z=0
$$

条件高斯为：

$$
\mathbf{p}\mid z=0
\sim
\mathcal{N}
\left(
\boldsymbol{\mu}_2,
\boldsymbol{\Sigma}_2
\right)
$$

其中：

$$
\boldsymbol{\mu}_2
=
\boldsymbol{\mu}_p
+
\mathbf{S}_{pz}S_{zz}^{-1}(0-\mu_z)
$$

即：

$$
\boldsymbol{\mu}_2
=
\boldsymbol{\mu}_p
-
\mathbf{S}_{pz}S_{zz}^{-1}\mu_z
$$

条件协方差为：

$$
\boldsymbol{\Sigma}_2
=
\mathbf{S}_{pp}
-
\mathbf{S}_{pz}S_{zz}^{-1}\mathbf{S}_{zp}
$$

由于 $S_{zz}$ 是标量，可以写成：

$$
\boldsymbol{\mu}_2
=
\boldsymbol{\mu}_p
-
\frac{\mu_z}{S_{zz}}\mathbf{S}_{pz}
$$

$$
\boldsymbol{\Sigma}_2
=
\mathbf{S}_{pp}
-
\frac{1}{S_{zz}}\mathbf{S}_{pz}\mathbf{S}_{pz}^T
$$

这一部分是标准条件高斯推导，可以保留。

---

## 3. 关键问题一：`plane_cond2d` 缺少 elev attenuation

### 3.1 当前文档使用的 alpha

当前 `plane_cond2d` 使用：

$$
\alpha_{g,p}
=
o_g
\exp
\left(
-\frac{1}{2}
q_{\mathrm{cond}}
\right)
$$

其中：

$$
q_{\mathrm{cond}}
=
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_2)^T
\boldsymbol{\Sigma}_2^{-1}
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_2)
$$

这个公式描述的是：

> 在已经条件到 $z=0$ 的前提下，平面内的 2D 条件高斯分布形状。

但是它不是严格的 3D 高斯在切片平面上的体密度值。

---

### 3.2 严格的 3D 高斯切片值

如果从 3D 高斯体密度出发，成像平面 $z=0$ 上的点为：

$$
\begin{bmatrix}
\boldsymbol{\xi}_p\\
0
\end{bmatrix}
$$

则 3D 高斯值应该为：

$$
G(\boldsymbol{\xi}_p,0)
=
o_g
\exp
\left[
-\frac{1}{2}
\begin{bmatrix}
\boldsymbol{\xi}_p-\boldsymbol{\mu}_p\\
-\mu_z
\end{bmatrix}^T
\mathbf{S}_c^{-1}
\begin{bmatrix}
\boldsymbol{\xi}_p-\boldsymbol{\mu}_p\\
-\mu_z
\end{bmatrix}
\right]
$$

利用条件高斯分解，有：

$$
\begin{bmatrix}
\boldsymbol{\xi}_p-\boldsymbol{\mu}_p\\
-\mu_z
\end{bmatrix}^T
\mathbf{S}_c^{-1}
\begin{bmatrix}
\boldsymbol{\xi}_p-\boldsymbol{\mu}_p\\
-\mu_z
\end{bmatrix}
=
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_2)^T
\boldsymbol{\Sigma}_2^{-1}
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_2)
+
\mu_z^2 S_{zz}^{-1}
$$

所以严格的 slice cross-section alpha 应该是：

$$
\alpha_{g,p}
=
o_g
\exp
\left[
-\frac{1}{2}
\left(
q_{\mathrm{cond}}
+
\frac{\mu_z^2}{S_{zz}}
\right)
\right]
$$

也可以写成：

$$
\alpha_{g,p}
=
o_g
\exp
\left(
-\frac{1}{2}q_{\mathrm{cond}}
\right)
\exp
\left(
-\frac{1}{2}\frac{\mu_z^2}{S_{zz}}
\right)
$$

其中第二项：

$$
\exp
\left(
-\frac{1}{2}\frac{\mu_z^2}{S_{zz}}
\right)
$$

就是 elev 方向的距离衰减项。

---

### 3.3 当前公式的风险

当前公式缺少 elev attenuation，因此会导致：

1. 只要高斯通过 elev corridor，即便其中心距离成像平面较远，也可能给出较强响应。
2. 远离平面的高斯不会按照 3D Gaussian density 自然衰减。
3. 模型更像是在做 “conditional plane projection”，而不是严格的 “slice cross-section evaluation”。

如果你的目标是物理意义更强的 slice-based volume reconstruction，建议加入 elev attenuation。

---

### 3.4 建议修改

将当前 alpha：

$$
\alpha
=
o\exp
\left(
-\frac{1}{2}q_{\mathrm{cond}}
\right)
$$

改为：

$$
\boxed{
\alpha
=
o\exp
\left[
-\frac{1}{2}
\left(
q_{\mathrm{cond}}
+
\frac{\mu_z^2}{S_{zz}}
\right)
\right]
}
$$

其中：

$$
q_{\mathrm{cond}}
=
(\boldsymbol{\xi}-\boldsymbol{\mu}_2)^T
\boldsymbol{\Sigma}_2^{-1}
(\boldsymbol{\xi}-\boldsymbol{\mu}_2)
$$

---

## 4. 关键问题二：Tile mapper 和 shader 使用的中心不一致

### 4.1 当前行为

Tile mapper 使用无条件平面中心：

$$
\boldsymbol{\mu}_p
=
\begin{bmatrix}
\mu_{\mathrm{lat}}\\
\mu_{\mathrm{ax}}
\end{bmatrix}
$$

但 `plane_cond2d` shader 实际计算 alpha 时使用条件均值：

$$
\boldsymbol{\mu}_2
=
\boldsymbol{\mu}_p
-
\frac{\mu_z}{S_{zz}}\mathbf{S}_{pz}
$$

二者差值为：

$$
\Delta\boldsymbol{\mu}
=
\boldsymbol{\mu}_2-\boldsymbol{\mu}_p
=
-\frac{\mu_z}{S_{zz}}\mathbf{S}_{pz}
$$

当 $\mu_z$ 较大，或者 $\mathbf{S}_{pz}$ 较大时，条件中心可能明显偏离无条件中心。

---

### 4.2 可能造成的问题

如果 tile mapper 只围绕 $\boldsymbol{\mu}_p$ 发 tile，而 shader 中真正有贡献的位置围绕 $\boldsymbol{\mu}_2$，那么会出现：

1. 理论上某些 pixel 应该收到该高斯贡献；
2. 但对应 tile 没有被 mapper 收录；
3. shader 根本不会遍历到该高斯；
4. 前向贡献被硬漏掉；
5. 反向梯度也自然消失。

这不是单纯的效率问题，而是前向函数本身被改变了。

---

### 4.3 建议修改方案 A：mapper 也使用条件均值

如果 `plane_cond2d` 是主要 shader，建议在 mapper 阶段也计算：

$$
\boldsymbol{\mu}_2
=
\boldsymbol{\mu}_p
-
\frac{\mu_z}{S_{zz}}\mathbf{S}_{pz}
$$

然后用：

$$
u_{\mathrm{pix}}
=
\frac{\mu_{2,\mathrm{lat}}}{\Delta_{\mathrm{lat}}}
+
w_{\mathrm{off}}
$$

$$
v_{\mathrm{pix}}
=
\frac{\mu_{2,\mathrm{ax}}-z_{\mathrm{near}}}{\Delta_{\mathrm{ax}}}
$$

作为 tile center。

---

### 4.4 建议修改方案 B：扩展 tile 半径

如果希望 mapper 继续保持 probe3d / plane_cond2d 共用逻辑，可以扩展半径：

$$
r_{\mathrm{mapper,lat}}
\ge
r_{\mathrm{base,lat}}
+
|\mu_{2,\mathrm{lat}}-\mu_{\mathrm{lat}}|
$$

$$
r_{\mathrm{mapper,ax}}
\ge
r_{\mathrm{base,ax}}
+
|\mu_{2,\mathrm{ax}}-\mu_{\mathrm{ax}}|
$$

这样即便 mapper 的中心仍然是 $\boldsymbol{\mu}_p$，也能覆盖条件中心附近的潜在贡献区域。

---

### 4.5 推荐

如果当前实验主要使用 `plane_cond2d`，优先推荐方案 A。

如果未来需要在同一个 mapper 中同时支持 `probe3d` 和 `plane_cond2d`，推荐方案 B 或者根据 `shader_weight_mode` 分支构造不同的 mapper center / radius。

---

## 5. 关键问题三：`delta` / `mu_2` 反向公式错误

### 5.1 当前文档中的写法

文档中定义：

$$
q
=
\boldsymbol{\delta}^T
\mathbf{I}_2
\boldsymbol{\delta}
$$

其中：

$$
\boldsymbol{\delta}
=
\boldsymbol{\xi}
-
\boldsymbol{\mu}_2
$$

并写出：

$$
\bar{q}
=
\bar{\mathrm{base}}
\left(
-\frac{1}{2}\mathrm{base}
\right)
$$

然后文档给出：

$$
\bar{\boldsymbol{\delta}}
=
-\bar q\mathbf{I}_2\boldsymbol{\delta}
$$

这个公式不正确。

---

### 5.2 正确链式法则

由于：

$$
q
=
\boldsymbol{\delta}^T
\mathbf{I}_2
\boldsymbol{\delta}
$$

且 $\mathbf{I}_2$ 对称，所以：

$$
\frac{\partial q}{\partial \boldsymbol{\delta}}
=
2\mathbf{I}_2\boldsymbol{\delta}
$$

因此：

$$
\boxed{
\bar{\boldsymbol{\delta}}
=
2\bar q\mathbf{I}_2\boldsymbol{\delta}
}
$$

又因为：

$$
\boldsymbol{\delta}
=
\boldsymbol{\xi}
-
\boldsymbol{\mu}_2
$$

所以：

$$
\frac{\partial \boldsymbol{\delta}}{\partial \boldsymbol{\mu}_2}
=
-\mathbf{I}
$$

因此：

$$
\boxed{
\bar{\boldsymbol{\mu}}_2
=
-\bar{\boldsymbol{\delta}}
=
-2\bar q\mathbf{I}_2\boldsymbol{\delta}
}
$$

---

### 5.3 一维 sanity check

考虑一维情况：

$$
\alpha
=
\exp
\left[
-\frac{1}{2}
(\xi-\mu)^2
\right]
$$

严格导数为：

$$
\frac{\partial \alpha}{\partial \mu}
=
\alpha(\xi-\mu)
$$

如果按照正确公式：

$$
q=(\xi-\mu)^2
$$

$$
\bar q
=
-\frac{1}{2}\alpha
$$

$$
\bar \mu
=
-2\bar q(\xi-\mu)
=
\alpha(\xi-\mu)
$$

结果正确。

如果使用文档当前公式：

$$
\bar\delta
=
-\bar q\delta
$$

则：

$$
\bar\mu
=
-\bar\delta
=
\bar q\delta
=
-\frac{1}{2}\alpha(\xi-\mu)
$$

不仅符号相反，而且系数也差了 2 倍。

---

### 5.4 建议修改

将文档与代码中相关 wire 反向改成：

$$
\bar q
=
\bar{\mathrm{base}}
\left(
-\frac{1}{2}\mathrm{base}
\right)
$$

$$
\bar{\boldsymbol{\delta}}
=
2\bar q\mathbf{I}_2\boldsymbol{\delta}
$$

$$
\bar{\boldsymbol{\mu}}_2
=
-2\bar q\mathbf{I}_2\boldsymbol{\delta}
$$

对 workspace 应写入：

$$
\mathrm{wrow}[0]
\mathrel{+}=
\bar{\mu}_{2,\mathrm{lat}}
$$

$$
\mathrm{wrow}[1]
\mathrel{+}=
\bar{\mu}_{2,\mathrm{ax}}
$$

即：

$$
\mathrm{wrow}[0]
\mathrel{+}=
-2\bar q
\left(
i_{00}\delta_{\mathrm{lat}}
+
i_{01}\delta_{\mathrm{ax}}
\right)
$$

$$
\mathrm{wrow}[1]
\mathrel{+}=
-2\bar q
\left(
i_{01}\delta_{\mathrm{lat}}
+
i_{11}\delta_{\mathrm{ax}}
\right)
$$

---

## 6. 关键问题四：inverse / clamp / branch 的反向不严格

### 6.1 2×2 inverse clamp

文档中使用：

$$
\mathrm{inv\_det}
=
\min
\left(
\frac{1}{\det_2},
10^8
\right)
$$

但反向里仍然使用普通矩阵逆的导数：

$$
\bar{\boldsymbol{\Sigma}}_2
=
-
\mathbf{I}_2
\bar{\mathbf{I}}_2
\mathbf{I}_2
$$

这个公式只在没有 clamp 的区域严格成立。

如果：

$$
\frac{1}{\det_2}
>
10^8
$$

前向已经进入饱和区，严格反向应该考虑 clamp 的导数，而不是继续使用普通 inverse backward。

---

### 6.2 `q_+ = max(q,0)` 的反向

若前向使用：

$$
q_+
=
\max(q,0)
$$

则严格子梯度为：

$$
\frac{\partial q_+}{\partial q}
=
\begin{cases}
1, & q>0\\
0, & q<0
\end{cases}
$$

因此：

$$
\bar q
=
\bar q_+
\cdot \mathbf{1}_{q>0}
$$

虽然理论上 $q=\delta^TI\delta$ 应该非负，但由于数值误差或非正定协方差，代码里加入了 `fmax(q,0)`。如果反向没有同步处理这个 branch，那么反向不是前向的严格导数。

---

### 6.3 alpha threshold / early stop / AABB

以下操作都是硬离散分支：

1. tile membership；
2. elev corridor；
3. AABB；
4. $\alpha<1/255$ skip；
5. $T<10^{-4}$ early stop；
6. 最多 replay 256 层；
7. Schur / determinant failure branch。

这些操作使得整体前向函数是分段的、不连续的。当前反向可以作为工程近似，但不能称为完整连续导数。

---

### 6.4 建议

文档中应该明确区分：

1. **理想连续模型的导数**；
2. **当前 CUDA 实现采用的工程近似梯度**；
3. **hard branch 处梯度置零或未定义的部分**。

建议在文档中新增一节：

```text
Differentiability Notes / Non-smooth Implementation Details
```

把所有不可微或非严格反向的分支集中列出。

---

## 7. 关键问题五：pose 不是严格 SE(3) 参数化

当前文档中 pose 的三个旋转列分别归一化：

$$
\hat{\mathbf{a}}_i
=
\frac{\mathbf{r}_i}{\|\mathbf{r}_i\|}
$$

然后使用：

$$
\hat{\mathbf{a}}_{\mathrm{lat}},
\hat{\mathbf{a}}_{\mathrm{ax}},
\hat{\mathbf{a}}_{\mathrm{ele}}
$$

作为探头坐标轴。

这个处理可以保证每个轴是单位向量，但不能保证：

$$
\hat{\mathbf{a}}_i^T\hat{\mathbf{a}}_j=0,\quad i\ne j
$$

也不能保证：

$$
\det R=1
$$

因此它不是严格的 SO(3) 或 SE(3) 参数化。

---

### 7.1 如果 pose 只作为输入

如果 pose 来自外部 tracker，且不需要优化，那么这个问题影响较小。此时它只是一个坐标投影实现细节。

---

### 7.2 如果 pose 需要优化

如果未来要优化 pose，则建议使用更严格的参数化，例如：

1. Lie algebra se(3)；
2. quaternion + normalization；
3. 6D rotation representation；
4. SVD / Gram-Schmidt 正交化；
5. 显式维护 $R\in SO(3)$。

否则，当前 `normalize3_backward` 得到的是独立列归一化梯度，不是严格的刚体姿态梯度。

---

## 8. 其他文档符号问题

### 8.1 `S_{05}` 符号不严谨

文档中写：

$$
\mathbf{S}_{pz}
=
\begin{bmatrix}
S_{02}\\
S_{05}
\end{bmatrix}
$$

如果 $S$ 是数学矩阵，第二项应该写成：

$$
S_{12}
$$

如果想表达 flattened row-major index，则应该写：

```text
Sc[2] = S_{02}
Sc[5] = S_{12}
```

建议不要混用数学矩阵下标和 flattened array 下标。

---

### 8.2 symmetric gradient 的 off-diagonal 约定需要明确

对于 $2\times2$ 对称矩阵：

$$
\mathbf{I}_2
=
\begin{bmatrix}
i_{00} & i_{01}\\
i_{01} & i_{11}
\end{bmatrix}
$$

如果 workspace 只存一个 $i_{01}$，那么：

$$
q
=
i_{00}\delta_0^2
+
2i_{01}\delta_0\delta_1
+
i_{11}\delta_1^2
$$

因此：

$$
\bar i_{01}
=
\bar q\cdot 2\delta_0\delta_1
$$

这一点文档是写对的。

但是后续从 packed symmetric gradient 转 full matrix gradient 时，必须明确：

- `wrow[4]` 是对单个参数 $i_{01}$ 的梯度；
- 还是对 full matrix 中 $I_{01}$ 一个 entry 的梯度；
- 在 inverse backward 中是否需要除以 2 或分配到两个 off-diagonal entries。

建议在文档中加入一张表，明确所有 symmetric packed entries 的梯度语义。

---

### 8.3 opacity 需要约束到 alpha 合理范围

Alpha compositing 默认要求：

$$
0\le \alpha \le 1
$$

但当前文档只写：

$$
\alpha=o_g\cdot \mathrm{base}
$$

如果 $o_g$ 没有经过 sigmoid、clamp 或 softplus + cap，则可能出现：

$$
\alpha>1
$$

甚至：

$$
\alpha<0
$$

此时：

$$
T\leftarrow T(1-\alpha)
$$

就不再有透明度意义。

建议明确：

1. $o_g$ 是否为 raw opacity；
2. 是否经过 sigmoid；
3. 是否 clamp 到 $[0,1]$；
4. 如果没有约束，alpha compositing 是否仍然只是数值 heuristic。

---

## 9. 推荐的修正版前向公式

如果希望更严格地表达 3D 高斯在成像平面上的 slice response，建议将单高斯单像素 alpha 写成：

$$
\boxed{
\alpha_{g,p}
=
o_g
\exp
\left[
-\frac{1}{2}
\left(
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_{2,g})^T
\boldsymbol{\Sigma}_{2,g}^{-1}
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_{2,g})
+
\frac{\mu_{z,g}^2}{S_{zz,g}}
\right)
\right]
}
$$

其中：

$$
\boldsymbol{\mu}_{2,g}
=
\boldsymbol{\mu}_{p,g}
-
\mathbf{S}_{pz,g}S_{zz,g}^{-1}\mu_{z,g}
$$

$$
\boldsymbol{\Sigma}_{2,g}
=
\mathbf{S}_{pp,g}
-
\mathbf{S}_{pz,g}S_{zz,g}^{-1}\mathbf{S}_{zp,g}
$$

然后 alpha compositing 为：

$$
C_p
=
\sum_{k=1}^{K}
\iota_{g_k}
\alpha_{g_k,p}
T_{k-1}
+
T_Kb
$$

其中：

$$
T_0=1
$$

$$
T_k=T_{k-1}(1-\alpha_{g_k,p})
$$

---

## 10. 推荐的修正版 `mu_2` 反向公式

定义：

$$
\boldsymbol{\delta}
=
\boldsymbol{\xi}
-
\boldsymbol{\mu}_2
$$

$$
q
=
\boldsymbol{\delta}^T
\mathbf{I}_2
\boldsymbol{\delta}
$$

$$
\mathrm{base}
=
\exp
\left(
-\frac{1}{2}q
\right)
$$

$$
\alpha
=
o\cdot \mathrm{base}
$$

则：

$$
\bar{\mathrm{base}}
=
\bar{\alpha}\cdot o
$$

$$
\bar q
=
-\frac{1}{2}\bar{\mathrm{base}}\cdot \mathrm{base}
$$

$$
\boxed{
\bar{\boldsymbol{\mu}}_2
=
-2\bar q\mathbf{I}_2\boldsymbol{\delta}
}
$$

展开为：

$$
\bar{\mu}_{2,\mathrm{lat}}
=
-2\bar q
\left(
i_{00}\delta_{\mathrm{lat}}
+
i_{01}\delta_{\mathrm{ax}}
\right)
$$

$$
\bar{\mu}_{2,\mathrm{ax}}
=
-2\bar q
\left(
i_{01}\delta_{\mathrm{lat}}
+
i_{11}\delta_{\mathrm{ax}}
\right)
$$

对 inverse entries：

$$
\bar i_{00}
=
\bar q\delta_{\mathrm{lat}}^2
$$

$$
\bar i_{01}
=
2\bar q\delta_{\mathrm{lat}}\delta_{\mathrm{ax}}
$$

$$
\bar i_{11}
=
\bar q\delta_{\mathrm{ax}}^2
$$

---

## 11. 如果加入 elev attenuation，新增反向项

若改为：

$$
q_{\mathrm{total}}
=
q_{\mathrm{cond}}
+
q_z
$$

其中：

$$
q_z
=
\frac{\mu_z^2}{S_{zz}}
$$

则：

$$
\alpha
=
o\exp
\left(
-\frac{1}{2}q_{\mathrm{total}}
\right)
$$

新增的反向项为：

$$
\bar{\mu}_z
\mathrel{+}=
\bar q_{\mathrm{total}}
\cdot
\frac{2\mu_z}{S_{zz}}
$$

$$
\bar S_{zz}
\mathrel{+}=
\bar q_{\mathrm{total}}
\cdot
\left(
-\frac{\mu_z^2}{S_{zz}^2}
\right)
$$

其中：

$$
\bar q_{\mathrm{total}}
=
-\frac{1}{2}\bar{\mathrm{base}}\cdot \mathrm{base}
$$

注意：原本 Schur 条件均值和条件协方差也已经依赖 $\mu_z$、$S_{zz}$、$S_{pz}$，所以这里的新增项需要和 Schur backward 的梯度累加在一起。

---

## 12. 建议的优先级

### 最高优先级

1. 修正 `delta` / `mu_2` 反向公式。
2. 检查 CUDA 代码中 `ours_plane_cond2d_bwd_wire` 是否和错误文档一致。
3. 若代码确实用了错误公式，优先修代码并做 finite-difference gradient check。

---

### 第二优先级

4. 决定 `plane_cond2d` 的真实建模目标：
   - 如果目标是 conditional projection，则当前前向可以保留，但文档不要称为严格 slice cross-section；
   - 如果目标是 3D Gaussian slice reconstruction，则加入 elev attenuation。

---

### 第三优先级

5. 修正 tile mapper 与 shader 中心不一致问题。
6. 对 mapper radius 做 conservative expansion，避免漏贡献。

---

### 第四优先级

7. 在文档中明确所有 hard branch、clamp、early stop 的不可微行为。
8. 明确 opacity 的范围约束。
9. 明确 symmetric packed gradient 的语义。
10. 若要优化 pose，考虑改为严格 SO(3) / SE(3) 参数化。

---

## 13. 最终建议表

| 模块 | 当前问题 | 严重程度 | 建议 |
|---|---|---:|---|
| `plane_cond2d` forward | 缺少 elev attenuation | 高 | 若做严格 slice，加入 $\exp(-0.5\mu_z^2/S_{zz})$ |
| Tile mapper | 使用无条件中心，shader 使用条件中心 | 高 | mapper 改用 $\mu_2$ 或扩大半径 |
| `mu_2` backward | `delta` 梯度符号和系数错误 | 最高 | 改为 $\bar\mu_2=-2\bar qI\delta$ |
| inverse backward | 未考虑 inv_det clamp | 中 | 文档注明近似，或实现 clamp-aware backward |
| `q_+=max(q,0)` | 反向需考虑 indicator | 中 | 加入 branch-aware gradient |
| hard culling | 不可微 | 中 | 文档明确梯度置零 |
| pose | 非严格 SE(3) | 中 | 若优化 pose，换 SO(3) 参数化 |
| opacity | alpha 范围未明确 | 中 | sigmoid / clamp / 文档说明 |
| 符号 | `S_{05}` 混用 flattened index | 低 | 改成 `S_{12}` 或 `Sc[5]` |

---

## 14. 推荐最终表述

建议在文档开头加入如下说明：

> 本文档描述的是当前 CUDA 实现中的 `plane_cond2d` 条件平面高斯 shader。该 shader 使用 3D 高斯在成像平面 $z=0$ 条件下的 2D 条件均值与条件协方差来计算平面内响应。需要注意，若不额外加入 $\exp(-\frac12\mu_z^2/S_{zz})$ 项，该模型并不等价于严格的 3D Gaussian density 在切片平面上的 cross-section，而更接近一种 conditional projection heuristic。

同时建议把当前 alpha 公式改写为两种模式：

### Conditional projection mode

$$
\alpha
=
o
\exp
\left(
-\frac12q_{\mathrm{cond}}
\right)
$$

### Strict slice cross-section mode

$$
\alpha
=
o
\exp
\left[
-\frac12
\left(
q_{\mathrm{cond}}
+
\frac{\mu_z^2}{S_{zz}}
\right)
\right]
$$

这样可以避免后续在 method 写作和代码实现之间产生概念混淆。
