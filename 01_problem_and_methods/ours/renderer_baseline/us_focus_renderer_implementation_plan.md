# US-Focus Gaussian Renderer 重构落地设计文档

面向 Codex / Cursor / 服务器端代码落地使用。

本文档目标：在现有 `ours_renderer` 基础上，新增一个更适合 freehand ultrasound slice reconstruction 的稳定渲染路径：

```text
shader_weight_mode = "us_focus"
compositing_mode  = "additive"
mapper_kind       = "tile"  # 首选
```

核心思想：不要继续使用 `plane_cond2d + alpha compositing` 作为主路线，而是参考 slice-based focus-aware Gaussian rendering 的思想，重构为 **A-line-aware / probe-conditioned / focus-weighted additive Gaussian accumulation**。

---

## 0. 为什么要重构

当前 `plane_cond2d + alpha` 路径已经比原始 3D 马氏距离更合理，但仍存在以下工程和数学风险：

1. 条件高斯依赖 Schur complement：

   $$
   \mu_2
   =
   \mu_p
   -
   \frac{\mu_z}{S_{zz}}S_{pz}
   $$

   当 \(S_{zz}\) 较小或 \(S_{pz}\) 较大时，条件中心可能大幅漂移，导致 tile 覆盖、排序和梯度都很敏感。

2. Alpha compositing 需要 depth sorting：

   $$
   C \leftarrow C + I\alpha T,\qquad
   T \leftarrow T(1-\alpha)
   $$

   一旦 \(\alpha>1\)、排序错误或 opacity 饱和，就容易出现大块遮挡、负 transmittance、LPIPS 爆炸。

3. 超声 slice 成像并不是可见光 front-to-back occlusion，更像多个散射贡献在成像平面上的加性累积。

4. 当前大块伪影更可能来自：
   - opacity 没有稳定约束；
   - alpha 饱和；
   - additive / alpha 配置混乱；
   - LPIPS 输入范围失控；
   - scale / covariance 梯度推歪。

因此建议新增一条稳定主线：

> 用 focus-weighted 2D Gaussian footprint 做 pixel contribution，用 additive accumulation 替代 alpha compositing，用 ultrasound-specific A-line / depth-adaptive focus 作为创新点。

---

## 1. 新路线总览

### 1.1 新 shader 名称

建议新增：

```text
shader_weight_mode = "us_focus"
```

其含义为：

> Ultrasound A-line-aware focus Gaussian contribution.

### 1.2 新 compositing 模式

建议首选：

```text
compositing_mode = "additive"
```

前向：

$$
I(p)
=
\sum_{g\in\mathcal{N}(p)}
\iota_g w_{g,p}
+
b
$$

其中：

- \(p\) 是当前 slice 上的像素；
- \(g\) 是高斯；
- \(\iota_g\) 是高斯强度；
- \(w_{g,p}\) 是高斯到像素的 focus-aware contribution weight；
- \(b\) 是可选 background intensity。

### 1.3 当前阶段不要使用 alpha transmittance

暂时不要做：

$$
T \leftarrow T(1-\alpha)
$$

因为这引入排序、alpha 范围、早停、replay backward 等大量复杂性。

超声 slice reconstruction 更合理的假设是：

> 每个像素强度由局部散射体贡献加性叠加，而不是前景遮挡背景。

---

## 2. 坐标系设定

### 2.1 世界坐标系高斯

每个高斯 \(g\) 有：

$$
\boldsymbol{\mu}_g^w\in\mathbb{R}^3
$$

$$
\mathbf{s}_g=(s_x,s_y,s_z)
$$

$$
\mathbf{q}_g
$$

$$
o_g
$$

$$
\iota_g
$$

世界协方差：

$$
\boldsymbol{\Sigma}_g^w
=
\mathbf{R}(\mathbf{q}_g)
\operatorname{diag}(s_x^2,s_y^2,s_z^2)
\mathbf{R}(\mathbf{q}_g)^T
$$

### 2.2 Probe / camera 坐标系

对第 \(i\) 张 slice，pose 给出 probe-to-world 变换。

定义 probe 三个单位轴：

$$
\hat{\mathbf{a}}_{\mathrm{lat},i},
\quad
\hat{\mathbf{a}}_{\mathrm{ax},i},
\quad
\hat{\mathbf{a}}_{\mathrm{ele},i}
$$

组成：

$$
\mathbf{A}_i
=
\begin{bmatrix}
\hat{\mathbf{a}}_{\mathrm{lat},i}^T\\
\hat{\mathbf{a}}_{\mathrm{ax},i}^T\\
\hat{\mathbf{a}}_{\mathrm{ele},i}^T
\end{bmatrix}
$$

探头坐标下高斯中心：

$$
\boldsymbol{\mu}_{c,g}
=
\mathbf{A}_i
(\boldsymbol{\mu}_g^w-\mathbf{t}_i)
=
\begin{bmatrix}
\mu_{\mathrm{lat}}\\
\mu_{\mathrm{ax}}\\
\mu_{\mathrm{ele}}
\end{bmatrix}
$$

探头坐标下协方差：

$$
\boldsymbol{\Sigma}_{c,g}
=
\mathbf{A}_i
\boldsymbol{\Sigma}_g^w
\mathbf{A}_i^T
$$

后续所有 `us_focus` 计算都在 probe-local frame 中完成。

---

## 3. 像素坐标与 A-line 结构

当前超声图像坐标：

- `col = x`：横向 lateral index；
- `row = y`：轴向 depth index；
- `H`：轴向采样数；
- `W`：横向 A-line 数；
- 每一列近似对应一条 A-line。

像素 \(p=(row,col)\) 对应的 probe-local 物理坐标：

$$
\xi_{\mathrm{lat}}(col)
=
(col-w_{\mathrm{off}})\Delta_{\mathrm{lat}}
$$

$$
\xi_{\mathrm{ax}}(row)
=
z_{\mathrm{near}}+row\Delta_{\mathrm{ax}}
$$

像素点：

$$
\boldsymbol{\xi}_p
=
\begin{bmatrix}
\xi_{\mathrm{lat}}\\
\xi_{\mathrm{ax}}
\end{bmatrix}
$$

注意：像素本身位于 slice 平面，因此：

$$
\xi_{\mathrm{ele}}=0
$$

---

## 4. 基础版 US-Focus 模型

### 4.1 设计思想

我们不再做条件高斯：

$$
p(\mathrm{lat},\mathrm{ax}\mid \mathrm{ele}=0)
$$

而是做：

> probe-local 3D Gaussian 被 ultrasound elevational focus kernel 调制后，在 image plane 上形成 2D footprint。

也就是：

$$
g_{\mathrm{focus}}(\mathbf{x})
=
g_c(\mathbf{x})
\cdot
h_{\mathrm{US}}(\mathbf{x})
$$

其中 \(h_{\mathrm{US}}\) 是超声成像的 focus / slice sensitivity。

### 4.2 最简单稳定版 focus kernel

基础版只对 elevational 方向加 focus prior：

$$
h_{\mathrm{US}}(\mathbf{x})
=
\exp
\left(
-\frac{1}{2}
\frac{x_{\mathrm{ele}}^2}
{\sigma_{\mathrm{ele}}^2}
\right)
$$

对应 precision 注入：

$$
\boldsymbol{\Lambda}_f
=
\begin{bmatrix}
0 & 0 & 0\\
0 & 0 & 0\\
0 & 0 & \frac{1}{\sigma_{\mathrm{ele}}^2}
\end{bmatrix}
$$

原始高斯 precision：

$$
\boldsymbol{\Lambda}_c
=
\boldsymbol{\Sigma}_c^{-1}
$$

Focus Gaussian precision：

$$
\boldsymbol{\Lambda}_e
=
\boldsymbol{\Lambda}_c
+
\boldsymbol{\Lambda}_f
$$

Focus Gaussian covariance：

$$
\boldsymbol{\Sigma}_e
=
\boldsymbol{\Lambda}_e^{-1}
$$

Focus Gaussian mean：

$$
\boldsymbol{\mu}_e
=
\boldsymbol{\Sigma}_e
\boldsymbol{\Lambda}_c
\boldsymbol{\mu}_c
$$

这一步等价于两个高斯相乘后的新高斯参数。

---

## 5. Ultrasound 创新版：depth-adaptive focus

为了和通用 GaussianPile 区分，建议不要使用常数 \(\sigma_z\)。对于超声，更合理的是深度相关的 elevational focus：

$$
\sigma_{\mathrm{ele}}
=
\sigma_{\mathrm{ele}}(a)
$$

其中：

$$
a = \mu_{\mathrm{ax}}
$$

或者：

$$
a = \xi_{\mathrm{ax}}(row)
$$

推荐初版用高斯中心的 axial depth：

$$
a_g = \mu_{\mathrm{ax},g}
$$

于是：

$$
\sigma_{\mathrm{ele}}(a_g)
=
\sigma_{\mathrm{ele},0}
+
\sigma_{\mathrm{ele},1}
\cdot
\max(a_g-z_{\mathrm{near}},0)
$$

再做 clamp：

$$
\sigma_{\mathrm{ele}}
=
\operatorname{clamp}
(
\sigma_{\mathrm{ele}},
\sigma_{\min},
\sigma_{\max}
)
$$

推荐默认参数：

```text
sigma_ele_0   = 1.0 mm
sigma_ele_1   = 0.00 ~ 0.03
sigma_ele_min = 0.5 mm
sigma_ele_max = 8.0 mm
```

如果想更稳定，第一版直接用常数：

```text
sigma_ele_const = slice_thickness_mm / 2
```

后续再升级为 depth-adaptive。

---

## 6. 更强创新版：A-line-aware focus precision

超声 beam 也可以对 lateral 方向做 depth-dependent beam width 建模：

$$
\sigma_{\mathrm{lat}}=\sigma_{\mathrm{lat}}(a)
$$

那么 focus precision 可以写成：

$$
\boldsymbol{\Lambda}_f(a)
=
\operatorname{diag}
\left(
\frac{1}{\sigma_{\mathrm{lat}}(a)^2},
0,
\frac{1}{\sigma_{\mathrm{ele}}(a)^2}
\right)
$$

注意：不建议初版对 axial 方向注入 focus precision，因为 axial 方向本来就是图像 row 方向，需要保留较强解析度。

其中：

$$
\sigma_{\mathrm{lat}}(a)
=
\sigma_{\mathrm{lat},0}
+
\sigma_{\mathrm{lat},1}
\cdot
\max(a-z_{\mathrm{near}},0)
$$

最终：

$$
\boldsymbol{\Lambda}_e
=
\boldsymbol{\Sigma}_c^{-1}
+
\boldsymbol{\Lambda}_f(a)
$$

这就是一个 ultrasound-specific 的改动，区别于通用 slice PSF。

---

## 7. 2D footprint 构造

从 Focus Gaussian 取 image-plane block：

$$
\boldsymbol{\mu}_{2d}
=
\begin{bmatrix}
\mu_{e,\mathrm{lat}}\\
\mu_{e,\mathrm{ax}}
\end{bmatrix}
$$

$$
\boldsymbol{\Sigma}_{2d}
=
\begin{bmatrix}
\Sigma_e[0,0] & \Sigma_e[0,1]\\
\Sigma_e[1,0] & \Sigma_e[1,1]
\end{bmatrix}
$$

2D inverse：

$$
\mathbf{Q}_{2d}
=
\boldsymbol{\Sigma}_{2d}^{-1}
$$

像素偏移：

$$
\mathbf{d}_{g,p}
=
\boldsymbol{\xi}_p
-
\boldsymbol{\mu}_{2d,g}
$$

二次型：

$$
q_{g,p}
=
\mathbf{d}_{g,p}^T
\mathbf{Q}_{2d,g}
\mathbf{d}_{g,p}
$$

plane footprint：

$$
\phi_{g,p}
=
\exp
\left(
-\frac{1}{2}q_{g,p}
\right)
$$

---

## 8. Soft slice response / opacity response

### 8.1 不建议直接照抄 GaussianPile 的 opacity_r

GaussianPile 使用：

$$
\mathrm{opacity}_r
=
\exp
\left[
-\frac12
\left(
\mu_c^T\Sigma_c^{-1}\mu_c
-
\mu_e^T\Sigma_e^{-1}\mu_e
\right)
\right]
$$

这在数学上来自高斯乘积的常数项。

但是为了形成自己的方法，建议用 ultrasound soft slice response：

$$
r_{g,i}
=
\exp
\left(
-\frac{1}{2}
\frac{\mu_{\mathrm{ele},g}^2}
{\sigma_{\mathrm{ele},g}^2 + \epsilon}
\right)
$$

其中：

$$
\mu_{\mathrm{ele},g}
=
(\mu_g^w-t_i)^T
\hat{\mathbf{a}}_{\mathrm{ele},i}
$$

也可以加入高斯自身 elevational variance：

$$
r_{g,i}
=
\exp
\left(
-\frac{1}{2}
\frac{\mu_{\mathrm{ele},g}^2}
{\Sigma_c[2,2]+\sigma_{\mathrm{ele}}(a_g)^2+\epsilon}
\right)
$$

推荐初版用第二个，更稳定：

$$
\boxed{
r_{g,i}
=
\exp
\left(
-\frac{1}{2}
\frac{\mu_{\mathrm{ele},g}^2}
{\Sigma_c[2,2]+\sigma_{\mathrm{ele}}(a_g)^2+\epsilon}
\right)
}
$$

这样如果高斯自身 elevational scale 较大，不会被过度衰减。

### 8.2 物理解释

这个项表示：

> 一个高斯中心距离当前 ultrasound slice 平面越远，它对当前 slice 的贡献越小；但如果该高斯本身在 elevational 方向更宽，或当前深度的 beam slice 更厚，则衰减更慢。

这比单纯 corridor hard culling 更平滑、更可微。

---

## 9. 能量归一化

为了避免 footprint 变大时总亮度也变大，建议加入 2D determinant normalization：

$$
n_g
=
\frac{1}
{\sqrt{\det(\boldsymbol{\Sigma}_{2d,g})+\epsilon}}
$$

然后：

$$
w_{g,p}
=
o_g
\cdot
r_{g,i}
\cdot
n_g
\cdot
\phi_{g,p}
$$

如果发现图像过暗或过亮，可以把归一化改成可控版本：

$$
n_g
=
\frac{1}
{(\det(\boldsymbol{\Sigma}_{2d,g})+\epsilon)^\gamma}
$$

其中：

```text
gamma = 0.0  # 不归一化
gamma = 0.25
gamma = 0.5  # GaussianPile 风格
```

推荐第一版：

```text
gamma = 0.25
```

比 \(0.5\) 温和，避免 det 太小时亮度爆炸。

---

## 10. 最终像素贡献公式

### 10.1 单高斯单像素权重

$$
\boxed{
w_{g,p}
=
o_g
\cdot
r_{g,i}
\cdot
\frac{1}
{(\det(\boldsymbol{\Sigma}_{2d,g})+\epsilon)^\gamma}
\cdot
\exp
\left(
-\frac{1}{2}
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_{2d,g})^T
\boldsymbol{\Sigma}_{2d,g}^{-1}
(\boldsymbol{\xi}_p-\boldsymbol{\mu}_{2d,g})
\right)
}
$$

### 10.2 像素强度

$$
\boxed{
I_i(p)
=
b
+
\sum_{g\in\mathcal{N}(p)}
\iota_g
w_{g,p}
}
$$

其中：

- \(b\)：background；
- \(\iota_g\)：高斯 intensity；
- \(o_g\)：高斯 occupancy / opacity；
- \(r_{g,i}\)：soft slice response；
- \(\mathcal{N}(p)\)：tile mapper 筛选出的候选高斯集合。

---

## 11. 参数约束建议

为避免再次出现 LPIPS 爆炸，强烈建议训练侧使用有界参数化。

### 11.1 Opacity / occupancy

不要直接优化 raw opacity。

推荐：

$$
o_g
=
o_{\max}
\cdot
\sigma(\theta^o_g)
$$

训练初期：

```python
opacity_max = 0.2
```

稳定后：

```python
opacity_max = 1.0
```

### 11.2 Intensity

若目标图像归一化到 \([0,1]\)，推荐：

$$
\iota_g=\sigma(\theta^\iota_g)
$$

或者允许轻微超出：

$$
\iota_g=1.5\sigma(\theta^\iota_g)
$$

### 11.3 Scale

推荐：

$$
s_g
=
s_{\min}
+
(s_{\max}-s_{\min})\sigma(\theta^s_g)
$$

初始建议：

```python
scale_min = 0.05 mm
scale_max = 3.0 mm
```

需要根据数据 spacing 调整。

---

## 12. CUDA 实现计划

以下按现有 `ours_renderer` 结构给出建议修改点。实际文件名可根据仓库当前情况微调。

---

### 12.1 新增 shader id

在 Python 和 CUDA 的 shader mode enum 中加入：

```text
SHADER_WEIGHT_US_FOCUS = 2
```

Python config：

```python
shader_weight_mode: str = "us_focus"
```

映射：

```python
{
    "probe3d": 0,
    "plane_cond2d": 1,
    "us_focus": 2,
}
```

建议位置：

```text
ours_renderer/config.py
ours_renderer/autograd.py
src/shader/wrapper_shader.cu
include/shader/shared.cuh
```

---

### 12.2 新增 device header

新增文件：

```text
include/shader/us_focus_device.cuh
```

建议包括以下函数：

```cpp
#pragma once

struct UsFocusEvalOut {
    float weight;      // w_g,p
    float intensity;   // iota_g
    float mu2_lat;
    float mu2_ax;
    float sig2_00;
    float sig2_01;
    float sig2_11;
    float inv2_00;
    float inv2_01;
    float inv2_11;
    float det2;
    float slice_response;
    float norm_factor;
    float quad;
};

__device__ inline bool us_focus_eval_at_pixel(
    int col,
    int row,
    int g,
    const float* gauss_means,
    const float* gauss_scales,
    const float* gauss_quats,
    const float* gauss_opacities,
    const float* gauss_intensities,
    const float* pose,
    const float* slice_grid_pack,
    const float* us_focus_params,
    UsFocusEvalOut* out
);
```

---

### 12.3 us_focus_params 设计

新增一个小参数数组，例如：

```text
us_focus_params[0] = sigma_ele_0
us_focus_params[1] = sigma_ele_1
us_focus_params[2] = sigma_ele_min
us_focus_params[3] = sigma_ele_max
us_focus_params[4] = sigma_lat_0
us_focus_params[5] = sigma_lat_1
us_focus_params[6] = sigma_lat_min
us_focus_params[7] = sigma_lat_max
us_focus_params[8] = det_gamma
us_focus_params[9] = det_eps
us_focus_params[10] = weight_min
us_focus_params[11] = use_lateral_focus  # 0 or 1
```

第一版可以只使用 elevational focus：

```text
use_lateral_focus = 0
```

---

### 12.4 核心前向 device 逻辑

伪代码：

```cpp
__device__ inline bool us_focus_eval_at_pixel(...) {
    // 1. load Gaussian params
    float3 mu_w = ...
    float3 scale = max(scale, 1e-6)
    float4 quat = ...

    // 2. build Rg, Sigma_w
    float Rg[9];
    quat_to_rot(quat, Rg);

    float Sigma_w[9];
    mat3_sigma_world(Rg, scale, Sigma_w);

    // 3. read probe axes and translation
    float3 a_lat, a_ax, a_ele, t;
    read_pose_axes_normalized(pose, &a_lat, &a_ax, &a_ele, &t);

    // 4. transform mean to probe frame
    float3 d = mu_w - t;
    float mu_lat = dot(d, a_lat);
    float mu_ax  = dot(d, a_ax);
    float mu_ele = dot(d, a_ele);

    // 5. build Sigma_c = A Sigma_w A^T
    float Sc[9];
    build_sigma_c_probe(Sigma_w, a_lat, a_ax, a_ele, Sc);
    add_jitter(Sc);

    // 6. invert Sc -> Lambda_c
    float Lc[9];
    if (!invert3x3_spd(Sc, Lc)) return false;

    // 7. compute depth-adaptive focus sigmas
    float depth = fmaxf(mu_ax - near_mm, 0.f);
    float sigma_ele = clamp(sigma_ele_0 + sigma_ele_1 * depth,
                            sigma_ele_min, sigma_ele_max);

    float sigma_lat = clamp(sigma_lat_0 + sigma_lat_1 * depth,
                            sigma_lat_min, sigma_lat_max);

    // 8. Lambda_e = Lambda_c + Lambda_f
    float Le[9] = Lc;
    if (use_lateral_focus) {
        Le[0] += 1.f / (sigma_lat * sigma_lat);
    }
    Le[8] += 1.f / (sigma_ele * sigma_ele);

    // 9. Sigma_e = inv(Le)
    float Se[9];
    if (!invert3x3_spd(Le, Se)) return false;

    // 10. mu_e = Se * Lc * mu_c
    float mu_c[3] = {mu_lat, mu_ax, mu_ele};
    float tmp[3] = Lc * mu_c;
    float mu_e[3] = Se * tmp;

    // 11. 2D footprint
    float mu2_lat = mu_e[0];
    float mu2_ax  = mu_e[1];

    float S2_00 = Se[0];
    float S2_01 = Se[1];
    float S2_11 = Se[4];

    // 12. invert 2D covariance
    float det2 = S2_00 * S2_11 - S2_01 * S2_01;
    if (det2 <= det_eps) return false;

    float inv_det2 = 1.f / det2;
    float Q00 =  S2_11 * inv_det2;
    float Q01 = -S2_01 * inv_det2;
    float Q11 =  S2_00 * inv_det2;

    // 13. pixel mm coordinate
    float xi_lat, xi_ax;
    sg_linear_pixel_to_plane_mm(col, row, slice_grid_pack, &xi_lat, &xi_ax);

    float dx = xi_lat - mu2_lat;
    float dy = xi_ax  - mu2_ax;

    float quad = Q00 * dx * dx + 2.f * Q01 * dx * dy + Q11 * dy * dy;
    if (!isfinite(quad) || quad < 0.f) quad = fmaxf(quad, 0.f);

    float phi = expf(-0.5f * quad);

    // 14. soft slice response
    float denom_ele = Sc[8] + sigma_ele * sigma_ele + 1e-6f;
    float slice_response = expf(-0.5f * mu_ele * mu_ele / denom_ele);

    // 15. determinant normalization
    float norm = powf(det2 + det_eps, -det_gamma);

    // 16. weight
    float o = gauss_opacities[g];
    float weight = o * slice_response * norm * phi;

    if (!isfinite(weight) || weight < weight_min) return false;

    out->weight = weight;
    out->intensity = gauss_intensities[g];
    ...
    return true;
}
```

---

## 13. Mapper 设计

### 13.1 推荐首版只支持 tile mapper

建议第一阶段不要改 pixel mapper，避免复杂度爆炸。

配置：

```python
mapper_kind = "tile"
```

### 13.2 Tile mapper footprint

mapper 需要使用与 shader 一致的 `mu2` 和 `Sigma2d` 来估计 coverage。

对于每个高斯：

1. 计算 `us_focus` 的 Focus Gaussian；
2. 得到：

   $$
   \mu_{2d}
   $$

   $$
   \Sigma_{2d}
   $$

3. 估计 2D ellipse 的 AABB。

保守半径：

$$
r_{\mathrm{lat}}
=
k_\sigma
\sqrt{\Sigma_{2d}[0,0]}
$$

$$
r_{\mathrm{ax}}
=
k_\sigma
\sqrt{\Sigma_{2d}[1,1]}
$$

如果考虑 off-diagonal 更严格，可用 eigenvalue：

$$
r
=
k_\sigma
\sqrt{\lambda_{\max}(\Sigma_{2d})}
$$

第一版推荐简单 AABB：

```cpp
r_lat = k_sigma * sqrtf(max(S2_00, eps));
r_ax  = k_sigma * sqrtf(max(S2_11, eps));
```

pixel center：

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

tile range：

```cpp
min_tx = floor((u_pix - r_pix_lat) / tile_size)
max_tx = ceil ((u_pix + r_pix_lat) / tile_size)
min_ty = floor((v_pix - r_pix_ax ) / tile_size)
max_ty = ceil ((v_pix + r_pix_ax ) / tile_size)
```

### 13.3 Additive 不需要 depth sorting

对于 `us_focus + additive`：

```text
sort_by_v = false
```

也可以继续按 tile id 排序，但 depth key 不需要影响合成结果。

CSR 只需要列出每个 tile 的候选高斯。

---

## 14. Forward shader 实现

### 14.1 Additive forward

在 tile CSR shader 中加入分支：

```cpp
if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS &&
    compositing_mode == COMPOSITING_ADDITIVE) {
    ...
}
```

对每个 pixel：

```cpp
float C = fixed_bg_color;

for each g in tile list:
    UsFocusEvalOut out;
    if (!us_focus_eval_at_pixel(..., &out)) continue;

    C += out.intensity * out.weight;

colors[p] = C;
```

### 14.2 输出范围建议

如果训练初期容易爆，可以临时加 debug-only clamp：

```cpp
C = fminf(fmaxf(C, 0.f), 1.f);
```

但不建议长期在 renderer 内 hard clamp，因为会造成梯度不连续。

更好的做法是在参数层控制：

```python
opacity = opacity_max * sigmoid(opacity_logit)
intensity = sigmoid(intensity_logit)
```

### 14.3 性能版 tile kernel：2D launch + shared memory

在质量闭环跑通后，`us_focus + additive` 的生产版 forward/backward 应改成 3DGS 风格的 tile-centric kernel：

```text
gridDim.x = num_tiles_x
gridDim.y = num_tiles_y
blockDim  = 2D pixel block inside one tile
```

每个 CUDA block 对应一个 image tile，block 内线程对应 tile 内像素。由于同一个 tile 内所有像素共享同一段候选高斯列表，kernel 内应分批把候选高斯的参数或预计算 footprint 加载到 shared memory：

```cpp
for candidate batch in tile_csr[tile]:
    load gaussian ids / footprint / opacity / intensity into shared memory
    __syncthreads()

    each pixel thread accumulates all candidates in the shared batch
    __syncthreads()
```

这样可以避免当前 pixel-centric 循环中对同一 tile 候选高斯的重复 global memory 读取，并为后续复用预计算 footprint 提供结构。首版 shared memory 内容建议优先缓存：

```text
gauss_id
mu2_lat, mu2_ax
inv2_00, inv2_01, inv2_11
det2 / norm_factor
slice_response
opacity, intensity
r_lat_mm, r_ax_mm
```

如果 shared memory 压力过高，可以先只缓存 `gauss_id + footprint compact struct`，完整高斯参数仍从 global 读取；性能验证后再决定是否进一步压缩字段或使用 half/packed layout。

该优化应同时规划 forward 和 backward：

- forward：一个 tile block 输出 tile 内所有像素颜色；
- backward：同样按 tile block 遍历候选高斯，像素线程计算局部贡献，再对高斯梯度做 block 内 reduce 后 atomicAdd 到 global grad；
- additive 路径不需要 depth sorting，因此 shared-memory batch 不需要维护 transmittance replay；
- 解析 scale/quaternion 梯度完成后再做该 kernel，避免把有限差分瓶颈固化进高速路径。

---

## 15. Backward 实现建议

### 15.1 第一阶段：先用 PyTorch reference 验证前向

在写复杂 CUDA backward 前，建议先做一个小规模 PyTorch reference：

```python
def us_focus_reference(
    means_w,
    scales,
    quats,
    opacities,
    intensities,
    pose,
    grid_pack,
    params,
):
    ...
```

对小规模：

```text
N = 10 ~ 100
H = 32
W = 32
```

验证 CUDA forward 与 PyTorch reference 一致。

### 15.2 第二阶段：CUDA backward 可以先只回传部分参数

为了快速稳定训练，可以第一版只回传：

- intensity；
- opacity；
- means。

暂时不回传：

- scale；
- quat；
- pose。

这样可以快速验证图像是否稳定，避免 covariance backward 复杂错误导致训练崩。

### 15.3 推荐 staged backward

#### Stage 1

```text
grad_intensity
grad_opacity
```

公式：

$$
I(p)=\sum_g \iota_g w_{g,p}
$$

所以上游梯度 \(\bar I_p\)：

$$
\bar{\iota}_g
\mathrel{+}=
\bar I_p w_{g,p}
$$

$$
\bar{o}_g
\mathrel{+}=
\bar I_p \iota_g
\frac{w_{g,p}}{o_g+\epsilon}
$$

#### Stage 2

加入 mean gradient。

对：

$$
w=o\cdot r\cdot n\cdot \phi
$$

$$
\phi=\exp(-\frac12 q)
$$

$$
q=d^TQd
$$

有：

$$
\bar q
=
\bar w
\cdot
o r n
\cdot
\left(
-\frac12\phi
\right)
$$

$$
\bar d
=
2\bar q Qd
$$

$$
\bar\mu_{2d}
=
-\bar d
$$

再通过：

$$
\mu_e
=
\Sigma_e\Lambda_c\mu_c
$$

回传到 \(\mu_c\)，最后：

$$
\bar\mu^w
=
A^T \bar\mu_c
$$

如果不想第一版实现完整链条，可以先近似：

```cpp
grad_mu_lat += grad_mu2_lat;
grad_mu_ax  += grad_mu2_ax;
grad_mu_ele += grad_slice_response;
```

但论文级方法最终需要完整链式。

#### Stage 3

加入 scale / quat / covariance gradient。

这个阶段需要完整矩阵微分，建议在 PyTorch 中先自动求导对照，再移植 CUDA。

---

## 16. us_focus 的完整反向数学框架

本节给 Codex 作为后续实现参考。第一版不一定全部实现。

### 16.1 前向定义

$$
\Lambda_c=\Sigma_c^{-1}
$$

$$
\Lambda_e=\Lambda_c+\Lambda_f
$$

$$
\Sigma_e=\Lambda_e^{-1}
$$

$$
\mu_e=\Sigma_e\Lambda_c\mu_c
$$

$$
\Sigma_2=P\Sigma_eP^T
$$

$$
\mu_2=P\mu_e
$$

其中：

$$
P=
\begin{bmatrix}
1&0&0\\
0&1&0
\end{bmatrix}
$$

$$
Q=\Sigma_2^{-1}
$$

$$
d=\xi-\mu_2
$$

$$
q=d^TQd
$$

$$
\phi=\exp(-q/2)
$$

$$
r=\exp
\left(
-\frac12\frac{\mu_{\mathrm{ele}}^2}
{\Sigma_c[2,2]+\sigma_{\mathrm{ele}}^2+\epsilon}
\right)
$$

$$
n=(\det\Sigma_2+\epsilon)^{-\gamma}
$$

$$
w=o r n \phi
$$

$$
I=\sum_g \iota_g w
$$

### 16.2 从 pixel loss 到 w

给定：

$$
\bar I_p=\frac{\partial L}{\partial I_p}
$$

则：

$$
\bar\iota
\mathrel{+}=
\bar I_p w
$$

$$
\bar w
=
\bar I_p \iota
$$

### 16.3 w 到各项

$$
\bar o
\mathrel{+}=
\bar w \cdot r n \phi
$$

$$
\bar r
=
\bar w \cdot o n \phi
$$

$$
\bar n
=
\bar w \cdot o r \phi
$$

$$
\bar\phi
=
\bar w \cdot o r n
$$

### 16.4 phi 到 q

$$
\bar q
=
\bar\phi
\left(
-\frac12\phi
\right)
$$

### 16.5 q 到 d 和 Q

$$
\bar d
=
2\bar q Qd
$$

$$
\bar Q
=
\bar q dd^T
$$

$$
\bar\mu_2
=
-\bar d
$$

### 16.6 Q 到 Sigma2

因为：

$$
Q=\Sigma_2^{-1}
$$

所以：

$$
\bar\Sigma_2^{(Q)}
=
-Q\bar Q Q
$$

### 16.7 determinant normalization

$$
n=(\det\Sigma_2+\epsilon)^{-\gamma}
$$

$$
\bar{\det}
=
\bar n
\cdot
(-\gamma)
(\det\Sigma_2+\epsilon)^{-\gamma-1}
$$

对于 2x2：

$$
\Sigma_2=
\begin{bmatrix}
a&b\\
b&c
\end{bmatrix}
$$

$$
\det=ac-b^2
$$

所以：

$$
\bar a
\mathrel{+}=
\bar{\det} c
$$

$$
\bar b
\mathrel{+}=
-2\bar{\det} b
$$

$$
\bar c
\mathrel{+}=
\bar{\det} a
$$

总的：

$$
\bar\Sigma_2
=
\bar\Sigma_2^{(Q)}
+
\bar\Sigma_2^{(\det)}
$$

### 16.8 slice response 反向

$$
r
=
\exp
\left(
-\frac12
\frac{\mu_{\mathrm{ele}}^2}{D}
\right)
$$

其中：

$$
D=\Sigma_c[2,2]+\sigma_{\mathrm{ele}}^2+\epsilon
$$

则：

$$
\bar u
=
\bar r
\left(
-\frac12 r
\right)
$$

其中：

$$
u=\frac{\mu_{\mathrm{ele}}^2}{D}
$$

所以：

$$
\bar\mu_{\mathrm{ele}}
\mathrel{+}=
\bar u
\frac{2\mu_{\mathrm{ele}}}{D}
$$

$$
\bar D
=
\bar u
\left(
-\frac{\mu_{\mathrm{ele}}^2}{D^2}
\right)
$$

$$
\bar\Sigma_c[2,2]
\mathrel{+}=
\bar D
$$

如果 \(\sigma_{\mathrm{ele}}\) 是 learnable 或 depth-dependent，也需要继续回传到其参数。

### 16.9 mu2 / Sigma2 回传到 mue / Sigmae

$$
\mu_2=P\mu_e
$$

所以：

$$
\bar\mu_e
\mathrel{+}=
P^T\bar\mu_2
$$

$$
\Sigma_2=P\Sigma_eP^T
$$

所以：

$$
\bar\Sigma_e
\mathrel{+}=
P^T\bar\Sigma_2P
$$

### 16.10 Sigmae inverse 回传

$$
\Sigma_e=\Lambda_e^{-1}
$$

所以：

$$
\bar\Lambda_e
=
-\Sigma_e\bar\Sigma_e\Sigma_e
$$

### 16.11 mue 回传

$$
\mu_e
=
\Sigma_e\Lambda_c\mu_c
$$

令：

$$
v=\Lambda_c\mu_c
$$

$$
\mu_e=\Sigma_ev
$$

则：

$$
\bar\Sigma_e
\mathrel{+}=
\bar\mu_e v^T
$$

$$
\bar v
=
\Sigma_e^T\bar\mu_e
$$

$$
\bar\Lambda_c
\mathrel{+}=
\bar v \mu_c^T
$$

$$
\bar\mu_c
\mathrel{+}=
\Lambda_c^T\bar v
$$

注意：这里新增的 \(\bar\Sigma_e\) 还需要继续通过 inverse 回传到 \(\Lambda_e\)。

### 16.12 Lambdae 到 Lambdac 和 focus params

$$
\Lambda_e=\Lambda_c+\Lambda_f
$$

所以：

$$
\bar\Lambda_c
\mathrel{+}=
\bar\Lambda_e
$$

$$
\bar\Lambda_f
\mathrel{+}=
\bar\Lambda_e
$$

### 16.13 Lambdac inverse 回传到 Sigmac

$$
\Lambda_c=\Sigma_c^{-1}
$$

所以：

$$
\bar\Sigma_c
=
-\Lambda_c\bar\Lambda_c\Lambda_c
$$

加上 slice response 对 \(\Sigma_c[2,2]\) 的梯度。

### 16.14 Sigmac / muc 回到世界参数

$$
\mu_c=A(\mu^w-t)
$$

所以：

$$
\bar\mu^w=A^T\bar\mu_c
$$

$$
\bar t=-A^T\bar\mu_c
$$

$$
\Sigma_c=A\Sigma^wA^T
$$

所以：

$$
\bar\Sigma^w=A^T\bar\Sigma_c A
$$

然后按已有代码从：

$$
\bar\Sigma^w
$$

回传到：

$$
\bar s,\bar q
$$

---

## 17. Python config 修改

建议新增：

```python
@dataclass
class OursRenderConfig:
    mapper_kind: str = "tile"
    compositing_mode: str = "additive"
    shader_weight_mode: str = "us_focus"

    us_focus_sigma_ele_0: float = 1.0
    us_focus_sigma_ele_1: float = 0.0
    us_focus_sigma_ele_min: float = 0.5
    us_focus_sigma_ele_max: float = 8.0

    us_focus_sigma_lat_0: float = 1.0
    us_focus_sigma_lat_1: float = 0.0
    us_focus_sigma_lat_min: float = 0.5
    us_focus_sigma_lat_max: float = 8.0

    us_focus_det_gamma: float = 0.25
    us_focus_det_eps: float = 1e-6
    us_focus_weight_min: float = 1e-6
    us_focus_use_lateral_focus: bool = False
```

并在调用 CUDA 前打包成 tensor / meta array。

---

## 18. 训练侧建议

### 18.1 初始训练配置

建议先用稳定配置：

```python
shader_weight_mode = "us_focus"
compositing_mode = "additive"
mapper_kind = "tile"
mapper_sort_by_v = False
```

### 18.2 参数约束

```python
opacity = 0.2 * torch.sigmoid(opacity_logit)
intensity = torch.sigmoid(intensity_logit)
scale = scale_min + (scale_max - scale_min) * torch.sigmoid(scale_logit)
```

### 18.3 Loss

初期不要使用 LPIPS。

先用：

```python
loss = F.l1_loss(pred, gt)
```

或者：

```python
loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * F.mse_loss(pred, gt)
```

稳定后再加：

```python
loss = l1 + lambda_ssim * ssim
```

最后再试 LPIPS。

### 18.4 LPIPS 输入

如果使用 LPIPS：

```python
pred_lpips = pred.clamp(0, 1)
gt_lpips = gt.clamp(0, 1)

pred_lpips = pred_lpips.repeat(1, 3, 1, 1) * 2 - 1
gt_lpips = gt_lpips.repeat(1, 3, 1, 1) * 2 - 1
```

---

## 19. Debug 统计项

每次 forward 建议统计：

```text
pred_min
pred_max
pred_mean
pred_finite

weight_min
weight_max
weight_mean
weight_nonzero_ratio

slice_response_min
slice_response_max
slice_response_mean

det2_min
det2_max
det2_mean

quad_min
quad_max
quad_mean

tiles_touched_mean
tiles_touched_max
candidate_gauss_per_tile_mean
```

尤其关注：

```text
pred_max 是否突然很大
weight_max 是否爆炸
det2 是否接近 0
slice_response 是否大面积接近 1
scale 是否异常大
```

---

## 20. 测试计划

### 20.1 CPU / PyTorch reference test

新增：

```text
tests/test_us_focus_reference.py
```

测试内容：

1. 随机生成小规模高斯；
2. 用 PyTorch 实现 `us_focus` forward；
3. 用 CUDA 实现同样 forward；
4. 比较：

```python
torch.allclose(cuda_out, ref_out, atol=1e-4, rtol=1e-3)
```

### 20.2 Special cases

#### Case 1：高斯正好在 slice 上

```text
mu_ele = 0
```

期望：

```text
slice_response ≈ 1
```

#### Case 2：高斯远离 slice

```text
mu_ele >> sigma_ele
```

期望：

```text
slice_response ≈ 0
```

#### Case 3：更大的 sigma_ele

期望：

```text
远离 slice 的高斯衰减变慢
```

#### Case 4：det_gamma = 0

期望：

```text
不做 determinant normalization
```

#### Case 5：use_lateral_focus = True

期望：

```text
lateral footprint 随 depth focus 参数变化
```

---

## 21. Codex 实施任务清单

### Task 1：新增 config 和 mode mapping

修改：

```text
ours_renderer/config.py
ours_renderer/autograd.py
src/shader/wrapper_shader.cu
include/shader/shared.cuh
```

新增：

```text
shader_weight_mode = "us_focus"
SHADER_WEIGHT_US_FOCUS = 2
```

新增 `us_focus_params` 打包逻辑。

---

### Task 2：新增 `us_focus_device.cuh`

实现：

```cpp
UsFocusEvalOut
us_focus_eval_at_pixel(...)
```

依赖已有工具：

```cpp
quat_to_rot
mat3_sigma_world
build_sigma_c_probe
invert3x3
invert2x2
sg_linear_pixel_to_plane_mm
```

如已有函数不可直接复用，封装成 shared utility。

---

### Task 3：修改 tile mapper

在：

```text
src/mapper/core_tile.cu
```

新增 `us_focus` footprint 计算：

```cpp
if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS) {
    compute_us_focus_footprint_mm(...)
}
```

返回：

```text
mu2_lat
mu2_ax
r_lat
r_ax
```

并用于 tile rect。

---

### Task 4：修改 forward shader

在：

```text
src/shader/forward.cu
```

新增：

```cpp
if (shader_weight_mode == SHADER_WEIGHT_US_FOCUS &&
    compositing_mode == COMPOSITING_ADDITIVE) {
    C += intensity * weight;
}
```

第一版只支持 additive。

如果用户误用：

```text
us_focus + alpha
```

直接报错或 fallback 到 additive。

### Task 4.5：新增生产版 2D tile launch kernel

在正确性版本通过后，为 `us_focus + additive + tile` 新增 tile-centric CUDA kernel：

```text
one block  = one tile
2D threads = pixels inside tile
shared memory = batched tile candidate Gaussian footprint / params
```

目标：

```text
1M 初始高斯时达到 130-150 optimization iter/s
```

实现顺序：

1. forward 先改为 tile block 输出整块 tile；
2. 使用 shared memory 分批缓存 tile 候选高斯 footprint；
3. backward 用相同 tile block 结构，像素线程累积局部贡献；
4. block 内 reduce 高斯梯度，最后 atomicAdd 到 global grad；
5. 与当前 scalar/candidate 循环版本做 image allclose 和 gradient finite-difference 对照。

---

### Task 5：新增 backward additive path

在：

```text
src/shader/us_focus.cu
```

或现有 backward 文件中新增：

```cpp
kernel_us_focus_additive_bwd(...)
```

第一版至少支持：

```text
grad_intensity
grad_opacity
grad_means
```

更完整版本再支持：

```text
grad_scales
grad_quats
grad_pose
grad_us_focus_params
```

---

### Task 6：新增测试

新增：

```text
tests/test_us_focus_forward.py
tests/test_us_focus_gradcheck.py
```

先测试 forward，再测试有限差分梯度。

---

### Task 7：训练脚本增加稳定模式

新增配置：

```yaml
renderer:
  shader_weight_mode: us_focus
  compositing_mode: additive
  mapper_kind: tile
  mapper_sort_by_v: false

  us_focus_sigma_ele_0: 1.0
  us_focus_sigma_ele_1: 0.0
  us_focus_sigma_ele_min: 0.5
  us_focus_sigma_ele_max: 8.0
  us_focus_det_gamma: 0.25
```

---

## 22. 第一版最小可用目标

第一版不要追求完整论文级反向。

最小闭环目标：

```text
us_focus forward CUDA works
tile mapper coverage works
additive rendering stable
pred range stable
L1 loss decreases
no large block artifacts
```

推荐第一版支持：

```text
grad_intensity
grad_opacity
grad_means
```

暂时冻结：

```text
scales
quats
pose
```

如果这个版本稳定，再逐步打开 scale / quat。

---

## 23. 与现有 plane_cond2d 的关系

不要立即删除 `plane_cond2d`。

建议保留三个路径：

```text
probe3d       # baseline
plane_cond2d # conditional Gaussian ablation
us_focus     # main proposed method
```

论文实验时可以做 ablation：

| Method | Description |
|---|---|
| 3D Mahalanobis / probe3d | 直接 3D 高斯评估 |
| plane_cond2d | 条件高斯切片 |
| us_focus constant | 常数 elevational focus |
| us_focus depth-adaptive | 深度自适应超声 focus |
| us_focus + A-line lateral | 加入 lateral beam profile |

这样创新路线更清楚。

---

## 24. 论文叙事建议

不要写成：

> We follow GaussianPile.

而是写成：

> Inspired by focus-aware slice formation, we design a probe-conditioned A-line-aware Gaussian-to-pixel accumulation model for freehand ultrasound reconstruction.

强调差异：

1. GaussianPile 是 generic slice-based volumetric reconstruction；
2. 本方法针对 freehand ultrasound；
3. 使用 probe-local A-line geometry；
4. 引入 depth-adaptive elevational beam profile；
5. 使用 soft slice response 替代 hard corridor / conditional slicing；
6. 与 relocalization / pose refinement 多任务耦合。

可命名为：

```text
A-line-aware Focus Gaussian Accumulation
```

或：

```text
Probe-conditioned Ultrasound Focus Splatting
```

---

## 25. 推荐最终公式汇总

### 25.1 Probe transform

$$
\mu_c=A_i(\mu^w-t_i)
$$

$$
\Sigma_c=A_i\Sigma^wA_i^T
$$

### 25.2 Focus precision

$$
\Lambda_c=\Sigma_c^{-1}
$$

$$
\Lambda_f(a)
=
\operatorname{diag}
\left(
\frac{\eta_{\mathrm{lat}}}{\sigma_{\mathrm{lat}}(a)^2},
0,
\frac{1}{\sigma_{\mathrm{ele}}(a)^2}
\right)
$$

其中：

$$
\eta_{\mathrm{lat}}\in\{0,1\}
$$

控制是否启用 lateral focus。

$$
\Lambda_e=\Lambda_c+\Lambda_f(a)
$$

$$
\Sigma_e=\Lambda_e^{-1}
$$

$$
\mu_e=\Sigma_e\Lambda_c\mu_c
$$

### 25.3 2D footprint

$$
\mu_{2d}=P\mu_e
$$

$$
\Sigma_{2d}=P\Sigma_eP^T
$$

$$
d_{g,p}=\xi_p-\mu_{2d,g}
$$

$$
q_{g,p}=d_{g,p}^T\Sigma_{2d,g}^{-1}d_{g,p}
$$

### 25.4 Soft slice response

$$
r_{g,i}
=
\exp
\left(
-\frac12
\frac{\mu_{\mathrm{ele},g}^2}
{\Sigma_c[2,2]+\sigma_{\mathrm{ele}}(a_g)^2+\epsilon}
\right)
$$

### 25.5 Contribution

$$
w_{g,p}
=
o_g
r_{g,i}
(\det\Sigma_{2d,g}+\epsilon)^{-\gamma}
\exp
\left(
-\frac12q_{g,p}
\right)
$$

### 25.6 Additive rendering

$$
I_i(p)
=
b
+
\sum_{g\in\mathcal{N}(p)}
\iota_gw_{g,p}
$$

---

## 26. 迁移顺序建议

按以下顺序执行最稳：

1. 新增 `us_focus` mode，不动旧路径；
2. 写 PyTorch reference；
3. 写 CUDA forward；
4. 写 tile mapper footprint；
5. 只跑 forward 可视化；
6. 支持 intensity / opacity backward；
7. 支持 mean backward；
8. 用 L1 loss 训练；
9. 稳定后支持 scale / quat backward；
10. 最后加 SSIM / LPIPS；
11. 再做 depth-adaptive / lateral focus ablation。

---

## 27. 最终建议

这次重构的核心目标不是把 GaussianPile 原样搬过来，而是把它的稳定思想改造成更适合超声的模型：

> 从 conditional slicing 转向 focus-weighted additive accumulation。

你的创新点应该放在：

1. ultrasound-specific A-line geometry；
2. depth-adaptive elevational focus；
3. probe-conditioned Gaussian-to-pixel contribution；
4. soft slice response；
5. freehand relocalization + reconstruction 联合优化；
6. additive acoustic accumulation 替代 alpha occlusion。

这样既能提高训练稳定性，又能在论文方法上和已有 slice-based Gaussian splatting 工作拉开差异。
