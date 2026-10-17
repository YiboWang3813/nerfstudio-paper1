# Ours Renderer 最新代码审查意见与伪影排查建议

本文整理针对用户最新上传的 `ours_renderer.zip` 的源码静态审查意见，重点回答两个问题：

1. Cursor 是否修复了之前指出的数学与实现问题；
2. 当前渲染图像出现大块伪影、LPIPS 爆炸的可能原因与排查路径。

> 说明：本文基于源码静态审查结论，没有在当前环境中实际编译和运行 CUDA。因此，涉及训练现象的部分是根据代码路径、数值稳定性和渲染公式进行的推断。

---

## 1. 总体结论

Cursor 确实已经修改了之前指出的几个关键问题，尤其是：

1. `plane_cond2d` 前向已经加入 elev attenuation；
2. tile mapper 已经开始使用 Schur 条件中心 `mu2`；
3. `delta / mu2` 反向符号和系数问题已经修复；
4. 部分 clamp / branch 的反向也做了一些处理。

但是，当前代码仍然存在几个非常可疑的风险点，足以导致：

- 渲染图像出现大块色斑 / 大块遮挡；
- 输出范围异常；
- alpha 合成失控；
- LPIPS loss 爆炸。

最需要优先排查的是：

1. opacity 是否没有约束，导致 alpha 大面积饱和；
2. 训练配置是否实际仍在跑默认的 `probe3d` / `additive`；
3. LPIPS 输入图像范围是否不对；
4. pixel mapper 的 depth 排序是否仍使用无条件中心；
5. covariance / pose backward 中 off-diagonal 梯度是否仍不严格。

---

## 2. 之前指出的问题是否已经修改

| 问题 | 当前状态 | 说明 |
|---|---:|---|
| `plane_cond2d` 缺少 elev attenuation | 已修改 | 前向中已经加入 `exp(-0.5 * mu_z^2 / Szz)` |
| mapper 和 shader 中心不一致 | tile mapper 已修改，pixel coverage 也已修改 | mapper footprint 中使用了 Schur 条件中心 |
| `delta / mu2` 反向符号和系数错误 | 已修改 | 反向已经改成类似 `-2 * dq * I * delta` 的形式 |
| `q=max(q,0)` 的不严格反向 | 部分修改 | 当前 plane weight 中对负 quad 做了返回处理 |
| inverse clamp 反向 | 部分修改 | clamp 激活时有梯度置零处理 |
| alpha 范围约束 | 前向有 clamp，但反向仍不完全严格 | 这是当前很重要的风险点 |

总体判断：

> Cursor 已经修复了主要的前向数学问题，但当前渲染伪影更可能来自 alpha / opacity 饱和、配置未切换到目标路径、LPIPS 输入范围异常，以及残留 backward 不严格问题。

---

## 3. 当前 `plane_cond2d` 前向已经更接近严格 slice cross-section

之前建议将 alpha 从：

$$
\alpha
=
o
\exp
\left(
-\frac{1}{2}q_{\mathrm{cond}}
\right)
$$

改成：

$$
\alpha
=
o
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

当前代码已经加入了类似逻辑：

```cpp
const float elev_quad = (mu_z * mu_z) / Sc[8];
const float elev_atten = expf(-0.5f * elev_quad);
const float base = base_plane * elev_atten;
float alpha = gauss_opacities[g] * base;
alpha = fminf(fmaxf(alpha, 0.f), 1.f);
```

这个方向是正确的。

其中：

$$
q_{\mathrm{total}}
=
q_{\mathrm{plane}}
+
q_{\mathrm{elev}}
$$

$$
q_{\mathrm{elev}}
=
\frac{\mu_z^2}{S_{zz}}
$$

因此：

$$
\alpha
=
o
\exp
\left(
-\frac{1}{2}q_{\mathrm{total}}
\right)
$$

这比原来的 conditional-only plane response 更接近真正的 3D Gaussian slice cross-section。

---

## 4. 当前最可疑问题一：alpha clamp 的反向没有同步处理

### 4.1 当前前向

当前前向中 alpha 被 clamp：

```cpp
float alpha = gauss_opacities[g] * base;
alpha = fminf(fmaxf(alpha, 0.f), 1.f);
```

即：

$$
\alpha
=
\operatorname{clamp}(o\cdot base, 0, 1)
$$

### 4.2 当前反向风险

但反向中仍然近似按未 clamp 的公式传播：

```cpp
atomicAdd(wrow + PLANE_COND2D_WSP_G_OPACITY, dL_dalpha * base);

const float dL_dbase_total = dL_dalpha * o;
```

如果：

$$
o\cdot base > 1
$$

那么前向：

$$
\alpha = 1
$$

严格来说，在饱和区：

$$
\frac{\partial \alpha}{\partial o}=0
$$

$$
\frac{\partial \alpha}{\partial base}=0
$$

但当前反向仍然继续把梯度传给 opacity、base、mean、scale、quat。

这会导致：

1. opacity 可能被继续错误推大；
2. scale / covariance 也可能被错误调整；
3. 大量 alpha 饱和为 1；
4. 渲染图像出现大块硬遮挡；
5. LPIPS 因局部大块结构异常而爆炸。

这非常符合当前“大块伪影”的现象。

---

### 4.3 建议修复

在 backward wire 中加入 clamp-aware mask：

```cpp
const float raw_alpha = o * base;
const float alpha_grad_mask =
    (raw_alpha > 0.f && raw_alpha < 1.f) ? 1.f : 0.f;

atomicAdd(
    wrow + PLANE_COND2D_WSP_G_OPACITY,
    dL_dalpha * base * alpha_grad_mask
);

const float dL_dbase_total =
    dL_dalpha * o * alpha_grad_mask;
```

也就是说：

$$
m_\alpha
=
\mathbf{1}_{0<o\cdot base<1}
$$

然后：

$$
\bar{o}
\mathrel{+}=
\bar{\alpha}\cdot base\cdot m_\alpha
$$

$$
\bar{base}
=
\bar{\alpha}\cdot o\cdot m_\alpha
$$

### 4.4 更推荐的做法

更好的做法是不要依赖 CUDA 中 hard clamp opacity，而是在 Python 模型侧对 opacity 做参数化：

```python
opacity = torch.sigmoid(opacity_logit)
```

这样天然满足：

$$
0 < o < 1
$$

如果仍然担心 alpha 过大，也可以进一步做：

```python
opacity = opacity_max * torch.sigmoid(opacity_logit)
```

例如：

```python
opacity_max = 0.2
```

训练早期先限制 alpha，稳定后再逐步放宽。

---

## 5. 当前最可疑问题二：实际训练配置可能仍然没有切到 `plane_cond2d + alpha`

### 5.1 默认配置风险

如果 `OursRenderConfig` 默认仍然是：

```python
compositing_mode: str = "additive"
shader_weight_mode: str = "probe3d"
```

那么即使 CUDA 代码已经修改，训练时也可能根本没有使用 `plane_cond2d`。

这会造成两类问题：

### 情况 A：实际跑的是 `probe3d`

`probe3d` 路径可能仍然没有 alpha clamp：

```cpp
float alpha = gauss_opacities[g] * base;
*out_alpha = alpha;
```

如果：

$$
\alpha > 1
$$

则 alpha compositing 中：

$$
T \leftarrow T(1-\alpha)
$$

会导致：

$$
T < 0
$$

后续颜色合成会失控。

### 情况 B：实际跑的是 `additive`

additive 模式前向类似：

$$
C_p = \sum_g \iota_g\alpha_{g,p}
$$

它没有 transmittance 抑制，所有贡献直接加起来。如果高斯数量多、opacity 或 intensity 较大，输出很容易爆炸。

这会直接导致：

- 渲染图像范围异常；
- LPIPS 输入分布异常；
- loss 爆炸。

---

### 5.2 必须确认的配置

训练启动时建议强制打印：

```python
print(config.shader_weight_mode)
print(config.compositing_mode)
print(config.mapper_kind)
print(config.mapper_sort_by_v)
```

你应该看到：

```text
shader_weight_mode = plane_cond2d
compositing_mode = alpha
mapper_kind = tile
mapper_sort_by_v = True
```

如果不是这个组合，需要先修配置。

---

## 6. 当前最可疑问题三：LPIPS 输入范围不对

LPIPS 对输入图像范围非常敏感。常见要求是：

1. 输入在 `[0,1]`，然后内部或外部转成 `[-1,1]`；
2. 或者直接输入 `[-1,1]`；
3. 通常是 3-channel image。

如果 renderer 输出是 raw intensity，例如：

```text
min = -20
max = 50
mean = 8
```

那么直接送入 LPIPS，loss 爆炸是正常的。

### 6.1 建议打印

训练中加入：

```python
with torch.no_grad():
    print(
        "pred:",
        pred.min().item(),
        pred.max().item(),
        pred.mean().item(),
        torch.isfinite(pred).all().item(),
    )
    print(
        "gt:",
        gt.min().item(),
        gt.max().item(),
        gt.mean().item(),
        torch.isfinite(gt).all().item(),
    )
```

### 6.2 建议临时关闭 LPIPS

在 renderer 稳定前，建议先只用：

```python
loss = l1_loss(pred, gt)
```

或者：

```python
loss = mse_loss(pred, gt)
```

等满足以下条件后再启用 LPIPS：

1. `pred` 没有 NaN / Inf；
2. `pred.min()` 和 `pred.max()` 接近合理图像范围；
3. 没有大块硬伪影；
4. L1 / MSE 能稳定下降。

### 6.3 LPIPS 前建议规范化

如果图像是单通道超声图像，可以先扩成 3 通道：

```python
pred_3 = pred.repeat(1, 3, 1, 1)
gt_3 = gt.repeat(1, 3, 1, 1)
```

若 LPIPS 期望 `[-1,1]`：

```python
pred_lpips = pred_3.clamp(0, 1) * 2 - 1
gt_lpips = gt_3.clamp(0, 1) * 2 - 1
```

注意：训练主 loss 不一定要 clamp，但送进 LPIPS 的图像最好范围稳定。

---

## 7. 当前最可疑问题四：pixel mapper depth 排序仍可能使用无条件中心

### 7.1 问题描述

pixel mapper 的 coverage 可能已经使用 `mu2`，但 depth 排序如果仍使用：

```cpp
depths[g] = (mu_y - axial_near_mm) / axial_spacing_mm_per_pix;
```

那么排序深度仍然是无条件中心：

$$
\mu_{\mathrm{ax}}
$$

而不是条件中心：

$$
\mu_{2,\mathrm{ax}}
$$

### 7.2 为什么这会造成问题

在 alpha compositing 中，顺序非常重要：

$$
C \leftarrow C + I\alpha T
$$

$$
T \leftarrow T(1-\alpha)
$$

如果高斯顺序错了，即使每个 alpha 都正确，最终颜色也可能明显错误。

对于 `plane_cond2d`，真正的平面响应中心是：

$$
\mu_{2,\mathrm{ax}}
=
\mu_{\mathrm{ax}}
-
\frac{\mu_z}{S_{zz}}S_{\mathrm{ax},z}
$$

因此排序最好也与这个条件中心一致。

### 7.3 建议修改

如果使用：

```python
mapper_kind = "pixel"
compositing_mode = "alpha"
shader_weight_mode = "plane_cond2d"
```

则建议把 depth 也改成：

```cpp
if (shader_weight_mode == SHADER_WEIGHT_PLANE_COND2D) {
    // compute mu2_ax using plane_cond2d_mapper_footprint_mm
    depths[g] = (mu2_ax - axial_near_mm) / axial_spacing_mm_per_pix;
} else {
    depths[g] = (mu_y - axial_near_mm) / axial_spacing_mm_per_pix;
}
```

如果当前使用的是 tile mapper，这个不是第一主因，但仍建议修正以保证两个 mapper 行为一致。

---

## 8. 当前最可疑问题五：Schur backward 到 full covariance 的 off-diagonal 处理仍可疑

### 8.1 问题描述

`schur_bwd()` 中可能只写了 upper-triangular 风格的梯度，例如：

```cpp
g_Sc[1] += g_s2_01;
g_Sc[2] += g_spz_lat;
g_Sc[5] += g_spz_ax;
```

但后续：

```cpp
plane_cond2d_sc_to_world_grads()
plane_cond2d_sc_to_axis_grads()
sigma_w_to_scale_rot_grads()
```

更像是在使用 full 3x3 matrix 的梯度。

如果 `g_Sc` 的 off-diagonal 语义不一致，会导致：

1. scale gradient 不严格；
2. quaternion / rotation gradient 不严格；
3. pose rotation gradient 不严格；
4. 训练长期后高斯形状可能被推歪；
5. 局部出现异常大 scale 或异常方向，造成大块伪影。

### 8.2 建议修复

如果后续函数期望 full symmetric matrix gradient，建议在 Schur backward 中对称写入：

```cpp
g_Sc[1] += 0.5f * g_s2_01;
g_Sc[3] += 0.5f * g_s2_01;

g_Sc[2] += 0.5f * g_spz_lat;
g_Sc[6] += 0.5f * g_spz_lat;

g_Sc[5] += 0.5f * g_spz_ax;
g_Sc[7] += 0.5f * g_spz_ax;
```

或者另一种方式是统一规定 `g_Sc` 使用 packed symmetric gradient，然后所有后续函数都按 packed 语义实现。

关键是必须统一：

- `g_Sc[1]` 到底代表单个参数 \(S_{01}\) 的梯度；
- 还是 full matrix entry \(S_{01}\) 的梯度；
- `S_{10}` 是否也需要显式存在；
- 后续求 \(\bar{\Sigma}^w\) 时 off-diagonal 是否会自动乘 2。

---

## 9. 当前测试覆盖不足

### 9.1 默认测试可能没有覆盖 `plane_cond2d`

如果测试默认没有显式设置：

```python
shader_weight_mode = "plane_cond2d"
compositing_mode = "alpha"
```

那么测试通过也不能说明当前新路径正确。

建议增加专门测试：

```python
OursRenderConfig(
    mapper_kind="tile",
    shader_weight_mode="plane_cond2d",
    compositing_mode="alpha",
    mapper_sort_by_v=True,
)
```

以及：

```python
OursRenderConfig(
    mapper_kind="pixel",
    shader_weight_mode="plane_cond2d",
    compositing_mode="alpha",
    mapper_sort_by_v=True,
)
```

### 9.2 CPU alpha reference 可能有乘两次 alpha 的问题

如果 CPU reference 中类似：

```python
contrib_I = int_sub.unsqueeze(1) * alpha
...
I = contrib_I[k]
C_alpha = C_alpha + I * a * T_alpha
```

那么实际是：

$$
C \mathrel{+}= I\alpha^2T
$$

而正确 alpha compositing 应该是：

$$
C \mathrel{+}= I\alpha T
$$

建议改成：

```python
I = int_sub[k]
a = alpha[k]
C_alpha = C_alpha + I * a * T_alpha
T_alpha = T_alpha * (1.0 - a)
```

否则 CPU/GPU 对照测试的意义会被削弱。

---

## 10. 大块伪影 + LPIPS 爆炸的原因排序

按照可能性从高到低排序：

### 10.1 opacity 没有限制，导致 alpha 大面积饱和

现象：

- 图像中出现大块近似常数区域；
- 某些高斯像硬遮挡一样盖住大片图像；
- pred max / min 异常；
- LPIPS 突然爆炸。

排查：

```python
print("opacity:", opacity.min().item(), opacity.max().item())
```

如果看到：

```text
opacity max > 1
```

或者 opacity 快速增大，基本可以确定这是主因之一。

修复：

```python
opacity = torch.sigmoid(opacity_logit)
```

或早期限制：

```python
opacity = 0.2 * torch.sigmoid(opacity_logit)
```

---

### 10.2 实际跑的是 `probe3d` 或 `additive`

现象：

- 你以为在跑 `plane_cond2d + alpha`；
- 但实际输出仍然像直接累加；
- 图像亮度很容易爆；
- LPIPS 明显不稳定。

排查：

```python
print(config.shader_weight_mode)
print(config.compositing_mode)
```

修复：

```python
config.shader_weight_mode = "plane_cond2d"
config.compositing_mode = "alpha"
config.mapper_sort_by_v = True
```

---

### 10.3 LPIPS 输入范围不对

现象：

- L1/MSE 可能还可以；
- 一加 LPIPS 就炸；
- pred range 明显不在 `[0,1]` 或 `[-1,1]`。

排查：

```python
print(pred.min(), pred.max(), pred.mean())
```

修复：

```python
pred_lpips = pred.clamp(0, 1)
gt_lpips = gt.clamp(0, 1)

pred_lpips = pred_lpips.repeat(1, 3, 1, 1) * 2 - 1
gt_lpips = gt_lpips.repeat(1, 3, 1, 1) * 2 - 1
```

---

### 10.4 scale 太大或训练中被错误梯度推大

现象：

- 一个高斯覆盖大片图像；
- 局部出现大面积模糊亮斑；
- tile touched count 异常大；
- alpha nonzero ratio 异常高。

排查：

```python
print("scale:", scales.min().item(), scales.max().item(), scales.mean().item())
```

修复：

```python
scale = scale_min + (scale_max - scale_min) * torch.sigmoid(scale_logit)
```

例如：

```python
scale_min = 0.05
scale_max = 3.0
```

具体数值应根据超声图像 spacing 和场景尺寸调整。

---

### 10.5 pixel mapper + alpha 排序错误

如果使用 pixel mapper，depth 排序错误会明显影响 alpha 合成结果。

排查：

```python
print(config.mapper_kind)
```

如果是：

```text
pixel
```

则优先修 depth 使用 `mu2_ax`。

---

## 11. 建议的立即排查步骤

### Step 1：确认 CUDA 真的重新编译

```bash
cd ours_renderer
rm -rf build
rm -f ours_renderer/_C*.so
./rebuild.sh
```

如果 `.so` 没更新，Python 可能仍然加载旧算子。

---

### Step 2：打印训练配置

必须确认：

```text
shader_weight_mode = plane_cond2d
compositing_mode = alpha
mapper_sort_by_v = True
```

建议训练开始时直接打印完整 config。

---

### Step 3：打印 renderer 输出范围

```python
with torch.no_grad():
    pred = renderer(...)
    print(
        "pred:",
        pred.min().item(),
        pred.max().item(),
        pred.mean().item(),
        torch.isfinite(pred).all().item(),
    )
```

如果 `pred.max()` 很大，LPIPS 爆炸不是 LPIPS 的根本问题，而是 renderer 输出已经炸了。

---

### Step 4：打印核心高斯参数范围

```python
print("opacity:", opacity.min().item(), opacity.max().item(), opacity.mean().item())
print("intensity:", intensity.min().item(), intensity.max().item(), intensity.mean().item())
print("scale:", scales.min().item(), scales.max().item(), scales.mean().item())
```

重点看：

1. opacity 是否大于 1；
2. opacity 是否小于 0；
3. intensity 是否远超图像范围；
4. scale 是否异常大；
5. scale 是否塌到 `1e-6` 附近。

---

### Step 5：暂时关闭 LPIPS

先只用：

```python
loss = torch.nn.functional.l1_loss(pred, gt)
```

或者：

```python
loss = torch.nn.functional.mse_loss(pred, gt)
```

等渲染图像稳定后，再加 LPIPS。

---

## 12. 建议的优先修复顺序

### 优先级 1：参数范围控制

1. opacity 使用 sigmoid 参数化；
2. intensity 限制到合理范围；
3. scale 使用 bounded parameterization。

推荐：

```python
opacity = opacity_max * torch.sigmoid(opacity_logit)
intensity = torch.sigmoid(intensity_logit)
scale = scale_min + (scale_max - scale_min) * torch.sigmoid(scale_logit)
```

训练初期可取：

```python
opacity_max = 0.2
```

---

### 优先级 2：确认配置

强制使用：

```python
shader_weight_mode = "plane_cond2d"
compositing_mode = "alpha"
mapper_sort_by_v = True
```

如果暂时不确定 pixel mapper depth 是否正确，先用：

```python
mapper_kind = "tile"
```

---

### 优先级 3：修 alpha clamp backward

加入：

```cpp
const float raw_alpha = o * base;
const float alpha_grad_mask =
    (raw_alpha > 0.f && raw_alpha < 1.f) ? 1.f : 0.f;
```

并让 opacity / base 相关梯度乘上这个 mask。

---

### 优先级 4：修 pixel mapper depth

如果使用 `pixel` mapper，则 `plane_cond2d` 路径的 depth 应该使用：

$$
\mu_{2,\mathrm{ax}}
$$

而不是：

$$
\mu_{\mathrm{ax}}
$$

---

### 优先级 5：修 off-diagonal gradient 语义

统一 `g_Sc` 的 symmetric matrix gradient 约定。

如果后续按 full matrix 使用，则 Schur backward 应对称写入。

---

### 优先级 6：修测试

1. CPU alpha reference 不要乘两次 alpha；
2. 增加 `plane_cond2d + alpha + tile` 测试；
3. 增加 `plane_cond2d + alpha + pixel` 测试；
4. 增加 finite-difference gradient check。

---

## 13. 推荐 debug 打印项

建议在训练前 10 iteration 每轮打印：

```python
def debug_stats(name, x):
    print(
        name,
        "min=", x.min().item(),
        "max=", x.max().item(),
        "mean=", x.mean().item(),
        "finite=", torch.isfinite(x).all().item(),
    )

debug_stats("pred", pred)
debug_stats("gt", gt)
debug_stats("opacity", opacity)
debug_stats("intensity", intensity)
debug_stats("scales", scales)
debug_stats("means", means)
```

如果能从 CUDA 返回中间统计，也建议打印：

```text
alpha min / max / mean
alpha >= 1 ratio
alpha <= 0 ratio
nonzero alpha ratio
tiles touched mean / max
effective layers per pixel mean / max
T final min / mean
```

其中最关键的是：

```text
alpha >= 1 ratio
```

如果这个比例不小，说明大块伪影很可能来自 alpha saturation。

---

## 14. 推荐临时稳定训练配置

为了先确认 renderer 稳定，可以采用以下临时策略：

```python
shader_weight_mode = "plane_cond2d"
compositing_mode = "alpha"
mapper_kind = "tile"
mapper_sort_by_v = True
```

参数约束：

```python
opacity = 0.2 * torch.sigmoid(opacity_logit)
intensity = torch.sigmoid(intensity_logit)
scale = scale_min + (scale_max - scale_min) * torch.sigmoid(scale_logit)
```

loss：

```python
loss = l1_loss(pred, gt)
```

暂时不要使用 LPIPS。

等图像稳定后，再逐步改为：

```python
opacity = torch.sigmoid(opacity_logit)
loss = l1_loss + lambda_lpips * lpips_loss
```

---

## 15. 最终判断

当前代码状态可以概括为：

> Cursor 已经把 `plane_cond2d` 的核心数学前向修得比较像样了，但训练稳定性仍然没有完全解决。现在的大块伪影和 LPIPS 爆炸，更可能来自 alpha / opacity 饱和、实际配置未切到 `plane_cond2d + alpha`、LPIPS 输入范围错误、pixel mapper depth 排序残留问题，以及 covariance backward 的 off-diagonal 梯度语义不统一。

最优先建议：

1. 先确认训练实际配置；
2. 先约束 opacity / intensity / scale；
3. 暂时关闭 LPIPS；
4. 打印 pred / alpha / opacity / scale 范围；
5. 修 alpha clamp backward；
6. 再修 pixel depth 和 off-diagonal covariance gradient。

只要先把输出范围和 alpha 饱和控制住，大块伪影和 LPIPS 爆炸大概率会明显缓解。
