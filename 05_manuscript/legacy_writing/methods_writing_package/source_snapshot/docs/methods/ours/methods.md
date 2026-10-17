# Ours Method Notes for Writing Agents

This document describes the current `ours` method with implementation-aligned
mathematical notation. It is intended for an agent writing the paper Methods
section. The formulas below are grounded in the current production path:

```text
configs_hydra/model/ours_us_focus.yaml
configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml
methods/ours/
ops/ours_renderer/
scene/container/single_sweep_container.py
```

The deployed method is:

```text
Ours = easy-pose canonical sweep prior
     + local-global ultrasound renderer
     + responsibility-aware density control
```

## 1. Symbol Table

### 1.1 Dataset, Images, and Poses

| Symbol | Code / config | Meaning |
| --- | --- | --- |
| \(N\) | sweep length | Number of frames in a single offline sweep. |
| \(i\) | `frame_id`, datum index | Frame index. |
| \(\mathcal T\) | train split | Training frame indices. |
| \(\mathcal E\) | test split | Held-out evaluation frame indices. |
| \(\mathbf I_i\) | `slice_meta.image` | Ground-truth B-mode image of frame \(i\), normalized to image tensor range. |
| \(\hat{\mathbf I}_i\) | `pred_image` | Rendered image from the Gaussian scene. |
| \(H,W\) | `dataset_meta.H`, `dataset_meta.W` | Image height and width. |
| \(p=(x,y)\) | CUDA pixel index | Pixel coordinate; \(x\) is lateral column and \(y\) is axial row. |
| \(\Delta_{\mathrm{lat}}\) | `delta_lateral_mm` | Lateral pixel spacing in mm. |
| \(\Delta_{\mathrm{ax}}\) | `delta_axial_mm` | Axial pixel spacing in mm. |
| \(z_{\mathrm{near}}\) | `near_mm` | Axial coordinate at row 0. |
| \(w_{\mathrm{off}}\) | `SG_LAT_COL_OFFSET` | Lateral column offset, usually \((W-1)/2\). |
| \(\boldsymbol{\xi}_p=(\xi_{\mathrm{lat}},\xi_{\mathrm{ax}},0)\) | `sg_linear_pixel_to_plane_mm` | Probe-local mm coordinate of pixel \(p\) on the slice plane. |
| \(\mathbf T_i\in SE(3)\) | `slice_meta.pose` | Probe-to-world pose for frame \(i\). |
| \(\tilde{\mathbf T}_i\) | `pose_mode=canonical_sweep` | Easy-pose canonical probe-to-world pose. |
| \(\mathbf a_{\mathrm{lat}},\mathbf a_{\mathrm{ax}},\mathbf a_{\mathrm{ele}}\) | pose columns 0,1,2 | Probe lateral, axial, and elevational axes in world coordinates. |
| \(\mathbf t_i\) | pose translation column | Probe origin in world coordinates. |

Pixel-to-plane mapping follows `ops/ours_renderer/include/shader/slice_grid_geometry.cuh`:

\[
\xi_{\mathrm{lat}}(x) = (x-w_{\mathrm{off}})\Delta_{\mathrm{lat}},
\qquad
\xi_{\mathrm{ax}}(y) = y\Delta_{\mathrm{ax}} + z_{\mathrm{near}}.
\]

The CUDA pose is read as column axes:

\[
\mathbf T_i =
\begin{bmatrix}
\mathbf r_{\mathrm{lat}} & \mathbf r_{\mathrm{ax}} & \mathbf r_{\mathrm{ele}} & \mathbf t_i\\
0&0&0&1
\end{bmatrix},
\qquad
\mathbf a_k = \frac{\mathbf r_k}{\|\mathbf r_k\|_2+\epsilon}.
\]

### 1.2 Gaussian Parameters

| Symbol | Code | Meaning |
| --- | --- | --- |
| \(G\) | `container.num_points` | Number of active Gaussians. |
| \(g\) | Gaussian index | Gaussian primitive index. |
| \(\boldsymbol{\mu}_g\in\mathbb R^3\) | `container.means` | Gaussian center in world mm. |
| \(\mathbf s_g=(s_{g1},s_{g2},s_{g3})\) | `container.scales` | Activated positive local scales in mm. |
| \(\mathbf q_g=(w,x,y,z)\) | `container.quats` | Unit quaternion in repository layout `(w,x,y,z)`. |
| \(\mathbf R_g\) | `ours_quat_to_rot` | Rotation matrix from \(\mathbf q_g\). |
| \(\alpha_g\) | `container.echo_alphas`, `opacities` | Echo alpha / local response amplitude. |
| \(\iota_g\) | `container.intensities` | Local intensity multiplier. In current `ours`, \(\iota_g=1\). |
| \(\tau_g\) | `container.attenuations` | Trainable global attenuation attribute, clamped to \([0,1]\) in CUDA formulas. |
| \(\boldsymbol{\Sigma}^w_g\) | CUDA covariance | World covariance. |
| \(\boldsymbol{\mu}^c_{ig}\) | probe coordinates | Gaussian center in probe coordinates of frame \(i\). |
| \(\boldsymbol{\Sigma}^c_{ig}\) | probe covariance | Gaussian covariance in probe coordinates. |

The active parameters are produced by `methods/ours/gaussian_container.py`.
For the production `ours_us_focus` path:

\[
\mathbf s_g =
s_{\min} + (s_{\max}-s_{\min})\sigma(\bar{\mathbf s}_g),
\quad
\alpha_g = \alpha_{\max}\sigma(\bar\alpha_g),
\quad
\tau_g = \tau_{\max}\sigma(\bar\tau_g),
\quad
\mathbf q_g = \frac{\bar{\mathbf q}_g}{\|\bar{\mathbf q}_g\|_2+\epsilon}.
\]

The production config uses:

```text
scale_activation = bounded_sigmoid
scale_min_mm = 0.05
scale_max_mm = 3.0
echo_alpha_max = 0.2
attenuation_max = 1.0
intensities = ones_like(echo_alphas)
```

The world covariance is:

\[
\boldsymbol{\Sigma}^w_g
=
\mathbf R_g
\operatorname{diag}(s_{g1}^2,s_{g2}^2,s_{g3}^2)
\mathbf R_g^\top.
\]

For frame \(i\), define the probe-axis matrix:

\[
\mathbf A_i =
\begin{bmatrix}
\mathbf a_{\mathrm{lat},i}^\top\\
\mathbf a_{\mathrm{ax},i}^\top\\
\mathbf a_{\mathrm{ele},i}^\top
\end{bmatrix}.
\]

Then:

\[
\boldsymbol{\mu}^c_{ig}
=
\mathbf A_i(\boldsymbol{\mu}_g-\mathbf t_i)
=
(\mu_{\mathrm{lat}},\mu_{\mathrm{ax}},\mu_{\mathrm{ele}})^\top,
\]

\[
\boldsymbol{\Sigma}^c_{ig}
=
\mathbf A_i \boldsymbol{\Sigma}^w_g \mathbf A_i^\top
+ \epsilon_{\Sigma}\mathbf I_3.
\]

The covariance jitter \(\epsilon_{\Sigma}=10^{-6}\) follows the CUDA path in
`plane_cond2d_device.cuh` / `us_focus_device.cuh`.

### 1.3 Renderer Hyperparameters

| Symbol | Code / config | Meaning |
| --- | --- | --- |
| \(k_\sigma\) | `slice_render.k_sigma` | Footprint radius multiplier. |
| \(T_s\) | `tile_size_px` | Tile size in pixels. |
| \(h_{\mathrm{elev}}\) | `elev_corridor_half_mm` | Elevational corridor half-width. |
| \(h_{\mathrm{slab}}\) | `slab_half_thickness_mm` | Extra slab half-thickness. |
| \(\sigma_{\mathrm{ele},0},\sigma_{\mathrm{ele},1}\) | `us_focus_sigma_ele_0/1` | Depth-dependent elevational focus width parameters. |
| \(\sigma_{\mathrm{lat},0},\sigma_{\mathrm{lat},1}\) | `us_focus_sigma_lat_0/1` | Depth-dependent lateral focus parameters. |
| \(\gamma_{\det}\) | `us_focus_det_gamma` | Determinant normalization exponent. |
| \(\epsilon_{\det}\) | `us_focus_det_eps` | Determinant stability epsilon. |
| \(w_{\min}\) | `us_focus_weight_min` | Minimum local response weight. |
| \(\beta\) | `ray_attenuation_beta` | Blend strength between no modulation and full transmission. |
| \(\lambda_{\mathrm{ray}}\) | `ray_attenuation_strength` | Ray response strength. |
| \(T_{\min}\) | `ray_attenuation_min_transmission` | Transmission lower clamp. |
| \(m_{\mathrm{ax}}\) | `ray_attenuation_axial_margin_mm` | Axial margin used for candidate depth cutoff. |

Production local-global renderer settings:

```text
shader_weight_mode = us_focus
compositing_mode = additive
mapper_kind = tile
ray_attenuation_enabled = true
ray_attenuation_source = attenuation
ray_attenuation_mapper_kind = ray_csr
ray_attenuation_formula = slice_depth
ray_attenuation_beta = 0.60
ray_attenuation_strength = 0.30
ray_attenuation_min_transmission = 0.55
ray_attenuation_candidate_k_sigma = 2.0
```

### 1.4 Responsibility Density Control Symbols

| Symbol | Code | Meaning |
| --- | --- | --- |
| \(e_p\) | `residual_abs[p]` | Absolute image residual \(|\hat I_p-I_p|\). |
| \(c_{gp}\) | CUDA contribution | Contribution magnitude of Gaussian \(g\) to pixel \(p\). |
| \(C_g\) | `resp_contribution` | Total contribution \(\sum_p c_{gp}\). |
| \(R_g\) | `resp_residual_contribution` | Residual contribution \(\sum_p c_{gp}e_p\). |
| \(\bar r_g\) | `residual_mean` | Contribution-normalized residual. |
| \(s_g^{\mathrm{dc}}\) | `score` | Responsibility densification score. |
| \(Q_q(\cdot)\) | `torch.quantile` | Quantile threshold. |
| \(\mathcal D\) | densify mask | Selected clone/split set. |
| \(\mathcal P\) | prune mask | Selected prune set. |

### 1.5 Easy-Pose Symbols

| Symbol | Code / config | Meaning |
| --- | --- | --- |
| \(d_0\) | `canonical_sweep_delta_mm` or `dataset_meta.delta_elevational_mm` | Base adjacent-frame spacing. |
| \(\rho\) | `canonical_sweep_delta_scale` | Spacing scale. |
| \(d=\rho d_0\) | `_canonical_sweep_delta_mm()` | Effective canonical spacing. |
| \(i_0\) | `canonical_sweep_ref_frame` | Reference frame index. |
| \(\mathbf u\) | `canonical_sweep_axis` | Canonical sweep axis, one of signed x/y/z. |
| \(o_i\) | `_make_canonical_sweep_offsets` | Scalar canonical offset for frame \(i\). |

## 2. Task Definition

Given one offline ultrasound sweep, the training data are:

\[
\mathcal D_{\mathrm{train}}
=
\{(\mathbf I_i,\mathbf T_i)\mid i\in\mathcal T\}.
\]

The goal is to optimize a Gaussian scene:

\[
\mathcal G
=
\{(\boldsymbol{\mu}_g,\mathbf s_g,\mathbf q_g,\alpha_g,\tau_g)\}_{g=1}^G
\]

such that the renderer \(\mathcal R_\Theta\) reproduces the observed training
slices and generalizes to held-out frames from the same sweep:

\[
\hat{\mathbf I}_i
=
\mathcal R_\Theta(\mathcal G,\mathbf T_i).
\]

The standard posed objective is:

\[
\min_{\mathcal G}
\sum_{i\in\mathcal T}
\mathcal L_{\mathrm{img}}
\left(
\mathcal R_\Theta(\mathcal G,\mathbf T_i),
\mathbf I_i
\right)
+
\lambda_s\mathcal L_{\mathrm{scale}}(\mathcal G).
\]

The production loss is configured in
`configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml`:

\[
\mathcal L
=
0.65\,\mathcal L_1(\hat{\mathbf I},\mathbf I)
+
0.35\,\mathcal L_{\mathrm{SSIM}}(\hat{\mathbf I},\mathbf I)
+
50\,\mathcal L_{\mathrm{scale}}.
\]

Here:

\[
\mathcal L_1
=
\frac{1}{HW}\sum_p |\hat I_p-I_p|,
\qquad
\mathcal L_{\mathrm{SSIM}}
=
1-\operatorname{SSIM}(\hat{\mathbf I},\mathbf I),
\]

and the scale-wall regularizer follows `utils/loss/gauss_loss.py`:

\[
\mathcal L_{\mathrm{scale}}
=
\frac{1}{3G}\sum_{g,k}
\left[
\max(s_{\min}^{\mathrm{wall}}-s_{gk},0)^2
+
\max(s_{gk}-s_{\max}^{\mathrm{wall}},0)^2
\right].
\]

For easy-pose, \(\mathbf T_i\) is replaced by \(\tilde{\mathbf T}_i\), defined
in Section 5.

## 3. Innovation I: Local-Global Ultrasound Renderer

The renderer is implemented by `ops/ours_renderer`. The Python entry point is
`methods/ours/render.py`, which calls:

```text
OursRenderer.forward(
    means, scales, quats, opacities, intensities,
    pose_probe_to_world, gauss_attenuations
)
```

In current `ours`, `opacities = echo_alphas`,
`intensities = ones_like(echo_alphas)`, and
`gauss_attenuations = attenuations` when
`ray_attenuation_source=attenuation`.

### 3.1 Local Coordinate Transform

For frame \(i\), Gaussian \(g\), and pixel \(p\), transform the Gaussian center
and covariance into probe coordinates:

\[
\boldsymbol{\mu}^c_{ig}
=
(\mu_{\mathrm{lat}},\mu_{\mathrm{ax}},\mu_{\mathrm{ele}})^\top,
\qquad
\boldsymbol{\Sigma}^c_{ig}
=
\begin{bmatrix}
S_{00}&S_{01}&S_{02}\\
S_{10}&S_{11}&S_{12}\\
S_{20}&S_{21}&S_{22}
\end{bmatrix}.
\]

The pixel plane coordinate is:

\[
\boldsymbol{\xi}_p
=
(\xi_{\mathrm{lat}},\xi_{\mathrm{ax}},0)^\top.
\]

The tile mapper first builds candidate lists by projecting Gaussians to the
probe plane. A Gaussian is rejected if it is outside the elevational corridor:

\[
|\mu_{\mathrm{ele}}|
>
h_{\mathrm{elev}} + h_{\mathrm{slab}} + k_\sigma \sigma_{\mathrm{ele},g}.
\]

Candidate construction is discrete and is not differentiated.

### 3.2 Ultrasound Focus Local Response

The production local branch uses `shader_weight_mode=us_focus`, implemented in
`ops/ours_renderer/include/shader/us_focus_device.cuh`.

Let:

\[
\mathbf \Lambda_c
=
(\boldsymbol{\Sigma}^c_{ig})^{-1}.
\]

Define depth:

\[
d_{\mathrm{ax}}
=
\max(\mu_{\mathrm{ax}}-z_{\mathrm{near}},0).
\]

The focus widths are:

\[
\sigma_{\mathrm{ele}}(d_{\mathrm{ax}})
=
\operatorname{clamp}
(\sigma_{\mathrm{ele},0}+\sigma_{\mathrm{ele},1}d_{\mathrm{ax}},
\sigma_{\mathrm{ele},\min},
\sigma_{\mathrm{ele},\max}),
\]

\[
\sigma_{\mathrm{lat}}(d_{\mathrm{ax}})
=
\operatorname{clamp}
(\sigma_{\mathrm{lat},0}+\sigma_{\mathrm{lat},1}d_{\mathrm{ax}},
\sigma_{\mathrm{lat},\min},
\sigma_{\mathrm{lat},\max}).
\]

The production setting has `us_focus_use_lateral_focus=false`, so the focus
precision update is elevational:

\[
\mathbf \Lambda_e
=
\mathbf \Lambda_c
+
\operatorname{diag}
\left(
0,\ 0,\ \sigma_{\mathrm{ele}}^{-2}
\right).
\]

The code supports an optional lateral term:

\[
\mathbf \Lambda_e
=
\mathbf \Lambda_c
+
\operatorname{diag}
\left(
\chi_{\mathrm{lat}}\sigma_{\mathrm{lat}}^{-2},\ 0,\ \sigma_{\mathrm{ele}}^{-2}
\right),
\]

where \(\chi_{\mathrm{lat}}\in\{0,1\}\). For the final method,
\(\chi_{\mathrm{lat}}=0\).

Invert the effective precision:

\[
\boldsymbol{\Sigma}^e_{ig}
=
\mathbf \Lambda_e^{-1}.
\]

The effective mean follows the product-of-Gaussians precision update:

\[
\boldsymbol{\mu}^e_{ig}
=
\boldsymbol{\Sigma}^e_{ig}
\mathbf \Lambda_c
\boldsymbol{\mu}^c_{ig}.
\]

The 2D local footprint on the image plane is:

\[
\boldsymbol{\mu}^{2D}_{ig}
=
\begin{bmatrix}
\mu^e_{\mathrm{lat}}\\
\mu^e_{\mathrm{ax}}
\end{bmatrix},
\]

\[
\boldsymbol{\Sigma}^{2D}_{ig}
=
\begin{bmatrix}
\Sigma^e_{00}+b_{\mathrm{lat}}^2 &
\frac{1}{2}(\Sigma^e_{01}+\Sigma^e_{10})\\
\frac{1}{2}(\Sigma^e_{01}+\Sigma^e_{10}) &
\Sigma^e_{11}+b_{\mathrm{ax}}^2
\end{bmatrix},
\]

where \(b_{\mathrm{lat}}\) and \(b_{\mathrm{ax}}\) are the optional inflation
terms `us_focus_inflate_lat_mm` and `us_focus_inflate_ax_mm`.

For a pixel \(p\), define:

\[
\mathbf d_{igp}
=
\begin{bmatrix}
\xi_{\mathrm{lat}}-\mu^{2D}_{\mathrm{lat}}\\
\xi_{\mathrm{ax}}-\mu^{2D}_{\mathrm{ax}}
\end{bmatrix},
\qquad
q_{igp}
=
\mathbf d_{igp}^\top
(\boldsymbol{\Sigma}^{2D}_{ig})^{-1}
\mathbf d_{igp}.
\]

The local 2D footprint response is:

\[
\phi_{igp}
=
\exp\left(-\frac{1}{2}q_{igp}\right).
\]

The elevational slice response in code uses the original probe elevational
center \(\mu_{\mathrm{ele}}\) and covariance entry \(S_{22}\):

\[
\eta_{ig}
=
\exp\left(
-\frac{1}{2}
\frac{\mu_{\mathrm{ele}}^2}
{S_{22}+\sigma_{\mathrm{ele}}^2+\epsilon_{\det}}
\right).
\]

The determinant normalization is:

\[
\nu_{ig}
=
\left(\det(\boldsymbol{\Sigma}^{2D}_{ig})+\epsilon_{\det}\right)^{-\gamma_{\det}}.
\]

The local response weight is:

\[
w^{\mathrm{loc}}_{igp}
=
\alpha_g\,
\eta_{ig}\,
\nu_{ig}\,
\phi_{igp}.
\]

The local branch output is additive in the production path:

\[
L_{ip}
=
\sum_{g\in\mathcal N^{\mathrm{tile}}_{ip}}
\iota_g w^{\mathrm{loc}}_{igp}.
\]

Since \(\iota_g=1\) in the current `OursGaussContainer`, this is effectively a
sum of echo-alpha-weighted local ultrasound focus responses.

### 3.3 Ray-CSR Global Propagation Modulation

The global branch is enabled by:

```text
ray_attenuation_enabled = true
ray_attenuation_mapper_kind = ray_csr
ray_attenuation_formula = slice_depth
```

For each Gaussian, the ray-CSR mapper computes:

\[
v_g
=
\frac{\mu_{\mathrm{ax}}-z_{\mathrm{near}}}{\Delta_{\mathrm{ax}}},
\]

and assigns the Gaussian to lateral ray bins whose tile-width interval overlaps
its lateral support. Candidates are sorted by ray bin and depth key. For a pixel
\(p=(x,y)\), the ray bin is:

\[
b(p)
=
\left\lfloor \frac{x}{T_s}\right\rfloor.
\]

Only candidates in the same ray bin with depth not after the pixel are used:

\[
\mathcal N^{\mathrm{ray}}_{ip}
=
\left\{
g \mid g\in \mathrm{CSR}(b(p)),\ 
v_g \le y + \frac{m_{\mathrm{ax}}}{\Delta_{\mathrm{ax}}}
\right\}.
\]

For the final `slice_depth` formula, define the isotropic average scale:

\[
\bar\sigma_g
=
\sqrt{\frac{s_{g1}^2+s_{g2}^2+s_{g3}^2}{3}}.
\]

The global response of Gaussian \(g\) for pixel \(p\) is:

\[
\rho_{igp}
=
\bar\sigma_g
\exp\left(
-\frac{1}{2}
\frac{
(\mu_{\mathrm{lat}}-\xi_{\mathrm{lat}})^2
+\mu_{\mathrm{ele}}^2
}
{\bar\sigma_g^2}
\right).
\]

The per-Gaussian attenuation factor is:

\[
\psi_{igp}
=
\lambda_{\mathrm{ray}}\rho_{igp},
\]

\[
T_{igp}
=
\tau_g + (1-\tau_g)\exp(-\psi_{igp}),
\qquad
\tau_g \leftarrow \operatorname{clamp}(\tau_g,0,1).
\]

The pixel transmission is:

\[
T_{ip}
=
\operatorname{clamp}
\left(
\prod_{g\in\mathcal N^{\mathrm{ray}}_{ip}} T_{igp},
T_{\min},1
\right).
\]

The final modulation map is:

\[
M_{ip}
=
1-\beta+\beta T_{ip}.
\]

The rendered pixel is:

\[
\hat I_{ip}
=
M_{ip} L_{ip}.
\]

This is exactly the local-global split used by `autograd.py`: local gradients
are multiplied by `modulation_hw`, and the ray-CSR backward path additionally
returns gradients to `attenuations`, `means`, `scales`, and, for the integral
variant, `quats`. For the final `slice_depth` variant, the ray response depends
on \(\tau_g,\boldsymbol{\mu}_g,\mathbf s_g\); quaternion supervision still
comes through the local branch.

### 3.4 Difference From Related Renderer Forms

A writing agent should preserve these distinctions:

1. Ours is not a pure local projected splatter: it adds \(M_{ip}\).
2. Ours is not a pure ray renderer: it keeps \(L_{ip}\) as the primary
   coordinate-aware local echo response.
3. Ours is not UltraGray with renamed variables: local and global branches have
   separate candidate structures and separate physical responsibilities.
4. In current code, `opacities` are echo alphas, `intensities` are fixed to one,
   and `attenuations` are only used by the global branch.

## 4. Innovation II: Responsibility-Aware Density Control

Density control is implemented in:

```text
methods/ours/dc_diagnostics.py
methods/ours/density_control.py
ops/ours_renderer/ours_renderer/diagnostics.py
ops/ours_renderer/src/shader/forward.cu
```

It runs only at configured density-control iterations:

\[
t_{\mathrm{dc}}\in
\{t\mid t_{\mathrm{start}}\le t\le t_{\mathrm{end}},
\ t\bmod f_{\mathrm{dc}}=0\}.
\]

Production timing:

\[
t_{\mathrm{start}}=1000,\qquad
t_{\mathrm{end}}=25000,\qquad
f_{\mathrm{dc}}=2000.
\]

### 4.1 Pixel Contribution Statistics

At a DC iteration, after rendering frame \(i\), define residual:

\[
e_{ip}
=
|\hat I_{ip}-I_{ip}|.
\]

The diagnostic pass replays the local tile candidate path. For additive
composition, the contribution is:

\[
c_{igp}
=
|\iota_g w^{\mathrm{loc}}_{igp}|.
\]

The CUDA path also supports alpha composition, where:

\[
c_{igp}
=
|\iota_g w^{\mathrm{loc}}_{igp}T^{\mathrm{front}}_{igp}|,
\]

but the production `ours_us_focus` path is additive.

The controller accumulates:

\[
C_g
=
\sum_{i,p} c_{igp},
\]

\[
R_g
=
\sum_{i,p} c_{igp}e_{ip},
\]

\[
S_g
=
\sum_{i,p}\mathbf 1[c_{igp}>0].
\]

The code also records `max_residual_hits`, but the validated final formula does
not use a max-hit multiplier.

### 4.2 Responsibility Score

The residual-weighted responsibility is:

\[
\bar r_g
=
\frac{R_g}{C_g+\epsilon}.
\]

The final densification score is:

\[
s_g^{\mathrm{dc}}
=
\bar r_g \log(1+C_g).
\]

This formula corresponds to `_responsibility_scores()` in
`methods/ours/density_control.py`.

Let:

\[
\mathcal S_+
=
\{g\mid s_g^{\mathrm{dc}}>0\}.
\]

The quantile score threshold is:

\[
\theta_s
=
\max
\left(
Q_{q_s}(\{s_g^{\mathrm{dc}}\mid g\in\mathcal S_+\}),
s_{\min}
\right),
\]

where the final config uses:

\[
q_s=0.98,\qquad s_{\min}=10^{-6}.
\]

A maximum fraction budget is also enforced:

\[
K
=
\max(1,\lfloor f_{\max}G\rfloor),
\qquad
f_{\max}=0.05.
\]

If the quantile threshold selects more than \(K\) positive Gaussians, the code
raises the threshold to the \(K\)-th largest score. The selected responsibility
set is:

\[
\mathcal D
=
\operatorname{TopK}_{K}
\left(
\{g\mid s_g^{\mathrm{dc}}\ge \theta_s\}
\right).
\]

### 4.3 Clone and Split

Let:

\[
s^{\max}_g
=
\max_k s_{gk}.
\]

The clone and split masks are:

\[
\mathcal D_{\mathrm{split}}
=
\{g\in\mathcal D\mid s^{\max}_g>\theta_{\mathrm{split}}\},
\]

\[
\mathcal D_{\mathrm{clone}}
=
\{g\in\mathcal D\mid
s^{\max}_g\le \theta_{\mathrm{split}},
\ s^{\max}_g>s_{\min}^{\mathrm{clone}}
\}.
\]

The code uses:

```text
theta_split = thresholds.split_scale
s_min_clone = 0.01
```

For each cloned Gaussian, all raw parameters are copied. For each split
Gaussian, the code creates two children. The raw scale tensor is shifted by
\(-\log 2\), and the means are perturbed:

\[
\bar{\mathbf s}_{g'} \leftarrow \bar{\mathbf s}_g-\log 2,
\]

\[
\boldsymbol{\mu}_{g'}
\leftarrow
\boldsymbol{\mu}_g
+
0.1\,\mathbf s_g\odot\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal N(\mathbf 0,\mathbf I).
\]

This is the exact `_generate_new_points()` behavior. The scale shift is applied
in raw scale space.

### 4.4 Contribution- and Attenuation-Aware Pruning

First define weak local-response candidates:

\[
\mathcal W
=
\{g\mid
\alpha_g < \theta_\alpha
\ \mathrm{or}\
\iota_g < \theta_\iota
\}.
\]

In current `ours`, \(\iota_g=1\), so pruning is usually driven by
\(\alpha_g\).

Under responsibility mode, pruning is additionally restricted to low
contribution:

\[
\theta_C
=
Q_{q_C}(\{C_g\}_{g=1}^{G}),
\qquad
q_C=0.10,
\]

\[
\mathcal C_{\mathrm{low}}
=
\{g\mid C_g\le \theta_C\}.
\]

High-attenuation Gaussians are protected:

\[
\theta_\tau
=
Q_{q_\tau}(\{\tau_g\}_{g=1}^{G}),
\qquad
q_\tau=0.90,
\]

\[
\mathcal A_{\mathrm{protect}}
=
\{g\mid \tau_g\ge \theta_\tau\}.
\]

The final prune set is:

\[
\mathcal P
=
\mathcal W
\cap
\mathcal C_{\mathrm{low}}
\cap
\left(\mathcal A_{\mathrm{protect}}\right)^c.
\]

This pruning rule is what makes the density controller compatible with the
local-global renderer: weak local echo Gaussians are not removed if their global
attenuation attribute is among the high-attenuation group.

### 4.5 What Is Not in the Final Formula

The code can compute high-residual max hits:

\[
H_g
=
\sum_{i,p}
\mathbf 1
\left[
g=\arg\max_{g'} c_{ig'p}
\right]
\mathbf 1[e_{ip}\ge Q_{0.90}(e_i)].
\]

Development ablations showed that the max-hit multiplier is not needed. It
should be mentioned only as an ablated/deprecated diagnostic, not as part of the
final paper method.

The code also supports optional structure weighting in `dc_diagnostics.py`, but
the final production config sets:

```text
structure_weight_enabled = false
```

Therefore, structure weighting should not appear in the main formula.

## 5. Innovation III: Easy-Pose Canonical Sweep Prior

Easy-pose is implemented in `scene/container/single_sweep_container.py`. It is
not a learned pose network and not the earlier residual pose optimizer.

### 5.1 Posed Input

The standard reconstruction input is:

\[
\mathcal D_{\mathrm{posed}}
=
\{(\mathbf I_i,\mathbf T_i)\}_{i=1}^{N},
\qquad
\mathbf T_i\in SE(3).
\]

This requires one tracked pose per frame.

### 5.2 Canonical Sweep Pose Construction

When:

```text
scene.loading.pose_mode = canonical_sweep
```

the loader replaces file poses with canonical poses. The effective spacing is:

\[
d_0
=
\begin{cases}
d_{\mathrm{cfg}}, & d_{\mathrm{cfg}}>0,\\
\Delta_{\mathrm{elev}}, & d_{\mathrm{cfg}}\le 0,
\end{cases}
\]

\[
d
=
\rho d_0,
\]

where:

```text
d_cfg = canonical_sweep_delta_mm
Delta_elev = dataset_meta.delta_elevational_mm
rho = canonical_sweep_delta_scale
```

For final uniform mode:

\[
o_i
=
(i-i_0)d.
\]

The axis is selected from signed coordinate axes:

\[
\mathbf u\in
\{\pm\mathbf e_x,\pm\mathbf e_y,\pm\mathbf e_z\}.
\]

The current code constructs:

\[
\tilde{\mathbf T}_i
=
\begin{bmatrix}
\mathbf I_3 & o_i\mathbf u\\
\mathbf 0^\top & 1
\end{bmatrix}.
\]

This is stricter than a general pose prior: the current implementation uses an
identity canonical rotation and zero reference origin, with translation only
along the configured canonical axis. In paper language, this can be described
as choosing a canonical sweep coordinate system and representing the sweep by a
single calibrated inter-frame spacing.

The code also supports two diagnostic spacing modes:

\[
o_i
=
\rho
\left(
\sum_{j<i}
\|\mathbf t_{j+1}^{\mathrm{file}}-\mathbf t_j^{\mathrm{file}}\|_2
-
\sum_{j<i_0}
\|\mathbf t_{j+1}^{\mathrm{file}}-\mathbf t_j^{\mathrm{file}}\|_2
\right)
\]

for `file_step_norm`, and:

\[
o_i
=
\rho
\left(
\sum_{j<i}
(\mathbf t_{j+1}^{\mathrm{file}}-\mathbf t_j^{\mathrm{file}})^\top\mathbf u
-
\sum_{j<i_0}
(\mathbf t_{j+1}^{\mathrm{file}}-\mathbf t_j^{\mathrm{file}})^\top\mathbf u
\right)
\]

for `file_axis_projection`. These are not the final main method; the validated
paper claim uses the uniform canonical sweep.

### 5.3 Easy-Pose Objective

The easy-pose training objective replaces tracked poses with canonical poses:

\[
\min_{\mathcal G}
\sum_{i\in\mathcal T}
\mathcal L
\left(
\mathcal R_\Theta(\mathcal G,\tilde{\mathbf T}_i),
\mathbf I_i
\right).
\]

The input dependency changes from:

\[
\{(\mathbf I_i,\mathbf T_i)\}_{i=1}^{N}
\quad\rightarrow\quad
\{\mathbf I_i\}_{i=1}^{N} + d.
\]

This reduces the pose requirement from a per-frame \(SE(3)\) sequence to a
single calibrated spacing and a canonical scan direction. The scoped claim is:

```text
regular or near-regular one-sweep reconstruction with low-cost calibrated spacing,
not arbitrary universal pose-free freehand reconstruction.
```

### 5.4 Spacing Stress Formula

The spacing stress used in validation is:

\[
d_\rho
=
\rho d,
\qquad
\rho\in
\{0.60,0.75,0.90,1.00,1.10,1.25,1.50\}.
\]

The corresponding canonical pose is:

\[
\tilde{\mathbf T}_i(d_\rho)
=
\begin{bmatrix}
\mathbf I_3 & (i-i_0)d_\rho \mathbf u\\
\mathbf 0^\top & 1
\end{bmatrix}.
\]

The validated conservative pass band is approximately:

\[
\rho\in[0.90,1.25].
\]

## 6. Training and Optimization Loop

The offline training callback is `methods/ours/gs_only_callbacks.py`. For each
iteration:

1. sample one training slice;
2. render \(\hat{\mathbf I}_i\);
3. compute image and scale losses;
4. backpropagate to Gaussian parameters;
5. update Gaussian parameters with `OursGaussSolver`;
6. if the iteration is a DC iteration, compute responsibility stats and modify
   the Gaussian topology.

In formula form:

\[
\mathcal G^{(t+\frac{1}{2})}
=
\operatorname{AdamStep}
\left(
\mathcal G^{(t)},
\nabla_{\mathcal G}\mathcal L^{(t)}
\right),
\]

and at density-control iterations:

\[
\mathcal G^{(t+1)}
=
\operatorname{Prune}
\left(
\operatorname{SplitClone}
(\mathcal G^{(t+\frac{1}{2})},\mathcal D),
\mathcal P
\right).
\]

Otherwise:

\[
\mathcal G^{(t+1)}
=
\mathcal G^{(t+\frac{1}{2})}.
\]

## 7. Code-Aligned Method Configuration

The final method should be associated with the following flags:

```text
model: ours_us_focus
model.slice_render.shader_weight_mode=us_focus
model.slice_render.compositing_mode=additive
model.slice_render.mapper_kind=tile
model.slice_render.ray_attenuation_enabled=true
model.slice_render.ray_attenuation_source=attenuation
model.slice_render.ray_attenuation_mapper_kind=ray_csr
model.slice_render.ray_attenuation_formula=slice_depth
model.slice_render.ray_attenuation_candidate_k_sigma=2.0
model.slice_render.ray_attenuation_beta=0.60
model.slice_render.ray_attenuation_strength=0.30
model.slice_render.ray_attenuation_min_transmission=0.55
model.offline.density_control.strategy=contribution_residual
model.offline.density_control.timing=1000/25000/2000
model.offline.density_control.responsibility.score_quantile=0.98
model.offline.density_control.responsibility.max_fraction=0.05
model.offline.density_control.responsibility.low_contribution_quantile=0.10
model.offline.density_control.responsibility.protect_attenuation_enabled=true
scene.loading.pose_mode=canonical_sweep      # easy-pose variant only
scene.loading.canonical_sweep_spacing_mode=uniform
```

## 8. Writing Notes and Boundaries

1. The main renderer formula should be the `us_focus` local branch multiplied
   by the ray-CSR `slice_depth` global modulation.
2. The `integral` ray formula, `probe3d`, and `plane_cond2d` are valid code
   paths and useful for appendix or implementation history, but they are not
   the final production method formula.
3. Responsibility DC should use
   \(s_g^{\mathrm{dc}}=\bar r_g\log(1+C_g)\). Do not include a max-hit
   multiplier in the final method.
4. Easy-pose should be written as a scoped acquisition prior. The current code
   uses identity canonical rotation and axis-aligned translation.
5. Report runtime overhead for local-global rendering and Gaussian count changes
   for responsibility DC.
