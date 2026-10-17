# Draft Methods Structure

This is a manuscript-facing skeleton. It should be rewritten into final paper
language, but the section order and technical claims are safe.

## 1. Problem Formulation

We consider offline reconstruction from a single B-mode ultrasound sweep. The
input consists of a sequence of 2D images and, for the tracked setting, the
corresponding probe-to-world poses. The objective is to optimize a Gaussian
scene that can render held-out slices from the same sweep.

Define the Gaussian scene as:

```text
G = {mu_g, s_g, q_g, alpha_g, tau_g}_{g=1}^G
```

where `mu` is in world millimetres, `s` are positive scales, `q` is a unit
quaternion, `alpha` is local echo response, and `tau` is the global attenuation
attribute.

## 2. Ultrasound Gaussian Scene Parameterization

Describe bounded scale activation, bounded echo alpha, bounded attenuation,
and fixed unit intensity. State quaternion layout `(w, x, y, z)` only if useful
for implementation detail.

## 3. Local-Global Ultrasound Renderer

Introduce the decomposition:

```text
Ihat_ip = L_ip * M_ip
M_ip = 1 - beta + beta * T_ip
```

Then split into local branch and global branch.

### 3.1 Local Focus-Aware Slice Response

Describe probe-coordinate transform, slice-grid coordinate mapping, tile
candidate construction, and `us_focus` response.

Use a compact formula:

```text
L_ip = sum_{g in N_tile(ip)} alpha_g eta_ig nu_ig phi_igp
```

where `eta` is elevational focus response, `nu` determinant normalization, and
`phi` the 2D footprint.

### 3.2 Ray-CSR Global Attenuation

Describe ray bins, axial sorting, and in-front candidate selection. State that
the production method uses integral attenuation with three quadrature samples.
Do not expand every CUDA detail unless the paper has space; put the full detail
in appendix if needed.

## 4. Responsibility-Aware Density Control

Start from the limitation of gradient-only DC in sparse ultrasound slices.
Define contribution and residual-weighted contribution:

```text
C_g = sum c_gp
R_g = sum c_gp e_p
rbar_g = R_g / (C_g + eps)
score_g = rbar_g log(1 + C_g)
```

Then describe densify and prune:

```text
densify high responsibility score
prune weak echo only if low contribution and not high attenuation
```

## 5. Easy-Pose Canonical Sweep Prior

Define the pose replacement:

```text
T_i -> Ttilde_i(d)
Ttilde_i = [I, (i - i0) d u; 0, 1]
```

State scope clearly: regular/near-regular one-sweep reconstruction with
calibrated spacing. Put negative-control limitations in Results or Discussion,
not as a main-method weakness.

## 6. Optimization And Implementation

State:

```text
30k iterations
1M initial Gaussians
0.8 L1 + 0.2 SSIM
final 30k checkpoint for comparison
DC timing 1000/25000/2000
CUDA tile renderer with ray-CSR global branch
```

Keep detailed hyperparameters in a table:

```text
renderer: us_focus + ray_csr integral q=3
beta: 0.80
strength: 0.55
min transmission: 0.35
candidate k_sigma: 2.0
DC: contribution_residual
score quantile: production config
protect attenuation quantile: 0.80
```

