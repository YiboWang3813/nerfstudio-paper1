# Ours Methods Source Of Truth

This file is the writing source of truth for the paper Methods section. It
resolves the current documentation into one final, production-aligned method
description.

## 1. Task And Representation

The task is offline one-sweep ultrasound scene reconstruction. Given a sequence
of B-mode slices and probe poses, optimize a 3D Gaussian scene that renders
held-out slices from the same sweep.

For frame `i`, the target image is `I_i` and the rendered image is `Ihat_i`.
The train/test split follows the production holdout NVS protocol. The primary
comparison row uses file poses:

```text
{I_i, T_i}_{i=1}^N -> Gaussian scene G
```

Easy-Pose replaces `T_i` with canonical sweep poses only in the pose-dependency
module and ablations.

Each Gaussian stores:

```text
mean mu_g in world mm
scale s_g in mm
quaternion q_g in repository layout (w, x, y, z)
echo alpha alpha_g
attenuation tau_g
```

In current Ours:

```text
opacities    = echo_alphas
intensities  = ones_like(echo_alphas)
attenuations = separate trainable global attenuation attribute
```

The production Gaussian constraints are bounded sigmoid scales, bounded echo
alpha, and bounded attenuation.

## 2. Final Renderer

The final renderer is a local-global ultrasound renderer:

```text
local branch  = us_focus coordinate-aware local slice response
global branch = ray-CSR attenuation modulation
output        = local response * global modulation
```

The deployed renderer is not a pure local splatter and not a pure ray renderer.
The local branch remains the image-forming branch; the global branch modulates
it.

### 2.1 Local Branch

For a Gaussian and frame, transform the Gaussian center and covariance from
world coordinates into the probe coordinate frame. Pixel coordinates are mapped
to probe-local millimetres by the slice-grid geometry:

```text
xi_lat(x) = (x - lateral_column_offset) * delta_lateral_mm
xi_ax(y)  = y * delta_axial_mm + z_near_mm
```

The local branch uses `shader_weight_mode=us_focus`, `mapper_kind=tile`, and
`compositing_mode=additive`.

The `us_focus` shader applies a depth-dependent elevational focus precision,
projects the effective covariance to the 2D slice plane, computes a Gaussian
footprint on the image plane, multiplies it by elevational slice response and
determinant normalization, then sums contributions:

```text
L_ip = sum_g alpha_g * eta_ig * det_norm_ig * phi_igp
```

Because intensities are fixed to one, local brightness is controlled by echo
alpha and footprint response.

### 2.2 Global Ray Attenuation Branch

The production branch uses ray-CSR candidates and integral attenuation:

```text
ray_attenuation_enabled = true
ray_attenuation_source = attenuation
ray_attenuation_mapper_kind = ray_csr
ray_attenuation_formula = integral
ray_attenuation_quadrature_points = 3
ray_attenuation_candidate_k_sigma = 2.0
ray_attenuation_beta = 0.80
ray_attenuation_strength = 0.55
ray_attenuation_min_transmission = 0.35
```

For each lateral ray bin, candidate Gaussians are sorted by axial depth. For a
pixel, candidates in front of or near the pixel depth contribute to a
multiplicative transmission map. The final rendered pixel is:

```text
Ihat_ip = L_ip * (1 - beta + beta * T_ip)
```

where `T_ip` is the clamped product of per-Gaussian transmission terms computed
by the CUDA ray branch. The exact implementation is in
`ops/ours_renderer/src/shader/ray_attenuation.cu`.

## 3. Responsibility-Aware Density Control

The density controller modifies Gaussian topology during training. It is a
capacity allocation mechanism, not a new image loss.

At a density-control iteration, the diagnostic pass replays local contributions
and accumulates per-Gaussian statistics:

```text
C_g = sum_p c_gp
R_g = sum_p c_gp * abs(Ihat_p - I_p)
```

The residual-weighted responsibility is:

```text
rbar_g = R_g / (C_g + eps)
score_g = rbar_g * log(1 + C_g)
```

Densification selects the high-score set under a quantile and max-fraction
budget. Clone/split follows the existing Gaussian topology rule based on active
scale size.

Pruning starts from weak local response, then applies contribution-aware and
attenuation-aware constraints:

```text
prune if weak echo AND low contribution AND not high attenuation
```

The final paper formula should not include the old max-hit multiplier. It is a
diagnostic/ablation artifact, not the final DC score.

## 4. Easy-Pose Canonical Sweep Prior

Easy-Pose replaces per-frame file poses with a canonical one-dimensional sweep
trajectory for regular or near-regular acquisitions.

The standard input is:

```text
{I_i, T_i}_{i=1}^N
```

Easy-Pose changes this to:

```text
{I_i}_{i=1}^N + calibrated adjacent-frame spacing d
```

The current implementation uses identity canonical rotation and axis-aligned
translation:

```text
Ttilde_i = [ I_3, (i - i0) * d * u
             0,   1                 ]
```

The final paper claim is scoped:

```text
regular or near-regular one-sweep reconstruction with calibrated spacing
```

Do not claim universal pose-free freehand reconstruction. `spine_phantom` is a
negative-control case for Easy-Pose because its geometry is not regular enough
for the canonical sweep prior.

## 5. Training Objective

The latest GS-strict production comparison row uses:

```text
loss = 0.8 * L1 + 0.2 * SSIM
disabled auxiliary training losses = scale_wall, highpass, LPIPS, DISTS, GMSD,
                                     and non-approved method auxiliary losses
checkpoint selection = final 30k checkpoint
iterations = 30000
initial Gaussians = 1000000
DC timing = start 1000, end 25000, freq 2000
```

Older pre-reset ablation documents used `0.65 L1 + 0.35 SSIM + 50 scale_wall`
and best-checkpoint selection. Do not use those settings for the final Methods
unless explicitly discussing historical development.

## 6. What To Emphasize In The Paper

Recommended contribution order:

1. Local-global ultrasound Gaussian renderer.
2. Responsibility-aware density control.
3. Easy-Pose canonical sweep prior.

Recommended framing:

```text
The renderer is the main reconstruction-quality contributor.
Responsibility DC is topology/control evidence and is near-neutral to weakly
positive in final image metrics.
Easy-Pose reduces pose dependency under a scoped regular-sweep protocol.
```

## 7. What Not To Say

Avoid these claims:

```text
Ours is universally pose-free.
Responsibility DC is the largest PSNR source.
The final ray branch is slice_depth.
Ours is just UltraGray with renamed variables.
The global branch can produce the image by itself.
The max-hit term is part of the final DC score.
3D cross-method SuperSplat figures are fair evidence.
```

