# Ours Introduction Notes for Writing Agents

This document is a writing-oriented introduction scaffold for the final `ours`
method. It is grounded in the current implementation and validated experiment
records under `methods/ours`, `ops/ours_renderer`, and `docs/methods/ours`.

Primary implementation references:

```text
methods/ours/render.py
methods/ours/gaussian_container.py
methods/ours/density_control.py
methods/ours/dc_diagnostics.py
scene/container/single_sweep_container.py
ops/ours_renderer/
configs_hydra/model/ours_us_focus.yaml
configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml
```

Primary validation reference:

```text
docs/methods/ours/validated_paper_innovations.md
```

## 1. Research Field

The work belongs to offline freehand ultrasound scene reconstruction. The input
is a single sweep of 2D B-mode ultrasound images, and the goal is to reconstruct
a 3D Gaussian representation that can render held-out slices from the same
sweep. This setting is different from ordinary multi-view natural-image 3DGS:

1. The observation is slice-like rather than perspective-surface-like.
2. Each image has a narrow field of view and only sees a thin local cross-section
   of the anatomy.
3. Ultrasound intensity is affected by both local echo response and propagation
   effects such as attenuation-like modulation.
4. The train/test frames come from one offline sweep. The method optimizes one
   scene per sweep rather than learning a generalizable feed-forward model.
5. Accurate per-frame probe poses are expensive in realistic clinical scanning
   because they require optical/electromagnetic tracking hardware or careful
   external calibration.

Therefore, directly applying standard 3DGS assumptions is not enough. Standard
3DGS density control was designed for camera views of surfaces, and a pure local
slice splatter only models nearby pixel response. At the same time, a pure
ray-based ultrasound renderer can overfit the propagation view of the problem
and underuse the coordinate-transform local response that is essential for
one-sweep slice reconstruction.

Ours is best written as an offline one-sweep ultrasound Gaussian reconstruction
framework with three connected innovations:

| Level | Innovation | Role |
| --- | --- | --- |
| Acquisition / pose input | Easy-pose canonical sweep prior | Reduces dependency on external per-frame tracking poses. |
| Rendering formulation | Local-global ultrasound renderer | Combines local echo response and global propagation modulation. |
| Optimization / topology | Responsibility-aware density control | Allocates Gaussian capacity using residual-contribution responsibility. |

## 2. Phenomena That Motivated the Innovations

### 2.1 Local Rendering Alone Misses Propagation-Style Effects

The original `ours` renderer follows the coordinate-transform logic used by
GaussianPile and slice-adapted 3DGS: transform world-mm Gaussians into the
probe/slice coordinate system and splat them locally on the image plane. This
matches the slice observation geometry well, but it is still local. It asks each
pixel to be explained only by Gaussians whose projected footprint overlaps that
pixel.

During development, this local-only behavior was found to be too weak for
ultrasound images. Ultrasound brightness is not only a local response; deeper
or laterally aligned structures can be globally modulated by upstream tissue and
attenuation-like propagation. This motivated a renderer with two separate
responsibilities:

```text
local branch  -> coordinate-aware echo response;
global branch -> ray-CSR propagation modulation;
final image   -> local response modulated by global transmission.
```

The key design decision is not to replace the local coordinate-transform
renderer with a pure ray renderer. Ours keeps the local tile branch because the
task is still one-sweep slice reconstruction, then adds a separate global branch
only for modulation.

### 2.2 Gradient-Only Density Control Is Not Reliable for Sparse Ultrasound Slices

Standard 3DGS density control uses accumulated mean-position gradients to decide
where to clone or split Gaussians. In one-sweep ultrasound reconstruction, high
gradient does not always mean that a Gaussian is responsible for visible
residual error. A Gaussian can have high gradient because of geometric
instability, while another Gaussian can be weak in gradient but actually explain
a residual-heavy anatomical structure or an attenuation-relevant region.

The development experiments showed that this problem is strongest on fragmented
or sparse slice coverage. The final density control therefore computes, only at
density-control iterations, how much each Gaussian contributes to pixels with
large residual. The density controller then densifies Gaussians with high
residual-weighted contribution and prunes weak Gaussians only when they also
have low contribution, while protecting high-attenuation Gaussians.

This is why the innovation should be framed as topology/capacity allocation,
not as a new photometric loss.

### 2.3 Full Pose-Free Learning Was Too Unstable, but Regular Sweeps Have a Useful Prior

Earlier attempts to use pose-estimation networks or to optimize poses directly
with the reconstruction loss were not reliable enough. The core issue was that
ultrasound GS reconstruction can explain many image changes by moving or
reshaping Gaussians, so the image reconstruction loss does not provide a strong
and unambiguous gradient for pose correction. Learned pose priors also required
substantial training time and still produced errors too large for GS to consume.

However, the target scenario is not arbitrary handheld motion. It is an offline
one-sweep acquisition. For regular or near-regular sweeps, the strongest
practical prior is the adjacent-frame spacing along the sweep direction. This
motivated easy-pose: replace per-frame 6-DoF tracked poses with a canonical
sweep trajectory parameterized by a calibrated spacing. This does not claim
universal pose-free reconstruction. It claims low-cost tracking-free
reconstruction under a scoped regular-sweep protocol.

## 3. Technical Summary of the Three Innovations

### 3.1 Local-Global Ultrasound Renderer

Implementation path:

```text
methods/ours/render.py
methods/ours/gaussian_container.py
ops/ours_renderer/ours_renderer/autograd.py
ops/ours_renderer/src/shader/forward.cu
ops/ours_renderer/src/shader/ray_attenuation.cu
```

The Gaussian container stores world-mm centers, scales, quaternions, echo
alphas, and attenuations. In the production path, `opacities` are the activated
`echo_alphas`, `intensities` are fixed to one, and `attenuations` are separate
trainable attributes for the global branch.

The local branch uses the `us_focus` shader. It transforms each Gaussian into
probe coordinates, applies an ultrasound focus-inspired footprint, and sums the
local echo response in an additive local tile renderer.

The global branch uses ray-CSR candidates. For each lateral ray bin, candidate
Gaussians are sorted by axial depth. For a pixel, only Gaussians before that
pixel along the sweep-depth direction contribute to a multiplicative
transmission. The final deployed formula uses the simplified `slice_depth`
response, not the earlier integral formula:

```text
response = sigma * exp(-0.5 * ((mu_lat - x_lat)^2 + mu_ele^2) / sigma^2)
T_i      = tau_i + (1 - tau_i) * exp(-strength * response)
T_pixel  = clamp(prod_i T_i, min_transmission, 1)
image    = local_response * (1 - beta + beta * T_pixel)
```

This renderer addresses the gap between local slice splatting and global
ultrasound propagation behavior.

Validation:

```text
datasets = synthetic_liver, spine_phantom, tus_rec_2024_qc, tdsc_abus_2023
protocol = 1M initialization, 30k iterations, best-checkpoint metrics
baseline = local-only tile renderer
ours     = local response + ray-CSR global attenuation
```

Main result from `validated_paper_innovations.md`:

| Dataset / sweep | PSNR delta | L1 reduction | SSIM delta | LPIPS reduction | Wall-time overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver / l1 | +0.1975 dB | 3.15% | +0.0045 | 15.45% | +16.13% |
| spine_phantom / left1 | +2.5043 dB | 16.92% | +0.0447 | 19.56% | +23.90% |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | +0.2353 dB | 3.35% | +0.0063 | 11.68% | +16.83% |
| tdsc_abus_2023 / validation_100 | +0.2717 dB | 4.50% | +0.0097 | 10.05% | +12.95% |
| Mean | +0.8022 dB | 6.98% | +0.0163 | 14.18% | +17.45% |

Takeaway for writing: this is the strongest accuracy contribution and should be
introduced first among the technical modules.

### 3.2 Responsibility-Aware Density Control

Implementation path:

```text
methods/ours/dc_diagnostics.py
methods/ours/density_control.py
ops/ours_renderer/ours_renderer/diagnostics.py
ops/ours_renderer/src/shader/forward.cu
```

At density-control iterations, the CUDA diagnostic pass reuses the local tile
candidate path and accumulates per-Gaussian contribution statistics:

```text
contribution              = sum pixel contribution;
residual_contribution     = sum contribution * abs(rendered - target);
support                   = number of contributing pixels;
max_residual_hits         = count of high-residual pixels where the Gaussian is max contributor.
```

The final responsibility score is:

```text
residual_mean = residual_contribution / contribution
score         = residual_mean * log(1 + contribution)
```

The densification set is selected by high score quantiles and a maximum fraction
budget. Clone/split still follows the existing Gaussian topology rule: large
Gaussians are split, smaller valid Gaussians are cloned. Pruning remains based
on weak echo alpha/intensity, but under responsibility mode it is additionally
restricted to low-contribution Gaussians and excludes high-attenuation Gaussians.

This innovation addresses the mismatch between gradient magnitude and actual
pixel responsibility in sparse/high-noise ultrasound slice reconstruction.

Validation:

```text
datasets = synthetic_liver, spine_phantom, tus_rec_2024_qc, tdsc_abus_2023
protocol = 1M initialization, 30k iterations, fixed DC timing 1000/25000/2000
renderer = final local-global renderer
baseline = gradient DC
ours     = contribution-residual responsibility DC
```

Main result:

| Dataset / sweep | Gradient PSNR | Responsibility PSNR | Delta PSNR | Delta LPIPS | Delta L1 | Delta SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver / l1 | 32.1320 | 32.1329 | +0.0009 | +0.0007 | -0.000016 | +0.00031 |
| spine_phantom / left1 | 30.2839 | 30.6411 | +0.3572 | -0.0394 | -0.000566 | +0.02432 |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | 32.2889 | 32.3076 | +0.0187 | -0.0021 | -0.000046 | +0.00052 |
| tdsc_abus_2023 / validation_100 | 27.4367 | 27.4492 | +0.0125 | -0.0023 | -0.000013 | +0.00009 |

Component ablation showed:

1. `responsibility_densify_only` is an upper-bound but keeps roughly one million
   Gaussians, so it is not the deployable method.
2. Removing attenuation protection weakens results.
3. Removing the deprecated max-hit multiplier is almost identical to full
   responsibility, so max-hit should not appear in the final formula.

Takeaway for writing: this module is a capacity-allocation innovation with
modest but consistent quantitative gains and strong mechanism evidence.

### 3.3 Easy-Pose Canonical Sweep Prior

Implementation path:

```text
scene/container/single_sweep_container.py
configs_hydra/scene/*.yaml
configs_hydra/experiment/preproduction/ideas/ours/easy_pose/four_dataset_feasibility.yaml
```

The standard training input uses per-frame file poses:

```text
{I_i, T_i}_{i=1}^N
```

With `scene.loading.pose_mode=canonical_sweep`, the loader replaces the file
poses with a canonical sweep trajectory. In the current code, the canonical
rotation is identity and the translation is a scalar offset along a configured
axis:

```text
T_i = [I, offset_i * axis; 0, 1]
offset_i = (i - ref) * delta               # uniform mode
delta = canonical_sweep_delta_mm if > 0 else dataset_meta.delta_elevational_mm
delta <- delta * canonical_sweep_delta_scale
```

The production claim is not that arbitrary handheld scans can be reconstructed
without pose. The claim is that, under regular or near-regular one-sweep
protocols, a low-cost calibrated spacing prior can replace per-frame tracking
poses.

Validation:

```text
datasets = synthetic_liver, tus_rec_2024_qc, tdsc_abus_2023
paired sweeps = 16
protocol = 1M initialization, 30k iterations, final renderer + responsibility DC
baseline = file pose
ours     = canonical sweep pose, scale 1.00
```

Dataset-level PSNR gaps:

| Dataset | Pairs | Mean gap | Median gap | Min gap | Max gap | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic_liver | 6 | -0.0211 | -0.0009 | -0.0795 | +0.0137 | Strong retention. |
| tus_rec_2024_qc | 5 | +0.2719 | +0.2554 | +0.1549 | +0.4005 | Consistent positive retention. |
| tdsc_abus_2023 | 5 | +0.0003 | -0.0006 | -0.0082 | +0.0096 | Near-perfect tie. |
| all | 16 | +0.0771 | n/a | -0.0795 | +0.4005 | All paired sweeps pass strong retention. |

Spacing stress used:

```text
rho in {0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50}
delta_rho = rho * delta
```

Conservative cross-dataset statement: robust around `rho in [0.90, 1.25]`.
`spine_phantom` should remain a negative-control/limitation case because its
motion is highly non-uniform and not a clean regular-sweep acquisition.

Takeaway for writing: easy-pose is a clinically motivated acquisition-cost
innovation. It reduces pose dependency rather than improving reconstruction
quality by itself.

## 4. Recommended Introduction Narrative

The introduction should not present the three modules as disconnected patches.
A coherent narrative is:

1. Medical ultrasound freehand reconstruction needs 3D scene formation from
   sparse, noisy, slice-like observations.
2. Existing Gaussian methods provide efficient scene representations, but their
   rendering and topology-control assumptions are inherited from natural-image
   or local slice splatting settings.
3. Real one-sweep ultrasound has three practical obstacles:
   - image formation is both local and propagation-modulated;
   - Gaussian topology should follow residual responsibility, not only gradient;
   - per-frame tracking poses are expensive and often unavailable.
4. Ours addresses these obstacles at three levels:
   - a local-global ultrasound renderer;
   - responsibility-aware density control;
   - an easy-pose canonical sweep prior.
5. Experiments show:
   - local-global rendering improves all four representative datasets;
   - responsibility DC gives consistent auxiliary gains and better topology
     behavior;
   - easy-pose preserves reconstruction quality on 16 regular/near-regular
     paired sweeps while replacing per-frame pose input with calibrated spacing.

Recommended contribution wording:

1. We propose a local-global ultrasound Gaussian renderer that combines
   coordinate-aware local echo splatting with ray-CSR global propagation
   modulation.
2. We introduce responsibility-aware density control, using residual-weighted
   Gaussian contribution to guide densification and contribution/attenuation
   aware pruning.
3. We develop an easy-pose canonical sweep prior for low-cost tracking-free
   one-sweep reconstruction, reducing per-frame `SE(3)` pose dependency to a
   calibrated sweep-spacing prior under regular scanning protocols.

## 5. Claims to Avoid

Do not claim:

1. universal pose-free reconstruction for arbitrary handheld scans;
2. responsibility DC as the largest standalone accuracy improvement;
3. local-global rendering as a copy of UltraGray;
4. that the renderer is free in runtime.

Instead, state:

1. easy-pose works under regular/near-regular one-sweep acquisition;
2. responsibility DC is an optimization/topology module;
3. local-global differs by preserving coordinate-transform local response while
   adding a separate ray-CSR modulation branch;
4. renderer runtime overhead is explicit and acceptable relative to the
   accuracy gain.
