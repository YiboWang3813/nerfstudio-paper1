# Validated Ours Innovations for Paper

Date: 2026-06-22

This file records the Ours innovations that are currently validated clearly
enough to be written into the paper. It stays under `docs/methods/ours/`
because these are method-owned results, not a separate paper-only artifact.

Validated innovations:

1. local-global ultrasound renderer;
2. responsibility-aware density control;
3. easy-pose / sweep-constrained pose-free prior.

The recommended paper narrative is not three unrelated tricks. Ours should be written as an offline one-sweep ultrasound Gaussian reconstruction framework with innovations at three levels:

| Level | Innovation | Role |
| --- | --- | --- |
| Rendering formulation | Local-global ultrasound renderer | Models local echo response and global propagation modulation. |
| Optimization / topology | Responsibility-aware density control | Allocates Gaussian capacity using residual-contribution responsibility instead of only mean gradients. |
| Acquisition / pose input | Easy-pose canonical sweep prior | Reduces per-frame tracking-pose dependency to a low-cost calibrated sweep spacing prior. |

## 1. Local-Global Ultrasound Renderer

Primary result document:

```text
docs/methods/ours/local_global_renderer/four_dataset_30k_validation.md
```

Development / implementation reports:

```text
docs/methods/ours/local_global_renderer/implementation_plan.md
docs/methods/ours/local_global_renderer/stage_a_local_global_ray_tile.md
docs/methods/ours/local_global_renderer/stage_b_separated_attenuation.md
docs/methods/ours/local_global_renderer/stage_c_ray_csr_attenuation.md
docs/methods/ours/local_global_renderer/stage_d_echo_alpha_ray_geometry.md
docs/methods/ours/local_global_renderer/stage_e_ultragray_ray_integral.md
docs/methods/ours/local_global_renderer/slice_depth_attenuation_formula.md
docs/methods/ours/local_global_renderer/ray_prefix_speed_optimization.md
docs/methods/ours/local_global_renderer/speed_optimization_backlog.md
```

### 1.1 Core Method

The baseline renderer is a local tile splatter. It projects Gaussians from world coordinates to the probe/slice pixel coordinate system and evaluates only local candidates around each pixel.

Ours keeps this local branch for echo response and adds a separate ray-CSR global branch for propagation modulation:

```text
local branch:   tile-based projected Gaussian response
global branch:  ray-CSR attenuation candidates along the slice-depth direction
fusion:         rendered echo response modulated by global transmission
```

Recommended final renderer overrides:

```text
model.slice_render.ray_attenuation_enabled=true
model.slice_render.ray_attenuation_source=attenuation
model.slice_render.ray_attenuation_mapper_kind=ray_csr
model.slice_render.ray_attenuation_formula=slice_depth
model.slice_render.ray_attenuation_candidate_k_sigma=2.0
model.slice_render.ray_attenuation_beta=0.60
model.slice_render.ray_attenuation_strength=0.30
model.slice_render.ray_attenuation_min_transmission=0.55
model.slice_render.ray_attenuation_quadrature_points=1
```

Paper-facing positioning:

- Different from GaussianPile / slice-adapted 3DGS: not only local projected splatting.
- Different from UltraGray: not a pure ray-candidate renderer; Ours keeps the coordinate-transform local response branch and adds a separate global modulation branch.
- The final formula should emphasize local response and global propagation as separated responsibilities, not as a single opacity/intensity attribute.

### 1.2 Main Ablation

Protocol:

```text
datasets = synthetic_liver, spine_phantom, tus_rec_2024_qc, tdsc_abus_2023
one representative sweep per dataset
1M Gaussian initialization
30k training iterations
standard density control
best-checkpoint test metrics
baseline = local-only tile renderer
ours = local tile response + ray-CSR global attenuation
```

Main result:

| Dataset / sweep | PSNR delta | L1 reduction | SSIM delta | LPIPS reduction | Wall-time overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver / l1 | +0.1975 dB | 3.15% | +0.0045 | 15.45% | +16.13% |
| spine_phantom / left1 | +2.5043 dB | 16.92% | +0.0447 | 19.56% | +23.90% |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | +0.2353 dB | 3.35% | +0.0063 | 11.68% | +16.83% |
| tdsc_abus_2023 / validation_100 | +0.2717 dB | 4.50% | +0.0097 | 10.05% | +12.95% |
| Mean | +0.8022 dB | 6.98% | +0.0163 | 14.18% | +17.45% |

Conclusion:

- The local-global renderer is the strongest reconstruction-quality innovation.
- It improves PSNR, L1, SSIM, and LPIPS on all four representative datasets.
- The observed wall-time overhead is bounded and acceptable for the gain.
- This should be the first technical contribution in the paper.

## 2. Responsibility-Aware Density Control

Primary result documents:

```text
docs/methods/ours/responsibility_density_control/idea_plan.md
docs/methods/ours/responsibility_density_control/fixed_protocol_validation.md
```

Production scripts:

```text
scripts/methods/ours/responsibility_density_control/slurm/submit_dc_fixed_protocol_main_30k.sh
scripts/methods/ours/responsibility_density_control/slurm/submit_dc_fixed_protocol_ablation_30k.sh
scripts/methods/ours/responsibility_density_control/visualize_responsibility_dc.py
scripts/methods/ours/responsibility_density_control/slurm/run_responsibility_dc_visuals.sh
```

### 2.1 Core Method

Vanilla 3DGS-style density control selects densification candidates mainly through accumulated Gaussian mean gradients. In one-sweep ultrasound reconstruction, this can allocate capacity to high-gradient but low-responsibility Gaussians and can prune weak but physically meaningful attenuating Gaussians.

The final responsibility DC uses residual-weighted contribution:

```math
C_i = \sum_p c_{ip},
```

where `c_ip` is the contribution of Gaussian `i` to pixel `p`.

Let `r_p` be the image residual. The residual-weighted responsibility is:

```math
\bar r_i =
\frac{\sum_p c_{ip} r_p}{\sum_p c_{ip} + \epsilon}.
```

The final densification score is:

```math
s_i = \bar r_i \log(1 + C_i).
```

The selected densification set is the top responsibility-score subset under residual and budget constraints:

```math
\mathcal D =
\{i \mid \bar r_i \ge Q_{0.90}(\bar r),\ s_i \ge Q_{0.98}(s)\},
\quad
|\mathcal D| \le 0.05N.
```

Pruning remains contribution-aware and protects high-attenuation Gaussians:

```math
\mathcal P =
\{i \mid C_i \le Q_{0.10}(C),\ \alpha_i \text{ weak},\ a_i < Q_{0.90}(a)\}.
```

Recommended final configuration:

```text
strategy = contribution_residual
mix_mode = responsibility
high_residual_quantile = 0.90
score_quantile = 0.98
max_fraction = 0.05
low_contribution_quantile = 0.10
protect_attenuation_enabled = true
protect_attenuation_quantile = 0.90
structure_weight_enabled = false
dc_start/end/freq = 1000/25000/2000
```

Important implementation decision:

- The max-hit multiplier should not be included in the final formula.
- The ablation showed `responsibility_no_max_hit` is almost indistinguishable from full responsibility.
- The cleaner final score is `s_i = bar_r_i * log(1 + C_i)`.

### 2.2 Main Fixed-Protocol Validation

Protocol:

```text
datasets = synthetic_liver, spine_phantom, tus_rec_2024_qc, tdsc_abus_2023
1M Gaussian initialization
30k training iterations
same fixed DC timing for all variants: 1000/25000/2000
final local-global renderer
best-checkpoint metrics
baseline = gradient DC
ours = responsibility DC
```

Main result:

| Dataset / sweep | Gradient PSNR | Responsibility PSNR | Delta PSNR | Delta LPIPS | Delta L1 | Delta SSIM | Time delta | Best Gaussians delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver / l1 | 32.1320 | 32.1329 | +0.0009 | +0.0007 | -0.000016 | +0.00031 | -2.7% | +8.8% |
| spine_phantom / left1 | 30.2839 | 30.6411 | +0.3572 | -0.0394 | -0.000566 | +0.02432 | +3.0% | +111.0% |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | 32.2889 | 32.3076 | +0.0187 | -0.0021 | -0.000046 | +0.00052 | -3.5% | +7.9% |
| tdsc_abus_2023 / validation_100 | 27.4367 | 27.4492 | +0.0125 | -0.0023 | -0.000013 | +0.00009 | +6.9% | +5.9% |

Conclusion:

- Responsibility DC improves PSNR on all four representative datasets.
- LPIPS, L1, and SSIM improve on three of four datasets.
- The expected effect size is smaller than the renderer because DC is an auxiliary topology/capacity allocator.
- The spine case provides the clearest mechanism evidence: gradient-only DC under-allocates capacity in fragmented sparse-slice coverage.

### 2.3 Component Ablation

Representative ablation rows:

| Dataset | Variant | Best PSNR | LPIPS | L1 | SSIM | Best Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| spine_phantom / left1 | responsibility_full | 30.6314 | 0.2124 | 0.015104 | 0.73894 | 84,273 |
| spine_phantom / left1 | responsibility_densify_only | 31.0093 | 0.1636 | 0.013888 | 0.78955 | 1,009,101 |
| spine_phantom / left1 | responsibility_no_attenuation_protect | 30.2282 | 0.2588 | 0.015653 | 0.71547 | 39,437 |
| spine_phantom / left1 | responsibility_no_max_hit | 30.6457 | 0.2115 | 0.015076 | 0.73915 | 84,398 |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | responsibility_full | 32.3032 | 0.1736 | 0.013719 | 0.85587 | 220,068 |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | responsibility_densify_only | 32.3507 | 0.1521 | 0.013581 | 0.85894 | 1,021,149 |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | responsibility_no_attenuation_protect | 32.2981 | 0.1748 | 0.013743 | 0.85551 | 206,597 |
| tus_rec_2024_qc / 010_LH_Per_L_DtP_qc_0005_0370 | responsibility_no_max_hit | 32.3046 | 0.1735 | 0.013729 | 0.85576 | 220,355 |

Interpretation:

- `responsibility_densify_only` is a useful upper-bound but keeps roughly one million Gaussians, so it is not a clean default.
- `responsibility_no_attenuation_protect` is consistently weaker, supporting attenuation-aware protection.
- `responsibility_no_max_hit` is effectively identical to full responsibility, so the max-hit term should be removed from the final formula.
- Mechanism visualizations show sparse selected Gaussian sets concentrated around residual-heavy structural regions instead of uniform FOV coverage.

## 3. Easy-Pose / Sweep-Constrained Pose-Free Prior

Primary result documents:

```text
docs/methods/ours/easy_pose/sweep_constrained_implementation_plan.md
docs/methods/ours/easy_pose/full_validation_and_spacing_stress.md
docs/methods/ours/easy_pose/canonical_geometry_diagnosis.md
```

Production / validation scripts:

```text
scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_l_full_validation_30k.sh
scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_m_spacing_tolerance_30k.sh
scripts/methods/ours/easy_pose/slurm/submit_pose_free_stage_k_spine_arclength_subset_30k.sh
scripts/methods/ours/easy_pose/select_arclength_uniform_frames.py
scripts/methods/ours/easy_pose/analyze_canonical_sweep_pose_geometry.py
scripts/methods/ours/easy_pose/analyze_spine_phantom_qc_windows.py
```

### 3.1 Core Method

The standard posed reconstruction input is:

```math
\{I_i, T_i\}_{i=1}^N \rightarrow \mathcal G,
\quad
T_i \in SE(3).
```

This requires a per-frame external tracking pose, with roughly `6N` pose degrees of freedom.

Easy-pose replaces the per-frame pose sequence with a canonical one-sweep trajectory controlled by a calibrated frame spacing:

```math
\tilde T_i(d) =
\begin{bmatrix}
R_0 & t_0 + (i-i_0)d\,\mathbf n \\
0 & 1
\end{bmatrix}.
```

Here:

- `R0` is the canonical probe orientation;
- `t0` is the canonical sweep origin;
- `n` is the elevational sweep direction;
- `d` is the calibrated adjacent-frame spacing;
- `i-i0` is the frame index offset.

The reconstruction objective changes from:

```math
\min_{\mathcal G}
\sum_{i \in \mathcal T}
\mathcal L(\mathcal R_\theta(\mathcal G, T_i), I_i)
```

to:

```math
\min_{\mathcal G}
\sum_{i \in \mathcal T}
\mathcal L(\mathcal R_\theta(\mathcal G, \tilde T_i(d)), I_i).
```

Paper-facing task definition:

```text
tracking-based one-sweep reconstruction:
  {I_i, T_i}_{i=1}^N -> G

easy-pose one-sweep reconstruction:
  {I_i}_{i=1}^N plus calibrated spacing d -> G
```

This should be framed as low-cost tracking-free reconstruction for regular or near-regular one-sweep scanning, not universal pose-free reconstruction.

### 3.2 Stage L Full Validation

Protocol:

```text
datasets = synthetic_liver, tus_rec_2024_qc, tdsc_abus_2023
16 paired sweeps
1M Gaussian initialization
30k training iterations
final local-global renderer
responsibility DC
baseline = file pose
ours = canonical sweep pose with scale 1.00
```

Dataset-level PSNR gaps:

| Dataset | Pairs | Mean gap | Median gap | Min gap | Max gap | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic_liver | 6 | -0.0211 | -0.0009 | -0.0795 | +0.0137 | Strong retention. |
| tus_rec_2024_qc | 5 | +0.2719 | +0.2554 | +0.1549 | +0.4005 | Consistent positive retention. |
| tdsc_abus_2023 | 5 | +0.0003 | -0.0006 | -0.0082 | +0.0096 | Near-perfect tie. |
| all | 16 | +0.0771 | n/a | -0.0795 | +0.4005 | All 16 paired sweeps pass strong retention. |

Conclusion:

- Easy-pose is validated as a pose-dependency reduction module on the three non-spine datasets.
- The claim should be "per-frame 6-DoF pose can be replaced by a low-cost canonical sweep prior under regular/near-regular one-sweep protocols."
- It should not be written as a universal pose-free method for arbitrary unconstrained handheld scans.

### 3.3 Stage M Spacing Calibration Tolerance

Spacing stress protocol:

```math
d_\rho = \rho d,
\quad
\rho \in \{0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50\}.
```

Corresponding canonical pose:

```math
\tilde T_i(d_\rho) =
\begin{bmatrix}
R_0 & t_0 + (i-i_0)\rho d\,\mathbf n \\
0 & 1
\end{bmatrix}.
```

Best-PSNR gap against file-pose reference:

| Dataset | Sweep | 0.60 | 0.75 | 0.90 | 1.00 | 1.10 | 1.25 | 1.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_liver | l1 | +0.240 | +0.174 | +0.076 | +0.004 | -0.062 | -0.189 | -0.408 |
| synthetic_liver | r2 | +0.254 | +0.164 | +0.068 | -0.002 | -0.071 | -0.183 | -0.403 |
| tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | -0.065 | +0.062 | +0.116 | +0.163 | +0.176 | +0.171 | +0.103 |
| tus_rec_2024_qc | 011_LH_Per_S_PtD_qc_0008_0447 | +0.160 | +0.287 | +0.369 | +0.390 | +0.401 | +0.381 | +0.322 |
| tdsc_abus_2023 | validation_100 | -0.895 | -0.397 | -0.099 | -0.007 | +0.063 | +0.128 | +0.089 |
| tdsc_abus_2023 | validation_101 | -0.672 | -0.302 | -0.105 | -0.005 | +0.049 | +0.041 | -0.095 |

Pass-band summary:

| Dataset | Sweep | Strong retention | Acceptable retention |
| --- | --- | --- | --- |
| synthetic_liver | l1 | 0.60-1.25 | 0.60-1.50 |
| synthetic_liver | r2 | 0.60-1.25 | 0.60-1.50 |
| tus_rec_2024_qc | 010_LH_Per_L_DtP_qc_0005_0370 | 0.60-1.50 | 0.60-1.50 |
| tus_rec_2024_qc | 011_LH_Per_S_PtD_qc_0008_0447 | 0.60-1.50 | 0.60-1.50 |
| tdsc_abus_2023 | validation_100 | 0.90-1.50 | 0.75-1.50 |
| tdsc_abus_2023 | validation_101 | 0.90-1.50 | 0.75-1.50 |

Conclusion:

- The canonical sweep prior is not brittle to small spacing calibration errors.
- A conservative cross-dataset claim is robust operation around `rho in [0.90, 1.25]`.
- Cleaner regular sweeps can tolerate a wider range.
- `spine_phantom` should remain a limitation or negative-control case because its motion is highly non-uniform and not rescued by arc-length frame selection.

## 4. Recommended Paper Contribution Wording

Recommended contribution list:

1. We propose a local-global ultrasound Gaussian renderer that combines coordinate-aware local splatting with ray-CSR global propagation modulation.
2. We introduce responsibility-aware density control, which uses residual-weighted Gaussian contribution to guide densification and contribution/attenuation-aware pruning.
3. We develop an easy-pose canonical sweep prior for low-cost tracking-free one-sweep reconstruction, reducing per-frame `SE(3)` pose dependency to a calibrated sweep spacing under regular scanning protocols.

Recommended caution:

- Do not claim universal pose-free reconstruction.
- Do not present responsibility DC as the largest standalone accuracy gain.
- Do not describe local-global renderer as a simple UltraGray copy; emphasize the dual candidate structure and coordinate-transform local response.
- Report Gaussian counts for responsibility DC because it can increase retained capacity.
- Report runtime overhead for local-global renderer because it adds a ray branch.

## 5. Final Evidence Checklist

| Innovation | Main validation | Ablation / stress | Status |
| --- | --- | --- | --- |
| Local-global renderer | Four datasets, local-only vs local-global, 1M/30k | Formula simplification, ray-CSR speed, no-DC control from development reports | Validated |
| Responsibility DC | Four datasets, gradient vs responsibility, fixed DC schedule | Densify-only, no attenuation protection, no max-hit, mechanism visualizations | Validated |
| Easy-pose | 16 paired sweeps, file pose vs canonical sweep | Spacing tolerance `0.60-1.50`, spine negative-control analysis | Validated with regular-sweep scope |

Overall conclusion:

The combination of easy-pose, local-global renderer, and responsibility DC is sufficient for a journal-level method contribution. The three components address different layers of the problem: acquisition pose dependency, ultrasound rendering physics, and Gaussian topology optimization.
