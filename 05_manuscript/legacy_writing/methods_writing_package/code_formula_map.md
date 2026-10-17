# Code To Formula Map

Use this file when checking Methods statements against implementation.

## Core Python Path

| Paper concept | Snapshot path | Original path | Notes |
| --- | --- | --- | --- |
| Render config builder | `source_snapshot/methods/ours/render.py` | `methods/ours/render.py` | Converts dataset metadata and model config to `OursRenderConfig`. |
| Gaussian parameterization | `source_snapshot/methods/ours/gaussian_container.py` | `methods/ours/gaussian_container.py` | Defines `echo_alphas`, `attenuations`, fixed unit intensities. |
| Responsibility DC | `source_snapshot/methods/ours/density_control.py` | `methods/ours/density_control.py` | Clone/split/prune and responsibility score. |
| DC diagnostics | `source_snapshot/methods/ours/dc_diagnostics.py` | `methods/ours/dc_diagnostics.py` | Contribution/residual stats wrapper. |
| Offline loop | `source_snapshot/methods/ours/gs_only_callbacks.py` | `methods/ours/gs_only_callbacks.py` | Training/test/save callbacks and DC timing calls. |
| Easy-Pose | `source_snapshot/scene/container/single_sweep_container.py` | `scene/container/single_sweep_container.py` | `canonical_sweep` pose construction. |

## Renderer Extension Path

| Paper concept | Snapshot path | Original path | Notes |
| --- | --- | --- | --- |
| Public renderer API | `source_snapshot/ops/ours_renderer/ours_renderer/api.py` | `ops/ours_renderer/ours_renderer/api.py` | `OursRenderer.forward(...)`. |
| Autograd bridge | `source_snapshot/ops/ours_renderer/ours_renderer/autograd.py` | `ops/ours_renderer/ours_renderer/autograd.py` | Packs meta/us-focus/ray params and calls CUDA. |
| Renderer config type | `source_snapshot/ops/ours_renderer/ours_renderer/types.py` | `ops/ours_renderer/ours_renderer/types.py` | Validates tile/additive/us_focus/ray attenuation options. |
| Local forward shader | `source_snapshot/ops/ours_renderer/src/shader/forward.cu` | `ops/ours_renderer/src/shader/forward.cu` | Tile/pixel local branch forward. |
| Local backward shader | `source_snapshot/ops/ours_renderer/src/shader/backward.cu` | `ops/ours_renderer/src/shader/backward.cu` | Gradients for local branch. |
| US focus code | `source_snapshot/ops/ours_renderer/include/shader/us_focus_device.cuh` | `ops/ours_renderer/include/shader/us_focus_device.cuh` | Focus-aware local response math. |
| Ray attenuation | `source_snapshot/ops/ours_renderer/src/shader/ray_attenuation.cu` | `ops/ours_renderer/src/shader/ray_attenuation.cu` | Ray-CSR candidate mapping and attenuation branch. |
| Slice grid geometry | `source_snapshot/ops/ours_renderer/include/shader/slice_grid_geometry.cuh` | `ops/ours_renderer/include/shader/slice_grid_geometry.cuh` | Pixel-to-mm mapping. |

## Production Configs

| Purpose | Snapshot path | Original path |
| --- | --- | --- |
| Base model defaults | `source_snapshot/configs_hydra/model/ours_us_focus.yaml` | `configs_hydra/model/ours_us_focus.yaml` |
| Main comparison Ours row | `source_snapshot/configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml` | `configs_hydra/experiment/production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours.yaml` |
| Ours ablation base | `source_snapshot/configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml` | `configs_hydra/experiment/production/ablations/ours/base_1m30k.yaml` |

## Formula Anchors

### Local Branch

Use:

```text
L_ip = sum_g alpha_g eta_ig nu_ig phi_igp
```

Code anchors:

```text
include/shader/us_focus_device.cuh
src/shader/forward.cu
ours_renderer/autograd.py
```

### Ray Branch

Use:

```text
Ihat_ip = L_ip * (1 - beta + beta * T_ip)
```

Production config:

```text
formula = integral
quadrature_points = 3
beta = 0.80
strength = 0.55
min_transmission = 0.35
candidate_k_sigma = 2.0
```

Code anchors:

```text
src/shader/ray_attenuation.cu
ours_renderer/autograd.py
ours_renderer/types.py
```

### Responsibility DC

Use:

```text
C_g = sum_p c_gp
R_g = sum_p c_gp e_p
rbar_g = R_g / (C_g + eps)
score_g = rbar_g * log(1 + C_g)
```

Code anchors:

```text
methods/ours/density_control.py
methods/ours/dc_diagnostics.py
ours_renderer/diagnostics.py
src/shader/forward.cu
```

### Easy-Pose

Use:

```text
Ttilde_i = [I_3, (i - i0) d u; 0, 1]
```

Code anchor:

```text
scene/container/single_sweep_container.py
```

## Evidence Docs In Snapshot

| Evidence | Snapshot path |
| --- | --- |
| Multi-method comparison lifecycle | `source_snapshot/docs/production/multi_comparisons.md` |
| Paper-facing quantitative tables | `source_snapshot/docs/production/comparison_final_outputs/quantitative_tabs.md` |
| Ours ablation lifecycle | `source_snapshot/docs/production/ours_ablations.md` |
| Comparison 2D figure config | `source_snapshot/docs/production/comparison_final_outputs/2d_rendering_figure_config.md` |
| Ablation figure config | `source_snapshot/docs/production/ablation_final_outputs/2d_ablation_figure_config.md` |
| 3D visualization decision | `source_snapshot/docs/production/comparison_final_outputs/3d_visualization_ablation_notes.md` |

