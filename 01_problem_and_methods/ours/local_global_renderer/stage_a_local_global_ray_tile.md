# Ours Local-Global Ray-Tile Stage A Report

Date: 2026-06-16

## Goal

Validate the proposed local-global ultrasound renderer split on TUS-REC-2024:

- local branch: existing tile-based slice Gaussian rasterization for local echo response;
- global branch: ray/tile attenuation modifier for long-range propagation;
- shared branch input: reuse the same world-to-probe projection and tile CSR where possible, then split only at candidate usage.

Stage A intentionally keeps the global branch simple and low-risk: no new learnable attenuation parameter, no ray-branch gradient to Gaussian attributes, and a bounded post-modulation of the local rendered image.

## Implementation

Branch: `dev/ours-local-global-ray-tile-20260616`

Main files:

- `ops/ours_renderer/src/shader/ray_attenuation.cu`
- `ops/ours_renderer/include/shader/ray_attenuation.cuh`
- `ops/ours_renderer/src/shader/wrapper_shader.cu`
- `ops/ours_renderer/include/shader/wrapper_shader.cuh`
- `ops/ours_renderer/ours_renderer/autograd.py`
- `ops/ours_renderer/ours_renderer/types.py`
- `methods/ours/render.py`
- `methods/ours/configs.py`
- `configs_hydra/experiment/preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_local_global_ray_tile_2k.yaml`

The integrated forward path calls the existing tile mapper once, runs the normal local renderer, then reuses the same tile CSR for a ray/tile attenuation pass. The final color is:

```text
color_out = color_local * (1 - beta + beta * T_ray)
```

where `T_ray = exp(-optical_depth)` is clamped by `ray_attenuation_min_transmission`. In Stage A, optical depth is derived from detached opacity and probe-frame Gaussian position/scale, so this is a diagnostic physical modifier rather than a fully learned attenuation field.

## Validation

Build and smoke checks:

- CUDA rebuild passed with `ops/ours_renderer/rebuild.sh`.
- Python binding smoke check confirmed `ProjectedSliceForwardRayTileCUDA`.
- Hydra dry run passed for `experiment=preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_local_global_ray_tile_2k`.
- 1-iter CUDA sanity run passed.
- 20-iter speed sanity passed for baseline and ray-tile configs.

Dataset/protocol:

- Dataset: `tus_rec_2024_qc`
- Sweep: `010_LH_Per_L_DtP_qc_0005_0370`
- Split: existing holdout interval protocol, train 325 / test 40
- Iterations: 2000
- Init Gaussians: 300000
- GPU: `gpu3` / RTX 4090

## Results

Baseline:

- Config: `experiment=preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_validation_2k`
- Output: `outputs/preproduction/ideas/ours/local_global_renderer/ours_local_global_ray_tile_20260616/baseline_2k`
- Job: `63695`

Stage A variants:

- `beta=0.05, strength=0.03, min_T=0.90`
  - Output: `outputs/preproduction/ideas/ours/local_global_renderer/ours_local_global_ray_tile_20260616/ray_beta005_strength003_2k`
  - Job: `63698`
- `beta=0.10, strength=0.05, min_T=0.85`
  - Output: `outputs/preproduction/ideas/ours/local_global_renderer/ours_local_global_ray_tile_20260616/ray_beta01_2k`
  - Job: `63696`
- `beta=0.20, strength=0.08, min_T=0.80`
  - Output: `outputs/preproduction/ideas/ours/local_global_renderer/ours_local_global_ray_tile_20260616/ray_beta020_strength008_2k`
  - Job: `63700`

| Variant | Test PSNR | Test L1 | Test SSIM | Test LPIPS | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 27.8516 | 0.0241 | 0.8101 | 0.2650 | 267410 |
| Ray A weak | 27.8203 | 0.0242 | 0.8095 | 0.2657 | 267417 |
| Ray A default | 27.7618 | 0.0244 | 0.8092 | 0.2655 | 267700 |
| Ray A strong | 27.6786 | 0.0247 | 0.8083 | 0.2671 | 268051 |

Speed:

- Baseline 20-iter sanity: about `13.28 it/s` including cold start and 2k image cost.
- Integrated ray-tile 20-iter sanity: about `11.27 it/s`.
- Full 2k ray-tile runs: final tqdm around `33 it/s`, with late-stage steady progress around `29-30 it/s` after evaluation stalls.
- The global branch is measurably slower than baseline because the current Stage A kernel scans upstream tiles per axial/lateral tile.

## Interpretation

Stage A did not improve reconstruction quality. The effect is monotonic enough to be useful: stronger global attenuation consistently hurts holdout PSNR/SSIM/LPIPS, and weaker attenuation merely approaches the baseline without exceeding it.

The likely failure mode is not the local-global split itself. It is the Stage A parameterization:

1. The global branch derives attenuation from the same opacity already optimized for local image synthesis, so it reuses a quantity with the wrong responsibility.
2. The attenuation is detached and multiplicative, so it can only darken/modulate the local render; it cannot learn a compensated physical decomposition.
3. The current ray/tile optical depth is coarse and tile-level, which is useful as an engineering probe but too blunt to become a quality-improving physical term.
4. The added compute is not justified by the current metrics.

## Decision

Do not promote the current Stage A formula to `main` as a production method.

Keep the dev-branch CUDA skeleton because it validates the desired clean structure:

- coordinate/tile mapping is shared;
- local response and global propagation are separate code paths;
- the integrated forward avoids a second tile mapper call;
- configuration can disable the branch and defaults remain baseline-compatible.

## Next Step

If this direction continues, Stage B should not simply make Stage A stronger. It should replace opacity-derived detached attenuation with a physically separated and regularized global field, preferably with:

- a dedicated bounded attenuation parameter or low-rank per-ray/per-depth profile;
- prefix/ray-bin accumulation instead of upstream tile scanning;
- explicit regularization so global propagation cannot duplicate local intensity;
- visualization of `T_ray` maps to verify that the learned/global field corresponds to plausible depth attenuation rather than arbitrary darkening.

Until Stage B exists, this idea should be treated as a clean negative result for "detached opacity-derived global attenuation".
