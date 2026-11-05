# UltraGray Reproduction Report

Legacy note: this report preserves historical reproduction evidence and may mention old `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/ultragray/` and should be launched with Hydra `experiment=...` aliases.

Date: 2026-05-27

## Scope

This report records the current UltraGray reproduction state after implementing and testing the method code under `methods/ultragray` and the CUDA renderer under `ops/ultragray_renderer`.

The main validated run is the 1M Gaussian, 10k iteration, no-density-control liver experiment:

- Config: `config_files/experiments/ultragray/liver_1m_10k_no_dc_fast4.yaml`
- Output root: `checkpoints/ultragray`
- Experiment tag: `ultragray_liver_1m_10k_no_dc_fast4_4080`
- Hardware: RTX 4080 through Slurm `gpu1`
- CUDA module: `cuda/11.8.0`
- Renderer build flag for Ada GPUs: `ULTRAGRAY_CUDA_SM89=1`

## Final Metrics

| Run | Iterations | Gaussians | Density control | PSNR | L1 | SSIM | LPIPS | Average speed |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `ultragray_liver_1m_10k_no_dc_fast4_4080` | 10000 | 1000000 | off | 26.7818 | 0.0324 | 0.5343 | 0.2711 | 46.27 it/s |
| `ultragray_liver_1m_10k_no_dc_fast` | 10000 | 1000000 | off | 26.8039 | 0.0323 | 0.5344 | 0.2677 | 43.34 it/s |
| `ultragray_liver_1m_10k_no_dc_geomfix` | 10000 | 1000000 | off | 26.5020 | 0.0337 | 0.5309 | 0.3788 | not gated |

The 4080 run completed the requested 1M/10k/no-DC reproduction with stable geometry and finite gradients. The quality is in the same range as the earlier A100 no-DC fast run, which is a useful cross-device sanity check.

## Geometry And Numerical Findings

- Pixel positions are now passed to the CUDA renderer in `(W, H, 3)` order, matching the operator's indexing convention. This fixed the earlier geometry mismatch where ray/sample positions were transposed relative to image coordinates.
- Ray origins and directions are explicitly generated and passed into the renderer. Ray perturbation now jitters origins rather than relying on implicit axial assumptions.
- The 1M Gaussian configs initialize scales at `[0.5, 0.5, 0.5]` mm and keep density control disabled for the main no-DC reproduction.
- The renderer uses bounded quadrature and local transmission windows to keep hit processing physically local in world-coordinate millimeters.

## Renderer Changes

- Mapper returns sorted hit depths in addition to CSR hit ranges.
- Shader forward/backward uses sorted depths for local transmission range lookup.
- Local transmission mode avoids integrating all ray hits when only a physical neighborhood is needed.
- Gaussian covariance/SigmaInv computation is skipped for transmission-only hits.
- CUDA kernels use restricted pointers where appropriate.
- Runtime profiling is controlled through `ULTRAGRAY_PROFILE_EVERY` and `ULTRAGRAY_PROFILE_WARMUP`.
- `ops/ultragray_renderer/setup.py` supports Ada cards by setting `ULTRAGRAY_CUDA_SM89=1` during build.

## Speed Profile

Representative 4080 profile samples from `ultragray_liver_1m_10k_no_dc_fast4_4080`:

| Step | Mapper GPU ms | Forward GPU ms | Backward GPU ms | Total hits | Mean hits/ray |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.259 | 1.577 | 7.016 | 837547 | 3271.7 |
| 1000 | not recorded | 1.763 | 7.888 | 908692 | 3549.6 |
| 2500 | not recorded | 2.257 | 10.476 | 1099691 | 4295.7 |
| 5000 | not recorded | 2.970 | 13.771 | 1304598 | 5096.1 |
| 9000 | not recorded | 3.902 | 17.681 | 1597990 | 6242.1 |
| 10000 | 0.336 | 3.322 | 15.571 | 1411090 | 5512.1 |

Early iterations can exceed 80 to 100 it/s, but the full 10k run settles closer to 40 to 45 it/s. The dominant cost is not just the designed physics model; it is the growing number of ray/Gaussian hit interactions that survive mapping and transmission windows. The current implementation reduces unnecessary work substantially, but a stable 100 it/s for the full 1M/10k/no-DC run would likely need a more aggressive spatial/ray culling strategy, chunk sizing changes, or a lower-cost transmission approximation.

## Cluster Notes

- RTX 4080 nodes are usable for the current reproduction path.
- RTX 4090 node discovery showed free GPUs, but Slurm could not allocate a second job because the node reports `RealMemory=1` and the existing allocation consumes the memory TRES.
- A100 was used only for earlier debugging; the final requested reproduction was completed on 4080.

## Checkpoint Locations

UltraGray debug runs were moved from `/home/boywang25/ultrasound-scene-reconstruction/outputs/debug_runs` to:

```text
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ultragray
```

All UltraGray experiment YAMLs now set `system.output_dir` to that root. The Slurm scripts also default `OUTPUT_ROOT` to the same path.
