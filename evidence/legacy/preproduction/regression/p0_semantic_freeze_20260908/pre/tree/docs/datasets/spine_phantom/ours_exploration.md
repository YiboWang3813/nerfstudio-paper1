# Spine Phantom Ours Exploration

Date: 2026-06-12
Branch: `main`
Stage: exploratory
GPU: RTX 4090 node `gpu3`

## Goal

Validate that the Ultra-NeRF `spine_phantom` dataset can be loaded through the production `ours` / `us_focus` path, including image-pose matching, holdout evaluation, online axial downsample, and the provisional `DatasetMeta`.

This is not a production comparison. The run is only a smoke-scale check that the dataset wiring and training path are viable.

## Dataset And Config

- Dataset root: `/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom`
- Sweep: `left1`
- Source image shape: `1300 x 164`
- Runtime image shape: `650 x 164`
- Online image conversion: `spine_phantom_shifted_float`
- Online axial downsample: factor `2`
- Train/test protocol: holdout interval with `holdout_trim_each=8`, `holdout_num_samples=30`
- Runtime split: `train=122`, `test=30`
- Alias: `experiment=preproduction/dataset_preflight/spine_phantom/spine_phantom_minimal`

The image-plane physical extents are currently operational/provisional:

- `near_mm=0.0`
- `far_mm=120.0`
- `width_mm=40.0`

These values are inherited by analogy from the available Ultra-NeRF synthetic-liver implementation and should not be described as paper-reported facts.

`delta_elevational_mm=0.79` is data-derived from adjacent pose translations, following `scripts/datasets/synthetic_liver/check_elevational_delta.py`. Across all available spine phantom sweeps, the adjacent-pose Euclidean step has mean `0.791 mm`, median `0.715 mm`, and 10%-trimmed mean `0.745 mm`. The per-sweep Euclidean means are `0.76-0.81 mm`, so `0.79 mm` is a reasonable inter-slice step default. This value should be described as pose-derived inter-slice spacing, not as a measured ultrasound beam thickness.

## Slurm Run

- Job: `62488`
- Runner: `scripts/methods/ours/easy_pose/slurm/run_spine_phantom_explore.sh`
- Output root: `outputs/preproduction/ideas/ours/easy_pose/spine_phantom_ours_explore`
- Run dir: `outputs/preproduction/ideas/ours/easy_pose/spine_phantom_ours_explore/spine_left1_axial2_minimal_62488`
- Summary: `outputs/preproduction/ideas/ours/easy_pose/spine_phantom_ours_explore/summary.tsv`
- Log: `logs/ours/ours_spine_exp_62488.out`

Key settings:

- Iterations: `2000`
- Initial Gaussians: `300000`
- Losses: `l1`, `ssim`, `scale_wall`
- Density control: enabled from `500` to `1500`, frequency `500`
- Metrics: `psnr`, `l1`, `ssim`, `lpips`

## Initial Result

Initial `left1` axial-2x test metrics at iter `2000`:

| PSNR | L1 | SSIM | LPIPS | Gaussians |
|---:|---:|---:|---:|---:|
| 30.944268 | 0.014121 | 0.773959 | 0.210557 | 211235 |

The run completed successfully. Setup verified:

- `spine_phantom/left1/images.npy` and `poses.npy` loaded.
- Axial downsample ran online: `H=1300 -> 650`.
- Runtime meta resolution was overridden: `(1300,164) -> (650,164)`.
- Holdout split was created from the same sweep.
- Renderer, loss pipeline, metrics, density control, checkpoint save, and scalar summary all completed.

## Interpretation

The dataset integration is viable enough for the next exploratory steps. The short run is also a practical signal that online axial-only 2x downsample does not add noticeable overhead relative to the renderer/training path.

The main unresolved risk is not software viability but image-plane physical calibration. `far_mm` and `width_mm` remain provisional, so any paper-facing result should either justify these numbers from source code/data provenance or label them as implementation assumptions. The inter-slice spacing is better supported because it can be estimated directly from poses.

## Follow-Up Checks

Two additional exploratory jobs were run to separate dataset viability from the axial-downsample choice.

| Job | Sweep | Runtime shape | Override | PSNR | L1 | SSIM | LPIPS | Gaussians |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 62488 | left1 | `650 x 164` | none | 30.944268 | 0.014121 | 0.773959 | 0.210557 | 211235 |
| 62490 | right1 | `650 x 164` | `scene.loading.sweep_name=right1` | 30.613616 | 0.015131 | 0.765624 | 0.203682 | 214436 |
| 62491 | left1 | `1300 x 164` | `scene.loading.axial_downsample_factor=1` | 29.882395 | 0.014990 | 0.757546 | 0.320475 | 204024 |

`right1` confirms that the loader, pose convention, holdout split, renderer, and metrics are not only working for `left1`. It loaded `146` frames and split into `116` train / `30` test frames.

The full-resolution `left1` check also completed successfully. It is only moderately slower in this short run (`~100 it/s` versus `~118 it/s` for axial-2x), so full-res is not blocked by compute at this scale. However, at the same 2000 iterations and 300k initialization, full-res is visibly harder: PSNR and SSIM drop, LPIPS worsens substantially, and the compose image shows a darker prediction side than the axial-2x run. This should not be overread as "full-res is worse" in an absolute sense, because the metric target is a harder high-frequency image. It does show that axial-2x is the more stable exploratory setting for fast iteration.

Current recommendation:

1. Keep `axial_downsample_factor=2` as the default exploratory spine_phantom setting.
2. Keep full-res as an explicit control, not the default, until a longer/full training schedule is tested.
3. If the paper needs to claim native-resolution reconstruction, run full-res with a tuned schedule instead of comparing it directly to the 2k axial-2x smoke result.

## Suggested Next Steps

1. Inspect more final render visualizations from all three runs.
2. Run one longer axial-2x exploratory job if the visuals are acceptable.
3. Run a longer full-res control only if native-resolution reporting becomes a concrete goal.
