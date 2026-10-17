# spine_phantom Dataset Notes

Audience: experiment-running agents. This file records the current production
dataset facts and config entry points.

## Production Status

`spine_phantom` is production-ready for the current Ours pipeline with the
shared p100k processed export.

Recommended production default:

- Scene config: `scene=spine_phantom`
- Main experiment config: `experiment=preproduction/dataset_preflight/spine_phantom/spine_phantom_preflight_10k`
- Working image shape: `896 x 112` (`100352` pixels/frame)
- Online image downsample: `1`
- Online axial downsample: `1`
- Initialization: `1M` Gaussians for production runs
- Iterations: train to `10k` or `15k`, evaluate periodically, report best
  checkpoint by the target metric

The earlier exploration used online axial downsampling from `1300x164` to
`650x164`. That raw/full-res path is no longer the production data layout under
`/home`.

## Local Data

Processed root:

`/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom`

Raw/original backup for this refresh:

`/tmp/omniultra_raw_data_20260614_p100k/spine_phantom`

Available sweep directories:

`left1`, `left1_1`, `left2`, `left3`, `left3_2`, `right1`, `right1_1`,
`right3`, `right3_2`

Typical files:

- `images.npy`: `(N, 896, 112)`, uint8
- `poses.npy`: `(N, 4, 4)`, float32
- `relative_poses.npy`: `(N, 4, 4)`, float32
- `metadata.json`: p100k source/target shape record

Example frame counts:

- `left1`: `152`
- `left1_1`: `153`
- `left2`: `151`

The standard scene loader uses absolute poses for this dataset.

## DatasetMeta

Defined in:

`scene/meta/dataset_meta/spine_phantom.py`

Key values:

| field | value |
| --- | --- |
| `name` | `spine_phantom` |
| `probe_type` | `linear` |
| `H, W` | `896, 112` |
| `near_mm, far_mm` | `0.0, 120.0` |
| `width_mm` | `40.0` |
| `delta_lateral_mm` | `40.0 / 112.0` |
| `delta_axial_mm` | `120.0 / 896.0` |
| `delta_elevational_mm` | `0.79` |
| `center_lateral` | `True` |
| `pose_init_type` | `probe->world` |

Important caveat: the Ultra-NeRF paper and available third-party code do not
fully expose the exact physical image-plane calibration for this dataset. The
current values are operational defaults:

- keep the retained physical field of view,
- use a plausible `40 mm` lateral span,
- use observed pose spacing for the elevational default.

## Image Handling

The processed p100k export has already converted the Ultra-NeRF shifted float
images into standard `uint8` intensity images. Runtime configs should use:

- `image_value_mode=uint8_255`
- `image_nan_value=0.0`
- `image_downsample_factor=1`
- `axial_downsample_factor=1`

Do not re-enable `spine_phantom_shifted_float` for the processed production
tree; it is only needed when reading the raw backup export.

## Config Entry Points

Scene config:

`configs_hydra/scene/spine_phantom.yaml`

Default scene values:

- `dataset_name=spine_phantom`
- `sweep_name=left1`
- `pose_type=absolute`
- `test_mode=holdout_interval`
- `holdout_trim_each=8`
- `holdout_num_samples=30`
- `image_value_mode=uint8_255`
- `image_downsample_factor=1`
- `axial_downsample_factor=1`
- default sampler `num_init=300000`

Experiment configs:

- `configs_hydra/experiment/preproduction/dataset_preflight/spine_phantom/spine_phantom_minimal.yaml`
- `configs_hydra/experiment/preproduction/dataset_preflight/spine_phantom/spine_phantom_preflight_10k.yaml`

## Agent Checklist

Before launching production:

- Confirm the target sweep exists under the local `spine_phantom` processed root.
- Keep `pose_type=absolute`.
- Keep `image_value_mode=uint8_255`.
- Keep both online downsample factors at `1`.
- Use `1M` initialization for production comparisons.
- If using a 30k budget for consistency, still report best checkpoint, not just
  final checkpoint.
