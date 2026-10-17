# synthetic_liver Dataset Notes

Audience: experiment-running agents. This file records the current production
dataset facts and config entry points.

## Production Status

`synthetic_liver` is the mature comparison dataset. The local production export
now uses the shared p100k resolution policy, so experiments should load
preprocessed `uint8` arrays directly and should not apply online downsampling.

Recommended production default:

- Scene config: `scene=liver`
- DatasetMeta: `synthetic_liver`
- Working image shape: `448 x 224` (`100352` pixels/frame)
- Production comparison scripts: `scripts/production/comparisons/synthetic_liver_1m30k_dc/`
- Common budget: `1M` initialization, `30k` iterations, density control enabled
- Evaluation split: use `test_images.npy` / `test_poses.npy` with
  `test_mode=interpolated_file`

## Local Data

Processed root:

`/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver`

Raw/original backup for this refresh:

`/tmp/omniultra_raw_data_20260614_p100k/synthetic_liver`

Primary production sweeps:

`l1`, `l2`, `l3`, `r1`, `r2`, `r3`

Each primary sweep contains:

- `images.npy`: `(200, 448, 224)`, uint8
- `poses.npy`: `(200, 4, 4)`, float32
- `relative_poses.npy`: `(200, 4, 4)`, float32
- `test_images.npy`: `(40, 448, 224)`, uint8
- `test_poses.npy`: `(40, 4, 4)`, float32
- `test_relative_poses.npy` when available from the source export
- `metadata.json`: p100k source/target shape record

Do not treat helper folders from old raw exports as standard sweep directories.
Only the processed sweep folders above are retained under `/home`.

## DatasetMeta

Defined in:

`scene/meta/dataset_meta/synthetic_liver.py`

Key values:

| field | value |
| --- | --- |
| `name` | `synthetic_liver` |
| `probe_type` | `linear` |
| `H, W` | `448, 224` |
| `near_mm, far_mm` | `0.0, 140.0` |
| `width_mm` | `80.0` |
| `delta_lateral_mm` | `80.0 / 224.0` |
| `delta_axial_mm` | `140.0 / 448.0` |
| `delta_elevational_mm` | `0.2685` |
| `center_lateral` | `True` |
| `pose_init_type` | `probe->world` |

The physical field of view is unchanged from the Ultra-NeRF synthetic liver
setup. Only the raster sampling changed.

## Config Entry Points

Scene configs:

- `configs_hydra/scene/liver.yaml`
- `configs_hydra/scene/liver_x2.yaml`

`scene=liver` default loading:

- `dataset_name=synthetic_liver`
- `images_name=images.npy`
- `sweep_name=l1`
- `pose_type=relative`
- `test_mode=interpolated_file`
- `test_images_name=test_images.npy`
- `test_poses_name=test_poses.npy`
- `image_value_mode=uint8_255`
- `image_downsample_factor=1`
- `axial_downsample_factor=1`

`scene=liver_x2` is now only a compatibility alias. It also points to the p100k
`images.npy`/`test_images.npy` files; do not expect `images_x2.npy` to exist in
the current `/home` processed data tree.

## Agent Checklist

Before launching production:

- Verify the intended sweep is one of `l1/l2/l3/r1/r2/r3`.
- Keep `pose_type=relative` for standard comparison runs.
- Do not request `images_x2.npy`; current production data is already resized.
- Keep both online downsample factors at `1`.
- For broad comparison, use the established `synthetic_liver_1m30k_dc` scripts
  instead of ad hoc one-off Slurm commands.
