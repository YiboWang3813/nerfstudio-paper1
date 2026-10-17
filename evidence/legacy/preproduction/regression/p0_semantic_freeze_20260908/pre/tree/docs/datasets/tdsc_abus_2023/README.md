# tdsc_abus_2023 Dataset Notes

Audience: experiment-running agents. This file records the current production
dataset facts and config entry points.

## Production Status

`tdsc_abus_2023` is newly adapted. The local production-format p100k export
contains all 30 retained validation volumes, but it should still be treated as
the least validated of the four datasets because the source is a 3D ABUS volume
dataset without tracked freehand poses.

Recommended first production checks:

- Scene config: `scene=tdsc_abus_2023`
- Working image shape: `280 x 356` (`99680` pixels/frame)
- Online image downsample: `1`
- Online axial downsample: `1`
- Pose model: generated parallel image-plane poses using documented slice
  spacing
- Start with short Ours smoke/preflight runs before broad comparison matrices

## Local Data

Processed root:

`/home/boywang25/ultrasound-scene-reconstruction/data/tdsc_abus_2023`

Raw/original and intermediate backups for this refresh:

- Raw NRRD validation split: `/tmp/tdsc_abus_2023_probe`
- Full-resolution standard sweep export:
  `/tmp/tdsc_abus_2023_standard_fullres_all/tdsc_abus_2023`
- Earlier single-sweep backup:
  `/tmp/omniultra_raw_data_20260614_p100k/tdsc_abus_2023`

Current processed sweeps:

- `validation_100` through `validation_129`
- `30` sweeps
- `10264` total frames

Typical files:

- `images.npy`: `(N, 280, 356)`, uint8
- `poses.npy`: `(N, 4, 4)`, float32
- `relative_poses.npy`: `(N, 4, 4)`, float32
- `metadata.json`: p100k source/target shape record

Frame counts vary by volume. Examples:

- `validation_100`: `353`
- `validation_103`: `330`
- `validation_123`: `354`
- `validation_129`: `348`

Some raw validation volumes have source shape `865 x 608 x 330` instead of
`865 x 682 x Z`. These are padded along the axial/Y dimension to the canonical
`865 x 682` image plane before p100k resizing, so all processed sweeps share one
DatasetMeta and one physical field of view. Padded cases:

`validation_103`, `validation_104`, `validation_106`, `validation_108`,
`validation_109`, `validation_111`, `validation_112`, `validation_114`,
`validation_115`, `validation_124`, `validation_127`, `validation_128`

Masks are not part of the current production dataloader path. Rebuild with
`scripts/datasets/shared/build_p100k_datasets.py --tdsc-include-masks` only if a future
experiment explicitly needs segmentation labels.

## Export Scripts

Raw NRRD to standard sweep export:

`scripts/datasets/tdsc_abus_2023/prepare.py`

p100k standardization:

`scripts/datasets/shared/build_p100k_datasets.py`

The raw export slices each volume along the documented slice axis and writes one
project sweep per volume. Because the dataset has no tracked probe poses, the
export creates parallel image-plane poses separated by the ABUS slice spacing.

Current 30-sweep preparation commands:

```bash
python scripts/datasets/tdsc_abus_2023/prepare.py \
  --raw-root /tmp/tdsc_abus_2023_probe \
  --out-root /tmp/tdsc_abus_2023_standard_fullres_all/tdsc_abus_2023 \
  --split Validation \
  --allow-variable-shape \
  --pad-to-target-shape

python scripts/datasets/shared/build_p100k_datasets.py \
  --raw-root /tmp/tdsc_abus_2023_standard_fullres_all \
  --out-root /home/boywang25/ultrasound-scene-reconstruction/data \
  --datasets tdsc_abus_2023 \
  --batch-size 32
```

## DatasetMeta

Defined in:

`scene/meta/dataset_meta/tdsc_abus_2023.py`

Key values:

| field | value |
| --- | --- |
| `name` | `tdsc_abus_2023` |
| `probe_type` | `linear` |
| `H, W` | `280, 356` |
| `near_mm, far_mm` | `0.0, 682 * 0.073` |
| `width_mm` | `865 * 0.200` |
| `delta_lateral_mm` | `(865 * 0.200) / 356` |
| `delta_axial_mm` | `(682 * 0.073) / 280` |
| `delta_elevational_mm` | `0.475674` |
| `center_lateral` | `True` |
| `pose_init_type` | `probe->world` |

The physical field of view comes from the source ABUS voxel spacing. The p100k
export changes raster sampling only; it does not change the retained physical
extent.

## Config Entry Points

Scene config:

`configs_hydra/scene/tdsc_abus_2023.yaml`

Default scene values:

- `dataset_name=tdsc_abus_2023`
- `dataset_dirname=tdsc_abus_2023`
- `sweep_name=validation_100`
- `pose_type=absolute`
- `test_mode=holdout_interval`
- `holdout_trim_each=10`
- `holdout_num_samples=40`
- `image_value_mode=uint8_255`
- `image_downsample_factor=1`
- `axial_downsample_factor=1`
- default sampler `num_init=300000`

## Agent Checklist

Before launching production:

- Confirm the target sweep exists under the processed root.
- Remember that poses are generated parallel slice poses, not tracked freehand
  probe poses.
- Keep both online downsample factors at `1`.
- Do not mix segmentation masks into reconstruction metrics unless the
  experiment explicitly declares that protocol.
