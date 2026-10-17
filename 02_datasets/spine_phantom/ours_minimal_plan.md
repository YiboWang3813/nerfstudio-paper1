# Spine Phantom Ours Minimal Plan

## Goal

Validate that the production `ours_us_focus` path can train and evaluate on the Ultra-NeRF `spine_phantom` data without starting a broad sweep.

## Dataset Setup

- Scene config: `scene=spine_phantom`
- Default sweep: `left1`
- Disk files: `data/spine_phantom/<sweep>/images.npy` and `poses.npy`
- Pose mode: `absolute`
- Eval mode: `holdout_interval`, because the spine phantom folders do not provide separate `test_images.npy` or `test_poses.npy`
- Holdout: trim 8 frames at each end, sample 30 test frames from the remaining interval
- Image values: processed `uint8_255`
- Resolution: p100k processed export, so runtime image shape is expected to be `(896, 112)`

## Metadata Assumptions

- Probe: linear
- Processed image shape: `896 x 112`
- Operational physical extent: `far_mm=120`, `width_mm=40`
- The p100k resize keeps the retained physical FOV and updates in-plane pixel spacing in DatasetMeta
- `delta_elevational_mm=0.79`, estimated from observed adjacent pose spacing across spine sweeps

These are operational defaults because the Ultra-NeRF paper and available third-party code do not expose exact spine phantom image-plane calibration.

## Minimal Experiment Config

Use:

```bash
python run.py experiment=preproduction/dataset_preflight/spine_phantom/spine_phantom_minimal
```

Do not launch this yet. First compose/dry-check the config, then inspect the first loaded scene batch if needed.

Key choices:

- Model: `model=ours_us_focus`
- Iterations: `2000`
- Initial Gaussians: `300000`
- Density control: enabled from `500` to `1500`, every `500`
- Metrics: `psnr`, `l1`, `ssim`, `lpips`
- Save once at final iteration

## First Execution Checklist

1. Compose config only and confirm `scene.dataset_name=spine_phantom`.
2. Instantiate `SceneFactory.build_scene` on CPU or a small GPU allocation and verify:
   - train frames match the selected sweep count
   - runtime meta is `H=896`, `W=112`
   - no NaNs remain in `slice_meta.image`
   - holdout train/test split is non-empty
3. Run the 2k minimal experiment on `left1`.
4. Inspect rendered/GT composites and scalar metrics.
5. If stable, repeat the same config with `scene.loading.sweep_name` overridden for one right-side sweep, for example `right1`.

## Success Criteria

- Loader completes without NaN image values.
- Runtime resolution and metadata spacing are consistent.
- Training reaches 2k iterations without renderer or density-control failures.
- Evaluation writes metrics and visual outputs for the holdout frames.
