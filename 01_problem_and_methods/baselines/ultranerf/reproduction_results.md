# UltraNeRF Reproduction Report

Legacy note: this report preserves historical reproduction evidence and may mention old `config_files/...` paths. Current runnable entries live under `configs_hydra/experiment/preproduction/baseline_reproduction/ultranerf/` and should be launched with Hydra `experiment=...` aliases.

Date: 2026-05-27

## Goal

Reproduce Ultra-NeRF on `synthetic_liver` with full-resolution `512 x 256` images, 200 source frames, `holdout_interval` split with 160 train frames and 40 test frames, and 30000 training iterations.

Acceptance target:

- `PSNR > 20`
- `SSIM ~= 0.5`

## Implementation

Implemented a PyTorch Ultra-NeRF path under `methods/ultranerf`:

- isotropic MLP field with NeRF positional encoding and 5 physical outputs.
- differentiable expectation-mode ultrasound renderer.
- linear-probe ray/grid extraction from existing `SliceMeta`.
- patch-based training with full-image test rendering.
- Engine callbacks, state, solver, checkpointing, metrics, debug maps, and manifest writing.

Integration was authorized for this task and completed in:

- `configs/base/models.py`
- `run.py`

## Experiment

Config:

```text
config_files/experiments/ultranerf/liver_fullres_30k_holdout.yaml
```

Slurm:

```text
job id: 59829
script: scripts/methods/baselines/ultranerf/slurm/run_ultranerf_fullres_30k.sh
stdout: logs/ultranerf/ultranerf_30k_59829.out
stderr: logs/ultranerf/ultranerf_30k_59829.err
```

Output:

```text
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout/
```

Key files:

```text
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout/checkpoints/iteration_30000_field.pth
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout/checkpoints/iteration_30000_optimizer.pth
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout/logs/scalars.tsv
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout/run_manifest.yaml
checkpoints/ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout/vis/item_1/compose.png
```

## Metrics

Final test metrics at iteration 30000:

| metric | value |
|---|---:|
| PSNR | 23.9653 |
| SSIM | 0.5544 |
| L1 | 0.03865 |
| LPIPS | 0.64794 |

Progress checkpoints:

| iteration | PSNR | SSIM | L1 | LPIPS |
|---:|---:|---:|---:|---:|
| 2500 | 21.5644 | 0.5123 | 0.0549 | 0.7357 |
| 10000 | 21.9882 | 0.5346 | 0.0449 | 0.7004 |
| 20000 | 22.8129 | 0.5483 | 0.0411 | 0.6641 |
| 25000 | 23.9074 | 0.5524 | 0.0382 | 0.6566 |
| 30000 | 23.9653 | 0.5544 | 0.0387 | 0.6479 |

## Validation

Local checks:

```bash
pytest -q -p no:cacheprovider tests/methods/ultranerf
python -m py_compile methods/ultranerf/args.py methods/ultranerf/configs.py methods/ultranerf/field.py methods/ultranerf/ray_sampler.py methods/ultranerf/render.py methods/ultranerf/solver.py methods/ultranerf/state.py methods/ultranerf/offline_callbacks.py methods/ultranerf/offline_runner.py methods/ultranerf/nerf_model.py run.py configs/base/models.py
git diff --check
```

Smoke Slurm run:

```text
job id: 59827
final at 500 iter: PSNR 19.6057, SSIM 0.5007
```

## Conclusion

The reproduction meets the requested threshold on `synthetic_liver`: final PSNR is above 20 and SSIM is around 0.5 under the specified 30000-iteration, 160/40 holdout protocol.

Remaining risk:

- This first version uses deterministic expectation rendering rather than Bernoulli sampling. It is stable and differentiable, but not a fully stochastic match to the paper renderer.
- Training uses random `128 x 128` full-resolution patches for efficiency, with full-image evaluation on the 40 holdout frames.
