# Hydra Experiment Configs

`configs_hydra/` is the Hydra config system. New runs should use lifecycle-aware
experiment aliases instead of long YAML paths:

```bash
python run.py experiment=preproduction/ideas/ours/pose_uncertainty_prior/pose_prior
python run.py experiment=preproduction/ideas/ours/renderer_baseline/main_10k
python run.py experiment=preproduction/baseline_reproduction/ultragray/main_10k_fast
python run.py experiment=production/comparisons/us_synthetic_recon_benchmark_v1/ours
```

Hydra creates the run directory automatically:

```text
outputs/YYYY-MM-DD/HH-MM-SS_<family>-<name>_<model>_<runid>/
  .hydra/
  .omniultra/
  config_dump.yaml
  resolved_config.yaml
  overrides.yaml
  run_manifest.yaml
  checkpoints/
  logs/
  vis/
```

Use `system.output_root` to choose the root without hand-writing a unique
folder name:

```bash
python run.py experiment=preproduction/baseline_reproduction/ultragray/smoke system.output_root=/tmp/omniultra_runs
```

Use `system.checkpoint_root` to keep the experiment folder under `outputs/`
while storing large checkpoint files on another filesystem. The run directory
still contains a `checkpoints` symlink, so existing scripts that read
`<run>/checkpoints/...` continue to work:

```bash
python run.py \
  experiment=preproduction/baseline_reproduction/ultragray/smoke \
  system.checkpoint_root=/tmp/$USER/omniultra_checkpoints
```

For a specific path, override `system.checkpoint_dir` directly:

```bash
python run.py \
  experiment=preproduction/baseline_reproduction/ultragray/smoke \
  system.checkpoint_dir=/tmp/$USER/omniultra_checkpoints/manual_run/checkpoints
```

Move existing checkpoint folders the same way with the maintenance script:

```bash
python scripts/maintenance/externalize_checkpoints.py \
  --outputs-root outputs \
  --checkpoint-root /tmp/$USER/omniultra_checkpoints
python scripts/maintenance/externalize_checkpoints.py \
  --outputs-root outputs \
  --checkpoint-root /tmp/$USER/omniultra_checkpoints \
  --apply
```

Use `task=dry_run` to validate the composed config and write run metadata
without starting training:

```bash
python run.py experiment=preproduction/baseline_reproduction/ultragray/smoke task=dry_run engine.iterations=1
```

## Layout

- `experiment/preproduction/baseline_reproduction/<method>/<name>.yaml`: baseline reproduction, diagnostics, and method preflights.
- `experiment/preproduction/ideas/ours/<idea>/<name>.yaml`: ours idea exploration and preproduction validation.
- `experiment/preproduction/dataset_preflight/<dataset>/<name>.yaml`: dataset-specific preflight aliases.
- `experiment/production/comparisons/<benchmark>/<method>.yaml`: production-grade fixed comparison aliases.
- `experiment/production/ablations/<method_or_idea>/<name>.yaml`: production-grade fixed ablation aliases when needed.
- `model/<method>.yaml`: method defaults.
- `model/ours_us_focus.yaml`: production model defaults for the ours us_focus renderer.
- `scene/*.yaml`: dataset and sampler defaults.
- `engine/*.yaml`: runner/callback defaults.
- `loss/default.yaml`, `metric/default.yaml`, `system/default.yaml`: shared defaults.

Experiment filenames should describe intent, not every parameter. Put specific
values in the config body or command-line overrides; each run stores the fully
resolved config and exact overrides in its output directory.

Production aliases should be thin wrappers over validated preproduction aliases
when possible. This keeps the final benchmark entry point explicit without
duplicating the underlying method configuration.

Preproduction scripts may still call a production alias when the purpose is to
probe or tune the exact final comparison entry point with command-line
overrides. In that case, keep the script under `scripts/preproduction/...` and
the output under `outputs/preproduction/...`; do not move the production alias
unless the final benchmark entry point itself is being retired.
