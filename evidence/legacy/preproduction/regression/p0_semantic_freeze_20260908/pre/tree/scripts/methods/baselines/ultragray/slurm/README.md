# UltraGray Slurm Scripts

These scripts reproduce the current UltraGray renderer tests and the final liver 1M/10k no-density-control run from main.

```bash
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
mkdir -p logs/ultragray checkpoints/ultragray
```

Default output root:

```text
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ultragray
```

Submit in order after changing UltraGray renderer or method code:

```bash
sbatch scripts/methods/baselines/ultragray/slurm/01_renderer_tests_4080.sh
sbatch scripts/methods/baselines/ultragray/slurm/02_liver_1m_10k_no_dc_fast4_4080.sh
```

Useful environment overrides:

```bash
PROJECT_DIR=/path/to/omniultra
CONDA_ENV=gaussian_splatting
CUDA_MODULE=cuda/11.8.0
REBUILD=1
WANDB_MODE=offline
OUTPUT_ROOT=/path/to/checkpoints/ultragray
EXP_TAG=custom_ultragray_run
ULTRAGRAY_PROFILE_EVERY=500
ULTRAGRAY_PROFILE_WARMUP=100
```

For RTX 4080/4090 builds, keep `ULTRAGRAY_CUDA_SM89=1`; PyTorch in this environment does not accept `TORCH_CUDA_ARCH_LIST=8.9` directly.
