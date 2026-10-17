# UltraGauss Slurm Scripts

These scripts reproduce the UltraGauss debugging and final liver baseline runs on the cluster. They assume the project runs from main:

```bash
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
mkdir -p logs/ultragauss
```

The default output root is:

```text
/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ultragauss
```

Submit in order when changing renderer or method code:

```bash
sbatch scripts/methods/baselines/ultragauss/slurm/01_renderer_reference.sh
sbatch scripts/methods/baselines/ultragauss/slurm/02_liver_smoke.sh
sbatch scripts/methods/baselines/ultragauss/slurm/03_liver_1m_10k_dc.sh
```

To rerun the precision-tril LR sweep used during debugging:

```bash
START_INDEX=0 END_INDEX=2 bash scripts/methods/baselines/ultragauss/slurm/04_precision_lr_sweep.sh
```

Useful environment overrides:

```bash
PROJECT_DIR=/path/to/omniultra
CONDA_ENV=gaussian_splatting
CUDA_MODULE=cuda/11.8.0
REBUILD=1
WANDB_MODE=offline
```

