#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ultranerf_smoke
#SBATCH -o logs/ultranerf/%x_%j.out
#SBATCH -e logs/ultranerf/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultranerf/smoke}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] WANDB_MODE: ${WANDB_MODE}"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_DIR}"
python run.py "experiment=${EXPERIMENT}" task=train

echo "[INFO] Finished at $(date)"
