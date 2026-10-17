#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J gpile_fwd
#SBATCH -o logs/gaussianpile/%x_%j.out
#SBATCH -e logs/gaussianpile/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

echo "[INFO] job=${SLURM_JOB_ID:-N/A} host=$(hostname) date=$(date)"
set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/ops/gaussianplie_renderer:${PYTHONPATH:-}"
pytest -q -p no:cacheprovider tests/ops/gaussianplie_renderer

