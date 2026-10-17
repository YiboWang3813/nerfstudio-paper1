#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ugr_tests_4080
#SBATCH -o logs/ultragray/%x_%j.out
#SBATCH -e logs/ultragray/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
export PYTHONDONTWRITEBYTECODE=1
export ULTRAGRAY_CUDA_SM89="${ULTRAGRAY_CUDA_SM89:-1}"

echo "============================================================"
echo "[INFO] UltraGray renderer test job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] CUDA module: ${CUDA_MODULE}"
echo "[INFO] ULTRAGRAY_CUDA_SM89: ${ULTRAGRAY_CUDA_SM89}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs/ultragray

echo "[INFO] Rebuilding ultragray_renderer"
bash ops/ultragray_renderer/rebuild.sh

echo "[INFO] Running UltraGray tests"
PYTHONPATH=. pytest -q -p no:cacheprovider tests/methods/ultragray tests/ops/ultragray_renderer

echo "============================================================"
echo "[INFO] UltraGray renderer test job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
