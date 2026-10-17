#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ultrags_base_build
#SBATCH -o logs/ultrags_baseline/%x_%j.out
#SBATCH -e logs/ultrags_baseline/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

echo "============================================================"
echo "[INFO] Rebuild job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

echo "[INFO] Python: $(which python)"
echo "[INFO] CUDA_HOME: ${CUDA_HOME:-N/A}"
which nvcc || true
nvcc --version || true
nvidia-smi || true

cd "${PROJECT_DIR}"
bash ops/ultrags_baseline_renderer/rebuild.sh
PYTHONPATH=. pytest -q -p no:cacheprovider tests/ops/ultrags_baseline_renderer

echo "============================================================"
echo "[INFO] Rebuild job finished successfully"
echo "============================================================"
