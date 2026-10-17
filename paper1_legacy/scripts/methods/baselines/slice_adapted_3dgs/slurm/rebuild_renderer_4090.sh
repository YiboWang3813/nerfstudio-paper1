#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J sa3dgs_build
#SBATCH -o logs/slice_adapted_3dgs/%x_%j.out
#SBATCH -e logs/slice_adapted_3dgs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"

echo "============================================================"
echo "[INFO] Slice-adapted 3DGS renderer build"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"

echo "[INFO] Python: $(which python)"
echo "[INFO] CUDA_HOME: ${CUDA_HOME:-N/A}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

bash ops/slice_adapted_3dgs_renderer/rebuild.sh
pytest -q -p no:cacheprovider tests/ops/slice_adapted_3dgs_renderer tests/methods/slice_adapted_3dgs

echo "============================================================"
echo "[INFO] Slice-adapted 3DGS renderer build finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

