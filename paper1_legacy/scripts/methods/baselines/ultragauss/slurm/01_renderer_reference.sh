#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ug_ref
#SBATCH -o logs/ultragauss/%x_%j.out
#SBATCH -e logs/ultragauss/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
REBUILD="${REBUILD:-0}"

echo "============================================================"
echo "[INFO] UltraGauss renderer reference job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Rebuild: ${REBUILD}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"

if [[ "${REBUILD}" == "1" ]]; then
  echo "[INFO] Rebuilding ultragauss_renderer"
  bash ops/ultragauss_renderer/rebuild.sh
fi

python -m pytest -q -p no:cacheprovider \
  tests/methods/ultragauss/test_reference_math.py \
  tests/methods/ultragauss/test_solver_scheduler.py \
  ops/ultragauss_renderer/tests/test_forward_reference.py

echo "============================================================"
echo "[INFO] UltraGauss renderer reference job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"

