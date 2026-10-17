#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ultrags_baseline_smoke
#SBATCH -o logs/ultrags_baseline/%x_%j.out
#SBATCH -e logs/ultrags_baseline/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultrags_baseline/smoke}"
OVERRIDES="${OVERRIDES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/baseline_reproduction/ultrags_baseline/smoke}"
EXP_NAME="${EXP_NAME:-${SLURM_JOB_NAME:-run}_${SLURM_JOB_ID:-manual}}"

echo "============================================================"
echo "[INFO] Slurm job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overrides: ${OVERRIDES:-<none>}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"

python run.py \
  experiment="${EXPERIMENT}" \
  hydra.run.dir="${OUTPUT_ROOT}/${EXP_NAME}" \
  system.output_root="${OUTPUT_ROOT}" \
  system.exp_tag="${EXP_NAME}" \
  ${OVERRIDES}

echo "============================================================"
echo "[INFO] Slurm job finished successfully"
echo "============================================================"
