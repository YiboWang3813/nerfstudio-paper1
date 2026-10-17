#!/bin/bash
#SBATCH -p gpu2
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J gpile_30k
#SBATCH -o logs/gaussianpile/%x_%j.out
#SBATCH -e logs/gaussianpile/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/gaussianpile/main_30k}"
EXP_TAG="${EXP_TAG:-gaussianpile_liver_1m_30000_dc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
OVERRIDES="${OVERRIDES:-}"
export CUDA_MODULE
export SLURM_SCRIPT="${BASH_SOURCE[0]}"

echo "[INFO] job=${SLURM_JOB_ID:-N/A} host=$(hostname) date=$(date)"
echo "[INFO] experiment=${EXPERIMENT} exp_tag=${EXP_TAG}"
echo "[INFO] output_root=${OUTPUT_ROOT:-<hydra-default>}"
echo "[INFO] overrides=${OVERRIDES:-<none>}"
set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/ops/gaussianplie_renderer:${PYTHONPATH:-}"
RUN_ARGS=(
    run.py
    "experiment=${EXPERIMENT}"
    "system.exp_tag=${EXP_TAG}"
)
if [[ -n "${OUTPUT_ROOT}" ]]; then
    RUN_ARGS+=("system.output_root=${OUTPUT_ROOT}")
fi
if [[ -n "${OVERRIDES}" ]]; then
    read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
    RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi
WANDB_MODE="${WANDB_MODE:-disabled}" python "${RUN_ARGS[@]}"
