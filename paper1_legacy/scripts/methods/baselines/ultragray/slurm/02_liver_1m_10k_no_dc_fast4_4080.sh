#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ugr_1m10k_4080
#SBATCH -o logs/ultragray/%x_%j.out
#SBATCH -e logs/ultragray/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultragray/main_10k_fast}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/preproduction/baseline_reproduction/ultragray}"
EXP_TAG="${EXP_TAG:-ultragray_liver_1m_10k_no_dc_fast4_4080}"
OVERRIDES="${OVERRIDES:-}"
REBUILD="${REBUILD:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
ULTRAGRAY_PROFILE_EVERY="${ULTRAGRAY_PROFILE_EVERY:-500}"
ULTRAGRAY_PROFILE_WARMUP="${ULTRAGRAY_PROFILE_WARMUP:-100}"
export PYTHONDONTWRITEBYTECODE=1
export ULTRAGRAY_CUDA_SM89="${ULTRAGRAY_CUDA_SM89:-1}"
export ULTRAGRAY_PROFILE_EVERY ULTRAGRAY_PROFILE_WARMUP

echo "============================================================"
echo "[INFO] UltraGray liver 1M/10k no-DC 4080 job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp tag: ${EXP_TAG}"
echo "[INFO] Overrides: ${OVERRIDES}"
echo "[INFO] Rebuild: ${REBUILD}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs/ultragray "${OUTPUT_ROOT}"

if [[ "${REBUILD}" == "1" ]]; then
  echo "[INFO] Rebuilding ultragray_renderer"
  bash ops/ultragray_renderer/rebuild.sh
fi

RUN_ARGS=(
  run.py
  "experiment=${EXPERIMENT}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_TAG}"
)

if [[ -n "${OVERRIDES}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi

echo "[INFO] Running: python ${RUN_ARGS[*]}"
WANDB_MODE="${WANDB_MODE}" python "${RUN_ARGS[@]}"

echo "============================================================"
echo "[INFO] UltraGray liver 1M/10k no-DC 4080 job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
