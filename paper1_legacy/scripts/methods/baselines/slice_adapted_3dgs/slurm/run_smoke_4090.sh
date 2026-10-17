#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J sa3dgs_smoke
#SBATCH -o logs/slice_adapted_3dgs/%x_%j.out
#SBATCH -e logs/slice_adapted_3dgs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/slice_adapted_3dgs/smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/baseline_reproduction/slice_adapted_3dgs/smoke_4090}"
EXP_NAME="${EXP_NAME:-${SLURM_JOB_NAME:-run}_${SLURM_JOB_ID:-manual}}"
OVERRIDES="${OVERRIDES:-}"
RUN_TESTS="${RUN_TESTS:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

echo "============================================================"
echo "[INFO] Slice-adapted 3DGS smoke job"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overrides: ${OVERRIDES:-<none>}"
echo "[INFO] Run tests: ${RUN_TESTS}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
export WANDB_MODE

echo "[INFO] Python: $(which python)"
echo "[INFO] CUDA_HOME: ${CUDA_HOME:-N/A}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "[INFO] Running method tests"
  pytest -q -p no:cacheprovider tests/methods/slice_adapted_3dgs
fi

RUN_ARGS=(
  run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${OUTPUT_ROOT}/${EXP_NAME}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_NAME}"
  "system.device=cuda"
)

if [[ -n "${OVERRIDES}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi

echo "[INFO] Running: python ${RUN_ARGS[*]}"
python "${RUN_ARGS[@]}"

echo "============================================================"
echo "[INFO] Slice-adapted 3DGS smoke job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
