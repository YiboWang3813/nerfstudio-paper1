#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
TASK_RUNNER="${TASK_RUNNER:-scripts/production/comparisons/${BENCHMARK}/run_ultragauss_tus_qc_chi2_5500_replacement_task.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/gs_ultragauss_tus_qc_chi2_5500_replacement.tsv}"
CHI2="${CHI2:-5.500}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
  "${PROJECT_DIR}/logs/comparison/${BENCHMARK}" \
  "${OUTPUT_ROOT}/tables"

echo "[INFO] Submitting production UltraGauss TUS-QC chi2=5.5 replacement"
echo "[INFO] Slurm jobs: 1"
echo "[INFO] GPUs: 2"
echo "[INFO] Workers per GPU: ${WORKERS_PER_GPU}"
echo "[INFO] Runner: ${TASK_RUNNER}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] CHI2=${CHI2}"

sbatch \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",CHI2="${CHI2}",WORKERS_PER_GPU="${WORKERS_PER_GPU}",WANDB_MODE="${WANDB_MODE}" \
  "${PROJECT_DIR}/${TASK_RUNNER}"
