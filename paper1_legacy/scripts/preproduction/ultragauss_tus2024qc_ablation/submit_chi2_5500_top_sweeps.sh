#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
TASK_RUNNER="${TASK_RUNNER:-scripts/preproduction/ultragauss_tus2024qc_ablation/run_ultragauss_tus2024qc_chi2_5500_top_sweeps_task.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ultragauss_tus2024qc_ablation}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/chi2_5500_top_sweeps.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-3}"

mkdir -p \
  "${PROJECT_DIR}/logs/preproduction/ultragauss_tus2024qc_ablation" \
  "${OUTPUT_ROOT}/tables"

echo "[INFO] Submitting UltraGauss TUS-REC 2024 QC chi2=5.5 top-sweep validation"
echo "[INFO] Array: ${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] Runner: ${TASK_RUNNER}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Summary: ${SUMMARY_PATH}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="${WANDB_MODE}" \
  "${PROJECT_DIR}/${TASK_RUNNER}"
