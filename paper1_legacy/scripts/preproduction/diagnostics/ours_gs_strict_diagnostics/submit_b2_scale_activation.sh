#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-ours_gs_strict_diagnostics}"
STAGE="${STAGE:-b2_scale_activation}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/diagnostics/ours_gs_strict_diagnostics}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/${USER}/omniultra_checkpoints}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/preproduction/diagnostics/ours_gs_strict_diagnostics/run_b2_scale_activation_task.sh}"

mkdir -p "${OUTPUT_ROOT}/tables" "${OUTPUT_ROOT}/notes" "${PROJECT_DIR}/logs/ours_gs_strict_diagnostics" "${CHECKPOINT_ROOT}"

echo "[INFO] Submitting Ours GS-strict diagnostics B2 scale activation"
echo "[INFO] Matrix: datasets=(tus_rec_2024_qc,tdsc_abus_2023), variants=(exp,bounded)"
echo "[INFO] Array=0-3%${MAX_PARALLEL}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SUMMARY_PATH=${SUMMARY_PATH}"
echo "[INFO] CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "[INFO] RUN_SCRIPT=${RUN_SCRIPT}"

sbatch \
  --array="0-3%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",STAGE="${STAGE}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",MAX_ACTIVE_OGSD_B2="${MAX_PARALLEL}" \
  "${RUN_SCRIPT}"
