#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-ours_gs_strict_diagnostics}"
STAGE="${STAGE:-a1_corridor_sweep}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/diagnostics/ours_gs_strict_diagnostics}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/${USER}/omniultra_checkpoints}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/preproduction/diagnostics/ours_gs_strict_diagnostics/run_a1_corridor_sweep_task.sh}"

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-11}"

mkdir -p "${OUTPUT_ROOT}/tables" "${OUTPUT_ROOT}/notes" "${PROJECT_DIR}/logs/ours_gs_strict_diagnostics" "${CHECKPOINT_ROOT}"

echo "[INFO] Submitting Ours GS-strict diagnostics A1 corridor sweep"
echo "[INFO] Matrix: datasets=(tus_rec_2024_qc,tdsc_abus_2023), method=ours, ratios=(0.50,0.75,1.00,1.25,1.50,2.00)"
echo "[INFO] Array=${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SUMMARY_PATH=${SUMMARY_PATH}"
echo "[INFO] CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "[INFO] RUN_SCRIPT=${RUN_SCRIPT}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",STAGE="${STAGE}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",MAX_ACTIVE_OGSD_A1="${MAX_PARALLEL}" \
  "${RUN_SCRIPT}"
