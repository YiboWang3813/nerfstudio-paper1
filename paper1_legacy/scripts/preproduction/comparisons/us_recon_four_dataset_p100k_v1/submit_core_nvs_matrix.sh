#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
TASK_RUNNER="${TASK_RUNNER:-scripts/preproduction/comparisons/${BENCHMARK}/run_comparison_task.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/main_nvs_per_run.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-159}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MAX_ACTIVE_US4CMP="${MAX_ACTIVE_US4CMP:-${MAX_PARALLEL}}"
WANDB_MODE="${WANDB_MODE:-offline}"
STAGE="${STAGE:-t01_core_nvs}"

mkdir -p "logs/comparison/${BENCHMARK}" "${OUTPUT_ROOT}/tables"

REQUESTED_COUNT=$((END_INDEX - START_INDEX + 1))
if [ "${REQUESTED_COUNT}" -lt 1 ]; then
  echo "[ERROR] Invalid array range: ${START_INDEX}-${END_INDEX}" >&2
  exit 2
fi

ACTIVE_US4CMP="$(squeue -h -u "${USER}" -n us4cmp 2>/dev/null | wc -l | tr -d ' ')"
AVAILABLE_US4CMP=$((MAX_ACTIVE_US4CMP - ACTIVE_US4CMP))
if [ "${AVAILABLE_US4CMP}" -lt "${REQUESTED_COUNT}" ]; then
  echo "[ERROR] Refusing to submit ${REQUESTED_COUNT} us4cmp task(s): active=${ACTIVE_US4CMP}, max=${MAX_ACTIVE_US4CMP}, available=${AVAILABLE_US4CMP}." >&2
  echo "[ERROR] Retry with a smaller START_INDEX/END_INDEX range after active jobs finish." >&2
  exit 75
fi

echo "[INFO] Submitting T01 core NVS for ${BENCHMARK}"
echo "[INFO] Array: ${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] Active us4cmp: ${ACTIVE_US4CMP}/${MAX_ACTIVE_US4CMP}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Summary: ${SUMMARY_PATH}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="${WANDB_MODE}",STAGE="${STAGE}",MAX_ACTIVE_US4CMP="${MAX_ACTIVE_US4CMP}" \
  "${TASK_RUNNER}"
