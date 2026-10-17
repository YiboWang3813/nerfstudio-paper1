#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
TASK_RUNNER="${TASK_RUNNER:-scripts/production/comparisons/${BENCHMARK}/run_gs_strict_task.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/main_nvs_per_run.tsv}"
DATASET_GROUP="${DATASET_GROUP:-synthetic_liver}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MAX_ACTIVE_US4GS="${MAX_ACTIVE_US4GS:-${MAX_PARALLEL}}"
WANDB_MODE="${WANDB_MODE:-offline}"
STAGE="${STAGE:-t01_gs_strict_nvs}"

case "${DATASET_GROUP}" in
  synthetic_liver) DEFAULT_END_INDEX=35 ;;
  spine_phantom) DEFAULT_END_INDEX=23 ;;
  tus_rec_2024_qc) DEFAULT_END_INDEX=29 ;;
  tdsc_abus_2023) DEFAULT_END_INDEX=29 ;;
  all) DEFAULT_END_INDEX=119 ;;
  *)
    echo "[ERROR] Unknown DATASET_GROUP=${DATASET_GROUP}" >&2
    exit 2
    ;;
esac

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-${DEFAULT_END_INDEX}}"

mkdir -p "logs/comparison/${BENCHMARK}" "${OUTPUT_ROOT}/tables"

REQUESTED_COUNT=$((END_INDEX - START_INDEX + 1))
if [ "${REQUESTED_COUNT}" -lt 1 ]; then
  echo "[ERROR] Invalid array range: ${START_INDEX}-${END_INDEX}" >&2
  exit 2
fi

ACTIVE_US4GS="$(squeue -h -u "${USER}" -n us4gs 2>/dev/null | wc -l | tr -d ' ')"
AVAILABLE_US4GS=$((MAX_ACTIVE_US4GS - ACTIVE_US4GS))
if [ "${AVAILABLE_US4GS}" -lt "${REQUESTED_COUNT}" ]; then
  echo "[ERROR] Refusing to submit ${REQUESTED_COUNT} us4gs task(s): active=${ACTIVE_US4GS}, max=${MAX_ACTIVE_US4GS}, available=${AVAILABLE_US4GS}." >&2
  echo "[ERROR] Retry with a smaller START_INDEX/END_INDEX range after active jobs finish." >&2
  exit 75
fi

echo "[INFO] Submitting GS-strict T01 NVS for ${BENCHMARK}"
echo "[INFO] Dataset group: ${DATASET_GROUP}"
echo "[INFO] Array: ${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] Active us4gs: ${ACTIVE_US4GS}/${MAX_ACTIVE_US4GS}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Summary: ${SUMMARY_PATH}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="${WANDB_MODE}",STAGE="${STAGE}",DATASET_GROUP="${DATASET_GROUP}",MAX_ACTIVE_US4GS="${MAX_ACTIVE_US4GS}" \
  "${TASK_RUNNER}"
