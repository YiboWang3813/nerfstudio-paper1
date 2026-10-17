#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
STAGE="${STAGE:-stage2_fixed_subset}"
DATASET_GROUP="${DATASET_GROUP:-synthetic_liver}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-2}"
END_ORIGINAL_TASK_ID="${END_ORIGINAL_TASK_ID:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}_pair_smoke}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/gs_pair_smoke_per_run.tsv}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/run_gs_strict_pair_task.sh}"

mkdir -p "${OUTPUT_ROOT}/tables" "${PROJECT_DIR}/logs/comparison/${BENCHMARK}"

echo "[INFO] Submitting ${BENCHMARK} GS-strict pair smoke"
echo "[INFO] DATASET_GROUP=${DATASET_GROUP}"
echo "[INFO] Pair array=${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] TASKS_PER_GPU=${TASKS_PER_GPU}"
echo "[INFO] Original task max=${END_ORIGINAL_TASK_ID}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SUMMARY_PATH=${SUMMARY_PATH}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",STAGE="${STAGE}",DATASET_GROUP="${DATASET_GROUP}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",TASKS_PER_GPU="${TASKS_PER_GPU}",END_ORIGINAL_TASK_ID="${END_ORIGINAL_TASK_ID}" \
  "${RUN_SCRIPT}"
