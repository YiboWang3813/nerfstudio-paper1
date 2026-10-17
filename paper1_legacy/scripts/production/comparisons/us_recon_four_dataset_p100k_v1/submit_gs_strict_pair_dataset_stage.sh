#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
STAGE="${STAGE:-stage3_full_sweep}"
DATASET_GROUP="${DATASET_GROUP:-synthetic_liver}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
START_INDEX="${START_INDEX:?START_INDEX pair index is required}"
END_INDEX="${END_INDEX:?END_INDEX pair index is required}"
END_ORIGINAL_TASK_ID="${END_ORIGINAL_TASK_ID:?END_ORIGINAL_TASK_ID is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/run_gs_strict_pair_task.sh}"

case "${STAGE}" in
  stage1_calibration)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/gs_stage1_per_run.tsv}"
    ;;
  stage2_30pct)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/gs_stage2_per_run.tsv}"
    ;;
  stage2_fixed_subset)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/gs_stage2_fixed_subset_per_run.tsv}"
    ;;
  stage3_full_sweep|t01_gs_strict_nvs)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/gs_full_sweep_per_run.tsv}"
    ;;
  *)
    SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
    ;;
esac

mkdir -p "${OUTPUT_ROOT}/tables" "${PROJECT_DIR}/logs/comparison/${BENCHMARK}"

echo "[INFO] Submitting ${BENCHMARK} GS-strict pair stage"
echo "[INFO] STAGE=${STAGE}"
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
