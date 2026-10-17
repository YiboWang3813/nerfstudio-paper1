#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/dc_tuning_synthetic_liver_l1_ours_dc.tsv}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/preproduction/comparisons/${BENCHMARK}/run_ours_dc_tuning_task.sh}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-7}"

mkdir -p "${OUTPUT_ROOT}/tables" "${PROJECT_DIR}/logs/comparison/${BENCHMARK}"

echo "[INFO] Submitting ours DC tuning: synthetic_liver/l1"
echo "[INFO] Array=${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] SUMMARY_PATH=${SUMMARY_PATH}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",MAX_ACTIVE_US4DC="${MAX_PARALLEL}" \
  "${RUN_SCRIPT}"
