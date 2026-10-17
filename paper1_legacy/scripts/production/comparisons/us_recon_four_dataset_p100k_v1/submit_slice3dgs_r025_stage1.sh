#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
STAGE="${STAGE:-slice3dgs_r025_stage1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}/external_checkpoints}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/run_slice3dgs_r025_task.sh}"

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-1}"

mkdir -p "${OUTPUT_ROOT}/tables" "${PROJECT_DIR}/logs/comparison/${BENCHMARK}" "${CHECKPOINT_ROOT}"

echo "[INFO] Submitting ${BENCHMARK} Stage-1 slice-adapted-3DGS r025"
echo "[INFO] Matrix: datasets=(tus_rec_2024_qc,tdsc_abus_2023), method=slice_adapted_3dgs, ratio=0.25"
echo "[INFO] Array=${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SUMMARY_PATH=${SUMMARY_PATH}"
echo "[INFO] CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "[INFO] RUN_SCRIPT=${RUN_SCRIPT}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",STAGE="${STAGE}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  "${RUN_SCRIPT}"
