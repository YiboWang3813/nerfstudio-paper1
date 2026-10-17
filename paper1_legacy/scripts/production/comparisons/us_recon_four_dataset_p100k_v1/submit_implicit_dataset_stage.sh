#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
STAGE="${STAGE:-implicit_stage1_diagnostic}"
DATASET_GROUP="${DATASET_GROUP:-all}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_${DATASET_GROUP}_per_run.tsv}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/run_implicit_task.sh}"

case "${STAGE}:${DATASET_GROUP}" in
  implicit_stage1_diagnostic:all) DEFAULT_END_INDEX=7 ;;
  implicit_stage1_diagnostic:synthetic_liver) DEFAULT_END_INDEX=1 ;;
  implicit_stage1_diagnostic:spine_phantom) DEFAULT_END_INDEX=1 ;;
  implicit_stage1_diagnostic:tus_rec_2024_qc) DEFAULT_END_INDEX=1 ;;
  implicit_stage1_diagnostic:tdsc_abus_2023) DEFAULT_END_INDEX=1 ;;
  implicit_stage1_calibration:all) DEFAULT_END_INDEX=7 ;;
  implicit_stage1_calibration:synthetic_liver) DEFAULT_END_INDEX=1 ;;
  implicit_stage1_calibration:spine_phantom) DEFAULT_END_INDEX=1 ;;
  implicit_stage1_calibration:tus_rec_2024_qc) DEFAULT_END_INDEX=1 ;;
  implicit_stage1_calibration:tdsc_abus_2023) DEFAULT_END_INDEX=1 ;;
  implicit_stage2_30pct:all) DEFAULT_END_INDEX=47 ;;
  implicit_stage2_30pct:synthetic_liver) DEFAULT_END_INDEX=3 ;;
  implicit_stage2_30pct:spine_phantom) DEFAULT_END_INDEX=5 ;;
  implicit_stage2_30pct:tus_rec_2024_qc) DEFAULT_END_INDEX=19 ;;
  implicit_stage2_30pct:tdsc_abus_2023) DEFAULT_END_INDEX=17 ;;
  implicit_stage3_full_sweep:all) DEFAULT_END_INDEX=155 ;;
  implicit_stage3_full_sweep:synthetic_liver) DEFAULT_END_INDEX=11 ;;
  implicit_stage3_full_sweep:spine_phantom) DEFAULT_END_INDEX=17 ;;
  implicit_stage3_full_sweep:tus_rec_2024_qc) DEFAULT_END_INDEX=65 ;;
  implicit_stage3_full_sweep:tdsc_abus_2023) DEFAULT_END_INDEX=59 ;;
  *)
    echo "[ERROR] Unsupported STAGE=${STAGE} DATASET_GROUP=${DATASET_GROUP}" >&2
    exit 2
    ;;
esac

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-${DEFAULT_END_INDEX}}"
METHOD_FILTER_EXPORT="${METHOD_FILTER:-}"
METHOD_FILTER_EXPORT="${METHOD_FILTER_EXPORT//,/:}"

mkdir -p "${OUTPUT_ROOT}/tables" "${PROJECT_DIR}/logs/comparison/${BENCHMARK}"

echo "[INFO] Submitting ${BENCHMARK} implicit ${STAGE}"
echo "[INFO] DATASET_GROUP=${DATASET_GROUP}"
echo "[INFO] Array=${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] SUMMARY_PATH=${SUMMARY_PATH}"
echo "[INFO] METHOD_FILTER=${METHOD_FILTER:-<default>}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",STAGE="${STAGE}",DATASET_GROUP="${DATASET_GROUP}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",MAX_ACTIVE_US4IMP="${MAX_PARALLEL}",METHOD_FILTER="${METHOD_FILTER_EXPORT}" \
  "${RUN_SCRIPT}"
