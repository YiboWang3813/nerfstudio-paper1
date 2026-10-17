#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ultragray_acc_speed_balance}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/ultragray_acc_speed_balance.tsv}"
RUNNER="${RUNNER:-${PROJECT_DIR}/scripts/methods/baselines/ultragray/slurm/05_acc_speed_balance_task.sh}"
REBUILD_FIRST="${REBUILD_FIRST:-0}"
WINDOWS=(1 2)

mkdir -p "${PROJECT_DIR}/logs/ultragray" "${OUTPUT_ROOT}/tables"

submit_one() {
  local dataset="$1"
  local scene="$2"
  local sweep="$3"
  local window="$4"
  local rebuild="$5"
  local exp_name="ultragray_${dataset}_${sweep}_lw${window}mm"
  local job_id
  job_id=$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",DATASET="${dataset}",SCENE="${scene}",SWEEP="${sweep}",LOCAL_WINDOW_MM="${window}",EXP_NAME="${exp_name}",REBUILD="${rebuild}",WANDB_MODE="offline" \
    "${RUNNER}")
  echo "[INFO] submitted job=${job_id} dataset=${dataset} sweep=${sweep} window=${window}mm rebuild=${rebuild} output=${OUTPUT_ROOT}/runs/${dataset}/${sweep}/lw${window}mm"
}

first=1
for window in "${WINDOWS[@]}"; do
  rebuild=0
  if [[ "${first}" == "1" && "${REBUILD_FIRST}" == "1" ]]; then
    rebuild=1
  fi
  submit_one tus_rec_2024_qc tus_rec_2024_qc 010_LH_Per_L_DtP_qc_0005_0370 "${window}" "${rebuild}"
  first=0
  submit_one tdsc_abus_2023 tdsc_abus_2023 validation_100 "${window}" 0
done

echo "[INFO] Summary path: ${SUMMARY_PATH}"
