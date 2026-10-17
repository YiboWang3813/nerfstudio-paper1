#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
TASK_RUNNER="${TASK_RUNNER:-scripts/preproduction/ultragauss_tus2024qc_ablation/run_ultragauss_tus2024qc_chi2_task.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ultragauss_tus2024qc_ablation}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/chi2_one_sweep.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
MAX_ACTIVE_UG_CHI2="${MAX_ACTIVE_UG_CHI2:-${MAX_PARALLEL}}"
WANDB_MODE="${WANDB_MODE:-offline}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-3}"

mkdir -p \
  "${PROJECT_DIR}/logs/preproduction/ultragauss_tus2024qc_ablation" \
  "${OUTPUT_ROOT}/tables"

requested_count=$((END_INDEX - START_INDEX + 1))
if [ "${requested_count}" -lt 1 ]; then
  echo "[ERROR] Invalid array range: ${START_INDEX}-${END_INDEX}" >&2
  exit 2
fi

set +e
active_jobs="$(squeue -h -u "${USER}" -n ug_chi2 2>/dev/null | wc -l | tr -d ' ')"
active_status=$?
set -e
if [ "${active_status}" -ne 0 ] || [ -z "${active_jobs}" ]; then
  echo "[WARN] Could not query active ug_chi2 jobs; relying on Slurm array %${MAX_PARALLEL} for throttling."
  active_jobs=0
fi

available_jobs=$((MAX_ACTIVE_UG_CHI2 - active_jobs))
if [ "${available_jobs}" -lt "${requested_count}" ]; then
  echo "[ERROR] Refusing to submit ${requested_count} ug_chi2 task(s): active=${active_jobs}, max=${MAX_ACTIVE_UG_CHI2}, available=${available_jobs}." >&2
  exit 75
fi

echo "[INFO] Submitting UltraGauss TUS-REC 2024 QC one-sweep chi2 ablation"
echo "[INFO] Array: ${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] Active ug_chi2: ${active_jobs}/${MAX_ACTIVE_UG_CHI2}"
echo "[INFO] Runner: ${TASK_RUNNER}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Summary: ${SUMMARY_PATH}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="${WANDB_MODE}",MAX_ACTIVE_UG_CHI2="${MAX_ACTIVE_UG_CHI2}" \
  "${PROJECT_DIR}/${TASK_RUNNER}"
