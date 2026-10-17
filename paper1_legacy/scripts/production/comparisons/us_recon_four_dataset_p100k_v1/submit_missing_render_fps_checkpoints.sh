#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
TASK_RUNNER="${TASK_RUNNER:-scripts/production/comparisons/${BENCHMARK}/run_missing_render_fps_checkpoint_task.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
STAGE="${STAGE:-paper_fps_checkpoint_regen_missing}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/render_fps_missing_checkpoint_regen_per_run.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-23}"
METHOD_FILTER_EXPORT="${METHOD_FILTER:-}"
METHOD_FILTER_EXPORT="${METHOD_FILTER_EXPORT//,/:}"

mkdir -p "${PROJECT_DIR}/logs/comparison/${BENCHMARK}" "${OUTPUT_ROOT}/tables"

echo "[INFO] Submitting missing render-FPS checkpoint regeneration"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Array: ${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}"
echo "[INFO] Methods: ${METHOD_FILTER:-ultrags_baseline,ultranerf,nerfus_no_prior}"
echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"

sbatch \
  --array="${START_INDEX}-${END_INDEX}%${MAX_PARALLEL}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",BENCHMARK="${BENCHMARK}",OUTPUT_ROOT="${OUTPUT_ROOT}",STAGE="${STAGE}",SUMMARY_PATH="${SUMMARY_PATH}",KEEP_CHECKPOINTS=1,METHOD_FILTER="${METHOD_FILTER_EXPORT}" \
  "${PROJECT_DIR}/${TASK_RUNNER}"
