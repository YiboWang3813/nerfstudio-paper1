#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
RUNNER="${RUNNER:-scripts/production/comparisons/us_synthetic_recon_benchmark_v1/run_comparison_6method.sh}"
TASK_RUNNER="${TASK_RUNNER:-scripts/production/comparisons/us_synthetic_recon_benchmark_v1/run_task_6method.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs/production/comparisons/us_synthetic_recon_benchmark_v1}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/per_run_metrics.tsv}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-41}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

mkdir -p logs/comparison "${OUTPUT_ROOT}/tables"

SBATCH_EXTRA_ARRAY=()
if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
  read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_EXTRA_ARGS}"
fi

dep=""
wave_jobs=()

for task_id in $(seq "${START_INDEX}" "${END_INDEX}"); do
  sbatch_args=("${SBATCH_EXTRA_ARRAY[@]}")
  if [[ -n "${dep}" ]]; then
    sbatch_args+=("--dependency=afterany:${dep}")
  fi

  echo "[SUBMIT] task_id=${task_id} dep=${dep:-none}"
  submit_out=$(
    sbatch \
      "${sbatch_args[@]}" \
      --parsable \
      --array="${task_id}-${task_id}" \
      --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUNNER="${RUNNER}",OUTPUT_ROOT="${OUTPUT_ROOT}",SUMMARY_PATH="${SUMMARY_PATH}",WANDB_MODE="${WANDB_MODE}" \
      "${TASK_RUNNER}"
  )
  echo "${submit_out}"
  job_id="${submit_out%%;*}"
  wave_jobs+=("${job_id}")

  if (( ${#wave_jobs[@]} >= MAX_PARALLEL )); then
    dep=$(IFS=:; echo "${wave_jobs[*]}")
    wave_jobs=()
  fi
done
