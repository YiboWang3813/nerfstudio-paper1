#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4gs2
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/run_gs_strict_task.sh}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"

pair_task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
first_task=$((pair_task_id * TASKS_PER_GPU))

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo "============================================================"
echo "[INFO] ${BENCHMARK} GS-strict pair task"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Pair task: ${pair_task_id}"
echo "[INFO] Original task range: ${first_task}-$((first_task + TASKS_PER_GPU - 1))"
echo "[INFO] Tasks per GPU: ${TASKS_PER_GPU}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "============================================================"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] RUN_SCRIPT is not executable: ${RUN_SCRIPT}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "logs/comparison/${BENCHMARK}"

pids=()
tasks=()
for ((offset = 0; offset < TASKS_PER_GPU; offset++)); do
  task_id=$((first_task + offset))
  if [[ -n "${END_ORIGINAL_TASK_ID:-}" && "${task_id}" -gt "${END_ORIGINAL_TASK_ID}" ]]; then
    continue
  fi

  tasks+=("${task_id}")
  (
    export SLURM_ARRAY_TASK_ID="${task_id}"
    export MAX_ACTIVE_US4GS=999999
    echo "[PAIR] Launching original task ${task_id} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"
    bash "${RUN_SCRIPT}"
  ) &
  pids+=("$!")
done

if (( ${#pids[@]} == 0 )); then
  echo "[ERROR] No child tasks were launched." >&2
  exit 2
fi

status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  task="${tasks[$i]}"
  if wait "${pid}"; then
    echo "[PAIR] Original task ${task} finished successfully."
  else
    child_status=$?
    echo "[PAIR] Original task ${task} failed with exit code ${child_status}." >&2
    status="${child_status}"
  fi
done

echo "============================================================"
echo "[INFO] Pair task finished with status ${status}"
echo "[INFO] Date: $(date)"
echo "============================================================"
exit "${status}"
