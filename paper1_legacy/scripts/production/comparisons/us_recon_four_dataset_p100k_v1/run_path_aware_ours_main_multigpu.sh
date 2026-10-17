#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:3
#SBATCH -J us4paours
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%j.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%j.err
#SBATCH --time=7-00:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
STAGE="${STAGE:-stage3_full_sweep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/ablations/ours/path_aware_pruning_update_20260831/main_ours}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/ours_per_run.tsv}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-77}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/run_gs_strict_task.sh}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS=(0)
fi

gpu_count="${#GPU_IDS[@]}"
total_slots=$((gpu_count * WORKERS_PER_GPU))
if (( total_slots < 1 )); then
  echo "[ERROR] total_slots=${total_slots}; check CUDA_VISIBLE_DEVICES and WORKERS_PER_GPU" >&2
  exit 2
fi
if (( START_INDEX < 0 || END_INDEX > 77 || START_INDEX > END_INDEX )); then
  echo "[ERROR] task range ${START_INDEX}-${END_INDEX} must lie within 0-77" >&2
  exit 2
fi

case_for_index() {
  local global_index="$1"
  if (( global_index < 6 )); then
    DATASET_GROUP_FOR_TASK=synthetic_liver
    LOCAL_INDEX_FOR_TASK="${global_index}"
  elif (( global_index < 15 )); then
    DATASET_GROUP_FOR_TASK=spine_phantom
    LOCAL_INDEX_FOR_TASK=$((global_index - 6))
  elif (( global_index < 48 )); then
    DATASET_GROUP_FOR_TASK=tus_rec_2024_qc
    LOCAL_INDEX_FOR_TASK=$((global_index - 15))
  else
    DATASET_GROUP_FOR_TASK=tdsc_abus_2023
    LOCAL_INDEX_FOR_TASK=$((global_index - 48))
  fi
}

mkdir -p "${OUTPUT_ROOT}/tables" "${OUTPUT_ROOT}/worker_logs" "${PROJECT_DIR}/logs/comparison/${BENCHMARK}"

echo "============================================================"
echo "[INFO] Path-aware Ours main-comparison rerun"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] GPU_IDS=${GPU_IDS[*]}"
echo "[INFO] WORKERS_PER_GPU=${WORKERS_PER_GPU}"
echo "[INFO] total_slots=${total_slots}"
echo "[INFO] task_range=${START_INDEX}-${END_INDEX} (78 cases total)"
echo "[INFO] experiment=production/comparisons/${BENCHMARK}/gs_strict/ours"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] summary_path=${SUMMARY_PATH}"
echo "[INFO] checkpoints=keep final 30k for FPS benchmark"
echo "============================================================"

run_slot() {
  local slot_id="$1"
  local gpu_id="$2"
  local global_index dataset_group local_index task_log

  for ((global_index = START_INDEX + slot_id; global_index <= END_INDEX; global_index += total_slots)); do
    case_for_index "${global_index}"
    dataset_group="${DATASET_GROUP_FOR_TASK}"
    local_index="${LOCAL_INDEX_FOR_TASK}"
    task_log="${OUTPUT_ROOT}/worker_logs/task_$(printf '%03d' "${global_index}").log"

    echo "[WORKER ${slot_id}] start global_index=${global_index} local_index=${local_index} gpu=${gpu_id} dataset_group=${dataset_group} output_root=${OUTPUT_ROOT} summary=${SUMMARY_PATH}" | tee -a "${task_log}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      TASK_INDEX="${local_index}" \
      METHOD_FILTER=ours \
      DATASET_GROUP="${dataset_group}" \
      STAGE="${STAGE}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      SUMMARY_PATH="${SUMMARY_PATH}" \
      KEEP_CHECKPOINTS=1 \
      MAX_ACTIVE_US4GS=9999 \
      bash "${RUN_SCRIPT}" >> "${task_log}" 2>&1
    task_exit=$?
    set -e
    if (( task_exit != 0 )); then
      echo "[WORKER ${slot_id}] FAILED global_index=${global_index} exit=${task_exit} log=${task_log}" | tee -a "${task_log}" >&2
      return "${task_exit}"
    fi
    echo "[WORKER ${slot_id}] finish global_index=${global_index} gpu=${gpu_id} dataset_group=${dataset_group}" | tee -a "${task_log}"
  done
}

pids=()
slot=0
for gpu_id in "${GPU_IDS[@]}"; do
  for ((worker = 0; worker < WORKERS_PER_GPU; worker++)); do
    run_slot "${slot}" "${gpu_id}" &
    pids+=("$!")
    slot=$((slot + 1))
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "[ERROR] At least one worker failed; inspect ${OUTPUT_ROOT}/worker_logs" >&2
fi
exit "${failed}"
