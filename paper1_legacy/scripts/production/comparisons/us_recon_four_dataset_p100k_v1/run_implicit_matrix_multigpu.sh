#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH -J us4impw
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%j.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
STAGE="${STAGE:-implicit_stage1_diagnostic}"
DATASET_GROUP="${DATASET_GROUP:-all}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
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

echo "============================================================"
echo "[INFO] ${BENCHMARK} implicit multi-GPU worker"
echo "[INFO] STAGE=${STAGE}"
echo "[INFO] DATASET_GROUP=${DATASET_GROUP}"
echo "[INFO] TASK_RANGE=${START_INDEX}-${END_INDEX}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] GPU_IDS=${GPU_IDS[*]}"
echo "[INFO] WORKERS_PER_GPU=${WORKERS_PER_GPU}"
echo "[INFO] total_slots=${total_slots}"
echo "[INFO] RUN_SCRIPT=${RUN_SCRIPT}"
echo "============================================================"

run_slot() {
  local slot_id="$1"
  local gpu_id="$2"
  local task_index

  for ((task_index = START_INDEX + slot_id; task_index <= END_INDEX; task_index += total_slots)); do
    echo "[WORKER ${slot_id}] start task_index=${task_index} gpu=${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      SLURM_ARRAY_TASK_ID="${task_index}" \
      MAX_ACTIVE_US4IMP=9999 \
      bash "${RUN_SCRIPT}"
    echo "[WORKER ${slot_id}] finish task_index=${task_index} gpu=${gpu_id}"
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

exit "${failed}"
