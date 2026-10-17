#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ss5gsw
#SBATCH -o logs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/synth_spine_5gs_full/%x_%A_%a.out
#SBATCH -e logs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/synth_spine_5gs_full/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-gs_strict_1m30k_dc_t1000_25000_f2000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/stage3_full_sweep_synthetic_liver_spine_phantom_5gs_per_run.tsv}"
STAGE="${STAGE:-stage3_full_sweep}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

METHODS=(ours ultragauss ultragray gaussianpile slice_adapted_3dgs)
DATASETS=(
  synthetic_liver synthetic_liver synthetic_liver synthetic_liver synthetic_liver synthetic_liver
  spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom
)
SCENES=(
  liver liver liver liver liver liver
  spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom spine_phantom
)
SWEEPS=(
  l1 l2 l3 r1 r2 r3
  left1 left1_1 left2 left3 left3_2 right1 right1_1 right3 right3_2
)
TRIMS=(10 10 10 10 10 10 8 8 8 8 8 8 8 8 8)

worker_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
method_count="${#METHODS[@]}"
total_tasks=$((${#DATASETS[@]} * method_count))

if (( worker_id < 0 || worker_id >= NUM_WORKERS )); then
  echo "[ERROR] worker_id=${worker_id} out of range for NUM_WORKERS=${NUM_WORKERS}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")"

summary_has_success() {
  local dataset="$1"
  local sweep="$2"
  local method="$3"
  [[ -f "${SUMMARY_PATH}" ]] || return 1
  awk -F '\t' -v d="${dataset}" -v s="${sweep}" -v m="${method}" '
    NR > 1 && $5 == d && $6 == s && $7 == m && $34 == "success" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${SUMMARY_PATH}"
}

run_one_task() {
  local task_id="$1"
  local dataset_idx=$((task_id / method_count))
  local method_idx=$((task_id % method_count))
  local dataset="${DATASETS[$dataset_idx]}"
  local scene="${SCENES[$dataset_idx]}"
  local sweep="${SWEEPS[$dataset_idx]}"
  local trim="${TRIMS[$dataset_idx]}"
  local method="${METHODS[$method_idx]}"
  local experiment="production/comparisons/${BENCHMARK}/gs_strict/${method}"
  local exp_name="${STAGE}_${dataset}_${sweep}_${method}"
  local output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${dataset}/${sweep}/${method}"
  local efficiency_path="${output_dir}/logs/efficiency.tsv"

  if summary_has_success "${dataset}" "${sweep}" "${method}"; then
    echo "[INFO] Skip existing success: ${dataset}/${sweep}/${method}"
    return 0
  fi

  mkdir -p "${output_dir}/logs" "${output_dir}/checkpoints" "${output_dir}/vis"

  local common_overrides=(
    "scene=${scene}"
    "scene.loading.sweep_name=${sweep}"
    "scene.loading.test_mode=holdout_interval"
    "scene.loading.holdout_trim_each=${trim}"
    "scene.loading.holdout_num_samples=0"
    "scene.loading.holdout_test_ratio=0.2"
    "scene.loading.pose_type=absolute"
    "scene.loading.pose_mode=file"
    "scene.loading.image_downsample_factor=1"
    "scene.loading.axial_downsample_factor=1"
    "system.seed=42"
    "engine.interval.test=30000"
    "engine.interval.save=30000"
    "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
    "metric.max_vis_frames=10"
    "metric.save_per_frame_metrics=true"
  )

  local run_args=(
    run.py
    "experiment=${experiment}"
    "hydra.run.dir=${output_dir}"
    "system.output_root=${OUTPUT_ROOT}"
    "system.exp_tag=${exp_name}"
    "${common_overrides[@]}"
  )

  echo "============================================================"
  echo "[INFO] Worker ${worker_id}/${NUM_WORKERS} task ${task_id}/${total_tasks}"
  echo "[INFO] Date: $(date)"
  echo "[INFO] Host: $(hostname)"
  echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
  echo "[INFO] Dataset: ${dataset}"
  echo "[INFO] Sweep: ${sweep}"
  echo "[INFO] Method: ${method}"
  echo "[INFO] Experiment: ${experiment}"
  echo "[INFO] Output dir: ${output_dir}"
  echo "============================================================"
  echo "[INFO] Running: ${PYTHON_BIN} ${run_args[*]}"

  efficiency_begin_run "${efficiency_path}" train
  set +e
  "${PYTHON_BIN}" "${run_args[@]}"
  local run_exit=$?
  set -e
  efficiency_end_run "${efficiency_path}" train "${run_exit}"
  if (( run_exit != 0 )); then
    echo "[ERROR] task ${task_id} failed with exit=${run_exit}" >&2
    return "${run_exit}"
  fi

  BENCHMARK="${BENCHMARK}" \
  PROTOCOL="${PROTOCOL}" \
  STAGE="${STAGE}" \
  DATASET="${dataset}" \
  SWEEP="${sweep}" \
  METHOD="${method}" \
  EXP_NAME="${exp_name}" \
  OUTPUT_DIR="${output_dir}" \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  EFFICIENCY_PATH="${efficiency_path}" \
  "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

  echo "[INFO] Completed task ${task_id}: ${dataset}/${sweep}/${method}"
}

echo "[INFO] ss5gs worker started: worker_id=${worker_id}, total_tasks=${total_tasks}, num_workers=${NUM_WORKERS}"
had_errors=0
for ((task_id = worker_id; task_id < total_tasks; task_id += NUM_WORKERS)); do
  set +e
  run_one_task "${task_id}"
  task_exit=$?
  set -e
  if (( task_exit != 0 )); then
    had_errors=1
    echo "[ERROR] Worker ${worker_id} task ${task_id} failed with exit=${task_exit}" >&2
    if (( CONTINUE_ON_ERROR == 0 )); then
      exit "${task_exit}"
    fi
  fi
done
if (( had_errors != 0 )); then
  echo "[WARN] ss5gs worker finished with failed tasks: worker_id=${worker_id}" >&2
fi
echo "[INFO] ss5gs worker finished: worker_id=${worker_id}"
