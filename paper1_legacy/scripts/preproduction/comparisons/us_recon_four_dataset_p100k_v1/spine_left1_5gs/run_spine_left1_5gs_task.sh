#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J spine5gs
#SBATCH -o logs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/spine_left1_5gs/%x_%A_%a.out
#SBATCH -e logs/preproduction/comparisons/us_recon_four_dataset_p100k_v1/spine_left1_5gs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-gs_strict_1m30k_dc_t1000_25000_f2000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/stage1_calibration_spine_phantom_left1_5gs_per_run.tsv}"
STAGE="${STAGE:-stage1_calibration}"
MAX_ACTIVE_SPINE5GS="${MAX_ACTIVE_SPINE5GS:-4}"

METHODS=(ours ultragauss ultragray gaussianpile slice_adapted_3dgs)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= ${#METHODS[@]} )); then
  echo "[ERROR] task_id=${task_id} out of range for ${#METHODS[@]} methods" >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n spine5gs -t RUNNING -o "%i" | sort -V)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_SPINE5GS )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_SPINE5GS; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done
    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active spine5gs jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_SPINE5GS=${MAX_ACTIVE_SPINE5GS}."
      exit 75
    fi
  fi
fi

dataset="spine_phantom"
scene="spine_phantom"
sweep="left1"
trim="8"
test_count="0"
method="${METHODS[$task_id]}"
experiment="production/comparisons/${BENCHMARK}/gs_strict/${method}"
exp_name="${STAGE}_${dataset}_${sweep}_${method}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${dataset}/${sweep}/${method}"

common_overrides=(
  "scene=${scene}"
  "scene.loading.sweep_name=${sweep}"
  "scene.loading.test_mode=holdout_interval"
  "scene.loading.holdout_trim_each=${trim}"
  "scene.loading.holdout_num_samples=${test_count}"
  "scene.loading.holdout_test_ratio=0.2"
  "scene.loading.pose_type=absolute"
  "scene.loading.pose_mode=file"
  "scene.loading.image_downsample_factor=1"
  "scene.loading.axial_downsample_factor=1"
  "system.seed=42"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] Spine phantom left1 5-GS comparison task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Method: ${method}"
echo "[INFO] Experiment: ${experiment}"
echo "[INFO] Output dir: ${output_dir}"
echo "============================================================"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/checkpoints" "${output_dir}/vis"

run_args=(
  run.py
  "experiment=${experiment}"
  "hydra.run.dir=${output_dir}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${exp_name}"
  "${common_overrides[@]}"
)

echo "[INFO] Running: ${PYTHON_BIN} ${run_args[*]}"
efficiency_path="${output_dir}/logs/efficiency.tsv"
efficiency_begin_run "${efficiency_path}" train
set +e
"${PYTHON_BIN}" "${run_args[@]}"
run_exit=$?
set -e
efficiency_end_run "${efficiency_path}" train "${run_exit}"
if (( run_exit != 0 )); then
  exit "${run_exit}"
fi

echo "[INFO] Summarizing scalar results"
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

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
