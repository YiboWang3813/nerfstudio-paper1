#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4thick
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-gs_strict_thickness_ratio_stage1_1m30k}"
STAGE="${STAGE:-thickness_ablation_stage1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
WANDB_MODE="${WANDB_MODE:-offline}"
MAX_ACTIVE_US4THICK="${MAX_ACTIVE_US4THICK:-2}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n us4thick -t RUNNING -o "%i" | sort -V)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_US4THICK )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_US4THICK; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done
    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active us4thick jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_US4THICK=${MAX_ACTIVE_US4THICK}."
      echo "[THROTTLE] Exiting ${current_job_id} before training; rerun this array index later."
      exit 75
    fi
  fi
fi

METHODS=(gaussianpile slice_adapted_3dgs)
DATASETS=(tus_rec_2024_qc tdsc_abus_2023)
SCENES=(tus_rec_2024_qc tdsc_abus_2023)
SWEEPS=(010_LH_Per_L_DtP_qc_0005_0370 validation_100)
TRIMS=(10 10)
TEST_COUNTS=(0 0)
DELTAS=(0.404000 0.475674)

RATIO_LABELS=(r050 r075 r100 r125 r150 r200 r300)
RATIO_VALUES=(0.50 0.75 1.00 1.25 1.50 2.00 3.00)
TUS_THICKNESS=(0.202000 0.303000 0.404000 0.505000 0.606000 0.808000 1.212000)
TDSC_THICKNESS=(0.237837 0.356756 0.475674 0.594593 0.713511 0.951348 1.427022)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
method_count="${#METHODS[@]}"
ratio_count="${#RATIO_LABELS[@]}"
dataset_count="${#DATASETS[@]}"
total_count=$((dataset_count * method_count * ratio_count))

if (( task_id < 0 || task_id >= total_count )); then
  echo "[ERROR] task_id=${task_id} is out of range for total_count=${total_count}" >&2
  exit 2
fi

dataset_idx=$((task_id / (method_count * ratio_count)))
within_dataset=$((task_id % (method_count * ratio_count)))
method_idx=$((within_dataset / ratio_count))
ratio_idx=$((within_dataset % ratio_count))

dataset="${DATASETS[$dataset_idx]}"
scene="${SCENES[$dataset_idx]}"
sweep="${SWEEPS[$dataset_idx]}"
trim="${TRIMS[$dataset_idx]}"
test_count="${TEST_COUNTS[$dataset_idx]}"
delta="${DELTAS[$dataset_idx]}"
method="${METHODS[$method_idx]}"
ratio_label="${RATIO_LABELS[$ratio_idx]}"
ratio_value="${RATIO_VALUES[$ratio_idx]}"
if [[ "${dataset}" == "tus_rec_2024_qc" ]]; then
  thickness_mm="${TUS_THICKNESS[$ratio_idx]}"
else
  thickness_mm="${TDSC_THICKNESS[$ratio_idx]}"
fi

experiment="production/comparisons/${BENCHMARK}/gs_strict/${method}"
exp_name="${STAGE}_${dataset}_${sweep}_${method}_${ratio_label}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${dataset}/${sweep}/${method}/${ratio_label}"

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

if [[ "${LIGHT_OUTPUTS:-0}" == "1" ]]; then
  common_overrides+=(
    "metric.max_vis_frames=0"
    "engine.interval.save=0"
  )
fi

case "${method}" in
  gaussianpile)
    thickness_override="model.slice_render.sigma_z_mm=${thickness_mm}"
    ;;
  slice_adapted_3dgs)
    thickness_override="model.render.slice_thickness_mm=${thickness_mm}"
    ;;
  *)
    echo "[ERROR] Unsupported method=${method}" >&2
    exit 2
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} thickness ablation task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Dataset: ${dataset}"
echo "[INFO] Sweep: ${sweep}"
echo "[INFO] Delta elevational mm: ${delta}"
echo "[INFO] Method: ${method}"
echo "[INFO] Ratio: ${ratio_value} (${ratio_label})"
echo "[INFO] Effective thickness mm: ${thickness_mm}"
echo "[INFO] Thickness override: ${thickness_override}"
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
  "${thickness_override}"
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
BENCHMARK="${BENCHMARK}" PROTOCOL="${PROTOCOL}" STAGE="${STAGE}" DATASET="${dataset}" SWEEP="${sweep}" METHOD="${method}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
