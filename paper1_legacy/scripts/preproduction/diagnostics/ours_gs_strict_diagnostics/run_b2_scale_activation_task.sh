#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ogsd_b2
#SBATCH -o logs/ours_gs_strict_diagnostics/%x_%A_%a.out
#SBATCH -e logs/ours_gs_strict_diagnostics/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-ours_gs_strict_diagnostics}"
PROTOCOL="${PROTOCOL:-b2_scale_activation_1m30k}"
STAGE="${STAGE:-b2_scale_activation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/diagnostics/ours_gs_strict_diagnostics}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/${STAGE}_per_run.tsv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/${USER}/omniultra_checkpoints}"
WANDB_MODE="${WANDB_MODE:-offline}"
MAX_ACTIVE_OGSD_B2="${MAX_ACTIVE_OGSD_B2:-4}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n ogsd_b2 -t RUNNING -o "%i" | sort -V)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_OGSD_B2 )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_OGSD_B2; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done
    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active ogsd_b2 jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_OGSD_B2=${MAX_ACTIVE_OGSD_B2}."
      echo "[THROTTLE] Exiting ${current_job_id} before training; rerun this array index later."
      exit 75
    fi
  fi
fi

DATASETS=(tus_rec_2024_qc tus_rec_2024_qc tdsc_abus_2023 tdsc_abus_2023)
SCENES=(tus_rec_2024_qc tus_rec_2024_qc tdsc_abus_2023 tdsc_abus_2023)
SWEEPS=(010_LH_Per_L_DtP_qc_0005_0370 010_LH_Per_L_DtP_qc_0005_0370 validation_100 validation_100)
TRIMS=(10 10 10 10)
TEST_COUNTS=(0 0 0 0)
CORRIDOR_HALVES=(0.202000 0.202000 0.237837 0.237837)
VARIANTS=(exp bounded exp bounded)
SCALE_ACTIVATIONS=(exp bounded_sigmoid exp bounded_sigmoid)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
total_count="${#DATASETS[@]}"
if (( task_id < 0 || task_id >= total_count )); then
  echo "[ERROR] task_id=${task_id} is out of range for total_count=${total_count}" >&2
  exit 2
fi

dataset="${DATASETS[$task_id]}"
scene="${SCENES[$task_id]}"
sweep="${SWEEPS[$task_id]}"
trim="${TRIMS[$task_id]}"
test_count="${TEST_COUNTS[$task_id]}"
corridor_half_mm="${CORRIDOR_HALVES[$task_id]}"
variant="${VARIANTS[$task_id]}"
scale_activation="${SCALE_ACTIVATIONS[$task_id]}"

method="ours"
experiment="production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours"
exp_name="${STAGE}_${dataset}_${sweep}_${method}_${variant}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${dataset}/${sweep}/${method}/${variant}"

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
  "system.checkpoint_root=${CHECKPOINT_ROOT}"
  "system.link_checkpoint_dir=true"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
  "++model.slice_render.elev_corridor_half_mm=${corridor_half_mm}"
  "model.gauss_container.constraints.scale_activation=${scale_activation}"
  "model.gauss_container.constraints.scale_min_mm=0.05"
  "model.gauss_container.constraints.scale_max_mm=3.0"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/us_recon_four_dataset_p100k_v1/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Dataset: ${dataset}"
echo "[INFO] Sweep: ${sweep}"
echo "[INFO] Variant: ${variant}"
echo "[INFO] Scale activation: ${scale_activation}"
echo "[INFO] Ours elev corridor half mm: ${corridor_half_mm}"
echo "[INFO] Experiment: ${experiment}"
echo "[INFO] Output dir: ${output_dir}"
echo "[INFO] Checkpoint root: ${CHECKPOINT_ROOT}"
echo "============================================================"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/vis"

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
BENCHMARK="${BENCHMARK}" PROTOCOL="${PROTOCOL}" STAGE="${STAGE}" DATASET="${dataset}" SWEEP="${sweep}" METHOD="${method}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
