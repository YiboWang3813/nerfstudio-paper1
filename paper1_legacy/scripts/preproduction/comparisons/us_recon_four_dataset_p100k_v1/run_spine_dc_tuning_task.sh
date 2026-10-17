#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J spdc
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-gs_strict_spine_left1_dc_tuning}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/dc_tuning_spine_phantom_left1_gaussianpile_ultrags.tsv}"
STAGE="${STAGE:-dc_tuning}"
DATASET="${DATASET:-spine_phantom}"
SCENE="${SCENE:-spine_phantom}"
SWEEP="${SWEEP:-left1}"
TRIM="${TRIM:-8}"
TEST_COUNT="${TEST_COUNT:-0}"
MAX_ACTIVE_SPDC="${MAX_ACTIVE_SPDC:-4}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  mapfile -t active_spdc < <(squeue -h -u "${USER}" -n spdc -t RUNNING -o "%i" | sort -V)
  if (( ${#active_spdc[@]} > MAX_ACTIVE_SPDC )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_SPDC; i++)); do
      if [[ "${active_spdc[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done

    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active spdc jobs (${#active_spdc[@]}) exceed MAX_ACTIVE_SPDC=${MAX_ACTIVE_SPDC}."
      echo "[THROTTLE] Exiting ${current_job_id} before training; rerun this array index later."
      exit 75
    fi
  fi
fi

VARIANT_NAMES=(
  gp_p005_obs4_score025
  gp_p002_obs4_score025
  gp_p005_obs8_score025
  gp_p002_obs8_score015
  ugs_op002_reset_on
  ugs_op001_reset_on
  ugs_op0005_reset_on
  ugs_op001_reset_off
  ugs_scale01_op005_reset_off
  ugs_scale01_op001_reset_off
  ugs_scale01_dist100_reset_off
  ugs_scale01_dist300_reset_off
)
METHODS=(
  gaussianpile
  gaussianpile
  gaussianpile
  gaussianpile
  ultrags_baseline
  ultrags_baseline
  ultrags_baseline
  ultrags_baseline
  ultrags_baseline
  ultrags_baseline
  ultrags_baseline
  ultrags_baseline
)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#VARIANT_NAMES[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#VARIANT_NAMES[@]} variants" >&2
  exit 2
fi

variant="${VARIANT_NAMES[$task_id]}"
method="${METHODS[$task_id]}"
experiment="production/comparisons/${BENCHMARK}/gs_strict/${method}"
exp_name="${STAGE}_${DATASET}_${SWEEP}_${variant}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${DATASET}/${SWEEP}/${variant}"

common_overrides=(
  "scene=${SCENE}"
  "scene.loading.sweep_name=${SWEEP}"
  "scene.loading.test_mode=holdout_interval"
  "scene.loading.holdout_trim_each=${TRIM}"
  "scene.loading.holdout_num_samples=${TEST_COUNT}"
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

case "${variant}" in
  gp_p005_obs4_score025)
    common_overrides+=(
      "model.offline.density_control.thresholds.prune_opacity=0.005"
      "model.offline.density_control.thresholds.min_prune_observations=4"
      "model.offline.density_control.thresholds.prune_observation_score=0.25"
    )
    ;;
  gp_p002_obs4_score025)
    common_overrides+=(
      "model.offline.density_control.thresholds.prune_opacity=0.002"
      "model.offline.density_control.thresholds.min_prune_observations=4"
      "model.offline.density_control.thresholds.prune_observation_score=0.25"
    )
    ;;
  gp_p005_obs8_score025)
    common_overrides+=(
      "model.offline.density_control.thresholds.prune_opacity=0.005"
      "model.offline.density_control.thresholds.min_prune_observations=8"
      "model.offline.density_control.thresholds.prune_observation_score=0.25"
    )
    ;;
  gp_p002_obs8_score015)
    common_overrides+=(
      "model.offline.density_control.thresholds.prune_opacity=0.002"
      "model.offline.density_control.thresholds.min_prune_observations=8"
      "model.offline.density_control.thresholds.prune_observation_score=0.15"
    )
    ;;
  ugs_op002_reset_on)
    common_overrides+=(
      "model.offline.density_control.thresholds.opacity_cull=0.002"
      "model.offline.density_control.enable.opacity_reset=true"
    )
    ;;
  ugs_op001_reset_on)
    common_overrides+=(
      "model.offline.density_control.thresholds.opacity_cull=0.001"
      "model.offline.density_control.enable.opacity_reset=true"
    )
    ;;
  ugs_op0005_reset_on)
    common_overrides+=(
      "model.offline.density_control.thresholds.opacity_cull=0.0005"
      "model.offline.density_control.enable.opacity_reset=true"
    )
    ;;
  ugs_op001_reset_off)
    common_overrides+=(
      "model.offline.density_control.thresholds.opacity_cull=0.001"
      "model.offline.density_control.enable.opacity_reset=false"
    )
    ;;
  ugs_scale01_op005_reset_off)
    common_overrides+=(
      "model.gauss_container.init_val.scale_2d=[0.1,0.1]"
      "model.gauss_container.init_val.opacity=0.05"
      "model.gauss_container.lr.scales_2d=0.001"
      "model.gauss_container.lr.opacities=0.03"
      "model.offline.density_control.thresholds.opacity_cull=0.001"
      "model.offline.density_control.thresholds.min_scale_2d=0.01"
      "model.offline.density_control.enable.opacity_reset=false"
    )
    ;;
  ugs_scale01_op001_reset_off)
    common_overrides+=(
      "model.gauss_container.init_val.scale_2d=[0.1,0.1]"
      "model.gauss_container.init_val.opacity=0.01"
      "model.gauss_container.lr.scales_2d=0.001"
      "model.gauss_container.lr.opacities=0.02"
      "model.offline.density_control.thresholds.opacity_cull=0.001"
      "model.offline.density_control.thresholds.min_scale_2d=0.01"
      "model.offline.density_control.enable.opacity_reset=false"
    )
    ;;
  ugs_scale01_dist100_reset_off)
    common_overrides+=(
      "model.render.physical_camera_distance_mm=100.0"
      "model.gauss_container.init_val.scale_2d=[0.1,0.1]"
      "model.gauss_container.init_val.opacity=0.05"
      "model.gauss_container.lr.scales_2d=0.001"
      "model.gauss_container.lr.opacities=0.03"
      "model.offline.density_control.thresholds.opacity_cull=0.001"
      "model.offline.density_control.thresholds.min_scale_2d=0.01"
      "model.offline.density_control.enable.opacity_reset=false"
    )
    ;;
  ugs_scale01_dist300_reset_off)
    common_overrides+=(
      "model.render.physical_camera_distance_mm=300.0"
      "model.gauss_container.init_val.scale_2d=[0.1,0.1]"
      "model.gauss_container.init_val.opacity=0.05"
      "model.gauss_container.lr.scales_2d=0.001"
      "model.gauss_container.lr.opacities=0.03"
      "model.offline.density_control.thresholds.opacity_cull=0.001"
      "model.offline.density_control.thresholds.min_scale_2d=0.01"
      "model.offline.density_control.enable.opacity_reset=false"
    )
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} spine DC tuning task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Method: ${method}"
echo "[INFO] Variant: ${variant}"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Sweep: ${SWEEP}"
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
BENCHMARK="${BENCHMARK}" PROTOCOL="${PROTOCOL}" STAGE="${STAGE}" DATASET="${DATASET}" SWEEP="${SWEEP}" METHOD="${method}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
