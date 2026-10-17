#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J tdsc_ours_echolr
#SBATCH -o logs/preproduction/ideas/ours/tdsc_echo_lr_refine/%x_%A_%a.out
#SBATCH -e logs/preproduction/ideas/ours/tdsc_echo_lr_refine/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-tdsc_echo_lr_refine}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
MAX_ACTIVE_TDSC_OURS_ECHOLR="${MAX_ACTIVE_TDSC_OURS_ECHOLR:-4}"

VARIANTS=(
  echo_init005_lr0010
  echo_init005_lr0015
  echo_init005_lr0020
  echo_init005_lr0025
)

ECHO_LR=(0.01 0.015 0.02 0.025)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= ${#VARIANTS[@]} )); then
  echo "[ERROR] task_id=${task_id} out of range for ${#VARIANTS[@]} variants" >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n tdsc_ours_echolr -t RUNNING -o "%i" | sort -V)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_TDSC_OURS_ECHOLR )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_TDSC_OURS_ECHOLR; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done
    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_TDSC_OURS_ECHOLR=${MAX_ACTIVE_TDSC_OURS_ECHOLR}."
      exit 75
    fi
  fi
fi

variant="${VARIANTS[$task_id]}"
dataset="tdsc_abus_2023"
scene="tdsc_abus_2023"
sweep="validation_100"
stage="stage_tdsc_validation100"
method="ours_b8_no_intprune_slice_depth_${variant}"
exp_name="${stage}_${dataset}_${sweep}_${method}"
output_dir="${OUTPUT_ROOT}/runs/${stage}/nvs/${dataset}/${sweep}/${method}"

common_overrides=(
  "scene=${scene}"
  "scene.loading.sweep_name=${sweep}"
  "scene.loading.test_mode=holdout_interval"
  "scene.loading.holdout_trim_each=10"
  "scene.loading.holdout_num_samples=0"
  "scene.loading.holdout_test_ratio=0.2"
  "scene.loading.pose_type=absolute"
  "scene.loading.pose_mode=file"
  "scene.loading.image_downsample_factor=1"
  "scene.loading.axial_downsample_factor=1"
  "system.seed=42"
  "engine.interval.save=30000"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
  "model.gauss_container.constraints.scale_activation=bounded_sigmoid"
  "model.gauss_container.constraints.scale_min_mm=0.03"
  "model.gauss_container.constraints.scale_max_mm=3.89"
  "model.gauss_container.constraints.echo_alpha_max=0.2"
  "model.gauss_container.init_val.echo_alphas=0.05"
  "model.gauss_container.lr.echo_alphas=${ECHO_LR[$task_id]}"
  "model.gauss_container.lr.attenuations=0.01"
  "model.offline.density_control.thresholds.prune_intensity=0.0"
  "model.slice_render.ray_attenuation_enabled=true"
  "model.slice_render.ray_attenuation_formula=slice_depth"
  "model.slice_render.ray_attenuation_beta=0.80"
  "model.slice_render.ray_attenuation_strength=0.55"
  "model.slice_render.ray_attenuation_min_transmission=0.35"
  "model.slice_render.ray_attenuation_quadrature_points=1"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/us_recon_four_dataset_p100k_v1/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] TDSC ours echo LR refine"
echo "[INFO] Variant: ${variant}"
echo "[INFO] echo_init=0.05 echo_lr=${ECHO_LR[$task_id]} attenuation_lr=0.01"
echo "[INFO] Output dir: ${output_dir}"
echo "============================================================"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/checkpoints" "${output_dir}/vis"

run_args=(
  run.py
  "experiment=production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours"
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

BENCHMARK="${BENCHMARK}" \
PROTOCOL="tdsc_ours_echo_lr_refine" \
STAGE="${stage}" \
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
