#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J tdsc_ours_ea2
#SBATCH -o logs/preproduction/ideas/ours/tdsc_echo_attenuation_tuning_v2/%x_%A_%a.out
#SBATCH -e logs/preproduction/ideas/ours/tdsc_echo_attenuation_tuning_v2/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-tdsc_echo_attenuation_tuning_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
MAX_ACTIVE_TDSC_OURS_EA2="${MAX_ACTIVE_TDSC_OURS_EA2:-4}"

VARIANTS=(
  emax025_elr008_alr003_att055
  emax030_elr008_alr003_att055
  emax025_elr008_alr003_attweak
  emax030_elr008_alr001_attweak
)

ECHO_ALPHA_MAX=(0.25 0.30 0.25 0.30)
ECHO_ALPHA_LR=(0.080 0.080 0.080 0.080)
ATTENUATION_LR=(0.003 0.003 0.003 0.001)
RAY_BETA=(0.80 0.80 0.60 0.60)
RAY_STRENGTH=(0.55 0.55 0.35 0.35)
RAY_MIN_TRANSMISSION=(0.35 0.35 0.55 0.55)

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
  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n tdsc_ours_ea2 -t RUNNING -o "%i" | sort -V)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_TDSC_OURS_EA2 )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_TDSC_OURS_EA2; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done
    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_TDSC_OURS_EA2=${MAX_ACTIVE_TDSC_OURS_EA2}."
      exit 75
    fi
  fi
fi

variant="${VARIANTS[$task_id]}"
dataset="tdsc_abus_2023"
scene="tdsc_abus_2023"
sweep="validation_100"
stage="stage_tdsc_validation100"
method="ours_${variant}"
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
  "model.gauss_container.constraints.echo_alpha_max=${ECHO_ALPHA_MAX[$task_id]}"
  "model.gauss_container.lr.echo_alphas=${ECHO_ALPHA_LR[$task_id]}"
  "model.gauss_container.lr.attenuations=${ATTENUATION_LR[$task_id]}"
  "model.slice_render.ray_attenuation_enabled=true"
  "model.slice_render.ray_attenuation_beta=${RAY_BETA[$task_id]}"
  "model.slice_render.ray_attenuation_strength=${RAY_STRENGTH[$task_id]}"
  "model.slice_render.ray_attenuation_min_transmission=${RAY_MIN_TRANSMISSION[$task_id]}"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/us_recon_four_dataset_p100k_v1/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] TDSC ours echo/attenuation tuning v2"
echo "[INFO] Variant: ${variant}"
echo "[INFO] echo_alpha_max=${ECHO_ALPHA_MAX[$task_id]} echo_alpha_lr=${ECHO_ALPHA_LR[$task_id]} attenuation_lr=${ATTENUATION_LR[$task_id]}"
echo "[INFO] ray_beta=${RAY_BETA[$task_id]} ray_strength=${RAY_STRENGTH[$task_id]} ray_min_transmission=${RAY_MIN_TRANSMISSION[$task_id]}"
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
PROTOCOL="tdsc_ours_bounded_scale_echo_attenuation_tuning_v2" \
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
