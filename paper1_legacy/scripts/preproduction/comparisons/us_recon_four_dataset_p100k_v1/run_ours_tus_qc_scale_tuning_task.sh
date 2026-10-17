#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4scale
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-gs_strict_1m30k_dc_t1000_25000_f2000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/tus_rec_2024_qc_ours_scale_tuning.tsv}"
STAGE="${STAGE:-ours_scale_tuning_tus_rec_2024_qc}"
DATASET="${DATASET:-tus_rec_2024_qc}"
SCENE="${SCENE:-tus_rec_2024_qc}"
SWEEP="${SWEEP:-010_LH_Per_L_DtP_qc_0005_0370}"
TRIM="${TRIM:-10}"
TEST_COUNT="${TEST_COUNT:-0}"
MAX_ACTIVE_US4SCALE="${MAX_ACTIVE_US4SCALE:-4}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n us4scale -t RUNNING -o "%i" | sort -V)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_US4SCALE )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_US4SCALE; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done

    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active us4scale jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_US4SCALE=${MAX_ACTIVE_US4SCALE}."
      echo "[THROTTLE] Exiting ${current_job_id} before training; rerun this array index later."
      exit 75
    fi
  fi
fi

VARIANT_NAMES=(
  tusq_scale_a_l65_s35_sw0
  tusq_scale_b_l65_s35_sw10
  tusq_scale_c_l65_s35_sw25
  tusq_scale_d_l80_s20_sw10
)
L1_WEIGHTS=(0.65 0.65 0.65 0.80)
SSIM_WEIGHTS=(0.35 0.35 0.35 0.20)
SCALE_WALL_WEIGHTS=(0.0 10.0 25.0 10.0)
DENSIFY_GRADS=(0.000015 0.000015 0.000015 0.000015)
SCORE_QUANTILES=(0.98 0.98 0.98 0.98)
MAX_FRACTIONS=(0.05 0.05 0.05 0.05)
LOW_CONTRIBUTION_Q=(0.10 0.10 0.10 0.10)
PROTECT_ATTENUATION_Q=(0.80 0.80 0.80 0.80)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#VARIANT_NAMES[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#VARIANT_NAMES[@]} TUS-QC scale tuning variants" >&2
  exit 2
fi

variant="${VARIANT_NAMES[$task_id]}"
l1_weight="${L1_WEIGHTS[$task_id]}"
ssim_weight="${SSIM_WEIGHTS[$task_id]}"
scale_wall_weight="${SCALE_WALL_WEIGHTS[$task_id]}"
densify_grad="${DENSIFY_GRADS[$task_id]}"
score_quantile="${SCORE_QUANTILES[$task_id]}"
max_fraction="${MAX_FRACTIONS[$task_id]}"
low_contribution_q="${LOW_CONTRIBUTION_Q[$task_id]}"
protect_attenuation_q="${PROTECT_ATTENUATION_Q[$task_id]}"

method="${variant}"
experiment="production/comparisons/${BENCHMARK}/gs_strict/ours"
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
  "++model.slice_render.elev_corridor_half_mm=0.202"
  "loss.lambdas.l1=${l1_weight}"
  "loss.lambdas.ssim=${ssim_weight}"
  "loss.lambdas.scale_wall=${scale_wall_weight}"
  "model.offline.density_control.thresholds.densify_grad=${densify_grad}"
  "model.offline.density_control.thresholds.prune_opacity=0.0001"
  "model.offline.density_control.responsibility.score_quantile=${score_quantile}"
  "model.offline.density_control.responsibility.max_fraction=${max_fraction}"
  "model.offline.density_control.responsibility.low_contribution_quantile=${low_contribution_q}"
  "model.offline.density_control.responsibility.protect_attenuation_quantile=${protect_attenuation_q}"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} ours TUS-QC scale tuning task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Variant: ${variant}"
echo "[INFO] l1=${l1_weight} ssim=${ssim_weight} scale_wall=${scale_wall_weight} densify_grad=${densify_grad}"
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
