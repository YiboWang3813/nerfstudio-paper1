#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J us4loss
#SBATCH -o logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.out
#SBATCH -e logs/comparison/us_recon_four_dataset_p100k_v1/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-gs_strict_1m30k_dc_t1000_25000_f2000_loss_l1_08_ssim_02}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/production/comparisons/${BENCHMARK}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/tus_rec_2024_qc_gs_loss_aligned_stage1.tsv}"
STAGE="${STAGE:-stage1_loss_aligned}"
DATASET="${DATASET:-tus_rec_2024_qc}"
SCENE="${SCENE:-tus_rec_2024_qc}"
SWEEP="${SWEEP:-010_LH_Per_L_DtP_qc_0005_0370}"
TRIM="${TRIM:-10}"
TEST_COUNT="${TEST_COUNT:-0}"
MAX_ACTIVE_US4LOSS="${MAX_ACTIVE_US4LOSS:-4}"
RUN_VARIANT="${RUN_VARIANT:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi

  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n us4loss -t RUNNING -o "%i" | sort -V)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_US4LOSS )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_US4LOSS; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done

    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active us4loss jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_US4LOSS=${MAX_ACTIVE_US4LOSS}."
      echo "[THROTTLE] Exiting ${current_job_id} before training; rerun this array index later."
      exit 75
    fi
  fi
fi

METHODS=(ours ultragauss ultragray gaussianpile ultrags_baseline slice_adapted_3dgs)

if [[ -n "${METHOD_OVERRIDE:-}" ]]; then
  method="${METHOD_OVERRIDE}"
else
  task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  if (( task_id < 0 || task_id >= ${#METHODS[@]} )); then
    echo "[ERROR] task_id=${task_id} is out of range for ${#METHODS[@]} loss-aligned GS methods" >&2
    exit 2
  fi
  method="${METHODS[$task_id]}"
fi

experiment="production/comparisons/${BENCHMARK}/gs_strict/${method}"
variant_suffix=""
if [[ -n "${RUN_VARIANT}" ]]; then
  variant_suffix="_${RUN_VARIANT}"
fi
exp_name="${STAGE}_${DATASET}_${SWEEP}_${method}${variant_suffix}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${DATASET}/${SWEEP}/${method}${variant_suffix}"

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
  "engine.interval.save=30000"
  "loss.needed_losses=[l1,ssim]"
  "loss.lambdas.l1=0.8"
  "loss.lambdas.ssim=0.2"
  "loss.lambdas.scale_wall=0.0"
  "loss.lambdas.highpass_l1=0.0"
  "loss.lambdas.lpips=0.0"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
)

if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  extra_overrides_array=(${EXTRA_OVERRIDES})
  common_overrides+=("${extra_overrides_array[@]}")
fi

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}/ops/ultragauss_renderer:${PROJECT_DIR}/ops/ultragray_renderer:${PROJECT_DIR}/ops/gaussianplie_renderer:${PROJECT_DIR}/ops/ultrags_baseline_renderer:${PROJECT_DIR}/ops/slice_adapted_3dgs_renderer:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/production/comparisons/${BENCHMARK}/efficiency_helpers.sh"

echo "============================================================"
echo "[INFO] ${BENCHMARK} GS loss-aligned TUS-QC task"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Method: ${method}"
echo "[INFO] Variant: ${RUN_VARIANT:-default}"
echo "[INFO] Loss: needed_losses=[l1,ssim], l1=0.8, ssim=0.2, scale_wall=0.0"
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  echo "[INFO] Extra overrides: ${EXTRA_OVERRIDES}"
fi
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
BENCHMARK="${BENCHMARK}" PROTOCOL="${PROTOCOL}" STAGE="${STAGE}" DATASET="${DATASET}" SWEEP="${SWEEP}" METHOD="${method}" EXP_NAME="${exp_name}" OUTPUT_DIR="${output_dir}" SUMMARY_PATH="${SUMMARY_PATH}" EFFICIENCY_PATH="${efficiency_path}" "${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
