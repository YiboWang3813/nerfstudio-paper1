#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ug_chi2
#SBATCH -o logs/preproduction/ultragauss_tus2024qc_ablation/%x_%A_%a.out
#SBATCH -e logs/preproduction/ultragauss_tus2024qc_ablation/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
PROTOCOL="${PROTOCOL:-preproduction_ultragauss_tus2024qc_chi2}"
EXPERIMENT="${EXPERIMENT:-production/comparisons/${BENCHMARK}/gs_strict/ultragauss}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ultragauss_tus2024qc_ablation}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/chi2_one_sweep.tsv}"
STAGE="${STAGE:-chi2_one_sweep}"
DATASET="${DATASET:-tus_rec_2024_qc}"
SCENE="${SCENE:-tus_rec_2024_qc}"
SWEEP="${SWEEP:-011_LH_Per_L_PtD_qc_0006_0690}"
TRIM="${TRIM:-10}"
TEST_COUNT="${TEST_COUNT:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
MAX_ACTIVE_UG_CHI2="${MAX_ACTIVE_UG_CHI2:-4}"

VARIANT_NAMES=(
  chi2_5991
  chi2_3000
  chi2_1500
  chi2_0750
)
CHI2_VALUES=(
  5.991
  3.000
  1.500
  0.750
)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#VARIANT_NAMES[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#VARIANT_NAMES[@]} variants" >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  current_job_id="${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    current_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  mapfile -t active_jobs < <(squeue -h -u "${USER}" -n ug_chi2 -t RUNNING -o "%i" 2>/dev/null | sort -V || true)
  if (( ${#active_jobs[@]} > MAX_ACTIVE_UG_CHI2 )); then
    keep_current=0
    for ((i = 0; i < MAX_ACTIVE_UG_CHI2; i++)); do
      if [[ "${active_jobs[$i]}" == "${current_job_id}" ]]; then
        keep_current=1
        break
      fi
    done
    if (( keep_current == 0 )); then
      echo "[THROTTLE] Active ug_chi2 jobs (${#active_jobs[@]}) exceed MAX_ACTIVE_UG_CHI2=${MAX_ACTIVE_UG_CHI2}."
      exit 75
    fi
  fi
fi

variant="${VARIANT_NAMES[$task_id]}"
chi2="${CHI2_VALUES[$task_id]}"
method="ultragauss_${variant}"
exp_name="${STAGE}_${DATASET}_${SWEEP}_${method}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${DATASET}/${SWEEP}/${method}"

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
  "engine.interval.save=0"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
  "model.render.chi2_threshold=${chi2}"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ultragauss_renderer:${PYTHONPATH:-}"

echo "============================================================"
echo "[INFO] UltraGauss TUS-REC 2024 QC chi2 ablation"
echo "[INFO] Protocol: ${PROTOCOL}"
echo "[INFO] Stage: ${STAGE}"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Variant: ${variant}"
echo "[INFO] chi2_threshold=${chi2}"
echo "[INFO] Output dir: ${output_dir}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/vis"

run_args=(
  run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${output_dir}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${exp_name}"
  "${common_overrides[@]}"
)

echo "[INFO] Running: ${PYTHON_BIN} ${run_args[*]}"
set +e
"${PYTHON_BIN}" "${run_args[@]}"
run_exit=$?
set -e
if (( run_exit != 0 )); then
  exit "${run_exit}"
fi

echo "[INFO] Summarizing scalar results"
BENCHMARK="${BENCHMARK}" \
PROTOCOL="${PROTOCOL}" \
STAGE="${STAGE}" \
DATASET="${DATASET}" \
SWEEP="${SWEEP}" \
METHOD="${method}" \
EXP_NAME="${exp_name}" \
OUTPUT_DIR="${output_dir}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
"${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

if [[ -d "${output_dir}/checkpoints" ]]; then
  echo "[INFO] Removing checkpoint payloads from ${output_dir}/checkpoints"
  find "${output_dir}/checkpoints" -type f -delete
fi

echo "============================================================"
echo "[INFO] Task finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
