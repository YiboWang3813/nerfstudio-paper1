#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ug55top
#SBATCH -o logs/preproduction/ultragauss_tus2024qc_ablation/%x_%A_%a.out
#SBATCH -e logs/preproduction/ultragauss_tus2024qc_ablation/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
BENCHMARK="${BENCHMARK:-us_recon_four_dataset_p100k_v1}"
EXPERIMENT="${EXPERIMENT:-production/comparisons/${BENCHMARK}/gs_strict/ultragauss}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ultragauss_tus2024qc_ablation}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/chi2_5500_top_sweeps.tsv}"
STAGE="${STAGE:-chi2_5500_top_sweeps}"
DATASET="${DATASET:-tus_rec_2024_qc}"
SCENE="${SCENE:-tus_rec_2024_qc}"
CHI2="${CHI2:-5.500}"
WANDB_MODE="${WANDB_MODE:-offline}"

SWEEPS=(
  011_LH_Per_L_DtP_qc_0005_0594
  011_RH_Per_C_PtD_qc_0186_0557_s02
  011_RH_Per_S_DtP_qc_0166_0553
  011_LH_Per_S_DtP_qc_0118_0529
)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_id < 0 || task_id >= ${#SWEEPS[@]} )); then
  echo "[ERROR] task_id=${task_id} is out of range for ${#SWEEPS[@]} sweeps" >&2
  exit 2
fi

sweep="${SWEEPS[$task_id]}"
method="ultragauss_chi2_5500"
exp_name="${STAGE}_${DATASET}_${sweep}_${method}"
output_dir="${OUTPUT_ROOT}/runs/${STAGE}/nvs/${DATASET}/${sweep}/${method}"

overrides=(
  "scene=${SCENE}"
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
  "engine.interval.save=0"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
  "model.render.chi2_threshold=${CHI2}"
)

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ultragauss_renderer:${PYTHONPATH:-}"

echo "============================================================"
echo "[INFO] UltraGauss TUS-REC 2024 QC chi2=5.5 top-sweep validation"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Sweep: ${sweep}"
echo "[INFO] chi2_threshold=${CHI2}"
echo "[INFO] Output dir: ${output_dir}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/vis"

run_args=(
  run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${output_dir}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${exp_name}"
  "${overrides[@]}"
)

"${PYTHON_BIN}" "${run_args[@]}"

BENCHMARK="${BENCHMARK}" \
PROTOCOL="preproduction_ultragauss_tus2024qc_chi2_5500_top_sweeps" \
STAGE="${STAGE}" \
DATASET="${DATASET}" \
SWEEP="${sweep}" \
METHOD="${method}" \
EXP_NAME="${exp_name}" \
OUTPUT_DIR="${output_dir}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
"${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

if [[ -d "${output_dir}/checkpoints" ]]; then
  find "${output_dir}/checkpoints" -type f -delete
fi

echo "[INFO] Task finished at $(date)"
