#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ugr_lwbal
#SBATCH -o logs/ultragray/%x_%j.out
#SBATCH -e logs/ultragray/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
PYTHON_BIN="${PYTHON_BIN:-/home/boywang25/miniconda3/envs/${CONDA_ENV}/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/ultragray/acc_speed_balance}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ultragray_acc_speed_balance}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/tables/ultragray_acc_speed_balance.tsv}"
DATASET="${DATASET:?DATASET is required}"
SCENE="${SCENE:?SCENE is required}"
SWEEP="${SWEEP:?SWEEP is required}"
LOCAL_WINDOW_MM="${LOCAL_WINDOW_MM:?LOCAL_WINDOW_MM is required}"
EXP_NAME="${EXP_NAME:-ultragray_${DATASET}_${SWEEP}_lw${LOCAL_WINDOW_MM}mm}"
WANDB_MODE="${WANDB_MODE:-offline}"
REBUILD="${REBUILD:-0}"
ULTRAGRAY_PROFILE_EVERY="${ULTRAGRAY_PROFILE_EVERY:-500}"
ULTRAGRAY_PROFILE_WARMUP="${ULTRAGRAY_PROFILE_WARMUP:-100}"

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE OUTPUT_ROOT SUMMARY_PATH
export ULTRAGRAY_CUDA_SM89="${ULTRAGRAY_CUDA_SM89:-1}"
export ULTRAGRAY_PROFILE_EVERY ULTRAGRAY_PROFILE_WARMUP
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ops/ultragray_renderer:${PYTHONPATH:-}"

output_dir="${OUTPUT_ROOT}/runs/${DATASET}/${SWEEP}/lw${LOCAL_WINDOW_MM}mm"

echo "============================================================"
echo "[INFO] UltraGray local-window accuracy/speed balance"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Sweep: ${SWEEP}"
echo "[INFO] Local window mm: ${LOCAL_WINDOW_MM}"
echo "[INFO] Output dir: ${output_dir}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs/ultragray "$(dirname "${SUMMARY_PATH}")" "${output_dir}/logs" "${output_dir}/checkpoints" "${output_dir}/vis"

if [[ "${REBUILD}" == "1" ]]; then
  echo "[INFO] Rebuilding ultragray_renderer"
  bash ops/ultragray_renderer/rebuild.sh
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

run_args=(
  run.py
  "experiment=${EXPERIMENT}"
  "hydra.run.dir=${output_dir}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_NAME}"
  "scene=${SCENE}"
  "scene.loading.sweep_name=${SWEEP}"
  "scene.loading.test_mode=holdout_interval"
  "scene.loading.holdout_trim_each=10"
  "scene.loading.holdout_num_samples=0"
  "scene.loading.holdout_test_ratio=0.2"
  "scene.loading.pose_type=absolute"
  "scene.loading.pose_mode=file"
  "scene.loading.image_downsample_factor=1"
  "scene.loading.axial_downsample_factor=1"
  "system.seed=42"
  "model.render.local_transmission_window_only=true"
  "model.render.local_transmission_max_window_mm=${LOCAL_WINDOW_MM}"
  "metric.needed_metrics=[psnr,l1,ssim,lpips,dists,gmsd]"
  "metric.max_vis_frames=10"
  "metric.save_per_frame_metrics=true"
)

echo "[INFO] Running: ${PYTHON_BIN} ${run_args[*]}"
"${PYTHON_BIN}" "${run_args[@]}"

echo "[INFO] Summarizing scalar results"
BENCHMARK="ultragray_acc_speed_balance" \
PROTOCOL="local_window_sweep" \
STAGE="preproduction" \
DATASET="${DATASET}" \
SWEEP="${SWEEP}" \
METHOD="ultragray_lw${LOCAL_WINDOW_MM}mm" \
EXP_NAME="${EXP_NAME}" \
OUTPUT_DIR="${output_dir}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
"${PYTHON_BIN}" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/summarize_run.py

echo "============================================================"
echo "[INFO] UltraGray local-window balance job finished"
echo "[INFO] Date: $(date)"
echo "============================================================"
