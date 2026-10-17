#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --array=0-26%4
#SBATCH -J ours_rayE_sweep
#SBATCH -o logs/ours/%x_%A_%a.out
#SBATCH -e logs/ours/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_echo_alpha_ray_csr_tail_2k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_fine_sweep_20260616}"
WANDB_MODE="${WANDB_MODE:-offline}"

BETAS=(0.40 0.50 0.60)
STRENGTHS=(0.15 0.20 0.30)
MIN_TRANSMISSIONS=(0.55 0.60 0.70)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_MIN_T=${#MIN_TRANSMISSIONS[@]}
NUM_STRENGTH=${#STRENGTHS[@]}

BETA_INDEX=$(( TASK_ID / (NUM_STRENGTH * NUM_MIN_T) ))
REST=$(( TASK_ID % (NUM_STRENGTH * NUM_MIN_T) ))
STRENGTH_INDEX=$(( REST / NUM_MIN_T ))
MIN_T_INDEX=$(( REST % NUM_MIN_T ))

BETA="${BETAS[$BETA_INDEX]}"
STRENGTH="${STRENGTHS[$STRENGTH_INDEX]}"
MIN_T="${MIN_TRANSMISSIONS[$MIN_T_INDEX]}"
EXP_NAME="stage_e_fine_b${BETA}_s${STRENGTH}_t${MIN_T}_2k"
EXP_NAME="${EXP_NAME//./p}"

OVERRIDES=(
  "engine.iterations=2000"
  "engine.interval.train.print=100"
  "engine.interval.test=1000"
  "engine.interval.save=2000"
  "loss.needed_losses=[l1,ssim,scale_wall]"
  "loss.lambdas.attenuation_tail=0.0"
  "model.slice_render.ray_attenuation_enabled=true"
  "model.slice_render.ray_attenuation_beta=${BETA}"
  "model.slice_render.ray_attenuation_strength=${STRENGTH}"
  "model.slice_render.ray_attenuation_min_transmission=${MIN_T}"
)

echo "============================================================"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Array task: ${TASK_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] beta=${BETA} strength=${STRENGTH} minT=${MIN_T}"
echo "[INFO] Overrides: ${OVERRIDES[*]}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u

python - <<'PY'
import torch
print("[INFO] torch:", torch.__version__)
print("[INFO] torch cuda:", torch.version.cuda)
print("[INFO] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] cuda device:", torch.cuda.get_device_name(0))
PY

nvidia-smi
cd "${PROJECT_DIR}"

python run.py \
  experiment="${EXPERIMENT}" \
  hydra.run.dir="${OUTPUT_ROOT}/${EXP_NAME}" \
  system.output_root="${OUTPUT_ROOT}" \
  system.exp_tag="${EXP_NAME}" \
  "${OVERRIDES[@]}"

echo "============================================================"
echo "[INFO] Finished at $(date)"
echo "============================================================"
