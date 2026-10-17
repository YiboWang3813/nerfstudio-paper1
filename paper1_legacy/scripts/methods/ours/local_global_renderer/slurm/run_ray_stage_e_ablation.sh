#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_rayE_ablate
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_echo_alpha_ray_csr_tail_2k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616}"
VARIANT="${VARIANT:-full_ray}"
EXP_NAME="${EXP_NAME:-stage_e_${VARIANT}_2k}"
WANDB_MODE="${WANDB_MODE:-offline}"

COMMON_OVERRIDES=(
  "engine.iterations=2000"
  "engine.interval.train.print=100"
  "engine.interval.test=1000"
  "engine.interval.save=2000"
  "loss.needed_losses=[l1,ssim,scale_wall]"
  "loss.lambdas.attenuation_tail=0.0"
)

case "${VARIANT}" in
  full_ray)
    VARIANT_OVERRIDES=(
      "model.slice_render.ray_attenuation_enabled=true"
      "model.slice_render.ray_attenuation_beta=0.20"
      "model.slice_render.ray_attenuation_strength=0.08"
      "model.slice_render.ray_attenuation_min_transmission=0.80"
    )
    ;;
  medium_ray)
    VARIANT_OVERRIDES=(
      "model.slice_render.ray_attenuation_enabled=true"
      "model.slice_render.ray_attenuation_beta=0.50"
      "model.slice_render.ray_attenuation_strength=0.20"
      "model.slice_render.ray_attenuation_min_transmission=0.60"
    )
    ;;
  strong_ray)
    VARIANT_OVERRIDES=(
      "model.slice_render.ray_attenuation_enabled=true"
      "model.slice_render.ray_attenuation_beta=1.00"
      "model.slice_render.ray_attenuation_strength=0.50"
      "model.slice_render.ray_attenuation_min_transmission=0.40"
    )
    ;;
  single_strength)
    VARIANT_OVERRIDES=(
      "model.slice_render.ray_attenuation_enabled=false"
      "model.slice_render.ray_attenuation_beta=0.0"
    )
    ;;
  *)
    echo "[ERROR] Unknown VARIANT=${VARIANT}. Expected single_strength, full_ray, medium_ray, or strong_ray." >&2
    exit 2
    ;;
esac

echo "============================================================"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Variant: ${VARIANT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Common overrides: ${COMMON_OVERRIDES[*]}"
echo "[INFO] Variant overrides: ${VARIANT_OVERRIDES[*]}"
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
  "${COMMON_OVERRIDES[@]}" \
  "${VARIANT_OVERRIDES[@]}"

echo "============================================================"
echo "[INFO] Finished at $(date)"
echo "============================================================"
