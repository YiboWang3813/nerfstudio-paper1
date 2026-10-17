#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_rayE_effect
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_full_ray_2k}"
OUT_JSON="${OUT_JSON:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_integral_stage_e_ablation_20260616/stage_e_full_ray_2k/ray_effect_diagnostic.json}"
ITERATION="${ITERATION:-2000}"
NUM_FRAMES="${NUM_FRAMES:-8}"

echo "============================================================"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Out json: ${OUT_JSON}"
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

python scripts/methods/ours/local_global_renderer/diagnose_ray_effect_checkpoint.py \
  --run-dir "${RUN_DIR}" \
  --iteration "${ITERATION}" \
  --num-frames "${NUM_FRAMES}" \
  --out-json "${OUT_JSON}"

echo "============================================================"
echo "[INFO] Finished at $(date)"
echo "============================================================"
