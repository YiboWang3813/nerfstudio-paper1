#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J ours_ray_csr_c
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/ideas/ours/local_global_renderer/tus_rec_2024_qc_separated_attenuation_ray_csr_tail_2k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/ideas/ours/local_global_renderer/ours_ray_csr_stage_c_20260616}"
EXP_NAME="${EXP_NAME:-${SLURM_JOB_NAME:-run}_${SLURM_JOB_ID:-manual}}"
OVERRIDES="${OVERRIDES:-engine.iterations=200 engine.interval.train.print=50 engine.interval.test=200 engine.interval.save=200}"
WANDB_MODE="${WANDB_MODE:-offline}"

echo "============================================================"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overrides: ${OVERRIDES}"
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
  ${OVERRIDES}

echo "============================================================"
echo "[INFO] Finished at $(date)"
echo "============================================================"
