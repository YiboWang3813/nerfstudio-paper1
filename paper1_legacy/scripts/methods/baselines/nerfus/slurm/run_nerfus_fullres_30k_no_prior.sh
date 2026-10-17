#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J nerfus_30k
#SBATCH -o logs/nerfus/%x_%j.out
#SBATCH -e logs/nerfus/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/nerfus/fullres_30k_no_prior}"
OVERRIDES="${OVERRIDES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/preproduction/baseline_reproduction/nerfus/no_prior_l1}"
EXP_NAME="${EXP_NAME:-${SLURM_JOB_NAME:-nerfus_30k}_${SLURM_JOB_ID:-manual}}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "============================================================"
echo "[INFO] Slurm job started"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Job name: ${SLURM_JOB_NAME:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Experiment: ${EXPERIMENT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overrides: ${OVERRIDES:-<none>}"
echo "[INFO] WANDB_MODE: ${WANDB_MODE}"
echo "============================================================"

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
set -u

echo "[INFO] Python: $(which python)"
python - <<'PY'
import sys
print("[INFO] Python version:", sys.version)
try:
    import torch
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)
    print("[INFO] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] cuda device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("[WARN] torch check failed:", repr(exc))
PY
nvidia-smi || true

cd "${PROJECT_DIR}"

RUN_ARGS=(
  run.py
  "experiment=${EXPERIMENT}"
  "task=train"
  "hydra.run.dir=${OUTPUT_ROOT}/${EXP_NAME}"
  "system.output_root=${OUTPUT_ROOT}"
  "system.exp_tag=${EXP_NAME}"
)

if [[ -n "${OVERRIDES}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi

python "${RUN_ARGS[@]}"

echo "============================================================"
echo "[INFO] Slurm job finished successfully"
echo "[INFO] Date: $(date)"
echo "============================================================"
