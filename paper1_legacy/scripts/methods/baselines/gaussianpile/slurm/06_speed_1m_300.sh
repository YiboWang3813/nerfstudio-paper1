#!/bin/bash
#SBATCH -p gpu2
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J gpile_speed
#SBATCH -o logs/gaussianpile/%x_%j.out
#SBATCH -e logs/gaussianpile/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:-preproduction/baseline_reproduction/gaussianpile/speed}"
EXP_TAG="${EXP_TAG:-gaussianpile_liver_1m_300_speed}"
TIME_FILE="${TIME_FILE:-${PROJECT_DIR}/checkpoints/gaussianpile/${EXP_TAG}.time}"
OVERRIDES="${OVERRIDES:-}"
export CUDA_MODULE
export TIME_FILE
export SLURM_SCRIPT="${BASH_SOURCE[0]}"

echo "[INFO] job=${SLURM_JOB_ID:-N/A} host=$(hostname) date=$(date)"
echo "[INFO] experiment=${EXPERIMENT} exp_tag=${EXP_TAG} time_file=${TIME_FILE}"
set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"
set -u
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/ops/gaussianplie_renderer:${PYTHONPATH:-}"
mkdir -p "$(dirname "${TIME_FILE}")"
start_sec="$(date +%s)"
RUN_ARGS=(run.py "experiment=${EXPERIMENT}" "system.exp_tag=${EXP_TAG}")
if [[ -n "${OVERRIDES}" ]]; then
  read -r -a OVERRIDE_ARRAY <<< "${OVERRIDES}"
  RUN_ARGS+=("${OVERRIDE_ARRAY[@]}")
fi
WANDB_MODE="${WANDB_MODE:-offline}" python "${RUN_ARGS[@]}"
end_sec="$(date +%s)"
echo "$((end_sec - start_sec))" > "${TIME_FILE}"

python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["TIME_FILE"])
if path.exists():
    seconds = float(path.read_text().strip())
    print(f"[INFO] wall_seconds={seconds:.6f}")
    print(f"[INFO] wall_iter_per_second={300.0 / seconds:.6f}")
PY
