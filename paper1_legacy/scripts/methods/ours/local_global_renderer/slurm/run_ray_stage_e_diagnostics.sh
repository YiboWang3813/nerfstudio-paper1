#!/usr/bin/env bash
#SBATCH -w gpu3
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J ours_rayE_diag
#SBATCH -o logs/ours/%x_%j.out
#SBATCH -e logs/ours/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
cd "${PROJECT_DIR}"

set +u
source ~/.bashrc
conda activate gaussian_splatting
module load cuda/11.8.0
set -u

export PYTHONPATH="${PROJECT_DIR}/ops/ours_renderer:${PROJECT_DIR}:${PYTHONPATH:-}"

python scripts/methods/ours/local_global_renderer/diagnose_ray_csr_gradients.py
python scripts/methods/ours/local_global_renderer/diagnose_echo_alpha_ray_autograd.py
