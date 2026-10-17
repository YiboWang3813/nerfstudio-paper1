#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -J p1gpu2smoke
#SBATCH -o logs/ours_ablation/p1gpu2smoke_%j.out
#SBATCH -e logs/ours_ablation/p1gpu2smoke_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python -u scripts/production/ablations/ours/p1_gpu2_smoke.py
