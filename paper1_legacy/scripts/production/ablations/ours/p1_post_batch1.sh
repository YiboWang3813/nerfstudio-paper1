#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 1
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH -J p1aggregate
#SBATCH -o logs/ours_ablation/p1aggregate_%j.out
#SBATCH -e logs/ours_ablation/p1aggregate_%j.err
set -eo pipefail
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
"$PYTHON" scripts/production/ablations/ours/p1_checkpoint_report.py
"$PYTHON" scripts/production/ablations/ours/p1_scientific_checkpoint.py
