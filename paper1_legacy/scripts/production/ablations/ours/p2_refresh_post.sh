#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 1
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH -J p2report
#SBATCH -o logs/ours_ablation/p2report_%j.out
#SBATCH -e logs/ours_ablation/p2report_%j.err
set -eo pipefail
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
export PYTHONPATH=ops/ours_renderer:. PYTHONDONTWRITEBYTECODE=1
/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/ablations/ours/p2_refresh_report.py
