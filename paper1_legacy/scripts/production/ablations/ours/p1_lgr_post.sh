#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 1
#SBATCH --gres=none
#SBATCH --time=00:20:00
#SBATCH -J p1lgrreport
#SBATCH -o logs/ours_ablation/p1lgrreport_%j.out
#SBATCH -e logs/ours_ablation/p1lgrreport_%j.err
set -eo pipefail
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/ablations/ours/p1_lgr_report.py --cleanup
