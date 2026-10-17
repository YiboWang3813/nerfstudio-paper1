#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 1
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH -J p1cleanupfinal
#SBATCH -o logs/ours_ablation/p1cleanupfinal_%j.out
#SBATCH -e logs/ours_ablation/p1cleanupfinal_%j.err
set -e
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/ablations/ours/p1_checkpoint_cleanup.py --apply
