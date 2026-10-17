#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH -J p1nodecheck
#SBATCH -o logs/ours_ablation/p1nodecheck_%j.out
#SBATCH -e logs/ours_ablation/p1nodecheck_%j.err
set -e
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
