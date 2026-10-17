#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 24
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH -J p1core
#SBATCH -o logs/ours_ablation/p1core_%j.out
#SBATCH -e logs/ours_ablation/p1core_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#GPUS[@]} == 4 ]]
pids=()
for slot in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" "$PYTHON" -u scripts/production/ablations/ours/p1_batch1_worker.py "$slot" 4 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
