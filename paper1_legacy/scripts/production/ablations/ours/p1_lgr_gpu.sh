#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu2
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH -J p1lgrconfirm
#SBATCH -o logs/ours_ablation/p1lgrconfirm_%j.out
#SBATCH -e logs/ours_ablation/p1lgrconfirm_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
"$PYTHON" -c 'import torch; assert torch.cuda.device_count()==2; ds=[torch.cuda.get_device_properties(i) for i in range(2)];print(ds);assert all("4080" in d.name and d.total_memory>=15*1024**3 for d in ds)'
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#GPUS[@]} == 2 ]]
pids=()
for slot in 0 1; do
  CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" "$PYTHON" -u scripts/production/ablations/ours/p1_lgr_confirmation.py worker "$slot" 2 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
