#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH -J fixedcandidate
#SBATCH -o logs/ours_ablation/fixedcandidate_%j.out
#SBATCH -e logs/ours_ablation/fixedcandidate_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONPATH=ops/ours_renderer:. PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -c 'import torch,os;ds=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())];print(ds);assert len(ds)==2 and all(os.environ["EXPECTED_GPU"] in d.name for d in ds)'
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#GPUS[@]} == 2 ]]
pids=()
for local_slot in 0 1; do
  slot=$((SLOT_OFFSET+local_slot))
  CUDA_VISIBLE_DEVICES="${GPUS[$local_slot]}" "$PYTHON" -u scripts/production/ablations/ours/p1_fixed_candidate.py worker "$slot" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
