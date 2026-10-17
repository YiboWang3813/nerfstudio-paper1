#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -J p0pre
#SBATCH -o logs/ours_ablation/p0pre_%j.out
#SBATCH -e logs/ours_ablation/p0pre_%j.err
set -euo pipefail
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
EVIDENCE="$PROJECT/outputs/preproduction/regression/p0_semantic_freeze_20260908"
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
export PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.6+PTX"
export PYTHONPATH="$EVIDENCE/pre/tree/ops/ours_renderer:$EVIDENCE/pre/tree"
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
nvidia-smi
nvcc --version
cd "$EVIDENCE/pre/tree/ops/ours_renderer"
"$PYTHON" setup.py build_ext --inplace
cd "$PROJECT"
"$PYTHON" scripts/production/ablations/ours/p0_renderer_probe.py "$EVIDENCE/pre/renderer_probe.pt"
