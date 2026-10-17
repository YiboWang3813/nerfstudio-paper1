#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -J p1probe4080
#SBATCH -o logs/ours_ablation/p1probe4080_%j.out
#SBATCH -e logs/ours_ablation/p1probe4080_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
hostname
nvidia-smi
sha256sum ops/ours_renderer/ours_renderer/_C.cpython-37m-x86_64-linux-gnu.so
"$PYTHON" -m pytest --import-mode=importlib -q -p no:cacheprovider \
  tests/ops/ours_renderer/test_p05_covariance.py \
  tests/ops/ours_renderer/test_p0_local_modes.py
