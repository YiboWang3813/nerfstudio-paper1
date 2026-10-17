#!/bin/bash
#SBATCH -p gpu1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -J p0check
#SBATCH -o logs/ours_ablation/p0check_%j.out
#SBATCH -e logs/ours_ablation/p0check_%j.err
set -euo pipefail
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
EVIDENCE="$PROJECT/outputs/preproduction/regression/p0_semantic_freeze_20260908"
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PROJECT EVIDENCE PYTHON
cd "$PROJECT"
PYTHONPATH="$EVIDENCE/pre/tree/ops/ours_renderer:$EVIDENCE/pre/tree" \
  "$PYTHON" scripts/production/ablations/ours/p0_renderer_probe.py "$EVIDENCE/pre/renderer_probe.pt"
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT"
"$PYTHON" scripts/production/ablations/ours/p0_renderer_probe.py "$EVIDENCE/post_renderer_probe.pt"
"$PYTHON" scripts/production/ablations/ours/p0_compare.py
"$PYTHON" -m pytest --import-mode=importlib -q -s -p no:cacheprovider \
  tests/methods/ours/test_p0_semantics.py tests/methods/ours/test_pruning_protection.py \
  tests/methods/ours/test_path_contribution_cuda.py tests/ops/ours_renderer/test_p0_local_modes.py \
  tests/ops/ours_renderer/validate_us_focus_reference.py
# Reuse the audited smoke-only tail, without rebuilding or running a formal matrix.
sed -n '/^for dataset in liver arm/,$p' scripts/production/ablations/ours/p0_post_gpu.sh | bash
