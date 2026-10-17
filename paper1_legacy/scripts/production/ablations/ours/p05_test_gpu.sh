#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -J p05test
#SBATCH -o logs/ours_ablation/p05test_%j.out
#SBATCH -e logs/ours_ablation/p05test_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONDONTWRITEBYTECODE=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" MAX_JOBS=8
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT"
cd "$PROJECT/ops/ours_renderer"
if [[ "${P05_SKIP_BUILD:-0}" != 1 ]]; then "$PYTHON" setup.py build_ext --inplace --force; fi
cd "$PROJECT"
nvidia-smi
nvcc --version
"$PYTHON" -m pytest --import-mode=importlib -q -s -p no:cacheprovider \
 tests/methods/ours/test_p0_semantics.py tests/methods/ours/test_pruning_protection.py \
 tests/methods/ours/test_path_contribution_cuda.py tests/ops/ours_renderer/test_p0_local_modes.py \
 tests/ops/ours_renderer/test_p05_covariance.py tests/ops/ours_renderer/validate_us_focus_reference.py
"$PYTHON" tests/ops/ours_renderer/validate_plane_cond2d.py
# Same six bounded real-data smokes, isolated from preserved P0 evidence.
sed -n '/^for dataset in liver arm/,$p' scripts/production/ablations/ours/p0_post_gpu.sh \
 | sed 's/p0_semantic_freeze_20260908/p05_corrected_covariance_20260908/g' | bash
