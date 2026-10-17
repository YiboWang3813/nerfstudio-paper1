#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH -J p05anchor
#SBATCH -o logs/ours_ablation/p05anchor_%j.out
#SBATCH -e logs/ours_ablation/p05anchor_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
# One worker at a time for an interpretable exclusive-GPU impact assay.
for dataset in liver arm; do
  if [[ "$dataset" == liver && -f outputs/preproduction/regression/p05_corrected_covariance_20260908/anchors/liver/checkpoints/iteration_30000_container.pth ]]; then
    "$PYTHON" scripts/production/ablations/ours/p05_validate_saved.py
    continue
  fi
  "$PYTHON" -u scripts/production/ablations/ours/p05_anchor.py "$dataset"
done
"$PYTHON" scripts/production/ablations/ours/p05_assay_report.py
EVIDENCE="$PROJECT/outputs/preproduction/regression/p05_corrected_covariance_20260908"
"$PYTHON" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/benchmark_render_fps.py \
  --source-tsv "$EVIDENCE/anchor_fps_input.tsv" --methods ours --max-runs 2 \
  --output-per-run "$EVIDENCE/anchor_fps.tsv" --output-summary "$EVIDENCE/anchor_fps_summary.tsv"
