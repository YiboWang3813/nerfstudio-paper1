#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -J fixedfps
#SBATCH -o logs/ours_ablation/fixedfps_%j.out
#SBATCH -e logs/ours_ablation/fixedfps_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
export PYTHONPATH=ops/ours_renderer:. PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -c 'import torch; p=torch.cuda.get_device_properties(0);print(p);assert "4090" in p.name'
"$PYTHON" scripts/production/ablations/ours/p2_fps_rebuild_report.py prepare
REPORT=outputs/production/comparisons/P2_fixed_full_final_20260909/fps_rebuild/reports
"$PYTHON" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/benchmark_render_fps.py \
 --source-tsv "$REPORT/fps_input.tsv" --methods ours --isolate-subprocesses \
 --warmup-frames 5 --timed-frames 20 --repeats 3 \
 --output-per-run "$REPORT/fixed_fps_per_run.tsv" --output-summary "$REPORT/fixed_fps_summary.tsv"
"$PYTHON" scripts/production/ablations/ours/p2_fps_rebuild_report.py aggregate
