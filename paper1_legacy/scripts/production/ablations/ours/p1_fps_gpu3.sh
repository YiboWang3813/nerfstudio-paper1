#!/bin/bash
#SBATCH -p gpu1
#SBATCH -w gpu3
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -J p1fairfps
#SBATCH -o logs/ours_ablation/p1fairfps_%j.out
#SBATCH -e logs/ours_ablation/p1fairfps_%j.err
set -eo pipefail
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
PROJECT=/home/boywang25/ultrasound-scene-reconstruction/omniultra
export PYTHONPATH="$PROJECT/ops/ours_renderer:$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
PYTHON=/home/boywang25/miniconda3/envs/gaussian_splatting/bin/python
"$PYTHON" -c 'import torch; p=torch.cuda.get_device_properties(0); print(p); assert "4090" in p.name'
"$PYTHON" -c 'import hashlib,json,pathlib; f=pathlib.Path("outputs/preproduction/regression/p05_corrected_covariance_20260908/freeze_v2"); h=json.loads((f/"hashes.json").read_text()); h.update(json.loads((f/"binary_hashes.json").read_text())); assert all(hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==v for n,v in h.items() if not n.startswith(("docs/","scripts/","tests/")))'
REPORT=outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports
"$PYTHON" scripts/production/comparisons/us_recon_four_dataset_p100k_v1/benchmark_render_fps.py \
  --source-tsv "$REPORT/main_fps_input.tsv" --methods ours --isolate-subprocesses \
  --output-per-run "$REPORT/corrected_main_fps.tsv" --output-summary "$REPORT/corrected_main_fps_summary.tsv"
"$PYTHON" scripts/production/ablations/ours/p1_fps_report.py
