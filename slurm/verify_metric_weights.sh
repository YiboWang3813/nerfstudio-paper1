#!/bin/bash
#SBATCH -J paper1-metric-check
#SBATCH -p gpu1
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH -o /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.out
#SBATCH -e /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.err
set -eo pipefail
source /home/boywang25/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
set -u
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
W=/home/boywang25/paper1
OUT=$W/results/migration/2026-10-17/gpu
for label in old new; do
    if [ "$label" = old ]; then
        root=/home/boywang25/ultrasound-scene-reconstruction/omniultra
        export TORCH_HOME=/home/boywang25/.cache/torch
    else
        root=$W/nerfstudio/paper1_legacy
        export TORCH_HOME=$W/checkpoints/pretrained/torch
    fi
    cd "$root"
    export PYTHONPATH="$root:$root/ops/ours_renderer"
    python "$W/protocol/tools/verify_metric_weights.py" --root "$root" --output "$OUT/metrics_${label}.json"
done
python - <<'PY'
import json,math
from pathlib import Path
p=Path('/home/boywang25/paper1/results/migration/2026-10-17/gpu')
a=json.loads((p/'metrics_old.json').read_text());b=json.loads((p/'metrics_new.json').read_text())
ok=a.keys()==b.keys() and all(math.isclose(a[k],b[k],abs_tol=1e-7,rel_tol=1e-6) for k in a)
r={'old':a,'new':b,'passed':ok,'downloads_allowed':False}
(p/'metric_weight_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
raise SystemExit(0 if ok else 1)
PY
