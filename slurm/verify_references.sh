#!/bin/bash
#SBATCH -J paper1-reference-check
#SBATCH -p gpu1
#SBATCH --nodelist=gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH -o /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.out
#SBATCH -e /home/boywang25/paper1/results/migration/2026-10-17/gpu/%x_%j.err
set -eo pipefail
source /home/boywang25/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting
source /etc/profile.d/lmod.sh
module load cuda/11.8.0
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
W=/home/boywang25/paper1
OUT=$W/results/migration/2026-10-17/gpu/reference_tests
mkdir -p "$OUT"
for label in old new; do
    if [ "$label" = old ]; then root=/home/boywang25/ultrasound-scene-reconstruction/omniultra; else root=$W/nerfstudio/paper1_legacy; fi
    cd "$root"
    export PYTHONPATH="$root:$root/ops/ours_renderer"
    set +e
    # Legacy tests do not inherit Nerfstudio's unrelated xdist/typeguard plugins.
    # The PDF-tool-dependent test is executed separately on the login CPU host.
    python -m pytest -q -o addopts= --rootdir="$root" -p no:cacheprovider --import-mode=importlib \
        -k 'not test_pdf_canvas_vector_and_frozen_inputs' \
        tests/ops/ours_renderer/test_p05_covariance.py \
        tests/methods/ours/test_pruning_protection.py \
        tests/methods/ours/test_path_contribution_cuda.py \
        tests/production/test_final_ablation_tables.py \
        tests/production/test_final_hyperparameter_figures.py \
        tests/production/test_final_paper_evidence.py > "$OUT/${label}.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$OUT/${label}.exit"
    set -e
done
python - <<'PY'
import json
from pathlib import Path
p=Path('/home/boywang25/paper1/results/migration/2026-10-17/gpu/reference_tests')
result={label:int((p/(label+'.exit')).read_text()) for label in ('old','new')}
(p/'status.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
raise SystemExit(0 if not any(result.values()) else 1)
PY
