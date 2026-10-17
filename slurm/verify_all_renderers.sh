#!/bin/bash
#SBATCH -J paper1-all-renderers
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
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
W=/home/boywang25/paper1
OUT=$W/results/migration/2026-10-17/gpu/all_renderers
mkdir -p "$OUT"
for label in old new; do
    if [ "$label" = old ]; then root=/home/boywang25/ultrasound-scene-reconstruction/omniultra; else root=$W/nerfstudio/paper1_legacy; fi
    cd "$root"
    export PYTHONPATH="$root:$root/ops/ours_renderer:$root/ops/ultragauss_renderer:$root/ops/ultragray_renderer:$root/ops/gaussianplie_renderer:$root/ops/slice_adapted_3dgs_renderer:$root/ops/ultrags_baseline_renderer:$root/ops/ultrags_renderer"
    set +e
    python -m pytest -q -o addopts= --rootdir="$root" --import-mode=importlib -p no:cacheprovider \
        tests/ops/ultragauss_renderer/test_forward_reference.py \
        tests/ops/ultragray_renderer/test_renderer_smoke.py \
        tests/ops/gaussianplie_renderer/test_reference_math.py \
        tests/ops/gaussianplie_renderer/test_reference_single_alpha.py \
        tests/ops/gaussianplie_renderer/test_additive_property.py \
        tests/ops/gaussianplie_renderer/test_cuda_reference_alignment.py \
        tests/ops/gaussianplie_renderer/test_cuda_wrapper.py \
        tests/ops/slice_adapted_3dgs_renderer/test_additive_splat_cuda.py \
        tests/ops/ultrags_baseline_renderer/test_import.py \
        tests/ops/ultrags_baseline_renderer/test_forward_shapes.py \
        tests/ops/ultrags_renderer/test_batched_equivalence.py > "$OUT/${label}.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$OUT/${label}.exit"
    set -e
done
python - <<'PY'
import json
from pathlib import Path
p=Path('/home/boywang25/paper1/results/migration/2026-10-17/gpu/all_renderers')
r={label:int((p/(label+'.exit')).read_text()) for label in ('old','new')}
(p/'status.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
raise SystemExit(0 if not any(r.values()) else 1)
PY
