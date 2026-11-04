#!/bin/bash
#SBATCH -J paper1-snapshot-replay
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
OUT=$W/results/migration/2026-10-17
cd "$W/nerfstudio"
python -m nerfstudio.scripts.ultrasound_replay \
    "$OUT/gpu/replay_old_liver/resolved_config.yaml" \
    --output "$OUT/frozen_snapshot_eval" --task test \
    --checkpoint-dir "$W/checkpoints/migration/old_liver" --resume 12
python - <<'PY'
import csv,json,math
from pathlib import Path
w=Path('/home/boywang25/paper1/results/migration/2026-10-17')
def metrics(path):
    with path.open() as f:return {r['key']:float(r['value']) for r in csv.DictReader(f,delimiter='\t') if r['step']=='12' and r['key'] in ['test/psnr','test/l1','test/ssim']}
a=metrics(w/'gpu/replay_old_liver/logs/scalars.tsv');b=metrics(w/'frozen_snapshot_eval/logs/scalars.tsv')
ok=a.keys()==b.keys() and all(math.isclose(a[k],b[k],abs_tol=1e-6,rel_tol=1e-5) for k in a)
prov=json.loads((w/'frozen_snapshot_eval/replay_provenance.json').read_text())
manifest=__import__('omegaconf').OmegaConf.load(str(w/'frozen_snapshot_eval/run_manifest.yaml'))
assert manifest.checkpoint_dir!=prov['source_checkpoint_dir']
r={'old':a,'frozen_snapshot_replay':b,'passed':ok,'historical_checkpoint_write_destination_separated':True}
(w/'snapshot_replay_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
raise SystemExit(0 if ok else 1)
PY
