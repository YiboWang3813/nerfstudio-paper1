"""Bounded 100-iteration, 1M GPU2 memory gate; never paper metrics."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PROTOCOL=ROOT/'docs/production/p1_paper_protocol_freeze'
OUT=ROOT/'outputs/preproduction/regression/p1_gpu2_memory_gate_20260908'
FREEZE=ROOT/'outputs/preproduction/regression/p05_corrected_covariance_20260908/freeze_v2'
for name,expected in json.loads((FREEZE/'hashes.json').read_text()).items():
    if not name.startswith(('docs/','scripts/','tests/')):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
for name,expected in json.loads((FREEZE/'binary_hashes.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
rows=json.loads((PROTOCOL/'batch1_matrix.json').read_text())
cases=list(csv.DictReader((PROTOCOL/'paper_final_dataset_protocol.tsv').open(),delimiter='\t'))
selected=[]
for ds in sorted({r['dataset'] for r in rows if r['group']=='main'}):
    case=max((r for r in cases if r['dataset']==ds and r['include_in_main']=='1'),key=lambda r:int(r['num_frames']))
    selected.append(next(r for r in rows if r['group']=='main' and r['dataset']==ds and r['sweep']==case['case_id']))
selected += [r for r in rows if r['settings']['local']=='none']
assert len(selected)==6
for row in selected:
    target=OUT/row['batch1_id']
    assert not target.exists(),target
    args=row['overrides']+['engine.iterations=100','engine.interval.test=100','engine.interval.save=100',
        'model.offline.density_control.timing.start_iter=1',
        'model.offline.density_control.timing.end_iter=100',
        'model.offline.density_control.timing.freq=50',
        'hydra.run.dir='+str(target),'system.exp_tag=p1_gpu2_memory_gate']
    # Same existing P0 instrumentation, no changes to training functions.
    env=dict(os.environ,P0_SMOKE_STATS=str(target/'smoke_stats.json'))
    print('SMOKE_START',row['batch1_id'],row['dataset'],row['sweep'],row['variant'],flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/production/ablations/ours/p0_smoke.py'),
        'experiment=production/ablations/ours/paper_semantic_freeze']+args,env=env,check=True)
    print('SMOKE_PASS',row['batch1_id'],flush=True)
