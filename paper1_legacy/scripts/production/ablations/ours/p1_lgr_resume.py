"""Resume only never-started LGR rows after a verified safe worker drain."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.production.ablations.ours.p1_lgr_confirmation import DOC,OUT,ROOT,guard


def plan():
    guard()
    state=subprocess.check_output(['sacct','-n','-P','-j','80077','--format=JobID,State'],universal_newlines=True)
    top=next(line for line in state.splitlines() if line.startswith('80077|'))
    assert top.split('|')[1] in ('FAILED','COMPLETED'),top
    matrix=json.loads((DOC/'lgr_multisweep_structural_matrix.json').read_text());done=[];remaining=[]
    for r in matrix:
        if not r['execution_authorized']:continue
        control=OUT/'control'/(r['batch1_id']+'.json')
        if control.exists():
            c=json.loads(control.read_text());assert c['exit_code']==0,c
            assert json.loads((Path(r['output_dir'])/'batch1_stats.json').read_text())['success'];done.append(r['batch1_id'])
        else:
            assert not Path(r['output_dir']).exists(),r['output_dir']
            assert not (OUT/'worker_logs'/(r['batch1_id']+'.log')).exists()
            remaining.append(r)
    assert len(done)+len(remaining)==36
    # Use prior same-variant representative run costs to balance work, not quality.
    prior=list(__import__('csv').DictReader((ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports/lgr_structural.tsv').open(),delimiter='\t'))
    estimates={(r['dataset'],r['label']):float(r['train_wall_time_sec']) for r in prior}
    bins=[[] for _ in range(4)];loads=[0.]*4
    for r in sorted(remaining,key=lambda r:estimates[r['dataset'],r['variant']],reverse=True):
        slot=min(range(4),key=lambda s:loads[s]);bins[slot].append(r['batch1_id']);loads[slot]+=estimates[r['dataset'],r['variant']]
    assert len({x for b in bins for x in b})==len(remaining)
    value=dict(previous_job=80077,completed_before_rebalance=done,slots=bins,estimated_seconds=loads,total_authorized_new=36)
    (DOC/'resume_partition.json').write_text(json.dumps(value,indent=2)+'\n')
    files=[DOC/'resume_partition.json',Path(__file__).resolve(),ROOT/'scripts/production/ablations/ours/p1_lgr_resume_gpu.sh']
    (DOC/'resume_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print(json.dumps(value),flush=True)


def worker(slot):
    guard()
    for p,h in json.loads((DOC/'resume_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    ids=json.loads((DOC/'resume_partition.json').read_text())['slots'][slot]
    matrix={r['batch1_id']:r for r in json.loads((DOC/'lgr_multisweep_structural_matrix.json').read_text())}
    control=OUT/'control';logs=OUT/'worker_logs'
    for row_id in ids:
        assert not (control/'RESUME_STOP').exists()
        r=matrix[row_id];assert not Path(r['output_dir']).exists();assert not (control/(row_id+'.json')).exists()
        start=time.time();print('START',row_id,flush=True)
        with (logs/(row_id+'.log')).open('x') as f:
            p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/production/ablations/ours/p1_lgr_confirmation.py'),'run',row_id],stdout=f,stderr=subprocess.STDOUT)
        result=dict(row_id=row_id,exit_code=p.returncode,worker=slot,job=os.environ.get('SLURM_JOB_ID'),train_start_unix=start,train_end_unix=time.time(),train_wall_time_sec=time.time()-start)
        (control/(row_id+'.json')).write_text(json.dumps(result,indent=2)+'\n');print('FINISH',result,flush=True)
        if p.returncode:
            (control/'RESUME_STOP').write_text(json.dumps(result));raise RuntimeError('No more runs after failure')


if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    else:worker(int(sys.argv[2]))
