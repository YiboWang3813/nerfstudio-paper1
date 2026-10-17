"""Four serial GPU workers, fixed matrix partitions, stop scheduling on any failure."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908'
PROTOCOL=ROOT/'docs/production/p1_paper_protocol_freeze'
slot=int(sys.argv[1]);slots=int(sys.argv[2])
rows=[r for r in json.loads((PROTOCOL/'batch1_matrix.json').read_text()) if r['execution_authorized']]
control=OUT/'control';control.mkdir(parents=True,exist_ok=True)
stop=control/'STOP'
for row in rows[slot::slots]:
    if stop.exists():
        raise RuntimeError('Batch stopped; inspect '+str(stop))
    target=Path(row['output_dir'])
    assert not target.exists(), 'Never overwrite '+str(target)
    logs=OUT/'worker_logs';logs.mkdir(exist_ok=True)
    start=time.time()
    print('START',row['batch1_id'],row['dataset'],row['sweep'],row['variant'],flush=True)
    with (logs/(row['batch1_id']+'.log')).open('x') as log:
        proc=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/production/ablations/ours/p1_batch1_run.py'),row['batch1_id']],stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT))
    result=dict(row_id=row['batch1_id'],exit_code=proc.returncode,
                train_start_unix=start,train_end_unix=time.time(),
                train_wall_time_sec=time.time()-start,worker=slot,
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),job=os.environ.get('SLURM_JOB_ID'))
    (control/(row['batch1_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('FINISH',json.dumps(result),flush=True)
    if proc.returncode:
        stop.write_text(json.dumps(result,indent=2)+'\n')
        raise RuntimeError('Training failed; no further tasks scheduled')
