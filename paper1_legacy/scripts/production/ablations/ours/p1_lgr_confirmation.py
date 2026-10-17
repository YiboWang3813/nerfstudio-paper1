"""Plan/execute only the authorized 11-case LGR confirmation; frozen training unchanged."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/production/p1_lgr_confirmation'
OUT=ROOT/'outputs/production/ablations/ours/paper_final_corrected_covariance_20260908/lgr_confirmation'
OLD=ROOT/'docs/production/p1_paper_protocol_freeze'
SHA='b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def guard():
    f=ROOT/'outputs/preproduction/regression/p05_corrected_covariance_20260908/freeze_v2'
    for name in ('hashes.json','binary_hashes.json'):
        for p,h in json.loads((f/name).read_text()).items():
            if not p.startswith(('scripts/','docs/','tests/')):
                assert sha(ROOT/p)==h,p


def plan():
    from omegaconf import OmegaConf
    from configs.setup import load_hydra_config
    from scripts.production.ablations.ours.p0_reuse import semantic
    guard();DOC.mkdir(exist_ok=True);(DOC/'exact_configs').mkdir(exist_ok=True)
    old=json.loads((OLD/'batch1_matrix.json').read_text())
    selected=[(d,s) for d,s in json.loads((OLD/'protocol_summary.json').read_text())['selected_multi_sweep'] if d!='tdsc_abus_2023']
    assert len(selected)==11
    byhash={r['semantic_config_sha256']:r for r in old}
    rows=[]
    for ds,sw in selected:
        base=next(r for r in old if r['group']=='main' and r['dataset']==ds and r['sweep']==sw)
        for label,local,glob in [('Plane','none','none'),('Fixed','fixed','none'),('Adaptive','adaptive','none'),('Simple','adaptive','slice_depth'),('Full','adaptive','integral')]:
            row=copy.deepcopy(base);row['batch1_id']='lc_{:03d}'.format(len(rows));row['variant']=label;row['group']='structural'
            row['settings'].update(local=local,global_mode=glob,prune='local_aware' if glob=='none' else 'path_aware')
            changes={'model.slice_render.local_conditioning_mode':local,
                     'model.slice_render.ray_attenuation_enabled':'false' if glob=='none' else 'true',
                     'model.slice_render.ray_attenuation_formula':'integral' if glob=='none' else glob,
                     'model.offline.density_control.pruning_policy':row['settings']['prune']}
            row['overrides']=[v for v in row['overrides'] if v.lstrip('+').split('=')[0] not in changes]+[k+'='+v for k,v in changes.items()]
            cfg=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',row['overrides']),resolve=True)
            eff=semantic(cfg);eff['metric']=cfg['metric'];digest=hashlib.sha256(json.dumps(eff,sort_keys=True).encode()).hexdigest()
            row['semantic_config_sha256']=digest;row['source_freeze_sha256']=SHA
            row['consumers']=['LGR_confirmation/'+label]
            reused=byhash.get(digest)
            row['execution_authorized']=reused is None;row['target_status']='NEW' if reused is None else 'FINAL-REUSE'
            row['reuse_batch1_id']=reused['batch1_id'] if reused else None
            row['output_dir']=reused['output_dir'] if reused else str(OUT/'runs'/ds/sw/row['batch1_id'])
            if reused:
                actual=OmegaConf.to_container(OmegaConf.load(Path(reused['output_dir'])/'resolved_config.yaml'),resolve=False)
                actual_eff=semantic(actual);actual_eff['metric']=actual['metric']
                assert actual_eff==eff,(ds,sw,label)
                stats=Path(reused['output_dir'])/('anchor_stats.json' if reused['target_status']=='FINAL-REUSE' else 'batch1_stats.json')
                assert json.loads(stats.read_text())['success']
            OmegaConf.save(OmegaConf.create(cfg),DOC/'exact_configs'/(row['batch1_id']+'.yaml'))
            rows.append(row)
    assert len(rows)==55 and sum(r['execution_authorized'] for r in rows)==36
    assert len({r['semantic_config_sha256'] for r in rows})==55
    (DOC/'lgr_multisweep_structural_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Reuse the exact guarded Batch1 runner; only protocol path and manifest label differ.
    original=ROOT/'scripts/production/ablations/ours/p1_batch1_run.py'
    sealed=json.loads((OLD/'submission_hashes.json').read_text())
    assert sha(original)==sealed[str(original.relative_to(ROOT))]
    source=original.read_text().replace("PROTOCOL = ROOT/'docs/production/p1_paper_protocol_freeze'","PROTOCOL = ROOT/'docs/production/p1_lgr_confirmation'").replace("PROTOCOL/'batch1_matrix.json'","PROTOCOL/'lgr_multisweep_structural_matrix.json'").replace("'Batch1_only'","'LGR_confirmation_only'")
    (DOC/'executed_runner.py').write_text(source)
    files=list((DOC/'exact_configs').glob('*.yaml'))+[DOC/'lgr_multisweep_structural_matrix.json',DOC/'executed_runner.py',Path(__file__).resolve(),ROOT/'scripts/production/ablations/ours/p1_lgr_gpu.sh']
    (DOC/'submission_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
    print('55 slots: 19 exact-config reuse, 36 NEW; no Batch2',flush=True)


def run(row_id):
    guard()
    code=DOC/'executed_runner.py'
    for p,h in json.loads((DOC/'submission_hashes.json').read_text()).items():assert sha(ROOT/p)==h,p
    # Preserve original runner __file__ root resolution without rewriting production.
    sys.argv=[str(code),row_id]
    exec(compile(code.read_text(),str(code),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/production/ablations/ours/p1_batch1_run.py')})


def worker(slot,slots):
    rows=[r for r in json.loads((DOC/'lgr_multisweep_structural_matrix.json').read_text()) if r['execution_authorized']]
    control=OUT/'control';logs=OUT/'worker_logs';control.mkdir(parents=True,exist_ok=True);logs.mkdir(exist_ok=True)
    for row in rows[slot::slots]:
        assert not (control/'STOP').exists()
        assert not Path(row['output_dir']).exists()
        start=time.time();print('START',row['batch1_id'],flush=True)
        with (logs/(row['batch1_id']+'.log')).open('x') as log:
            p=subprocess.run([sys.executable,'-u',str(Path(__file__)),'run',row['batch1_id']],stdout=log,stderr=subprocess.STDOUT)
        result=dict(row_id=row['batch1_id'],exit_code=p.returncode,worker=slot,job=os.environ.get('SLURM_JOB_ID'),train_start_unix=start,train_end_unix=time.time(),train_wall_time_sec=time.time()-start)
        (control/(row['batch1_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('FINISH',result,flush=True)
        if p.returncode:
            (control/'STOP').write_text(json.dumps(result));raise RuntimeError('Stop on failed run')


if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    elif sys.argv[1]=='run':run(sys.argv[2])
    elif sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
