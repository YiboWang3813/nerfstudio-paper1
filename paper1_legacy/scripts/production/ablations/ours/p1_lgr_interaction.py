"""Submission-only 2x3 interaction study: 44 reuse + 22 Fixed/global runs."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.production.ablations.ours import p1_lgr_confirmation as base

ROOT=base.ROOT
DOC=ROOT/'docs/production/p1_lgr_interaction'
OUT=ROOT/'outputs/production/ablations/ours/paper_final_corrected_covariance_20260908/lgr_interaction'


def plan():
    from omegaconf import OmegaConf
    from configs.setup import load_hydra_config
    from scripts.production.ablations.ours.p0_reuse import semantic
    base.guard();DOC.mkdir(exist_ok=True);(DOC/'exact_configs').mkdir(exist_ok=True)
    assert not (DOC/'interaction_matrix.json').exists(),'Do not overwrite a sealed plan'
    old=json.loads((base.DOC/'lgr_multisweep_structural_matrix.json').read_text())
    known={r['semantic_config_sha256']:r for r in old}
    cases=sorted({(r['dataset'],r['sweep']) for r in old});assert len(cases)==11
    rows=[]
    for ds,sw in cases:
        for local in ('fixed','adaptive'):
            for glob,label in [('none','None'),('slice_depth','Simple'),('integral','Full')]:
                template=next(r for r in old if r['dataset']==ds and r['sweep']==sw and r['settings']['local']=='adaptive' and r['settings']['global_mode']==glob)
                row=copy.deepcopy(template);row['batch1_id']='li_{:03d}'.format(len(rows));row['variant']=local.title()+'+'+label
                row['settings']['local']=local
                row['overrides']=[x for x in row['overrides'] if x.lstrip('+').split('=')[0]!='model.slice_render.local_conditioning_mode']+['model.slice_render.local_conditioning_mode='+local]
                cfg=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',row['overrides']),resolve=True)
                effective=semantic(cfg);effective['metric']=cfg['metric']
                digest=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest();row['semantic_config_sha256']=digest
                reused=known.get(digest);row['execution_authorized']=reused is None;row['target_status']='NEW' if reused is None else 'FINAL-REUSE'
                row['reuse_confirmation_id']=reused['batch1_id'] if reused else None
                row['consumers']=['LGR_interaction/'+row['variant']]
                row['output_dir']=reused['output_dir'] if reused else str(OUT/'runs'/ds/sw/row['batch1_id'])
                row['source_freeze_sha256']=base.SHA
                ref=Path(template['output_dir'])/'batch1_stats.json'
                if not ref.exists():ref=Path(template['output_dir'])/'anchor_stats.json'
                stats=json.loads(ref.read_text());row['estimated_seconds']=stats.get('wrapper_wall_seconds',stats.get('wall_seconds',600))
                if reused:
                    actual=OmegaConf.to_container(OmegaConf.load(Path(reused['output_dir'])/'resolved_config.yaml'),resolve=False)
                    eff=semantic(actual);eff['metric']=actual['metric'];assert eff==effective
                    p=Path(reused['output_dir'])/'batch1_stats.json'
                    if not p.exists():p=Path(reused['output_dir'])/'anchor_stats.json'
                    assert json.loads(p.read_text())['success']
                else:
                    assert local=='fixed' and glob in ('slice_depth','integral')
                    assert not Path(row['output_dir']).exists()
                OmegaConf.save(OmegaConf.create(cfg),DOC/'exact_configs'/(row['batch1_id']+'.yaml'))
                rows.append(row)
    assert len(rows)==66 and len({r['semantic_config_sha256'] for r in rows})==66
    new=[r for r in rows if r['execution_authorized']];assert len(new)==22
    bins=[[] for _ in range(4)];loads=[0.]*4
    for r in sorted(new,key=lambda r:r['estimated_seconds'],reverse=True):
        slot=min(range(4),key=lambda s:loads[s]);bins[slot].append(r['batch1_id']);loads[slot]+=r['estimated_seconds']
    (DOC/'interaction_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
    (DOC/'worker_partition.json').write_text(json.dumps(dict(slots=bins,estimated_seconds=loads),indent=2)+'\n')
    original=ROOT/'scripts/production/ablations/ours/p1_batch1_run.py'
    sealed=json.loads((base.OLD/'submission_hashes.json').read_text());assert base.sha(original)==sealed[str(original.relative_to(ROOT))]
    code=original.read_text().replace("PROTOCOL = ROOT/'docs/production/p1_paper_protocol_freeze'","PROTOCOL = ROOT/'docs/production/p1_lgr_interaction'").replace("PROTOCOL/'batch1_matrix.json'","PROTOCOL/'interaction_matrix.json'").replace("'Batch1_only'","'LGR_interaction_only'")
    (DOC/'executed_runner.py').write_text(code)
    files=list((DOC/'exact_configs').glob('*.yaml'))+[DOC/'interaction_matrix.json',DOC/'worker_partition.json',DOC/'executed_runner.py',Path(__file__).resolve(),ROOT/'scripts/production/ablations/ours/p1_lgr_interaction_gpu.sh',Path(base.__file__).resolve()]
    (DOC/'submission_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):base.sha(p) for p in files},indent=2)+'\n')
    print('66 slots: 44 exact reuse +22 NEW. Worker estimates:',loads,flush=True)


def worker(slot):
    base.guard()
    for p,h in json.loads((DOC/'submission_hashes.json').read_text()).items():assert base.sha(ROOT/p)==h,p
    rows={r['batch1_id']:r for r in json.loads((DOC/'interaction_matrix.json').read_text())}
    ids=json.loads((DOC/'worker_partition.json').read_text())['slots'][slot]
    control=OUT/'control';logs=OUT/'worker_logs';control.mkdir(parents=True,exist_ok=True);logs.mkdir(exist_ok=True)
    for rid in ids:
        assert not (control/'STOP').exists();assert not Path(rows[rid]['output_dir']).exists()
        start=time.time();print('START',rid,flush=True)
        with (logs/(rid+'.log')).open('x') as f:
            p=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'run',rid],stdout=f,stderr=subprocess.STDOUT)
        value=dict(row_id=rid,exit_code=p.returncode,worker=slot,job=os.environ.get('SLURM_JOB_ID'),train_start_unix=start,train_end_unix=time.time(),train_wall_time_sec=time.time()-start)
        (control/(rid+'.json')).write_text(json.dumps(value,indent=2)+'\n');print('FINISH',value,flush=True)
        if p.returncode:(control/'STOP').write_text(json.dumps(value));raise RuntimeError('Stop on failure')


if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
    elif sys.argv[1]=='run':
        base.DOC=DOC
        base.run(sys.argv[2])
