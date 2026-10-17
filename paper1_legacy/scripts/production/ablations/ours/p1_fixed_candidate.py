"""Submission-only full-protocol Fixed+Full design validation; no algorithm edits."""
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.production.ablations.ours import p1_lgr_confirmation as base

ROOT=base.ROOT
DOC=ROOT/'docs/production/p1_fixed_main_candidate'
OUT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/fixed_full_candidate'
MATRIX='fixed_full_main_candidate_matrix.json'


def effective(cfg):
    from scripts.production.ablations.ours.p0_reuse import semantic
    result=semantic(cfg);result['metric']=cfg['metric'];return result


def plan():
    from omegaconf import OmegaConf
    from configs.setup import load_hydra_config
    base.guard();DOC.mkdir(exist_ok=True);(DOC/'exact_configs').mkdir(exist_ok=True)
    assert not (DOC/MATRIX).exists(),'Never overwrite a sealed plan'
    old=json.loads((base.OLD/'batch1_matrix.json').read_text())
    mains=[r for r in old if r['group']=='main'];assert len(mains)==78
    protocol=list(csv.DictReader((base.OLD/'paper_final_dataset_protocol.tsv').open(),delimiter='\t'))
    assert {(r['dataset'],r['sweep']) for r in mains}=={(r['dataset'],r['case_id']) for r in protocol if r['include_in_main']=='1'}
    previous=json.loads((ROOT/'docs/production/p1_lgr_interaction/interaction_matrix.json').read_text())
    known={r['semantic_config_sha256']:r for r in previous}
    rows=[]
    for template in mains:
        ds,sw=template['dataset'],template['sweep'];row=copy.deepcopy(template)
        row.update(batch1_id='fc_{:03d}'.format(len(rows)),group='structural',variant='Fixed+Full_candidate',study='paper_wide_local_design_selection')
        row['adaptive_output_dir']=template['output_dir'];row['adaptive_semantic_config_sha256']=template['semantic_config_sha256']
        row['settings']['local']='fixed'
        row['overrides']=[x for x in row['overrides'] if x.lstrip('+').split('=')[0]!='model.slice_render.local_conditioning_mode']+['model.slice_render.local_conditioning_mode=fixed']
        cfg=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',row['overrides']),resolve=True)
        eff=effective(cfg)
        actual=OmegaConf.to_container(OmegaConf.load(Path(template['output_dir'])/'resolved_config.yaml'),resolve=False)
        adapt=effective(actual);expected=copy.deepcopy(adapt);expected['model']['slice_render']['local_conditioning_mode']='fixed'
        assert eff==expected,(ds,sw,'More than local-mode changed')
        digest=hashlib.sha256(json.dumps(eff,sort_keys=True).encode()).hexdigest();row['semantic_config_sha256']=digest
        reused=known.get(digest);row['execution_authorized']=reused is None;row['target_status']='NEW' if reused is None else 'FINAL-REUSE'
        row['reuse_interaction_id']=reused['batch1_id'] if reused else None
        row['output_dir']=reused['output_dir'] if reused else str(OUT/'runs'/ds/sw/row['batch1_id'])
        row['source_freeze_sha256']=base.SHA;row['consumers']=['FixedFull_main_candidate']
        if reused:
            re=OmegaConf.to_container(OmegaConf.load(Path(reused['output_dir'])/'resolved_config.yaml'),resolve=False)
            assert effective(re)==eff
            st=json.loads((Path(reused['output_dir'])/'batch1_stats.json').read_text());assert st['success'] and st['finite_final_parameters'] and st['source_freeze_sha256']==base.SHA
        else:assert not Path(row['output_dir']).exists()
        p=Path(template['output_dir'])/'batch1_stats.json'
        if not p.exists():p=p.with_name('anchor_stats.json')
        s=json.loads(p.read_text());assert s['success']
        row['estimated_seconds']=s.get('wrapper_wall_seconds',s.get('wall_seconds'))
        OmegaConf.save(OmegaConf.create(cfg),DOC/'exact_configs'/(row['batch1_id']+'.yaml'));rows.append(row)
    assert len({r['semantic_config_sha256'] for r in rows})==78
    new=[r for r in rows if r['execution_authorized']];assert len(new)==67
    bins=[[] for _ in range(4)];loads=[0.]*4
    for r in sorted(new,key=lambda r:r['estimated_seconds'],reverse=True):
        slot=min(range(4),key=lambda s:loads[s]);bins[slot].append(r['batch1_id']);loads[slot]+=r['estimated_seconds']
    (DOC/MATRIX).write_text(json.dumps(rows,indent=2)+'\n')
    (DOC/'worker_partition.json').write_text(json.dumps(dict(slots=bins,estimated_seconds=loads),indent=2)+'\n')
    original=ROOT/'scripts/production/ablations/ours/p1_batch1_run.py'
    sealed=json.loads((base.OLD/'submission_hashes.json').read_text());assert base.sha(original)==sealed[str(original.relative_to(ROOT))]
    code=original.read_text().replace("PROTOCOL = ROOT/'docs/production/p1_paper_protocol_freeze'","PROTOCOL = ROOT/'docs/production/p1_fixed_main_candidate'").replace("PROTOCOL/'batch1_matrix.json'","PROTOCOL/'"+MATRIX+"'").replace("'Batch1_only'","'FixedFull_candidate_only'")
    (DOC/'executed_runner.py').write_text(code)
    files=list((DOC/'exact_configs').glob('*.yaml'))+[DOC/MATRIX,DOC/'worker_partition.json',DOC/'executed_runner.py',Path(__file__).resolve(),Path(base.__file__).resolve()]
    files += [ROOT/'scripts/production/ablations/ours'/n for n in ('p1_fixed_candidate_gpu.sh','p1_fixed_candidate_report.py','p1_fixed_candidate_post.sh')]
    (DOC/'submission_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):base.sha(p) for p in files},indent=2)+'\n')
    print('78 Fixed slots:11 exact reuse+67 NEW;78 Adaptive reuse. Worker seconds:',loads,flush=True)


def worker(slot):
    base.guard()
    for p,h in json.loads((DOC/'submission_hashes.json').read_text()).items():assert base.sha(ROOT/p)==h,p
    rows={r['batch1_id']:r for r in json.loads((DOC/MATRIX).read_text())}
    ids=json.loads((DOC/'worker_partition.json').read_text())['slots'][slot]
    control=OUT/'control';logs=OUT/'worker_logs';control.mkdir(parents=True,exist_ok=True);logs.mkdir(exist_ok=True)
    for rid in ids:
        assert not (control/'STOP').exists();assert not Path(rows[rid]['output_dir']).exists()
        start=time.time();print('START',rid,flush=True)
        with (logs/(rid+'.log')).open('x') as f:
            proc=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'run',rid],stdout=f,stderr=subprocess.STDOUT)
        value=dict(row_id=rid,exit_code=proc.returncode,worker=slot,job=os.environ.get('SLURM_JOB_ID'),train_start_unix=start,train_end_unix=time.time(),train_wall_time_sec=time.time()-start)
        (control/(rid+'.json')).write_text(json.dumps(value,indent=2)+'\n');print('FINISH',value,flush=True)
        if proc.returncode:(control/'STOP').write_text(json.dumps(value));raise RuntimeError('Stop on failure')


if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
    elif sys.argv[1]=='run':base.DOC=DOC;base.run(sys.argv[2])
