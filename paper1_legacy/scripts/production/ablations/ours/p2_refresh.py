"""Final Fixed refresh: exact semantic dedup, no LGR/CGA hyperparameter execution."""
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from scripts.production.ablations.ours import p1_fixed_candidate as worker_impl
from scripts.production.ablations.ours.p1_fixed_candidate import effective

ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/production/p2_fixed_final'
OUT=ROOT/'outputs/production/ablations/ours/P2_fixed_full_final_20260909'
MAIN=ROOT/'outputs/production/comparisons/P2_fixed_full_final_20260909'
FREEZE=ROOT/'outputs/preproduction/regression/P2_fixed_full_final_20260909/freeze'
MATRIX='final_refresh_matrix.json'
REP=[('synthetic_liver','l1'),('tus_rec_2024_qc','010_LH_Per_L_DtP_qc_0005_0370')]
SR='model.slice_render.'
DC='model.offline.density_control.'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def guard():
    for name in ('hashes.json','binary_hashes.json'):
        for p,h in json.loads((FREEZE/name).read_text()).items():
            if not p.startswith(('docs/','scripts/','tests/')):assert sha(ROOT/p)==h,p


def plan():
    from omegaconf import OmegaConf
    from configs.setup import load_hydra_config
    DOC.mkdir(exist_ok=True);(DOC/'exact_configs').mkdir(exist_ok=True);assert not (DOC/MATRIX).exists()
    candidate=json.loads((ROOT/'docs/production/p1_fixed_main_candidate/fixed_full_main_candidate_matrix.json').read_text())
    mains={(r['dataset'],r['sweep']):r for r in candidate};assert len(mains)==78
    known={}
    for file in ['docs/production/p1_fixed_main_candidate/fixed_full_main_candidate_matrix.json','docs/production/p1_lgr_interaction/interaction_matrix.json','docs/production/p1_lgr_confirmation/lgr_multisweep_structural_matrix.json','docs/production/p1_paper_protocol_freeze/batch1_matrix.json']:
        for r in json.loads((ROOT/file).read_text()):known.setdefault(r['semantic_config_sha256'],r)
    rows=[];byhash={};consumers=[]
    def add(ds,sw,consumer,pose='file',spacing='uniform',rho=1.,local='fixed',glob='integral',dense='residual_guided',prune='path_aware'):
        template=mains[ds,sw];row=copy.deepcopy(template)
        changes={SR+'local_conditioning_mode':local,SR+'ray_attenuation_enabled':str(glob!='none').lower(),SR+'ray_attenuation_formula':'integral' if glob=='none' else glob,DC+'densification_policy':dense,DC+'pruning_policy':prune,'scene.loading.pose_mode':pose,'scene.loading.canonical_sweep_spacing_mode':spacing,'scene.loading.canonical_sweep_delta_scale':str(rho),'experiment.sensorless_required':str(pose=='canonical_sweep' and spacing=='uniform').lower()}
        overrides=[v for v in template['overrides'] if v.lstrip('+').split('=')[0] not in changes]+[k+'='+v for k,v in changes.items()]
        cfg=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',overrides),resolve=True)
        digest=hashlib.sha256(json.dumps(effective(cfg),sort_keys=True).encode()).hexdigest()
        if digest not in byhash:
            rid='p2_{:03d}'.format(len(rows));reused=known.get(digest)
            row.update(batch1_id=rid,group='main' if consumer=='main' else 'structural',variant=consumer,overrides=overrides,semantic_config_sha256=digest,source_freeze_sha256='P2_PENDING',execution_authorized=False,target_status='FINAL-REUSE' if reused else 'NEW',consumers=[])
            row['settings']=dict(pose=pose,spacing=spacing,rho=rho,local=local,global_mode=glob,dense=dense,prune=prune)
            row['output_dir']=reused['output_dir'] if reused else str(OUT/'runs'/ds/sw/rid)
            row['execution_source_sha256']=reused['source_freeze_sha256'] if reused else 'P2_PENDING'
            row['reuse_id']=reused['batch1_id'] if reused else None
            if reused:
                actual=OmegaConf.to_container(OmegaConf.load(Path(reused['output_dir'])/'resolved_config.yaml'),resolve=False)
                assert effective(actual)==effective(cfg),(consumer,ds,sw)
            else:assert not Path(row['output_dir']).exists()
            src=Path(template['output_dir'])/'batch1_stats.json';row['estimated_seconds']=json.loads(src.read_text())['wrapper_wall_seconds']
            OmegaConf.save(OmegaConf.create(cfg),DOC/'exact_configs'/(rid+'.yaml'));rows.append(row);byhash[digest]=row
        row=byhash[digest];row['consumers'].append(consumer)
        consumers.append(dict(dataset=ds,sweep=sw,consumer=consumer,row_id=row['batch1_id']))
    for ds,sw in mains:add(ds,sw,'main')
    for ds,sw in REP:
        for l,c,p in itertools.product((0,1),repeat=3):
            add(ds,sw,'joint/L{}C{}P{}'.format(l,c,p),pose='canonical_sweep' if p else 'file',local='fixed' if l else 'none',glob='integral' if l else 'none',dense='residual_guided' if c else 'gradient',prune='path_aware' if c else 'opacity')
        for label,pose,spacing in [('MP','file','uniform'),('PP-S','canonical_sweep','file_step_norm'),('PP-A','canonical_sweep','file_axis_projection'),('PPP','canonical_sweep','uniform')]:add(ds,sw,'ppp/'+label,pose=pose,spacing=spacing)
    selected=json.loads((ROOT/'docs/production/p1_paper_protocol_freeze/protocol_summary.json').read_text())['selected_multi_sweep'];assert len(selected)==16
    for ds,sw in selected:
        for pose,label in [('file','MP'),('canonical_sweep','PPP')]:add(ds,sw,'multi/'+label,pose=pose)
    for ds,sw in REP:
        for rho in (.6,.75,.9,1.,1.1,1.25,1.5):add(ds,sw,'rho/'+str(rho),pose='canonical_sweep',rho=rho)
        for label,dense,prune in [('GD','gradient','opacity'),('RGD','residual_guided','none'),('RGD+OP','residual_guided','opacity'),('RGD+LP','residual_guided','local_aware'),('Full','residual_guided','path_aware')]:add(ds,sw,'cga/'+label,dense=dense,prune=prune)
    for ds,sw in selected:
        if ds=='tdsc_abus_2023':continue
        for label,local,glob in [('Plane','none','none'),('Fixed','fixed','none'),('Adaptive','adaptive','none'),('Fixed+Simple','fixed','slice_depth'),('Fixed+Full','fixed','integral')]:add(ds,sw,'lgr/'+label,local=local,glob=glob)
    new=[r for r in rows if r['target_status']=='NEW'];assert len(new)<=50,'Stop: exceeds user budget gate'
    assert not any(r['target_status']=='NEW' for r in rows if any(c=='main' or c.startswith('lgr/') for c in r['consumers']))
    bins=[[] for _ in range(4)];loads=[0.]*4
    for r in sorted(new,key=lambda r:r['estimated_seconds'],reverse=True):
        slot=min(range(4),key=lambda s:loads[s]);bins[slot].append(r['batch1_id']);loads[slot]+=r['estimated_seconds']
    for name,value in [(MATRIX,rows),('consumers.json',consumers),('worker_partition.json',dict(slots=bins,estimated_seconds=loads)),('final_fixed_joint_matrix.json',[dict(c,**{'run':next(r for r in rows if r['batch1_id']==c['row_id'])}) for c in consumers if c['consumer'].startswith('joint/')])]:
        (DOC/name).write_text(json.dumps(value,indent=2)+'\n')
    print('P2 plan:',len(rows),'unique;',len(new),'NEW; main/LGR zero NEW; seconds',loads)


def seal():
    guard();digest=(FREEZE/'snapshot.sha256').read_text().strip()
    (DOC/'freeze.json').write_text(json.dumps(dict(source_sha256=digest,archive=str(FREEZE/'source.tar.gz')),indent=2)+'\n')
    rows=json.loads((DOC/MATRIX).read_text())
    for r in rows:
        r['source_freeze_sha256']=digest;r['execution_authorized']=r['target_status']=='NEW'
        if r['execution_authorized']:r['execution_source_sha256']=digest
    (DOC/MATRIX).write_text(json.dumps(rows,indent=2)+'\n')
    consumers=json.loads((DOC/'consumers.json').read_text())
    (DOC/'final_fixed_joint_matrix.json').write_text(json.dumps([dict(c,run=next(r for r in rows if r['batch1_id']==c['row_id'])) for c in consumers if c['consumer'].startswith('joint/')],indent=2)+'\n')
    original=ROOT/'scripts/production/ablations/ours/p1_batch1_run.py'
    code=original.read_text().replace("PROTOCOL = ROOT/'docs/production/p1_paper_protocol_freeze'","PROTOCOL = ROOT/'docs/production/p2_fixed_final'").replace("PROTOCOL/'batch1_matrix.json'","PROTOCOL/'"+MATRIX+"'").replace("ROOT/'outputs/preproduction/regression/p05_corrected_covariance_20260908/freeze_v2'","ROOT/'outputs/preproduction/regression/P2_fixed_full_final_20260909/freeze'").replace("'b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2'",repr(digest)).replace("'Batch1_only'","'P2_Final_Method_Refresh'")
    (DOC/'executed_runner.py').write_text(code)
    files=list((DOC/'exact_configs').glob('*.yaml'))+[DOC/MATRIX,DOC/'freeze.json',DOC/'executed_runner.py',DOC/'worker_partition.json',Path(__file__).resolve(),Path(worker_impl.__file__).resolve()]
    files += [ROOT/'scripts/production/ablations/ours'/n for n in ('p2_refresh_gpu.sh','p2_refresh_report.py','p2_refresh_post.sh')]
    (DOC/'submission_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
    print('SEALED',digest,'NEW',sum(r['execution_authorized'] for r in rows))


if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    elif sys.argv[1]=='seal':seal()
    elif sys.argv[1]=='run':
        guard();sys.argv=[str(DOC/'executed_runner.py'),sys.argv[2]]
        exec(compile((DOC/'executed_runner.py').read_text(),str(DOC/'executed_runner.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/production/ablations/ours/p1_batch1_run.py')})
    elif sys.argv[1]=='worker':
        worker_impl.base.guard=guard;worker_impl.DOC=DOC;worker_impl.OUT=OUT;worker_impl.MATRIX=MATRIX;worker_impl.__file__=__file__
        worker_impl.worker(int(sys.argv[2]))
