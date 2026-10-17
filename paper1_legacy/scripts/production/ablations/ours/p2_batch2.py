"""Authorized final numeric sensitivity only; P2 algorithms/defaults stay frozen."""
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from scripts.production.ablations.ours import p2_refresh as base
from scripts.production.ablations.ours.p0_reuse import flatten

ROOT, FREEZE, SR, DC = base.ROOT, base.FREEZE, base.SR, base.DC
DOC = ROOT/'docs/production/p2_batch2_final'
OUT = ROOT/'outputs/production/ablations/ours/P2_fixed_full_final_20260909/batch2'
MATRIX = 'batch2_exact_matrix.json'
PANELS = [
    ('sigma_ele', SR+'us_focus_sigma_ele_0', [.8, 1., 1.5, 2.], 1.),
    ('beta', SR+'ray_attenuation_beta', [.4, .6, .8, 1.], .8),
    ('lambda_ray', SR+'ray_attenuation_strength', [.2, .35, .55, .75, 1.], .55),
    ('Nq', SR+'ray_attenuation_quadrature_points', [1, 2, 3], 3),
    ('half_window', SR+'ray_attenuation_quadrature_window_mm', [1., 2., 3., 4.], 2.),
]


def plan():
    from configs.setup import load_hydra_config
    from omegaconf import OmegaConf
    base.guard()
    assert not (DOC/MATRIX).exists(), 'Never overwrite a sealed matrix'
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC/'exact_configs').mkdir(exist_ok=True)
    old = json.loads((base.DOC/base.MATRIX).read_text())
    known = {r['semantic_config_sha256']: r for r in old}
    settings = [(n, str(v), {k: v}, v == default) for n,k,vs,default in PANELS for v in vs]
    for q,b in itertools.product([.96,.98,.99], [.001,.005,.05]):
        settings.append(('qd_by_B', '{}_{}'.format(q,b), {DC+'responsibility.score_quantile':q, DC+'responsibility.max_fraction':b}, q==.98 and b==.05))
    for q,a in itertools.product([.05,.1,.15], [.7,.8,.9]):
        settings.append(('qp_by_qA', '{}_{}'.format(q,a), {DC+'responsibility.low_contribution_quantile':q, DC+'responsibility.path_contribution_quantile':a}, q==.1 and a==.8))
    for a in [5e-5,1e-4,5e-4]:
        settings.append(('tau_alpha', str(a), {DC+'thresholds.prune_opacity':a}, a==1e-4))
    rows, unique, points = [], {}, []
    for ds,sw in base.REP:
        anchor = next(r for r in old if r['dataset']==ds and r['sweep']==sw and 'main' in r['consumers'])
        acfg = OmegaConf.to_container(OmegaConf.load(Path(anchor['output_dir'])/'resolved_config.yaml'), resolve=False)
        aflat = flatten(base.effective(acfg))
        for panel,label,changes,is_default in settings:
            overrides = [v for v in anchor['overrides'] if v.lstrip('+').split('=')[0] not in changes]+[k+'='+str(v) for k,v in changes.items()]
            cfg = OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze', overrides), resolve=True)
            eff = base.effective(cfg); flat = flatten(eff)
            changed = {k for k in set(flat)|set(aflat) if flat.get(k)!=aflat.get(k)}
            assert changed <= set(changes), (panel, changed)
            assert bool(changed) != is_default
            sr = cfg['model']['slice_render']; dc = cfg['model']['offline']['density_control']
            assert sr['local_conditioning_mode']=='fixed' and sr['ray_attenuation_formula']=='integral'
            assert sr['us_focus_sigma_ele_min'] <= sr['us_focus_sigma_ele_0'] <= sr['us_focus_sigma_ele_max']
            assert dc['pruning_policy']=='path_aware' and dc['responsibility']['protect_attenuation_mode']=='path_contribution'
            assert cfg['scene']['loading']['pose_mode']=='file'
            digest = hashlib.sha256(json.dumps(eff,sort_keys=True).encode()).hexdigest()
            if digest not in unique:
                reused = known.get(digest); rid = 'h2_{:03d}'.format(len(rows))
                row = copy.deepcopy(anchor)
                row.update(batch1_id=rid, group='structural', variant=panel+'/'+label, study='Final numeric hyperparameter analysis',
                           overrides=overrides, semantic_config_sha256=digest, execution_authorized=reused is None,
                           target_status='FINAL-REUSE' if reused else 'NEW', consumers=[], reuse_id=reused['batch1_id'] if reused else None,
                           execution_source_sha256=reused['execution_source_sha256'] if reused else anchor['source_freeze_sha256'],
                           output_dir=reused['output_dir'] if reused else str(OUT/'runs'/ds/sw/rid), changed_keys=sorted(changed))
                if reused:
                    actual = OmegaConf.to_container(OmegaConf.load(Path(reused['output_dir'])/'resolved_config.yaml'),resolve=False)
                    assert base.effective(actual)==eff
                    assert json.loads((Path(reused['output_dir'])/'batch1_stats.json').read_text())['success']
                else:
                    assert not Path(row['output_dir']).exists()
                OmegaConf.save(OmegaConf.create(cfg),DOC/'exact_configs'/(rid+'.yaml'))
                rows.append(row); unique[digest]=row
            row=unique[digest]; row['consumers'].append(panel+'/'+label)
            points.append(dict(dataset=ds,sweep=sw,panel=panel,point=label,row_id=row['batch1_id'],is_default=is_default,changes=changes))
    new=[r for r in rows if r['execution_authorized']]
    assert len(new)==66 and len(rows)==68 and len(points)==82
    assert len(new)<=70
    bins=[[] for _ in range(4)]; loads=[0.]*4
    for r in sorted(new,key=lambda r:r['estimated_seconds'],reverse=True):
        slot=min(range(4),key=lambda s:loads[s]); bins[slot].append(r['batch1_id']); loads[slot]+=r['estimated_seconds']
    for name,value in [(MATRIX,rows),('panel_points.json',points),('worker_partition.json',dict(slots=bins,estimated_seconds=loads)),
                       ('budget.json',dict(new_runs=len(new),reused_unique=2,logical_slots=len(points),worker_hours=sum(loads)/3600,four_worker_hours=max(loads)/3600,headroom_fraction=.3))]:
        (DOC/name).write_text(json.dumps(value,indent=2)+'\n')
    print('EXACT DEDUP: 82 slots = 66 NEW + 16 reused panel-default slots (2 unique default runs).')
    print('Worker seconds:',loads)


def seal():
    base.guard()
    digest=(FREEZE/'snapshot.sha256').read_text().strip()
    code=(base.DOC/'executed_runner.py').read_text().replace("ROOT/'docs/production/p2_fixed_final'", "ROOT/'docs/production/p2_batch2_final'").replace('final_refresh_matrix.json', MATRIX).replace('P2_Final_Method_Refresh', 'P2_Final_Hyperparameter_Batch2')
    (DOC/'executed_runner.py').write_text(code)
    files=list((DOC/'exact_configs').glob('*.yaml'))+[DOC/MATRIX,DOC/'panel_points.json',DOC/'worker_partition.json',DOC/'executed_runner.py',Path(__file__).resolve(),Path(base.worker_impl.__file__).resolve(),Path(base.__file__).resolve()]
    files += [ROOT/'scripts/production/ablations/ours'/n for n in ('p2_batch2_gpu.sh','p2_batch2_post.sh','p2_batch2_report.py','p2_refresh_report.py','p0_reuse.py')]
    (DOC/'submission_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):base.sha(p) for p in files},indent=2)+'\n')
    (DOC/'freeze.json').write_text(json.dumps(dict(source_sha256=digest,archive=str(FREEZE/'source.tar.gz')),indent=2)+'\n')
    print('SEALED',digest)


if __name__=='__main__':
    if sys.argv[1]=='plan': plan()
    elif sys.argv[1]=='seal': seal()
    elif sys.argv[1]=='run':
        base.guard(); sys.argv=[str(DOC/'executed_runner.py'),sys.argv[2]]
        exec(compile((DOC/'executed_runner.py').read_text(),str(DOC/'executed_runner.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/production/ablations/ours/p1_batch1_run.py')})
    elif sys.argv[1]=='worker':
        w=base.worker_impl; w.base.guard=base.guard; w.DOC=DOC; w.OUT=OUT; w.MATRIX=MATRIX; w.__file__=__file__
        w.worker(int(sys.argv[2]))
