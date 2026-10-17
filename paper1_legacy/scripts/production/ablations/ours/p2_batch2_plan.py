"""Propose only: four LGR panels and minimal CGA evidence, never submit jobs."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
from scripts.production.ablations.ours.p2_refresh import ROOT,DOC,FREEZE,MATRIX,REP,SR,DC,effective,guard


def main():
    from configs.setup import load_hydra_config
    from omegaconf import OmegaConf
    guard();current=json.loads((DOC/MATRIX).read_text());known={r['semantic_config_sha256']:r for r in current};unique={};points=[]
    dest=DOC/'batch2_proposal';dest.mkdir(exist_ok=True);(dest/'exact_configs').mkdir(exist_ok=True)
    settings=[]
    for name,key,values in [('beta',SR+'ray_attenuation_beta',[.4,.8,1.]),('lambda',SR+'ray_attenuation_strength',[.2,.55,1.]),('Nq',SR+'ray_attenuation_quadrature_points',[1,2,3]),('half_window',SR+'ray_attenuation_quadrature_window_mm',[1.,2.,4.])]:
        for value in values:settings.append((name,str(value),{key:value}))
    for q,b in itertools.product([.94,.98,.99],[.001,.005,.05]):settings.append(('qd_by_B','{}_{}'.format(q,b),{DC+'responsibility.score_quantile':q,DC+'responsibility.max_fraction':b}))
    for qp,qa in itertools.product([.05,.1,.2],[.6,.8,.9]):settings.append(('qp_by_qA','{}_{}'.format(qp,qa),{DC+'responsibility.low_contribution_quantile':qp,DC+'responsibility.path_contribution_quantile':qa}))
    for a in (1e-5,1e-4,1e-3):settings.append(('tau_alpha',str(a),{DC+'thresholds.prune_opacity':a}))
    for ds,sw in REP:
        anchor=next(r for r in current if r['dataset']==ds and r['sweep']==sw and 'main' in r['consumers'])
        for panel,label,changes in settings:
            overrides=[x for x in anchor['overrides'] if x.lstrip('+').split('=')[0] not in changes]+[k+'='+str(v) for k,v in changes.items()]
            cfg=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',overrides),resolve=True)
            digest=hashlib.sha256(json.dumps(effective(cfg),sort_keys=True).encode()).hexdigest()
            if digest not in unique:
                reused=known.get(digest);rid='b2_{:03d}'.format(len(unique))
                unique[digest]=dict(id=rid,dataset=ds,sweep=sw,semantic_config_sha256=digest,source_freeze_sha256=(FREEZE/'snapshot.sha256').read_text().strip(),execution_authorized=False,status='FINAL-REUSE' if reused else 'PROPOSED_NEW',reuse_row_id=reused['batch1_id'] if reused else None,overrides=overrides,consumers=[],estimated_seconds=anchor['estimated_seconds'])
                OmegaConf.save(OmegaConf.create(cfg),dest/'exact_configs'/(rid+'.yaml'))
            unique[digest]['consumers'].append(panel+'/'+label);points.append(dict(dataset=ds,panel=panel,point=label,row_id=unique[digest]['id']))
    rows=list(unique.values());new=[r for r in rows if r['status']=='PROPOSED_NEW'];assert len(new)==52 and len(rows)==54
    (dest/'batch2_exact_proposed_matrix.json').write_text(json.dumps(rows,indent=2)+'\n');(dest/'panel_points.json').write_text(json.dumps(points,indent=2)+'\n')
    worker=sum(r['estimated_seconds'] for r in new)/3600
    (dest/'budget.json').write_text(json.dumps(dict(execution_authorized=False,new_runs=52,unique_physical_configs=54,reused_default_runs=2,panel_slots=len(points),estimated_worker_hours=worker,estimated_four_worker_hours=worker/4,headroom_fraction=.25,quality_population_new_runs=0,optional_split_sweep_additional_runs=4),indent=2)+'\n')
    (dest/'README.md').write_text('''# Proposed Batch2 — NOT authorized, NOT submitted

Exactly52 new30k runs on Liver/Arm representatives, MP+Fixed+Full+FullCGA,
same P2 freeze/seed/split.54 unique configs include2 reused full defaults.
No PPP reruns. No gamma/Tmin or additional renderer panels.

- LGR beta[.4,.8,1],lambda[.2,.55,1],Nq[1,2,3],half-window[1,2,4]mm:
  16new runs after default reuse.
- CGA qd[.94,.98,.99] x B[.001,.005,.05]:16new runs.
- CGA qp[.05,.10,.20] x qA[.6,.8,.9]:16new runs.
- tau_alpha[1e-5,1e-4,1e-3]:4new runs.
- Quality–population panel: reuse final five CGA variants,0new runs. It is a
  cross-variant trade-off, not a matched-capacity causal isolation. If a true split
  threshold panel is preferred instead, [.4,.6,1]mm adds4new runs (total56),
  not included in this proposed52-run authorization request.

Budget rationale: final Fixed default selected responsibility fractions are
approximately.00174–.00340(Liver) and.00091–.00156(Arm). The old B=.02/.05/.12
grid would mostly leave the budget inactive. B=.001/.005/.05 spans a potentially
binding cap and the unchanged loose default. This is a hypothesis to test, not
an asserted plateau. Final default local-contribution thresholds were0; if qp
points also yield zero thresholds, report quantile saturation rather than
claiming unrestricted robustness. Inspect event thresholds along with quality/#G.

Ranks/test optima do not retune defaults. These52 runs are only a proposed plan;
all execution_authorized flags are false. Await explicit user approval.
''')
    print('BATCH2 PROPOSAL ONLY:52NEW,2reuse; worker hours',worker)


if __name__=='__main__':main()
