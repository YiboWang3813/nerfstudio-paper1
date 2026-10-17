"""Freeze authorized Batch 1 exact Hydra overrides; no training submission."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf
from configs.setup import load_hydra_config
from scripts.production.ablations.ours.p0_reuse import flatten, semantic

ROOT = Path(__file__).resolve().parents[4]
P05 = ROOT / 'outputs/preproduction/regression/p05_corrected_covariance_20260908'
OUT = ROOT / 'docs/production/p1_paper_protocol_freeze'
SHA = 'b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2'
TAG = 'paper_final_corrected_covariance_20260908'


def literal(v):
    if isinstance(v, bool):
        return str(v).lower()
    return json.dumps(v)


def main():
    assert hashlib.sha256((P05/'freeze_v2/source.tar.gz').read_bytes()).hexdigest() == SHA
    hashes = json.loads((P05/'freeze_v2/hashes.json').read_text())
    for p, h in hashes.items():
        if not p.startswith(('docs/', 'scripts/', 'tests/')):
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h, p
    plans = json.loads((P05/'minimal_run_matrix.json').read_text())
    rows = [r for r in plans if r['group'] in ('main','structural','multi_sweep','hyper_PPP')]
    mainrows = {(r['dataset'],r['sweep']):r for r in rows if r['group']=='main'}
    seen, anchor_check, main_diffs = {}, {}, {}
    for i,r in enumerate(rows):
        r['batch1_id'] = 'b1_{:03d}'.format(i)
        r['source_freeze_sha256'] = SHA
        r['execution_authorized'] = r['target_status'] != 'FINAL-REUSE'
        # Remove Batch 2 consumer labels, not the shared physical experiment.
        r['consumers'] = [c for c in r['consumers'] if not c.startswith(('hyper_LGR/', 'hyper_CGA/'))]
        m = mainrows[r['dataset'],r['sweep']]
        # Always inherit exact audited main image/split/calibration config.
        exact = []
        for prefix in ('scene','metric'):
            for k,v in flatten(m['exact_main_'+prefix+'_config'],prefix+'.').items():
                exact.append('++'+k+'='+literal(v))
        # Existing variant overrides then deliberately replace only pose/control fields.
        r['overrides'] = [x for x in r['overrides'] if x.startswith('scene=')] + exact + [x for x in r['overrides'] if not x.startswith('scene=')]
        r['overrides'] += ['engine.iterations=30000','engine.interval.test=30000','engine.interval.save=30000','wandb.enable=false']
        c = OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',r['overrides']),resolve=True)
        effective = semantic(c)
        effective['metric'] = c['metric']
        digest = hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()
        assert digest not in seen, ('duplicate config', r['batch1_id'],seen.get(digest))
        seen[digest]=r['batch1_id']
        r['semantic_config_sha256']=digest
        target = ROOT/'outputs/production'/('comparisons' if r['group']=='main' else 'ablations/ours')/TAG/'runs'/r['dataset']/r['sweep']/r['batch1_id']
        r['output_dir'] = str(target)
        if r['group']=='main':
            old = OmegaConf.to_container(OmegaConf.load(Path(r['source'])/'resolved_config.yaml'),resolve=False)
            a,b = flatten(semantic(old)),flatten(semantic(c))
            diff={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}
            main_diffs[r['batch1_id']]=diff
            assert not diff, (r['batch1_id'], diff)
        if r['target_status']=='FINAL-REUSE':
            name='liver' if r['dataset']=='synthetic_liver' else 'arm'
            old=OmegaConf.to_container(OmegaConf.load(P05/'anchors'/name/'resolved_config.yaml'),resolve=False)
            a,b=flatten(semantic(old)),flatten(semantic(c))
            diff={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}
            assert not diff and old['metric']==c['metric'],diff
            r['output_dir']=str(P05/'anchors'/name)
            anchor_check[r['batch1_id']]=dict(semantic_differences=diff,metric_equal=True,source=r['output_dir'])
        cfgdir=OUT/'exact_configs';cfgdir.mkdir(exist_ok=True)
        OmegaConf.save(OmegaConf.create(c),cfgdir/(r['batch1_id']+'.yaml'))
    assert len(rows)==134 and sum(r['execution_authorized'] for r in rows)==132
    for name,value in [('batch1_matrix.json',rows),('batch1_anchor_reuse.json',anchor_check),('batch1_main_config_differences.json',main_diffs)]:
        (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
    summary=dict(total_unique=134,final_reuse=2,remaining=132,
                 remaining_by_group=Counter(r['group'] for r in rows if r['execution_authorized']),
                 source_freeze=SHA, exact_config_dedup=True, batch2_authorized=False)
    (OUT/'batch1_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
