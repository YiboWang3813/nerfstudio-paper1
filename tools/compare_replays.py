"""Compare migration replay signatures and numeric training/checkpoint records."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import torch


def tensors(blob, prefix=''):
    if torch.is_tensor(blob):
        yield prefix, blob
    elif isinstance(blob, dict):
        for key in sorted(blob, key=str):
            yield from tensors(blob[key], prefix+'/'+str(key))
    elif isinstance(blob, (tuple, list)):
        for i, value in enumerate(blob):
            yield from tensors(value, prefix+'/'+str(i))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output
    old = json.loads((out/'old_signatures.json').read_text())
    new = json.loads((out/'new_signatures.json').read_text())
    failures = []
    config_match = old['configs'] == new['configs']
    data_match = old['datasets'] == new['datasets']
    if not config_match: failures.append('config signatures differ')
    if not data_match: failures.append('dataset signatures differ')
    report = {'config_alias_count': len(old['configs']), 'config_signatures_identical': config_match,
              'existing_config_failures': {k:v for k,v in old['configs'].items() if 'error_type' in v},
              'four_dataset_signatures_identical': data_match, 'checkpoint_comparisons': [], 'scalar_comparisons': [],
              'tolerances': {'checkpoint_atol': 1e-5, 'checkpoint_rtol': 1e-4, 'scalar_atol': 1e-4, 'scalar_rtol': 1e-4}}
    for scene in old['datasets']:
        roots = [out/('replay_'+label+'_'+scene) for label in ('old', 'new')]
        scalar_paths = [r/'logs/scalars.tsv' for r in roots]
        scalar_rows = []
        for path in scalar_paths:
            with path.open() as f:
                scalar_rows.append(list(csv.DictReader(f, delimiter='\t')))
        a,b=scalar_rows
        if len(a)!=len(b): failures.append('scalar row count:'+scene)
        max_diff = 0.0
        for left,right in zip(a,b):
            if any(word in left.get('key', '').lower() for word in ('time','fps','duration','elapsed','timestamp')):
                continue
            for key in left:
                if any(word in key.lower() for word in ('time','fps','duration','elapsed','timestamp')): continue
                try: x,y=float(left[key]),float(right[key])
                except (TypeError, ValueError):
                    if left[key]!=right[key]: failures.append('scalar identity:'+scene+':'+key)
                    continue
                if not math.isfinite(x) or not math.isfinite(y):
                    failures.append('nonfinite scalar:'+scene+':'+key)
                elif not math.isclose(x,y,abs_tol=1e-4,rel_tol=1e-4):
                    failures.append('scalar:'+scene+':'+key)
                max_diff=max(max_diff,abs(x-y))
        report['scalar_comparisons'].append({'scene':scene,'rows':len(a),'max_abs_difference':max_diff})
        # Only model parameters are compared; wall-clock fields are not numeric outputs.
        left_files = sorted((roots[0]/'checkpoints').glob('*container.pth'))
        right_files = sorted((roots[1]/'checkpoints').glob('*container.pth'))
        if not left_files or len(left_files)!=len(right_files): failures.append('checkpoint count:'+scene)
        for lp,rp in zip(left_files,right_files):
            aa=dict(tensors(torch.load(str(lp),map_location='cpu')))
            bb=dict(tensors(torch.load(str(rp),map_location='cpu')))
            if aa.keys()!=bb.keys(): failures.append('tensor names:'+scene)
            max_diff=0.0
            for key in aa.keys() & bb.keys():
                x,y=aa[key],bb[key]
                if x.shape!=y.shape:
                    failures.append('tensor shape:'+scene+':'+key); continue
                if x.numel(): max_diff=max(max_diff,float((x.double()-y.double()).abs().max()))
                if not torch.allclose(x,y,atol=1e-5,rtol=1e-4): failures.append('tensor:'+scene+':'+key)
            report['checkpoint_comparisons'].append({'scene':scene,'file':lp.name,'tensor_count':len(aa),'max_abs_difference':max_diff})
    reference = out/'reference_tests'
    exits = {label:int((reference/(label+'.exit') if (reference/(label+'.exit')).exists() else out/(label+'_pytest.exit')).read_text()) for label in ('old','new')}
    report['pytest_exit_codes']=exits
    if any(exits.values()): failures.append('pytest failed; inspect old/new logs')
    pdf_exits={label:int((reference/('cpu_pdf_'+label+'.exit')).read_text()) if (reference/('cpu_pdf_'+label+'.exit')).exists() else None for label in ('old','new')}
    report['cpu_pdf_test_exit_codes']=pdf_exits
    if any(code!=0 for code in pdf_exits.values()): failures.append('CPU PDF verification missing or failed')
    report['failures']=failures
    report['passed']=not failures
    (out/'replay_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
