"""Keep fixed-state equivalence separate from independent-run determinism."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import torch

W=Path(__file__).resolve().parents[2]
P=W/'results/migration/2026-10-17/gpu'


def scalars(label,scene):
    with (P/('replay_'+label+'_'+scene)/'logs/scalars.tsv').open() as f:
        return {(r['step'],r['key']):float(r['value']) for r in csv.DictReader(f,delimiter='\t')}


def main():
    old=torch.load(str(P/'frozen_old.pth'),map_location='cpu')
    new=torch.load(str(P/'frozen_new.pth'),map_location='cpu')
    report={'frozen_checkpoint_tolerance':{'atol':1e-6,'rtol':1e-5},'frozen_comparisons':[], 'independent_training_controls':[], 'failures':[]}
    for scene in old:
        for a,b in zip(old[scene],new[scene]):
            assert a['frame_id']==b['frame_id']
            for field in ['prediction','depth']:
                x,y=a[field],b[field]
                maximum=float((x-y).abs().max())
                ok=torch.allclose(x,y,atol=1e-6,rtol=1e-5)
                report['frozen_comparisons'].append({'scene':scene,'frame':a['frame_id'],'field':field,'max_abs_difference':maximum,'passed':ok})
                if not ok:report['failures'].append(scene+':'+field)
        a,b,c=scalars('old',scene),scalars('new',scene),scalars('old_repeat',scene)
        for key in a:
            if key[1] in ('test/psnr','test/ssim','test/l1'):
                report['independent_training_controls'].append({'scene':scene,'step':key[0],'metric':key[1],'old':a[key],'migrated':b[key],'old_repeat':c[key],'migration_abs_difference':abs(a[key]-b[key]),'old_repeat_abs_difference':abs(a[key]-c[key])})
    report['fixed_state_passed']=not report['failures']
    report['independent_training_note']='Separate seeded CUDA training trajectories can diverge through atomic reductions and dynamic topology; tensor-by-index equality is not certified. These paired repeat controls quantify observed numerical variability, not a statistical equivalence claim.'
    (P/'frozen_control_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['failures']:raise SystemExit(1)


if __name__=='__main__':main()
