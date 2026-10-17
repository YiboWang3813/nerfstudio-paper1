"""Strict paired interaction analysis and inherited-crop contact sheets only."""
import csv
import hashlib
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

from scripts.production.ablations.ours.p1_lgr_interaction import ROOT,DOC,OUT
from scripts.production.ablations.ours.p1_lgr_report import paired_statistics,write

METRICS=('psnr','ssim','lpips')


def interaction_delta(a_none,a_global,f_none,f_global):
    return (a_global-a_none)-(f_global-f_none)


def main():
    from scripts.production.ablations.ours.summarize_ablation_run import read_series,value_at
    from scripts.production.ablations.ours.p1_scientific_checkpoint import table
    from scripts.production.ablations.ours.p1_lgr_confirmation import guard
    guard();matrix=json.loads((DOC/'interaction_matrix.json').read_text())
    audit=json.loads((ROOT/'docs/production/p1_paper_protocol_freeze/baseline_case_frame_audit.json').read_text())
    report=OUT/'reports';report.mkdir(parents=True,exist_ok=True);rows=[];pending=[]
    for r in matrix:
        run=Path(r['output_dir']);p=run/'batch1_stats.json'
        if not p.exists():p=run/'anchor_stats.json'
        if not p.exists() or not json.loads(p.read_text()).get('success'):
            pending.append(r['batch1_id']);continue
        state=json.loads(p.read_text())
        if r['execution_authorized']:
            assert json.loads((OUT/'control'/(r['batch1_id']+'.json')).read_text())['exit_code']==0
            assert state['finite_final_parameters'] and state['semantic_config_sha256']==r['semantic_config_sha256']
            assert state['source_freeze_sha256']==r['source_freeze_sha256']
        series=read_series(run/'logs/scalars.tsv')
        row=dict(dataset=r['dataset'],sweep=r['sweep'],local=r['settings']['local'],global_mode=r['settings']['global_mode'],row_id=r['batch1_id'],reused=not r['execution_authorized'],output_dir=str(run),gpu=state.get('gpu','see manifest'))
        for m in METRICS:
            row[m]=value_at(series,'test/'+m,30000);assert row[m] is not None and math.isfinite(row[m])
        frames=list(csv.DictReader((run/'eval/per_frame_metrics_iteration_030000.tsv').open(),delimiter='\t'))
        expected=next(x['expected_test_indices'] for x in audit if x['dataset']==r['dataset'] and x['sweep']==r['sweep'])
        assert [int(x['index']) for x in frames]==expected
        assert all(math.isfinite(float(x[m])) for x in frames for m in METRICS)
        rows.append(row)
    (report/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=66,pending=pending),indent=2)+'\n')
    if rows:write(report/'per_case.tsv',rows)
    assert not pending,'Incomplete study; no partial mean promoted to final'
    means=[];contrasts=[];per_pair=[];interactions=[]
    for ds in sorted({r['dataset'] for r in rows}):
        group=[r for r in rows if r['dataset']==ds];cases=sorted({r['sweep'] for r in group});lookup={(r['sweep'],r['local'],r['global_mode']):r for r in group}
        for local in ('fixed','adaptive'):
            for glob in ('none','slice_depth','integral'):
                rr=[r for r in group if r['local']==local and r['global_mode']==glob];assert len(rr)==len(cases)
                means.append(dict(dataset=ds,local=local,global_mode=glob,n=len(rr),**{m:st.mean(r[m] for r in rr) for m in METRICS}))
        pairs=[('Adaptive-Fixed under '+g,('fixed',g),('adaptive',g)) for g in ('none','slice_depth','integral')]
        pairs += [(l+': '+a+' -> '+b,(l,a),(l,b)) for l in ('fixed','adaptive') for a,b in [('none','slice_depth'),('slice_depth','integral')]]
        for label,before,after in pairs:
            for m in METRICS:
                delta=[lookup[(sw,)+after][m]-lookup[(sw,)+before][m] for sw in cases]
                contrasts.append(dict(dataset=ds,contrast=label,metric=m,**paired_statistics(delta,m)))
                per_pair.extend(dict(dataset=ds,sweep=sw,contrast=label,metric=m,delta=d) for sw,d in zip(cases,delta))
        for a,b in [('none','slice_depth'),('slice_depth','integral'),('none','integral')]:
            for m in METRICS:
                dd=[interaction_delta(lookup[sw,'adaptive',a][m],lookup[sw,'adaptive',b][m],lookup[sw,'fixed',a][m],lookup[sw,'fixed',b][m]) for sw in cases]
                interactions.append(dict(dataset=ds,global_transition=a+' -> '+b,metric=m,**paired_statistics(dd,m)))
                per_pair.extend(dict(dataset=ds,sweep=sw,contrast='Interaction '+a+' -> '+b,metric=m,delta=d) for sw,d in zip(cases,dd))
    write(report/'six_row_means_long.tsv',means);write(report/'paired_statistics.tsv',contrasts);write(report/'per_case_paired_deltas.tsv',per_pair);write(report/'interaction_difference_in_differences.tsv',interactions)
    wide=[]
    for local in ('fixed','adaptive'):
        for glob in ('none','slice_depth','integral'):
            entry=dict(local=local,global_mode=glob)
            for ds,label in [('synthetic_liver','Liver'),('tus_rec_2024_qc','Arm')]:
                r=next(r for r in means if r['dataset']==ds and r['local']==local and r['global_mode']==glob)
                for m in METRICS:entry[label+'_'+m]=r[m]
            wide.append(entry)
    write(report/'six_row_interaction_table.tsv',wide)
    text=['# LGR local-global interaction paired report',
          '66 slots =44 exact-config reuse +22 new Fixed+Simple/Full; MP,1M,30k,seed42, same11 cases and corrected source freeze. No algorithm/default change. Internal analysis, not automatically a paper table.',
          'Paired deltas are after-minus-before. Std is sample std(ddof=1); numerical ties abs<=1e-6; LPIPS wins reverse the sign. Counts are not significance tests. Mixed4080/4090 hardware and one seed; no independent-subject inference from Arm segments.',
          'No-global CGA falls back to local-aware pruning. These are trained whole-system interactions, including adaptation responses, not isolated fixed-topology renderer derivatives.',
          '## Six-row means',table(wide,list(wide[0])),
          '## Adaptive-Fixed under each global mode and global transitions within each local mode',table(contrasts,['dataset','contrast','metric','mean_delta','std_paired_delta','wins','ties','losses','min_delta','max_delta']),
          '## Difference-in-differences',
          '(Adaptive_after-Adaptive_before) - (Fixed_after-Fixed_before). Positive PSNR/SSIM means a larger global benefit under Adaptive; negative LPIPS means a larger LPIPS reduction under Adaptive. A nonzero interaction does not by itself mean Adaptive+Full is better in absolute quality.',
          table(interactions,['dataset','global_transition','metric','mean_delta','std_paired_delta','wins','ties','losses']),
          'No automatic CaseA/B/C decision, production change, or Batch2 submission. Review practical effect size and consistent directions across metrics before choosing the design.']
    (report/'lgr_local_global_interaction_paired.md').write_text('\n\n'.join(text)+'\n')
    images(rows,report)
    if '--cleanup' in sys.argv:cleanup(matrix,audit,report)
    print('COMPLETE 66 slots;22 new; paired statistics/contact sheets ready; no Batch2')


def images(rows,report):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    specs=json.loads((ROOT/'docs/production/ablation_final_outputs/manifests/paper_ablation_internal_visual_manifest.json').read_text())
    out=report/'contact_sheets';out.mkdir(exist_ok=True);manifest=[]
    for ds,label in [('synthetic_liver','Liver'),('tus_rec_2024_qc','Arm')]:
        spec=next(s for s in specs if s['group']=='Renderer' and s['dataset_label']==('Synthetic' if label=='Liver' else 'TUS-QC'))
        sw='l1' if label=='Liver' else '010_LH_Per_L_DtP_qc_0005_0370';item=spec['item'];crop=spec['crop_xywh'];x,y,w,h=crop
        rr=[r for r in rows if r['dataset']==ds and r['sweep']==sw];lookup={(r['local'],r['global_mode']):r for r in rr}
        for name,keys in [('full_fixed_vs_adaptive',[('fixed','integral'),('adaptive','integral')]),('six_combinations',[(l,g) for l in ('fixed','adaptive') for g in ('none','slice_depth','integral')])]:
            gt=Path(lookup[keys[0]]['output_dir'])/'vis'/('item_'+str(item))/'gt.png';gh=hashlib.sha256(gt.read_bytes()).hexdigest();entries=[]
            for local,glob in keys:
                path=Path(lookup[local,glob]['output_dir'])/'vis'/('item_'+str(item))
                assert hashlib.sha256((path/'gt.png').read_bytes()).hexdigest()==gh
                entries.append((local+' / '+{'none':'None','slice_depth':'Simple','integral':'Full'}[glob],path/'pred.png'))
            entries.append(('GT',gt));fig,axes=plt.subplots(2,len(entries),figsize=(2.8*len(entries),5),squeeze=False)
            for i,(title,path) in enumerate(entries):
                im=np.asarray(Image.open(path).convert('L'))/255.
                for ax,array in [(axes[0,i],im),(axes[1,i],im[y:y+h,x:x+w])]:ax.imshow(array,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
                axes[0,i].set_title(title,fontsize=9)
                manifest.append(dict(dataset=ds,sweep=sw,sheet=name,item=item,crop_xywh=crop,label=title,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gt_sha256=gh))
            fig.tight_layout();fig.savefig(out/(label+'_'+name+'.png'),dpi=180);fig.savefig(out/(label+'_'+name+'.pdf'));plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def cleanup(matrix,audit,report):
    from scripts.production.ablations.ours.p1_checkpoint_cleanup import validated_targets
    count=total=0
    for r in matrix:
        if not r['execution_authorized']:continue
        control=json.loads((OUT/'control'/(r['batch1_id']+'.json')).read_text())
        expected=next(x['expected_test_indices'] for x in audit if x['dataset']==r['dataset'] and x['sweep']==r['sweep'])
        for p in validated_targets(r,control,expected,(OUT/'runs',)):
            entry=dict(row_id=r['batch1_id'],path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),timestamp=time.time(),reason='Interaction metrics,frames and representative exports validated')
            ledger=report/'checkpoint_cleanup.jsonl'
            with ledger.open('a') as f:f.write(json.dumps(dict(entry,event='delete_intent'))+'\n')
            p.unlink()
            with ledger.open('a') as f:f.write(json.dumps(dict(entry,event='deleted'))+'\n')
            count+=1;total+=entry['bytes']
    print('Validated new checkpoint files deleted:',count,'bytes:',total)


if __name__=='__main__':main()
