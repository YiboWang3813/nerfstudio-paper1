"""Paired design review only: six metrics, population, engineering and frozen crops."""
import csv
import hashlib
import json
import math
import statistics as st
import sys
import time
from pathlib import Path
from scripts.production.ablations.ours.p1_fixed_candidate import ROOT,DOC,OUT,MATRIX,effective,base
from scripts.production.ablations.ours.p1_lgr_report import write

METRICS=('psnr','l1','ssim','lpips','dists','gmsd')
HIGHER={'psnr','ssim'}
COUNTS={'synthetic_liver':6,'spine_phantom':9,'tus_rec_2024_qc':33,'tdsc_abus_2023':30}


def paired(delta,metric):
    assert metric in METRICS and len(delta)>1 and all(math.isfinite(d) for d in delta)
    oriented=[d if metric in HIGHER else -d for d in delta]
    return dict(n=len(delta),mean_delta=st.mean(delta),std_paired_delta=st.stdev(delta),median_delta=st.median(delta),
                wins=sum(d>1e-6 for d in oriented),ties=sum(abs(d)<=1e-6 for d in oriented),losses=sum(d< -1e-6 for d in oriented),tie_absolute_tolerance=1e-6)


def rank(value,baselines,metric):
    return 1+sum(b>value if metric in HIGHER else b<value for b in baselines)


def read(path):
    return list(csv.DictReader(path.open(),delimiter='\t'))


def collect(row,local,expected):
    from omegaconf import OmegaConf
    from scripts.production.ablations.ours.summarize_ablation_run import read_series,value_at
    path=Path(row['output_dir'] if local=='fixed' else row['adaptive_output_dir'])
    sp=path/'batch1_stats.json'
    if not sp.exists():sp=path/'anchor_stats.json'
    if not sp.exists():return None
    state=json.loads(sp.read_text())
    if not state.get('success'):return None
    assert state['finite_final_parameters']
    digest=row['semantic_config_sha256'] if local=='fixed' else row['adaptive_semantic_config_sha256']
    cfg=OmegaConf.to_container(OmegaConf.load(path/'resolved_config.yaml'),resolve=False)
    assert hashlib.sha256(json.dumps(effective(cfg),sort_keys=True).encode()).hexdigest()==digest
    if sp.name=='batch1_stats.json':
        assert state['source_freeze_sha256']==base.SHA and state['semantic_config_sha256']==digest
    if local=='fixed' and row['execution_authorized']:
        control=OUT/'control'/(row['batch1_id']+'.json')
        if not control.exists():return None
        assert json.loads(control.read_text())['exit_code']==0
    series=read_series(path/'logs/scalars.tsv')
    result=dict(dataset=row['dataset'],sweep=row['sweep'],local=local,output_dir=str(path),semantic_config_sha256=digest,source_freeze_sha256=base.SHA,gpu=state.get('gpu','see manifest'))
    for metric in METRICS:
        result[metric]=value_at(series,'test/'+metric,30000);assert result[metric] is not None and math.isfinite(result[metric])
    frames=read(path/'eval/per_frame_metrics_iteration_030000.tsv')
    assert [int(r['index']) for r in frames]==expected
    assert all(math.isfinite(float(r[m])) for r in frames for m in METRICS)
    result['final_G']=value_at(series,'train/num_gaussians',30000)
    assert result['final_G']>0 and math.isfinite(result['final_G'])
    result['training_wrapper_wall_seconds']=state.get('wrapper_wall_seconds',state.get('wall_seconds'))
    for k in ('peak_local_tile_gaussian_pairs','peak_global_ray_gaussian_pairs','peak_allocated_bytes','peak_reserved_bytes'):
        result[k]=state[k];assert math.isfinite(result[k]) and result[k]>=0
    return result


def images(rows,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from scripts.production.comparisons.us_recon_four_dataset_p100k_v1 import make_tmi_all_datasets_2d_rendering_v2 as drawing
    out=report/'contact_sheets';out.mkdir(exist_ok=True);manifest=[]
    lookup={(r['dataset'],r['sweep'],r['local']):r for r in rows}
    for spec in drawing.ROWS:
        x,y,w,h=spec.crop;entries=[];hashes=[]
        for mode in ('fixed','adaptive'):
            path=Path(lookup[spec.dataset,spec.sweep,mode]['output_dir'])/'vis'/('item_'+str(spec.item))
            hashes.append(hashlib.sha256((path/'gt.png').read_bytes()).hexdigest())
            entries.append((mode.title()+' + Full',path/'pred.png'))
        assert len(set(hashes))==1
        entries.append(('GT',path/'gt.png'));fig,axes=plt.subplots(2,3,figsize=(8.4,5),squeeze=False)
        for i,(label,p) in enumerate(entries):
            im=np.asarray(Image.open(p).convert('L'))/255.
            for ax,array in [(axes[0,i],im),(axes[1,i],im[y:y+h,x:x+w])]:
                assert array.size;ax.imshow(array,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
            axes[0,i].set_title(label,fontsize=10)
            manifest.append(dict(dataset=spec.dataset,sweep=spec.sweep,label=label,item=spec.item,crop_xywh=list(spec.crop),source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gt_sha256=hashes[0]))
        fig.tight_layout();name=spec.dataset+'_fixed_vs_adaptive';fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    assert len(manifest)==12
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def cleanup(matrix,expected,report):
    from scripts.production.ablations.ours.p1_checkpoint_cleanup import validated_targets
    n=total=0
    for row in matrix:
        if not row['execution_authorized']:continue
        c=json.loads((OUT/'control'/(row['batch1_id']+'.json')).read_text())
        for p in validated_targets(row,c,expected[row['dataset'],row['sweep']],(OUT/'runs',)):
            e=dict(row_id=row['batch1_id'],path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),timestamp=time.time(),reason='Candidate metrics/configs/exports/population/provenance verified; no FPS authorized')
            with (report/'checkpoint_cleanup.jsonl').open('a') as f:f.write(json.dumps(dict(e,event='delete_intent'))+'\n')
            p.unlink()
            with (report/'checkpoint_cleanup.jsonl').open('a') as f:f.write(json.dumps(dict(e,event='deleted'))+'\n')
            n+=1;total+=e['bytes']
    print('Validated checkpoint cleanup:',n,'files,',total,'bytes')


def main():
    from scripts.production.ablations.ours.p1_scientific_checkpoint import table
    base.guard()
    for p,h in json.loads((DOC/'submission_hashes.json').read_text()).items():assert base.sha(ROOT/p)==h,p
    matrix=json.loads((DOC/MATRIX).read_text());assert len(matrix)==78
    audit=json.loads((base.OLD/'baseline_case_frame_audit.json').read_text())
    expected={(r['dataset'],r['sweep']):r['expected_test_indices'] for r in audit}
    report=OUT/'reports';report.mkdir(exist_ok=True,parents=True);rows=[];pending=[]
    for row in matrix:
        for mode in ('fixed','adaptive'):
            r=collect(row,mode,expected[row['dataset'],row['sweep']])
            if r is None:pending.append(row['batch1_id']+'/'+mode)
            else:rows.append(r)
    (report/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=156,pending=pending),indent=2)+'\n')
    if rows:write(report/'per_case.tsv',rows)
    assert not pending,'No final aggregate from an incomplete candidate matrix'
    summaries=[];pairs=[];engineering=[];ranking=[]
    frozen=read(ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports/main_corrected_table.tsv')
    for ds,n in COUNTS.items():
        rr=[r for r in rows if r['dataset']==ds];cases=sorted({r['sweep'] for r in rr});assert len(cases)==n and len(rr)==2*n
        lookup={(r['sweep'],r['local']):r for r in rr}
        baseline=[r for r in frozen if r['dataset']==ds and r['method']!='ours_corrected'];assert len(baseline)==7 and all(int(r['n'])==n for r in baseline)
        current=next(r for r in frozen if r['dataset']==ds and r['method']=='ours_corrected')
        for m in METRICS:
            f=[lookup[sw,'fixed'][m] for sw in cases];a=[lookup[sw,'adaptive'][m] for sw in cases];delta=[x-y for x,y in zip(f,a)]
            assert abs(st.mean(a)-float(current['final_'+m]))<1e-10,'Adaptive aggregate changed'
            summaries.append(dict(dataset=ds,metric=m,fixed_mean=st.mean(f),adaptive_mean=st.mean(a),**paired(delta,m)))
            pairs.extend(dict(dataset=ds,sweep=sw,metric=m,fixed=x,adaptive=y,delta=x-y) for sw,x,y in zip(cases,f,a))
            bs=[float(r['final_'+m]) for r in baseline]
            ranking.append(dict(dataset=ds,metric=m,adaptive_mean=st.mean(a),fixed_mean=st.mean(f),adaptive_rank=rank(st.mean(a),bs,m),fixed_rank=rank(st.mean(f),bs,m)))
        for mode in ('fixed','adaptive'):
            selected=[r for r in rr if r['local']==mode]
            for field in ('final_G','training_wrapper_wall_seconds','peak_local_tile_gaussian_pairs','peak_global_ray_gaussian_pairs','peak_allocated_bytes','peak_reserved_bytes'):
                vals=[r[field] for r in selected];assert all(v is not None and math.isfinite(v) for v in vals)
                engineering.append(dict(dataset=ds,local=mode,field=field,n=n,mean=st.mean(vals),std=st.stdev(vals),median=st.median(vals),minimum=min(vals),maximum=max(vals)))
    write(report/'paired_six_metrics.tsv',summaries);write(report/'per_case_paired_deltas.tsv',pairs);write(report/'representation_efficiency.tsv',engineering);write(report/'main_rank_comparison.tsv',ranking)
    (report/'fixed_vs_adaptive_main_rank_audit.md').write_text('\n\n'.join(['# Fixed candidate vs current Adaptive rank audit','Each design separately replaces the Ours slot among the same seven frozen baselines (eight methods, not nine). Unrounded values, identical 6/9/33/30 cases. This is a candidate audit, not a replacement of the official main table. Rank is 1 plus the number of strictly better methods.',table(ranking,list(ranking[0])),'Rank is not the criterion for selecting the local design. No baseline/Methods/default changes.'])+'\n')
    text=['# Paper-wide paired Fixed+Full candidate validation','78/78 Fixed and 78/78 Adaptive complete:11 Fixed reuse+67 new runs; all Adaptive reused. Corrected P0.5 source, MP,Full global,Full CGA,1M,30k,seed42. Only local conditioning mode differs.',
          'Fixed minus Adaptive; sample standard deviation(ddof1), median paired difference; ties abs<=1e-6. PSNR/SSIM higher is better; L1/LPIPS/DISTS/GMSD lower is better. No significance tests. Arm segments are not independent subjects.',
          '## Six-metric paired results',table(summaries,list(summaries[0])),
          '## Representation and engineering observations',table(engineering,list(engineering[0])),
          'Training-wrapper wall time includes shared-worker/hardware differences and is only an engineering observation. No FPS comparison has been performed. Candidate pair counts are maxima per run; their dataset mean/std and maximum are explicitly distinguished.',
          '## Candidate ranks',table(ranking,list(ranking[0])),
          'Frozen main qualitative cases/crops are used without reselection; see contact_sheets/manifest.json. Await visual/scientific review and F1/F2/F3 decision. No automatic default change, dataset-specific mode, Methods edit, or Batch2 submission.']
    (report/'FIXED_CANDIDATE_PAIRED.md').write_text('\n\n'.join(text)+'\n')
    images(rows,report)
    if '--cleanup' in sys.argv:cleanup(matrix,expected,report)
    print('COMPLETE 78 paired cases /67 new; candidate tables and contact sheets ready. PAUSE.')


if __name__=='__main__':main()
