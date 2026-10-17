"""Aggregate authorized corrected artifacts; incomplete datasets never become final rows."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.production.ablations.ours.summarize_ablation_run import read_series, value_at

ROOT=Path(__file__).resolve().parents[4]
PROTOCOL=ROOT/'docs/production/p1_paper_protocol_freeze'
OUT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908'
METRICS=('psnr','l1','ssim','lpips','dists','gmsd')
HIGH={'psnr','ssim'}


def efficiency_metadata(rows):
    """Missing efficiency never excludes a quality case or becomes zero."""
    values=[]
    for row in rows:
        try:
            value=float(row.get('train_wall_time_sec', ''))
        except (ValueError, TypeError):
            continue
        if math.isfinite(value) and value > 0:
            values.append(value)
    complete=len(values)==len(rows)
    return dict(train_wall_time_sec=statistics.mean(values) if complete and values else '',
                time_available_cases=len(values),time_expected_cases=len(rows),
                available_case_time_mean=statistics.mean(values) if values else '',
                time_status='complete' if complete else 'incomplete_not_paper_aggregate')


def tsv(path, rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def key(row):
    return (row['dataset'],row['sweep'],json.dumps(row['settings'],sort_keys=True))


def main():
    report=OUT/'reports';report.mkdir(parents=True,exist_ok=True)
    matrix=json.loads((PROTOCOL/'batch1_matrix.json').read_text())
    old=json.loads((PROTOCOL/'legacy_main_selected_rows.json').read_text())
    latest_path=ROOT/'outputs/production/ablations/ours/path_aware_pruning_update_20260831/main_ours_gpu3/tables/ours_per_run.tsv'
    latest={(r['dataset'],r['sweep']):r for r in csv.DictReader(latest_path.open(),delimiter='\t')}
    audit=json.loads((PROTOCOL/'baseline_case_frame_audit.json').read_text())
    available, pending={},[]
    for row in matrix:
        run=Path(row['output_dir'])
        path=run/('anchor_stats.json' if row['target_status']=='FINAL-REUSE' else 'batch1_stats.json')
        if not path.exists() or not json.loads(path.read_text()).get('success'):
            pending.append(row['batch1_id']);continue
        stats=json.loads(path.read_text())
        series=read_series(run/'logs/scalars.tsv')
        metrics={m:value_at(series,'test/'+m,30000) for m in METRICS}
        assert all(v is not None and math.isfinite(float(v)) for v in metrics.values()),row['batch1_id']
        ng=value_at(series,'train/num_gaussians',30000)
        assert ng is not None
        frames=list(csv.DictReader((run/'eval/per_frame_metrics_iteration_030000.tsv').open(),delimiter='\t'))
        # Frozen expected frame IDs are independent of pose ablation mode.
        expected=next(r['expected_test_indices'] for r in audit if r['dataset']==row['dataset'] and r['sweep']==row['sweep'])
        assert [int(r['index']) for r in frames]==expected,row['batch1_id']
        control=OUT/'control'/(row['batch1_id']+'.json')
        wall=json.loads(control.read_text())['train_wall_time_sec'] if control.exists() else stats.get('wall_seconds',stats.get('wrapper_wall_seconds'))
        r=dict(row_id=row['batch1_id'],group=row['group'],variant=row['variant'],dataset=row['dataset'],sweep=row['sweep'],
               output_dir=str(run),final_step=30000,final_num_gaussians=ng,train_wall_time_sec=wall,
               peak_local_candidates=stats['peak_local_tile_gaussian_pairs'],peak_ray_candidates=stats['peak_global_ray_gaussian_pairs'],
               peak_allocated_bytes=stats['peak_allocated_bytes'])
        r.update({'final_'+k:v for k,v in metrics.items()})
        available[row['batch1_id']]=r
    tsv(report/'completed_per_run.tsv',list(available.values()))
    matched=[]
    for r in available.values():
        if r['group']!='main':continue
        baseline=latest[r['dataset'],r['sweep']]
        entry=dict(dataset=r['dataset'],sweep=r['sweep'],corrected_run=r['output_dir'],matched_legacy_run=baseline['output_dir'])
        for metric in METRICS+('num_gaussians',):
            entry['corrected_'+metric]=r['final_'+metric]
            entry['matched_legacy_'+metric]=float(baseline['final_'+metric])
            entry['delta_'+metric]=r['final_'+metric]-float(baseline['final_'+metric])
        matched.append(entry)
    tsv(report/'main_matched_legacy_per_run_deltas.tsv',matched)
    write_json(report/'progress.json',dict(total_unique=len(matrix),completed=len(available),
         corrected_anchor_reuse=2,new_training_completed=len(available)-2,remaining=len(pending),pending=pending,
         final_scientific_checkpoint_ready=not pending))
    fps=[dict(benchmark='paper_final_corrected_covariance',stage='Batch1',dataset=r['dataset'],sweep=r['sweep'],method='ours',
              final_step=30000,final_num_gaussians=r['final_num_gaussians'],output_dir=r['output_dir'])
         for r in available.values() if r['group']=='main']
    tsv(report/'main_fps_input.tsv',fps)
    dataset_results=[]
    for ds in sorted({r['dataset'] for r in matrix if r['group']=='main'}):
        selected=[r for r in matrix if r['group']=='main' and r['dataset']==ds]
        if any(r['batch1_id'] not in available for r in selected):
            continue
        corrected=[available[r['batch1_id']] for r in selected]
        dataset_results.append(dict(dataset=ds,method='ours_corrected',n=len(corrected),
          **{'final_'+m:statistics.mean(r['final_'+m] for r in corrected) for m in METRICS},
          final_num_gaussians=statistics.mean(r['final_num_gaussians'] for r in corrected),
          train_wall_time_sec=statistics.mean(r['train_wall_time_sec'] for r in corrected)))
        for method in sorted({r['method'] for r in old}):
            legacy=[r for r in old if r['dataset']==ds and r['method']==method]
            assert {r['sweep'] for r in legacy}=={r['sweep'] for r in selected}
            dataset_results.append(dict(dataset=ds,method='ours_legacy' if method=='ours' else method,n=len(legacy),
             **{'final_'+m:statistics.mean(float(r['final_'+m]) for r in legacy) for m in METRICS},
             **efficiency_metadata(legacy)))
    tsv(report/'main_complete_dataset_aggregates.tsv',dataset_results)
    rankings=[]
    for ds in sorted({r['dataset'] for r in dataset_results}):
        group=[r for r in dataset_results if r['dataset']==ds and r['method']!='ours_legacy']
        for m in METRICS:
            ordered=sorted(group,key=lambda r:r['final_'+m],reverse=m in HIGH)
            rankings.append(dict(dataset=ds,metric=m,best=ordered[0]['method'],second=ordered[1]['method'],
                corrected_rank=next(i+1 for i,r in enumerate(ordered) if r['method']=='ours_corrected'),
                ranking=' > '.join(r['method'] for r in ordered)))
    tsv(report/'main_rankings.tsv',rankings)
    corrected_rows=[r for r in dataset_results if r['method']=='ours_corrected']
    if len(corrected_rows)==4:
        tsv(report/'main_ours_corrected_row.tsv',corrected_rows)
        proposal=['% Corrected quality-only Ours rows; canonical table integration pending.',
                  '% Columns: dataset, PSNR, L1, SSIM, LPIPS, DISTS, GMSD. FPS is separately gated.']
        for r in corrected_rows:
            proposal.append(r['dataset'].replace('_',r'\_')+' & '+' & '.join('{:.4f}'.format(r['final_'+m]) for m in METRICS)+r' \\')
        (report/'main_ours_corrected_quality_rows.tex').write_text('\n'.join(proposal)+'\n')
        timing=[]
        for label,pair in [('synthetic_phantom',('synthetic_liver','spine_phantom')),('real',('tus_rec_2024_qc','tdsc_abus_2023'))]:
            for method in sorted({r['method'] for r in dataset_results}):
                members=[r for r in dataset_results if r['method']==method and r['dataset'] in pair]
                assert len(members)==2
                complete=all(isinstance(r['train_wall_time_sec'],(int,float)) for r in members)
                timing.append(dict(group=label,method=method,equal_dataset_mean_train_seconds=statistics.mean(r['train_wall_time_sec'] for r in members) if complete else '',
                    time_status='complete' if complete else 'incomplete_use_frozen_baseline_panel',
                    timing_caveat='4080SUPER plus reused4090 anchors' if method=='ours_corrected' else 'historical4090 concurrency'))
        tsv(report/'equal_dataset_training_time.tsv',timing)
    lookup={key(r):r for r in matrix}

    def locate(ds,sw,**changes):
        settings=dict(pose='file',spacing='uniform',rho=1.,local='adaptive',global_mode='integral',dense='residual_guided',prune='path_aware')
        settings.update(changes)
        if settings['global_mode']=='none' and settings['prune']=='path_aware':
            settings['prune']='local_aware'
        row=lookup[(ds,sw,json.dumps(settings,sort_keys=True))]
        return available.get(row['batch1_id'])

    representatives=[('synthetic_liver','l1'),('tus_rec_2024_qc','010_LH_Per_L_DtP_qc_0005_0370')]
    joint,ppp,lgr,cga,rho,population=[],[],[],[],[],[]
    for ds,sw in representatives:
        for p in (0,1):
            for l in (0,1):
                for c in (0,1):
                    r=locate(ds,sw,pose='canonical_sweep' if p else 'file',global_mode='integral' if l else 'none',dense='residual_guided' if c else 'gradient',prune='path_aware' if c else 'opacity')
                    if r:joint.append(dict(r,PPP=p,LGR=l,CGA=c))
        for label,pose,spacing in [('MP','file','uniform'),('PP-S','canonical_sweep','file_step_norm'),('PP-A','canonical_sweep','file_axis_projection'),('PPP','canonical_sweep','uniform')]:
            r=locate(ds,sw,pose=pose,spacing=spacing)
            if r:ppp.append(dict(r,label=label))
        for label,local,glob in [('Plane','none','none'),('Fixed','fixed','none'),('Adaptive','adaptive','none'),('Simple','adaptive','slice_depth'),('Full','adaptive','integral')]:
            r=locate(ds,sw,local=local,global_mode=glob)
            if r:lgr.append(dict(r,label=label))
        for label,dense,prune in [('GD','gradient','opacity'),('RGD','residual_guided','none'),('RGD+OP','residual_guided','opacity'),('RGD+LP','residual_guided','local_aware'),('Full','residual_guided','path_aware')]:
            r=locate(ds,sw,dense=dense,prune=prune)
            if r:
                cga.append(dict(r,label=label))
                with (Path(r['output_dir'])/'logs/scalars.tsv').open() as f:
                    population.extend(dict(dataset=ds,label=label,iteration=int(x['step']),num_gaussians=float(x['value'])) for x in csv.DictReader(f,delimiter='\t') if x['key']=='train/num_gaussians')
        for value in (.6,.75,.9,1.,1.1,1.25,1.5):
            r=locate(ds,sw,pose='canonical_sweep',rho=value)
            if r:rho.append(dict(r,rho=value))
    for name,rows in [('joint',joint),('ppp_internal',ppp),('lgr_structural',lgr),('cga_structural',cga),('rho',rho),('population',population)]:
        tsv(report/(name+'.tsv'),rows)
    multi=[]
    selected=json.loads((PROTOCOL/'protocol_summary.json').read_text())['selected_multi_sweep']
    for ds,sw in selected:
        mp=locate(ds,sw);pseudo=locate(ds,sw,pose='canonical_sweep')
        if mp and pseudo:
            multi.append(dict(dataset=ds,sweep=sw,MP_PSNR=mp['final_psnr'],PPP_PSNR=pseudo['final_psnr'],
             MP_SSIM=mp['final_ssim'],PPP_SSIM=pseudo['final_ssim'],delta_PSNR=pseudo['final_psnr']-mp['final_psnr'],delta_SSIM=pseudo['final_ssim']-mp['final_ssim']))
    tsv(report/'ppp_multisweep_paired.tsv',multi)
    summaries=[]
    for ds in sorted({d for d,s in selected}):
        pairs=[r for r in multi if r['dataset']==ds]
        if len(pairs)!=sum(d==ds for d,s in selected):continue
        r=dict(dataset=ds,n=len(pairs))
        for field in ('MP_PSNR','PPP_PSNR','MP_SSIM','PPP_SSIM','delta_PSNR','delta_SSIM'):
            vals=[x[field] for x in pairs];r[field+'_mean']=statistics.mean(vals);r[field+'_std']=statistics.stdev(vals)
        summaries.append(r)
    tsv(report/'ppp_multisweep_mean_std.tsv',summaries)
    qualitative=[]
    for ds,(sw,item) in json.loads((PROTOCOL/'protocol_summary.json').read_text())['qualitative'].items():
        new=locate(ds,sw)
        if not new:continue
        gt=Path(new['output_dir'])/'vis'/('item_'+str(item))/'gt.png'
        oldrows=[r for r in old if r['dataset']==ds and r['sweep']==sw]
        gh=hashlib.sha256(gt.read_bytes()).hexdigest()
        assert all(hashlib.sha256((Path(r['output_dir'])/'vis'/('item_'+str(item))/'gt.png').read_bytes()).hexdigest()==gh for r in oldrows)
        qualitative.append(dict(dataset=ds,sweep=sw,item=item,corrected=str(gt.parent),gt_sha256=gh,baseline_gt_identical=True))
    write_json(report/'qualitative_paths.json',qualitative)
    print(json.dumps(dict(completed=len(available),pending=len(pending),complete_datasets=len(dataset_results)//9),indent=2))


if __name__=='__main__':
    main()
