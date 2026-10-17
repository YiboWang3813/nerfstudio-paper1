"""Final Fixed evidence only; legacy/adaptive quality is never relabeled as final."""
import csv
import hashlib
import json
import math
import shutil
import statistics as st
import sys
from pathlib import Path
from scripts.production.ablations.ours.p2_refresh import ROOT,DOC,OUT,MAIN,FREEZE,MATRIX,guard,effective
from scripts.production.ablations.ours.p1_lgr_report import write
from scripts.production.ablations.ours.summarize_ablation_run import read_series,value_at

METRICS=('psnr','l1','ssim','lpips','dists','gmsd')
HIGH={'psnr','ssim'}


def read(path):return list(csv.DictReader(path.open(),delimiter='\t'))


def collect(r,expected):
    from omegaconf import OmegaConf
    path=Path(r['output_dir']);p=path/'batch1_stats.json'
    if not p.exists():p=path/'anchor_stats.json'
    if not p.exists():return None
    stats=json.loads(p.read_text())
    if not stats.get('success'):return None
    if r['execution_authorized']:
        control=OUT/'control'/(r['batch1_id']+'.json')
        if not control.exists():return None
        assert json.loads(control.read_text())['exit_code']==0 and stats['finite_final_parameters']
        assert stats['source_freeze_sha256']==r['source_freeze_sha256']
    cfg=OmegaConf.to_container(OmegaConf.load(path/'resolved_config.yaml'),resolve=False)
    assert hashlib.sha256(json.dumps(effective(cfg),sort_keys=True).encode()).hexdigest()==r['semantic_config_sha256']
    frame=read(path/'eval/per_frame_metrics_iteration_030000.tsv');assert [int(x['index']) for x in frame]==expected
    assert all(math.isfinite(float(x[m])) for x in frame for m in METRICS)
    s=read_series(path/'logs/scalars.tsv')
    row=dict(dataset=r['dataset'],sweep=r['sweep'],row_id=r['batch1_id'],output_dir=str(path),execution_source_sha256=r['execution_source_sha256'],validated_equivalent_p2_sha256=r['source_freeze_sha256'])
    for m in METRICS:
        row['final_'+m]=value_at(s,'test/'+m,30000);assert isinstance(row['final_'+m],float) and math.isfinite(row['final_'+m])
    row['final_num_gaussians']=value_at(s,'train/num_gaussians',30000);assert row['final_num_gaussians']>0
    control=path.parents[3]/'control'/(path.name+'.json')
    row['train_wall_time_sec']=json.loads(control.read_text())['train_wall_time_sec'] if control.exists() else stats.get('wall_seconds')
    return row


def main_tables(rows,report):
    from scripts.production.ablations.ours.p1_scientific_checkpoint import table
    old=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports'
    final=[]
    for ds in sorted({r['dataset'] for r in rows}):
        rr=[r for r in rows if r['dataset']==ds]
        final.append(dict(dataset=ds,method='ours_fixed_final',n=len(rr),**{'final_'+m:st.mean(r['final_'+m] for r in rr) for m in METRICS},final_num_gaussians=st.mean(r['final_num_gaussians'] for r in rr),train_wall_time_sec=st.mean(r['train_wall_time_sec'] for r in rr)))
    write(report/'main_ours_fixed_final.tsv',final)
    for r in read(old/'main_corrected_table.tsv'):
        if r['method']=='ours_corrected':continue
        final.append(dict(dataset=r['dataset'],method=r['method'],n=int(r['n']),**{'final_'+m:float(r['final_'+m]) for m in METRICS},final_num_gaussians=r['final_num_gaussians'],train_wall_time_sec=r['train_wall_time_sec']))
    assert len(final)==32;write(report/'main_fixed_final_table.tsv',final);rankings=[]
    for ds in sorted({r['dataset'] for r in final}):
        rr=[r for r in final if r['dataset']==ds];assert len(rr)==8 and len({r['n'] for r in rr})==1
        for m in METRICS:
            ordered=sorted(rr,key=lambda r:r['final_'+m],reverse=m in HIGH)
            for r in ordered:
                v=r['final_'+m];rank=1+sum(x['final_'+m]>v if m in HIGH else x['final_'+m]<v for x in rr)
                rankings.append(dict(dataset=ds,method=r['method'],metric=m,raw_mean=v,displayed='{:.4f}'.format(v),rank=rank,best=ordered[0]['method'],second_best=ordered[1]['method']))
    write(report/'main_fixed_final_values_display_ranks.tsv',rankings)
    fps=ROOT/'outputs/production/comparisons/P2_fixed_full_final_20260909/fps_rebuild/reports'
    for name in ('fixed_fps_per_run.tsv','fixed_fps_datasets.tsv','fixed_fps_panels.tsv','replica_vs_original.tsv'):shutil.copy2(fps/name,report/name)
    frows=read(fps/'fixed_fps_datasets.tsv');eff=[]
    for r in final:
        if r['method']=='ours_fixed_final':eff.append(dict(dataset=r['dataset'],time_seconds=r['train_wall_time_sec'],fps=float(next(x['fps_mean'] for x in frows if x['dataset']==r['dataset']))))
    write(report/'main_fixed_efficiency_datasets.tsv',eff)
    panels=[]
    for label,ds in [('synthetic_phantom',('synthetic_liver','spine_phantom')),('real',('tus_rec_2024_qc','tdsc_abus_2023'))]:
        rr=[r for r in eff if r['dataset'] in ds];panels.append(dict(panel=label,time_seconds=st.mean(r['time_seconds'] for r in rr),fps=st.mean(r['fps'] for r in rr)))
    write(report/'main_fixed_efficiency_panels.tsv',panels)
    shutil.copy2(old/'baseline_efficiency_frozen_panels.tsv',report/'baseline_efficiency_frozen_panels.tsv')
    ours=[r for r in rankings if r['method']=='ours_fixed_final']
    text=['# Final Fixed main claim audit','Only Fixed+Full quality is final Ours; same frozen6/9/33/30 cases and seven FINAL-REUSE baselines. No78-case retraining. Ranking uses unrounded values.',table([r for r in final if r['method']=='ours_fixed_final'],list(final[0])),table(ours,list(ours[0])),
          '## Supported / update numbers','PSNR and L1 rank1 on all4 datasets. SSIM rank1 on Liver/Spine/Breast, rank2 on Arm. LPIPS rank1 on Spine/Arm/Breast, rank2 on Liver. DISTS rank1 on Arm/Breast, rank2 on Liver/Spine. GMSD rank1 on all4. All old Adaptive/legacy numerical values and derived percentages must be replaced.',
          '## Weaken / do not claim','Do not claim all metrics rank1. Fixed design trades lower Liver PSNR/L1/SSIM for better LPIPS/DISTS/GMSD; Breast DISTS is systematically worse than Adaptive despite remaining rank1 among baselines. Fixed usually retains more Gaussians; no efficiency-superiority or all-perceptual-gains claim.',
          '## Efficiency',table(eff,list(eff[0])),table(panels,list(panels[0])),'FPS uses separately reconstructed Fixed models for the same two frozen cases per dataset,5warmup/20frames/3passes,pureforward on4090. Replica-vs-original deltas are retained; quality comes from the original78 models, not the replicas. Time uses outer training-process wall seconds and equal-dataset panel means; shared-worker/hardware heterogeneity precludes controlled speedup claims. Frozen baseline efficiency is reused verbatim, including historical aggregation caveats.','Canonical LaTeX is not modified; this report is the final numerical source.']
    (report/'main_fixed_final_claim_audit.md').write_text('\n\n'.join(text)+'\n')
    from scripts.production.comparisons.us_recon_four_dataset_p100k_v1 import make_tmi_all_datasets_2d_rendering_v2 as drawing
    old_sources=json.loads((ROOT/'docs/production/p1_paper_protocol_freeze/legacy_main_selected_rows.json').read_text())
    sources={(r['dataset'],r['sweep'],r['method']):r for r in old_sources}
    for r in rows:sources[r['dataset'],r['sweep'],'ours']=r
    image_dir=report/'qualitative';image_dir.mkdir(exist_ok=True)
    drawing._read_rows=lambda:sources;drawing.FIGURE_DIR=image_dir;drawing.MANIFEST_DIR=image_dir;drawing.CONFIG_PATH=image_dir/'provenance.md'
    drawing.main()
    drawing.CONFIG_PATH.write_text('Final Fixed panels from original78-case quality runs; unchanged frozen baseline/case/frame/crop selections. FPS replicas are not image sources. See main and P2 data-source manifests.\n')
    provenance=image_dir/'paper_all_datasets_2d_rendering_manifest.json'
    value=json.loads(provenance.read_text());value['summary_files']=[str(DOC/MATRIX),str(ROOT/'docs/production/p1_paper_protocol_freeze/legacy_main_selected_rows.json')]
    provenance.write_text(json.dumps(value,indent=2)+'\n')


def main():
    from scripts.production.ablations.ours.p1_scientific_checkpoint import table
    guard();matrix=json.loads((DOC/MATRIX).read_text());consumers=json.loads((DOC/'consumers.json').read_text())
    audit=json.loads((ROOT/'docs/production/p1_paper_protocol_freeze/baseline_case_frame_audit.json').read_text());expected={(r['dataset'],r['sweep']):r['expected_test_indices'] for r in audit}
    report=OUT/'reports';report.mkdir(exist_ok=True,parents=True);mr=MAIN/'reports';mr.mkdir(exist_ok=True,parents=True)
    rows={};pending=[]
    only_main='--main-only' in sys.argv
    for r in matrix:
        if only_main and 'main' not in r['consumers']:continue
        value=collect(r,expected[r['dataset'],r['sweep']])
        if value is None:pending.append(r['batch1_id'])
        else:rows[r['batch1_id']]=value
    write(report/'completed_per_run.tsv',list(rows.values()))
    (report/'progress.json').write_text(json.dumps(dict(completed=len(rows),pending=pending,main_only=only_main),indent=2)+'\n')
    mainrows=[rows[c['row_id']] for c in consumers if c['consumer']=='main'];assert len(mainrows)==78
    main_tables(mainrows,mr)
    if only_main:return
    assert not pending,'No final partial refresh report'
    groups={key:[] for key in ('joint','ppp','multi','rho','cga','lgr')}
    for c in consumers:
        if '/' not in c['consumer']:continue
        group,label=c['consumer'].split('/',1);groups[group].append(dict(rows[c['row_id']],label=label))
    for name,rr in groups.items():write(report/(name+'_per_case.tsv'),rr)
    assert [len(groups[k]) for k in ('joint','ppp','multi','rho','cga','lgr')]==[16,8,32,14,10,55]
    lgr=[];multi=[]
    for ds in sorted({r['dataset'] for r in groups['lgr']}):
        for label in ('Plane','Fixed','Adaptive','Fixed+Simple','Fixed+Full'):
            rr=[r for r in groups['lgr'] if r['dataset']==ds and r['label']==label]
            lgr.append(dict(dataset=ds,label=label,n=len(rr),**{m:st.mean(r['final_'+m] for r in rr) for m in ('psnr','ssim','lpips')}))
    for ds in sorted({r['dataset'] for r in groups['multi']}):
        for label in ('MP','PPP'):
            rr=[r for r in groups['multi'] if r['dataset']==ds and r['label']==label]
            multi.append(dict(dataset=ds,label=label,n=len(rr),PSNR_mean=st.mean(r['final_psnr'] for r in rr),PSNR_std=st.stdev(r['final_psnr'] for r in rr),SSIM_mean=st.mean(r['final_ssim'] for r in rr),SSIM_std=st.stdev(r['final_ssim'] for r in rr)))
    write(report/'lgr_final_multisweep.tsv',lgr);write(report/'ppp_multisweep_mean_std.tsv',multi)
    effects=[]
    for ds in sorted({r['dataset'] for r in groups['joint']}):
        rr={r['label']:r for r in groups['joint'] if r['dataset']==ds}
        for pos,module in ((1,'LGR'),(3,'CGA'),(5,'PPP')):
            for m in ('psnr','ssim','lpips'):
                delta=[]
                for label,r in rr.items():
                    if label[pos]!='0':continue
                    other=label[:pos]+'1'+label[pos+1:];delta.append(rr[other]['final_'+m]-r['final_'+m])
                effects.append(dict(dataset=ds,module=module,metric=m,mean_effect=st.mean(delta),min_effect=min(delta),max_effect=max(delta)))
    write(report/'joint_paired_effects.tsv',effects)
    population=[];events=[];totals=[]
    for r in groups['cga']:
        s=read_series(Path(r['output_dir'])/'logs/scalars.tsv')
        population.extend(dict(dataset=r['dataset'],label=r['label'],iteration=step,num_gaussians=n) for step,n in s['train/num_gaussians'])
        event={}
        for key,series in s.items():
            if key.startswith('train/dc/'):
                for step,v in series:
                    if step%2000==0:event.setdefault(step,{})[key.split('/')[-1]]=v
        spawned=pruned=protected=0
        for step,e in sorted(event.items()):
            if 'num_gaussians_before' not in e:continue
            add=int(e['num_gaussians_after']-e['num_gaussians_before']+e['prune_weak']);assert add>=0
            spawned+=add;pruned+=int(e['prune_weak']);protected+=int(e.get('path_protected_candidate_count',0))
            events.append(dict(dataset=r['dataset'],label=r['label'],iteration=step,event_json=json.dumps(e,sort_keys=True),spawned=add,pruned=int(e['prune_weak']),path_protected_candidate_decisions=int(e.get('path_protected_candidate_count',0))))
        assert 1000000+spawned-pruned==int(r['final_num_gaussians'])
        totals.append(dict(dataset=r['dataset'],label=r['label'],final_G=r['final_num_gaussians'],final_G_k=r['final_num_gaussians']/1000,total_spawned=spawned,total_pruned=pruned,path_protected_candidate_decisions=protected,PSNR=r['final_psnr'],LPIPS=r['final_lpips']))
    write(report/'cga_population.tsv',population);write(report/'cga_events.tsv',events);write(report/'cga_event_totals.tsv',totals)
    differences=[]
    for ds in sorted({r['dataset'] for r in totals}):
        a=next(r for r in totals if r['dataset']==ds and r['label']=='RGD+LP');b=next(r for r in totals if r['dataset']==ds and r['label']=='Full')
        differences.append(dict(dataset=ds,extra_final_G=b['final_G']-a['final_G'],delta_PSNR=b['PSNR']-a['PSNR'],delta_LPIPS=b['LPIPS']-a['LPIPS']))
    write(report/'cga_full_minus_lp.tsv',differences)
    figures(groups,population,report)
    contact_sheets(groups,report)
    fields=['dataset','label','final_psnr','final_ssim','final_lpips','final_num_gaussians']
    text=['# P2 final Fixed refresh results','Main78 and LGR55slots are exact-config reuse. New structural runs use P2; reused rows preserve their original execution source hashes with a separate P2 equivalence certification. No Batch2 launched.','## Joint',table(groups['joint'],fields),table(effects,list(effects[0])),'LGR off is Plane(no conditioning,no global), not the former Adaptive local-only setting. CGA off is gradient+opacity; no-global CGA on degrades to local-aware.','## PPP representative',table(groups['ppp'],fields),'## PPP multi-sweep',table(multi,list(multi[0])),'Only PSNR/SSIM are claimed here; no all-perceptual-invariance claim.','## Rho',table(groups['rho'],fields),'rho1 is fixed nominal spacing, not a test-selected optimum.','## CGA',table(totals,list(totals[0])),table(differences,list(differences[0])),'Protected decisions are event-summed, not unique Gaussian identities.','## LGR',table(lgr,list(lgr[0])),'Plane→Fixed tests finite slice. Fixed/Adaptive compares alternatives, not progressive modules. Fixed→Simple→Full tests global formulations; no monotonicity or independent acoustic-attribute claim.','All raw data, configs,source lineage and exported images are preserved. Await scientific review. No automatic Hyperparameter Batch2.']
    (report/'P2_FINAL_REFRESH.md').write_text('\n\n'.join(text)+'\n')
    manifest=[dict(c,output_dir=rows[c['row_id']]['output_dir'],execution_source_sha256=rows[c['row_id']]['execution_source_sha256'],validated_equivalent_p2_sha256=rows[c['row_id']]['validated_equivalent_p2_sha256']) for c in consumers]
    (report/'paper_final_data_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('P2 FINAL REFRESH COMPLETE. Checkpoints retained. PAUSE; no Batch2.')


def contact_sheets(groups,report):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    joint=json.loads((ROOT/'docs/production/ablation_final_outputs/manifests/paper_ablation_joint_visual_manifest.json').read_text())
    internal=json.loads((ROOT/'docs/production/ablation_final_outputs/manifests/paper_ablation_internal_visual_manifest.json').read_text())
    dest=report/'contact_sheets';dest.mkdir(exist_ok=True);manifest=[]
    for spec in joint:
        ds=spec['dataset'];sw=spec['sweep']
        for group,label in [('joint',None),('ppp','Easy-Pose'),('cga','DC'),('lgr','Renderer')]:
            s=spec if label is None else next(s for s in internal if s['group']==label and s['dataset_label']==('Synthetic' if ds=='synthetic_liver' else 'TUS-QC'))
            rr=[r for r in groups[group] if r['dataset']==ds and r['sweep']==sw];item=s['item'];x,y,w,h=s['crop_xywh']
            entries=[];gt_hash=None
            for r in rr:
                path=Path(r['output_dir'])/'vis'/('item_'+str(item));digest=hashlib.sha256((path/'gt.png').read_bytes()).hexdigest()
                if gt_hash is None:gt_hash=digest
                assert gt_hash==digest
                entries.append((r['label'],path/'pred.png'))
            entries.append(('GT',path/'gt.png'));fig,axes=plt.subplots(2,len(entries),figsize=(2.5*len(entries),5),squeeze=False)
            for i,(name,p) in enumerate(entries):
                im=np.asarray(Image.open(p).convert('L'))/255.
                for ax,array in ((axes[0,i],im),(axes[1,i],im[y:y+h,x:x+w])):ax.imshow(array,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
                axes[0,i].set_title(name,fontsize=9)
                manifest.append(dict(group=group,dataset=ds,sweep=sw,label=name,item=item,crop_xywh=s['crop_xywh'],source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gt_sha256=gt_hash))
            fig.tight_layout()
            for ext in ('png','pdf'):fig.savefig(dest/(group+'_'+ds+'.'+ext),dpi=180)
            plt.close(fig)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def figures(groups,population,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,ds in zip(axes,sorted({r['dataset'] for r in population})):
        for label in ('GD','RGD','RGD+OP','RGD+LP','Full'):
            rr=sorted([r for r in population if r['dataset']==ds and r['label']==label],key=lambda r:r['iteration'])
            ax.step([r['iteration'] for r in rr],[r['num_gaussians']/1e6 for r in rr],where='post',label=label)
        ax.set_title(ds);ax.set_xlabel('Iteration');ax.set_ylabel('Gaussians (million)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(report/('cga_population.'+ext),dpi=180)
    plt.close(fig);fig,axes=plt.subplots(2,3,figsize=(10,5.5))
    for i,ds in enumerate(sorted({r['dataset'] for r in groups['rho']})):
        rr=sorted([r for r in groups['rho'] if r['dataset']==ds],key=lambda r:float(r['label']))
        for j,m in enumerate(('psnr','ssim','lpips')):
            ax=axes[i,j];ax.plot([float(r['label']) for r in rr],[r['final_'+m] for r in rr],'-o');ax.axvline(1.,color='gray',ls='--');ax.set_title(ds+' '+m);ax.set_xlabel('rho');ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(report/('rho_sensitivity.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
