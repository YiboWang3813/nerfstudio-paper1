"""Strict postprocessing of final sensitivity; never tunes or submits training."""
import json
import math
import shutil
import statistics as st
from pathlib import Path
from scripts.production.ablations.ours import p2_batch2 as batch
from scripts.production.ablations.ours import p2_refresh_report as previous
from scripts.production.ablations.ours.p1_lgr_report import write
from scripts.production.ablations.ours.summarize_ablation_run import read_series

ROOT,DOC,OUT=batch.ROOT,batch.DOC,batch.OUT
NAMES={'synthetic_liver':'Liver','tus_rec_2024_qc':'Arm'}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.production.ablations.ours.p1_scientific_checkpoint import table
    batch.base.guard(); previous.OUT=OUT
    rows=json.loads((DOC/batch.MATRIX).read_text()); points=json.loads((DOC/'panel_points.json').read_text())
    audit=json.loads((ROOT/'docs/production/p1_paper_protocol_freeze/baseline_case_frame_audit.json').read_text())
    expected={(r['dataset'],r['sweep']):r['expected_test_indices'] for r in audit}
    dest=OUT/'reports'; dest.mkdir(parents=True,exist_ok=True)
    collected={}; missing=[]; event_rows=[]
    for r in rows:
        v=previous.collect(r,expected[r['dataset'],r['sweep']])
        if v is None: missing.append(r['batch1_id']); continue
        stats=json.loads((Path(r['output_dir'])/'batch1_stats.json').read_text())
        for key in ['peak_local_tile_gaussian_pairs','peak_global_ray_gaussian_pairs','total_global_ray_gaussian_pairs','forward_calls','peak_allocated_bytes','gpu']:
            v[key]=stats.get(key)
        calls=stats.get('forward_calls',0)
        v['mean_global_candidate_interactions']=stats.get('total_global_ray_gaussian_pairs',0)/calls if calls else None
        series=read_series(Path(r['output_dir'])/'logs/scalars.tsv')
        for key, values in series.items():
            if key.startswith('train/dc/'):
                for step,value in values:
                    if step % 2000==0:
                        event_rows.append(dict(dataset=r['dataset'],row_id=r['batch1_id'],iteration=step,statistic=key,value=value))
        collected[r['batch1_id']]=v
    (dest/'progress.json').write_text(json.dumps(dict(completed=len(collected),expected=len(rows),pending=missing),indent=2)+'\n')
    assert not missing, 'Missing/failed rows: '+str(missing)
    assert len(collected)==68
    write(dest/'completed_per_run.tsv',list(collected.values()))
    write(dest/'adaptation_event_statistics.tsv',event_rows)
    result=[]
    for p in points:
        r=collected[p['row_id']]
        default=next(q for q in points if q['dataset']==p['dataset'] and q['panel']==p['panel'] and q['is_default'])
        d=collected[default['row_id']]
        result.append(dict(r,panel=p['panel'],point=p['point'],is_default=p['is_default'],
                           **{'delta_'+m:r['final_'+m]-d['final_'+m] for m in ['psnr','ssim','lpips']}))
    write(dest/'all_panel_points.tsv',result)
    summaries=[]
    for panel in dict.fromkeys(p['panel'] for p in points):
        subset=[r for r in result if r['panel']==panel];write(dest/(panel+'.tsv'),subset)
        for ds in NAMES:
            rr=[r for r in subset if r['dataset']==ds]; d=next(r for r in rr if r['is_default'])
            s=dict(dataset=ds,panel=panel,n=len(rr),default_point=d['point'])
            for m in ['psnr','ssim','lpips']:
                vals=[r['final_'+m] for r in rr]
                s[m+'_min']=min(vals);s[m+'_max']=max(vals);s[m+'_range']=max(vals)-min(vals)
                s[m+'_default']=d['final_'+m]
            s['G_min']=min(r['final_num_gaussians'] for r in rr);s['G_max']=max(r['final_num_gaussians'] for r in rr)
            summaries.append(s)
    write(dest/'parameter_ranges.tsv',summaries)
    # H2: requested primary draft, plus LPIPS companion to expose trade-offs.
    for metric in ['psnr','lpips']:
        fig,axes=plt.subplots(1,5,figsize=(15,3),constrained_layout=True)
        for ax,(panel,key,values,default) in zip(axes,batch.PANELS):
            for ds in NAMES:
                rr=sorted([r for r in result if r['panel']==panel and r['dataset']==ds],key=lambda r:float(r['point']))
                ax.plot([float(r['point']) for r in rr],[r['delta_'+metric] for r in rr],marker='o',label=NAMES[ds])
            ax.axhline(0,color='gray',lw=.6);ax.axvline(default,color='gray',ls='--',lw=.8)
            ax.set_title(panel);ax.set_xlabel('mm' if panel in ['sigma_ele','half_window'] else 'value')
            ax.set_ylabel('Delta '+metric.upper());ax.legend(fontsize=7)
        for ext in ['png','pdf']:fig.savefig(str(dest/('H2_LGR_delta_'+metric+'.'+ext)),dpi=200)
        plt.close(fig)
    # H3 grids: quality and population have separate raw and plotted views.
    for metric in ['delta_psnr','final_num_gaussians']:
        fig,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
        for i,ds in enumerate(NAMES):
            for j,panel in enumerate(['qd_by_B','qp_by_qA']):
                rr=[r for r in result if r['panel']==panel and r['dataset']==ds]
                xs=sorted({float(r['point'].split('_')[1]) for r in rr});ys=sorted({float(r['point'].split('_')[0]) for r in rr})
                z=np.empty((len(ys),len(xs)))
                for r in rr:
                    y,x=map(float,r['point'].split('_'));z[ys.index(y),xs.index(x)]=r[metric]/(1000 if metric=='final_num_gaussians' else 1)
                ax=axes[i,j]; im=ax.imshow(z,origin='lower',aspect='auto',cmap='viridis')
                ax.set_xticks(range(len(xs)));ax.set_xticklabels(xs);ax.set_yticks(range(len(ys)));ax.set_yticklabels(ys)
                for y in range(len(ys)):
                    for x in range(len(xs)):ax.text(x,y,'{:.3f}'.format(z[y,x]),ha='center',va='center',color='white',fontsize=8)
                d=next(r for r in rr if r['is_default']);dy,dx=map(float,d['point'].split('_'))
                ax.scatter([xs.index(dx)],[ys.index(dy)],s=500,facecolors='none',edgecolors='red',linewidths=2)
                ax.set_title(NAMES[ds]+' '+panel);ax.set_xlabel('B' if j==0 else 'q_A');ax.set_ylabel('q_d' if j==0 else 'q_p')
                fig.colorbar(im,ax=ax,label='G(k)' if metric=='final_num_gaussians' else 'Delta PSNR')
        for ext in ['png','pdf']:fig.savefig(str(dest/('H3_CGA_grids_'+metric+'.'+ext)),dpi=200)
        plt.close(fig)
    # No repeated defaults in population scatter and no artificial PSNR/G ratio.
    scatter={}
    for r in result:
        if r['panel'] in ['qd_by_B','qp_by_qA','tau_alpha']:
            scatter.setdefault(r['row_id'],dict(dataset=r['dataset'],row_id=r['row_id'],label=r['panel']+'/'+r['point'],final_num_gaussians=r['final_num_gaussians'],psnr=r['final_psnr'],lpips=r['final_lpips'],output_dir=r['output_dir']))
    old=batch.base.OUT/'reports'
    for r in previous.read(old/'cga_per_case.tsv'):
        if r['label']=='Full':continue # already represented by exact default
        scatter[r['row_id']]=dict(dataset=r['dataset'],row_id=r['row_id'],label='structural/'+r['label'],final_num_gaussians=float(r['final_num_gaussians']),psnr=float(r['final_psnr']),lpips=float(r['final_lpips']),output_dir=r['output_dir'])
    write(dest/'quality_population_scatter.tsv',list(scatter.values()))
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for i,ds in enumerate(NAMES):
        rr=sorted([r for r in result if r['panel']=='tau_alpha' and r['dataset']==ds],key=lambda r:float(r['point']))
        axes[i,0].plot([float(r['point']) for r in rr],[r['delta_psnr'] for r in rr],'-o');axes[i,0].set_xscale('log');axes[i,0].axvline(1e-4,ls='--',color='gray')
        axes[i,0].set_title(NAMES[ds]+' tau_alpha');axes[i,0].set_ylabel('Delta PSNR')
        for j,metric in enumerate(['psnr','lpips'],1):
            for group,marker in [('structural','s'),('qd_by_B','o'),('qp_by_qA','^'),('tau_alpha','x')]:
                rrr=[r for r in scatter.values() if r['dataset']==ds and r['label'].startswith(group)]
                axes[i,j].scatter([r['final_num_gaussians']/1000 for r in rrr],[r[metric] for r in rrr],label=group,marker=marker,s=25)
            axes[i,j].set_xlabel('Final G(k)');axes[i,j].set_ylabel(metric.upper());axes[i,j].legend(fontsize=6)
    for ext in ['png','pdf']:fig.savefig(str(dest/('H3_CGA_tau_quality_population.'+ext)),dpi=200)
    plt.close(fig)
    for ext in ['png','pdf']:shutil.copy2(old/('rho_sensitivity.'+ext),dest/('H1_PPP_rho.'+ext))
    shutil.copy2(old/'rho_per_case.tsv',dest/'H1_PPP_rho.tsv')
    text=['# Final Batch2 numeric sensitivity (no production retuning)',
          '66 newly trained configs + 2 exact reused defaults; 82 panel slots. MP, Fixed+Full, accumulated-A_g CGA, same P2 source and metrics. B uses .001/.005/.05; sigma uses .8/1/1.5/2 mm to respect the existing .8 mm clamp. No significance testing.',
          table(summaries,list(summaries[0])),
          'Inspect raw LPIPS companion with delta-PSNR; do not infer a universal optimum or broad plateau from a small grid. Check adaptation_event_statistics.tsv for inactive budgets and zero local-contribution thresholds. Wall time is shared-worker engineering metadata, NOT a controlled FPS comparison. No new benchmark per point.',
          'All panels retain raw PSNR/SSIM/LPIPS and default-relative deltas. H3 additionally retains population and raw event statistics. Figures are drafts, not approved paper layouts. PPP H1 is reused unchanged.',
          'No further training is planned or automatically submitted. Final scientific interpretation and qualitative audit must be reviewed before marking the paper evidence complete.']
    (dest/'BATCH2_RESULTS.md').write_text('\n\n'.join(text)+'\n')
    (dest/'data_sources.json').write_text(json.dumps(dict(source_freeze=str(batch.FREEZE),matrix=str(DOC/batch.MATRIX),panels=str(DOC/'panel_points.json'),per_run=list(collected.values()),H1_source=str(old/'rho_per_case.tsv'),structural_population_source=str(old/'cga_per_case.tsv')),indent=2)+'\n')
    print('Batch2 complete:66 NEW success,2 reused; raw data,H1/H2/H3 drafts ready. No new training.')


if __name__=='__main__':report()
