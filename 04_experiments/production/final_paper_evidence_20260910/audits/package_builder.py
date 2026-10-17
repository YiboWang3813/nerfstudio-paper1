"""Package existing final evidence only. No training, algorithm edits or retuning."""
import csv
import hashlib
import json
import math
import shutil
import statistics as st
from collections import defaultdict, Counter
from pathlib import Path
from scripts.production.ablations.ours import p2_batch2 as b
from scripts.production.ablations.ours.p1_scientific_checkpoint import table

ROOT=b.ROOT
DEST=ROOT/'docs/production/final_paper_evidence_20260910'
H=b.OUT/'reports'
S=b.base.OUT/'reports'
M=b.base.MAIN/'reports'
DS={'synthetic_liver':'Liver','spine_phantom':'Spine','tus_rec_2024_qc':'Arm','tdsc_abus_2023':'Breast'}
MET=['psnr','l1','ssim','lpips','dists','gmsd']
SOURCE=[]
READS={}


def read(p):
    if DEST not in p.parents:READS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    for r in rows:
        for k,v in list(r.items()):
            if v in ['True','False']:r[k]=v=='True'
            elif v:
                try:r[k]=float(v)
                except ValueError:pass
    return rows


def cp(src,relative):
    dest=DEST/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest);SOURCE.append(dict(source=str(src),target=relative,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))


def md(relative,text):
    (DEST/relative).write_text(text.strip()+'\n')


def export(relative,rows):
    """Raw JSON/TSV, human display, and apply-ready tabular (no canonical edits)."""
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    path=DEST/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tsv').open('w') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)
    path.with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
    display=[{k:('{:.6f}'.format(v) if isinstance(v,float) else v) for k,v in r.items()} for r in rows]
    path.with_name(path.name+'_display.md').write_text(table(display,fields)+'\n')
    # TeX fragments intended for manual inclusion; no compiler project assumed.
    def esc(v):return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    if not any(k in fields for k in ['output_dir','source_config','overrides']):
        text='\\begin{tabular}{'+'l'*len(fields)+'}\n'+ ' & '.join(map(esc,fields))+r' \\'+'\n\\hline\n'
        for r in display:text+=' & '.join(esc(r.get(k,'')) for k in fields)+r' \\'+'\n'
        path.with_suffix('.tex').write_text(text+'\\end{tabular}\n')


def provenance(r):
    out=r.get('output_dir','')
    return dict(r,source_config=str(Path(out)/'resolved_config.yaml') if out else '')


def wide(rows,metrics):
    result=[]
    for label in dict.fromkeys(r['label'] for r in rows):
        v={'variant':label}
        for ds in ['synthetic_liver','tus_rec_2024_qc']:
            r=next(x for x in rows if x['label']==label and x['dataset']==ds)
            for short,key in metrics:v[DS[ds]+'_'+short]=r[key]
        result.append(v)
    return result


def event_audit(points):
    events=defaultdict(dict)
    for r in read(H/'adaptation_event_statistics.tsv'):
        events[r['row_id'],int(r['iteration'])][r['statistic'].split('/')[-1]]=r['value']
    summaries=[];details=[]
    for r in points:
        if r['panel'] not in ['qd_by_B','qp_by_qA','tau_alpha']:continue
        ee=[(t,v) for (rid,t),v in events.items() if rid==r['row_id']]
        counts=Counter();selected=[];prot=[]
        for t,v in ee:
            n=v['num_gaussians_before'];sel=v['responsibility_selected'];selected.append(sel)
            rec=dict(dataset=DS[r['dataset']],panel=r['panel'],point=r['point'],row_id=r['row_id'],iteration=t)
            rec.update(v)
            if r['panel']=='qd_by_B':
                q,cap=map(float,str(r['point']).split('_'));k=max(1,int(n*cap));positive=v['responsibility_positive']
                expected_q=math.floor((positive-1)*(1-q)+1e-5)+1
                label='budget' if k<expected_q and abs(sel-k)<=2 else 'quantile' if expected_q<k and abs(sel-expected_q)<=2 else 'tie_or_other'
                rec.update(budget_k=k,unique_score_quantile_expected_count=expected_q,limiter_inference=label);counts[label]+=1
            counts['zero_C_threshold']+=v['local_contribution_threshold']==0
            prot.append(v['attenuation_protect_count']/n)
            details.append(rec)
        summaries.append(dict(dataset=DS[r['dataset']],panel=r['panel'],point=r['point'],row_id=r['row_id'],events=len(ee),
            budget_events=counts['budget'],quantile_events=counts['quantile'],tie_or_other_events=counts['tie_or_other'],zero_C_threshold_events=counts['zero_C_threshold'],
            selected_min=min(selected),selected_max=max(selected),path_protection_fraction_min=min(prot),path_protection_fraction_max=max(prot),
            path_protected_candidate_decisions=sum(v['path_protected_candidate_count'] for t,v in ee),
            total_pruned=sum(v['prune_weak'] for t,v in ee),total_spawned=sum(v['num_gaussians_after']-v['num_gaussians_before']+v['prune_weak'] for t,v in ee)))
    export('cga/selection_behavior_summary',summaries)
    export('cga/selection_behavior_events',details)
    return summaries


def curves():
    rho=read(S/'rho_per_case.tsv');allpoints=read(H/'all_panel_points.tsv');rr=[];ranges=[]
    for ds in ['synthetic_liver','tus_rec_2024_qc']:
        sub=[r for r in rho if r['dataset']==ds];d=next(r for r in sub if float(r['label'])==1)
        for r in sub:rr.append(provenance(dict(r,rho=float(r['label']),delta_psnr=r['final_psnr']-d['final_psnr'],delta_lpips=r['final_lpips']-d['final_lpips'])))
        v=dict(dataset=DS[ds])
        for m in ['psnr','ssim','lpips']:
            best=(max if m!='lpips' else min)(sub,key=lambda r:r['final_'+m])
            vals=[r['final_'+m] for r in sub]
            v.update({m+'_range':max(vals)-min(vals),m+'_best_rho':best['label'],m+'_default_minus_best':d['final_'+m]-best['final_'+m]})
        ranges.append(v)
    export('ppp/rho_data',[provenance(r) for r in rr]);export('ppp/rho_ranges',ranges)
    export('lgr/hyperparameter_data',[provenance(r) for r in allpoints if r['panel'] in [p[0] for p in b.PANELS]])
    export('cga/hyperparameter_data',[provenance(r) for r in allpoints if r['panel'] not in [p[0] for p in b.PANELS]])
    for panel,*_ in b.PANELS:export('lgr/'+panel+'_data',[provenance(r) for r in allpoints if r['panel']==panel])
    for panel in ['qd_by_B','qp_by_qA','tau_alpha']:export('cga/'+panel+'_data',[provenance(r) for r in allpoints if r['panel']==panel])
    export('lgr/hyperparameter_ranges',[r for r in read(H/'parameter_ranges.tsv') if r['panel'] in [p[0] for p in b.PANELS]])
    export('cga/hyperparameter_ranges',[r for r in read(H/'parameter_ranges.tsv') if r['panel'] not in [p[0] for p in b.PANELS]])
    event=event_audit(allpoints)
    return rr,allpoints,event


def figures(rho,points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.2,'lines.markersize':4})
    datasets=['synthetic_liver','tus_rec_2024_qc'];colors=['#0072B2','#D55E00'];markers=['o','s']
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(str(DEST/(name+'.'+ext)),dpi=400,bbox_inches='tight',pad_inches=.03)
        plt.close(fig)
    def line(ax,rows,field,metric,default):
        for ds,c,m in zip(datasets,colors,markers):
            sub=sorted([r for r in rows if r['dataset']==ds],key=lambda r:float(r[field]));x=[float(r[field]) for r in sub]
            y=[r[metric] for r in sub];ax.plot(x,y,color=c,marker=m,label=DS[ds])
        ax.axhline(0,color='.6',lw=.6);ax.axvline(default,color='.4',ls=':',lw=1)
        ax.plot([default],[0],marker='*',color='black',ms=8,zorder=6,label='Default')
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig,axes=plt.subplots(2,1,figsize=(3.5,3.7),constrained_layout=True)
    for ax,metric,title in zip(axes,['delta_psnr','delta_lpips'],['(a) Reconstruction','(b) Perceptual trade-off']):
        line(ax,rho,'rho',metric,1);ax.set_title(title,loc='left');ax.set_ylabel('$\\Delta$ PSNR (dB)' if metric=='delta_psnr' else '$\\Delta$ LPIPS');ax.set_xlabel('$\\rho$');ax.set_xticks([.6,.75,.9,1,1.1,1.25,1.5])
    axes[0].legend(ncol=3,loc='best',handlelength=1)
    save(fig,'ppp/H1')
    titles=['Slice width','Composition weight','Attenuation strength','Quadrature order','Integration half-window']
    labels=['$\\sigma_{ele}$ (mm)','$\\beta$','$\\lambda_{ray}$','$N_q$','$w$ (mm)']
    for metric,name in [('delta_psnr','H2'),('delta_lpips','H2_LPIPS_review')]:
        fig,axs=plt.subplots(2,3,figsize=(7.16,4.15),constrained_layout=True)
        for i,(panel,key,values,default) in enumerate(b.PANELS):
            ax=axs.flat[i];line(ax,[r for r in points if r['panel']==panel],'point',metric,default)
            ax.set_title('('+chr(97+i)+') '+titles[i],loc='left');ax.set_xlabel(labels[i]);ax.set_xticks(values)
            ax.set_ylabel('$\\Delta$ PSNR (dB)' if metric=='delta_psnr' else '$\\Delta$ LPIPS')
        axs.flat[5].axis('off');handles,labs=axs.flat[0].get_legend_handles_labels();axs.flat[5].legend(handles,labs,loc='center',frameon=False)
        axs.flat[5].text(.05,.1,'Relative to the fixed default.\nDataset curves are paired by setting.',fontsize=8,transform=axs.flat[5].transAxes)
        save(fig,'lgr/'+name)
    scatter=read(H/'quality_population_scatter.tsv');export('cga/quality_population',scatter)
    # Four evidence classes, each with Liver/Arm rows. One double-column figure.
    fig,axs=plt.subplots(2,4,figsize=(7.16,4.0),constrained_layout=True)
    bound=max(abs(r['delta_psnr']) for r in points if r['panel'] in ['qd_by_B','qp_by_qA'])
    for i,ds in enumerate(datasets):
        for j,panel in enumerate(['qd_by_B','qp_by_qA']):
            ax=axs[i,j];sub=[r for r in points if r['dataset']==ds and r['panel']==panel]
            xs=sorted({float(r['point'].split('_')[1]) for r in sub});ys=sorted({float(r['point'].split('_')[0]) for r in sub});z=np.zeros((3,3))
            for r in sub:
                y,x=map(float,r['point'].split('_'));z[ys.index(y),xs.index(x)]=r['delta_psnr']
            ax.imshow(z,origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
            for iy in range(3):
                    for ix in range(3):ax.text(ix,iy,'{:+.3f}'.format(z[iy,ix]) if abs(z[iy,ix])>=.0005 else '0.000',ha='center',va='center',fontsize=7,color='black' if abs(z[iy,ix])<bound*.6 else 'white')
            d=next(r for r in sub if r['is_default']);dy,dx=map(float,d['point'].split('_'))
            ax.plot(xs.index(dx),ys.index(dy),marker='s',ms=23,mfc='none',mec='black',mew=1.2)
            ax.set_xticks(range(3));ax.set_xticklabels(xs);ax.set_yticks(range(3));ax.set_yticklabels(ys)
            ax.set_xlabel('$B$' if j==0 else '$q_A$');ax.set_ylabel(DS[ds]+' / '+('$q_d$' if j==0 else '$q_p$'))
            ax.set_title(('(a) Densification' if j==0 else '(b) Pruning')+'\n$\\Delta$ PSNR (dB)')
        ax=axs[i,2];sub=sorted([r for r in points if r['panel']=='tau_alpha' and r['dataset']==ds],key=lambda r:float(r['point']))
        ax.plot([float(r['point']) for r in sub],[r['delta_psnr'] for r in sub],'-o',color=colors[i]);ax.set_xscale('log');ax.plot([1e-4],[0],'k*',ms=8)
        ax.set_xticks([5e-5,1e-4,5e-4]);ax.set_xticklabels(['5e-5','1e-4','5e-4']);ax.minorticks_off();ax.set_xlabel('$\\tau_\\alpha$');ax.set_ylabel('$\\Delta$ PSNR (dB)');ax.set_title('(c) Opacity threshold')
        ax=axs[i,3]
        for label,m,c in [('structural','s','#444444'),('qd_by_B','o','#0072B2'),('qp_by_qA','^','#009E73'),('tau_alpha','x','#D55E00')]:
            sub=[r for r in scatter if r['dataset']==ds and r['label'].startswith(label)]
            ax.scatter([r['final_num_gaussians']/1000 for r in sub],[r['psnr'] for r in sub],marker=m,color=c,s=12,label=label)
        default=next(r for r in points if r['dataset']==ds and r['is_default']);ax.plot([default['final_num_gaussians']/1000],[default['final_psnr']],'k*',ms=7)
        ax.set_xlabel('Final #G (k)');ax.set_ylabel('PSNR (dB)');ax.set_title('(d) Quality–population');ax.ticklabel_format(useOffset=False,axis='y')
    handles,_=axs[0,3].get_legend_handles_labels()
    fig.legend(handles,['CGA variants','$q_d \\times B$','$q_p \\times q_A$','$\\tau_\\alpha$'],loc='upper center',bbox_to_anchor=(.5,1.06),ncol=4,frameon=False,fontsize=7)
    save(fig,'cga/H3')
    # Internal review companions retain independent units rather than normalized blending.
    for src,dest in [('H3_CGA_grids_final_num_gaussians','cga/H3_population_review'),('H3_CGA_tau_quality_population','cga/H3_LPIPS_population_review')]:
        for ext in ['png','pdf']:cp(H/(src+'.'+ext),dest+'.'+ext)


def package():
    from omegaconf import OmegaConf
    b.base.guard()
    for folder in ['main','joint','ppp','lgr','cga','qualitative','audits']:(DEST/folder).mkdir(parents=True,exist_ok=True)
    assert json.loads((H/'progress.json').read_text())['pending']==[]
    controls=[json.loads(p.read_text()) for p in (b.OUT/'control').glob('*.json')];assert len(controls)==66 and all(r['exit_code']==0 for r in controls)
    matrix=json.loads((b.DOC/b.MATRIX).read_text());finalmatrix=json.loads((b.base.DOC/b.base.MATRIX).read_text())
    source_rows=[]
    for kind,rows in [('structural_main',finalmatrix),('hyperparameter',matrix)]:
        for r in rows:
            path=Path(r['output_dir']);cfgpath=path/'resolved_config.yaml'
            cfg=OmegaConf.to_container(OmegaConf.load(cfgpath),resolve=False)
            assert hashlib.sha256(json.dumps(b.base.effective(cfg),sort_keys=True).encode()).hexdigest()==r['semantic_config_sha256']
            snapshot='audits/resolved_configs/'+kind+'_'+r['batch1_id']+'.yaml';cp(cfgpath,snapshot)
            source_rows.append(dict(group=kind,row_id=r['batch1_id'],dataset=r['dataset'],sweep=r['sweep'],output_dir=str(path),source_config=str(cfgpath),packaged_config=snapshot,
                semantic_config_sha256=r['semantic_config_sha256'],execution_source_sha256=r['execution_source_sha256'],validated_p2_source_sha256=r['source_freeze_sha256'],consumers=r['consumers']))
    export('audits/run_sources',source_rows)
    cp(b.DOC/b.MATRIX,'audits/batch2_exact_matrix.json');cp(b.base.DOC/b.base.MATRIX,'audits/final_refresh_matrix.json')
    cp(ROOT/'docs/production/p1_paper_protocol_freeze/legacy_main_selected_rows.json','audits/baseline_sources.json')
    cp(ROOT/'docs/production/p1_paper_protocol_freeze/paper_final_dataset_protocol.tsv','audits/paper_final_dataset_protocol.tsv')
    cp(ROOT/'docs/production/p1_paper_protocol_freeze/paper_final_metric_protocol.md','audits/paper_final_metric_protocol.md')
    main=read(M/'main_fixed_final_table.tsv');ranks=read(M/'main_fixed_final_values_display_ranks.tsv')
    completed={r['row_id']:r for r in read(S/'completed_per_run.tsv')}
    for ds,n in [('synthetic_liver',6),('spine_phantom',9),('tus_rec_2024_qc',33),('tdsc_abus_2023',30)]:
        rr=[completed[r['batch1_id']] for r in finalmatrix if r['dataset']==ds and 'main' in r['consumers']]
        assert len(rr)==n
        aggregate=next(r for r in main if r['dataset']==ds and r['method']=='ours_fixed_final')
        for m in MET:assert abs(st.mean(r['final_'+m] for r in rr)-aggregate['final_'+m])<1e-12
    for r in ranks:
        rr=[v for v in main if v['dataset']==r['dataset']];key='final_'+r['metric']
        val=next(v[key] for v in rr if v['method']==r['method']);assert val==r['raw_mean']
        rank=1+sum(v[key]>val if r['metric'] in ['psnr','ssim'] else v[key]<val for v in rr)
        assert rank==r['rank']
    expected={'synthetic_liver':[1,1,1,2,2,1],'spine_phantom':[1,1,1,1,2,1],'tus_rec_2024_qc':[1,1,2,1,1,1],'tdsc_abus_2023':[1,1,1,1,1,1]}
    for ds,rank in expected.items():assert [next(r['rank'] for r in ranks if r['dataset']==ds and r['method']=='ours_fixed_final' and r['metric']==m) for m in MET]==rank
    export('main/final_main_table_data',main);export('main/final_main_rank_data',ranks)
    cp(M/'main_fixed_final_values_display_ranks.tsv','main/final_main_exact_display_ranks.tsv')
    for name in ['fixed_fps_per_run','fixed_fps_datasets','fixed_fps_panels','main_fixed_efficiency_datasets','main_fixed_efficiency_panels','baseline_efficiency_frozen_panels','replica_vs_original']:
        cp(M/(name+'.tsv'),'main/final_'+name+'.tsv')
    export('main/final_fps',read(M/'main_fixed_efficiency_datasets.tsv'))
    cp(M/'main_fixed_final_claim_audit.md','main/final_main_claims.md')
    jr=read(S/'joint_per_case.tsv');jw=wide(jr,[('PSNR','final_psnr'),('SSIM','final_ssim'),('LPIPS','final_lpips')])
    jw=[dict(LGR=int(r['variant'][1]),CGA=int(r['variant'][3]),PPP=int(r['variant'][5]),**r) for r in jw]
    export('joint/final_joint_table',jw)
    export('joint/final_joint_sources',[provenance(r) for r in jr]);cp(S/'joint_paired_effects.tsv','joint/paired_effects.tsv')
    pr=read(S/'ppp_per_case.tsv');export('ppp/final_ppp_table',wide(pr,[('PSNR','final_psnr'),('SSIM','final_ssim'),('LPIPS','final_lpips')]))
    export('ppp/final_ppp_sources',[provenance(r) for r in pr]);export('ppp/final_ppp_multisweep',read(S/'ppp_multisweep_mean_std.tsv'));cp(S/'multi_per_case.tsv','ppp/multisweep_sources.tsv')
    lr=read(S/'lgr_final_multisweep.tsv');export('lgr/final_lgr_table',wide(lr,[('PSNR','psnr'),('SSIM','ssim'),('LPIPS','lpips')]))
    cp(S/'lgr_per_case.tsv','lgr/structural_sources.tsv')
    transitions=[]
    for ds in ['synthetic_liver','tus_rec_2024_qc']:
        sub={r['label']:r for r in lr if r['dataset']==ds}
        for a,z in [('Plane','Fixed'),('Fixed','Adaptive'),('Fixed','Fixed+Simple'),('Fixed+Simple','Fixed+Full')]:
            transitions.append(dict(dataset=DS[ds],transition=a+' -> '+z,**{'delta_'+m:sub[z][m]-sub[a][m] for m in ['psnr','ssim','lpips']}))
    export('lgr/structural_transitions',transitions)
    cr=read(S/'cga_per_case.tsv')
    for r in cr:r['G_k']=r['final_num_gaussians']/1000
    export('cga/final_cga_table',wide(cr,[('PSNR','final_psnr'),('LPIPS','final_lpips'),('G_k','G_k')]))
    export('cga/structural_sources',[provenance(r) for r in cr]);deltas=[]
    for ds in ['synthetic_liver','tus_rec_2024_qc']:
        sub={r['label']:r for r in cr if r['dataset']==ds}
        for a,z in [('GD','RGD'),('RGD','RGD+OP'),('RGD','RGD+LP'),('RGD+LP','Full')]:
            ga=sub[a]['final_num_gaussians'];gz=sub[z]['final_num_gaussians']
            deltas.append(dict(dataset=DS[ds],transition=a+' -> '+z,delta_G=gz-ga,population_percent=100*(gz/ga-1),delta_psnr=sub[z]['final_psnr']-sub[a]['final_psnr'],delta_lpips=sub[z]['final_lpips']-sub[a]['final_lpips']))
    export('cga/structural_deltas',deltas)
    for name in ['cga_population.tsv','cga_population.png','cga_population.pdf','cga_event_totals.tsv','cga_events.tsv']:cp(S/name,'cga/'+name.replace('cga_','',1))
    rho,points,events=curves();figures(rho,points)
    # Package figures at original coordinates, not regenerated/cherry-picked selections.
    for p in (S/'contact_sheets').glob('*'):cp(p,'qualitative/contact_sheets/'+p.name)
    for p in (M/'qualitative').glob('*'):cp(p,'qualitative/main/'+p.name)
    for name in ['final_qualitative_audit.md','qualitative_integrity_checks.json']:cp(b.DOC/name,'qualitative/'+name)
    cp(S/'contact_sheets/manifest.json','qualitative/contact_sheet_manifest.json')
    narrative(rho,points,events,transitions,deltas)
    cp(Path(__file__).resolve(),'audits/package_builder.py')
    files={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='final_data_source_manifest.json'}
    (DEST/'audits/final_data_source_manifest.json').write_text(json.dumps(dict(P2_source_sha256=(b.FREEZE/'snapshot.sha256').read_text().strip(),source_freeze=str(b.FREEZE),run_sources=source_rows,copied_sources=SOURCE,analytical_inputs=READS,package_file_sha256=files,training_submitted=False),indent=2)+'\n')
    print('FINAL PACKAGE:',DEST, 'files',len(files),'NO TRAINING')


def narrative(rho,points,events,transitions,deltas):
    def short(rows,keys):return table([{k:r[k] for k in keys} for r in rows],keys)
    rho_table=short(rho,['dataset','rho','final_psnr','final_ssim','final_lpips','delta_psnr','delta_lpips'])
    md('ppp/analysis.md', '''# PPP: final interpretation

## Representative and multi-sweep semantics

MP uses measured 6-DoF geometry. PP-S uses measured adjacent translation norms;
PP-A uses signed projection on a predefined axis. PPP uses retained indices,
nominal d0*rho spacing, predefined signed axis and reference frame, identity
rotations. Loader bookkeeping/QC may read poses; the claim is only that measured
probe geometry is not used to construct the PPP trajectory. This is not an
unposed raw-data acquisition/QC claim.

Representative table: final_ppp_table.tsv. Multi-sweep: final_ppp_multisweep.tsv
(Liver6, Arm5, Breast5; case-equally weighted mean and sample SD, not SEM).
Default MP/PPP slots are reused from matched final Fixed runs, not Adaptive.
PSNR/SSIM performance is comparable or better on average; do not assert formal
equivalence or all-perceptual preservation. Liver representative LPIPS rises
from .157044 to .182353; Arm changes .112040 to .108021.

## Rho: all points
'''+rho_table+'''

Ranges, per-metric sampled best and signed default-minus-best gaps are in
rho_ranges.tsv. Liver PSNR/SSIM strictly decrease and LPIPS strictly improves
as rho increases (.6→1.5). Arm PSNR/SSIM improve up to1.25 then decline at1.5;
LPIPS decreases up to1.25 then rises. Default Arm PSNR is only .0689481dB below
the sampled best. Liver default is .3395511dB below the sampled PSNR best at.6,
but that low-rho point has worse LPIPS. The LPIPS-best Liver rho is1.5.

SUPPORTED wording: “rho=1 represents the nominal acquisition spacing and is fixed
across datasets without per-sequence tuning, while perturbing the global spacing
scale leads to dataset-dependent reconstruction/perceptual trade-offs.” This is
supported within the two-representative sensitivity protocol. Nominal spacing
comes from stored dataset protocol constants, not confirmed raw device metadata.
Do not call rho1 an empirical optimum, claim a universal plateau, or extrapolate
insensitivity beyond the tested range. H1 shows both PSNR and LPIPS deltas.
''')
    sections=[]
    for panel,*_ in b.PANELS:
        rr=[r for r in points if r['panel']==panel]
        sections.append('## '+panel+'\n\n'+short(rr,['dataset','point','is_default','final_psnr','final_ssim','final_lpips','delta_psnr','delta_ssim','delta_lpips']))
    md('lgr/analysis.md', '''# LGR: structural and numeric interpretation

## Final structural table (Liver6 / Arm5 means)

See final_lgr_table.tsv and structural_sources.tsv. Plane/Fixed/Adaptive are
local-only; Simple/Full both use Fixed, MP and contribution-aware adaptation.
Global-path availability also controls CGA global protection: these are complete
system comparisons, not isolation with a frozen Gaussian population.

'''+short(transitions,list(transitions[0]))+'''

Plane→Fixed has the largest, consistent benefit (PSNR/SSIM/LPIPS improved on all
11 matched cases). Fixed→Adaptive increases Liver PSNR/SSIM but worsens LPIPS;
Arm worsens all three. Choose the shared simpler Fixed design, not a progressive
five-row addition story. Fixed→Simple: Arm improves all three aggregate metrics;
Liver PSNR/SSIM fall slightly while LPIPS improves. Simple→Full: Arm aggregate
PSNR/SSIM/LPIPS further improve; Liver PSNR slightly decreases and SSIM/LPIPS
are nearly unchanged with small unfavorable mean deltas. “Significant” must not
mean statistical significance; no unplanned significance test was conducted.

## Numerical interpretation

- **sigma_ele: materially sensitive, not unstable.** Widths .8/1/1.5/2mm;
  .8 is the existing implementation clamp minimum. Liver PSNR/SSIM rise
  monotonically with width, while LPIPS worsens; Arm all three metrics worsen
  with width. PSNR ranges .5873178/.7704966dB; LPIPS ranges .035017312/.021516102.
  Liver PSNR-best is2mm, Arm PSNR-best .8mm; both LPIPS-best .8mm. At1mm,
  Arm loses only .0446587dB and .002940552 LPIPS relative to.8mm; Liver gains
  .2024098dB over.8 at a .004937364 LPIPS cost. Wider2mm degrades Arm by
  .7258379dB relative to1mm. No tested narrow-width catastrophic failure and no
  evidence below.8mm. Supported: finite-slice width materially affects quality,
  while fixed1mm is a shared cross-domain operating point, not a universal optimum.
- **beta: moderately sensitive, with domain/metric trade-off.** .4/.6/.8/1;
  PSNR ranges .0912602/.2570081dB and LPIPS .016957068/.014544701.
  Increasing beta monotonically decreases Liver PSNR/SSIM but improves LPIPS;
  Arm PSNR/SSIM/LPIPS all improve. Default.8 is an interior shared setting,
  not the best on every metric. Do not lump beta with near-flat lambda/window.
- **lambda_ray: stable over the tested .2–1 range.** PSNR ranges
  .0171847/.0541477dB; LPIPS .001888730/.001020012. Changes are small and
  not uniformly monotonic. Default.55 lies in this stable observed region.
  “Reconstruction remains stable over a broad tested attenuation-strength range”
  is supported for these two cases; avoid absolute insensitivity claims.
- **Nq: non-monotonic; NO convergence/saturation claim.** Nq1 is close to3;
  Nq2 is worse on both datasets. 2→3 yields +.4096548/+.7780779dB,
  +.006553606/+.005368477 SSIM, and -.008574640/-.003282091 LPIPS.
  1→3 yields only +.0093911/+.0207719dB; Liver SSIM slightly falls and Arm
  LPIPS slightly worsens. Nq3 is the PSNR-best of the sampled settings but these
  trained-model endpoints do not establish convergence to an exact integral.
  CUDA forward/backward use standard Gauss–Legendre nodes/weights:1point at0,
  2points at±1/sqrt(3),3points at0 and±sqrt(3/5). No explicit node/weight bug
  was found. Center-sampling parity for sharply localized peaks is a plausible
  mechanism, not an experimentally identified cause; learned geometry and CGA
  co-adapt. Do not write “very low order is insufficient, then saturates.”
- **Half-window: stable across1–4mm; no established cost penalty.** PSNR
  ranges .0206915/.0244875dB; LPIPS .001247229/.001559514. Default2mm is
  reasonable; no monotonic quality improvement with larger windows. Mean global
  associations vary only ~.19% Liver/~.94% Arm across all values (denominator=min).
  These counts mix training/evaluation forwards and co-adapted populations.
  Nq remains3; increasing the window does not increase quadrature sample count.
  No controlled per-point FPS measurement exists; shared-worker wall time is not
  speed evidence. Supported: a larger window was not necessary for comparable
  quality in these cases. NOT supported: larger windows substantially raise cost.

All range endpoints/default-relative deltas are saved unrounded in data files.
No test-selected retuning, new integration order or extra experiment is proposed.

'''+ '\n\n'.join(sections))
    cg=[]
    for panel in ['qd_by_B','qp_by_qA','tau_alpha']:
        cg.append('## '+panel+' raw points\n\n'+short([r for r in points if r['panel']==panel],['dataset','point','final_psnr','final_lpips','final_num_gaussians','delta_psnr','delta_lpips']))
    md('cga/analysis.md', '''# CGA: capacity regulation, not an accuracy-boost claim

## Structural changes
'''+short(deltas,list(deltas[0]))+'''

RGD grows population relative to GD; OP and LP reduce it relative to unpruned
RGD. Full retains additional primitives relative to LP. Full−LP PSNR is
-.0018494dB Liver and +.0386384dB Arm; LPIPS -.000497626 and +.000701491.
SUPPORTED: “path-aware pruning selectively retains additional primitives while
keeping reconstruction quality within a narrow range.” NOT “improves all metrics.”
Protected counts are event-summed decisions, not distinct Gaussian identities.

## qd x B: effective selection controls

PSNR ranges .0144313/.0131710dB; LPIPS .001205274/.001137021;
G ranges675559–711076 /707735–724242. Default(.98,.05) lies in the observed
stable-quality region. qd sets the positive-responsibility quantile; B adds an
approximately capped candidate budget relative to total population. Both feed
a threshold with >= ties, NOT strict TopK.

selection_behavior_summary.tsv classifies every point's12 adaptation events.
Limiter attribution is inferred from logged positive-score count, total N,
selected count, code quantile/topk rule and a2-count tolerance; original full
score vectors/pre-budget quantile thresholds are not archived. Do not claim
exact selected Gaussian identity equivalence across independent runs.

Liver: B=.001 is budget-limited12/12 at qd.96/.98 and10/12 at.99;
(.96,.005) budget9/12, quantile3/12. All other points quantile12/12.
Arm: B=.001 budget12/12 at.96,11/12 at.98,0/12 at.99;
all B=.005/.05 points quantile12/12. No unexplained count-classification cases.
Where the cap is loose, changing B does not change the selection rule for a
given score vector. Small differences across separately trained models are not
evidence of an active budget effect. Where the cap binds, qd changes can be masked.
Thus quality stability is supported, but not equivalence of all combinations.

## qp x qA: local saturation vs genuinely active path protection

PSNR ranges .0144032/.0090822dB; LPIPS .001576368/.001072631;
G ranges661306–722969 /701034–728740. All18 runs x12 events have local
contribution threshold exactly0 for qp=.05/.10/.15. The <= rule selects tied
zero-C primitives, so qp does NOT continuously increase pruning aggressiveness
in this sampled interval. This is local quantile saturation, not evidence that
an active qp control has no effect.

qA is NOT saturated: protection fractions track~30%,20%,10% for qA=.7,.8,.9.
At default qp=.1, path-protected candidate decisions are429680/234708/79070
Liver and216056/119549/50881 Arm. Raising qA narrows protection and lowers
final population. Its thresholds and decisions genuinely change. Lower qA means
more protection / less aggressive pruning, not more pruning. Default(.1,.8)
is in the stable-quality region, but the qp claim must acknowledge zero ties.

## tau_alpha

PSNR ranges .0159422/.0117541dB; LPIPS .002303379/.001112590;
G ranges565867–732376 /639733–738377. Raising threshold5e-5→5e-4 monotonically
reduces population while PSNR changes little. Default1e-4 is intermediate in
population, not guaranteed metric optimum. Together the grids show capacity
regulation without a large reconstruction penalty for the sampled settings.

## Quality–population: KEEP

The scatter adds the unpruned RGD populations (~1.051M Liver /1.023M Arm),
which are outside the hyperparameter grid populations, and connects structural
and numeric controls. It illustrates a broad population span with a narrow PSNR
band, not a fitted causal relation, optimum, efficiency ratio or Pareto proof.
No connecting trend line; datasets separate; independent axis scales labeled.
LPIPS counterpart is retained for review. H3 keeps four evidence classes.

## Claim adjudication

A quality stability: **SUPPORTED** for tested settings/cases, not universal.
B tunable population: **SUPPORTED**, especially tau_alpha/qA and active budgets.
C controls work as expected: **PARTIALLY SUPPORTED** as a blanket claim:
qd/B and qA show intended behavior, but qp lies on a zero-threshold plateau.
Residual responsibility targets weighted reconstruction error by code; final
metrics/scatter alone do not spatially prove that every added primitive lies in
an under-reconstructed region. Use formulation plus event/structural evidence,
not a standalone sensitivity proof of spatial reallocation.

'''+ '\n\n'.join(cg)+'\n\n## Event audit\n\n'+short(events,list(events[0])))
    md('joint/analysis.md', '''# Final Joint audit

L/C/P labels mean LGR/CGA/PPP (1=on). All16 slots follow final semantics;
LGR-on is Fixed+Full. LGR-off intentionally uses Plane/no-global, not Fixed or
Adaptive local-only. CGA-off is GD+opacity; CGA-on is residual-guided/path-aware,
degrading to local-aware without global path statistics. P0=MP/P1=uniformPPP.
Reused corrected Plane rows are compatible by effective config, not relabeled
Adaptive runs. Source IDs/configs are in final_joint_sources.tsv/run_sources.tsv.

Across four factorial backgrounds per dataset, LGR mean PSNR effects are
+3.8633dB Liver and+10.3391dB Arm; it is the dominant fidelity component.
CGA primarily regulates capacity; neither its contribution nor PPP should be
framed as a universal independent accuracy booster. PPP removes measured geometry
from pseudo-trajectory construction, with quality interpretation supported by
matched full-method and multi-sweep comparisons. Factorial background effects
are not independent seeds or a significance test. Liver PPP LPIPS trade-offs
must remain visible. The Main78 comparison remains the frozen MP evaluation
protocol; do not falsely label these Main results as PPP-trained end-to-end runs.
''')
    md('audits/figure_table_audit.md', '''# Figure/table decisions

All source tables retain raw unrounded TSV/JSON and row/config provenance;
display fragments are separate. Main ranks/best/second-best use raw means, never
rounded ties. Multi-sweep is case-equally weighted; SD is not SEM. Hyperparameter
curves have no invented confidence bands (single seed per setting).

H1: KEEP two single-column stacked panels, delta PSNR and delta LPIPS. Shared
nominal default star/dotted line, Liver blue circles/Arm orange squares.

H2 choice comparison:

| Option | Readability/information | Verdict |
|---|---|---|
| A five delta-PSNR panels only | Clean but conceals sigma/beta perceptual trade-offs if unexplained | Not alone |
| B normalized PSNR+LPIPS in every panel | Four lines plus mixed units/scaling in small axes; normalization can exaggerate/compress changes | Do not recommend |
| C delta-PSNR main plus explicit prose on LPIPS | Best clarity with raw LPIPS table/companion retained; must mention sigma AND beta trade-offs | RECOMMEND |

H2 uses5panels in2x3 at double-column width7.16in with legend in remaining cell;
not five cramped panels across one column. Main text MUST describe the two
Liver trade-offs separately: wider sigma improves PSNR but worsens LPIPS; larger beta does
the reverse. The LPIPS companion is an internal review asset, not a promised
Supplementary figure. H1 already includes both metrics because the trade-off is
central. Nq non-monotonicity must be discussed explicitly, no convergence caption.

H3: KEEP four evidence classes, Liver/Arm rows at double-column width. qd/B and
qp/qA heatmaps annotate delta PSNR with a shared symmetric color scale; default
cell outlined. Tau/default marker and population scatter occupy remaining columns.
Population includes unpruned structural controls, not only duplicated grid points.
No PSNR/G ratio or fitted trend. Raw population heatmaps and LPIPS scatter are
review assets. Default star includes both dataset deltas at0, naturally coincident.

Fonts: vector PDF embedded Type42, 8pt nominal text,7pt ticks/heatmap values.
PNG400dpi. No dual y-axis or normalized metric blending. These are layout
recommendations, not claims of journal-mandated dimensions. H2/H3 should NOT be
shrunk to single column (would halve text sizes); if single-column is mandatory,
stack/reflow panels instead. Captions must define all parameter units, metric
directions, default markers, MP isolation and single-seed representative scope.
''')
    md('audits/final_claim_matrix.md', '''# Final claim adjudication

| Claim | Status | Qualification |
|---|---|---|
| Best PSNR/L1/GMSD on all four Main datasets, other metrics best/second | SUPPORTED | Raw ranks checked; frozen MP Main protocol,8 methods,6/9/33/30 cases |
| Main results demonstrate full PPP method on78cases | NOT SUPPORTED | Main was deliberately frozen MP; PPP has separate16-case validation |
| PPP avoids measured geometry for constructing pseudo-trajectory | SUPPORTED | Not pose-file-free loading/QC, not invariance of all perceptual metrics |
| rho1 empirical optimum/plateau | NOT SUPPORTED | Nominal shared spacing; domain/metric trade-offs |
| Finite-slice conditioning essential in compared cases | SUPPORTED | Strongest11-case structural transition |
| Adaptive independently improves accuracy | NOT SUPPORTED | Domain/metric trade-off; Fixed final |
| Full globally improves all metrics over Simple | NOT SUPPORTED | Arm mean benefit; small Liver trade-off |
| Fixed1mm shared operating point | SUPPORTED | Material width sensitivity, no universal optimum claim |
| All LGR parameters insensitive | NOT SUPPORTED | sigma/beta trade-offs; Nq2 dip |
| Nq1→2→3 converges/saturates | NOT SUPPORTED | Nq2 worse;1close3 |
| lambda/window stable over tested ranges | SUPPORTED | Two representative cases only |
| Larger window materially increases cost | NOT SUPPORTED | No controlled per-point timing; associations nearly flat |
| CGA quality stable while capacity changes | SUPPORTED | Tested controls; no universal guarantee |
| Every qp/qA grid control active | PARTIALLY SUPPORTED | qp zero-C saturation; qA genuinely active |
| Path-aware pruning improves every metric | NOT SUPPORTED | Extra retention, narrow mixed metric changes |
| More speckle means more anatomical detail | NOT SUPPORTED | Require matched GT pattern; no scatterer/anatomical inference |
''')
    md('audits/FINAL_EXPERIMENT_COMPLETENESS.md', '''# Final Experiment Completeness

| Evidence category | Status | Basis |
|---|---|---|
| Main comparison | COMPLETE |78Fixed+Full MP cases;7unchanged baselines;same cases;raw ranks |
| Joint component ablation | COMPLETE |16slots,8combinations x2representatives,final semantics |
| PPP internal | COMPLETE |MP/PP-S/PP-A/PPP x2,Fixed+Full |
| PPP multi-sweep | COMPLETE |6Liver+5Arm+5Breast MP/PPP,PSNR/SSIM |
| PPP rho sensitivity | COMPLETE |7values x2,existing final runs |
| LGR internal | COMPLETE |5rows,6Liver+5Arm multi-sweep |
| LGR hyperparameters | COMPLETE |5parameter axes,raw metrics/deltas;Nq non-monotonicity disclosed |
| CGA internal | COMPLETE |5variants x2,PSNR/LPIPS/population |
| CGA population dynamics | COMPLETE |Curves,event/spawn/prune/protection statistics |
| CGA hyperparameters | COMPLETE |Two3x3 grids,tau3points;active/saturated controls audited |
| FPS | COMPLETE |Fixed models,5warmup/20frames/3passes,2frozen cases/dataset |
| Qualitative | COMPLETE |4Main datasets+8internal sheets;GT/crop/config/source integrity |

**Are there any scientifically necessary additional training experiments? NO.**

No explicit implementation bug was identified by this final audit. Nq's
non-monotonic trained-model performance is an observed limitation, not sufficient
evidence of a coding error and not a reason to launch new runs. Completeness is
conditional on the restrained supported claims in final_claim_matrix.md, NOT on
proving every previously proposed optimistic claim. No multi-seed/generalization
beyond this protocol, clinical validation or exact numerical convergence claim.

Main deliberately retains MP, even though the final system also offers/uses PPP
in its acquisition-guided formulation. State the Main pose protocol transparently;
do not relabel MP numbers as PPP results. This accepted scope is not a missing run.

No more training, no expanded search, no test-driven default changes. Production
remains the P2 Fixed freeze. Canonical LaTeX source is still unconfirmed; assets
and prose/patch suggestions are prepared without editing03_method.tex or04_results.
Human manuscript integration/layout is editorial work, not incomplete experiments.
''')
    md('README.md', '''# Final paper evidence — 2026-09-10

Experiment design and training are closed. This package contains only analyses,
tables, figures and audits of existing P2 Fixed evidence; no new training.

Final renderer:Fixed finite-slice1mm + full causal ray-integrated global;
full accumulated-A_g CGA; acquisition-guided PPP. Main/LGR/CGA sensitivity uses
the explicitly frozen MP isolation protocol; PPP evidence is separate.

Start with audits/FINAL_EXPERIMENT_COMPLETENESS.md and final_claim_matrix.md.
Main claims/ranks/FPS in main/;Joint/PPP/LGR/CGA final tables and analyses in
their named folders. H1 is single-column;H2/H3 double-column. Qualitative source
integrity and restrained visual claims are in qualitative/.

Every table retains raw unrounded TSV/JSON, displayed Markdown and, where
appropriate, an apply-ready tabular fragment. Figures are PNG/PDF. Long raw
source tables intentionally include run IDs/output paths/resolved configs;
audits/run_sources.tsv maps all source versions/effective-config hashes.
Reused corrected P0.5 runs preserve their original execution SHA plus P2
equivalence certification, never falsely claimed as P2-executed. Baselines retain
their own legacy lineage. No legacy-bug Ours metrics enter final tables.

Important findings: Nq2 is worse than1/3 (not saturation); qp is zero-C saturated;
qA actively changes protection; widths/beta have metric/domain trade-offs;
lambda/window and CGA quality are stable over the tested settings. No universal
best-default, statistical-significance or “all parameters insensitive” claim.

No scientifically necessary extra training is identified. No new tasks are
submitted by this packager. Default remains unchanged. No canonical LaTeX edited.

Reproduce packaging only:
`PYTHONPATH=ops/ours_renderer:. PYTHONDONTWRITEBYTECODE=1 /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python scripts/production/ablations/ours/final_paper_evidence.py`
''')
    mainclaims=DEST/'main/final_main_claims.md'
    mainclaims.write_text(mainclaims.read_text()+r'''

## Approved numerical claim template

\methodname{} achieves the best PSNR, L1, and GMSD on all four datasets,
while attaining best or second-best SSIM, LPIPS, and DISTS throughout.

This is 100% accurate for the frozen eight-method, within-sweep Main comparison,
using unrounded means and MP for Ours. Do not remove this experimental scope or
imply these78runs used PPP. FPS uses separately rebuilt Fixed model replicas on
the frozen two cases/dataset; it is pure forward, not end-to-end throughput.
''')
    for group,name in [('ppp','rho_ranges'),('lgr','hyperparameter_ranges'),('cga','hyperparameter_ranges')]:
        path=DEST/group/'analysis.md'
        path.write_text(path.read_text()+'\n\n## Complete ranges / default positions\n\n'+(DEST/group/(name+'_display.md')).read_text())
    scatter=read(H/'quality_population_scatter.tsv');stats=[]
    for ds in ['synthetic_liver','tus_rec_2024_qc']:
        rr=[r for r in scatter if r['dataset']==ds]
        stats.append(dict(dataset=DS[ds],unique_points=len(rr),G_min=min(r['final_num_gaussians'] for r in rr),G_max=max(r['final_num_gaussians'] for r in rr),
                          PSNR_min=min(r['psnr'] for r in rr),PSNR_max=max(r['psnr'] for r in rr),LPIPS_min=min(r['lpips'] for r in rr),LPIPS_max=max(r['lpips'] for r in rr)))
    export('cga/quality_population_ranges',stats)
    p=DEST/'cga/analysis.md';p.write_text(p.read_text()+'\n\n## Combined quality–population extent\n\n'+table(stats,list(stats[0])))
    qp=DEST/'qualitative/final_qualitative_audit.md'
    qp.write_text(qp.read_text()+'''

## Final explicit visual adjudication

| Group | SUPPORTED | WEAK — qualify | UNSUPPORTED — do not write |
|---|---|---|---|
| Main Liver | Major shadow/echo pattern retained | Fine texture similarity to GT | Every speckle is recovered anatomy |
| Main Spine | Bright layered bands retained | Superiority over strongest baseline in this crop | Perfect thin-structure recovery |
| Main Arm | Layer separation and coarse echo arrangement | Small boundary-sharpness ranking between strong methods | Artifact-free / best in every region |
| Main Breast | Broad pattern and localized texture retained | Fine-band similarity | More speckle means more true detail |
| Joint Liver/Arm | LGR-on visibly improves over Plane | CGA/PPP small visual gains under full renderer | Three equally strong independent accuracy boosts |
| PPP Liver/Arm | Coarse reconstruction content retained | Perceptual similarity | Identical texture or all-metric preservation |
| LGR Liver/Arm | Plane-to-finite-slice improvement | Fixed/Simple/Full fine differences | Five monotonically improving visual rows |
| CGA Liver/Arm | Similar content across five policies | Tiny visual differences | Full universally improves image quality |

All9sheets were visually inspected in the initial final qualitative audit; this
package retains exact copies and hashes of those reviewed assets. Integrity checks
were re-run during packaging. No new case/crop or contrast selection was made.
''')
    md('audits/paper_ready_prose_drafts.md', r'''
# Suggested prose / captions (not applied to canonical LaTeX)

## Local/global formulation

Finite-slice conditioning yields the largest and most consistent improvement
over the plane-only control. Comparing fixed and depth-adaptive widths reveals
dataset- and metric-dependent trade-offs, motivating the shared simpler fixed
formulation. Adding simple global transmission improves Arm reconstruction and
Liver LPIPS at a modest Liver PSNR/SSIM cost. Full causal integration further
improves the Arm means, with small changes on Liver.

## Hyperparameters

Finite-slice width materially affects reconstruction: wider slices improve Liver
PSNR but worsen LPIPS, whereas Arm favors the narrower tested widths. The fixed
1-mm setting provides a shared operating point. Increasing beta trades Liver
PSNR for LPIPS, while improving both on Arm. Attenuation strength and integration
half-window produce small changes over the tested ranges. Quadrature results
are non-monotonic: the two-point rule yields lower fidelity than either the
one- or three-point rule in these trained systems; these results do not establish
numerical convergence. The three-point production setting remains unchanged.

## CGA

Across the tested density-control settings, reconstruction quality varies little
while model population changes. Tight candidate budgets restrict densification,
whereas loose budgets leave selection quantile-controlled. Local pruning
quantiles share a zero-contribution threshold over the tested range; in contrast,
the accumulated-path quantile actively changes protection and final population.
Thus the plateau must be interpreted jointly with the effective selection rules,
not as universal insensitivity to every pruning parameter.

## H1 caption

Sensitivity to the nominal spacing multiplier rho using Fixed+Full and CGA on
the Liver/Arm representatives. Values are relative to rho=1 (black star/dotted
line); higher delta PSNR and lower delta LPIPS are favorable. No per-sequence
tuning is performed. Lines connect sampled settings only; no uncertainty bands
are implied.

## H2 caption

Renderer-parameter sensitivity under MP and the final Fixed+Full/CGA design.
Panels vary width, beta, attenuation strength, quadrature order and integration
half-window, holding other settings fixed. Black stars/dotted lines indicate
production defaults. Vertical axes show default-relative PSNR in dB and have
different ranges. Width and beta also exhibit perceptual trade-offs discussed
in the text; Nq results are non-monotonic rather than a convergence curve.

## H3 caption

CGA parameter analysis on Liver(top) and Arm(bottom). (a,b) Annotated default-
relative PSNR for densification/pruning grids; outlined cells mark defaults.
The heatmaps share a symmetric color scale; annotations give exact displayed dB
differences (rounded to.001). (c) Opacity-threshold sensitivity. (d) PSNR versus
final population across structural controls and hyperparameter settings, with
the default marked by a star. Axes in(d) differ between datasets. Black outlines
in the grids and stars in(c,d) identify the same unchanged default. The local
contribution quantile is saturated at zero in(b); the path quantile is active.
The scatter indicates a quality–capacity operating range, not a fitted causal
law or a claim that fewer primitives always improve quality.
''')


if __name__=='__main__':package()
