"""Scientific checkpoint from complete Batch 1 tables; no training or claim rewriting."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports'
METRICS=('psnr','l1','ssim','lpips','dists','gmsd')
HIGH={'psnr','ssim'}


def read(name):
    rows=list(csv.DictReader((OUT/(name+'.tsv')).open(),delimiter='\t'))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except (ValueError,TypeError):pass
    return rows


def table(rows, fields):
    lines=['| '+' | '.join(fields)+' |','|'+'|'.join('---' for _ in fields)+'|']
    for r in rows:
        lines.append('| '+' | '.join('{:.6g}'.format(r[k]) if isinstance(r.get(k),float) else str(r.get(k,'')) for k in fields)+' |')
    return '\n'.join(lines)+'\n'


def main():
    progress=json.loads((OUT/'progress.json').read_text())
    assert progress['remaining']==0,'Incomplete Batch 1: no final scientific checkpoint'
    mainrows=read('main_complete_dataset_aggregates')
    assert len(mainrows)==36
    rankings=read('main_rankings')
    joint=read('joint');ppp=read('ppp_internal');lgr=read('lgr_structural');cga=read('cga_structural');rho=read('rho')
    assert [len(x) for x in (joint,ppp,lgr,cga,rho)]==[16,8,10,10,14]
    text=['# Batch 1 scientific checkpoint\n',
          'All quality numbers below are corrected-covariance final30k, seed42. Baselines are reused on identical case/frame IDs. Legacy Ours is impact-reference only. Single-seed contrasts are not significance tests. Batch2 is NOT authorized.\n',
          '## Main comparison\n',table(mainrows,['dataset','method','n']+['final_'+m for m in METRICS]+['train_wall_time_sec']),
          table(rankings,['dataset','metric','best','second','corrected_rank'])]
    deltas=[];claims=[]
    for ds in sorted({r['dataset'] for r in mainrows}):
        rows=[r for r in mainrows if r['dataset']==ds]
        ours=next(r for r in rows if r['method']=='ours_corrected')
        old=next(r for r in rows if r['method']=='ours_legacy')
        deltas.append(dict(dataset=ds,**{m:ours['final_'+m]-old['final_'+m] for m in METRICS}))
        categories={'NeRF':['ultranerf','nerfus_no_prior'],
                    'ultrasound_Gaussian':['ultragauss','ultragray','ultrags_baseline'],
                    'adapted_Gaussian':['slice_adapted_3dgs','gaussianpile']}
        for category,methods in categories.items():
            baselines=[r for r in rows if r['method'] in methods]
            for m in METRICS:
                best=sorted(baselines,key=lambda r:r['final_'+m],reverse=m in HIGH)[0]
                delta=ours['final_'+m]-best['final_'+m]
                claims.append(dict(dataset=ds,comparison=category,metric=m,strongest_baseline=best['method'],
                     baseline=best['final_'+m],corrected=ours['final_'+m],delta=delta,
                     relative_reduction_pct=-100*delta/best['final_'+m] if m not in HIGH else '',
                     directional_advantage_supported=delta>0 if m in HIGH else delta<0))
    text+=['## Corrected minus legacy Ours\n',table(deltas,['dataset']+list(METRICS)),
           'These deltas compare the old published main-table row, not an isolated covariance intervention. The newer accumulated-A_g legacy main configs match the corrected configs and are separately compared in main_matched_legacy_per_run_deltas.tsv. Neither historical binary is treated as strictly reproducible; no causal/statistical significance claim is made.\n',
           '## Numerical-claim audit\n',
           'All old exact Ours values/percentages require replacement even when the direction survives. The following table recomputes every dataset/category/metric contrast against its strongest baseline. Best-PSNR/L1/SSIM and GMSD/perceptual claims must follow the explicit ranks above, not the old narrative.\n',
           table(claims,['dataset','comparison','metric','strongest_baseline','baseline','corrected','delta','relative_reduction_pct','directional_advantage_supported']),
           'Training-time advantage claims are not hardware-controlled: new training uses4080SUPER, anchors/legacy use4090, and concurrency differs. Old240–353FPS is not corrected evidence. Fair corrected FPS is separately gated on4090 measurements.\n',
           '## Joint factorial\n',table(joint,['dataset','PPP','LGR','CGA','final_psnr','final_ssim','final_lpips','final_num_gaussians'])]
    effects=[]
    for ds in sorted({r['dataset'] for r in joint}):
        rows=[r for r in joint if r['dataset']==ds]
        for module in ('PPP','LGR','CGA'):
            other=[m for m in ('PPP','LGR','CGA') if m!=module]
            for metric in ('psnr','ssim','lpips','num_gaussians'):
                diffs=[]
                for r in rows:
                    if r[module]!=0:continue
                    on=next(s for s in rows if s[module]==1 and all(s[m]==r[m] for m in other))
                    diffs.append(on['final_'+metric]-r['final_'+metric])
                effects.append(dict(dataset=ds,module=module,metric=metric,mean_effect=statistics.mean(diffs),min_effect=min(diffs),max_effect=max(diffs)))
    text += [table(effects,['dataset','module','metric','mean_effect','min_effect','max_effect']),
       'Effects are paired across four settings of the other modules, not independent replicates. LGR-off means adaptive local without global transmission; CGA-on then uses local-aware fallback, not an all-protected path rule.\n',
       '## PPP\n',table(ppp,['dataset','label','final_psnr','final_ssim','final_lpips']),
       table(read('ppp_multisweep_mean_std'),['dataset','n','MP_PSNR_mean','MP_PSNR_std','PPP_PSNR_mean','PPP_PSNR_std','MP_SSIM_mean','PPP_SSIM_mean','delta_PSNR_mean','delta_PSNR_std']),
       'Geometry-use claim is confined to pseudo-trajectory construction. Arm subset preprocessing used measured-pose QC.\n',
       '## LGR structural\n',table(lgr,['dataset','label','final_psnr','final_ssim','final_lpips','final_num_gaussians'])]
    pairs=[]
    for ds in sorted({r['dataset'] for r in lgr}):
        for a,b in zip(('Plane','Fixed','Adaptive','Simple'),('Fixed','Adaptive','Simple','Full')):
            before=next(r for r in lgr if r['dataset']==ds and r['label']==a)
            after=next(r for r in lgr if r['dataset']==ds and r['label']==b)
            pairs.append(dict(dataset=ds,transition=a+' -> '+b,delta_PSNR=after['final_psnr']-before['final_psnr'],delta_SSIM=after['final_ssim']-before['final_ssim'],delta_LPIPS=after['final_lpips']-before['final_lpips']))
    text += [table(pairs,['dataset','transition','delta_PSNR','delta_SSIM','delta_LPIPS']),
       'No monotonic improvement is assumed. Plane/Fixed isolates finite-slice conditioning; Fixed/Adaptive isolates depth dependence. Global-on also enables contribution-based path protection, so Adaptive/Simple is a whole-system structural contrast, not a renderer-only fixed-topology estimate. Simple/Full compares the specified preceding-depth approximation to causal integration. No attribute-decoupling ablation is claimed.\n',
       '## CGA structural and population\n',table(cga,['dataset','label','final_psnr','final_lpips','final_num_gaussians'])]
    changes=[]
    for ds in sorted({r['dataset'] for r in cga}):
        mapping={r['label']:r for r in cga if r['dataset']==ds}
        for a,b in [('GD','RGD'),('RGD','RGD+OP'),('RGD','RGD+LP'),('RGD+LP','Full')]:
            x,y=mapping[a],mapping[b]
            changes.append(dict(dataset=ds,transition=a+' -> '+b,delta_G=y['final_num_gaussians']-x['final_num_gaussians'],delta_G_pct=100*(y['final_num_gaussians']/x['final_num_gaussians']-1),delta_PSNR=y['final_psnr']-x['final_psnr'],delta_LPIPS=y['final_lpips']-x['final_lpips']))
    text += [table(changes,['dataset','transition','delta_G','delta_G_pct','delta_PSNR','delta_LPIPS']),
       'RGD vs GD also removes conventional pruning by definition. RGD+OP restores it; RGD+LP tests contribution-aware pruning; Full adds global protection. Final population differences are net training outcomes, not a count of the exact same Gaussians preserved.\n',
       '## Representative rho\n',table(rho,['dataset','rho','final_psnr','final_ssim','final_lpips'])]
    optima=[]
    for ds in sorted({r['dataset'] for r in rho}):
        rows=[r for r in rho if r['dataset']==ds];default=next(r for r in rows if r['rho']==1.)
        for m in ('psnr','ssim','lpips'):
            best=sorted(rows,key=lambda r:r['final_'+m],reverse=m in HIGH)[0]
            optima.append(dict(dataset=ds,metric=m,best_sampled_rho=best['rho'],default=default['final_'+m],best_sampled=best['final_'+m],default_minus_best=default['final_'+m]-best['final_'+m]))
    text += [table(optima,['dataset','metric','best_sampled_rho','default','best_sampled','default_minus_best']),
       'Best sampled is descriptive, not a retuned paper default. No LGR/CGA robustness/plateau claim is validated by Batch1; those grids remain Batch2 decisions.\n']
    population=read('population');fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for ax,ds in zip(axes,sorted({r['dataset'] for r in population})):
        for label in ('GD','RGD','RGD+OP','RGD+LP','Full'):
            rows=sorted((r for r in population if r['dataset']==ds and r['label']==label),key=lambda r:r['iteration'])
            ax.step([r['iteration'] for r in rows],[r['num_gaussians']/1e6 for r in rows],where='post',label=label)
        ax.set_title(ds);ax.set_xlabel('Iteration');ax.set_ylabel('Gaussians (million)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'population_dynamics.pdf');fig.savefig(OUT/'population_dynamics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,5.5))
    for i,ds in enumerate(sorted({r['dataset'] for r in rho})):
        rows=sorted((r for r in rho if r['dataset']==ds),key=lambda r:r['rho'])
        for j,m in enumerate(('psnr','ssim','lpips')):
            ax=axes[i,j];ax.plot([r['rho'] for r in rows],[r['final_'+m] for r in rows],'-o');ax.axvline(1.,color='gray',ls='--');ax.set_title(ds+' / '+m.upper());ax.set_xlabel('rho');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'rho_sensitivity.pdf');fig.savefig(OUT/'rho_sensitivity.png',dpi=180);plt.close(fig)
    text += ['## Artifacts and next gate\n',
             'See qualitative_paths.json (fixed inherited cases, identical baseline GT hashes), population_dynamics.pdf/png, rho_sensitivity.pdf/png, per-run tables, exact config/provenance in each output directory. Do not launch Batch2 automatically.\n']
    (OUT/'SCIENTIFIC_CHECKPOINT.md').write_text('\n'.join(text))
    (OUT/'claim_audit.json').write_text(json.dumps(claims,indent=2)+'\n')
    print('Complete quality checkpoint written; fair4090 FPS and manual qualitative interpretation must still be verified.')


if __name__=='__main__':
    main()
