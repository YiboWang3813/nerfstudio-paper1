"""Single-column figures from sealed numeric evidence only; no model execution."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import yaml

# Installed skill QA: fail before exporting misaligned multi-panel figures.
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'docs/production/final_paper_evidence_20260910'
OUT = ROOT / 'docs/production/final_hyperparameter_figures_20260916'
DS = ['synthetic_liver', 'tus_rec_2024_qc']
NAMES = dict(zip(DS, ['Liver', 'Arm']))
COLORS = ['#356A93', '#BD633E']
HEATMAP = LinearSegmentedColormap.from_list('paper_blue_gray_orange', ['#356A93', '#F5F3EF', '#BD633E'])
MARKERS = ['o', 's']
PANELS = [
    ('sigma_ele', [.8, 1., 1.5, 2.], 1., r'$\sigma_{\mathrm{ele}}$ (mm)', 'Finite-slice width', 'us_focus_sigma_ele_0'),
    ('beta', [.4, .6, .8, 1.], .8, r'$\beta$', 'Composition weight', 'ray_attenuation_beta'),
    ('lambda_ray', [.2, .35, .55, .75, 1.], .55, r'$\lambda_{\mathrm{ray}}$', 'Attenuation strength', 'ray_attenuation_strength'),
    ('Nq', [1, 2, 3], 3, r'$N_q$', 'Quadrature order', 'ray_attenuation_quadrature_points'),
    ('half_window', [1, 2, 3, 4], 2, r'$w$ (mm)', 'Integration half-window', 'ray_attenuation_quadrature_window_mm'),
]
HASHES = {}
CFG = {}
FIGURES = []


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(rel):
    p = SRC / rel
    HASHES[rel] = digest(p)
    if p.suffix == '.json':
        return json.loads(p.read_text())
    if p.suffix == '.md':
        return p.read_text()
    with p.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(rel, rows):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w = csv.DictWriter(f, keys, delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def text(rel, value):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value.rstrip() + '\n')


def style():
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','Times','DejaVu Serif'],
        'mathtext.fontset':'dejavuserif', 'font.size':9,
        'axes.labelsize':10, 'axes.titlesize':9.5, 'xtick.labelsize':8.5,
        'ytick.labelsize':8.5, 'legend.fontsize':8.5, 'axes.linewidth':.8,
        'lines.linewidth':1.35, 'lines.markersize':4.0, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white', 'axes.unicode_minus':True})


def save(fig, rel):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    plot_axes = [ax for ax in fig.axes if ax.get_label() != 'qa_colorbar']
    if len(plot_axes) > 1:
        require_matplotlib_panel_alignment(fig, axes=plot_axes,
            json_out=str(p)+'.alignment.json', overlay_svg=str(p)+'.alignment.svg',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    # No bbox_inches=tight: preserve exactly 3.5 inches in the PDF MediaBox.
    fig.savefig(p.with_suffix('.svg'))
    fig.savefig(p.with_suffix('.pdf'), metadata={'Creator':'ULGSplat frozen evidence plotting', 'CreationDate':None})
    fig.savefig(p.with_suffix('.png'), dpi=400)
    fig.savefig(p.with_name(p.name+'_size_review.png'), dpi=150)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            b = artist.get_window_extent(renderer)
            if b.width > 0 and b.height > 0 and (b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1):
                clipped.append(artist.get_text())
    FIGURES.append(dict(path=str(p),width_in=float(fig.get_figwidth()),height_in=float(fig.get_figheight()),png_dpi=400,outside_canvas_text=clipped))


def axes_style(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.8)
        spine.set_edgecolor('black')
    ax.set_axisbelow(True)
    ax.grid(True,axis='y',color='.90',linewidth=.5)
    ax.tick_params(width=.7, length=3, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(useOffset=False, axis='y')


def legend_handles():
    return [Line2D([],[],color=c,marker=m,ls='-' if ds==DS[0] else '--',label=NAMES[ds]) for ds,c,m in zip(DS,COLORS,MARKERS)] + [Line2D([],[],color='black',marker='*',ls='none',markersize=6,label='Default')]


def line(ax, rows, field, metric, default):
    for ds,c,m in zip(DS,COLORS,MARKERS):
        rr = sorted([r for r in rows if r['dataset']==ds],key=lambda r:float(r[field]))
        ax.plot([float(r[field]) for r in rr],[float(r[metric]) for r in rr],color=c,marker=m,ls='-' if ds==DS[0] else '--',label=NAMES[ds])
    ax.axhline(0,color='.65',lw=.55,zorder=0)
    ax.axvline(default,color='.35',lw=1.,ls='--',zorder=1)
    ax.plot(default,0,'k*',ms=6.5,zorder=5)
    axes_style(ax)


def prepare():
    manifest=read('audits/final_data_source_manifest.json')
    runindex={r['row_id']:r for r in read('audits/run_sources.tsv')}
    rho=read('ppp/rho_data.tsv'); plgr=read('lgr/hyperparameter_data.tsv'); cgas=read('cga/hyperparameter_data.tsv')
    structural=read('cga/structural_sources.tsv')
    for rel in ['ppp/analysis.md','lgr/analysis.md','cga/analysis.md']:
        read(rel)
    for r in rho+plgr+cgas+structural:
        idx=runindex[r['row_id']]
        rel=idx['packaged_config']
        if rel not in CFG:
            HASHES[rel]=digest(SRC/rel)
            CFG[rel]=yaml.safe_load((SRC/rel).read_text())
        c=CFG[rel];sr=c['model']['slice_render'];dc=c['model']['offline']['density_control']
        assert sr['local_conditioning_mode']=='fixed'
        assert sr['ray_attenuation_enabled'] and sr['ray_attenuation_formula']=='integral'
        assert sr['ray_attenuation_source']=='attenuation'
        assert c['engine']['iterations']==30000 and c['system']['seed']==42 and c['scene']['sampler']['num_init']==1000000
        assert idx['validated_p2_source_sha256']==manifest['P2_source_sha256']
        assert idx['output_dir']==r['output_dir']
        r['packaged_config']=str(SRC/rel)
        if 'rho' in r:
            assert c['scene']['loading']['pose_mode']=='canonical_sweep'
            assert c['scene']['loading']['canonical_sweep_spacing_mode']=='uniform'
            assert float(c['scene']['loading']['canonical_sweep_delta_scale'])==float(r['rho'])
        else:
            assert c['scene']['loading']['pose_mode']=='file'
        if 'panel' in r or 'rho' in r:
            assert dc['densification_policy']=='residual_guided' and dc['pruning_policy']=='path_aware'
            assert dc['responsibility']['protect_attenuation_mode']=='path_contribution'
        if 'panel' in r:
            panel=r['panel'];point=r['point']
            if panel in [p[0] for p in PANELS]:
                _,values,default,_,_,key=next(p for p in PANELS if p[0]==panel)
                assert float(sr[key])==float(point)
                assert (r['is_default']=='True')==(float(point)==default)
            elif panel=='qd_by_B':
                q,b=map(float,point.split('_'))
                assert dc['responsibility']['score_quantile']==q and dc['responsibility']['max_fraction']==b
            elif panel=='qp_by_qA':
                q,a=map(float,point.split('_'))
                assert dc['responsibility']['low_contribution_quantile']==q and dc['responsibility']['path_contribution_quantile']==a
            else:
                assert panel=='tau_alpha' and dc['thresholds']['prune_opacity']==float(point)
    ranges=[]
    for data in [rho,plgr,cgas]:
        for ds in DS:
            panels=sorted({r.get('panel','rho') for r in data})
            for panel in panels:
                rr=[r for r in data if r['dataset']==ds and r.get('panel','rho')==panel]
                base=next(r for r in rr if float(r['rho'])==1) if panel=='rho' else next(r for r in rr if r['is_default']=='True')
                for r in rr:
                    r['is_default']=r is base
                    for m in ['psnr','ssim','lpips']:
                        delta=Decimal(r['final_'+m])-Decimal(base['final_'+m])
                        if 'delta_'+m in r:
                            assert abs(delta-Decimal(r['delta_'+m]))<Decimal('1e-12')
                        r['delta_'+m]=str(delta)
                        r['default_'+m]=base['final_'+m]
                ranges.append(dict(dataset=NAMES[ds],panel=panel,**{m+'_range':str(max(Decimal(r['final_'+m]) for r in rr)-min(Decimal(r['final_'+m]) for r in rr)) for m in ['psnr','ssim','lpips']}))
    for panel,values,default,*_ in PANELS:
        for ds in DS:
            assert sorted(float(r['point']) for r in plgr if r['dataset']==ds and r['panel']==panel)==values
    for ds in DS:
        assert sorted(float(r['rho']) for r in rho if r['dataset']==ds)==[.6,.75,.9,1.,1.1,1.25,1.5]
    for r in cgas:
        if r['panel']=='qd_by_B':
            x,y=map(float,r['point'].split('_'));assert x in [.96,.98,.99] and y in [.001,.005,.05]
            assert r['is_default']==(x==.98 and y==.05)
        if r['panel']=='qp_by_qA':
            x,y=map(float,r['point'].split('_'));assert x in [.05,.1,.15] and y in [.7,.8,.9]
            assert r['is_default']==(x==.1 and y==.8)
        if r['panel']=='tau_alpha':
            assert float(r['point']) in [5e-5,1e-4,5e-4]
            assert r['is_default']==(float(r['point'])==1e-4)
    for folder,data in [('H1_AFGM',rho),('H2_PLGR',plgr),('H3_CGAS',cgas)]:
        write(folder+'/plotted_data.tsv',data)
    write('verified_ranges.tsv',ranges)
    for rel in ['lgr/hyperparameter_ranges.tsv','cga/hyperparameter_ranges.tsv']:
        for r in read(rel):
            actual=next(v for v in ranges if v['dataset']==NAMES[r['dataset']] and v['panel']==r['panel'])
            for m in ['psnr','ssim','lpips']:
                assert abs(Decimal(actual[m+'_range'])-Decimal(r[m+'_range']))<Decimal('1e-12')
    assert all(float(r['psnr_range'])<.016 for r in ranges if r['panel'] in ['qd_by_B','qp_by_qA','tau_alpha'])
    events=read('cga/selection_behavior_summary.tsv')
    qp=[r for r in events if r['panel']=='qp_by_qA']
    assert sum(int(r['events']) for r in qp)==216
    assert all(r['events']==r['zero_C_threshold_events'] for r in qp)
    write('H3_CGAS/selection_behavior_audit.tsv',events)
    # Deduplicate repeated default memberships by actual run path.
    scatter={}
    for r in cgas+structural:
        key=(r['dataset'],r['output_dir'])
        tag=r.get('panel','structural')+'/'+str(r.get('point',r.get('label')))
        if key not in scatter:
            scatter[key]=dict(dataset=r['dataset'],row_id=r['row_id'],output_dir=r['output_dir'],packaged_config=r['packaged_config'],final_G=r['final_num_gaussians'],G_k=str(Decimal(r['final_num_gaussians'])/1000),PSNR=r['final_psnr'],LPIPS=r['final_lpips'],memberships=tag,structural_variant='',is_default=False)
        else:
            assert scatter[key]['PSNR']==r['final_psnr']
            scatter[key]['memberships']+=';'+tag
        if 'panel' not in r:
            scatter[key]['structural_variant']='Full CGAS' if r['label']=='Full' else r['label'].replace('+',' + ')
        scatter[key]['is_default'] |= bool(r.get('is_default',False)) or r.get('label')=='Full'
    scatter=list(scatter.values())
    legacy_scatter=read('cga/quality_population.tsv')
    assert {(r['dataset'],r['output_dir']) for r in legacy_scatter}=={(r['dataset'],r['output_dir']) for r in scatter}
    for r in legacy_scatter:
        x=next(x for x in scatter if x['dataset']==r['dataset'] and x['output_dir']==r['output_dir'])
        assert Decimal(x['PSNR'])==Decimal(r['psnr']) and Decimal(x['final_G'])==Decimal(r['final_num_gaussians'])
    write('H4_CGAS_population/plotted_data.tsv',scatter)
    for rel,h in HASHES.items():
        if rel!='audits/final_data_source_manifest.json':assert manifest['package_file_sha256'][rel]==h,rel
    text('source_audit.json',json.dumps(dict(source_root=str(SRC),P2_source_sha256=manifest['P2_source_sha256'],input_sha256=HASHES,unique_configs_checked=len(CFG),scatter_unique_runs=len(scatter),qp_zero_threshold_events=216,no_training=True,no_jobs=True,no_defaults_changed=True),indent=2))
    return rho,plgr,cgas,scatter,ranges


def h1(rows):
    fig,ax=plt.subplots(figsize=(3.5,2.25))
    fig.subplots_adjust(left=.20,right=.975,bottom=.25,top=.82)
    line(ax,rows,'rho','delta_psnr',1.)
    ax.set_xlabel(r'Spacing scale $\rho$');ax.set_ylabel(r'$\Delta$PSNR (dB)')
    ax.set_xticks([.75,1.,1.25,1.5]);ax.set_xticklabels(['0.75','1.00','1.25','1.50'])
    fig.legend(handles=legend_handles(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.55,.99),columnspacing=1.0,handlelength=1.3)
    save(fig,'H1_AFGM/H1_candidate_A');save(fig,'H1_AFGM/H1');plt.close(fig)
    # A genuine single-axis trade-off plot: separate dimensional axes, no dual scale.
    fig,ax=plt.subplots(figsize=(3.5,2.65))
    fig.subplots_adjust(left=.20,right=.97,bottom=.22,top=.84)
    for ds,c,m in zip(DS,COLORS,MARKERS):
        rr=sorted([r for r in rows if r['dataset']==ds],key=lambda r:float(r['rho']))
        ax.plot([float(r['delta_psnr']) for r in rr],[float(r['delta_lpips']) for r in rr],color=c,marker=m,ls='-' if ds==DS[0] else '--')
        for rho in ([.6,1.5] if ds==DS[0] else [.6,1.25,1.5]):
            r=next(r for r in rr if float(r['rho'])==rho)
            offset=(3,5) if rho==.6 else (-4,-11) if ds==DS[0] else ((-4,-12) if rho==1.25 else (-38,14))
            ax.annotate(r'$\rho$='+str(rho),(float(r['delta_psnr']),float(r['delta_lpips'])),xytext=offset,textcoords='offset points',fontsize=6.5,color=c)
    ax.plot(0,0,'k*',ms=6.5);ax.axhline(0,lw=.55,color='.7');ax.axvline(0,lw=.55,color='.7')
    ax.set_xlabel(r'$\Delta$PSNR (dB) $\rightarrow$ better');ax.set_ylabel(r'$\Delta$LPIPS ($\downarrow$ better)');axes_style(ax);ax.margins(.18,.25)
    ax.grid(False)
    ax.set_xticks([-.6,-.4,-.2,0,.2,.4])
    fig.legend(handles=legend_handles(),loc='upper center',bbox_to_anchor=(.55,.99),ncol=3,frameon=False,handlelength=1.3)
    save(fig,'H1_AFGM/H1_candidate_B');plt.close(fig)


def h2(rows):
    for metric,name in [('delta_psnr','H2'),('delta_lpips','H2_lpips_review')]:
        fig,axs=plt.subplots(5,1,figsize=(3.5,6.8))
        fig.subplots_adjust(left=.22,right=.975,top=.925,bottom=.07,hspace=.88)
        for i,(ax,(panel,values,default,xlabel,title,key)) in enumerate(zip(axs,PANELS)):
            line(ax,[r for r in rows if r['panel']==panel],'point',metric,default)
            ax.set_xticks(values);ax.set_xticklabels([f'{v:g}' for v in values])
            ax.set_xlabel(xlabel,labelpad=1.5)
            ax.set_ylabel(r'$\Delta$PSNR (dB)' if metric=='delta_psnr' else r'$\Delta$LPIPS',labelpad=3)
            ax.set_title('('+chr(97+i)+') '+title,loc='left',pad=4,fontsize=9)
            ax.margins(x=.08,y=.18)
        fig.legend(handles=legend_handles(),loc='upper center',bbox_to_anchor=(.56,.995),ncol=3,frameon=False,columnspacing=1.0,handlelength=1.3)
        save(fig,'H2_PLGR/'+name);plt.close(fig)


def h3(rows):
    fig=plt.figure(figsize=(3.5,5.65))
    gs=fig.add_gridspec(3,2,left=.21,right=.975,bottom=.12,top=.855,hspace=.88,wspace=.55,height_ratios=[1,1,.78])
    for i,panel in enumerate(['qd_by_B','qp_by_qA']):
        for j,ds in enumerate(DS):
            ax=fig.add_subplot(gs[i,j]);rr=[r for r in rows if r['dataset']==ds and r['panel']==panel]
            xs=sorted({float(r['point'].split('_')[1]) for r in rr});ys=sorted({float(r['point'].split('_')[0]) for r in rr})
            z=np.zeros((3,3))
            for r in rr:
                y,x=map(float,r['point'].split('_'));z[ys.index(y),xs.index(x)]=float(r['delta_psnr'])
            im=ax.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,z,vmin=-.016,vmax=.016,cmap=HEATMAP,shading='flat',rasterized=False)
            for y in range(3):
                for x in range(3):
                    value=z[y,x];label=f'{value:+.3f}' if abs(value)>=.0005 else '.000'
                    label=label.replace('+0.','+.').replace('-0.','−.')
                    ax.text(x,y,label,ha='center',va='center',fontsize=7,color='white' if abs(value)>.008 else 'black')
            default=next(r for r in rr if r['is_default']);dy,dx=map(float,default['point'].split('_'))
            ax.plot(xs.index(dx)-.30,ys.index(dy)-.28,'k*',ms=5.5,zorder=5)
            ax.invert_yaxis()
            ax.set_xticks(range(3));ax.set_xticklabels([f'{x:g}' for x in xs],fontsize=8)
            ax.set_yticks(range(3));ax.set_yticklabels([f'{x:.2f}' for x in ys],fontsize=8)
            ax.set_xlabel('$B$' if i==0 else '$q_A$',labelpad=5)
            ax.set_ylabel('$q_d$' if i==0 else '$q_p$',labelpad=1)
            ax.tick_params(length=2,pad=2)
            ax.set_title(f'({chr(97+i*2+j)}) {NAMES[ds]}',loc='left',pad=4)
    cax=fig.add_axes([.24,.94,.65,.023])
    cax.set_label('qa_colorbar')
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.01,0,.01]);cb.ax.tick_params(labelsize=8,length=2,pad=2)
    cb.solids.set_rasterized(False)
    cb.ax.xaxis.set_label_position('top');cb.set_label(r'$\Delta$PSNR relative to default (dB)',labelpad=4,fontsize=9)
    ax=fig.add_subplot(gs[2,:]);rr=[r for r in rows if r['panel']=='tau_alpha'];line(ax,rr,'point','delta_psnr',1e-4)
    ax.set_xscale('log');ax.set_xticks([5e-5,1e-4,5e-4]);ax.set_xticklabels([r'$5\!\times\!10^{-5}$',r'$10^{-4}$',r'$5\!\times\!10^{-4}$']);ax.minorticks_off()
    ax.set_xlim(3.8e-5,7.4e-4)
    ax.set_xlabel(r'$\tau_\alpha$',labelpad=1);ax.set_ylabel(r'$\Delta$PSNR (dB)',labelpad=2);ax.set_title('(e) Opacity threshold',loc='left',pad=4)
    fig.legend(handles=legend_handles(),loc='lower center',bbox_to_anchor=(.55,.005),ncol=3,frameon=False,handlelength=1.2,columnspacing=.9)
    save(fig,'H3_CGAS/H3');plt.close(fig)


def h4(rows):
    fig,axs=plt.subplots(2,1,figsize=(3.5,3.8),sharex=True)
    fig.subplots_adjust(left=.23,right=.975,bottom=.15,top=.86,hspace=.43)
    for ax,ds,c in zip(axs,DS,COLORS):
        rr=[r for r in rows if r['dataset']==ds]
        for r in rr:
            x,y=float(r['G_k']),float(r['PSNR'])
            if r['is_default']:
                ax.plot(x,y,marker='*',color='black',ms=7,zorder=6)
            elif r['structural_variant']:
                ax.scatter(x,y,marker='D',s=19,c=c,zorder=4)
            else:
                ax.scatter(x,y,marker='o',s=16,facecolors='none',edgecolors=c,lw=.8,zorder=2)
            if r['structural_variant']=='RGD':
                ax.annotate('RGD (no pruning)',(x,y),xytext=(-8,12),textcoords='offset points',ha='right',fontsize=7,arrowprops=dict(arrowstyle='-',color='.4',lw=.5))
        axes_style(ax);ax.grid(False);ax.set_title(NAMES[ds],loc='left',color='black',pad=3)
        ax.set_ylabel('PSNR (dB)');ax.set_xlim(530,1090);ax.margins(y=.23)
    axs[1].set_xlabel('Final Gaussian population, #G (k)',fontsize=9);axs[1].set_xticks([600,800,1000])
    handles=[Line2D([],[],marker='o',mfc='none',mec='.3',ls='none',label='Sensitivity'),Line2D([],[],marker='D',color='.3',ls='none',label='Structural'),Line2D([],[],marker='*',color='k',ls='none',ms=7,label='Default')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,.99),ncol=3,frameon=False,columnspacing=.75,handlelength=.9)
    save(fig,'H4_CGAS_population/H4');plt.close(fig)


def reports(ranges,scatter):
    text('H1_AFGM/analysis.md',r'''# H1 — AFGM spacing sensitivity

Authoritative source: `docs/production/final_paper_evidence_20260910/ppp/rho_data.tsv` (legacy path only). Frozen Fixed+Full renderer, Full CGAS, AFGM; rho=.60,.75,.90,1,1.10,1.25,1.50. Default rho=1.

Candidate A (H1.pdf/png): single panel, spacing on x and default-relative PSNR in dB on y. Liver/Arm have consistent colors/markers; star and dashed vertical reference mark the nominal setting. Recommended: direct parameter reading, no incompatible units or dual axes. H1_candidate_A is the same figure.

Candidate B: single-panel PSNR–LPIPS trade-off trajectory, each quantity on its own dimensional axis. Endpoints and Arm rho=1.25 are labeled, other samples remain in plotted_data.tsv. Scientifically valid but less direct: rho is not the x-axis, and nearby Arm points crowd the default. Keep for internal review, not preferred for the main paper. It does NOT treat dB and LPIPS as comparable magnitudes.

Liver increasing rho monotonically lowers PSNR/SSIM and improves LPIPS. Arm sampled PSNR peaks at rho=1.25, only 0.0689481 dB above rho=1. This is a sampled optimum, not a continuous optimum. rho=1 represents nominal acquisition spacing without per-sequence tuning; no empirical-default-optimum or universal-plateau claim. If candidate A is used, prose must retain the Liver perceptual trade-off rather than equating PSNR with all quality.
''')
    text('H2_PLGR/analysis.md',r'''# H2 — PLGR numerical sensitivity

Source: `docs/production/final_paper_evidence_20260910/lgr/hyperparameter_data.tsv`, checked against archived configs and `lgr/hyperparameter_ranges.tsv`. All local modes are Fixed, never Adaptive. MP and Full CGAS are fixed.

Five vertically stacked panels at 3.5-inch width. Each displays Liver/Arm delta PSNR against one parameter; defaults are starred. Shared typography, independently labeled y scales (do not infer equal sensitivity from equal visual line slope). No smoothing. LPIPS companion uses the same layout and original units, not a blended metric.

| Parameter | Actual completed values | Default |
|---|---|---|
| sigma_ele (mm) | 0.8,1,1.5,2 | 1 |
| beta | 0.4,0.6,0.8,1 | 0.8 |
| lambda_ray | 0.2,0.35,0.55,0.75,1 | 0.55 |
| Nq | 1,2,3 | 3 |
| half-window (mm) | 1,2,3,4 | 2 |

- Width is materially sensitive: PSNR ranges 0.5873178 / 0.7704966 dB (Liver/Arm). Larger widths increase Liver PSNR but reduce Arm PSNR; metric/domain trade-offs remain. 1 mm is a shared operating point, not universal optimum.
- Beta is moderately sensitive: ranges 0.0912602 / 0.2570081 dB, with opposite dataset trends. LPIPS trade-offs are retained in the review, not hidden by a blanket robustness claim.
- Lambda is stable over the tested interval: ranges 0.0171847 / 0.0541477 dB.
- Nq is non-monotonic: q=2 is poor, q=1 and q=3 are substantially better. Range 0.4096548 / 0.7780779 dB. No convergence/saturation claim or retuning.
- Half-window quality ranges 0.0206915 / 0.0244875 dB. No controlled runtime benchmark per point exists; no claim that larger windows increase measured cost. Raw candidate statistics are retained in plotted_data.tsv but co-adapted Gaussian populations prevent a controlled cost interpretation.
''')
    text('H3_CGAS/analysis.md',r'''# H3 — CGAS selection/pruning sensitivity

Source: `docs/production/final_paper_evidence_20260910/cga/hyperparameter_data.tsv`; selection interpretation checked against `cga/selection_behavior_summary.tsv`. MP, Fixed+Full renderer, accumulated path contribution. No old high-tau results.

Two columns (Liver/Arm), two 3x3 heatmap rows, then a spanning tau_alpha marker-line panel. All colors share a zero-centered ±0.016 dB scale, not separate rescaling per heatmap; cell text is delta PSNR to three decimals. A black star marks the default grid cell. Sampled maxima are deliberately not highlighted: this is robustness characterization, not parameter selection. Rows increase downward, matching the reference. The line panel uses default stars. Tau is plotted on a logarithmic x-axis at its actual sampled values; a line panel preserves numeric spacing better than a categorical 1D strip. Small cell differences are descriptive, not significance; zero-looking rounded cells can contain nonzero raw differences.

Actual grids: qd={.96,.98,.99} x B={.001,.005,.05}, default(.98,.05); qp={.05,.10,.15} x qA={.70,.80,.90}, default(.10,.80); tau_alpha={5e-5,1e-4,5e-4}, default1e-4. Early proposed B={.02,.05,.08} was NOT the executed grid and must not be used to label these figures.

Every dataset-specific scan has PSNR range <0.016 dB (largest 0.0159422). This supports quality stability within the tested protocol, not independent activity of every control. All 216 events in the 18 qp-by-qA settings have local contribution threshold zero: qp is saturated. Apparent heatmap variation across qp cannot be attributed to an active local threshold; training numerical variability/co-adaptation is not a controlled seed test. qA genuinely changes protection fractions (~30/20/10%) and population. B is partially active: low budgets can bind; higher budgets are often quantile-limited. See saved selection_behavior_audit.tsv for the event-based evidence.
''')
    rr=[]
    for ds in DS:
        sub=[r for r in scatter if r['dataset']==ds]
        rr.append(dict(dataset=NAMES[ds],unique_runs=len(sub),G_min=min(float(r['final_G']) for r in sub),G_max=max(float(r['final_G']) for r in sub),PSNR_min=min(float(r['PSNR']) for r in sub),PSNR_max=max(float(r['PSNR']) for r in sub),PSNR_range=max(float(r['PSNR']) for r in sub)-min(float(r['PSNR']) for r in sub)))
    write('H4_CGAS_population/ranges.tsv',rr)
    text('H4_CGAS_population/analysis.md',r'''# H4 — CGAS quality–population relationship

Authoritative sources: `cga/hyperparameter_data.tsv` and `cga/structural_sources.tsv` under the sealed final evidence package; independently matched against `cga/quality_population.tsv`. Plot each physical output run once, merging repeated default memberships by output path. Original memberships/IDs are in plotted_data.tsv. No point jitter or regression line.

KEEP. The unpruned RGD cases (1,050,672 Liver / 1,022,586 Arm) extend far beyond the sensitivity scans and add information absent from H3. Sensitivity points use hollow circles; structural strategies use diamonds; the production Full CGAS default uses a star. Separate vertically aligned Liver/Arm axes retain actual PSNR units and show narrow within-dataset changes without a ~3.4-dB inter-dataset gap flattening both clouds. The x-axis is shared; y-scales are explicitly dataset-specific.

Population changes substantially while PSNR stays within 0.0277768 dB for Liver and 0.1160482 dB for Arm across the combined evidence. This is an observed quality–capacity trade-off, not a monotonic causal law, efficiency ratio, or argument that fewer primitives are always better. GD vs unpruned RGD changes pruning as well as densification. No fitted lines. Overlapping points remain overlapping: multiplicity is not invented by offsets.
''')
    text('FINAL_HYPERPARAMETER_FIGURE_AUDIT.md',r'''# Final single-column hyperparameter figure audit

No training, jobs, hyperparameter/default changes, manuscript edits or ZIP. Existing frozen per-run values only; arithmetic deltas recomputed and matched to sealed values to 1e-12. Original frozen files are unchanged. All plotted renderer configurations verify fixed local + full integral global; archived config paths/hashes are in the TSVs/source_audit.json. Corrected older Fixed runs reused as defaults retain execution hashes and P2 equivalence, never relabeled Adaptive results.

The user-specified reference remains the typography baseline: serif text, matching DejaVu Serif mathtext and closed four-sided axes. The installed nature-figure skill guided the subsequent revision: restrained #356A93 / #BD633E palette, circle/solid versus square/dashed dataset curves, sparse neutral grid, physically aligned panel edges and editable SVG export. Black stars identify defaults; sampled maxima are no longer highlighted. The four-figure scientific structure and all numerical data remain unchanged. See FIGURE_CONTRACT.md and SKILL_REVISION_AUDIT.md for provenance and rendered QA.

| Figure | Recommended layout | Interpretation |
|---|---|---|
| H1 AFGM | One parameter-line panel, delta PSNR; candidate B trade-off plot kept separately | Nominal rho=1, not an empirical optimum; dataset/metric-dependent response |
| H2 PLGR | Five vertically stacked sensitivity panels; LPIPS companion retained | Width/beta trade-offs, stable lambda/window, NON-monotonic Nq |
| H3 CGAS | Two pairs of heatmaps + one tau marker-line panel | Stable quality, but qp saturated and other controls only partly/genuinely active |
| H4 CGAS population | Two aligned raw-PSNR scatter facets | KEEP: structural unpruned RGD extends the explored capacity range |

Exact plotted values/defaults are in each analysis.md and plotted_data.tsv. Use actual executed sigma_ele=.8/1/1.5/2 and B=.001/.005/.05, not early proposed points. Structural RGD in H4 intentionally has pruning off; “all full CGAS” applies to sensitivity points, not structural controls.

Figures preserve exactly 3.5-inch PDF width (no tight-bbox resizing), embedded TrueType/vector plotting (including heatmap cells/colorbar), 400-dpi PNG, 8–8.5-pt principal tick/legend type and 9–10-pt labels. Optional trade-off annotations remain smaller. The separate *_size_review.png files are 150-dpi small-size previews, not extra paper figures. H2 is 6.8 inches tall after tightening panel spacing; it still occupies substantial column height, a layout cost rather than a reason to combine parameters.

For H1 A, prose must explicitly retain the LPIPS trade-off. Different H2 y-axis ranges must not be mistaken for comparable sensitivities. H3 cells use shared color scaling and rounded labels; small numerical differences are not significant-effects claims. H4 dataset-specific PSNR scales are clearly labeled, with no fitted trend or dual axis. No controlled cost claim is made for half-window changes.

This package is for visual review, not automatic manuscript inclusion. No additional experiment is required or authorized.
''')


def main():
    OUT.mkdir(parents=True,exist_ok=True);style()
    rho,plgr,cgas,scatter,ranges=prepare()
    h1(rho);h2(plgr);h3(cgas);h4(scatter);reports(ranges,scatter)
    text('render_audit.json',json.dumps(FIGURES,indent=2))
    assert all(f['width_in']==3.5 for f in FIGURES)
    print(json.dumps(FIGURES,indent=2))


if __name__=='__main__':main()
