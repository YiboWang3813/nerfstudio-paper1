"""Reporting-only Batch1 tables, inherited-frame contact sheets and CGA accounting."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
REPORT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports'
DOC=ROOT/'docs/production/p1_paper_protocol_freeze'
METRICS=('psnr','l1','ssim','lpips','dists','gmsd')


def read(name):
    return list(csv.DictReader((REPORT/(name+'.tsv')).open(),delimiter='\t'))


def write(name,rows):
    if not rows:return
    with (REPORT/(name+'.tsv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),delimiter='\t');w.writeheader();w.writerows(rows)


def tables():
    aggregate=read('main_complete_dataset_aggregates');long=[]
    for ds in sorted({r['dataset'] for r in aggregate}):
        group=[r for r in aggregate if r['dataset']==ds and r['method']!='ours_legacy']
        for m in METRICS:
            ordered=sorted(group,key=lambda r:float(r['final_'+m]),reverse=m in ('psnr','ssim'))
            for i,r in enumerate(ordered):
                v=float(r['final_'+m]);long.append(dict(dataset=ds,method=r['method'],metric=m,raw_mean=v,displayed='{:.4f}'.format(v),rank=i+1,best=ordered[0]['method'],second_best=ordered[1]['method']))
    write('main_corrected_values_display_ranks',long)
    final_table=[r for r in aggregate if r['method']!='ours_legacy']
    assert len(final_table)==32
    write('main_corrected_table',final_table)
    for name in ('ppp_internal','cga_structural'):
        rows=read(name)
        for r in rows:r['final_G_k']=float(r['final_num_gaussians'])/1000
        write(name+'_candidate',rows)
    events=[];summaries=[]
    for r in read('cga_structural'):
        event={}
        with (Path(r['output_dir'])/'logs/scalars.tsv').open() as f:
            for x in csv.DictReader(f,delimiter='\t'):
                if x['key'].startswith('train/dc/'):
                    event.setdefault(int(x['step']),{})[x['key'].split('/')[-1]]=float(x['value'])
        # last_event_stats may be logged repeatedly; adaptation iteration is identified
        # by changes of the (before,after) population and the frozen 2000-step schedule.
        selected={step:e for step,e in event.items() if step%2000==0 and 'num_gaussians_before' in e}
        spawned=pruned=protected=0
        for step,e in sorted(selected.items()):
            add=int(e['num_gaussians_after']-e['num_gaussians_before']+e['prune_weak'])
            assert add>=0
            spawned+=add;pruned+=int(e['prune_weak']);protected+=int(e.get('path_protected_candidate_count',0))
            events.append(dict(dataset=r['dataset'],label=r['label'],iteration=step,spawned=add,pruned=int(e['prune_weak']),path_protected_candidate_decisions=int(e.get('path_protected_candidate_count',0)),**e))
        assert 1000000+spawned-pruned==int(float(r['final_num_gaussians'])),(r['dataset'],r['label'],spawned,pruned)
        summaries.append(dict(dataset=r['dataset'],label=r['label'],events=len(selected),initial_G=1000000,final_G=int(float(r['final_num_gaussians'])),total_spawned=spawned,total_pruned=pruned,path_protected_candidate_decisions=protected,PSNR=float(r['final_psnr']),LPIPS=float(r['final_lpips'])))
    write('cga_event_accounting',events);write('cga_population_accounting',summaries)
    delta=[]
    for ds in sorted({r['dataset'] for r in summaries}):
        a=next(r for r in summaries if r['dataset']==ds and r['label']=='RGD+LP');b=next(r for r in summaries if r['dataset']==ds and r['label']=='Full')
        delta.append(dict(dataset=ds,extra_final_G=b['final_G']-a['final_G'],delta_PSNR=b['PSNR']-a['PSNR'],delta_LPIPS=b['LPIPS']-a['LPIPS']))
    write('cga_full_minus_lp',delta)


def efficiency():
    # Preserve old paper baseline panel numbers verbatim (published precision),
    # rather than recomputing a changed aggregate from incomplete raw metadata.
    source=ROOT/'docs/production/comparison_final_outputs/quantitative_tabs.md'
    frozen=[];panel=None
    for line in source.read_text().splitlines():
        if line.startswith('## Table 1'):panel='synthetic_phantom'
        if line.startswith('## Table 2'):panel='real'
        if line.startswith('| ') and panel:
            parts=[s.strip() for s in line.strip('|').split('|')]
            if len(parts)==15 and parts[0] not in ('Method','Ours'):
                try:t=float(parts[-2]);fps=float(parts[-1])
                except ValueError:continue
                frozen.append(dict(panel=panel,method=parts[0],training_seconds=t,FPS=fps,status='FINAL-REUSE_frozen_display_precision',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    assert len(frozen)==14
    write('baseline_efficiency_frozen_panels',frozen)
    raw=read('corrected_main_fps');assert len(raw)==78 and all(r['status']=='success' for r in raw)
    fair=read('fair_fps_comparison');main=read('main_ours_corrected_row');datasets=[]
    for r in main:
        ds=r['dataset'];fps=[float(x['render_fps']) for x in raw if x['dataset']==ds];f=next(x for x in fair if x['dataset']==ds and x['method']=='ours')
        datasets.append(dict(dataset=ds,num_training_cases=int(r['n']),training_seconds=float(r['train_wall_time_sec']),all_case_FPS_mean=statistics.mean(fps),all_case_min_FPS=min(fps),paper_matched_two_case_FPS=float(f['render_fps_mean']),paper_FPS_cases=f['selected_ids']))
    write('corrected_efficiency_datasets',datasets)
    for panel,pair in [('synthetic_phantom',('synthetic_liver','spine_phantom')),('real',('tus_rec_2024_qc','tdsc_abus_2023'))]:
        members=[r for r in datasets if r['dataset'] in pair]
        frozen.append(dict(panel=panel,method='Ours corrected',training_seconds=statistics.mean(r['training_seconds'] for r in members),FPS=statistics.mean(r['paper_matched_two_case_FPS'] for r in members),status='corrected_equal_dataset_mean',source='corrected_efficiency_datasets.tsv',source_sha256=''))
    write('main_corrected_efficiency_panels',frozen)
    mapping={'UltraNeRF':'ultranerf','NeRF-US w/o prior':'nerfus_no_prior','UltraGauss':'ultragauss','UltraGray':'ultragray','UltraGS baseline':'ultrags_baseline','Slice-adapted 3DGS':'slice_adapted_3dgs','GaussianPile':'gaussianpile','Ours corrected':'ours_corrected'}
    quality={(r['dataset'],r['method']):r for r in read('main_corrected_table')}
    for panel,pair in [('synthetic_phantom',('synthetic_liver','spine_phantom')),('real',('tus_rec_2024_qc','tdsc_abus_2023'))]:
        combined=[]
        for e in frozen:
            if e['panel']!=panel:continue
            row=dict(method=mapping[e['method']],training_seconds=e['training_seconds'],FPS=e['FPS'],efficiency_status=e['status'])
            for ds in pair:
                q=quality[ds,row['method']]
                for m in METRICS:row[ds+'_'+m]=float(q['final_'+m])
            combined.append(row)
        assert len(combined)==8
        write('main_corrected_panel_'+panel,combined)


def claims():
    from scripts.production.ablations.ours.p1_scientific_checkpoint import table
    values=read('main_corrected_values_display_ranks')
    ours=[r for r in values if r['method']=='ours_corrected']
    source=ROOT/'docs/production/experiment_results_io/04_experiments_and_results.tex'
    rows=read('main_complete_dataset_aggregates')
    lookup={(r['dataset'],r['method']):r for r in rows}
    comparisons=[]
    targets={'synthetic_liver':['nerfus_no_prior','ultrags_baseline','slice_adapted_3dgs'],
             'spine_phantom':['nerfus_no_prior','ultragray','gaussianpile'],
             'tus_rec_2024_qc':['nerfus_no_prior','ultragray','slice_adapted_3dgs'],
             'tdsc_abus_2023':['nerfus_no_prior','ultragauss','slice_adapted_3dgs']}
    for ds,methods in targets.items():
        a=lookup[ds,'ours_corrected']
        for method in methods:
            b=lookup[ds,method]
            comparisons.append(dict(dataset=ds,baseline=method,baseline_PSNR=float(b['final_psnr']),corrected_PSNR=float(a['final_psnr']),PSNR_delta=float(a['final_psnr'])-float(b['final_psnr']),baseline_SSIM=float(b['final_ssim']),corrected_SSIM=float(a['final_ssim']),L1_reduction_percent=100*(1-float(a['final_l1'])/float(b['final_l1']))))
    write('main_results_claim_replacements',comparisons)
    eff=read('corrected_efficiency_datasets');panels=read('main_corrected_efficiency_panels')
    text=['# Corrected main claim audit',
          'Audit target is the supplied Results exchange copy, not a confirmed canonical LaTeX source. No manuscript was edited. Source SHA256: '+hashlib.sha256(source.read_bytes()).hexdigest(),
          'Quality uses all frozen 6/9/33/30 cases, eight methods, equal sweep weights and unrounded ranks. Baselines quality are FINAL-REUSE.',
          '## A. Still supported',
          '- PSNR first on Liver, Spine and Breast; L1 first on all four. SSIM first on Liver/Spine/Breast, second on Arm. Breast wins all six metrics.\n- Reconstruction accuracy remains above both NeRF methods on the evaluated datasets.\n- Perceptual distances improve against legacy Ours, but are not best on every dataset.\n- Pure-forward rendering exceeds a stated 30 FPS reference threshold on all measured cases; this does not establish end-to-end clinical/display latency.',
          '## B. Exact numbers must change',
          'All historical Ours values, PSNR increments, L1 percentages and 240–353 FPS text require replacement. The following rows cover the named baseline contrasts in the three comparison subsections; raw values are saved, not inferred from displayed rounding.',
          table(comparisons,['dataset','baseline','baseline_PSNR','corrected_PSNR','PSNR_delta','baseline_SSIM','corrected_SSIM','L1_reduction_percent']),
          'Dataset counts must read 6 Liver / 9 Spine / 33 evaluated Arm QC segments / 30 Breast, separating 48 raw Arm sweeps. Single4090 training wording is incorrect: Batch1 used4080SUPER except two4090 anchors. d0 is fixed nominal/calibrated spacing, not verified runtime device metadata.',
          '## C. Must weaken',
          '- "Strongest fidelity across all four datasets" -> specify metric/dataset wins, including Arm exceptions.\n- "Consistently sharper and richer than all baselines" -> inherited-case visual observations only; UltraGray/3DGS remain competitive. See qualitative audit.\n- PPP preserves multi-sweep PSNR/SSIM, not all quality metrics; Liver representative LPIPS changes about .180 to .223.\n- CGA quality stays in a narrow band; no significant-PSNR or LP-always-better claim.\n- Full LGR and Adaptive advantages await the 11-case confirmation; no universal monotonic claim.\n- rho=1 is the nominal fixed spacing, not an observed optimum/plateau.',
          '## D. No longer supported / do not use',
          '- Best PSNR/L1/SSIM on every dataset: false for Arm PSNR and SSIM.\n- Arm PSNR exceeds UltraGray by .43dB: now lower by .0468816030303dB.\n- Arm PSNR exceeds adapted3DGS: now lower by .1857811dB; Ours is third.\n- Outperforms both adapted Gaussian methods on every dataset/metric: false.\n- Less training time than both adapted baselines: not supported by current recorded times and not hardware-controlled.\n- Covariance correction accelerates training: not established by shared-worker wall time.\n- Final caption claiming uniformly finer structures than every baseline: not established by the inherited images.',
          '## Full rank and display audit',table(ours,['dataset','metric','raw_mean','displayed','rank','best','second_best']),
          '## Corrected efficiency',table(eff,list(eff[0])),table(panels,list(panels[0])),
          'Paper FPS uses the same two cases per dataset as every baseline, then equal-weight dataset means for Liver+Spine and Arm+Breast. All78-case FPS means/minima are provided separately and are not substituted into a two-case comparison. Timing means cover all frozen cases. FPS uses verified4090, 5warmup,20heldout frames,3passes, pure forward.',
          'Baseline efficiency panel cells are reused verbatim at their frozen published precision. Historical Time aggregation was inconsistent (GS pooled sweeps, implicit equal datasets); this audit does not silently redefine baseline cells. Corrected Ours uses the P1 equal-dataset rule. These Time columns must carry the heterogeneity caveat, not support a controlled efficiency superiority claim. Raw baseline missing-time coverage is separately reported in main_complete_dataset_aggregates.tsv; missing is never zero.',
          'No Batch2 or algorithm modification is authorized by this report.']
    (DOC/'main_corrected_claim_audit.md').write_text('\n\n'.join(text)+'\n')


def images():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from scripts.production.comparisons.us_recon_four_dataset_p100k_v1 import make_tmi_all_datasets_2d_rendering_v2 as drawing
    out=REPORT/'contact_sheets';out.mkdir(exist_ok=True)
    allrows=read('completed_per_run');manifest=[]
    def sheet(name,rows,item,crop):
        gt=Path(rows[0][1])/'vis'/('item_'+str(item))/'gt.png'
        gh=hashlib.sha256(gt.read_bytes()).hexdigest()
        for label,run in rows:assert hashlib.sha256((Path(run)/'vis'/('item_'+str(item))/'gt.png').read_bytes()).hexdigest()==gh
        entries=[(label,Path(run)/'vis'/('item_'+str(item))/'pred.png') for label,run in rows]+[('GT',gt)]
        fig,axes=plt.subplots(2,len(entries),figsize=(2.3*len(entries),4.6),squeeze=False)
        x,y,w,h=crop
        for i,(label,path) in enumerate(entries):
            im=np.asarray(Image.open(path).convert('L'))/255.
            axes[0,i].imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[0,i].set_title(label,fontsize=9)
            axes[1,i].imshow(im[y:y+h,x:x+w],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            for ax in axes[:,i]:ax.axis('off')
            manifest.append(dict(sheet=name,label=label,item=item,crop_xywh=crop,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gt_sha256=gh))
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    old=json.loads((DOC/'legacy_main_selected_rows.json').read_text())
    summary={(r['dataset'],r['sweep'],r['method']):r for r in old}
    for r in allrows:
        if r['group']=='main':summary[r['dataset'],r['sweep'],'ours']=r
    drawing._read_rows=lambda:summary
    drawing.FIGURE_DIR=out;drawing.MANIFEST_DIR=out;drawing.CONFIG_PATH=out/'main_figure_provenance.md'
    drawing.main()
    drawing.CONFIG_PATH.write_text('# Corrected main qualitative figure\n\nRegenerate with `PYTHONPATH=. python scripts/production/ablations/ours/p1_finish_reports.py images` in the project environment.\n\nOnly corrected Ours panels replace legacy Ours; baseline images, inherited cases/items/crops and the original TMI v2 layout are unchanged. Inputs: frozen legacy_main_selected_rows.json plus corrected completed_per_run.tsv. See paper_all_datasets_2d_rendering_manifest.json and contact_sheet_manifest.json for resolved sources and image hashes.\n')
    # Existing generator emits an old summary-files hint; replace only that hint.
    mp=out/'paper_all_datasets_2d_rendering_manifest.json';value=json.loads(mp.read_text());value['summary_files']=[str(DOC/'legacy_main_selected_rows.json'),str(REPORT/'completed_per_run.tsv')];mp.write_text(json.dumps(value,indent=2)+'\n')
    for spec in drawing.ROWS:
        rs=[(label,summary[spec.dataset,spec.sweep,method]['output_dir']) for method,label in drawing.METHODS if method!='gt']
        sheet('main_'+spec.label,rs,spec.item,list(spec.crop))
    jointspec=json.loads((ROOT/'docs/production/ablation_final_outputs/manifests/paper_ablation_joint_visual_manifest.json').read_text())
    internal=json.loads((ROOT/'docs/production/ablation_final_outputs/manifests/paper_ablation_internal_visual_manifest.json').read_text())
    for spec in jointspec:
        ds=spec['dataset'];name='Liver' if ds=='synthetic_liver' else 'Arm'
        jr=[r for r in read('joint') if r['dataset']==ds]
        sheet('joint_'+name,[('P{} L{} C{}'.format(r['PPP'],r['LGR'],r['CGA']),r['output_dir']) for r in jr],spec['item'],spec['crop_xywh'])
        for table,group in [('ppp_internal','Easy-Pose'),('lgr_structural','Renderer'),('cga_structural','DC')]:
            inherited=next(s for s in internal if s['group']==group and s['dataset_label']==('Synthetic' if name=='Liver' else 'TUS-QC'))
            rows=[(r['label'],r['output_dir']) for r in read(table) if r['dataset']==ds]
            sheet(table+'_'+name,rows,inherited['item'],inherited['crop_xywh'])
    (out/'contact_sheet_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    {'tables':tables,'efficiency':efficiency,'images':images,'claims':claims}[sys.argv[1]]()
