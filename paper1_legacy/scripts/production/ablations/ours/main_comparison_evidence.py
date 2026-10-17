"""Extract existing frozen Main evidence; never train, evaluate or edit manuscript."""
import csv
import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'docs/production/final_paper_evidence_20260910'
OUT=P/'main_comparison'
getcontext().prec=40
DATA=[('synthetic_liver','Liver',6),('spine_phantom','Spine',9),('tus_rec_2024_qc','Arm',33),('tdsc_abus_2023','Breast',30)]
METHODS=[('ultranerf','UltraNeRF'),('nerfus_no_prior','NeRF-US'),('ultragauss','UltraGauss'),('ultragray','UltraGRay'),('ultrags_baseline','UltraGS'),('slice_adapted_3dgs','3DGS'),('gaussianpile','GaussianPile'),('ours_fixed_final','ULGSplat')]
NAMES=dict(METHODS)
METRICS=['psnr','l1','ssim','lpips','dists','gmsd']
HIGH={'psnr','ssim'}
CAT={'NeRF-based':['ultranerf','nerfus_no_prior'],'Ultrasound Gaussian':['ultragauss','ultragray','ultrags_baseline'],'Adapted Gaussian':['slice_adapted_3dgs','gaussianpile']}
INPUTS={}


def read(rel):
    p=P/rel;INPUTS[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix=='.json':return json.loads(p.read_text())
    if p.suffix in ['.md','.tex']:return p.read_text()
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def tsv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,keys,delimiter='\t');w.writeheader();w.writerows(rows)


def markdown(rows,keys):
    def fmt(v):return str(v) if v not in [None,''] else 'NA'
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in rows)+'\n'


def dec(v):return Decimal(v)


def main():
    OUT.mkdir(exist_ok=True)
    manifest=read('audits/final_data_source_manifest.json')
    table=read('main/final_main_table_data.tsv');oldrank=read('main/final_main_exact_display_ranks.tsv')
    frozen=read('audits/final_refresh_matrix.json');baseline=read('audits/baseline_sources.json')
    protocol=read('audits/paper_final_dataset_protocol.tsv')
    read('audits/paper_final_metric_protocol.md');read('main/final_main_claims.md')
    read('qualitative/final_qualitative_audit.md')
    fps=read('main/final_fixed_fps_datasets.tsv');fpsruns=read('main/final_fixed_fps_per_run.tsv')
    eff=read('main/final_baseline_efficiency_frozen_panels.tsv');ourpanel=read('main/final_main_fixed_efficiency_panels.tsv')
    assert len(table)==32 and len(fpsruns)==8
    lookup={(r['dataset'],r['method']):r for r in table}
    source=[];config_checks=[]
    from omegaconf import OmegaConf
    from configs.setup import load_hydra_config
    for ds,name,n in DATA:
        selected={r['case_id'] for r in protocol if r['dataset']==ds and r['include_in_main']=='1'}
        assert len(selected)==n
        ours=[r for r in frozen if r['dataset']==ds and 'main' in r['consumers']]
        assert {r['sweep'] for r in ours}==selected
        for method,label in METHODS:
            assert dec(lookup[ds,method]['n'])==n
            if method!='ours_fixed_final':
                rr=[r for r in baseline if r['dataset']==ds and r['method']==method]
                assert len(rr)==n and {r['sweep'] for r in rr}==selected
                for r in rr:
                    source.append(dict(dataset=name,method=label,case_id=r['sweep'],output_dir=r['output_dir'],resolved_config=r['output_dir']+'/resolved_config.yaml',packaged_config='',status='FINAL-REUSE baseline',source_table=r.get('source_table','')))
            else:
                for r in ours:
                    rel='audits/resolved_configs/structural_main_'+r['batch1_id']+'.yaml'
                    p=P/rel;INPUTS[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
                    cfg=OmegaConf.to_container(OmegaConf.load(p),resolve=False);sr=cfg['model']['slice_render'];dc=cfg['model']['offline']['density_control']
                    assert cfg['scene']['loading']['pose_mode']=='file'
                    assert sr['local_conditioning_mode']=='fixed' and sr['us_focus_sigma_ele_0']==1.
                    assert sr['ray_attenuation_enabled'] and sr['ray_attenuation_formula']=='integral' and sr['ray_attenuation_source']=='attenuation'
                    assert dc['densification_policy']=='residual_guided' and dc['pruning_policy']=='path_aware'
                    assert dc['responsibility']['protect_attenuation_mode']=='path_contribution'
                    source.append(dict(dataset=name,method=label,case_id=r['sweep'],output_dir=r['output_dir'],resolved_config=r['output_dir']+'/resolved_config.yaml',packaged_config=rel,status='FINAL Fixed+Full MP',source_table='audits/final_refresh_matrix.json'))
                # Current production resolved with the frozen Main MP overrides, not a training call.
                r=ours[0];old=OmegaConf.to_container(OmegaConf.load(P/('audits/resolved_configs/structural_main_'+r['batch1_id']+'.yaml')),resolve=False)
                current=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',r['overrides']),resolve=True)
                fields=['model','scene','loss','engine','metric']
                assert all(current[k]==old[k] for k in fields) and current['system']['seed']==old['system']['seed']
                config_checks.append(dict(dataset=name,anchor_row=r['batch1_id'],current_vs_frozen='EXACT MATCH for model/scene/loss/engine/metric/seed',pose_mode=current['scene']['loading']['pose_mode'],local_mode=current['model']['slice_render']['local_conditioning_mode'],global_formula=current['model']['slice_render']['ray_attenuation_formula']))
    # Verify all read frozen files against the sealed package. Do not chase external output paths.
    for rel,h in INPUTS.items():
        if rel!='audits/final_data_source_manifest.json':assert manifest['package_file_sha256'][rel]==h,rel
    assert len(source)==624
    tsv('final_main_case_sources.tsv',source)
    (OUT/'current_config_audit.json').write_text(json.dumps(config_checks,indent=2)+'\n')
    rows=[];ranks=[];rankmap={}
    for ds,label,n in DATA:
        for metric in METRICS:
            values={m:dec(lookup[ds,m]['final_'+metric]) for m,_ in METHODS}
            unique=sorted(set(values.values()),reverse=metric in HIGH)
            for m,name in METHODS:
                v=values[m];rank=1+sum(x>v if metric in HIGH else x<v for x in values.values())
                best=v==unique[0];second=v==unique[1]
                old=next(r for r in oldrank if r['dataset']==ds and r['method']==m and r['metric']==metric)
                assert rank==int(old['rank'])
                assert best==(old['best']==m) and second==(old['second_best']==m)
                ranks.append(dict(dataset=label,method=name,metric=metric,unrounded_value=str(v),rank=rank,best=best,second_best=second,expected_latex='bold' if best else 'underline' if second else 'plain',frozen_rank_metadata_matches=True))
                rankmap[ds,m,metric]=(best,second)
        for m,name in METHODS:
            r=lookup[ds,m];panel='synthetic_phantom' if ds in ['synthetic_liver','spine_phantom'] else 'real'
            if m=='ours_fixed_final':
                f=next(x for x in fps if x['dataset']==ds);e=next(x for x in ourpanel if x['panel']==panel)
                datasetfps=f['fps_mean'];pfps=e['fps'];ptime=e['time_seconds'];status='Frozen Fixed FPS: 2 cases/dataset'
            else:
                aliases={'UltraNeRF':'UltraNeRF','NeRF-US':'NeRF-US w/o prior','UltraGauss':'UltraGauss','UltraGRay':'UltraGray','UltraGS':'UltraGS baseline','3DGS':'Slice-adapted 3DGS','GaussianPile':'GaussianPile'}
                e=next(x for x in eff if x['panel']==panel and x['method']==aliases[name]);datasetfps='';pfps=e['FPS'];ptime=e['training_seconds'];status='Dataset FPS unavailable in allowed package; panel FPS frozen display precision'
            rows.append(dict(dataset=label,dataset_key=ds,method=name,method_key=m,n=n,pose_protocol='MP',**{k:r['final_'+k] for k in METRICS},
                             training_time_dataset_seconds=r['train_wall_time_sec'],fps_dataset=datasetfps,efficiency_panel=panel,training_time_panel_seconds=ptime,fps_panel=pfps,
                             efficiency_status=status,quality_source='main/final_main_table_data.tsv'))
    tsv('final_main_table_raw.tsv',rows);tsv('final_main_ranks.tsv',ranks)
    # Display formatting only after ranking; no rounding of stored raw means.
    tex=['% Evidence-only table bodies; not an edit of any manuscript.','% Columns: method + six quality metrics + dataset time(s) + dataset FPS.','% Bold = raw-value best; underline = raw-value second-best. NA is unavailable, never zero.']
    md=['# Final Main Comparison','Unrounded decimal strings are preserved below. **Bold**=best; <u>underline</u>=second-best, based on raw ranks.','NeRF-US denotes the frozen no-prior variant; 3DGS denotes slice-adapted 3DGS. Baseline dataset FPS is unavailable; use the separate frozen panel efficiency table.']
    for ds,label,n in DATA:
        md.append('## '+label+' (n='+str(n)+')');disp=[]
        tex += ['% '+label+' (n='+str(n)+')',r'\multicolumn{9}{l}{'+label+' ($n='+str(n)+r'$)} \\',r'% Method & PSNR & L1 & SSIM & LPIPS & DISTS & GMSD & Time(s) & FPS']
        for r in [x for x in rows if x['dataset']==label]:
            item={'Method':r['method']};tc=[r['method']]
            for k in METRICS:
                first,second=rankmap[ds,r['method_key'],k];v=r[k];item[k.upper()]=('**'+v+'**') if first else ('<u>'+v+'</u>') if second else v
                s=format(dec(v),'.4f');tc.append(r'\textbf{'+s+'}' if first else r'\underline{'+s+'}' if second else s)
            item['Time(s)']=r['training_time_dataset_seconds'];item['Dataset FPS']=r['fps_dataset'];disp.append(item)
            tc += [format(dec(r[k]),'.2f') if r[k] else '--' for k in ['training_time_dataset_seconds','fps_dataset']]
            tex.append(' & '.join(tc)+r' \\')
        md.append(markdown(disp,list(disp[0])));tex.append(r'\hline')
    (OUT/'final_main_table_markdown.md').write_text('\n\n'.join(md)+'\n')
    (OUT/'final_main_table_latex.tex').write_text('\n'.join(tex)+'\n')
    panels=[]
    for panel in ['synthetic_phantom','real']:
        for _,name in METHODS:
            r=next(r for r in rows if r['efficiency_panel']==panel and r['method']==name)
            panels.append(dict(panel=panel,method=name,training_seconds=r['training_time_panel_seconds'],fps=r['fps_panel'],precision='raw stored mean' if name=='ULGSplat' else 'frozen historical display precision'))
    tsv('final_main_efficiency_panels.tsv',panels)
    (OUT/'final_main_efficiency_panels.md').write_text(markdown(panels,list(panels[0])))
    deltas=[];category=[]
    for ds,label,n in DATA:
        ours=lookup[ds,'ours_fixed_final']
        for cat,methods in CAT.items():
            for m in methods:
                r=lookup[ds,m];wide=dict(category=cat,dataset=label,baseline=NAMES[m])
                for k in METRICS:
                    o,v=dec(ours['final_'+k]),dec(r['final_'+k]);diff=o-v
                    pct=(v-o)/v*100 if k not in HIGH and v!=0 else ''
                    deltas.append(dict(category=cat,dataset=label,baseline=NAMES[m],metric=k,ours=str(o),baseline_value=str(v),ours_minus_baseline=str(diff),relative_reduction_percent=str(pct),unit='dB' if k=='psnr' else 'metric units',favorable_to_ours=diff>0 if k in HIGH else diff<0,ssim_guidance='Use absolute difference, not percentage' if k=='ssim' else ''))
                    wide['delta_'+k]=format(diff,'.6f')
                    if pct!='':wide[k+'_reduction_percent']=format(pct,'.3f')
                category.append(wide)
    tsv('final_main_deltas.tsv',deltas);tsv('final_main_category_evidence.tsv',category)
    cons=[]
    for cat,methods in CAT.items():
        for method in methods:
            for k in METRICS:
                rr=[r for r in deltas if r['baseline']==NAMES[method] and r['metric']==k];w=sum(r['favorable_to_ours'] for r in rr);t=sum(dec(r['ours_minus_baseline'])==0 for r in rr)
                cons.append(dict(category=cat,baseline=NAMES[method],metric=k,wins=w,ties=t,losses=4-w-t,consistent_all_four=w==4))
    tsv('final_main_consistency.tsv',cons)
    for cat in CAT:
        sub=[r for r in category if r['category']==cat]
        (OUT/('category_'+cat.lower().replace(' ','_').replace('-','_')+'.md')).write_text('# '+cat+'\n\nAbsolute deltas are Ours minus baseline; positive reduction percentages favor Ours. SSIM uses absolute deltas.\n\n'+markdown(sub,list(sub[0])))
    claims=[r for r in ranks if r['method']=='ULGSplat']
    assert all(r['rank']==1 for r in claims if r['metric'] in ['psnr','l1','gmsd']) and all(r['rank']<=2 for r in claims)
    tsv('final_ulgsplat_ranks.tsv',claims)
    analysis(rows,claims,panels,cons,config_checks)
    (OUT/'source_audit.json').write_text(json.dumps(dict(inputs=INPUTS,authoritative_freeze=manifest['P2_source_sha256'],case_counts={name:n for _,name,n in DATA},MP_all_methods='frozen metric protocol; Ours independently checked in78 archived configs',current_production=config_checks,
        quality_rows=32,rank_cells=192,delta_cells=168,raw_rank_metadata_matches=True,manuscript_bold_underline='NOT INSPECTED: canonical manuscript outside allowed evidence; correct marks generated',baseline_dataset_FPS='UNAVAILABLE; panel aggregates only',training_or_evaluation_executed=False,manuscript_modified=False),indent=2)+'\n')
    print('Main evidence extraction complete:',OUT,'32 quality rows /192 ranks /168 deltas; NO training or reevaluation.')


def analysis(rows,claims,panels,cons,configs):
    own=[{k:r[k] for k in ['dataset','psnr','l1','ssim','lpips','dists','gmsd','training_time_dataset_seconds','fps_dataset']} for r in rows if r['method']=='ULGSplat']
    text='''# Main Comparison: evidence for rewriting (not manuscript prose)

## Authoritative inputs and protocol

All inputs are confined to `docs/production/final_paper_evidence_20260910/`
plus current production Hydra configuration. External output paths are recorded
for provenance only; no raw output was rerendered/re-evaluated or accessed to
fill missing efficiency metadata.

- `main/final_main_table_data.tsv`: authoritative32 unrounded quality/time rows.
- `main/final_main_exact_display_ranks.tsv`: frozen192 rank/display assignments,
  independently recomputed and verified here. Its values are not re-averaged.
- `audits/paper_final_dataset_protocol.tsv`: selected IDs,6/9/33/30.
- `audits/paper_final_metric_protocol.md`: MP for all methods, held-out frames,
  final30000 iteration, frame mean within case then equal-case dataset mean.
- `audits/final_refresh_matrix.json`, `audits/run_sources.tsv`, and packaged
  `audits/resolved_configs/structural_main_p2_*.yaml`: corrected Fixed row lineage.
- `audits/baseline_sources.json`: seven baseline methods' exact selected IDs
  match every dataset; historical Ours entries in this file are NOT quality inputs.
- `main/final_fixed_fps_per_run.tsv`, `final_fixed_fps_datasets.tsv`,
  `final_main_fixed_efficiency_panels.tsv`: final Fixed efficiency.
- `main/final_baseline_efficiency_frozen_panels.tsv`: historical baseline panel
  efficiency; precision is the frozen displayed precision, not invented raw digits.
- `qualitative/final_qualitative_audit.md`: only previously supported visual claims.

Current config entry points: `configs_hydra/experiment/production/ablations/ours/paper_semantic_freeze.yaml`
and its inherited production configs (including `configs_hydra/model/ours_us_focus.yaml`).
78 archived Ours configs verify MP,Fixed1mm,Full integral,independent attenuation,
residual-guided/path-aware accumulated-A_g CGA. Current Hydra resolved with frozen
Main overrides matches the archived model/scene/loss/engine/metric/seed on one
anchor from each dataset. Reading the base model YAML alone would be misleading:
Full global settings are supplied through experiment inheritance/overrides.
Current system's PPP default does not retroactively change the Main MP protocol.

P2 source SHA256: f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27.
The copied historical metric-protocol document names P0.5 in its header. That is
the protocol's origin, NOT the final default version; P2 lineage and actual Fixed
configs override any mistaken inference that the Main is Adaptive. Reused corrected
P0.5 Fixed runs retain original execution hashes plus certified P2 equivalence.

Method order follows the user's supplied manuscript order. Short names map to
UltraNeRF / NeRF-US(no prior) / UltraGauss / UltraGRay / UltraGS baseline /
slice-adapted3DGS / GaussianPile / ULGSplat. Do not imply a full-prior NeRF-US
comparison or an unmodified general-purpose3DGS implementation.

## Overall quantitative findings

1. Same6Liver/9Spine/33Arm/30Breast cases across eight methods; no method-specific subset.
2. ULGSplat ranks first in PSNR,L1,GMSD on all four datasets.
3. SSIM ranks first on Liver/Spine/Breast and second on Arm.
4. LPIPS ranks first on Spine/Arm/Breast and second on Liver; DISTS first on
   Arm/Breast and second on Liver/Spine. No “best in every metric” claim.
5. Compared with both NeRF baselines, all six dataset-level metrics improve on
   every dataset, within this frozen single-seed MP reconstruction protocol.
6. Against specialized Gaussian methods, gains are less uniform: UltraGRay wins
   Liver LPIPS and Liver/Spine DISTS. These exceptions are scientifically important.
7. On Arm, ULGSplat narrowly exceeds adapted3DGS PSNR but has slightly lower SSIM;
   quality advantages should not be generalized metric-by-metric without exceptions.
8. Fixed pure-forward throughput is201.3–307.4FPS across the dataset aggregates;
   this is not end-to-end frame-rate validation or proof of fastest training.

## Rank / formatting audit

Expected quoted claim: **SUPPORTED** for the frozen eight-method Main table.
All192 independently recomputed raw ranks and best/second identities match the
frozen rank metadata. Ranking uses Decimal representations of stored unrounded
means, higher for PSNR/SSIM and lower otherwise. No raw ties were found.
The package's previous generic tabular did not encode bold/underline; the current
canonical manuscript is not included in the allowed scope, so its existing marks
cannot be verified. `final_main_table_latex.tex` generates the correct marks in
the requested order; it is a new evidence fragment, not a manuscript edit.
Four-decimal display can produce apparent ties (e.g.Arm L1); marks still follow
raw ranks. `final_main_table_markdown.md` retains unrounded strings.

'''+markdown(own,list(own[0]))+'''

## NeRF-based comparison

**Strongest supported observations:** broad six-metric improvement over both
frozen NeRF baselines across all four datasets. State the within-sweep MP protocol,
not general superiority of every architecture/training budget.

**Recommended balanced numbers:** PSNR gains (Liver/Spine/Arm/Breast) are
+7.3465/+3.9077/+7.7442/+9.5409dB versus UltraNeRF and
+6.9534/+3.5859/+4.6855/+6.9862dB versus NeRF-US(no prior).
Pair these with cross-dataset ranges rather than a cherry-picked maximum:
L1 reductions27.47–69.47% /25.52–50.89%; LPIPS reductions63.55–74.42% /
69.00–78.38%, respectively. All exact per-dataset/per-baseline values are in
category_nerf_based.md and final_main_deltas.tsv.

**Avoid:** “9.54dB everywhere”; attributing the entire difference solely to one
local/global mechanism; claiming superiority to prior-enabled NeRF-US; significance
or generalization beyond the evaluated protocol. The module attribution belongs
to the separate controlled ablations, not this cross-method table alone.

## Ultrasound Gaussian comparison

**Consistent:** all four datasets improve PSNR,L1,SSIM,GMSD versus UltraGauss,
UltraGRay and UltraGS. Against UltraGauss/UltraGS, LPIPS/DISTS also improve4/4.
**Exceptions:** versus UltraGRay, Liver LPIPS is .003124051 higher(worse),
Liver DISTS .0131947227 higher, Spine DISTS .0266813377 higher. Do not hide these.

**Recommended numbers:** versus UltraGauss, PSNR gains1.2293–2.7947dB across
datasets. Versus UltraGS: +1.0093/+0.5718/+2.6409/+2.6997dB (Liver/Spine/Arm/Breast).
For the stronger UltraGRay comparison, explicitly show modest PSNR gains
+1.3670/+0.4164/+0.2960/+1.8309dB and its perceptual exceptions. On real Arm/Breast,
ULGSplat improves every metric over UltraGRay; LPIPS reductions24.42%/35.89%.

**Avoid:** all-perceptual superiority over every specialized Gaussian baseline;
explaining ranking as proof of a specific mechanism without ablation evidence;
claiming the largest real-data gain applies to both real datasets identically.

## Adapted Gaussian comparison

**Consistent:** GaussianPile is improved on all six metrics across4/4 datasets.
Adapted3DGS is improved in PSNR,L1,LPIPS,DISTS,GMSD across4/4, but Arm SSIM is
lower by .0024977911 (.886903376 vs .889401169 approximately; use raw table).

**Recommended numbers:** versus GaussianPile, PSNR gains+3.7948/+2.8183/+1.4865/
+2.1367dB. Versus adapted3DGS:+2.2962/+3.4458/+0.1571/+1.9071dB. The narrow
Arm PSNR margin (+.1571252dB) plus the SSIM reversal is more informative than
only citing the largest gain. Arm LPIPS decreases .0092132911 (8.45%), while
GMSD decreases .0007850496 (1.73%). These support a qualified mixed-metric
comparison, not a claim of overwhelming superiority on Arm.

**Avoid:** all metrics improve versus adapted3DGS; using the tiny Arm GMSD margin
as evidence of statistical significance; equating additional speckle with anatomy.

## Efficiency: measured values and limits

Final Fixed FPS: Liver208.55991882404066,Spine201.31913657661812,
Arm262.7712231425593,Breast307.42536520694114. Synthetic/phantom204.9395277003294;
real285.0982941747502. Dataset FPS is the arithmetic mean over the same two frozen
cases per dataset, NOT all6/9/33/30 quality cases. Panels equally average their two
dataset means. Retained per-run rows verify5warmup,20measured held-out frames,
3passes(60timed frames),final30000 checkpoints. RTX4090 and pure-forward CUDA
timing are documented in the frozen final claim audit; raw FPS rows label device
only as cuda, so GPU identity is inherited provenance, not independently probed.
Setup,I/O,metrics,backward and training are excluded. Fixed FPS models were rebuilt
under matching configs because original checkpoints were deleted; quality means
still come from the original78runs, not these replicas.

Training time(s): Liver517.4052810668945,Spine509.2855300108592,
Arm459.9817534793507,Breast378.76087275346123; panels513.3454055388769 and
419.371313116406. Boundary is the outer training process, including setup/eval/
checkpoint writing, not just optimization. Historical wrappers have small timing-
boundary differences; new/old runs involve shared-worker concurrency and mixed
4080SUPER/4090 hardware. No controlled cross-method speedup claim is justified.

Baseline per-dataset time is reproduced where present, but GaussianPile/Breast
is missing. It is not imputed as0 or replaced by a partial mean. Frozen baseline
panel times are reused verbatim, NOT recalculated from the shown dataset-time
columns; e.g.their historical aggregation differs. This means the baseline panel
time rows should be labeled historical reported efficiency, not represented as
uniform fresh measurements. Dataset-level baseline FPS is absent from this
allowed package; only panel FPS is available. NA is an explicit limitation, not
a reason to rerun or regenerate results.

**Safe:** measured Fixed pure-forward throughput exceeds a conventional30FPS
reference on these benchmark cases; it is compatible with fast rendering.
**Unsafe:** fastest method, end-to-end real-time clinical pipeline, controlled
training acceleration, speedup caused by covariance/local design. ULGSplat's
synthetic panel FPS is lower than every baseline's frozen panel FPS; on the real
panel it exceeds GaussianPile but not the other baselines. Present quality and
throughput jointly without inventing an efficiency win.

## Main qualitative evidence only

| Dataset | SUPPORTED | WEAK / UNSUPPORTED — exclude strong claims |
|---|---|---|
| Liver | Major shadow/echo pattern and superficial bright region retained | Fine texture equivalence; every speckle recovered; universally best local detail |
| Spine | Bright layered/curved echo bands retained | Unique superiority over strongest baseline; perfect thin-structure preservation |
| Arm | Layer separation and coarse echo arrangement retained | Fine boundary ranking between strong methods; artifact-free; best in every region |
| Breast | Broad pattern and localized echo texture retained | Fine-band equivalence; more speckle means more anatomical detail |

These are inherited illustrative cases, not newly selected or representative
samples proving dataset-wide visual ranking. No figures were regenerated.

## Remaining limitations / inconsistencies to disclose

1. Canonical manuscript marks/order cannot be independently inspected within the
   allowed package. New body follows the explicit method order supplied by the user.
2. Dataset-level baseline FPS is unavailable; GaussianPile Breast time is missing.
   Preserve NA and the separate frozen panel efficiency values, never disaggregate.
3. Historical baseline panel time aggregation is not consistently the arithmetic
   mean of the dataset times displayed here; hardware/concurrency also differ.
4. P0.5 header in archived protocol is historical; final actual Ours is Fixed P2-
   equivalent. Main is MP for all methods, not the PPP configuration.
5. NeRF-US/3DGS short labels need variant qualifiers (no prior / slice-adapted).

No quality-rank discrepancy was found. None of these issues authorizes training,
default changes, result regeneration, or automatic manuscript edits.
'''
    # Keep prose examples anchored to exact computed data, not hand-entered rounded copies.
    text=text.replace('.886903376 vs .889401169','.886903378 vs .889401169')
    (OUT/'final_main_comparison_analysis.md').write_text(text)
    (OUT/'README.md').write_text('''# Main Comparison rewrite evidence

Read final_main_comparison_analysis.md first. All files are extracted/derived
from the frozen parent package and current production config only. No new
training/evaluation or canonical manuscript editing.

- final_main_table_raw.tsv:32rows, all raw quality values, dataset/panel efficiency distinguished.
- final_main_table_markdown.md:unrounded quality display with correct raw-rank marks.
- final_main_table_latex.tex:new table-body fragment (4decimal quality,2decimal efficiency).
- final_main_ranks.tsv:192cells, best/second checked against frozen metadata.
- final_main_deltas.tsv:168baseline/metric comparisons, no SSIM percentage claims.
- category_*.md:compact evidence per baseline family; final_main_consistency.tsv records exceptions.
- final_main_efficiency_panels.tsv/md:16frozen efficiency rows, not reconstructed baseline dataset FPS.
- final_main_case_sources.tsv:624case/method source references; archived Ours configs included in parent package.
- current_config_audit.json/source_audit.json:protocol,hash and current-config checks.

NA values remain missing; historical displayed FPS values retain their precision.
The LaTeX file is an evidence export, not a change to the paper source.
''')


if __name__=='__main__':main()
