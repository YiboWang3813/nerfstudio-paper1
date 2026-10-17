"""Retain eight models, compare replica metrics, then aggregate same-case FPS."""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path
from scripts.production.ablations.ours.p2_fps_rebuild import ROOT,DOC,OUT,MATRIX,base
from scripts.production.ablations.ours.p1_lgr_report import write
from scripts.production.ablations.ours.summarize_ablation_run import read_series,value_at


def main():
    base.guard();matrix=json.loads((DOC/MATRIX).read_text());report=OUT/'reports';report.mkdir(exist_ok=True,parents=True)
    if sys.argv[1]=='prepare':
        metrics=[];inputs=[]
        for r in matrix:
            run=Path(r['output_dir']);s=json.loads((run/'batch1_stats.json').read_text());c=json.loads((OUT/'control'/(r['batch1_id']+'.json')).read_text())
            assert c['exit_code']==0 and s['success'] and s['finite_final_parameters'] and s['source_freeze_sha256']==base.SHA and s['semantic_config_sha256']==r['semantic_config_sha256']
            assert (run/'checkpoints/iteration_30000_container.pth').is_file()
            new=read_series(run/'logs/scalars.tsv');old=read_series(Path(r['quality_source_dir'])/'logs/scalars.tsv')
            for key in ['test/'+m for m in ('psnr','l1','ssim','lpips','dists','gmsd')]+['train/num_gaussians']:
                a=value_at(old,key,30000);b=value_at(new,key,30000);assert a is not None and b is not None and math.isfinite(b)
                metrics.append(dict(dataset=r['dataset'],sweep=r['sweep'],metric=key,original=a,replica=b,delta=b-a))
            inputs.append(dict(dataset=r['dataset'],sweep=r['sweep'],method='ours',final_step=30000,final_num_gaussians=s['final_num_gaussians'],output_dir=str(run)))
        write(report/'replica_vs_original.tsv',metrics);write(report/'fps_input.tsv',inputs)
        print('8/8 models verified; replicas do NOT replace original quality metrics. MAX DELTAS', {m:max(abs(r['delta']) for r in metrics if r['metric']==m) for m in {r['metric'] for r in metrics}})
    elif sys.argv[1]=='aggregate':
        raw=list(csv.DictReader((report/'fixed_fps_per_run.tsv').open(),delimiter='\t'));assert len(raw)==8 and all(r['status']=='success' for r in raw)
        assert {(r['dataset'],r['sweep']) for r in raw}=={(r['dataset'],r['sweep']) for r in matrix}
        rows=[]
        for ds in sorted({r['dataset'] for r in raw}):
            rr=[r for r in raw if r['dataset']==ds];assert len(rr)==2
            rows.append(dict(dataset=ds,n=2,fps_mean=st.mean(float(r['render_fps']) for r in rr),selected_ids=';'.join(sorted(r['sweep'] for r in rr))))
        write(report/'fixed_fps_datasets.tsv',rows);lookup={r['dataset']:r['fps_mean'] for r in rows}
        panels=[dict(panel='synthetic_phantom',fps_mean=st.mean(lookup[d] for d in ('synthetic_liver','spine_phantom'))),dict(panel='real',fps_mean=st.mean(lookup[d] for d in ('tus_rec_2024_qc','tdsc_abus_2023')))]
        write(report/'fixed_fps_panels.tsv',panels)
        print('Fixed FPS COMPLETE',rows,panels,'MODELS RETAINED')


if __name__=='__main__':main()
