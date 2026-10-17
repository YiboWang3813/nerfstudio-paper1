"""Strict paired reporting for all 55 authorized LGR slots; no training changes."""
import csv
import hashlib
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/production/p1_lgr_confirmation'
OUT=ROOT/'outputs/production/ablations/ours/paper_final_corrected_covariance_20260908/lgr_confirmation'
METRICS=('psnr','ssim','lpips')
LABELS=('Plane','Fixed','Adaptive','Simple','Full')


def paired_statistics(delta,metric):
    oriented=[d if metric!='lpips' else -d for d in delta]
    return dict(n=len(delta),mean_delta=st.mean(delta),std_paired_delta=st.stdev(delta),
                wins=sum(d>1e-6 for d in oriented),ties=sum(abs(d)<=1e-6 for d in oriented),
                losses=sum(d< -1e-6 for d in oriented),tie_absolute_tolerance=1e-6,
                min_delta=min(delta),max_delta=max(delta))


def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)


def main():
    from scripts.production.ablations.ours.summarize_ablation_run import read_series,value_at
    matrix=json.loads((DOC/'lgr_multisweep_structural_matrix.json').read_text())
    audit=json.loads((ROOT/'docs/production/p1_paper_protocol_freeze/baseline_case_frame_audit.json').read_text())
    report=OUT/'reports';report.mkdir(parents=True,exist_ok=True);rows=[];pending=[]
    for r in matrix:
        run=Path(r['output_dir']);stats=run/'batch1_stats.json'
        if not stats.exists():stats=run/'anchor_stats.json'
        if not stats.exists() or not json.loads(stats.read_text()).get('success'):
            pending.append(r['batch1_id']);continue
        state=json.loads(stats.read_text())
        if r['execution_authorized']:
            c=json.loads((OUT/'control'/(r['batch1_id']+'.json')).read_text());assert c['exit_code']==0
            assert state['source_freeze_sha256']==r['source_freeze_sha256'] and state['semantic_config_sha256']==r['semantic_config_sha256']
            assert state['finite_final_parameters']
        series=read_series(run/'logs/scalars.tsv');row=dict(dataset=r['dataset'],sweep=r['sweep'],variant=r['variant'],row_id=r['batch1_id'],reused=not r['execution_authorized'],output_dir=str(run),gpu=state.get('gpu','see run manifest'),source_freeze_sha256=r['source_freeze_sha256'],semantic_config_sha256=r['semantic_config_sha256'])
        for m in METRICS:
            row[m]=value_at(series,'test/'+m,30000);assert row[m] is not None and math.isfinite(row[m])
        frames=list(csv.DictReader((run/'eval/per_frame_metrics_iteration_030000.tsv').open(),delimiter='\t'))
        expected=next(x['expected_test_indices'] for x in audit if x['dataset']==r['dataset'] and x['sweep']==r['sweep'])
        assert [int(x['index']) for x in frames]==expected
        row['final_G']=value_at(series,'train/num_gaussians',30000)
        rows.append(row)
    (report/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=55,pending=pending),indent=2)+'\n')
    if rows:write(report/'per_case.tsv',rows)
    assert not pending,'No final paired report from an incomplete matrix: '+str(pending)
    means=[];paired=[];transitions=[]
    for ds in sorted({r['dataset'] for r in rows}):
        subset=[r for r in rows if r['dataset']==ds];cases=sorted({r['sweep'] for r in subset});lookup={(r['sweep'],r['variant']):r for r in subset}
        for label in LABELS:
            rr=[r for r in subset if r['variant']==label];assert len(rr)==len(cases)
            for m in METRICS:means.append(dict(dataset=ds,variant=label,metric=m,n=len(rr),mean=st.mean(r[m] for r in rr),std=st.stdev(r[m] for r in rr)))
        for a,b in zip(LABELS[:-1],LABELS[1:]):
            for m in METRICS:
                delta=[lookup[sw,b][m]-lookup[sw,a][m] for sw in cases]
                for sw,d in zip(cases,delta):paired.append(dict(dataset=ds,sweep=sw,before=a,after=b,metric=m,delta=d))
                transitions.append(dict(dataset=ds,before=a,after=b,metric=m,**paired_statistics(delta,m)))
    write(report/'variant_means.tsv',means);write(report/'paired_deltas.tsv',paired);write(report/'transition_statistics.tsv',transitions)
    from scripts.production.ablations.ours.p1_scientific_checkpoint import table
    text=['# LGR 11-case structural confirmation','55 slots: 19 exact-config reuse + 36 newly trained; MP, seed42, 1M, 30k, same frozen split. No hyperparameter experiments.',
          'No-global variants use local-aware CGA fallback; all other training settings unchanged. Differences are whole-system trained contrasts, not isolated fixed-topology renderer effects.',
          'Means and sample standard deviations (ddof=1) are sweep-wise. W/T/L is per metric, direction-aware, with numerical tie tolerance 1e-6; wins are NOT significance or clinical/practical effect claims. Single seed; Arm segments are not independent subjects.',
          '## Variant means',table(means,['dataset','variant','metric','n','mean','std']),
          '## Paired transitions',table(transitions,['dataset','before','after','metric','mean_delta','std_paired_delta','wins','ties','losses','min_delta','max_delta']),
          'See per_case.tsv and paired_deltas.tsv for every individual case. No default retuning or Batch2 launch is authorized by these results.']
    (report/'LGR_CONFIRMATION.md').write_text('\n\n'.join(text))
    print('COMPLETE 55-slot LGR paired report')
    if '--cleanup' in sys.argv:
        from scripts.production.ablations.ours.p1_checkpoint_cleanup import validated_targets
        total=0;files=0
        for r in matrix:
            if not r['execution_authorized']:continue
            control=json.loads((OUT/'control'/(r['batch1_id']+'.json')).read_text())
            expected=next(x['expected_test_indices'] for x in audit if x['dataset']==r['dataset'] and x['sweep']==r['sweep'])
            for path in validated_targets(r,control,expected,(OUT/'runs',)):
                entry=dict(row_id=r['batch1_id'],path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),timestamp=time.time(),reason='LGR completed; metrics/frame IDs/all exports verified; no FPS needed')
                ledger=report/'checkpoint_cleanup.jsonl'
                with ledger.open('a') as f:f.write(json.dumps(dict(entry,event='delete_intent'))+'\n')
                path.unlink()
                with ledger.open('a') as f:f.write(json.dumps(dict(entry,event='deleted'))+'\n')
                total+=entry['bytes'];files+=1
        print('Removed validated new LGR checkpoints:',files,'files,',total,'bytes; no reused artifacts touched')


if __name__=='__main__':main()
