"""Output-only retention policy. Never changes training or touches active checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PROTOCOL=ROOT/'docs/production/p1_paper_protocol_freeze'
MAIN=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908'
ABLATION=ROOT/'outputs/production/ablations/ours/paper_final_corrected_covariance_20260908'
METRICS=('psnr','l1','ssim','lpips','dists','gmsd')


def read_tsv(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def validated_targets(row, control, expected_indices, allowed_roots, fps_complete=False):
    """Return exact final filenames only, after checking finished exported evidence."""
    if row.get('target_status')=='FINAL-REUSE' or not control or control.get('exit_code')!=0:
        return []
    run=Path(row['output_dir'])
    resolved=run.resolve()
    if run.is_symlink() or not any(str(resolved).startswith(str(p.resolve())+'/') for p in allowed_roots):
        raise ValueError('Out-of-scope run: '+str(run))
    stats=json.loads((run/'batch1_stats.json').read_text())
    if not stats.get('success') or not stats.get('finite_final_parameters'):
        return []
    for name in ('resolved_config.yaml','run_manifest.yaml','logs/scalars.tsv'):
        if not (run/name).is_file() or not (run/name).stat().st_size:return []
    values={r['key']:float(r['value']) for r in read_tsv(run/'logs/scalars.tsv') if int(r['step'])==30000}
    if not all(math.isfinite(values.get('test/'+m,float('nan'))) for m in METRICS):return []
    n=values.get('train/num_gaussians',0)
    if not math.isfinite(n) or n<=0 or int(n)!=int(stats['final_num_gaussians']):return []
    frames=read_tsv(run/'eval/per_frame_metrics_iteration_030000.tsv')
    if [int(r['index']) for r in frames]!=expected_indices:return []
    if not all(math.isfinite(float(r[m])) for r in frames for m in METRICS):return []
    # All exported representative frames must be intact before dropping re-render ability.
    images=list((run/'vis').glob('item_*'))
    if len(images)!=min(10,len(frames)):return []
    if not all((p/name).is_file() and (p/name).stat().st_size for p in images for name in ('pred.png','gt.png','diff.png','compose.png')):return []
    names=['iteration_30000_solver.pth']
    if row['group']!='main' or fps_complete:names.append('iteration_30000_container.pth')
    targets=[]
    for name in names:
        path=run/'checkpoints'/name
        if not path.exists():continue
        if path.is_symlink() or path.resolve().parent!=resolved/'checkpoints':
            raise ValueError('Unsafe checkpoint path: '+str(path))
        if path.is_file():targets.append(path)
    return targets


def sweep(apply=False):
    matrix=json.loads((PROTOCOL/'batch1_matrix.json').read_text())
    audit=json.loads((PROTOCOL/'baseline_case_frame_audit.json').read_text())
    expected={(r['dataset'],r['sweep']):r['expected_test_indices'] for r in audit}
    reports=MAIN/'reports'
    fps_complete=False
    if (reports/'FAIR_FPS_CHECKPOINT.md').exists() and (reports/'SCIENTIFIC_CHECKPOINT.md').exists():
        fps=read_tsv(reports/'corrected_main_fps.tsv')
        wanted={r['output_dir'] for r in matrix if r['group']=='main'}
        fps_complete=(len(fps)==78 and {r['output_dir'] for r in fps}==wanted
                      and all(r['status']=='success' and math.isfinite(float(r['render_fps'])) and float(r['render_fps'])>0 for r in fps))
    actions=[]
    for row in matrix:
        cp=MAIN/'control'/(row['batch1_id']+'.json')
        if not cp.exists():continue
        control=json.loads(cp.read_text())
        try:
            targets=validated_targets(row,control,expected[row['dataset'],row['sweep']],
                (MAIN/'runs',ABLATION/'runs'),fps_complete=fps_complete)
        except (OSError,ValueError,KeyError) as exc:
            print('RETAIN',row['batch1_id'],str(exc),flush=True);continue
        for path in targets:
            entry=dict(timestamp=time.time(),row_id=row['batch1_id'],path=str(path),bytes=path.stat().st_size,
                       reason='optimizer unnecessary after completed run' if 'solver' in path.name else 'all required inference/export outputs verified',
                       fps_complete=fps_complete,applied=apply)
            if apply:
                entry['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                ledger=MAIN/'provenance/checkpoint_cleanup.jsonl'
                ledger.parent.mkdir(parents=True,exist_ok=True)
                # Record intent before unlink and completion afterward for auditability.
                with ledger.open('a') as f:f.write(json.dumps(dict(entry,event='delete_intent'))+'\n')
                path.unlink()
                with ledger.open('a') as f:f.write(json.dumps(dict(entry,event='deleted'))+'\n')
            actions.append(entry)
    print(json.dumps(dict(apply=apply,files=len(actions),bytes=sum(r['bytes'] for r in actions),actions=actions)),flush=True)
    return len(actions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        sweep(args.apply)
        if not args.watch or (MAIN/'control/STOP').exists():break
        matrix=json.loads((PROTOCOL/'batch1_matrix.json').read_text())
        # Stop the watcher after training; the dependent final cleanup handles FPS leftovers.
        completed=list((MAIN/'control').glob('b1_*.json'))
        if len(completed)>=sum(r['execution_authorized'] for r in matrix):break
        time.sleep(60)


if __name__=='__main__':main()
