"""Remove only the six completed P1 GPU2 100-step smoke checkpoint pairs."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
SMOKE=ROOT/'outputs/preproduction/regression/p1_gpu2_memory_gate_20260908'
LEDGER=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/provenance/checkpoint_cleanup.jsonl'
IDS=('b1_001','b1_009','b1_038','b1_068','b1_096','b1_097')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    targets=[]
    for identity in IDS:
        run=SMOKE/identity
        assert not run.is_symlink() and run.resolve().parent==SMOKE.resolve()
        stats=json.loads((run/'smoke_stats.json').read_text())
        assert stats.get('success') and stats.get('finite_outputs') and stats.get('paper_result') is False
        assert [v for v in stats['overrides'] if v.startswith('engine.iterations=')][-1]=='engine.iterations=100'
        metrics={r['key']:float(r['value']) for r in csv.DictReader((run/'logs/scalars.tsv').open(),delimiter='\t') if int(r['step'])==100}
        assert all(math.isfinite(metrics.get('test/'+m,float('nan'))) for m in ('psnr','ssim','l1','lpips','dists','gmsd'))
        for name in ('resolved_config.yaml','run_manifest.yaml','eval/per_frame_metrics_iteration_000100.tsv'):
            assert (run/name).is_file() and (run/name).stat().st_size
        assert list((run/'vis').glob('item_*/pred.png'))
        for kind in ('container','solver'):
            p=run/'checkpoints'/('iteration_100_'+kind+'.pth')
            if not p.exists():continue
            assert p.is_file() and not p.is_symlink() and p.resolve().parent==run.resolve()/'checkpoints'
            targets.append(p)
    records=[]
    for p in targets:
        record=dict(timestamp=time.time(),path=str(p),bytes=p.stat().st_size,
                    reason='completed P1 100-step smoke; only exported validation evidence required',applied=args.apply)
        if args.apply:
            record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            with LEDGER.open('a') as f:f.write(json.dumps(dict(record,event='delete_intent'))+'\n')
            p.unlink()
            with LEDGER.open('a') as f:f.write(json.dumps(dict(record,event='deleted'))+'\n')
        records.append(record)
    print(json.dumps(dict(files=len(records),bytes=sum(r['bytes'] for r in records),applied=args.apply,records=records),indent=2))


if __name__=='__main__':main()
