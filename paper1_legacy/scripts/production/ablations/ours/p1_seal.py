"""Seal the protocol, submission scripts and data inputs separately from P0.5 source."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'docs/production/p1_paper_protocol_freeze'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    summary=json.loads((OUT/'protocol_summary.json').read_text())
    records=[]
    for ds,sweeps in summary['local_processed_ids'].items():
        for sw in sweeps:
            path=ROOT.parent/'data'/ds/sw
            hashes={p.name:sha(p) for p in path.iterdir() if p.name in ('images.npy','poses.npy','relative_poses.npy','metadata.json')}
            records.append(dict(dataset=ds,sweep=sw,path=str(path),hashes=hashes))
    (OUT/'data_hashes.json').write_text(json.dumps(records,indent=2)+'\n')
    files=list(OUT.glob('*.json'))+list(OUT.glob('*.md'))+list(OUT.glob('*.tsv'))+list((OUT/'exact_configs').glob('*.yaml'))
    files=[p for p in files if p.name!='submission_hashes.json']
    files+=list((ROOT/'scripts/production/ablations/ours').glob('p1*.py'))
    files+=[ROOT/'scripts/production/ablations/ours/p1_batch1_gpu.sh',
            ROOT/'scripts/production/ablations/ours/p0_reuse.py',ROOT/'scripts/production/ablations/ours/p0_plan.py',
            ROOT/'scripts/production/comparisons/us_recon_four_dataset_p100k_v1/benchmark_render_fps.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    (OUT/'submission_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (OUT/'git_status_at_submission.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=str(ROOT)))
    (OUT/'base_commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)))
    print('PROTOCOL_SUBMISSION_SHA256',sha(OUT/'submission_hashes.json'),flush=True)


if __name__=='__main__':
    main()
