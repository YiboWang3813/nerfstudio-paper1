"""Verify Git numeric/report snapshots against relocated original run files."""
import hashlib
import json
from pathlib import Path

W=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


if __name__=='__main__':
    root=W/'results/evidence/legacy';source=W/'results/legacy/outputs'
    redactions={r['public_copy'] for r in json.loads((W/'protocol/migration/public_redactions.json').read_text())}
    rows=[];failures=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root);original=source/rel
        expected=sha(original) if original.is_file() else None;actual=sha(p)
        redacted=str(p) in redactions
        rows.append({'path':str(rel),'source_sha256':expected,'git_snapshot_sha256':actual,'redacted_public_copy':redacted})
        if expected is None or (actual!=expected and not redacted):failures.append(str(rel))
    (W/'protocol/migration/numeric_snapshot_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'snapshot_files':len(rows),'documented_redactions':sum(r['redacted_public_copy'] for r in rows),'failures':failures,'passed':not failures}
    (W/'results/migration/2026-10-17/numeric_snapshot_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)
