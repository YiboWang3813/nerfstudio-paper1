"""Archive exact submission artifacts; production archive remains P0.5 unchanged."""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PROTOCOL=ROOT/'docs/production/p1_paper_protocol_freeze'
OUT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/provenance'
OUT.mkdir(parents=True,exist_ok=True)
version=sys.argv[1] if len(sys.argv)>1 else 'v1'
assert version in ('v1','v2')
stem='p1_gpu2_submission' if version=='v1' else 'p1_gpu2_submission_v2'
target=OUT/(stem+'.tar.gz')
assert not target.exists(),target
hashes=json.loads((PROTOCOL/'submission_hashes.json').read_text())
for name,expected in hashes.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
extra=['docs/production/p1_paper_protocol_freeze/gpu2_migration.md',
       'docs/production/p1_paper_protocol_freeze/submission_hashes.json',
       'scripts/production/ablations/ours/p1_gpu2_batch.sh',
       'scripts/production/ablations/ours/p1_gpu2_smoke.py',
       'scripts/production/ablations/ours/p1_gpu2_smoke.sh',
       'scripts/production/ablations/ours/p1_migration_archive.py']
for name in extra:
    hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
with tarfile.open(str(target),'w:gz') as archive:
    for name in sorted(hashes):
        archive.add(str(ROOT/name),arcname=name,recursive=False)
    content=json.dumps(hashes,indent=2).encode()
    item=tarfile.TarInfo('migration_hashes.json');item.size=len(content)
    archive.addfile(item,io.BytesIO(content))
(OUT/('migration_hashes.json' if version=='v1' else 'migration_hashes_v2.json')).write_text(json.dumps(hashes,indent=2)+'\n')
(OUT/(stem+'.sha256')).write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'\n')
print('SUBMISSION_ARCHIVE',target)
print('SHA256',(OUT/(stem+'.sha256')).read_text())
