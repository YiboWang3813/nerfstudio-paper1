"""Archive the scheduled, non-training report/FPS pipeline and its job dependencies."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/provenance'
names=['p1_checkpoint_report.py','p1_scientific_checkpoint.py','p1_post_batch1.sh',
       'p1_fps_gpu3.sh','p1_fps_report.py','p1_postprocess_archive.py']
paths=[ROOT/'scripts/production/ablations/ours'/n for n in names]
paths += [ROOT/'docs/production/p1_paper_protocol_freeze/execution_status.md']
archive=OUT/'p1_postprocess_v1.tar.gz'
assert not archive.exists()
with tarfile.open(str(archive),'w:gz') as stream:
    for p in paths:stream.add(str(p),arcname=str(p.relative_to(ROOT)),recursive=False)
record=dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            file_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            jobs=dict(training=79940,aggregate_after_success=79942,fps_after_aggregate_success=79943),
            production_source_sha256='b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2',
            batch2_submitted=False)
(OUT/'p1_postprocess_v1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
