"""Check relocated files, source identities, docs and figure/run references."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(workspace):
    state = workspace/'results/migration/2026-10-17'
    failures = []
    code = json.loads((state/'code_import_before_sanitization.json').read_text())
    for row in code:
        path = Path(row['new_path'])
        if row['relative_path'] == 'README.md':
            continue  # The sole source-content change is recorded credential redaction.
        if not path.is_file() or sha(path) != row['sha256']:
            failures.append('code:'+str(path))
    docs = json.loads((workspace/'docs/00_index/legacy_path_map.json').read_text())
    for row in docs:
        path = workspace/'docs'/row['new_relative_path']
        if not path.is_file() or sha(path) != row['sha256']:
            failures.append('docs:'+str(path))
    moves = [json.loads(line) for line in (state/'moves.jsonl').read_text().splitlines()]
    for row in moves:
        path = Path(row['new'])
        if not path.exists():
            failures.append('missing_move:'+str(path))
            continue
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) != (row['device'], row['inode']):
            failures.append('inode:'+str(path))
        if row['compatibility_alias'] and Path(row['old']).resolve() != path.resolve():
            failures.append('alias:'+row['old'])
    package = workspace/'docs/04_experiments/production/final_paper_evidence_20260910'
    manifest = json.loads((package/'audits/final_data_source_manifest.json').read_text())
    for rel, digest in manifest['package_file_sha256'].items():
        path = package/rel
        if not path.is_file() or sha(path) != digest:
            failures.append('evidence:'+str(path))
    figure = json.loads((workspace/'docs/04_experiments/production/comparison_final_outputs/manifests/paper_all_datasets_2d_rendering_manifest.json').read_text())
    refs = [p['source_image'] for p in figure['panels'] if 'source_image' in p]
    for ref in refs:
        if not Path(ref).is_file():
            failures.append('figure_source:'+ref)
    report = {'code_files_checked': len(code)-1, 'docs_files_checked': len(docs), 'atomic_moves_checked': len(moves),
              'evidence_hashes_checked': len(manifest['package_file_sha256']), 'figure_images_checked': len(refs),
              'failures': failures, 'passed': not failures,
              'scope': 'File identities, source content, docs, selected evidence and figure references; GPU/config replay reported separately.'}
    (state/'integrity_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    verify(p.parse_args().workspace.resolve())
