"""Import selected files in source-mtime order, without forging commit dates."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess


def commit_groups(repo, records, label):
    groups = {}
    for path, stamp in records:
        day = dt.datetime.fromtimestamp(stamp/1e9, tz=dt.timezone.utc).date().isoformat()
        groups.setdefault(day, []).append(path)
    committed = []
    for day, files in sorted(groups.items()):
        # Explicit pathspecs, including PDFs ignored by upstream defaults.
        for i in range(0, len(files), 100):
            subprocess.run(['git', 'add', '-f', '--']+sorted(files[i:i+100]), cwd=repo, check=True)
        changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=repo).returncode
        if changed:
            subprocess.run(['git', 'commit', '-m', '{}; source mtime {}'.format(label, day)], cwd=repo, check=True)
            sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip()
            committed.append({'source_mtime_day_utc': day, 'commit': sha, 'files': sorted(files)})
    return committed


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['code', 'docs', 'ops', 'results'], required=True)
    p.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    w = args.workspace
    if args.kind == 'code':
        repo = w/'nerfstudio'
        entries = json.loads((w/'results/migration/2026-10-17/code_import_before_sanitization.json').read_text())
        records = [('paper1_legacy/'+r['relative_path'], r['mtime_ns']) for r in entries]
    elif args.kind == 'docs':
        repo = w/'docs'
        entries = json.loads((repo/'00_index/legacy_path_map.json').read_text())
        records = [(r['new_relative_path'], r['mtime_ns']) for r in entries]
    elif args.kind == 'results':
        repo = w/'results'
        records = [(str(p.relative_to(repo)), p.stat().st_mtime_ns) for p in (repo/'evidence').rglob('*') if p.is_file()]
    else:
        repo = w/'gsplat'
        records = []
        for base, dirs, files in __import__('os').walk(str(repo/'ultrasound_ops'), followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('build', '__pycache__', '.git') and not d.endswith('.egg-info')]
            for name in files:
                path = Path(base)/name
                if path.is_symlink() or path.suffix in ('.so', '.pyc', '.o', '.a'):
                    continue
                records.append((str(path.relative_to(repo)), path.stat().st_mtime_ns))
    history = commit_groups(repo, records, 'Import preserved legacy '+args.kind)
    dest = w/'protocol/migration'/('import_commits_'+args.kind+'.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(history, indent=2)+'\n')
