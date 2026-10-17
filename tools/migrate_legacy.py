"""Relocate the legacy workspace without changing algorithm/source evidence."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/boywang25/ultrasound-scene-reconstruction/omniultra')
LEGACY = WORKSPACE/'nerfstudio/paper1_legacy'
OPS = WORKSPACE/'gsplat/ultrasound_ops'
STATE = WORKSPACE/'results/migration/2026-10-17'
DOC_MAP = {
    'datasets': '02_datasets', 'methods': '01_problem_and_methods',
    'operations': '03_engineering/legacy_operations',
    'experiments': '04_experiments/protocol_notes',
    'production': '04_experiments/production',
    'paper': '05_manuscript/legacy_writing', 'literature': '06_literature',
    'archive': '90_archive/legacy_archive',
}

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def command(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)

def alias(old, new):
    if old.is_symlink():
        assert old.resolve() == new.resolve(), (old, new)
    else:
        assert not old.exists(), old
        old.parent.mkdir(parents=True, exist_ok=True)
        old.symlink_to(new, target_is_directory=new.is_dir())

def relocate(old, new, keep_alias=True):
    if old.is_symlink():
        assert old.resolve() == new.resolve(), (old, new)
        return
    if not old.exists():
        assert new.exists(), (old, new)
        return
    assert not new.exists(), new
    before = old.stat()
    new.parent.mkdir(parents=True, exist_ok=True)
    assert before.st_dev == new.parent.stat().st_dev, 'Expected same-filesystem atomic move'
    os.rename(str(old), str(new))
    after = new.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    row = {'old': str(old), 'new': str(new), 'device': before.st_dev,
           'inode': before.st_ino, 'mtime_ns': before.st_mtime_ns,
           'identity_preserved': True, 'compatibility_alias': keep_alias}
    with (STATE/'moves.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    if keep_alias:
        alias(old, new)
    print('MOVED', old, '->', new, flush=True)

def import_code():
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE/'private').mkdir(exist_ok=True, mode=0o700)
    if not (STATE/'source_git.json').exists():
        get = lambda *args: subprocess.check_output(['git', *args], cwd=SOURCE).decode()
        save(STATE/'source_git.json', {'source': str(SOURCE), 'head': get('rev-parse', 'HEAD').strip(),
             'status_porcelain': get('status', '--porcelain=v1', '-uall').splitlines(),
             'tracked_files': list(filter(None, get('ls-files', '-z').split('\0')))})
        (STATE/'private/working_tree.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=SOURCE))
        shutil.copy2(SOURCE/'README.md', STATE/'private/original_README.md')
    excluded = ['/.git', '/docs', '/ops', '/outputs', '/logs', '/wandb', '/artifacts', '/workspace',
                '/.agents', '/.codex', '.pytest_cache', '.vscode', '__pycache__', '*.pyc', '*.egg-info', 'build']
    LEGACY.mkdir(parents=True, exist_ok=True)
    command('rsync', '-a', *[v for p in excluded for v in ('--exclude', p)], str(SOURCE)+'/', str(LEGACY)+'/')
    rows = []
    for p in sorted(LEGACY.rglob('*')):
        if p.is_file():
            original = SOURCE/p.relative_to(LEGACY)
            stat = original.stat()
            digest = sha(original)
            assert sha(p) == digest, p
            rows.append({'old_path': str(original), 'new_path': str(p), 'relative_path': str(p.relative_to(LEGACY)),
                         'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': digest})
    rows.sort(key=lambda r: (r['mtime_ns'], r['relative_path']))
    save(STATE/'code_import_before_sanitization.json', rows)
    # Preserve original locally, but do not publish apparent credentials in the fork.
    readme = LEGACY/'README.md'
    text = readme.read_text()
    marker = '以下为历史 **wandb** 备忘'
    if marker in text:
        text = text.split(marker)[0]+'历史凭据备忘已从公开迁移副本移除；原件保存在本地迁移私有快照。\n'
        readme.write_text(text)
    relocate(SOURCE.parent/'data', WORKSPACE/'data')
    relocate(SOURCE/'outputs', WORKSPACE/'results/legacy/outputs')
    relocate(SOURCE/'ops', OPS)
    for name in ('logs', 'wandb', 'artifacts', 'workspace'):
        if (SOURCE/name).exists():
            relocate(SOURCE/name, WORKSPACE/'results/legacy'/name)
    for name, target in {'ops': OPS, 'outputs': WORKSPACE/'results/legacy/outputs',
                         'logs': WORKSPACE/'results/legacy/logs', 'wandb': WORKSPACE/'results/legacy/wandb',
                         'artifacts': WORKSPACE/'results/legacy/artifacts', 'workspace': WORKSPACE/'results/legacy/workspace'}.items():
        if target.exists():
            alias(LEGACY/name, target)

def import_docs():
    old = SOURCE/'docs'
    staging = STATE/'private/docs_staging'
    if not old.is_symlink() and old.exists():
        rows = []
        for p in old.rglob('*'):
            if p.is_file():
                stat = p.stat()
                rows.append({'old_relative_path': str(p.relative_to(old)), 'mtime_ns': stat.st_mtime_ns,
                             'size_bytes': stat.st_size, 'sha256': sha(p)})
        rows.sort(key=lambda r: (r['mtime_ns'], r['old_relative_path']))
        save(STATE/'docs_source_inventory.json', rows)
        relocate(old, staging, keep_alias=False)
    compatibility = WORKSPACE/'compat/docs'
    compatibility.mkdir(parents=True, exist_ok=True)
    mapping = []
    inventory = json.loads((STATE/'docs_source_inventory.json').read_text())
    for name, new in DOC_MAP.items():
        if (staging/name).exists():
            relocate(staging/name, WORKSPACE/'docs'/new, keep_alias=False)
        if (WORKSPACE/'docs'/new).exists():
            alias(compatibility/name, WORKSPACE/'docs'/new)
    for p in list(staging.iterdir()) if staging.exists() else []:
        if p.name == 'README.md':
            dest = WORKSPACE/'docs/90_archive/legacy_docs_index.md'
        else:
            dest = WORKSPACE/'docs/90_archive/root_files'/p.name
        relocate(p, dest, keep_alias=False)
        alias(compatibility/p.name, dest)
    alias(old, compatibility)
    alias(LEGACY/'docs', compatibility)
    for row in inventory:
        rel = Path(row['old_relative_path'])
        if rel.parts[0] in DOC_MAP:
            new = Path(DOC_MAP[rel.parts[0]])/Path(*rel.parts[1:])
        elif rel.name == 'README.md':
            new = Path('90_archive/legacy_docs_index.md')
        else:
            new = Path('90_archive/root_files')/rel
        dest = WORKSPACE/'docs'/new
        assert dest.is_file() and sha(dest) == row['sha256'], dest
        mapping.append(dict(row, new_relative_path=str(new)))
    save(WORKSPACE/'docs/00_index/legacy_path_map.json', mapping)
    save(WORKSPACE/'protocol/migration/docs_chronology.json', mapping)
    print('DOCS_VERIFIED', len(mapping), flush=True)

def externalize_weights():
    root = WORKSPACE/'results/legacy/outputs'
    summary = {'checkpoint_directories': [], 'standalone_weights': [], 'externalized_builds': []}
    for base, dirs, files in os.walk(str(root), followlinks=False):
        if 'checkpoints' in dirs:
            old = Path(base)/'checkpoints'
            new = WORKSPACE/'checkpoints/legacy'/old.relative_to(root)
            if old.is_symlink():
                summary['checkpoint_directories'].append({'run': str(Path(base).relative_to(root)), 'path': str(old.resolve()), 'preexisting_external': True})
            else:
                relocate(old, new)
                summary['checkpoint_directories'].append({'run': str(Path(base).relative_to(root)), 'path': str(new), 'preexisting_external': False})
            dirs.remove('checkpoints')
        for name in files:
            if Path(name).suffix.lower() in ('.pth', '.pt', '.ckpt'):
                old = Path(base)/name
                if old.is_symlink():
                    continue
                new = WORKSPACE/'checkpoints/legacy/standalone'/old.relative_to(root)
                relocate(old, new)
                summary['standalone_weights'].append({'old': str(old), 'new': str(new)})
    # Installed Python 3.7 extensions are historical artifacts, not source files.
    for base, dirs, files in os.walk(str(OPS), followlinks=False):
        for name in list(dirs):
            if name == 'build' or name.endswith('.egg-info'):
                old = Path(base)/name
                new = WORKSPACE/'cache/legacy_builds'/old.relative_to(OPS)
                if not old.is_symlink():
                    relocate(old, new)
                summary['externalized_builds'].append(str(new))
                dirs.remove(name)
        for name in files:
            if name.endswith('.so'):
                old = Path(base)/name
                new = WORKSPACE/'cache/legacy_builds'/old.relative_to(OPS)
                if not old.is_symlink():
                    relocate(old, new)
                summary['externalized_builds'].append(str(new))
    save(STATE/'checkpoint_relocation.json', summary)
    print('WEIGHT_AND_BUILD_RELOCATION_DONE', len(summary['checkpoint_directories']), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['code_data', 'docs', 'weights'])
    args = parser.parse_args()
    {'code_data': import_code, 'docs': import_docs, 'weights': externalize_weights}[args.stage]()
