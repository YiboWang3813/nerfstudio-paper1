"""Verify every formerly tracked and untracked source file has a tracked home."""
import ast
import json
from pathlib import Path
import subprocess

W=Path(__file__).resolve().parents[2]


def main():
    source=json.loads((W/'protocol/migration/source_git.json').read_text())
    docs={r['old_relative_path']:r['new_relative_path'] for r in json.loads((W/'docs/00_index/legacy_path_map.json').read_text())}
    sets={name:set(filter(None,subprocess.check_output(['git','ls-files','-z'],cwd=str(W/name)).decode().split('\0'))) for name in ['nerfstudio','gsplat','docs','results']}
    def locate(path):
        if path.startswith('docs/'):return 'docs',docs.get(path[5:],'MISSING')
        if path.startswith('ops/'):return 'gsplat','ultrasound_ops/'+path[4:]
        if path.startswith('outputs/'):return 'results','evidence/legacy/'+path[8:]
        return 'nerfstudio','paper1_legacy/'+path
    def missing(paths):
        result=[]
        for path in paths:
            repo,new=locate(path)
            if new not in sets[repo]:result.append(path)
        return result
    untracked=[]
    for line in source['status_porcelain']:
        if line.startswith('?? '):
            path=line[3:]
            if path.startswith('"'):path=ast.literal_eval(path).encode('latin1').decode('utf-8')
            untracked.append(path)
    report={'old_tracked_files':len(source['tracked_files']),'old_untracked_files':len(untracked),
            'missing_tracked':missing(source['tracked_files']),'missing_untracked':missing(untracked)}
    report['passed']=not report['missing_tracked'] and not report['missing_untracked']
    (W/'protocol/migration/git_import_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
