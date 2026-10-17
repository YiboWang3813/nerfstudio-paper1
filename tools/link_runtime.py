"""Restore ignored compatibility links after checking out the workspace forks."""
import os
from pathlib import Path

W = Path(__file__).resolve().parents[2]
LEGACY = W/'nerfstudio/paper1_legacy'
DOCS = {
    'datasets':'02_datasets', 'methods':'01_problem_and_methods',
    'operations':'03_engineering/legacy_operations', 'experiments':'04_experiments/protocol_notes',
    'production':'04_experiments/production', 'paper':'05_manuscript/legacy_writing',
    'literature':'06_literature', 'README.md':'90_archive/legacy_docs_index.md',
}


def link(path, target):
    if path.is_symlink():
        if path.resolve()!=target.resolve(): raise RuntimeError('Conflicting link: '+str(path))
        return
    if path.exists(): raise RuntimeError('Refusing to overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.symlink_to(os.path.relpath(str(target),str(path.parent)),target_is_directory=target.is_dir())


if __name__=='__main__':
    for old,new in DOCS.items():
        link(W/'compat/docs'/old,W/'docs'/new)
    for old,target in {
        'ops':W/'gsplat/ultrasound_ops','docs':W/'compat/docs',
        'outputs':W/'results/legacy/outputs','logs':W/'results/legacy/logs',
        'wandb':W/'results/legacy/wandb','artifacts':W/'results/legacy/artifacts',
        'workspace':W/'results/legacy/workspace',
    }.items(): link(LEGACY/old,target)
    print('Paper1 runtime links verified. External data/checkpoints and ABI-compatible binaries must be provisioned separately.')
