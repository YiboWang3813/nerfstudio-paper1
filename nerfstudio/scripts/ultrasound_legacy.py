"""Run preserved ultrasound experiments from the Paper1 workspace.

This compatibility command deliberately keeps the legacy training loop. Native
Nerfstudio Model/Pipeline adaptation is a separate, equivalence-tested stage.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main():
    fork = Path(__file__).resolve().parents[2]
    workspace = Path(os.environ.get('PAPER1_WORKSPACE', str(fork.parent))).resolve()
    legacy = fork/'paper1_legacy'
    if not (legacy/'run.py').is_file():
        raise RuntimeError('Missing preserved Paper1 legacy source: '+str(legacy))
    args = list(sys.argv[1:])
    keys = {arg.lstrip('+').split('=', 1)[0] for arg in args if '=' in arg}
    defaults = {
        'scene.loading.baseroot': workspace/'data',
        'system.output_root': workspace/'results/runs',
        'system.checkpoint_root': workspace/'checkpoints/runs',
    }
    for key, value in defaults.items():
        if key not in keys:
            args.append('{}={}'.format(key, value))
    # Keep arbitrary caller-provided Hydra overrides; do not rewrite old aliases.
    sys.path.insert(0, str(legacy))
    sys.argv = [str(legacy/'run.py')] + args
    os.chdir(str(legacy))
    runpy.run_path(str(legacy/'run.py'), run_name='__main__')


if __name__ == '__main__':
    main()
