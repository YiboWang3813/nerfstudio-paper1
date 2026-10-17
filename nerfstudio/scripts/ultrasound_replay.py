"""Replay a frozen composed config without inheriting today's alias defaults."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from nerfstudio.scripts.ultrasound_legacy import configure_workspace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path,help='Historical resolved_config.yaml')
    p.add_argument('--output',type=Path,required=True,help='New, empty run directory')
    p.add_argument('--task',choices=['train','test','dry_run'],default='dry_run')
    p.add_argument('--checkpoint-dir',type=Path,help='Explicit source checkpoint directory for test/resume')
    p.add_argument('--resume',type=int,help='Explicit checkpoint iteration for test/resume')
    args=p.parse_args()
    source=args.config.expanduser().resolve();output=args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Output must be absent or empty; refusing to overwrite existing evidence.')
    if args.task=='test' and (args.checkpoint_dir is None or args.resume is None):
        p.error('Test replay requires --checkpoint-dir and --resume; never guess a historical checkpoint.')
    if (args.checkpoint_dir is None)!=(args.resume is None):
        p.error('--checkpoint-dir and --resume must be supplied together.')
    if args.resume is not None and args.resume<=0:
        p.error('--resume must be a positive checkpoint iteration.')
    fork=Path(__file__).resolve().parents[2]
    workspace=Path(os.environ.get('PAPER1_WORKSPACE',str(fork.parent))).resolve()
    legacy=configure_workspace(fork,workspace)
    from omegaconf import OmegaConf
    from configs.setup import setup_experiment
    from utils.experiment_management import short_run_id, save_experiment_snapshot, update_run_manifest
    raw=OmegaConf.load(str(source))
    if args.task=='train' and str(raw.engine.resume or '') not in ('','0') and args.resume is None:
        p.error('This historical config resumed training; supply its source checkpoint directory and iteration explicitly.')
    original_id=str(raw.system.run_id)
    raw.task=args.task
    raw.engine.task=args.task
    raw.system.run_id=short_run_id()
    raw.system.run_name='replay_{}_{}'.format(original_id,raw.system.run_id)
    raw.system.output_root=str(workspace/'results/replays')
    raw.system.output_dir=str(output)
    raw.system.hydra_output_dir=str(output)
    raw.system.checkpoint_root=str(workspace/'checkpoints/replays')
    # Loading historical weights and writing new weights have separate locations.
    raw.system.checkpoint_dir=''
    raw.scene.loading.baseroot=str(workspace/'data')
    if args.resume is not None:raw.engine.resume=str(args.resume)
    os.chdir(str(legacy))
    cfg=setup_experiment(raw,['frozen_config='+str(source),'replay_task='+args.task],hydra_output_dir=str(output))
    cfg.system.config_path='frozen_config:'+str(source)
    save_experiment_snapshot(cfg,str(output),cfg.system.config_path,['frozen_config='+str(source),'replay_task='+args.task])
    update_run_manifest(str(output),{'config_path':cfg.system.config_path,'replayed_config_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    source_manifest=source.parent/'run_manifest.yaml'
    provenance={'frozen_config':str(source),'frozen_config_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'original_run_id':original_id,'task':args.task,
                'source_checkpoint_dir':str(args.checkpoint_dir.expanduser().resolve()) if args.checkpoint_dir else None,
                'source_checkpoint_iteration':args.resume,
                'source_manifest':str(source_manifest) if source_manifest.is_file() else None,
                'source_manifest_sha256':hashlib.sha256(source_manifest.read_bytes()).hexdigest() if source_manifest.is_file() else None,
                'source_execution_identity':'Read original source manifest; current execution identity is in new run_manifest.yaml.',
                'semantic_policy':'Use the entire resolved config; only task, machine paths and new run identity are overridden.'}
    (output/'replay_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if args.task=='dry_run':
        print('Frozen configuration replay validated:',output)
        return
    from run import _register_methods
    from engine.engine import Engine
    _register_methods(cfg.model.name)
    engine=Engine(cfg)
    if args.checkpoint_dir is not None:
        source_weights=args.checkpoint_dir.expanduser().resolve()
        io=engine.state.io_manager
        original_load=io.load_checkpoint
        def load_historical(items,step):
            expected=[source_weights/('iteration_{}_{}.pth'.format(step,name)) for name,_ in items]
            missing=[str(path) for path in expected if not path.is_file()]
            if missing:raise FileNotFoundError('Incomplete historical checkpoint: '+repr(missing))
            destination=io.checkpoint_dir
            try:
                io.checkpoint_dir=str(source_weights)
                return original_load(items,step)
            finally:
                io.checkpoint_dir=destination
        io.load_checkpoint=load_historical
    engine.run()


if __name__=='__main__':main()
