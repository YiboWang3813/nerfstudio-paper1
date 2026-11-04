"""Render the same saved checkpoint through either source tree, without training."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    sys.path.insert(0,str(root))
    from omegaconf import OmegaConf
    from configs.setup import prepare_config
    from engine.engine import Engine
    from run import _register_methods
    from methods.ours.render import render
    w=Path('/home/boywang25/paper1');base=w/'results/migration/2026-10-17/gpu'
    data={}
    for scene in ['liver','spine_phantom','tdsc_abus_2023','tus_rec_2024_qc']:
        run=base/('replay_old_'+scene)
        cfg=prepare_config(OmegaConf.load(str(run/'resolved_config.yaml')))
        cfg.system.output_dir=str(args.output.parent/('eval_logs_'+args.output.stem+'_'+scene))
        cfg.system.checkpoint_dir=str(run/'checkpoints')
        cfg.engine.resume='12'
        cfg.system.seed=42
        _register_methods(cfg.model.name)
        engine=Engine(cfg)
        engine.trigger('on_setup')
        state=engine.state
        frames=[]
        with torch.no_grad():
            for index in state.scene.test_indices:
                meta=state.scene.test_container.get(index)
                pred,depth,_=render(state.model_cfg,state.scene.dataset_meta,meta,state.gauss_container,is_jitter=False,return_intermediates=False,collect_path_stats=False)
                frames.append({'frame_id':int(index),'prediction':pred.cpu(),'depth':depth.cpu()})
        data[scene]=frames
        engine.trigger('on_finish')
        del engine,state
        torch.cuda.empty_cache()
    torch.save(data,str(args.output))
    print('FROZEN_CHECKPOINT_RENDER_COMPLETE',root,flush=True)


if __name__=='__main__':main()
