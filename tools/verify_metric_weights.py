"""Verify all six metrics using only provisioned pretrained weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();sys.path.insert(0,str(args.root.resolve()))
from configs.setup import load_hydra_config
from engine.computer import MetricComputer

def refuse_download(*args,**kwargs):
    raise RuntimeError('Offline metric check attempted a download; provision the missing weight explicitly.')

torch.hub.download_url_to_file=refuse_download
cfg=load_hydra_config('production/ablations/ours/paper_semantic_freeze')
cfg.metric.needed_metrics=['psnr','l1','ssim','lpips','dists','gmsd']
computer=MetricComputer(cfg,torch.device('cuda'))
torch.manual_seed(42)
pred=torch.rand(64,64,device='cuda');gt=torch.rand(64,64,device='cuda')
with torch.no_grad():metrics=computer.compute({'gt_image':gt},{'pred_image':pred})
result={key:float(value) for key,value in metrics.items()}
assert all(__import__('math').isfinite(v) for v in result.values())
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(result)
