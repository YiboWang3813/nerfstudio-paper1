"""Execute one explicitly authorized Batch 1 row using unchanged P0.5 production."""
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from ours_renderer import autograd
from engine.engine import Engine
from scripts.production.ablations.ours.p0_reuse import semantic

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT/'docs/production/p2_batch2_final'
FREEZE = ROOT/'outputs/preproduction/regression/P2_fixed_full_final_20260909/freeze'
SHA = 'f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27'
row_id = sys.argv[1]
matrix_path = PROTOCOL/'batch2_exact_matrix.json'
matrix = json.loads(matrix_path.read_text())
row = next(r for r in matrix if r['batch1_id']==row_id)
assert row['execution_authorized'] and row['source_freeze_sha256']==SHA
assert row['group'] in ('main','structural','multi_sweep','hyper_PPP')
target = Path(row['output_dir'])
assert not (target/'resolved_config.yaml').exists(), 'Never overwrite any attempted run'
for filename,expected in json.loads((FREEZE/'hashes.json').read_text()).items():
    if not filename.startswith(('scripts/','docs/','tests/')):
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected, filename
binary_hashes=json.loads((FREEZE/'binary_hashes.json').read_text())
for filename,expected in binary_hashes.items():
    assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected, filename
submission=json.loads((PROTOCOL/'submission_hashes.json').read_text())
for filename,expected in submission.items():
    assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected, filename
stats=dict(success=False,peak_local_tile_gaussian_pairs=0,peak_global_ray_gaussian_pairs=0,
           total_local_tile_gaussian_pairs=0,total_global_ray_gaussian_pairs=0,forward_calls=0,
           source_freeze_sha256=SHA,row_id=row_id,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
           matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),binary_hashes=binary_hashes)


def instrument(name, local_index, ray_index=None):
    original=getattr(autograd._C,name)
    def wrapped(*args):
        out=original(*args)
        size=out[local_index].numel()
        stats['peak_local_tile_gaussian_pairs']=max(stats['peak_local_tile_gaussian_pairs'],size)
        stats['total_local_tile_gaussian_pairs']+=size
        if ray_index is not None:
            size=out[ray_index].numel()
            stats['peak_global_ray_gaussian_pairs']=max(stats['peak_global_ray_gaussian_pairs'],size)
            stats['total_global_ray_gaussian_pairs']+=size
        stats['forward_calls']+=1
        if stats['forward_calls']%1000==0:
            assert torch.isfinite(out[0]).all().item(), 'Nonfinite render'
        return out
    setattr(autograd._C,name,wrapped)


instrument('ProjectedSliceForwardCUDA',2)
instrument('ProjectedSliceForwardRayTileCUDA',3,8)
original_run=Engine.run


def audited_run(engine):
    actual=OmegaConf.to_container(engine.cfg,resolve=True)
    normalized=semantic(actual);normalized['metric']=actual['metric']
    digest=hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()
    assert digest==row['semantic_config_sha256'], 'Resolved config differs from frozen matrix'
    stats['semantic_config_sha256']=digest
    manifest_path=target/'run_manifest.yaml'
    manifest=OmegaConf.load(manifest_path)
    manifest.paper_final_source_sha256=SHA
    manifest.paper_final_batch='P2_Final_Hyperparameter_Batch2'
    manifest.paper_final_row_id=row_id
    manifest.paper_final_matrix_sha256=stats['matrix_sha256']
    manifest.paper_final_submission_hashes=submission
    manifest.paper_final_binary_hashes=binary_hashes
    OmegaConf.save(manifest,manifest_path)
    original_run(engine)
    assert engine.state.iteration==30000
    for key,value in engine.state.gauss_container._param_dict.items():
        assert torch.isfinite(value.tensor).all().item(), key
    stats['finite_final_parameters']=True
    stats['final_num_gaussians']=len(engine.state.gauss_container.means)


Engine.run=audited_run
sys.argv=[str(ROOT/'run.py'),'experiment=production/ablations/ours/paper_semantic_freeze']+row['overrides']+[
    'hydra.run.dir='+str(target),'system.output_root='+str(target.parents[3]),'system.exp_tag='+row_id]
torch.cuda.reset_peak_memory_stats()
start=time.monotonic()
try:
    runpy.run_path(str(ROOT/'run.py'),run_name='__main__')
    stats['success']=True
finally:
    torch.cuda.synchronize()
    stats.update(wrapper_wall_seconds=time.monotonic()-start,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name(),
                 overrides=sys.argv[1:])
    target.mkdir(parents=True,exist_ok=True)
    (target/'batch1_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('BATCH1_STATS',json.dumps(stats),flush=True)
