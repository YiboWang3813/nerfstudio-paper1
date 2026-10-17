"""Only the two user-authorized corrected MP 1M/30k anchors; no matrix runner."""
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
from scripts.production.ablations.ours.p0_plan import common, REP

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/preproduction/regression/p05_corrected_covariance_20260908"
name = sys.argv[1]
assert name in ("liver", "arm") and len(sys.argv) == 2
ds, sw = REP[0 if name == "liver" else 1]
run_dir = OUT / "anchors" / name
assert not (run_dir / "resolved_config.yaml").exists(), "Never overwrite an existing anchor"
freeze = OUT / "freeze_v2"
hashes = json.loads((freeze / "hashes.json").read_text())
for filename, expected in hashes.items():
    if filename.startswith(("methods/", "ops/", "scene/", "configs", "engine/", "loss", "metric", "utils/")):
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
binary_hashes = json.loads((freeze / "binary_hashes.json").read_text())
for filename, expected in binary_hashes.items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
stats = dict(peak_local_tile_gaussian_pairs=0, peak_global_ray_gaussian_pairs=0,
             total_local_tile_gaussian_pairs=0, total_global_ray_gaussian_pairs=0,
             forward_calls=0, finite_sampled_outputs=True, approved_scope="two corrected MP anchors")


def instrument(name, local_index, ray_index=None):
    original = getattr(autograd._C, name)
    def wrapped(*args):
        out = original(*args)
        stats["peak_local_tile_gaussian_pairs"] = max(stats["peak_local_tile_gaussian_pairs"], out[local_index].numel())
        stats["total_local_tile_gaussian_pairs"] += out[local_index].numel()
        if ray_index is not None:
            stats["peak_global_ray_gaussian_pairs"] = max(stats["peak_global_ray_gaussian_pairs"], out[ray_index].numel())
            stats["total_global_ray_gaussian_pairs"] += out[ray_index].numel()
        stats["forward_calls"] += 1
        if stats["forward_calls"] % 1000 == 0:
            assert torch.isfinite(out[0]).all().item(), "Nonfinite corrected render"
        return out
    setattr(autograd._C, name, wrapped)


instrument("ProjectedSliceForwardCUDA", 2)
instrument("ProjectedSliceForwardRayTileCUDA", 3, 8)
original_run = Engine.run


def audited_run(engine):
    manifest_path = run_dir / "run_manifest.yaml"
    manifest = OmegaConf.load(manifest_path)
    manifest.p05_source_snapshot = str(freeze)
    manifest.p05_snapshot_sha256 = (freeze / "snapshot.sha256").read_text().strip()
    manifest.p05_config_sha256 = hashlib.sha256((run_dir / "resolved_config.yaml").read_bytes()).hexdigest()
    manifest.p05_binary_hashes = binary_hashes
    manifest.p05_scope = "authorized MP full LGR full CGA impact anchor; not a matrix"
    OmegaConf.save(manifest, manifest_path)
    original_run(engine)
    assert engine.state.iteration == 30000
    for key, value in engine.state.gauss_container._param_dict.items():
        assert torch.isfinite(value.tensor).all().item(), "Nonfinite final parameter: " + key
    stats["finite_final_parameters"] = True


Engine.run = audited_run
sys.argv = [str(ROOT / "run.py"), "experiment=production/ablations/ours/paper_semantic_freeze"] + common(ds, sw)
sys.argv += ["engine.iterations=30000", "engine.interval.test=30000", "engine.interval.save=30000",
             "wandb.enable=false", "hydra.run.dir=" + str(run_dir),
             "system.output_root=" + str(OUT / "anchors"), "system.exp_tag=p05_corrected_" + name]
torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
try:
    runpy.run_path(str(ROOT / "run.py"), run_name="__main__")
    stats["success"] = True
finally:
    torch.cuda.synchronize()
    stats.update(wall_seconds=time.monotonic() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                 overrides=sys.argv[1:])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "anchor_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("P05_ANCHOR", json.dumps(stats))
