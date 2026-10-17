"""Instrument short, real-data P0 runs; refuses formal training budgets."""
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch
from ours_renderer import autograd

if not any(arg.startswith("engine.iterations=") and 1 <= int(arg.split("=", 1)[1]) <= 500 for arg in sys.argv):
    raise SystemExit("P0 smoke requires explicit engine.iterations in [1, 500]")

stats = dict(peak_local_tile_gaussian_pairs=0, peak_global_ray_gaussian_pairs=0,
             forward_calls=0, finite_outputs=True, paper_result=False)


def instrument(name, local_index, ray_index=None):
    original = getattr(autograd._C, name)
    def wrapped(*args):
        out = original(*args)
        stats["peak_local_tile_gaussian_pairs"] = max(stats["peak_local_tile_gaussian_pairs"], out[local_index].numel())
        if ray_index is not None:
            stats["peak_global_ray_gaussian_pairs"] = max(stats["peak_global_ray_gaussian_pairs"], out[ray_index].numel())
        stats["forward_calls"] += 1
        if not torch.isfinite(out[0]).all().item():
            stats["finite_outputs"] = False
            raise RuntimeError("Nonfinite P0 smoke render")
        return out
    setattr(autograd._C, name, wrapped)


instrument("ProjectedSliceForwardCUDA", 2)
instrument("ProjectedSliceForwardRayTileCUDA", 3, 8)
torch.cuda.reset_peak_memory_stats()
start = time.monotonic()
try:
    runpy.run_path(str(Path(__file__).resolve().parents[4] / "run.py"), run_name="__main__")
    stats["success"] = True
finally:
    torch.cuda.synchronize()
    stats.update(wall_seconds=time.monotonic() - start,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 gpu=torch.cuda.get_device_name(), overrides=sys.argv[1:])
    destination = Path(os.environ["P0_SMOKE_STATS"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(stats, indent=2) + "\n")
    print("P0_SMOKE", json.dumps(stats))
