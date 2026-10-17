"""Summarize completed P0 diagnostics without interpreting smoke PSNR as evidence."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908"


def main():
    rows = []
    for dataset in ("liver", "arm"):
        for mode in ("adaptive", "fixed", "none"):
            path = OUT / "smoke" / (dataset + "_" + mode)
            stats = json.loads((path / "smoke_stats.json").read_text())
            log = (path / "console.log").read_text()
            initial = int(re.search(r"Init Gaussians.*?N=(\d+)", log).group(1))
            events = [tuple(map(int, m)) for m in re.findall(
                r"\[Ours DC\] Iter (\d+) \| Split: (\d+) Clone: (\d+) \| Prune: (\d+)", log)]
            # Existing spawning retains parents and adds two children per split.
            final = initial + sum(2 * s + c - p for _, s, c, p in events)
            row = dict(dataset=dataset, mode=mode, initial_gaussians=initial,
                       final_gaussians_from_events=final, adaptation_events=events, **stats)
            assert stats.get("success") and stats["finite_outputs"] and stats["forward_calls"] == 100
            assert len(events) == 2
            rows.append(row)
    (OUT / "smoke_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(r["dataset"], r["mode"], r["peak_local_tile_gaussian_pairs"],
              round(r["peak_allocated_bytes"] / 2**30, 3), round(r["wall_seconds"], 2),
              r["final_gaussians_from_events"])
    regression = json.loads((OUT / "default_renderer_regression.json").read_text())
    paired = {k: v for k, v in regression.items() if "repeat/" not in k}
    print("pre/post max_abs", max(v["max_abs"] for v in paired.values()))
    print("within-build repeat max_abs", max(v["max_abs"] for k, v in regression.items() if "repeat/" in k))


if __name__ == "__main__":
    main()
