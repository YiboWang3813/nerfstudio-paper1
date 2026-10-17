"""Finish the audit artifacts after tests; compare snapshots without training."""
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908"


def main():
    before = torch.load(OUT / "pre/renderer_probe.pt", map_location="cpu")
    after = torch.load(OUT / "post_renderer_probe.pt", map_location="cpu")
    result = {}
    for key in before:
        a, b = before[key], after[key]
        result[key] = dict(bitwise_equal=torch.equal(a, b), max_abs=(a-b).abs().max().item())
    for label, data in (("pre", before), ("post", after)):
        for name in ("pred", "path", "support", "grad_means", "grad_scales", "grad_quats", "grad_alpha", "grad_tau", "grad_pose"):
            a, b = data["0/" + name], data["1/" + name]
            result[label + "_repeat/" + name] = dict(bitwise_equal=torch.equal(a, b), max_abs=(a-b).abs().max().item())
    (OUT / "default_renderer_regression.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    previous = json.loads((OUT / "pre/hashes.json").read_text())
    changes, patch = {}, []
    names = set(previous)
    names.update(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0"))
    for name in sorted(n for n in names if n and not n.startswith(("outputs/", "logs/", "artifacts/", "wandb/"))):
        old_hash = previous.get(name)
        current = ROOT / name
        if not current.is_file():
            continue
        digest = hashlib.sha256(current.read_bytes()).hexdigest()
        if digest == old_hash:
            continue
        changes[name] = dict(pre=old_hash, post=digest)
        try:
            old = (OUT / "pre/tree" / name).read_text().splitlines(True) if old_hash else []
            new = current.read_text().splitlines(True)
            patch.extend(difflib.unified_diff(old, new, "a/" + name, "b/" + name))
        except UnicodeDecodeError:
            pass
    (OUT / "p0_only.patch").write_text("".join(patch))
    (OUT / "p0_changed_hashes.json").write_text(json.dumps(changes, indent=2) + "\n")
    for name in ("scene/container/single_sweep_container.py", "ops/ours_renderer/src/shader/ray_attenuation.cu"):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == previous[name]
    assert all(v["max_abs"] < 1e-4 for k, v in result.items() if not k.endswith("support"))


if __name__ == "__main__":
    main()
