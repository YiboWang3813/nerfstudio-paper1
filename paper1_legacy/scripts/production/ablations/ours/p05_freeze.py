"""Archive corrected source plus exact anchors, binary and environment provenance."""
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
from configs.setup import load_hydra_config
from scripts.production.ablations.ours.p0_snapshot import capture
from scripts.production.ablations.ours.p0_plan import common, REP
from scripts.production.ablations.ours.p0_reuse import semantic, flatten

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/preproduction/regression/p05_corrected_covariance_20260908"
PRE = ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908"


def main():
    destination = OUT / (sys.argv[1] if len(sys.argv) > 1 else "freeze")
    capture(destination)
    previous = json.loads((PRE / "final/hashes.json").read_text())
    current = json.loads((destination / "hashes.json").read_text())
    # Read original P0 final bytes directly from the preserved archive.
    import tarfile
    patch, covariance_patch = [], []
    with tarfile.open(PRE / "final/source.tar.gz") as archive:
        for name, digest in current.items():
            if previous.get(name) == digest:
                continue
            try:
                old = archive.extractfile(name).read().decode().splitlines(True) if name in previous else []
                new = (ROOT / name).read_text().splitlines(True)
                delta = list(difflib.unified_diff(old, new, "a/" + name, "b/" + name))
                patch.extend(delta)
                if name.startswith("ops/ours_renderer/"):
                    covariance_patch.extend(delta)
            except UnicodeDecodeError:
                pass
    (destination / "p05_since_p0.patch").write_text("".join(patch))
    (destination / "covariance.patch").write_text("".join(covariance_patch))
    shutil.copy2(PRE / "p0_only.patch", destination / "p0_infrastructure.patch")
    # Preserve source dependencies of the pre-P0 regression tests, not just hashes.
    shutil.copy2(PRE / "pre/source.tar.gz", destination / "pre_p0_source.tar.gz")
    shutil.copy2(PRE / "final/source.tar.gz", destination / "p0_final_source.tar.gz")
    binaries = {}
    for path in (ROOT / "ops/ours_renderer/ours_renderer").glob("_C*.so"):
        binaries[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, destination / path.name)
    (destination / "binary_hashes.json").write_text(json.dumps(binaries, indent=2) + "\n")
    (destination / "pip_freeze.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
    (destination / "conda_explicit.txt").write_bytes(subprocess.check_output([
        "/home/boywang25/miniconda3/bin/conda", "list", "--explicit", "-p", str(Path(sys.executable).parents[1])]))
    (destination / "host_compiler.txt").write_bytes(subprocess.check_output(["gcc", "--version"]))
    (destination / "python_version.txt").write_text(sys.version + "\n" + sys.executable + "\n")
    for path in (ROOT / "logs/ours_ablation").glob("p05test_*.out"):
        shutil.copy2(path, destination / path.name)
    for path in (ROOT / "logs/ours_ablation").glob("p05test_*.err"):
        shutil.copy2(path, destination / path.name)
    for path in (OUT / "smoke").glob("*/*"):
        if path.name in ("resolved_config.yaml", "smoke_stats.json", "run_manifest.yaml", "console.log"):
            target = destination / "smoke" / path.parent.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / path.name)
    config_comparison = {}
    for ds, sw in REP:
        cfg = load_hydra_config("production/ablations/ours/paper_semantic_freeze", common(ds, sw))
        OmegaConf.save(cfg, destination / ("anchor_" + ds + ".yaml"), resolve=True)
        legacy = ROOT / "outputs/production/ablations/ours/path_aware_pruning_update_20260831/stage2_t03_t01/runs/t03" / ds / sw / "d_full_path_aware_pruning/resolved_config.yaml"
        a = flatten(semantic(OmegaConf.to_container(OmegaConf.load(legacy), resolve=False)))
        b = flatten(semantic(OmegaConf.to_container(cfg, resolve=True)))
        differences = {k: [a.get(k), b.get(k)] for k in set(a) | set(b) if a.get(k) != b.get(k)}
        config_comparison[ds] = differences
        assert not differences, differences
    (destination / "matched_stage2_config_differences.json").write_text(json.dumps(config_comparison, indent=2) + "\n")
    print("SOURCE_SHA256", (destination / "snapshot.sha256").read_text())


if __name__ == "__main__":
    main()
