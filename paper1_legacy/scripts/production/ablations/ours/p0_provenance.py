"""Write a source/config provenance sidecar for an explicitly selected run.

This does not launch training or change historical run manifests. Future runners
must reference this sidecar from their experiment manifest before submission.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hashes = json.loads((args.snapshot / "hashes.json").read_text())
    critical = {n: h for n, h in hashes.items() if n.startswith(
        ("methods/ours/", "ops/ours_renderer/", "configs_hydra/", "scene/"))}
    changed = [n for n, h in critical.items() if not (ROOT / n).is_file() or digest(ROOT / n) != h]
    if changed:
        raise RuntimeError("Snapshot no longer matches: " + repr(changed))
    payload = dict(source_snapshot=str(args.snapshot.resolve()),
                   snapshot_sha256=(args.snapshot / "snapshot.sha256").read_text().strip(),
                   base_commit=(args.snapshot / "base_commit.txt").read_text().strip(),
                   config_path=str(args.config.resolve()), config_sha256=digest(args.config),
                   critical_source_hashes=critical,
                   renderer_binaries={str(p.relative_to(ROOT)): digest(p) for p in
                       (ROOT / "ops/ours_renderer/ours_renderer").glob("_C*.so")},
                   formal_training_authorized=False)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
