"""Freeze all verified medical sweeps locally, one ZIP at a time. No cloud I/O."""

import argparse
import json
import os
import shutil
from pathlib import Path

from package_dataset import package
from prepare_manifest import GEOMETRY, digest, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    raw = Path("/home/yibo/shared/datasets/paper1-unpublished/eehpc-20261017")
    destination = Path("/home/yibo/shared/datasets")
    packages = Path("/home/yibo/shared/cache/dataset-packages/paper1")
    rows = [json.loads(line) for line in args.source_index.read_text().splitlines()]
    rows = [row for row in rows if row["path"].split("/")[0] in GEOMETRY]
    if shutil.disk_usage(destination).free < sum(row["size_bytes"] for row in rows) + 2 * 1024**3:
        raise OSError("Insufficient space for sequential package staging")
    for row in rows:
        path = raw / row["path"]
        if digest(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise ValueError(f"Source verification failed: {row['path']}")
    for dataset in GEOMETRY:
        selected = [row for row in rows if row["path"].startswith(dataset + "/")]
        root = destination / dataset / args.version
        root.mkdir(parents=True, exist_ok=True)
        for row in selected:
            relative = Path(row["path"]).relative_to(dataset)
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(raw / row["path"], dest)
            elif digest(dest) != row["sha256"]:
                raise ValueError(f"Destination differs: {dest}")
        sweeps = sorted({Path(row["path"]).parts[1] for row in selected if len(Path(row["path"]).parts) >= 3})
        receipts = []
        for name in sweeps:
            sweep = root / name
            manifest = prepare(
                sweep, dataset, args.version, args.source_index, trim=8 if dataset == "spine_phantom" else 10
            )
            output = packages / dataset / args.version / (name + ".zip")
            if output.exists():
                receipt = json.loads(output.with_suffix(".receipt.json").read_text())
                if digest(output) != receipt["sha256"] or digest(manifest) != receipt["files"]["manifest.json"]:
                    raise ValueError("Existing package does not match frozen manifest")
            else:
                receipt = package(sweep, output)
            receipts.append(receipt)
            print(dataset, name, receipt["sha256"], flush=True)
        aux = [
            {"path": Path(row["path"]).name, "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
            for row in selected
            if len(Path(row["path"]).parts) == 2
        ]
        value = {
            "schema_version": 1,
            "dataset": dataset,
            "version": args.version,
            "source_index_sha256": digest(args.source_index),
            "packages": receipts,
            "auxiliary_files": aux,
            "publication_status": "unpublished_local_working_copy",
            "disposable": False,
            "r2_requests": 0,
        }
        catalog = root / "dataset.json"
        if catalog.exists() and json.loads(catalog.read_text()) != value:
            raise ValueError("Frozen catalog differs")
        catalog.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
