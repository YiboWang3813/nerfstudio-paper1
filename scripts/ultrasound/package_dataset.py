"""Create and locally verify one self-contained dataset ZIP. Never contacts R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(sweep: Path, output: Path):
    manifest = json.loads((sweep / "manifest.json").read_text())
    files = [sweep / item["path"] for item in manifest["provenance"]["files"]] + [sweep / "manifest.json"]
    for split in manifest["splits"].values():
        for name in ("images", "poses"):
            file = sweep / split[name]["path"]
            if file.resolve().parent != sweep.resolve() or file not in files:
                raise ValueError("Every runtime payload must be included inside the ZIP.")
    expected = {file.name: sha256(file) for file in files}
    for item in manifest["provenance"]["files"]:
        if expected[item["path"]] != item["sha256"]:
            raise ValueError("A source payload changed after manifest freeze; create a new dataset version.")
    output.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(file.stat().st_size for file in files) + 64 * 1024 * 1024
    if shutil.disk_usage(output.parent).free < needed:
        raise OSError(f"Insufficient space for verified package staging: require {needed} bytes.")
    if output.exists():
        raise FileExistsError("A versioned dataset package must not be overwritten.")
    partial = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for file in files:
            archive.write(file, file.name)
    with zipfile.ZipFile(partial, "r") as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("Unexpected package members.")
        for name, digest in expected.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"ZIP verification failed: {name}")
    partial.rename(output)
    receipt = {
        "schema_version": 1,
        "dataset_version": manifest["dataset_version"],
        "sweep": sweep.name,
        "package": str(output),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "files": expected,
        "verification": "all ZIP members read back and SHA256 checked",
        "publication_status": "not_uploaded",
        "r2_requests": 0,
        "source_may_be_deleted": False,
    }
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.sweep.resolve(), args.output.resolve()), indent=2))
