"""Verify a transferred sweep and add a frozen native manifest. No cloud access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# Physical extents are preserved from the frozen dataset definitions. Spine
# extents remain operational calibration assumptions, not measured ground truth.
GEOMETRY = {
    "synthetic_liver": (80.0, 140.0, 0.2685, True),
    "spine_phantom": (40.0, 120.0, 0.79, True),
    "tus_rec_2024_qc": (640 * 0.229389190673828, 480 * 0.220979690551758, 0.404, False),
    "tdsc_abus_2023": (865 * 0.2, 682 * 0.073, 0.475674, True),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(sweep: Path, dataset: str, version: str, source_manifest: Path, *, trim: int = 10, ratio: float = 0.2):
    source = {row["path"]: row for row in map(json.loads, source_manifest.read_text().splitlines())}
    receipts = []
    for file in sorted(sweep.iterdir()):
        if file.name in ("manifest.json", "publication.json") or not file.is_file():
            continue
        relative = f"{dataset}/{sweep.name}/{file.name}"
        expected = source.get(relative)
        if expected is None:
            raise ValueError(f"Missing frozen source identity: {relative}")
        sha = digest(file)
        if sha != expected["sha256"] or file.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"Transferred payload differs from source: {relative}")
        receipts.append({"path": file.name, "sha256": sha, "size_bytes": file.stat().st_size})
    images = np.load(sweep / "images.npy", mmap_mode="r", allow_pickle=False)
    n, h, w = images.shape
    if trim < 0 or not 0 < ratio < 1 or n - 2 * trim <= 1:
        raise ValueError("Invalid holdout interval.")
    count = max(1, min(n - 2 * trim - 1, round(n * ratio)))
    test = np.unique(np.rint(np.linspace(trim, n - 1 - trim, count)).astype(np.int64)).tolist()
    if len(test) != count:
        raise ValueError("Holdout rounding collapsed indices.")
    train = sorted(set(range(n)) - set(test))
    width, depth, elev, center = GEOMETRY[dataset]
    files = {item["path"]: item for item in receipts}
    payload = {name: {"path": name + ".npy", "sha256": files[name + ".npy"]["sha256"]} for name in ("images", "poses")}
    metadata = json.loads((sweep / "metadata.json").read_text())
    value = {
        "schema_version": 1,
        "dataset_version": version,
        "pose_convention": "probe_to_world_mm",
        "image_value_mode": metadata.get(
            "image_value_mode", "spine_phantom_shifted_float" if dataset == "spine_phantom" else "uint8_255"
        ),
        "geometry": {
            "height": h,
            "width": w,
            "lateral_spacing_mm": width / w,
            "axial_spacing_mm": depth / h,
            "near_mm": 0.0,
            "elevational_spacing_mm": elev,
            "center_lateral": center,
            "probe_type": "linear",
        },
        "splits": {
            name: {**payload, "indices": ids, "frame_ids": ids} for name, ids in (("train", train), ("test", test))
        },
        "split_protocol": {"name": "holdout_interval", "trim_each": trim, "ratio": ratio, "count": count},
        "provenance": {
            "source": "eehpc:/home/boywang25/paper1/data/" + dataset + "/" + sweep.name,
            "source_manifest_sha256": digest(source_manifest),
            "files": receipts,
        },
    }
    output = sweep / "manifest.json"
    if output.exists() and json.loads(output.read_text()) != value:
        raise ValueError("Refusing to overwrite an existing different dataset manifest; create a new version.")
    output.write_text(json.dumps(value, indent=2) + "\n")
    if not (sweep / "publication.json").exists():
        (sweep / "publication.json").write_text(
            json.dumps({"status": "unpublished_local_working_copy", "disposable": False, "r2_requests": 0}, indent=2)
            + "\n"
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--dataset", choices=GEOMETRY, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--trim", type=int, default=10)
    args = parser.parse_args()
    print(prepare(args.sweep.resolve(), args.dataset, args.version, args.source_manifest.resolve(), trim=args.trim))
