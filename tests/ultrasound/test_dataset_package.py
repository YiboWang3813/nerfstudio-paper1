"""A package is readable alone and cannot silently publish modified source bytes."""

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def test_package_roundtrip_and_frozen_payload_guard(manifest, tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/ultrasound/package_dataset.py"
    spec = importlib.util.spec_from_file_location("dataset_package", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = json.loads(manifest.read_text())
    data["provenance"] = {
        "files": [
            {"path": name + ".npy", "sha256": data["splits"]["train"][name]["sha256"]} for name in ("images", "poses")
        ]
    }
    manifest.write_text(json.dumps(data))
    output = tmp_path / "packages/sweep.zip"
    receipt = module.package(manifest.parent, output)
    assert receipt["r2_requests"] == 0 and receipt["publication_status"] == "not_uploaded"
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"manifest.json", "images.npy", "poses.npy"}
    with (manifest.parent / "images.npy").open("ab") as stream:
        stream.write(b"unpublished change")
    with pytest.raises(ValueError, match="changed after manifest freeze"):
        module.package(manifest.parent, tmp_path / "packages/modified.zip")
    assert not (tmp_path / "packages/modified.zip").exists()
