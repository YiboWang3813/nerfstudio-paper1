import hashlib
import json
import numpy as np
import pytest


@pytest.fixture
def manifest(tmp_path):
    images = np.arange(6 * 16 * 16, dtype=np.float32).reshape(6, 16, 16) % 256
    poses = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
    poses[:, 2, 3] = np.arange(6) * 0.5
    payload = {}
    for name, array in (("images", images), ("poses", poses)):
        path = tmp_path / (name + ".npy")
        np.save(path, array)
        payload[name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    value = {
        "schema_version": 1,
        "dataset_version": "test-fixture-v1",
        "pose_convention": "probe_to_world_mm",
        "image_value_mode": "uint8_255",
        "geometry": {
            "height": 16,
            "width": 16,
            "lateral_spacing_mm": 0.5,
            "axial_spacing_mm": 0.5,
            "near_mm": 2.0,
            "elevational_spacing_mm": 0.5,
            "center_lateral": True,
            "probe_type": "linear",
        },
        "splits": {
            name: {**payload, "indices": ids, "frame_ids": [100 + i for i in ids]}
            for name, ids in (("train", [0, 2, 4, 5]), ("test", [1, 3]))
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path
