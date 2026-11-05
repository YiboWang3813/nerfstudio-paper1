"""Contract tests: geometry, split leakage, payload identity and sampler resume."""

import json

import pytest
import torch

from nerfstudio.data.datamanagers.ultrasound_datamanager import UltrasoundDataManagerConfig
from nerfstudio.data.dataparsers.ultrasound_dataparser import UltrasoundDataParserConfig


def test_geometry_and_values(manifest):
    manager = UltrasoundDataManagerConfig(dataparser=UltrasoundDataParserConfig(data=manifest)).setup(device="cpu")
    camera, batch = manager.next_eval_image(0)
    assert batch["frame_id"] == 101
    assert batch["image"].shape == (16, 16, 1)
    assert batch["image"].min() >= 0 and batch["image"].max() <= 1
    assert torch.allclose(camera.camera_to_worlds[0, :3, :3], torch.diag(torch.tensor([1.0, -1.0, -1.0])))
    assert camera.camera_to_worlds[0, 2, 3] == 0.5
    assert manager.train_dataparser_outputs.dataparser_scale == 1.0


def test_canonical_sweep_uses_explicit_spacing(manifest):
    parser = UltrasoundDataParserConfig(
        data=manifest, pose_mode="canonical", canonical_spacing_mm=2.0, canonical_reference_frame_id=100
    ).setup()
    output = parser.get_dataparser_outputs("test")
    assert output.cameras.camera_to_worlds[0, 2, 3] == 2.0
    assert output.cameras.camera_to_worlds[1, 2, 3] == 6.0
    with pytest.raises(ValueError, match="explicit positive"):
        UltrasoundDataParserConfig(data=manifest, pose_mode="canonical").setup().get_dataparser_outputs("train")


def test_sampler_checkpoint_restores_exact_next_frames(manifest):
    config = UltrasoundDataManagerConfig(dataparser=UltrasoundDataParserConfig(data=manifest))
    first = config.setup(device="cpu")
    for step in range(3):
        first.next_train(step)
    state = first.state_dict()
    expected = [first.next_train(i)[1]["frame_id"] for i in range(12)]
    second = config.setup(device="cpu")
    second.load_state_dict(state)
    assert [second.next_train(i)[1]["frame_id"] for i in range(12)] == expected


def test_rejects_leakage_and_modified_payload(manifest):
    data = json.loads(manifest.read_text())
    data["splits"]["test"]["indices"][0] = 0
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leakage"):
        UltrasoundDataParserConfig(data=manifest).setup().get_dataparser_outputs("train")
    payload = manifest.parent / "images.npy"
    with payload.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        UltrasoundDataParserConfig(data=manifest).setup().get_dataparser_outputs("train")


@pytest.mark.parametrize("mode,expected", [("file_step_norm", [2.5, 5.5]), ("file_axis_projection", [2.0, -1.0])])
def test_nonuniform_canonical_spacing_preserves_source_steps_and_holdout(manifest, mode, expected):
    import hashlib  # noqa: PLC0415 - import under the shared-runtime lock

    import numpy as np  # noqa: PLC0415 - import under the shared-runtime lock

    data = json.loads(manifest.read_text())
    pose_file = manifest.parent / data["splits"]["train"]["poses"]["path"]
    poses = np.load(pose_file)
    poses[:4, :3, 3] = [[0, 0, 0], [3, 0, 4], [3, 0, 2], [3, 0, -2]]
    np.save(pose_file, poses)
    digest = hashlib.sha256(pose_file.read_bytes()).hexdigest()
    for split in data["splits"].values():
        split["poses"]["sha256"] = digest
    manifest.write_text(json.dumps(data))
    parser = UltrasoundDataParserConfig(
        data=manifest,
        pose_mode="canonical",
        canonical_spacing_mm=1.0,
        canonical_reference_frame_id=100,
        canonical_spacing_mode=mode,
        canonical_spacing_scale=0.5,
    ).setup()
    output = parser.get_dataparser_outputs("test")
    torch.testing.assert_close(output.cameras.camera_to_worlds[:, 2, 3], torch.tensor(expected))


def test_default_uniform_canonical_identity_keeps_existing_checkpoint_contract(manifest):
    import hashlib  # noqa: PLC0415 - import under the shared-runtime lock

    cfg = UltrasoundDataParserConfig(data=manifest, pose_mode="canonical", canonical_spacing_mm=2.0)
    output = cfg.setup().get_dataparser_outputs("train")
    old_identity = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "pose_mode": "canonical",
        "canonical_spacing_mm": 2.0,
        "canonical_axis": "z",
        "canonical_reference_frame_id": 0,
        "pose_validation": "rigid",
    }
    assert (
        output.metadata["parser_identity_sha256"]
        == hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
    )
