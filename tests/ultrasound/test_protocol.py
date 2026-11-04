"""Scientific configuration and metric contracts independent of full budgets."""

import hashlib
import json

import numpy as np
import pytest
import torch
import yaml

from nerfstudio.cameras.ultrasound_poses import (
    UltrasoundCameraOptimizer,
    UltrasoundPoseConfig,
    UltrasoundUncertaintyConfig,
    frame_weights,
)
from nerfstudio.configs.ultrasound_protocol import METHODS, production_config, update_dataclass


@pytest.mark.parametrize("method", METHODS)
def test_frozen_protocol_and_native_serialization(method, manifest, tmp_path):
    data = json.loads(manifest.read_text())
    data["dataset_version"] = "p100k-native-v2"
    data["split_protocol"] = {"trim_each": 8, "ratio": 0.2}
    path = tmp_path / "spine_phantom/p100k-native-v2/left1/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    (path.parent.parent / "dataset.json").write_text(
        json.dumps(
            {
                "packages": [
                    {"sweep": "left1", "files": {"manifest.json": hashlib.sha256(path.read_bytes()).hexdigest()}}
                ]
            }
        )
    )
    cfg = production_config(method, path, tmp_path / "protocol-not-launched")
    restored = yaml.load(yaml.dump(cfg), Loader=yaml.Loader)
    assert restored.pipeline.model._target is cfg.pipeline.model._target
    assert cfg.max_num_iterations == 30000
    assert cfg.pipeline.model.full_metrics
    if method in ("nerfus", "ultranerf"):
        assert cfg.pipeline.model.mse_weight == 0.1 and cfg.pipeline.model.ssim_weight == 0.9
        assert cfg.pipeline.model.l1_weight == 0
    else:
        assert cfg.pipeline.model.num_gaussians == 1000000
    with pytest.raises(ValueError, match="Unsupported"):
        update_dataclass(cfg.pipeline.model, {"unknown_scientific_option": True})


def test_pose_is_probe_local_and_residuals_are_bounded():
    config = UltrasoundPoseConfig(enabled=True, trainable=True, residual_bound_trans_mm=0.2)
    pose = torch.eye(4)[None]
    pose[0, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    component = UltrasoundCameraOptimizer(config, [99], pose)
    component.step = 1
    with torch.no_grad():
        component.pose_delta_trans[0, 0] = 0.5
    component.clamp_residuals()
    result = pose[0, :3, :3] @ component(torch.tensor(0))[:3, 3]
    torch.testing.assert_close(result, torch.tensor([0.0, 0.2, 0.0]))
    groups = {}
    component.get_param_groups(groups)
    assert set(groups) == {"pose_rot", "pose_trans"}


def test_uncertainty_mapping_and_invalid_inputs(tmp_path):
    path = tmp_path / "weights.npy"
    np.save(path, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    cfg = UltrasoundUncertaintyConfig(enabled=True, uncertainty_abspath=str(path), weight_mode="exp")
    expected = 0.1 + 0.9 * torch.exp(-torch.arange(3, dtype=torch.float32))
    torch.testing.assert_close(frame_weights(cfg, [10, 20, 30]), expected)
    with pytest.raises(ValueError):
        frame_weights(cfg, [10, 20])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual perceptual GPU execution required")
def test_full_metric_protocol_is_separate_from_train_modules():
    from nerfstudio.model_components.ultrasound_metrics import evaluator, full_reference_metrics

    pred = torch.rand(64, 64, 1, device="cuda")
    target = torch.rand_like(pred)
    result = full_reference_metrics(pred, target)
    assert set(result) == {"psnr", "l1", "ssim", "lpips", "dists", "gmsd"}
    assert all(np.isfinite(value) for value in result.values())
    p = pred.permute(2, 0, 1)[None].expand(-1, 3, -1, -1)
    t = target.permute(2, 0, 1)[None].expand_as(p)
    for name, metric in evaluator("cuda:0").items():
        assert result[name] == pytest.approx(float(metric(p, t).mean()), rel=1e-6, abs=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA pose gradient required")
def test_native_pose_uncertainty_and_mc_training(manifest, tmp_path):
    from nerfstudio.configs.ultrasound import ultrasound_config

    cfg = ultrasound_config()
    cfg.pipeline.datamanager.dataparser.data = manifest
    cfg.pipeline.model.num_gaussians = 32
    pose = cfg.pipeline.model.pose
    pose.enabled = pose.trainable = True
    pose.start_iter = 1
    pose.lambda_prior = 0.01
    u = cfg.pipeline.model.uncertainty
    weights = tmp_path / "weights.npy"
    np.save(weights, np.full(4, 0.4, dtype=np.float32))
    u.enabled = True
    u.uncertainty_abspath = str(weights)
    u.mc_enabled = True
    u.mc_samples = 2
    u.mc_trans_sigma_mm = 0.05
    from nerfstudio.engine.optimizers import AdamOptimizerConfig

    for name in ("pose_rot", "pose_trans"):
        cfg.optimizers[name] = {"optimizer": AdamOptimizerConfig(lr=3e-5, eps=1e-15), "scheduler": None}
    cfg.output_dir = tmp_path / "runs"
    cfg.experiment_name = "pose-mc"
    cfg.timestamp = "test"
    cfg.max_num_iterations = 3
    cfg.logging.local_writer.enable = False
    cfg.steps_per_eval_all_images = cfg.steps_per_eval_image = cfg.steps_per_eval_batch = 0
    trainer = cfg.setup()
    trainer.setup()
    trainer.train()
    component = trainer.pipeline.model.camera_optimizer
    assert torch.isfinite(component.pose_delta_trans).all()
    assert component.pose_delta_trans.abs().max() > 0
    state = trainer.pipeline.model.state_dict()
    assert not any("lpips" in key or "dists" in key for key in state)
