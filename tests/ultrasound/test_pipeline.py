"""Native Trainer integration: update, eval, checkpoint and exact continuation."""

import copy

import pytest
import torch

from nerfstudio.configs.ultrasound import ultrasound_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("density", [False, True, "path"])
def test_native_trainer_checkpoint_and_continuation(manifest, tmp_path, density):
    config = ultrasound_config()
    config.pipeline.datamanager.dataparser.data = manifest
    config.pipeline.model.num_gaussians = 32
    config.pipeline.model.renderer.elev_corridor_half_mm = 2.0
    strategy = config.pipeline.model.density
    strategy.enabled = bool(density)
    strategy.refine_start, strategy.refine_stop, strategy.refine_every = 1, 4, 2
    strategy.gradient_threshold = 0.0
    strategy.pruning = "none"
    if density == "path":
        strategy.pruning = "path_aware"
        strategy.densification = "residual_guided"
        renderer = config.pipeline.model.renderer
        renderer.ray_attenuation_enabled = True
        renderer.ray_attenuation_source = "attenuation"
        renderer.ray_attenuation_mapper_kind = "ray_csr"
        renderer.ray_attenuation_beta = 0.6
        renderer.ray_attenuation_strength = 0.3
    config.output_dir = tmp_path / "runs"
    config.experiment_name = "native-contract"
    config.timestamp = "first"
    config.max_num_iterations = 3
    config.steps_per_eval_batch = 0
    config.steps_per_eval_image = 0
    config.steps_per_eval_all_images = 0
    config.logging.local_writer.enable = False
    trainer = config.setup()
    trainer.setup()
    before = trainer.pipeline.model.gaussians["means"].detach().clone()
    trainer.train()
    assert not torch.equal(before, trainer.pipeline.model.gaussians["means"])
    metrics, _ = trainer.pipeline.get_eval_image_metrics_and_images(3)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    camera, _ = trainer.pipeline.datamanager.next_eval_image(3)
    camera.rescale_output_resolution(0.5)
    trainer.pipeline.model.eval()
    with torch.no_grad():
        preview = trainer.pipeline.model.get_outputs_for_camera(camera)
    assert preview["rgb"].shape == (8, 8, 3)
    trainer.pipeline.model.train()
    checkpoint = next(trainer.checkpoint_dir.glob("*.ckpt"))
    # Continue both uninterrupted and checkpoint-loaded trainers by one native step.
    expected_frame = trainer.pipeline.datamanager._order[trainer.pipeline.datamanager._cursor].item()
    from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location

    def run_iteration(current):
        for callback in current.callbacks:
            callback.run_callback_at_location(3, Location.BEFORE_TRAIN_ITERATION)
        current.train_iteration(3)
        for callback in current.callbacks:
            callback.run_callback_at_location(3, Location.AFTER_TRAIN_ITERATION)

    run_iteration(trainer)
    expected = {name: p.detach().clone() for name, p in trainer.pipeline.model.gaussians.items()}
    restored_config = copy.deepcopy(config)
    restored_config.timestamp = "restored"
    restored_config.load_checkpoint = checkpoint
    restored = restored_config.setup()
    restored.setup()
    manager = restored.pipeline.datamanager
    assert manager._order[manager._cursor].item() == expected_frame
    run_iteration(restored)
    for name, parameter in restored.pipeline.model.gaussians.items():
        torch.testing.assert_close(parameter, expected[name], atol=1e-7, rtol=1e-6)
        assert restored.optimizers.optimizers[name].param_groups[0]["params"][0] is parameter
    if density:
        assert len(expected["means"]) > 32
        assert restored.pipeline.model.strategy_state["last_event"]["step"] == 4
    assert not any(key.startswith(("engine", "methods", "scene", "ours_renderer")) for key in __import__("sys").modules)
