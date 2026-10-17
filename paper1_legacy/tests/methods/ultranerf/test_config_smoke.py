import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from methods.ultranerf.configs import ModelConfig


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.name == "ultranerf"
    assert cfg.field.depth == 8
    assert cfg.field.use_viewdirs is False
    assert cfg.ray.train_patch_height > 0
    assert cfg.field.alpha_activation == "abs"
    assert cfg.render.boundary_sampling_mode == "bernoulli"
    assert cfg.render.scatter_sampling_mode == "bernoulli"
    assert cfg.render.eval_render_mode == "mc_average"
    assert cfg.render.eval_mc_samples > 1
