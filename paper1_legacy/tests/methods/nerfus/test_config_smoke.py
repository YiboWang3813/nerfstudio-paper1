import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from methods.nerfus.configs import ModelConfig


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.name == "nerfus"
    assert cfg.field.depth == 8
    assert cfg.field.use_viewdirs is False
    assert cfg.ray.train_patch_height > 0
    assert cfg.field.alpha_activation == "abs"
    assert cfg.field.width == 256
    assert cfg.render.boundary_sampling_mode == "expectation"
    assert cfg.render.scatter_sampling_mode == "expectation"
    assert cfg.render.eval_render_mode == "mc_average"
    assert cfg.render.eval_mc_samples > 1
    assert cfg.solver.optimizer == "radam"
    assert cfg.solver.lr == 5.0e-4
    assert cfg.solver.lr_final == 5.0e-5
