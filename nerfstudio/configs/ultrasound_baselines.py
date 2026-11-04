"""Native method configurations; historical config conversion is offline only."""

import importlib
from dataclasses import fields

from nerfstudio.configs.ultrasound import ultrasound_config
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.ultrasound_scheduler import UltrasoundExponentialSchedulerConfig
from nerfstudio.models.ultrasound_baselines import UltrasoundBaselineModelConfig

GAUSSIAN_METHODS = ("ultragauss", "ultragray", "gaussianpile", "slice_adapted_3dgs", "ultrags_baseline")


def baseline_config(method, algorithm=None):
    if method not in GAUSSIAN_METHODS:
        raise ValueError(method)
    if algorithm is None:
        module = importlib.import_module("nerfstudio.model_components.ultrasound_baselines." + method + ".configs")
        algorithm = module.ModelConfig()
    config = ultrasound_config()
    config.method_name = "ultrasound-" + method
    config.pipeline.model = UltrasoundBaselineModelConfig(algorithm=algorithm)
    config.optimizers = {}
    for item in fields(algorithm.gauss_container.lr):
        name = item.name
        lr = getattr(algorithm.gauss_container.lr, name)
        factor = 0.1
        if method in ("slice_adapted_3dgs", "ultrags_baseline"):
            factor = getattr(
                algorithm.gauss_container.scheduler,
                "position_lr_final_scale" if name == "means" else "default_lr_final_scale",
            )
        scheduler = UltrasoundExponentialSchedulerConfig(final_factor=factor)
        if method == "gaussianpile" and name != "means":
            scheduler = None
        config.optimizers[name] = {"optimizer": AdamOptimizerConfig(lr=lr, eps=1e-15), "scheduler": scheduler}
    return config


def field_config(method, algorithm=None):
    from nerfstudio.engine.optimizers import RAdamOptimizerConfig
    from nerfstudio.models.ultrasound_fields import UltrasoundFieldModelConfig

    if method not in ("nerfus", "ultranerf"):
        raise ValueError(method)
    if algorithm is None:
        algorithm = importlib.import_module(
            "nerfstudio.model_components.ultrasound_baselines." + method + ".configs"
        ).ModelConfig()
    config = ultrasound_config()
    config.method_name = "ultrasound-" + method
    config.pipeline.model = UltrasoundFieldModelConfig(algorithm=algorithm)
    optimizer = RAdamOptimizerConfig if algorithm.solver.optimizer == "radam" else AdamOptimizerConfig
    factor = algorithm.solver.lr_final / algorithm.solver.lr if method == "nerfus" else algorithm.solver.lr_decay_rate
    config.optimizers = {
        "field": {
            "optimizer": optimizer(lr=algorithm.solver.lr, eps=1e-8),
            "scheduler": UltrasoundExponentialSchedulerConfig(
                final_factor=factor, max_steps=algorithm.solver.lr_decay_steps, first_update=0, clamp_at_final=False
            ),
        }
    }
    return config
