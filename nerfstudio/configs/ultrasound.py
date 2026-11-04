"""Native method registration. Development preset pending production equivalence."""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.ultrasound_scheduler import UltrasoundExponentialSchedulerConfig
from nerfstudio.pipelines.ultrasound import UltrasoundPipelineConfig


def ultrasound_config():
    rates = {"means": 0.00016, "scales": 0.005, "quats": 0.001, "echo_alphas": 0.08, "attenuations": 0.01}
    return TrainerConfig(
        method_name="ultrasound-dev",
        max_num_iterations=30000,
        mixed_precision=False,
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_eval_all_images=30000,
        steps_per_save=2000,
        pipeline=UltrasoundPipelineConfig(),
        optimizers={
            name: {
                "optimizer": AdamOptimizerConfig(lr=lr, eps=1e-15),
                "scheduler": UltrasoundExponentialSchedulerConfig() if name == "means" else None,
            }
            for name, lr in rates.items()
        },
        viewer=ViewerConfig(),
        vis="tensorboard",
    )
