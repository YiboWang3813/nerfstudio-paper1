from engine.registry import register_state
from engine.state import State

from scene.scene import Scene

from .configs import ModelConfig
from .gaussian_container import OursGaussContainer
from .gaussian_solver import OursGaussSolver
from .density_control import OursOfflineDensityController
from .pose_optimization import OursPoseManager


@register_state("OursOfflineState")
class OursOfflineState(State):
    scene: Scene = None
    gauss_container: OursGaussContainer = None
    gauss_solver: OursGaussSolver = None
    dc: OursOfflineDensityController = None
    pose_manager: OursPoseManager = None
    pose_uncertainty_weights = None
    #: BBox radius (mm) from ``build_init_gaussians`` for logging / optional losses.
    scene_radius: float = 0.0

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model
