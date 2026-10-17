from engine.registry import register_state
from engine.state import State
from scene.scene import Scene

from .configs import ModelConfig
from .density_control import GaussianPileOfflineDensityController
from .gaussian_container import GaussianPileGaussContainer
from .gaussian_solver import GaussianPileGaussSolver


@register_state("GaussianPileOfflineState")
class GaussianPileOfflineState(State):
    scene: Scene = None
    gauss_container: GaussianPileGaussContainer = None
    gauss_solver: GaussianPileGaussSolver = None
    dc: GaussianPileOfflineDensityController = None
    scene_radius: float = 0.0

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

