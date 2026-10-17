from engine.state import State
from engine.registry import register_state

from scene.scene import Scene
from .gaussian_container import UltraGrayGaussContainer
from .gaussian_solver import UltraGrayGaussSolver
from .density_control import UltraGrayOfflineDensityController


@register_state("UltraGrayOfflineState")
class UltraGrayOfflineState(State):
    scene: Scene = None
    gauss_container: UltraGrayGaussContainer = None
    gauss_solver: UltraGrayGaussSolver = None
    dc: UltraGrayOfflineDensityController = None
