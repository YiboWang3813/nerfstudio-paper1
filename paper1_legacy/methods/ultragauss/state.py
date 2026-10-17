from engine.state import State
from engine.registry import register_state

from scene.scene import Scene
from .gaussian_container import UltraGaussContainer
from .gaussian_solver import UltraGaussSolver
from .density_control import UltraGaussOfflineDensityController


@register_state("UltraGaussOfflineState")
class UltraGaussOfflineState(State):
    scene: Scene = None
    gauss_container: UltraGaussContainer = None
    gauss_solver: UltraGaussSolver = None
    dc: UltraGaussOfflineDensityController = None
