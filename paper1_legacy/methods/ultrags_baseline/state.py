from engine.registry import register_state
from engine.state import State
from scene.scene import Scene

from .density_control import UltraGSBaselineDensityController
from .gaussian_container import UltraGSBaselineContainer
from .gaussian_solver import UltraGSBaselineSolver


@register_state("UltraGSBaselineOfflineState")
class UltraGSBaselineOfflineState(State):
    scene: Scene = None
    gauss_container: UltraGSBaselineContainer = None
    gauss_solver: UltraGSBaselineSolver = None
    dc: UltraGSBaselineDensityController = None
