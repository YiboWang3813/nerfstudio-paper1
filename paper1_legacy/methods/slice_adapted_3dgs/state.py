from engine.registry import register_state
from engine.state import State
from scene.scene import Scene

from .density_control import SliceAdapted3DGSDensityController
from .gaussian_container import SliceAdapted3DGSContainer
from .gaussian_solver import SliceAdapted3DGSSolver


@register_state("SliceAdapted3DGSOfflineState")
class SliceAdapted3DGSOfflineState(State):
    scene: Scene = None
    gauss_container: SliceAdapted3DGSContainer = None
    gauss_solver: SliceAdapted3DGSSolver = None
    dc: SliceAdapted3DGSDensityController = None
