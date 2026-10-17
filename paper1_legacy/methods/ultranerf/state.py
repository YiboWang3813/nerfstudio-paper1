from engine.registry import register_state
from engine.state import State


@register_state("UltraNeRFOfflineState")
class UltraNeRFOfflineState(State):
    scene = None
    field = None
    solver = None
