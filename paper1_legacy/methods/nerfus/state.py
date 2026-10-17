from engine.registry import register_state
from engine.state import State


@register_state("NerfusOfflineState")
class NerfusOfflineState(State):
    scene = None
    field = None
    solver = None
