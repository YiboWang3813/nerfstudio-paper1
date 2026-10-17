# methods/hooks/gaussian_model.py

from common.debug.debug_scope import DebugScope
from common.debug.debug_utils import range_stats, grad_stats
import torch


class GaussianModelDebugHook:
    """
    Debug hook for GaussianModel (UltraGS-style, minimal).

    Policy:
    - Forward: log PHYSICAL (activated) parameters via model.<name>
    - Backward: log RAW parameter gradients via model._<name>.grad
    - NO norm, NO aggregation semantics
    - Flatten everything and show real distributions
    """

    # ============================================================
    # Forward: physical / activated parameters
    # ============================================================
    @staticmethod
    @torch.no_grad()
    def on_forward(model):
        if not DebugScope.enabled():
            return

        stats = {}

        for name in model.param_cfg.keys():
            if not hasattr(model, name):
                continue

            # physical / activated value
            value = getattr(model, name)

            # flatten and log
            v = value.reshape(-1)
            stats.update(range_stats(
                v,
                f"debug/gauss/{name}"
            ))

        DebugScope.update(stats)

    # ============================================================
    # Backward: raw parameter gradients
    # ============================================================
    @staticmethod
    def on_backward(model):
        if not DebugScope.enabled():
            return

        stats = {}

        for name, cfg in model.param_cfg.items():
            raw_name = cfg["param"]
            if not hasattr(model, raw_name):
                continue

            raw_param = getattr(model, raw_name)
            grad = raw_param.grad
            if grad is None:
                continue

            # flatten and log
            g = grad.reshape(-1)
            stats.update(grad_stats(
                g,
                f"debug/grad/{name}"
            ))

        DebugScope.update(stats)
