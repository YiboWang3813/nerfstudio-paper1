from gaussian.gauss_solver import GaussSolver
from gaussian.lr_scheduler_manager import get_expon_lr_func
from configs.root import RootConfig

from .configs import ModelConfig
from .gaussian_container import UltraGaussContainer


class UltraGaussSolver(GaussSolver):
    def __init__(self, cfg: RootConfig, container: UltraGaussContainer):
        super().__init__(container)
        self.cfg = cfg

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def _setup_lr_scheduler(self):
        max_steps = int(self.cfg.engine.iterations)
        for name, spec in self.container._specs.items():
            lr0 = float(spec.lr)
            self.lr_mgr.register(
                name,
                get_expon_lr_func(lr_init=lr0, lr_final=lr0 * 0.1, max_steps=max_steps),
            )
