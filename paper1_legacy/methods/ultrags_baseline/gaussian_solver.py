from gaussian.gauss_solver import GaussSolver
from gaussian.lr_scheduler_manager import get_expon_lr_func
from configs.root import RootConfig

from .configs import ModelConfig
from .gaussian_container import UltraGSBaselineContainer


class UltraGSBaselineSolver(GaussSolver):
    def __init__(self, cfg: RootConfig, container: UltraGSBaselineContainer):
        super().__init__(container)
        self.cfg = cfg

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def _setup_lr_scheduler(self):
        max_steps = int(self.cfg.engine.iterations)
        scheduler_cfg = self.model_cfg.gauss_container.scheduler
        for name, spec in self.container._specs.items():
            lr0 = float(spec.lr)
            if name == "means":
                lr_final = lr0 * float(scheduler_cfg.position_lr_final_scale)
                delay_mult = float(scheduler_cfg.position_lr_delay_mult)
            else:
                lr_final = lr0 * float(scheduler_cfg.default_lr_final_scale)
                delay_mult = 1.0
            self.lr_mgr.register(
                name,
                get_expon_lr_func(
                    lr_init=lr0,
                    lr_final=lr_final,
                    lr_delay_mult=delay_mult,
                    max_steps=max_steps,
                ),
            )
