from configs.root import RootConfig
from gaussian.gauss_solver import GaussSolver
from gaussian.lr_scheduler_manager import get_expon_lr_func

from .configs import ModelConfig
from .gaussian_container import GaussianPileGaussContainer


class GaussianPileGaussSolver(GaussSolver):
    def __init__(self, cfg: RootConfig, container: GaussianPileGaussContainer):
        super().__init__(container)
        self.cfg = cfg

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def _setup_lr_scheduler(self):
        max_steps = int(self.cfg.engine.iterations)
        final_mult = float(self.model_cfg.gauss_container.lr_final_mult)
        lr_cfg = self.model_cfg.gauss_container.lr
        for name in ("means", "scales", "quats", "opacities"):
            lr = float(getattr(lr_cfg, name))
            self.lr_mgr.register(
                name,
                get_expon_lr_func(lr_init=lr, lr_final=lr * final_mult, max_steps=max_steps),
            )
