from configs.root import RootConfig
from gaussian.gauss_solver import GaussSolver
from gaussian.lr_scheduler_manager import get_expon_lr_func

from .configs import ModelConfig
from .gaussian_container import OursGaussContainer


class OursGaussSolver(GaussSolver):
    def __init__(self, cfg: RootConfig, container: OursGaussContainer):
        super().__init__(container)
        self.cfg = cfg

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def _setup_lr_scheduler(self):
        if self.cfg.engine.stream == "online":
            return
        means_lr = self.model_cfg.gauss_container.lr.means
        means_scheduler = get_expon_lr_func(
            lr_init=means_lr,
            lr_final=means_lr * 0.1,
            max_steps=int(self.cfg.engine.iterations),
        )
        self.lr_mgr.register("means", means_scheduler)
