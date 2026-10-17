import torch

from .configs import ModelConfig
from .field import UltraNeRFField


class UltraNeRFSolver:
    def __init__(self, cfg: ModelConfig, field: UltraNeRFField):
        self.cfg = cfg
        self.field = field
        self.optimizer = None
        self.scheduler = None

    def setup(self) -> None:
        opt_name = str(self.cfg.solver.optimizer).lower()
        if opt_name != "adam":
            raise ValueError(f"Unsupported UltraNeRF optimizer '{self.cfg.solver.optimizer}'")
        self.optimizer = torch.optim.Adam(self.field.parameters(), lr=float(self.cfg.solver.lr))
        decay_steps = max(1, int(self.cfg.solver.lr_decay_steps))
        decay_rate = float(self.cfg.solver.lr_decay_rate)
        gamma = decay_rate ** (1.0 / float(decay_steps))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def step_lr(self, _iteration: int) -> None:
        self.scheduler.step()
