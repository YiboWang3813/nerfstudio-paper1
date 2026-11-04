"""Native scheduler preserving the ultrasound loop's one-based update timing."""

from dataclasses import dataclass, field
from typing import Type

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig


@dataclass
class UltrasoundExponentialSchedulerConfig(SchedulerConfig):
    _target: Type = field(default_factory=lambda: UltrasoundExponentialScheduler)
    final_factor: float = 0.1
    max_steps: int = 30000


class UltrasoundExponentialScheduler(Scheduler):
    def get_scheduler(self, optimizer, lr_init):
        cfg = self.config
        if cfg.max_steps <= 0 or cfg.final_factor <= 0 or lr_init <= 0:
            raise ValueError("Invalid ultrasound exponential schedule.")

        def factor(step):
            t = np.clip((step + 1) / cfg.max_steps, 0, 1)
            return float(np.exp(np.log(lr_init) * (1 - t) + np.log(lr_init * cfg.final_factor) * t) / lr_init)

        return LambdaLR(optimizer, factor)
