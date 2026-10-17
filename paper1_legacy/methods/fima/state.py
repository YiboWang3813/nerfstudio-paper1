from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from engine.registry import register_state
from engine.state import State

from scene.scene import Scene

from .configs import FiMAConfig
from .sequence_dataset import FiMASequenceSampler


@register_state("FiMAOfflineState")
class FiMAOfflineState(State):
    scene: Optional[Scene] = None
    model_cfg: Optional[FiMAConfig] = None

    pose_estimator: Optional[nn.Module] = None
    pem_optimizer: Optional[torch.optim.Optimizer] = None
    pem_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    pose_refiner: Optional[torch.nn.Module] = None
    pose_refiner_optimizer: Optional[torch.optim.Optimizer] = None

    sequence_sampler: Optional[FiMASequenceSampler] = None
    rng_np: Optional[np.random.RandomState] = None
    pem_target_mean: Optional[torch.Tensor] = None
    pem_target_std: Optional[torch.Tensor] = None
