from dataclasses import dataclass, field
from typing import Any
from omegaconf import MISSING

# 从base拿到基础Config
from .base.scene import SceneConfig
from .base.engine import EngineConfig
from .base.system import SystemConfig, WandbConfig
from .base.loss import LossConfig
from .base.metric import MetricConfig

@dataclass
class ExperimentConfig:
    family: str = "default"
    name: str = "default"
    protocol: str = ""
    description: str = ""
    sensorless_required: bool = False


# ================= 4. Root Config =================
@dataclass
class RootConfig:
    task: str = "train"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    model: Any = MISSING
