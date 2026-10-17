from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict

from configs.root import RootConfig
from engine.computer import LossComputer, MetricComputer

@dataclass
class State:
    cfg: RootConfig
    iteration: int = 0
    device: Any = None

    # 运行时字典（方法侧约定键名；loss/metric 统一读取）
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # 通用组件
    data_generator: Any = None
    model: Any = None
    optimizer: Any = None

    # 核心组件（通用、与具体方法解耦）
    loss_computer: Optional[LossComputer] = None
    metric_computer: Optional[MetricComputer] = None
    # 辅助组件
    scalar_manager: Any = None
    io_manager: Any = None


    ckpt_items: List = field(default_factory=list)
    stop_training: bool = False