from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricPSNRConfig:
    max_val: float = 1.0


@dataclass
class MetricConfig:
    # 默认不启用任何指标；由具体方法/实验配置声明 needed_metrics。
    needed_metrics: List[str] = field(default_factory=list)
    psnr_cfg: MetricPSNRConfig = field(default_factory=MetricPSNRConfig)
    max_vis_frames: int = 10
    save_per_frame_metrics: bool = True
