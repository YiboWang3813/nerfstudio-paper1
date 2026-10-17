from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING


@dataclass
class TrainIntervalConfig:
    step: int = 20   # 每隔多少步记一次 log
    print: int = 100 # 每隔多少步打印一次 console


@dataclass
class IntervalConfig:
    train: TrainIntervalConfig = field(default_factory=TrainIntervalConfig)
    debug: int = 1000
    test: int = 2000
    save: int = 5000


@dataclass
class EngineConfig:
    name: str = "default_engine"
    state: str = MISSING  # STATE_REGISTRY 中的名字，如 "UltraGSOfflineState"
    task: str = "train"   # train | test
    stream: str = "offline" # offline | online

    iterations: int = 30000
    accum_steps: int = 1
    resume: str = ""

    #: Online stream（train）下，在「本帧标准训练」之前用当前高斯 + 该帧 GT 位姿渲染并打测试指标；
    #: 仅在 ``stream_eval_interval_frames > 0`` 时启用。合法帧区间为 [skip_head, dataset_len - skip_tail)。
    stream_eval_skip_head_frames: int = 0
    stream_eval_skip_tail_frames: int = 0
    stream_eval_interval_frames: int = 0  # 0=关闭；>0 时从 skip_head 起每隔该间隔测一帧

    needed_callbacks: List[str] = field(default_factory=list)  # CALLBACK_REGISTRY 的 key 列表
    interval: IntervalConfig = field(default_factory=IntervalConfig)
    