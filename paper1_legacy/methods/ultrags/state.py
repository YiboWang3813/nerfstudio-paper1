from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from engine.state import State
from engine.registry import register_state

from scene.scene import Scene
from .config import ModelConfig
from .gauss_container import UltraGSGaussContainer
from .gauss_solver import UltraGSGaussSolver
from .density_control import UltraGSOfflineDensityController

if TYPE_CHECKING:
    from .density_control import UltraGSOnlineDensityController
from .local_window_manager import LocalWindowManager
from .gaussian_state_manager import GaussianStateManager
from .pose_refiner import PoseRefiner


@register_state("UltraGSOfflineState")
class UltraGSOfflineState(State):
    scene: Scene = None
    gauss_container: UltraGSGaussContainer = None
    gauss_solver: UltraGSGaussSolver = None
    dc: UltraGSOfflineDensityController = None


@register_state("UltraGSOnlineState")
class UltraGSOnlineState(State):
    """
    Online Streaming State:
    - ``model_cfg``：与 ``gauss_container.model_cfg`` 同步，插件直接读配置无需经 container
    - ``inputs/outputs``：统一运行时字典；loss/metric 计算与在线帧 scratch 均写入其中
    - ``stream_*``：当前 stream 迭代的数据上下文（索引、slice、增量高斯）
    - ``lwm`` / ``gsm`` / ``pose_carryover``：窗格预算、membership、跨帧位姿缓存
    """
    @dataclass
    class OnlinePoseCarryover:
        """跨帧保留的位姿量（供下一帧 pose loss 与先验）；每帧插件末尾更新。
        它们是跨帧要传递的 不是只在当前帧使用的"""

        P_refined_prev: Optional[torch.Tensor] = None
        P_refined_prevprev: Optional[torch.Tensor] = None
        P_coarse_prev: Optional[torch.Tensor] = None
        pose_refine_loss: float = 0.0
        #: 模式 B：每全局帧 id 一次采样的粗姿态（GT+Noise）
        coarse_by_frame: Dict[int, Any] = field(default_factory=dict)

    # 方法专用配置
    model_cfg: Optional[ModelConfig] = None

    # 组件
    scene: Scene = None
    gauss_container: UltraGSGaussContainer = None
    gauss_solver: UltraGSGaussSolver = None
    dc: Optional["UltraGSOnlineDensityController"] = None
    lwm: LocalWindowManager = None
    gsm: GaussianStateManager = None
    pose_estimator: Any = None
    pose_refiner: PoseRefiner = None
    pose_refiner_optimizer: Any = None

    # 跨帧运行时；子类非 dataclass 时 field() 无效，在 Online setup 里 new OnlinePoseCarryover()
    pose_carryover: Optional[OnlinePoseCarryover] = None

    # 当前帧运行时
    #: 当前 stream 帧的全局 slice 索引（与 scheduler 一致）
    frame_idx: int = 0
    stream_frame_id: int = 0
    stream_slice_meta: Any = None
    stream_new_gaussians: Optional[dict] = None
    #: 已完成的 stream 帧数（0-based，用于 seed_every / 日志取模）
    stream_frame_count: int = 0
