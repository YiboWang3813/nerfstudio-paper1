from dataclasses import dataclass, field
from typing import List


@dataclass
class LossSSIMConfig:
    window_size: int = 11
    sigma: float = 1.5
    downsample_enabled: bool = False
    downsample_factor: int = 1
    downsample_mode: str = "area"
    clamp_inputs: bool = False


@dataclass
class LossL1Config:
    clamp_inputs: bool = False


@dataclass
class LossMarcoSSIMConfig:
    blur_window_size: int = 21
    blur_sigma: float = 3.0
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5


@dataclass
class LossLPIPSConfig:
    metric_name: str = "lpips"


@dataclass
class LossHighPassL1Config:
    kernel_size: int = 9
    sigma: float = 2.0


@dataclass
class LossScaleWallConfig:
    min_scales_mm: float = 0.1
    max_scales_mm: float = 1.0


@dataclass
class LossScaleAnisoConfig:
    max_aniso_ratio: float = 5.0


@dataclass
class LossOpacityConfig:
    target_opacity: float = 0.5


@dataclass
class LossAttenuationTailConfig:
    max_attenuation: float = 0.05


@dataclass
class LossPoseAnchorConfig:
    rot_weight: float = 1.0
    trans_weight: float = 1.0


@dataclass
class LossPosePriorConfig:
    rot_weight: float = 1.0
    trans_weight: float = 1.0


@dataclass
class LossPoseSmoothConfig:
    rot_weight: float = 1.0
    trans_weight: float = 1.0


@dataclass
class LossPoseAccConfig:
    rot_weight: float = 1.0
    trans_weight: float = 1.0


@dataclass
class LossPoseAuxRecConfig:
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    window_size: int = 11
    sigma: float = 1.5


@dataclass
class LossPoseRegistrationConfig:
    photo_weight: float = 1.0
    grad_weight: float = 0.2
    charbonnier_eps: float = 1e-3


@dataclass
class LossPoseNeighborRegistrationConfig:
    photo_weight: float = 1.0
    ssim_weight: float = 0.2
    charbonnier_eps: float = 1e-3
    window_size: int = 11
    sigma: float = 1.5
    min_overlap_ratio: float = 0.3


@dataclass
class LossPoseCycleConfig:
    photo_weight: float = 1.0
    ssim_weight: float = 0.1
    charbonnier_eps: float = 1e-3
    window_size: int = 11
    sigma: float = 1.5
    min_overlap_ratio: float = 0.3


@dataclass
class LossPoseConstVelConfig:
    rot_weight: float = 1.0
    trans_weight: float = 1.0


@dataclass
class LossPoseGtDeltaConfig:
    rot_weight: float = 1.0
    trans_weight: float = 1.0


@dataclass
class LossPoseOccConfConfig:
    charbonnier_epsilon: float = 1e-3
    confidence_gamma: float = 1.0
    confidence_min: float = 0.05
    detach_confidence: bool = True


@dataclass
class LossLambdaConfig:
    l1: float = 0.0
    mse: float = 0.0
    ssim: float = 0.0
    sobel: float = 0.0
    highpass_l1: float = 0.0
    marco_ssim: float = 0.0
    lpips: float = 0.0
    diff_tv: float = 0.0
    scale_wall: float = 0.0
    scale_aniso: float = 0.0
    opacity: float = 0.0
    attenuation_l1: float = 0.0
    attenuation_tail: float = 0.0
    pose_anchor: float = 0.0
    pose_prior: float = 0.0
    pose_smooth: float = 0.0
    pose_acc: float = 0.0
    pose_aux_rec: float = 0.0
    pose_registration: float = 0.0
    pose_neighbor_registration: float = 0.0
    pose_cycle: float = 0.0
    pose_const_vel: float = 0.0
    pose_gt_delta: float = 0.0
    pose_occ_conf: float = 0.0


@dataclass
class LossConfig:
    # 默认不启用任何损失；由具体方法/实验配置声明 needed_losses 与 lambdas。
    needed_losses: List[str] = field(default_factory=list)
    lambdas: LossLambdaConfig = field(default_factory=LossLambdaConfig)

    l1_cfg: LossL1Config = field(default_factory=LossL1Config)
    ssim_cfg: LossSSIMConfig = field(default_factory=LossSSIMConfig)
    highpass_l1_cfg: LossHighPassL1Config = field(default_factory=LossHighPassL1Config)
    marco_ssim_cfg: LossMarcoSSIMConfig = field(default_factory=LossMarcoSSIMConfig)
    lpips_cfg: LossLPIPSConfig = field(default_factory=LossLPIPSConfig)
    scale_wall_cfg: LossScaleWallConfig = field(default_factory=LossScaleWallConfig)
    scale_aniso_cfg: LossScaleAnisoConfig = field(default_factory=LossScaleAnisoConfig)
    opacity_cfg: LossOpacityConfig = field(default_factory=LossOpacityConfig)
    attenuation_tail_cfg: LossAttenuationTailConfig = field(default_factory=LossAttenuationTailConfig)
    pose_anchor_cfg: LossPoseAnchorConfig = field(default_factory=LossPoseAnchorConfig)
    pose_prior_cfg: LossPosePriorConfig = field(default_factory=LossPosePriorConfig)
    pose_smooth_cfg: LossPoseSmoothConfig = field(default_factory=LossPoseSmoothConfig)
    pose_acc_cfg: LossPoseAccConfig = field(default_factory=LossPoseAccConfig)
    pose_aux_rec_cfg: LossPoseAuxRecConfig = field(default_factory=LossPoseAuxRecConfig)
    pose_registration_cfg: LossPoseRegistrationConfig = field(default_factory=LossPoseRegistrationConfig)
    pose_neighbor_registration_cfg: LossPoseNeighborRegistrationConfig = field(
        default_factory=LossPoseNeighborRegistrationConfig
    )
    pose_cycle_cfg: LossPoseCycleConfig = field(default_factory=LossPoseCycleConfig)
    pose_const_vel_cfg: LossPoseConstVelConfig = field(default_factory=LossPoseConstVelConfig)
    pose_gt_delta_cfg: LossPoseGtDeltaConfig = field(default_factory=LossPoseGtDeltaConfig)
    pose_occ_conf_cfg: LossPoseOccConfConfig = field(default_factory=LossPoseOccConfConfig)
