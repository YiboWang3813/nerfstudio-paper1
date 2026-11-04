"""Serializable native method dataclasses; physical values retain source units."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RenderConfig:
    sh_degree: int = 2
    active_sh_degree_start: int = 0
    sh_degree_step_iter: int = 1000
    camera_mode: str = "released_pose_fov"
    physical_camera_distance_mm: float = 200.0
    dar_mode: str = "released_disabled"
    fov_x_deg: float = 60.0
    fov_y_deg: float = 60.0
    fov_x_min_deg: float = 20.0
    fov_x_max_deg: float = 140.0
    fov_y_min_deg: float = 20.0
    fov_y_max_deg: float = 140.0
    znear: float = 0.01
    zfar: float = 1000.0
    bg_color: float = 0.0
    scale_modifier: float = 1.0
    enable_slice_cull: bool = True
    slice_thickness_ratio: float = 0.5
    slice_thickness_mm: Optional[float] = None
    slice_cull_std: float = 3.0
    slice_cull_scale_multiplier: float = 3.0
    opacity_slice_attenuation: bool = True
    debug: bool = False
    output_grayscale: bool = True


@dataclass
class GaussContainerConfig_LRConfig:
    means: float = 0.00016
    sh_dc: float = 0.0025
    sh_rest: float = 0.000125
    opacities: float = 0.05
    scales_2d: float = 0.005
    quats: float = 0.001
    fov_x: float = 0.001
    fov_y: float = 0.001


@dataclass
class GaussContainerConfig_InitValConfig:
    opacity: float = 0.1
    scale_2d: Tuple[float, float] = (0.3, 0.3)
    sh_dc: float = 0.0
    fov_x: float = 0.0
    fov_y: float = 0.0


@dataclass
class GaussContainerConfig_SchedulerConfig:
    position_lr_final_scale: float = 0.01
    default_lr_final_scale: float = 0.1
    position_lr_delay_mult: float = 0.01


@dataclass
class GaussContainerConfig_ConstraintsConfig:
    opacity_max: float = 0.0


@dataclass
class GaussContainerConfig:
    lr: GaussContainerConfig_LRConfig = field(default_factory=GaussContainerConfig_LRConfig)
    init_val: GaussContainerConfig_InitValConfig = field(default_factory=GaussContainerConfig_InitValConfig)
    scheduler: GaussContainerConfig_SchedulerConfig = field(default_factory=GaussContainerConfig_SchedulerConfig)
    constraints: GaussContainerConfig_ConstraintsConfig = field(default_factory=GaussContainerConfig_ConstraintsConfig)


@dataclass
class OfflineConfig_DiagnosticsConfig:
    enable_render_stats: bool = True
    train_step: int = 100


@dataclass
class OfflineConfig_DensityControlConfig_EnableConfig:
    densify: bool = False
    prune: bool = False
    opacity_reset: bool = False


@dataclass
class OfflineConfig_DensityControlConfig_TimingConfig:
    start_iter: int = 500
    end_iter: int = 30000
    freq: int = 0
    densification_interval: int = 100
    opacity_reset_interval: int = 3000


@dataclass
class OfflineConfig_DensityControlConfig_ThresholdsConfig:
    densify_grad: float = 0.0002
    opacity_cull: float = 0.05
    percent_dense: float = 0.01
    n_max: Optional[int] = None
    min_scale_2d: float = 0.0
    max_scale_2d: float = 0.0


@dataclass
class OfflineConfig_DensityControlConfig:
    enable: OfflineConfig_DensityControlConfig_EnableConfig = field(
        default_factory=OfflineConfig_DensityControlConfig_EnableConfig
    )
    timing: OfflineConfig_DensityControlConfig_TimingConfig = field(
        default_factory=OfflineConfig_DensityControlConfig_TimingConfig
    )
    thresholds: OfflineConfig_DensityControlConfig_ThresholdsConfig = field(
        default_factory=OfflineConfig_DensityControlConfig_ThresholdsConfig
    )


@dataclass
class OfflineConfig:
    density_control: OfflineConfig_DensityControlConfig = field(default_factory=OfflineConfig_DensityControlConfig)
    diagnostics: OfflineConfig_DiagnosticsConfig = field(default_factory=OfflineConfig_DiagnosticsConfig)


@dataclass
class ModelConfig:
    name: str = "ultrags_baseline"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
