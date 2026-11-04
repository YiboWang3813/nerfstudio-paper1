"""Serializable native method dataclasses; physical values retain source units."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RenderConfig:
    backend: str = "torch_reference"
    camera_mode: str = "slice_plane_physical"
    physical_camera_distance_mm: float = 200.0
    slice_thickness_mm: Optional[float] = 0.3
    slice_thickness_ratio: Optional[float] = None
    thickness_scale: float = 1.0
    cull_std: float = 3.0
    max_points_per_chunk: int = 1024
    max_pixels_per_chunk: int = 8192
    min_covariance: float = 0.0001
    normalize_by_footprint: bool = True
    clamp_output: bool = True
    bg_color: float = 0.0
    output_grayscale: bool = True
    debug: bool = False


@dataclass
class GaussContainerConfig_LRConfig:
    means: float = 0.00016
    sh_dc: float = 0.0025
    sh_rest: float = 0.000125
    opacities: float = 0.05
    scales_3d: float = 0.005
    quats: float = 0.001


@dataclass
class GaussContainerConfig_InitValConfig:
    opacity: float = 0.1
    scale_3d: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    sh_dc: float = 0.0


@dataclass
class GaussContainerConfig_SchedulerConfig:
    position_lr_final_scale: float = 0.01
    default_lr_final_scale: float = 0.1
    position_lr_delay_mult: float = 0.01


@dataclass
class GaussContainerConfig_ConstraintsConfig:
    opacity_max: float = 0.0
    min_scale_3d: float = 0.001
    max_scale_3d: float = 0.0


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
    densify: bool = True
    prune: bool = True
    opacity_reset: bool = False


@dataclass
class OfflineConfig_DensityControlConfig_TimingConfig:
    start_iter: int = 1000
    end_iter: int = 25000
    freq: int = 2000
    opacity_reset_interval: int = 3000


@dataclass
class OfflineConfig_DensityControlConfig_ThresholdsConfig:
    densify_grad: float = 1e-05
    split_scale_mm: float = 0.6
    min_scale_mm: float = 5e-05
    max_scale_mm: float = 5.0
    prune_opacity: float = 0.01


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
    n_max: int = 2000000


@dataclass
class OfflineConfig:
    density_control: OfflineConfig_DensityControlConfig = field(default_factory=OfflineConfig_DensityControlConfig)
    diagnostics: OfflineConfig_DiagnosticsConfig = field(default_factory=OfflineConfig_DiagnosticsConfig)


@dataclass
class ModelConfig:
    name: str = "slice_adapted_3dgs"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
