"""Serializable native method dataclasses; physical values retain source units."""

from dataclasses import dataclass, field


@dataclass
class RenderConfig:
    chi2_threshold: float = 7.815
    mapper_sort_by_depth: bool = True
    fixed_bg_alpha: float = 1e-06
    fixed_bg_intensity: float = 0.0
    diag_beta: float = 0.0001
    jitter_radius_ratio: float = 0.25
    enable_shadow: bool = False
    shadow_delta: float = 0.0


@dataclass
class GaussContainerConfig_LRConfig:
    means: float = 0.00016
    precision_tril: float = 0.05
    opacities: float = 0.05
    intensities: float = 0.05


@dataclass
class GaussContainerConfig_InitValConfig:
    opacity: float = 0.731
    intensity: float = 0.5
    diag_raw_min: float = 4.0
    diag_raw_max: float = 5.0
    offdiag_std: float = 0.01


@dataclass
class GaussContainerConfig:
    lr: GaussContainerConfig_LRConfig = field(default_factory=GaussContainerConfig_LRConfig)
    init_val: GaussContainerConfig_InitValConfig = field(default_factory=GaussContainerConfig_InitValConfig)


@dataclass
class OfflineConfig_DiagnosticsConfig:
    enable_render_stats: bool = True
    train_step: int = 100


@dataclass
class OfflineConfig_DensityControlConfig_EnableConfig:
    densify: bool = True
    prune: bool = True


@dataclass
class OfflineConfig_DensityControlConfig_TimingConfig:
    start_iter: int = 1000
    end_iter: int = 25000
    freq: int = 2000


@dataclass
class OfflineConfig_DensityControlConfig_ThresholdsConfig:
    densify_grad: float = 1e-05
    split_scale_mm: float = 0.6
    min_scale_mm: float = 5e-05
    max_scale_mm: float = 5.0
    prune_opacity: float = 0.01
    prune_intensity: float = 0.01


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
    name: str = "ultragauss"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
