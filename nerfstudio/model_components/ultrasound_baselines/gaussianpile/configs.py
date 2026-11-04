"""Serializable native method dataclasses; physical values retain source units."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RenderConfig:
    enable_input_checks: bool = False


@dataclass
class GaussInitConfig:
    num_points: int = 1000000


@dataclass
class GaussianPileSliceRenderConfig:
    backend: str = "ours_focus"
    tile_size_px: int = 16
    radius_sigma: float = 3.0
    sigma_z_mm: Optional[float] = None
    sigma_z_ratio: Optional[float] = None
    prefilter_elevational_radius_mm: float = 0.0
    det_eps: float = 1e-06
    cov_eps: float = 1e-06
    opacity_cull_threshold: float = 1e-06
    render_chunk_size: int = 250000


@dataclass
class GaussContainerConfig_LRConfig:
    means: float = 0.0006
    scales: float = 0.002
    quats: float = 0.001
    opacities: float = 0.02


@dataclass
class GaussContainerConfig_InitValConfig:
    opacities: float = 0.1


@dataclass
class GaussContainerConfig_ConstraintConfig:
    scale_activation: str = "bounded_sigmoid"
    scale_min_mm: float = 0.05
    scale_max_mm: float = 3.0
    opacity_max: float = 1.0


@dataclass
class GaussContainerConfig:
    lr: GaussContainerConfig_LRConfig = field(default_factory=GaussContainerConfig_LRConfig)
    lr_final_mult: float = 0.1
    init_val: GaussContainerConfig_InitValConfig = field(default_factory=GaussContainerConfig_InitValConfig)
    constraints: GaussContainerConfig_ConstraintConfig = field(default_factory=GaussContainerConfig_ConstraintConfig)


@dataclass
class OfflineConfig_DiagnosticsConfig:
    enable_render_stats: bool = True
    train_step: int = 100
    all_black_eps: float = 1e-06
    warn_on_all_black_test: bool = True
    fail_on_all_black_test: bool = False
    restore_best_on_finish: bool = True


@dataclass
class OfflineConfig_DensityControlConfig_EnableConfig:
    densify: bool = True
    prune: bool = True


@dataclass
class OfflineConfig_DensityControlConfig_TimingConfig:
    start_iter: int = 500
    end_iter: int = 25000
    freq: int = 500
    grad_accum_interval: int = 1


@dataclass
class OfflineConfig_DensityControlConfig_ThresholdsConfig:
    densify_grad: float = 5e-05
    split_scale: float = 0.6
    min_scale_mm: float = 0.01
    prune_opacity: float = 0.02
    prune_observation_score: float = 0.25
    min_prune_observations: int = 2
    max_prune_fraction_per_step: float = 0.0
    min_keep_ratio: float = 0.0


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
    name: str = "gaussianpile"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_init: GaussInitConfig = field(default_factory=GaussInitConfig)
    slice_render: GaussianPileSliceRenderConfig = field(default_factory=GaussianPileSliceRenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
