"""Serializable native method dataclasses; physical values retain source units."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RenderConfig:
    k_sigma: float = 3.0
    elev_cutoff_margin_mm: float = 0.0
    ops_max_offset_mm: float = 2.0
    ops_train_only: bool = True
    quadrature_window_radius_mm: float = 2.0
    local_transmission_window_only: bool = False
    local_transmission_max_window_mm: float = 17.0
    elev_beam_width_mm: float = 2.0


@dataclass
class GaussContainerConfig_LRConfig:
    means: float = 0.0001
    scales: float = 0.005
    quats: float = 0.005
    attenuations: float = 0.0005
    sh_dc: float = 0.005
    sh_rest: float = 1e-05


@dataclass
class GaussContainerConfig_InitValConfig:
    tau: float = 0.99
    scale_mm: float = 0.5
    sh_dc: float = 0.1


@dataclass
class GaussContainerConfig:
    lr: GaussContainerConfig_LRConfig = field(default_factory=GaussContainerConfig_LRConfig)
    init_val: GaussContainerConfig_InitValConfig = field(default_factory=GaussContainerConfig_InitValConfig)


@dataclass
class OfflineConfig_DensityControlConfig_EnableConfig:
    densify: bool = True
    prune: bool = True


@dataclass
class OfflineConfig_DensityControlConfig_TimingConfig:
    start_iter: int = 1000
    end_iter: int = 20000
    freq: int = 2500


@dataclass
class OfflineConfig_DensityControlConfig_ThresholdsConfig:
    densify_grad: float = 1e-05
    split_scale_mm: float = 0.5
    min_scale_mm: float = 5e-05
    max_scale_mm: float = 5.0


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
    n_max: Optional[int] = None


@dataclass
class OfflineConfig:
    density_control: OfflineConfig_DensityControlConfig = field(default_factory=OfflineConfig_DensityControlConfig)
    sh_degree1_start_iter: int = 1000


@dataclass
class ModelConfig:
    name: str = "ultragray"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
