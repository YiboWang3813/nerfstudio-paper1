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
class GaussContainerConfig:
    @dataclass
    class LRConfig:
        means: float = 1e-4
        scales: float = 5e-3
        quats: float = 5e-3
        attenuations: float = 5e-4
        sh_dc: float = 5e-3
        sh_rest: float = 1e-5

    @dataclass
    class InitValConfig:
        tau: float = 0.99
        scale_mm: float = 0.5
        sh_dc: float = 0.1

    lr: LRConfig = field(default_factory=LRConfig)
    init_val: InitValConfig = field(default_factory=InitValConfig)


@dataclass
class OfflineConfig:
    @dataclass
    class DensityControlConfig:
        @dataclass
        class EnableConfig:
            densify: bool = True
            prune: bool = True

        @dataclass
        class TimingConfig:
            start_iter: int = 1000
            end_iter: int = 20000
            freq: int = 2500

        @dataclass
        class ThresholdsConfig:
            densify_grad: float = 1e-5
            split_scale_mm: float = 0.5
            min_scale_mm: float = 5e-5
            max_scale_mm: float = 5.0

        enable: EnableConfig = field(default_factory=EnableConfig)
        timing: TimingConfig = field(default_factory=TimingConfig)
        thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
        n_max: Optional[int] = None

    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)
    sh_degree1_start_iter: int = 1000


@dataclass
class ModelConfig:
    name: str = "ultragray"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
