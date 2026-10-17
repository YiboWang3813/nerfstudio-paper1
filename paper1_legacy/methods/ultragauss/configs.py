from dataclasses import dataclass, field


@dataclass
class RenderConfig:
    chi2_threshold: float = 7.815
    mapper_sort_by_depth: bool = True
    fixed_bg_alpha: float = 1e-6
    fixed_bg_intensity: float = 0.0
    diag_beta: float = 1e-4
    jitter_radius_ratio: float = 0.25
    enable_shadow: bool = False
    shadow_delta: float = 0.0


@dataclass
class GaussContainerConfig:
    @dataclass
    class LRConfig:
        means: float = 1.6e-4
        precision_tril: float = 5e-2
        opacities: float = 5e-2
        intensities: float = 5e-2

    @dataclass
    class InitValConfig:
        opacity: float = 0.731
        intensity: float = 0.5
        diag_raw_min: float = 4.0
        diag_raw_max: float = 5.0
        offdiag_std: float = 0.01

    lr: LRConfig = field(default_factory=LRConfig)
    init_val: InitValConfig = field(default_factory=InitValConfig)


@dataclass
class OfflineConfig:
    @dataclass
    class DiagnosticsConfig:
        enable_render_stats: bool = True
        train_step: int = 100

    @dataclass
    class DensityControlConfig:
        @dataclass
        class EnableConfig:
            densify: bool = True
            prune: bool = True

        @dataclass
        class TimingConfig:
            start_iter: int = 1000
            end_iter: int = 25000
            freq: int = 2000

        @dataclass
        class ThresholdsConfig:
            densify_grad: float = 1e-5
            split_scale_mm: float = 0.6
            min_scale_mm: float = 5e-5
            max_scale_mm: float = 5.0
            prune_opacity: float = 1e-2
            prune_intensity: float = 1e-2

        enable: EnableConfig = field(default_factory=EnableConfig)
        timing: TimingConfig = field(default_factory=TimingConfig)
        thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
        n_max: int = 2_000_000

    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass
class ModelConfig:
    name: str = "ultragauss"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
