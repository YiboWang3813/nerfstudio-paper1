from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RenderConfig:
    enable_input_checks: bool = False


@dataclass
class GaussInitConfig:
    num_points: int = 1_000_000


@dataclass
class GaussianPileSliceRenderConfig:
    backend: str = "ours_focus"
    tile_size_px: int = 16
    radius_sigma: float = 3.0
    sigma_z_mm: Optional[float] = None
    sigma_z_ratio: Optional[float] = None
    prefilter_elevational_radius_mm: float = 0.0
    det_eps: float = 1.0e-6
    cov_eps: float = 1.0e-6
    opacity_cull_threshold: float = 1.0e-6
    render_chunk_size: int = 250_000


@dataclass
class GaussContainerConfig:
    @dataclass
    class LRConfig:
        means: float = 6.0e-4
        scales: float = 2.0e-3
        quats: float = 1.0e-3
        opacities: float = 2.0e-2

    @dataclass
    class InitValConfig:
        opacities: float = 0.1

    @dataclass
    class ConstraintConfig:
        scale_activation: str = "bounded_sigmoid"
        scale_min_mm: float = 0.05
        scale_max_mm: float = 3.0
        opacity_max: float = 1.0

    lr: LRConfig = field(default_factory=LRConfig)
    lr_final_mult: float = 0.1
    init_val: InitValConfig = field(default_factory=InitValConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)


@dataclass
class OfflineConfig:
    @dataclass
    class DiagnosticsConfig:
        enable_render_stats: bool = True
        train_step: int = 100
        all_black_eps: float = 1.0e-6
        warn_on_all_black_test: bool = True
        fail_on_all_black_test: bool = False
        restore_best_on_finish: bool = True

    @dataclass
    class DensityControlConfig:
        @dataclass
        class EnableConfig:
            densify: bool = True
            prune: bool = True

        @dataclass
        class TimingConfig:
            start_iter: int = 500
            end_iter: int = 25_000
            freq: int = 500
            grad_accum_interval: int = 1

        @dataclass
        class ThresholdsConfig:
            densify_grad: float = 5.0e-5
            split_scale: float = 0.6
            min_scale_mm: float = 0.01
            prune_opacity: float = 0.02
            prune_observation_score: float = 0.25
            min_prune_observations: int = 2
            max_prune_fraction_per_step: float = 0.0
            min_keep_ratio: float = 0.0

        enable: EnableConfig = field(default_factory=EnableConfig)
        timing: TimingConfig = field(default_factory=TimingConfig)
        thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
        n_max: int = 2_000_000

    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass
class ModelConfig:
    name: str = "gaussianpile"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_init: GaussInitConfig = field(default_factory=GaussInitConfig)
    slice_render: GaussianPileSliceRenderConfig = field(default_factory=GaussianPileSliceRenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
