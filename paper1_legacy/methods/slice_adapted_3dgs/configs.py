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
    min_covariance: float = 1.0e-4
    normalize_by_footprint: bool = True
    clamp_output: bool = True
    bg_color: float = 0.0
    output_grayscale: bool = True
    debug: bool = False


@dataclass
class GaussContainerConfig:
    @dataclass
    class LRConfig:
        means: float = 1.6e-4
        sh_dc: float = 2.5e-3
        sh_rest: float = 1.25e-4
        opacities: float = 5.0e-2
        scales_3d: float = 5.0e-3
        quats: float = 1.0e-3

    @dataclass
    class InitValConfig:
        opacity: float = 0.1
        scale_3d: Tuple[float, float, float] = (0.3, 0.3, 0.3)
        sh_dc: float = 0.0

    @dataclass
    class SchedulerConfig:
        position_lr_final_scale: float = 0.01
        default_lr_final_scale: float = 0.1
        position_lr_delay_mult: float = 0.01

    @dataclass
    class ConstraintsConfig:
        opacity_max: float = 0.0
        min_scale_3d: float = 1.0e-3
        max_scale_3d: float = 0.0

    lr: LRConfig = field(default_factory=LRConfig)
    init_val: InitValConfig = field(default_factory=InitValConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)


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
            opacity_reset: bool = False

        @dataclass
        class TimingConfig:
            start_iter: int = 1000
            end_iter: int = 25000
            freq: int = 2000
            opacity_reset_interval: int = 3000

        @dataclass
        class ThresholdsConfig:
            densify_grad: float = 1.0e-5
            split_scale_mm: float = 0.6
            min_scale_mm: float = 5.0e-5
            max_scale_mm: float = 5.0
            prune_opacity: float = 1.0e-2

        enable: EnableConfig = field(default_factory=EnableConfig)
        timing: TimingConfig = field(default_factory=TimingConfig)
        thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
        n_max: int = 2_000_000

    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass
class ModelConfig:
    name: str = "slice_adapted_3dgs"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
