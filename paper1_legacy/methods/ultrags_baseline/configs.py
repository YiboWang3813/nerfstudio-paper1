from dataclasses import dataclass, field
from typing import Optional
from typing import Tuple


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
class GaussContainerConfig:
    @dataclass
    class LRConfig:
        means: float = 1.6e-4
        sh_dc: float = 2.5e-3
        sh_rest: float = 1.25e-4
        opacities: float = 5.0e-2
        scales_2d: float = 5.0e-3
        quats: float = 1.0e-3
        fov_x: float = 1.0e-3
        fov_y: float = 1.0e-3

    @dataclass
    class InitValConfig:
        opacity: float = 0.1
        scale_2d: Tuple[float, float] = (0.3, 0.3)
        sh_dc: float = 0.0
        fov_x: float = 0.0
        fov_y: float = 0.0

    @dataclass
    class SchedulerConfig:
        position_lr_final_scale: float = 0.01
        default_lr_final_scale: float = 0.1
        position_lr_delay_mult: float = 0.01

    @dataclass
    class ConstraintsConfig:
        opacity_max: float = 0.0

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
            densify: bool = False
            prune: bool = False
            opacity_reset: bool = False

        @dataclass
        class TimingConfig:
            start_iter: int = 500
            end_iter: int = 30000
            freq: int = 0
            densification_interval: int = 100
            opacity_reset_interval: int = 3000

        @dataclass
        class ThresholdsConfig:
            densify_grad: float = 2.0e-4
            opacity_cull: float = 5.0e-2
            percent_dense: float = 0.01
            n_max: Optional[int] = None
            min_scale_2d: float = 0.0
            max_scale_2d: float = 0.0

        enable: EnableConfig = field(default_factory=EnableConfig)
        timing: TimingConfig = field(default_factory=TimingConfig)
        thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)

    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass
class ModelConfig:
    name: str = "ultrags_baseline"
    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
