from dataclasses import dataclass, field as dc_field


@dataclass
class FieldConfig:
    depth: int = 8
    width: int = 256
    pe_num_freqs: int = 10
    include_input: bool = True
    skip_layer: int = 4
    use_viewdirs: bool = False
    alpha_activation: str = "abs"
    alpha_bias_init: float = 0.0


@dataclass
class RayConfig:
    coordinate_unit: str = "mm"
    coordinate_scale: float = 100.0
    train_patch_height: int = 128
    train_patch_width: int = 128
    eval_chunk_size: int = 65536


@dataclass
class RenderConfig:
    boundary_sampling_mode: str = "expectation"
    scatter_sampling_mode: str = "expectation"
    eval_render_mode: str = "mc_average"
    eval_mc_samples: int = 4
    eval_seed: int = 20260604
    eval_save_stochastic: bool = True
    enable_psf: bool = True
    psf_kernel_size: int = 7
    psf_sigma_lateral: float = 2.0
    psf_sigma_axial: float = 1.0
    attenuation_scale: float = 0.01
    alpha_max: float = 0.0
    boundary_attenuation_scale: float = 1.0
    reflection_gain: float = 1.0
    eps: float = 1.0e-6
    output_mode: str = "clamp"
    output_gain: float = 1.0
    relaxed_temperature: float = 0.1


@dataclass
class SolverConfig:
    optimizer: str = "radam"
    lr: float = 5.0e-4
    lr_final: float = 5.0e-5
    lr_decay_steps: int = 300000


@dataclass
class OfflineConfig:
    @dataclass
    class DiagnosticsConfig:
        enable_render_stats: bool = True
        train_step: int = 100
        save_debug_maps: bool = True

    diagnostics: DiagnosticsConfig = dc_field(default_factory=DiagnosticsConfig)


@dataclass
class ModelConfig:
    name: str = "nerfus"
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    ray: RayConfig = dc_field(default_factory=RayConfig)
    render: RenderConfig = dc_field(default_factory=RenderConfig)
    solver: SolverConfig = dc_field(default_factory=SolverConfig)
    offline: OfflineConfig = dc_field(default_factory=OfflineConfig)
