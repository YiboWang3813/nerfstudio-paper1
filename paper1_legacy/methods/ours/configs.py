from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RenderConfig:
    enable_input_checks: bool = False


@dataclass
class GaussInitConfig:
    """Offline init via ``Scene.build_init_gaussians()`` (image ``pixel_pos`` world mm + ``scene.sampler``)."""

    #: Target count; written to ``scene.sampler.num_init`` before sampling.
    num_points: int = 1_000_000


@dataclass
class SceneNormConfig:
    """Reserved for checkpoint migration helpers only (``geometry.pose_probe_world_to_normalized``)."""

    margin: float = 1.15


@dataclass
class OursSliceRenderConfig:
    local_conditioning_mode: str = "adaptive"  # none | fixed | adaptive
    tile_size_px: int = 16
    #: Number of elevational slices used with ``DatasetMeta.delta_elevational_mm`` to form corridor half-width
    #: ``num_slices_elevational * delta_elevational_mm / 2`` (mm) when ``elev_corridor_half_mm`` is unset.
    num_slices_elevational: int = 2
    #: If set, overrides corridor half-width (mm). Use ``slab_thickness_ratio * delta_elevational_mm`` to match ultrags.
    elev_corridor_half_mm: Optional[float] = None
    slab_half_thickness_mm: float = 0.0
    k_sigma: float = 2.0
    mapper_kind: str = "tile"
    num_pixels_lateral: int = 5
    num_pixels_axial: int = 5
    mapper_sort_by_v: bool = True
    compositing_mode: str = "alpha"
    fixed_bg_color: float = 0.0
    shader_weight_mode: str = "probe3d"  # "probe3d" | "plane_cond2d" | "us_focus"
    us_focus_sigma_ele_0: float = 1.0
    us_focus_sigma_ele_1: float = 0.0
    us_focus_sigma_ele_min: float = 0.5
    us_focus_sigma_ele_max: float = 8.0
    us_focus_sigma_lat_0: float = 1.0
    us_focus_sigma_lat_1: float = 0.0
    us_focus_sigma_lat_min: float = 0.5
    us_focus_sigma_lat_max: float = 8.0
    us_focus_det_gamma: float = 0.25
    us_focus_det_eps: float = 1e-6
    us_focus_weight_min: float = 1e-6
    us_focus_use_lateral_focus: bool = False
    us_focus_inflate_lat_mm: float = 0.0
    us_focus_inflate_ax_mm: float = 0.0
    ray_attenuation_enabled: bool = False
    ray_attenuation_source: str = "opacity"  # "opacity" | "attenuation"
    ray_attenuation_beta: float = 0.0
    ray_attenuation_strength: float = 0.0
    ray_attenuation_min_transmission: float = 0.75
    ray_attenuation_weight_min: float = 1e-4
    ray_attenuation_axial_margin_mm: float = 0.0
    ray_attenuation_quadrature_window_mm: float = 2.0
    ray_attenuation_quadrature_points: int = 3
    ray_attenuation_mapper_kind: str = "tile_prefix"
    ray_attenuation_formula: str = "integral"
    ray_attenuation_candidate_k_sigma: float = -1.0
    render_chunk_size: int = 256


@dataclass
class GaussContainerConfig:
    @dataclass
    class LRConfig:
        means: float = 1.6e-4
        scales: float = 5e-3
        quats: float = 1e-3
        echo_alphas: float = 5e-2
        opacities: float = 5e-2
        intensities: float = 1e-3
        attenuations: float = 1e-2

    @dataclass
    class InitValConfig:
        echo_alphas: float = 0.1
        opacities: float = 0.1
        intensities: float = 0.1
        attenuations: float = 0.02

    @dataclass
    class ConstraintConfig:
        #: Default keeps legacy behavior: scale = exp(raw).
        #: Use "bounded_sigmoid" for stable us_focus runs:
        #: scale = scale_min_mm + (scale_max_mm - scale_min_mm) * sigmoid(raw).
        scale_activation: str = "exp"
        scale_min_mm: float = 0.05
        scale_max_mm: float = 3.0
        echo_alpha_max: float = 1.0
        opacity_max: float = 1.0
        intensity_max: float = 1.0
        attenuation_max: float = 1.0

    lr: LRConfig = field(default_factory=LRConfig)
    init_val: InitValConfig = field(default_factory=InitValConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)


@dataclass
class OfflineConfig:
    @dataclass
    class DensityControlConfig:
        @dataclass
        class DCResponsibilityConfig:
            enabled: bool = False
            high_residual_quantile: float = 0.90
            score_quantile: float = 0.98
            min_score: float = 1e-6
            max_fraction: float = 0.05
            mix_mode: str = "responsibility"
            union_gradient_fraction: float = 0.50
            union_gradient_min_factor: float = 0.25
            structure_weight_enabled: bool = False
            structure_boundary_weight: float = 0.50
            structure_highpass_weight: float = 0.50
            structure_speckle_weight: float = 0.25
            structure_max_weight: float = 2.50
            low_contribution_quantile: float = 0.10
            protect_attenuation_enabled: bool = True
            # high_tau keeps legacy behavior (P1). low_tau protects strong attenuation potential (P2).
            # path_effect protects high (1 - tau) * diagnostic_support Gaussians (P3 approximation).
            # path_contribution uses the exact accumulated A_g = sum(1 - T_igp) statistic.
            protect_attenuation_mode: str = "high_tau"
            protect_attenuation_quantile: float = 0.90
            path_contribution_threshold_mode: str = "quantile"  # "quantile" | "fixed"
            path_contribution_quantile: float = 0.80
            path_contribution_threshold: float = 0.0

        @dataclass
        class DCEnableConfig:
            densify: bool = True
            prune: bool = True

        @dataclass
        class DCTimingConfig:
            start_iter: int = 500
            end_iter: int = 7500
            freq: int = 500

        @dataclass
        class DCThresholdsConfig:
            densify_grad: float = 6e-6
            split_scale: float = 0.6
            prune_opacity: float = 5e-3
            prune_intensity: float = 0.01
            # Optional physical-scale pruning threshold. A non-positive value
            # keeps legacy behavior; production sweeps can enable this via
            # overrides to match other GS DC policies without clamping params.
            max_scale_mm: float = 0.0

        enable: DCEnableConfig = field(default_factory=DCEnableConfig)
        timing: DCTimingConfig = field(default_factory=DCTimingConfig)
        thresholds: DCThresholdsConfig = field(default_factory=DCThresholdsConfig)
        strategy: str = "gradient"  # "gradient" | "contribution_residual"
        # None preserves legacy aliases/overrides. Paper freeze aliases set both explicitly.
        densification_policy: Optional[str] = None  # gradient | residual_guided
        pruning_policy: Optional[str] = None  # opacity | none | local_aware | path_aware
        responsibility: DCResponsibilityConfig = field(default_factory=DCResponsibilityConfig)

    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)


@dataclass
class PoseOptimizationConfig:
    enabled: bool = False
    prior_poses_abspath: str = ""
    noise_enabled: bool = False
    noise_rot_deg: float = 0.0
    noise_trans_mm: float = 0.0
    noise_seed: int = 20260526
    trainable: bool = False
    start_iter: int = 0
    end_iter: int = -1
    lr: float = 3.0e-5
    lr_rot: Optional[float] = None
    lr_trans: Optional[float] = None
    optimize_rotation: bool = True
    optimize_translation: bool = True
    lambda_prior: float = 0.0
    lambda_prior_rot: Optional[float] = None
    lambda_prior_trans: Optional[float] = None
    lambda_smooth_rot: float = 0.0
    lambda_smooth_trans: float = 0.0
    residual_bound_rot_deg: float = 0.0
    residual_bound_trans_mm: float = 0.0
    pose_only_start_iter: int = -1
    pose_only_end_iter: int = -1
    log_interval: int = 500
    eval_train_count: int = 64


@dataclass
class PoseUncertaintyConfig:
    enabled: bool = False
    uncertainty_abspath: str = ""
    weight_mode: str = "direct"  # "direct" | "exp"
    min_weight: float = 0.1
    max_weight: float = 1.0
    alpha: float = 1.0
    image_loss_names: tuple = ("l1", "ssim", "mse", "lpips")
    decouple_density_control: bool = True
    log_interval: int = 500
    mc_enabled: bool = False
    mc_samples: int = 2
    mc_trans_sigma_mm: float = 0.0
    mc_rot_sigma_deg: float = 0.0


@dataclass
class ModelConfig:
    name: str = "ours"

    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_init: GaussInitConfig = field(default_factory=GaussInitConfig)
    scene_norm: SceneNormConfig = field(default_factory=SceneNormConfig)
    slice_render: OursSliceRenderConfig = field(default_factory=OursSliceRenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    pose_optimization: PoseOptimizationConfig = field(default_factory=PoseOptimizationConfig)
    pose_uncertainty: PoseUncertaintyConfig = field(default_factory=PoseUncertaintyConfig)
