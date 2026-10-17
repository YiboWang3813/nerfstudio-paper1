import torch

from ultragauss_renderer import UltraGaussRenderer, UltraGaussRenderConfig, SliceGeometry

from .configs import ModelConfig
from .gaussian_container import UltraGaussContainer
from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta


def render(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: UltraGaussContainer,
    is_jitter: bool = False,
):
    means = container.means
    precision_tril = container.precision_tril
    opacities = container.opacities
    intensities = container.intensities

    pixel_pos = slice_meta.pixel_pos
    probe_origin = slice_meta.probe_origin
    if is_jitter:
        jitter_radius = float(cfg.render.jitter_radius_ratio) * float(dataset_meta.delta_elevational_mm)
        jitter_scalar = (torch.rand(1, device=pixel_pos.device) - 0.5) * 2.0 * jitter_radius
        jitter_vec = slice_meta.vec_elevational * jitter_scalar
        pixel_pos = (pixel_pos + jitter_vec).contiguous()
        probe_origin = (probe_origin + jitter_vec).contiguous()

    renderer = UltraGaussRenderer(
        UltraGaussRenderConfig(
            H=int(dataset_meta.H),
            W=int(dataset_meta.W),
            lateral_spacing_mm_per_pix=float(dataset_meta.delta_lateral_mm),
            axial_spacing_mm_per_pix=float(dataset_meta.delta_axial_mm),
            axial_near_mm=float(dataset_meta.near_mm),
            lateral_col_offset=(float(dataset_meta.W - 1) * 0.5 if dataset_meta.center_lateral else 0.0),
            chi2_threshold=float(cfg.render.chi2_threshold),
            mapper_sort_by_depth=bool(cfg.render.mapper_sort_by_depth),
            fixed_bg_alpha=float(cfg.render.fixed_bg_alpha),
            fixed_bg_intensity=float(cfg.render.fixed_bg_intensity),
            diag_beta=float(cfg.render.diag_beta),
            enable_shadow=bool(cfg.render.enable_shadow),
            shadow_delta=float(cfg.render.shadow_delta),
        )
    )

    pred_image, alpha_image, depth_image, intermediates = renderer(
        gauss_means_world=means,
        gauss_precision_tril=precision_tril,
        gauss_opacities=opacities,
        gauss_intensities=intensities,
        pixel_positions_world=pixel_pos,
        geometry=SliceGeometry(
            origin_world=probe_origin,
            axis_lateral_world=slice_meta.vec_lateral,
            axis_axial_world=slice_meta.vec_axial,
            axis_elevational_world=slice_meta.vec_elevational,
        ),
        return_intermediates=True,
    )
    intermediates["alpha_image"] = alpha_image
    return pred_image, depth_image, intermediates
