import torch

from ultragray_renderer import UltragrayRenderer, UltragrayRenderConfig, ProbeGeometry

from .configs import ModelConfig
from .gaussian_container import UltraGrayGaussContainer
from .render_geometry import pixel_positions_for_renderer
from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta


def render(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: UltraGrayGaussContainer,
    is_jitter: bool = False,
    active_sh_degree1: bool = True,
):
    means = container.means
    scales = container.scales
    quats = container.quats
    attenuations = container._param_dict["attenuations"].tensor
    if active_sh_degree1:
        sh_coeffs = container.sh_coeffs
    else:
        sh_coeffs = torch.cat(
            [
                container._param_dict["sh_dc"].value,
                torch.zeros_like(container._param_dict["sh_rest"].value),
            ],
            dim=1,
        )

    pixel_pos = slice_meta.pixel_pos
    ray_origins = slice_meta.ray_origins
    ray_dirs = slice_meta.ray_dirs
    probe_origin = slice_meta.probe_origin
    if is_jitter:
        jitter_radius = 0.25 * dataset_meta.delta_elevational_mm
        jitter_scalar = (torch.rand(1, device=pixel_pos.device) - 0.5) * 2.0 * jitter_radius
        jitter_vec = slice_meta.vec_elevational * jitter_scalar
        pixel_pos = (pixel_pos + jitter_vec).contiguous()
        ray_origins = (ray_origins + jitter_vec).contiguous()
        probe_origin = (probe_origin + jitter_vec).contiguous()

    w = int(dataset_meta.W)
    h = int(dataset_meta.H)
    pixel_positions = pixel_positions_for_renderer(pixel_pos, dataset_meta)
    ray_origins = ray_origins.contiguous()
    ray_dirs = ray_dirs.contiguous()
    ops_max_offset_mm = float(cfg.render.ops_max_offset_mm)
    if cfg.render.ops_train_only and not is_jitter:
        ops_max_offset_mm = 0.0

    renderer = UltragrayRenderer(
        UltragrayRenderConfig(
            image_width=w,
            image_height=h,
            width_mm=float(dataset_meta.width_mm),
            near_mm=float(dataset_meta.near_mm),
            lateral_col_offset=(float(dataset_meta.W - 1) * 0.5 if dataset_meta.center_lateral else 0.0),
            elev_cutoff_margin_mm=float(cfg.render.elev_cutoff_margin_mm),
            k_sigma=float(cfg.render.k_sigma),
            ops_max_offset_mm=ops_max_offset_mm,
            quadrature_window_radius_mm=float(cfg.render.quadrature_window_radius_mm),
            local_transmission_window_only=bool(cfg.render.local_transmission_window_only),
            local_transmission_max_window_mm=float(cfg.render.local_transmission_max_window_mm),
        )
    )

    pred_image = renderer(
        means=means,
        scales=scales,
        quats=quats,
        attenuations=attenuations,
        sh_coeffs=sh_coeffs,
        pixel_positions=pixel_positions,
        ray_origins=ray_origins,
        ray_dirs=ray_dirs,
        geometry=ProbeGeometry(
            origin_world=probe_origin,
            axis_lateral_world=slice_meta.vec_lateral,
            axis_elevational_world=slice_meta.vec_elevational,
            axis_axial_world=slice_meta.vec_axial,
        ),
    )
    return pred_image, None, {}
