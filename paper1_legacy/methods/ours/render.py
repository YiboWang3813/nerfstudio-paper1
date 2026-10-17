import torch

from ours_renderer import OursRenderConfig
from ours_renderer.grid_geometry import build_slice_grid_pack
from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta

from .configs import ModelConfig
from .gaussian_container import OursGaussContainer


def build_ours_render_config(dataset_meta: DatasetMeta, model_cfg: ModelConfig) -> OursRenderConfig:
    sr = model_cfg.slice_render
    if sr.elev_corridor_half_mm is not None:
        elev_half = float(sr.elev_corridor_half_mm)
    else:
        elev_half = float(sr.num_slices_elevational) * float(dataset_meta.delta_elevational_mm) * 0.5
    return OursRenderConfig(
        local_conditioning_mode=str(sr.local_conditioning_mode),
        H=int(dataset_meta.H),
        W=int(dataset_meta.W),
        lateral_spacing_mm_per_pix=float(dataset_meta.delta_lateral_mm),
        axial_spacing_mm_per_pix=float(dataset_meta.delta_axial_mm),
        axial_near_mm=float(dataset_meta.near_mm),
        tile_size_px=int(sr.tile_size_px),
        elev_corridor_half_mm=elev_half,
        slab_half_thickness_mm=float(sr.slab_half_thickness_mm),
        k_sigma=float(sr.k_sigma),
        mapper_kind=str(sr.mapper_kind),
        num_pixels_lateral=int(sr.num_pixels_lateral),
        num_pixels_axial=int(sr.num_pixels_axial),
        mapper_sort_by_v=bool(sr.mapper_sort_by_v),
        compositing_mode=str(sr.compositing_mode),
        fixed_bg_color=float(sr.fixed_bg_color),
        shader_weight_mode=str(sr.shader_weight_mode),
        us_focus_sigma_ele_0=float(sr.us_focus_sigma_ele_0),
        us_focus_sigma_ele_1=float(sr.us_focus_sigma_ele_1),
        us_focus_sigma_ele_min=float(sr.us_focus_sigma_ele_min),
        us_focus_sigma_ele_max=float(sr.us_focus_sigma_ele_max),
        us_focus_sigma_lat_0=float(sr.us_focus_sigma_lat_0),
        us_focus_sigma_lat_1=float(sr.us_focus_sigma_lat_1),
        us_focus_sigma_lat_min=float(sr.us_focus_sigma_lat_min),
        us_focus_sigma_lat_max=float(sr.us_focus_sigma_lat_max),
        us_focus_det_gamma=float(sr.us_focus_det_gamma),
        us_focus_det_eps=float(sr.us_focus_det_eps),
        us_focus_weight_min=float(sr.us_focus_weight_min),
        us_focus_use_lateral_focus=bool(sr.us_focus_use_lateral_focus),
        us_focus_inflate_lat_mm=float(sr.us_focus_inflate_lat_mm),
        us_focus_inflate_ax_mm=float(sr.us_focus_inflate_ax_mm),
        ray_attenuation_enabled=bool(sr.ray_attenuation_enabled),
        ray_attenuation_source=str(sr.ray_attenuation_source),
        ray_attenuation_beta=float(sr.ray_attenuation_beta),
        ray_attenuation_strength=float(sr.ray_attenuation_strength),
        ray_attenuation_min_transmission=float(sr.ray_attenuation_min_transmission),
        ray_attenuation_weight_min=float(sr.ray_attenuation_weight_min),
        ray_attenuation_axial_margin_mm=float(sr.ray_attenuation_axial_margin_mm),
        ray_attenuation_quadrature_window_mm=float(sr.ray_attenuation_quadrature_window_mm),
        ray_attenuation_quadrature_points=int(sr.ray_attenuation_quadrature_points),
        ray_attenuation_mapper_kind=str(sr.ray_attenuation_mapper_kind),
        ray_attenuation_formula=str(sr.ray_attenuation_formula),
        ray_attenuation_candidate_k_sigma=float(sr.ray_attenuation_candidate_k_sigma),
        render_chunk_size=int(sr.render_chunk_size),
        enable_input_checks=model_cfg.render.enable_input_checks,
    )


def build_slice_grid_geometry(dataset_meta: DatasetMeta, device: torch.device) -> torch.Tensor:
    return build_slice_grid_pack(dataset_meta, device=device)


def render(
    model_cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: OursGaussContainer,
    is_jitter: bool = False,
    return_intermediates: bool = False,
    pose_override: torch.Tensor = None,
    collect_path_stats: bool = False,
):
    del is_jitter
    if return_intermediates:
        raise NotImplementedError("ours_renderer path does not expose intermediates.")

    pose_src = pose_override if pose_override is not None else slice_meta.pose
    pose_w = pose_src.to(device=container.means.device, dtype=container.means.dtype)

    renderer = container.ensure_ours_renderer(dataset_meta, model_cfg)
    grid = build_slice_grid_geometry(dataset_meta, container.means.device)
    renderer.set_slice_grid_geometry(grid)
    ray_attr = container.attenuations if model_cfg.slice_render.ray_attenuation_source == "attenuation" else None
    rendered = renderer(
        container.means,
        container.scales,
        container.quats,
        container.opacities,
        container.intensities,
        pose_w,
        ray_attr,
        collect_path_stats=collect_path_stats,
    )
    if collect_path_stats:
        pred, path_contribution, path_support = rendered
        intermediates = {
            "path_contribution": path_contribution,
            "path_support": path_support,
        }
    else:
        pred = rendered
        intermediates = {}
    depth = torch.zeros_like(pred)
    return pred, depth, intermediates


def render_debug(*args, **kwargs):
    return render(*args, **kwargs)
