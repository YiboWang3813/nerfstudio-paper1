import math

import torch

from .camera_adapter import build_virtual_camera
from .configs import ModelConfig
from .gaussian_container import UltraGSBaselineContainer
from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta


def _resolve_tanfovs(cfg: ModelConfig, camera, container: UltraGSBaselineContainer):
    fov_x = float(camera.FoVx)
    fov_y = float(camera.FoVy)
    if str(cfg.render.dar_mode) == "global_offset":
        fov_x += float(container.fov_x_offsets.mean().detach().item())
        fov_y += float(container.fov_y_offsets.mean().detach().item())
        fov_x = min(max(fov_x, math.radians(float(cfg.render.fov_x_min_deg))), math.radians(float(cfg.render.fov_x_max_deg)))
        fov_y = min(max(fov_y, math.radians(float(cfg.render.fov_y_min_deg))), math.radians(float(cfg.render.fov_y_max_deg)))
    return math.tan(fov_x * 0.5), math.tan(fov_y * 0.5)


def _view_points(points: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=-1)
    return homogeneous.matmul(view.to(device=points.device, dtype=points.dtype))[..., :3]


def _resolve_slice_thickness_mm(cfg: ModelConfig, dataset_meta: DatasetMeta) -> float:
    legacy_absolute = getattr(cfg.render, "slice_thickness_mm", None)
    if legacy_absolute is not None:
        return max(float(legacy_absolute), 1.0e-6)
    ratio = float(getattr(cfg.render, "slice_thickness_ratio", 0.5))
    return max(ratio * float(dataset_meta.delta_elevational_mm), 1.0e-6)


def _slice_candidates_and_attenuation(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    camera,
    container: UltraGSBaselineContainer,
):
    n = int(container.num_points)
    device = container.means.device
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0, 1), dtype=container.means.dtype, device=device)

    if not bool(getattr(cfg.render, "enable_slice_cull", True)):
        indices = torch.arange(n, device=device)
        attenuation = torch.ones((n, 1), dtype=container.means.dtype, device=device)
        return indices, attenuation

    if str(getattr(cfg.render, "camera_mode", "")) != "slice_plane_physical":
        indices = torch.arange(n, device=device)
        attenuation = torch.ones((n, 1), dtype=container.means.dtype, device=device)
        return indices, attenuation

    means_cam = _view_points(container.means, camera.world_view_transform)
    rel_z = means_cam[:, 2] - float(cfg.render.physical_camera_distance_mm)

    # UltraGS uses 2D disks rather than 3D ellipsoids, so the learned disk radius is
    # only a guard against over-culling. Keep physical thickness in charge of focus.
    max_scale = torch.max(container.scales_2d, dim=1).values
    thickness = _resolve_slice_thickness_mm(cfg, dataset_meta)
    scale_proxy = max_scale * float(getattr(cfg.render, "slice_cull_scale_multiplier", 3.0))
    scale_proxy = torch.minimum(scale_proxy, torch.full_like(scale_proxy, thickness))
    sigma_z = torch.sqrt(scale_proxy.square() + thickness * thickness).clamp_min(1.0e-6)
    keep = rel_z.abs() <= float(getattr(cfg.render, "slice_cull_std", 3.0)) * sigma_z
    indices = torch.nonzero(keep, as_tuple=False).squeeze(-1)

    if indices.numel() == 0:
        return indices, torch.empty((0, 1), dtype=container.means.dtype, device=device)

    if bool(getattr(cfg.render, "opacity_slice_attenuation", True)):
        selected_rel_z = rel_z[indices]
        selected_sigma_z = sigma_z[indices]
        attenuation = torch.exp(-0.5 * (selected_rel_z / selected_sigma_z).square()).unsqueeze(-1)
    else:
        attenuation = torch.ones((indices.shape[0], 1), dtype=container.means.dtype, device=device)
    return indices, attenuation.to(dtype=container.means.dtype)


def _empty_render(cfg: ModelConfig, camera, container: UltraGSBaselineContainer, screenspace_points: torch.Tensor):
    h = int(camera.image_height)
    w = int(camera.image_width)
    device = container.means.device
    dtype = container.means.dtype
    channels = 1 if bool(cfg.render.output_grayscale) else 3
    pred = torch.full((channels, h, w), float(cfg.render.bg_color), dtype=dtype, device=device)
    pred = pred + screenspace_points.sum() * 0.0
    depth = torch.zeros((1, h, w), dtype=dtype, device=device)
    alpha = torch.zeros((1, h, w), dtype=dtype, device=device)
    allmap = torch.cat([depth, alpha], dim=0)
    n = int(container.num_points)
    return pred, depth, {
        "viewspace_points": screenspace_points,
        "visibility_filter": torch.zeros((n,), dtype=torch.bool, device=device),
        "radii": torch.zeros((n,), dtype=torch.int32, device=device),
        "alpha_image": alpha,
        "allmap": allmap,
        "slice_candidate_count": 0,
        "rendered_gaussian_count": 0,
    }


def render(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: UltraGSBaselineContainer,
):
    from ultrags_baseline_renderer import GaussianRasterizationSettings, GaussianRasterizer

    camera = build_virtual_camera(cfg.render, dataset_meta, slice_meta)
    screenspace_points = torch.zeros_like(
        container.means,
        dtype=container.means.dtype,
        requires_grad=True,
        device=container.means.device,
    )
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    render_indices, slice_attenuation = _slice_candidates_and_attenuation(cfg, dataset_meta, camera, container)
    if render_indices.numel() == 0:
        return _empty_render(cfg, camera, container, screenspace_points)

    tanfovx, tanfovy = _resolve_tanfovs(cfg, camera, container)
    background = torch.full((3,), float(cfg.render.bg_color), dtype=torch.float32, device=container.means.device)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=float(cfg.render.scale_modifier),
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=int(container.active_sh_degree),
        campos=camera.camera_center,
        prefiltered=False,
        debug=bool(cfg.render.debug),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_rgb, radii, allmap = rasterizer(
        means3D=container.means[render_indices],
        means2D=screenspace_points[render_indices],
        shs=container.sh_coeffs[render_indices],
        colors_precomp=None,
        opacities=container.opacities[render_indices] * slice_attenuation,
        scales=container.scales_2d[render_indices],
        rotations=container.quats[render_indices],
        cov3D_precomp=None,
    )
    if bool(cfg.render.output_grayscale):
        pred = 0.299 * rendered_rgb[0] + 0.587 * rendered_rgb[1] + 0.114 * rendered_rgb[2]
        pred = pred.unsqueeze(0)
    else:
        pred = rendered_rgb

    render_alpha = allmap[1:2] if allmap.shape[0] > 1 else None
    render_depth_expected = allmap[0:1] if allmap.shape[0] > 0 else None
    if render_depth_expected is not None and render_alpha is not None:
        render_depth_expected = torch.nan_to_num(render_depth_expected / torch.clamp(render_alpha, min=1.0e-8), 0, 0)

    full_radii = torch.zeros((int(container.num_points),), dtype=radii.dtype, device=radii.device)
    full_radii[render_indices] = radii
    visibility = torch.zeros((int(container.num_points),), dtype=torch.bool, device=radii.device)
    visibility[render_indices] = radii > 0

    return pred, render_depth_expected, {
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility,
        "radii": full_radii,
        "alpha_image": render_alpha,
        "allmap": allmap,
        "slice_candidate_count": int(render_indices.numel()),
        "rendered_gaussian_count": int((radii > 0).sum().detach().item()),
    }
