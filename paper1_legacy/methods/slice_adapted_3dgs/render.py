import torch

from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta

from .camera_adapter import build_slice_camera, pixel_grid
from .configs import ModelConfig
from .gaussian_container import SliceAdapted3DGSContainer


def _resolve_slice_thickness_mm(render_cfg, dataset_meta: DatasetMeta) -> float:
    configured = getattr(render_cfg, "slice_thickness_mm", None)
    if configured is not None:
        return max(float(configured), 1.0e-6)
    ratio = getattr(render_cfg, "slice_thickness_ratio", None)
    if ratio is not None:
        return max(float(ratio) * float(dataset_meta.delta_elevational_mm), 1.0e-6)
    return max(float(dataset_meta.delta_elevational_mm), 1.0e-6)


def _view_points(points: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], dim=-1).matmul(view)[..., :3]


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=-1, eps=1.0e-8)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        [
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _camera_covariances(scales: torch.Tensor, quats: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    rot = _quat_to_matrix(quats)
    cov_world = rot.matmul(torch.diag_embed(scales * scales)).matmul(rot.transpose(-1, -2))
    r_wc = view[:3, :3]
    return r_wc.transpose(0, 1).unsqueeze(0).matmul(cov_world).matmul(r_wc.unsqueeze(0))


def _coarse_slice_filter(
    rel_z: torch.Tensor,
    scales: torch.Tensor,
    render_cfg,
    dataset_meta: DatasetMeta = None,
) -> torch.Tensor:
    """Conservative slab cull before full covariance construction.

    The largest 3D scale bounds any camera-axis standard deviation after rotation,
    so this cannot remove a Gaussian that the exact sigma_z test would keep.
    """
    max_var = scales.amax(dim=-1).square()
    min_cov = torch.as_tensor(float(render_cfg.min_covariance), dtype=scales.dtype, device=scales.device)
    max_var = torch.maximum(max_var, min_cov)
    thickness_mm = (
        _resolve_slice_thickness_mm(render_cfg, dataset_meta)
        if dataset_meta is not None
        else max(float(render_cfg.slice_thickness_mm), 1.0e-6)
    )
    sigma_bound = torch.sqrt(max_var + thickness_mm**2)
    radius = float(render_cfg.cull_std) * sigma_bound * float(render_cfg.thickness_scale)
    return torch.abs(rel_z) <= radius


def _empty_render(camera, dtype, device, render_cfg, visibility_filter):
    pred = torch.full(
        (3, int(camera.image_height), int(camera.image_width)),
        float(render_cfg.bg_color),
        dtype=dtype,
        device=device,
    )
    alpha = torch.zeros((1, int(camera.image_height), int(camera.image_width)), dtype=dtype, device=device)
    return pred, None, {
        "visibility_filter": visibility_filter,
        "alpha_image": alpha,
        "camera": camera,
        "coarse_candidate_count": int(visibility_filter.sum().detach().item()),
        "rendered_gaussian_count": 0,
    }


def _render_reference(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: SliceAdapted3DGSContainer,
):
    render_cfg = cfg.render
    slice_thickness_mm = _resolve_slice_thickness_mm(render_cfg, dataset_meta)
    camera = build_slice_camera(render_cfg, dataset_meta, slice_meta)
    device = container.means.device
    dtype = container.means.dtype

    means_cam = _view_points(container.means, camera.world_view_transform.to(device=device, dtype=dtype))
    rel_z = means_cam[:, 2] - float(camera.focal_depth_mm)
    scales_all = container.scales_3d
    coarse_keep = _coarse_slice_filter(rel_z, scales_all, render_cfg, dataset_meta)
    if not bool(coarse_keep.any().item()):
        return _empty_render(camera, dtype, device, render_cfg, coarse_keep)

    candidate_indices = torch.nonzero(coarse_keep, as_tuple=False).squeeze(-1)
    means_cam = means_cam[candidate_indices]
    rel_z = rel_z[candidate_indices]
    scales = scales_all[candidate_indices]
    quats = container.activated_param("quats", candidate_indices)

    cov_cam = _camera_covariances(scales, quats, camera.world_view_transform.to(device=device, dtype=dtype))
    min_cov = float(render_cfg.min_covariance)
    cov2d = cov_cam[:, :2, :2]
    eye2 = torch.eye(2, dtype=dtype, device=device).unsqueeze(0)
    cov2d = cov2d + eye2 * min_cov
    var_z = torch.clamp(cov_cam[:, 2, 2], min=min_cov)
    sigma_z = torch.sqrt(var_z + slice_thickness_mm**2)

    exact_keep = torch.abs(rel_z) <= float(render_cfg.cull_std) * sigma_z * float(render_cfg.thickness_scale)
    full_keep = torch.zeros_like(coarse_keep)
    full_keep[candidate_indices] = exact_keep
    if not bool(exact_keep.any().item()):
        return _empty_render(camera, dtype, device, render_cfg, full_keep)

    render_indices = candidate_indices[exact_keep]
    means2d = means_cam[exact_keep, :2]
    cov2d = cov2d[exact_keep]
    sigma_z = sigma_z[exact_keep]
    rel_z = rel_z[exact_keep]
    colors = container.dc_colors_for(render_indices).to(dtype=dtype)
    opacities = container.activated_param("opacities", render_indices).to(dtype=dtype)

    det = torch.clamp(cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0], min=min_cov)
    inv = torch.empty_like(cov2d)
    inv[:, 0, 0] = cov2d[:, 1, 1] / det
    inv[:, 1, 1] = cov2d[:, 0, 0] / det
    inv[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv[:, 1, 0] = -cov2d[:, 1, 0] / det

    attenuation = torch.exp(-0.5 * (rel_z / torch.clamp(sigma_z, min=1.0e-6)) ** 2).unsqueeze(-1)
    weights = opacities * attenuation
    if bool(render_cfg.normalize_by_footprint):
        weights = weights / torch.sqrt(det).unsqueeze(-1)

    if str(render_cfg.backend) == "cuda" and device.type == "cuda":
        if dtype != torch.float32:
            raise ValueError("slice_adapted_3dgs cuda backend currently supports float32 only.")
        from slice_adapted_3dgs_renderer import additive_splat_bounded

        splat_flat, alpha_flat = additive_splat_bounded(
            int(camera.image_height),
            int(camera.image_width),
            float(camera.half_width_mm),
            float(camera.half_height_mm),
            means2d,
            inv,
            weights,
            colors,
            float(render_cfg.cull_std),
        )
        image_flat = splat_flat + float(render_cfg.bg_color)
    else:
        pixels = pixel_grid(camera, device=device, dtype=dtype)
        image_flat = torch.full((pixels.shape[0], 3), float(render_cfg.bg_color), dtype=dtype, device=device)
        alpha_flat = torch.zeros((pixels.shape[0], 1), dtype=dtype, device=device)

        point_chunk = max(1, int(render_cfg.max_points_per_chunk))
        pixel_chunk = max(1, int(render_cfg.max_pixels_per_chunk))
        for p0 in range(0, pixels.shape[0], pixel_chunk):
            p = pixels[p0 : p0 + pixel_chunk]
            out = torch.zeros((p.shape[0], 3), dtype=dtype, device=device)
            alpha_out = torch.zeros((p.shape[0], 1), dtype=dtype, device=device)
            for g0 in range(0, means2d.shape[0], point_chunk):
                g1 = g0 + point_chunk
                d = p[:, None, :] - means2d[None, g0:g1, :]
                inv_g = inv[g0:g1]
                power = -0.5 * torch.einsum("pgi,gij,pgj->pg", d, inv_g, d)
                gaussian = torch.exp(torch.clamp(power, min=-50.0, max=0.0))
                weighted = gaussian.unsqueeze(-1) * weights[g0:g1].unsqueeze(0)
                out = out + (weighted * colors[g0:g1].unsqueeze(0)).sum(dim=1)
                alpha_out = alpha_out + weighted.sum(dim=1)
            image_flat[p0 : p0 + p.shape[0]] = image_flat[p0 : p0 + p.shape[0]] + out
            alpha_flat[p0 : p0 + p.shape[0]] = alpha_out

    if bool(render_cfg.clamp_output):
        image_flat = image_flat.clamp(0.0, 1.0)
    rendered_rgb = image_flat.transpose(0, 1).reshape(3, int(camera.image_height), int(camera.image_width))
    alpha = alpha_flat.transpose(0, 1).reshape(1, int(camera.image_height), int(camera.image_width))
    return rendered_rgb, None, {
        "visibility_filter": full_keep,
        "alpha_image": alpha,
        "camera": camera,
        "coarse_candidate_count": int(candidate_indices.numel()),
        "rendered_gaussian_count": int(render_indices.numel()),
    }


def render(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: SliceAdapted3DGSContainer,
):
    backend = str(cfg.render.backend)
    if backend not in {"torch_reference", "cuda"}:
        raise ValueError(f"slice_adapted_3dgs supports backend='torch_reference' or 'cuda', got {backend!r}.")
    rendered_rgb, depth, info = _render_reference(cfg, dataset_meta, slice_meta, container)
    if bool(cfg.render.output_grayscale):
        pred = 0.299 * rendered_rgb[0] + 0.587 * rendered_rgb[1] + 0.114 * rendered_rgb[2]
        pred = pred.unsqueeze(0)
    else:
        pred = rendered_rgb
    return pred, depth, info
