import torch

from gaussianplie_renderer import GaussianPlieRenderConfig
from gaussianplie_renderer.grid_geometry import build_slice_grid_pack
from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta

from .configs import ModelConfig
from .gaussian_container import GaussianPileGaussContainer


def _resolve_sigma_z_mm(dataset_meta: DatasetMeta, model_cfg: ModelConfig) -> float:
    configured = model_cfg.slice_render.sigma_z_mm
    if configured is not None:
        return float(configured)
    ratio = getattr(model_cfg.slice_render, "sigma_z_ratio", None)
    if ratio is not None:
        return max(float(ratio) * float(dataset_meta.delta_elevational_mm), 1.0e-6)
    return float(dataset_meta.delta_elevational_mm)


def build_gaussianpile_render_config(
    dataset_meta: DatasetMeta,
    model_cfg: ModelConfig,
) -> GaussianPlieRenderConfig:
    sr = model_cfg.slice_render
    sigma_z = _resolve_sigma_z_mm(dataset_meta, model_cfg)
    return GaussianPlieRenderConfig(
        H=int(dataset_meta.H),
        W=int(dataset_meta.W),
        lateral_spacing_mm_per_pix=float(dataset_meta.delta_lateral_mm),
        axial_spacing_mm_per_pix=float(dataset_meta.delta_axial_mm),
        axial_near_mm=float(dataset_meta.near_mm),
        lateral_col_offset=(float(dataset_meta.W - 1) * 0.5 if dataset_meta.center_lateral else 0.0),
        backend=str(sr.backend),
        tile_size_px=int(sr.tile_size_px),
        radius_sigma=float(sr.radius_sigma),
        sigma_z_mm=float(sigma_z),
        prefilter_elevational_radius_mm=float(sr.prefilter_elevational_radius_mm),
        det_eps=float(sr.det_eps),
        cov_eps=float(sr.cov_eps),
        opacity_cull_threshold=float(sr.opacity_cull_threshold),
        render_chunk_size=int(sr.render_chunk_size),
        enable_input_checks=bool(model_cfg.render.enable_input_checks),
    )


def build_slice_grid_geometry(dataset_meta: DatasetMeta, device: torch.device) -> torch.Tensor:
    return build_slice_grid_pack(dataset_meta, device=device)


def render(
    model_cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: GaussianPileGaussContainer,
    is_jitter: bool = False,
    return_intermediates: bool = False,
    pose_override: torch.Tensor = None,
):
    del is_jitter
    if return_intermediates:
        raise NotImplementedError("gaussianplie_renderer path does not expose full intermediates yet.")

    pose_src = pose_override if pose_override is not None else slice_meta.pose
    pose_w = pose_src.to(device=container.means.device, dtype=container.means.dtype)

    renderer = container.ensure_gaussianpile_renderer(dataset_meta, model_cfg)
    grid = build_slice_grid_geometry(dataset_meta, container.means.device)
    renderer.set_slice_grid_geometry(grid)

    means = container.means
    scales = container.scales
    quats = container.quats
    alphas = container.alphas

    prefilter_radius = float(model_cfg.slice_render.prefilter_elevational_radius_mm)
    if prefilter_radius > 0.0:
        axis_elev = slice_meta.vec_elevational.to(device=means.device, dtype=means.dtype)
        origin = slice_meta.probe_origin.to(device=means.device, dtype=means.dtype)
        signed_dist = means.matmul(axis_elev) - origin.dot(axis_elev)
        active = signed_dist.abs() <= prefilter_radius
        if bool(active.any()):
            means = means[active]
            scales = scales[active]
            quats = quats[active]
            alphas = alphas[active]

    render_kwargs = {
        "pixel_positions_world": slice_meta.pixel_pos,
        "origin_world": slice_meta.probe_origin,
        "axis_lateral_world": slice_meta.vec_lateral,
        "axis_axial_world": slice_meta.vec_axial,
        "axis_elevational_world": slice_meta.vec_elevational,
    }
    chunk_size = max(1, int(model_cfg.slice_render.render_chunk_size))
    if means.shape[0] > chunk_size:
        pred = None
        for start in range(0, means.shape[0], chunk_size):
            end = min(start + chunk_size, means.shape[0])
            part = renderer(
                means[start:end],
                scales[start:end],
                quats[start:end],
                alphas[start:end],
                pose_w,
                **render_kwargs,
            )
            pred = part if pred is None else pred + part
    else:
        pred = renderer(
            means,
            scales,
            quats,
            alphas,
            pose_w,
            **render_kwargs,
        )
    depth = torch.zeros_like(pred)
    return pred, depth, {}


def render_debug(*args, **kwargs):
    return render(*args, **kwargs)
