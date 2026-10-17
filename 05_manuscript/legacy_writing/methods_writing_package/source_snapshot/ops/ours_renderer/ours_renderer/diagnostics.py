import torch

from .autograd import _us_focus_params_vector
from .grid_geometry import shader_weight_mode_id
from .types import OursRenderConfig

try:
    from . import _C  # type: ignore[attr-defined]
except Exception:
    _C = None


def contribution_residual_stats(
    config: OursRenderConfig,
    gauss_means: torch.Tensor,
    gauss_scales: torch.Tensor,
    gauss_quats: torch.Tensor,
    gauss_opacities: torch.Tensor,
    gauss_intensities: torch.Tensor,
    pose_probe_to_world: torch.Tensor,
    slice_grid_geometry: torch.Tensor,
    residual_abs: torch.Tensor,
    high_residual_threshold: float,
):
    """Return per-Gaussian local-response responsibility stats for DC iterations.

    The diagnostic pass intentionally reuses the local tile candidate path. It does not alter the
    differentiable render graph and is intended to run only when density control fires.
    """
    if _C is None or not hasattr(_C, "ProjectedSliceContributionStatsCUDA"):
        raise RuntimeError("ours_renderer._C does not expose ProjectedSliceContributionStatsCUDA.")
    if config.mapper_kind != "tile":
        raise ValueError("contribution_residual_stats currently supports mapper_kind='tile' only.")
    if not gauss_means.is_cuda:
        raise ValueError("contribution_residual_stats requires CUDA tensors.")

    device = gauss_means.device
    grid = slice_grid_geometry.to(device=device, dtype=torch.float32).contiguous()
    us_focus_params = _us_focus_params_vector(config, device)
    pose_f = pose_probe_to_world.to(device=device, dtype=torch.float32).contiguous()
    residual_f = residual_abs.to(device=device, dtype=torch.float32).contiguous().view(-1)
    return _C.ProjectedSliceContributionStatsCUDA(
        gauss_means.float().contiguous(),
        gauss_scales.float().contiguous(),
        gauss_quats.float().contiguous(),
        gauss_opacities.float().contiguous(),
        gauss_intensities.float().contiguous(),
        pose_f,
        grid,
        us_focus_params,
        residual_f,
        int(config.H),
        int(config.W),
        float(config.lateral_spacing_mm_per_pix),
        float(config.axial_spacing_mm_per_pix),
        float(config.axial_near_mm),
        int(config.tile_size_px),
        float(config.elev_corridor_half_mm),
        float(config.slab_half_thickness_mm),
        float(config.k_sigma),
        bool(config.mapper_sort_by_v),
        0 if config.compositing_mode == "additive" else 1,
        float(config.fixed_bg_color),
        int(shader_weight_mode_id(config.shader_weight_mode)),
        float(high_residual_threshold),
    )
