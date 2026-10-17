import torch
import torch.nn.functional as F

from ours_renderer import contribution_residual_stats

from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta

from .configs import ModelConfig
from .gaussian_container import OursGaussContainer
from .render import build_ours_render_config, build_slice_grid_geometry


def _as_2d_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        return image
    if image.dim() == 3:
        if image.shape[0] in (1, 3):
            return image.mean(dim=0)
        if image.shape[-1] in (1, 3):
            return image.mean(dim=-1)
    raise ValueError(f"expected 2D grayscale or 3D single/RGB image, got shape={tuple(image.shape)}")


def _robust_unit(x: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    positive = x[x > 0]
    if positive.numel() == 0:
        return torch.zeros_like(x)
    scale = torch.quantile(positive.flatten(), min(max(float(q), 0.0), 1.0)).clamp_min(1e-6)
    return (x / scale).clamp(0.0, 1.0)


def _avg_pool_2d(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = int(kernel_size) // 2
    x4 = x[None, None]
    return F.avg_pool2d(F.pad(x4, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)[0, 0]


def _structure_weight_map(
    pred_image: torch.Tensor,
    gt_image: torch.Tensor,
    rcfg,
) -> torch.Tensor:
    gt = _as_2d_image(gt_image.detach()).to(dtype=torch.float32)
    residual = _as_2d_image((pred_image.detach() - gt_image.detach()).abs()).to(dtype=torch.float32)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gt.device,
        dtype=gt.dtype,
    )
    sobel_y = sobel_x.t()
    gt4 = gt[None, None]
    gx = F.conv2d(F.pad(gt4, (1, 1, 1, 1), mode="reflect"), sobel_x[None, None])[0, 0]
    gy = F.conv2d(F.pad(gt4, (1, 1, 1, 1), mode="reflect"), sobel_y[None, None])[0, 0]
    boundary = _robust_unit(torch.sqrt(gx * gx + gy * gy + 1e-12))

    residual_smooth = _avg_pool_2d(residual, kernel_size=5)
    highpass = _robust_unit((residual - residual_smooth).abs())

    gt_mean = _avg_pool_2d(gt, kernel_size=7)
    gt_sq_mean = _avg_pool_2d(gt * gt, kernel_size=7)
    local_var = (gt_sq_mean - gt_mean * gt_mean).clamp_min(0.0)
    speckle = _robust_unit(local_var)

    weight = torch.ones_like(gt, dtype=torch.float32)
    weight = weight + float(getattr(rcfg, "structure_boundary_weight", 0.50)) * boundary
    weight = weight + float(getattr(rcfg, "structure_highpass_weight", 0.50)) * highpass
    weight = weight + float(getattr(rcfg, "structure_speckle_weight", 0.25)) * speckle
    max_weight = max(1.0, float(getattr(rcfg, "structure_max_weight", 2.50)))
    return weight.clamp(1.0, max_weight).to(device=gt_image.device, dtype=gt_image.dtype)


def _apply_spatial_weight(residual: torch.Tensor, weight_2d: torch.Tensor) -> torch.Tensor:
    if residual.dim() == 2:
        return residual * weight_2d
    if residual.dim() == 3:
        if residual.shape[0] in (1, 3):
            return residual * weight_2d.unsqueeze(0)
        if residual.shape[-1] in (1, 3):
            return residual * weight_2d.unsqueeze(-1)
    raise ValueError(f"cannot apply 2D weight to residual shape={tuple(residual.shape)}")


def compute_contribution_residual_stats(
    model_cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    container: OursGaussContainer,
    pred_image: torch.Tensor,
    gt_image: torch.Tensor,
    high_residual_quantile: float,
):
    """Compute per-Gaussian residual responsibility stats for a single rendered slice."""
    residual = (pred_image.detach() - gt_image.detach()).abs()
    rcfg = model_cfg.offline.density_control.responsibility
    diag_stats = {"responsibility_structure_weight_enabled": 0.0}
    if bool(getattr(rcfg, "structure_weight_enabled", False)):
        weight = _structure_weight_map(pred_image, gt_image, rcfg)
        residual = _apply_spatial_weight(residual, weight)
        diag_stats.update(
            {
                "responsibility_structure_weight_enabled": 1.0,
                "responsibility_structure_weight_mean": float(weight.mean().detach().item()),
                "responsibility_structure_weight_max": float(weight.max().detach().item()),
            }
        )
    q = float(high_residual_quantile)
    q = min(max(q, 0.0), 1.0)
    high_residual_threshold = float(torch.quantile(residual.flatten(), q).detach().item())

    cfg = build_ours_render_config(dataset_meta, model_cfg)
    grid = build_slice_grid_geometry(dataset_meta, container.means.device)
    pose_w = slice_meta.pose.to(device=container.means.device, dtype=container.means.dtype)
    stats = contribution_residual_stats(
        cfg,
        container.means.detach(),
        container.scales.detach(),
        container.quats.detach(),
        container.opacities.detach(),
        container.intensities.detach(),
        pose_w.detach(),
        grid,
        residual,
        high_residual_threshold,
    )
    return (*stats, diag_stats)
