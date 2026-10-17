"""FiMA PEM 监督：L1 / MSE / Pearson。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .configs import FiMALossConfig
from .pose_geometry import angle_wrap_diff


def pearson_corr_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """x,y 同 shape，展平后 Pearson；保持 ``x`` 的 dtype 以便反传。"""
    x = x.reshape(-1)
    y = y.reshape(-1)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.norm() * ym.norm()).clamp_min(eps)
    return (xm * ym).sum() / denom


def pem_supervision_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: FiMALossConfig,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    pred/target: (B, 6) 或 (B, K, 6)，最后一维 [tx,ty,tz, rx,ry,rz]。
    valid_mask: optional (B,) 或 (B, K)，只在有效监督槽位上计算 loss。
    """
    if valid_mask is not None:
        mask = valid_mask.to(device=pred.device, dtype=torch.bool)
        if mask.ndim == pred.ndim:
            mask = mask.squeeze(-1)
        if mask.shape != pred.shape[:-1]:
            raise ValueError(f"valid_mask shape {tuple(mask.shape)} does not match pred prefix {tuple(pred.shape[:-1])}")
        pred = pred[mask]
        target = target[mask]
        if pred.numel() == 0:
            zero = target.new_tensor(0.0)
            return zero, {"total": zero.detach(), "l1_trans": zero.detach(), "l1_rot": zero.detach()}

    parts: Dict[str, torch.Tensor] = {}
    total = pred.new_tensor(0.0)
    if bool(getattr(cfg, "standardize_targets", False)):
        if target_mean is None or target_std is None:
            raise ValueError("pem_supervision_loss: standardize_targets=True requires target_mean/target_std")
        view_shape = (1,) * (pred.ndim - 1) + (6,)
        mean = target_mean.to(device=pred.device, dtype=pred.dtype).reshape(view_shape)
        std = target_std.to(device=pred.device, dtype=pred.dtype).reshape(view_shape).clamp_min(
            float(getattr(cfg, "target_std_min", 1.0e-6))
        )
        pred_for_loss = (pred - mean) / std
        target_for_loss = (target - mean) / std
    else:
        pred_for_loss = pred
        target_for_loss = target

    trans_p, rot_p = pred_for_loss[..., :3], pred_for_loss[..., 3:6]
    trans_t, rot_t = target_for_loss[..., :3], target_for_loss[..., 3:6]
    rot_diff = rot_p - rot_t if bool(getattr(cfg, "standardize_targets", False)) else angle_wrap_diff(rot_p, rot_t)

    if cfg.use_l1:
        lt = F.l1_loss(trans_p, trans_t)
        lr = rot_diff.abs().mean()
        l1 = lt + lr
        parts["l1_trans"] = lt.detach()
        parts["l1_rot"] = lr.detach()
        total = total + cfg.lambda_l1 * l1

    if cfg.use_mse:
        mt = F.mse_loss(trans_p, trans_t)
        mr = (rot_diff * rot_diff).mean()
        mse = mt + mr
        parts["mse_trans"] = mt.detach()
        parts["mse_rot"] = mr.detach()
        total = total + cfg.lambda_mse * mse

    if cfg.use_pearson:
        if cfg.pearson_mode == "batch_flat":
            pc = pearson_corr_1d(pred_for_loss, target_for_loss)
            pl = 1.0 - pc
            parts["pearson"] = pc.detach()
            total = total + cfg.lambda_pearson * pl
        else:
            # per_channel：对 (B, ...) 展平 batch 维后每个欧拉/平移通道单独相关
            flat_p = pred_for_loss.reshape(-1, pred_for_loss.shape[-1])
            flat_t = target_for_loss.reshape(-1, target_for_loss.shape[-1])
            losses = []
            cors = []
            for c in range(6):
                pc = pearson_corr_1d(flat_p[:, c], flat_t[:, c])
                cors.append(pc)
                losses.append(1.0 - pc)
            pl = torch.stack(losses).mean()
            parts["pearson_mean"] = torch.stack(cors).mean().detach()
            total = total + cfg.lambda_pearson * pl

    parts["total"] = total
    return total, parts


def heteroscedastic_pose_loss(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    cfg: FiMALossConfig,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Two-scalar heteroscedastic NLL for [tx,ty,tz,rx,ry,rz] pose outputs."""
    if bool(getattr(cfg, "standardize_targets", False)):
        if target_mean is None or target_std is None:
            raise ValueError("heteroscedastic_pose_loss: standardize_targets=True requires target_mean/target_std")
        view_shape = (1,) * (pred.ndim - 1) + (6,)
        mean = target_mean.to(device=pred.device, dtype=pred.dtype).reshape(view_shape)
        std = target_std.to(device=pred.device, dtype=pred.dtype).reshape(view_shape).clamp_min(
            float(getattr(cfg, "target_std_min", 1.0e-6))
        )
        pred_l = (pred - mean) / std
        target_l = (target - mean) / std
    else:
        pred_l = pred
        target_l = target

    if valid_mask is not None:
        mask = valid_mask.to(device=pred.device, dtype=torch.bool)
        if mask.ndim == pred.ndim:
            mask = mask.squeeze(-1)
        if mask.shape != pred.shape[:-1]:
            raise ValueError(f"valid_mask shape {tuple(mask.shape)} does not match pred prefix {tuple(pred.shape[:-1])}")
        pred_l = pred_l[mask]
        target_l = target_l[mask]
        logvar = logvar[mask]
        if pred_l.numel() == 0:
            zero = target.new_tensor(0.0)
            return zero, {"total": zero.detach(), "nll": zero.detach()}

    logvar = logvar.to(dtype=pred_l.dtype).clamp(float(logvar_min), float(logvar_max))
    trans_err = ((pred_l[..., :3] - target_l[..., :3]) ** 2).sum(dim=-1)
    rot_err = (angle_wrap_diff(pred_l[..., 3:6], target_l[..., 3:6]) ** 2).sum(dim=-1)
    lv_trans = logvar[..., 0]
    lv_rot = logvar[..., 1]
    nll_trans = 0.5 * torch.exp(-lv_trans) * trans_err + 0.5 * lv_trans
    nll_rot = 0.5 * torch.exp(-lv_rot) * rot_err + 0.5 * lv_rot
    nll = (nll_trans + nll_rot).mean()
    total = nll * float(getattr(cfg, "lambda_heteroscedastic", 1.0))
    return total, {
        "total": total,
        "nll": nll.detach(),
        "nll_trans": nll_trans.mean().detach(),
        "nll_rot": nll_rot.mean().detach(),
        "logvar_trans": lv_trans.mean().detach(),
        "logvar_rot": lv_rot.mean().detach(),
    }
