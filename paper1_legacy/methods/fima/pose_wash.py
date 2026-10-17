"""冻结 PEM 后的 PoseRefiner 内循环（无 GS 渲染）。"""
from __future__ import annotations

from typing import Tuple

import torch

from .pose_wash_losses import PoseGtDeltaLossBatch


def optimize_pose_refiner_batch(
    pose_refiner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: PoseGtDeltaLossBatch,
    p_coarse: torch.Tensor,
    p_gt: torch.Tensor,
    num_inner_steps: int,
) -> Tuple[torch.Tensor, float]:
    """
    p_coarse / p_gt: (B, 4, 4)，requires_grad 无关（coarse 应对 PEM 路径 detach）。
    返回最后一轮的 ``p_refined`` 与平均 loss。
    """
    loss_acc = 0.0
    p_ref = None
    for _ in range(max(1, int(num_inner_steps))):
        optimizer.zero_grad(set_to_none=True)
        p_ref = pose_refiner(p_coarse)
        loss = loss_fn(p_coarse, p_ref, p_gt)
        loss.backward()
        optimizer.step()
        loss_acc += float(loss.detach().item())
    mean_loss = loss_acc / max(1, int(num_inner_steps))
    return p_ref, mean_loss


def gate_accept_refined(
    loss_coarse: float,
    loss_refined: float,
    *,
    min_improve: float,
    use_abs_improve: bool,
) -> bool:
    if use_abs_improve:
        improve = loss_coarse - loss_refined
    else:
        improve = (loss_coarse - loss_refined) / max(loss_coarse, 1e-8)
    return improve >= min_improve
