from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def build_confidence_from_occupancy(
    occupancy_hw: torch.Tensor,
    *,
    gamma: float = 1.0,
    min_conf: float = 0.05,
    detach: bool = True,
) -> torch.Tensor:
    """将 occupancy 映射为 confidence: high-occupancy => high-confidence。"""
    occ = occupancy_hw
    if detach:
        occ = occ.detach()
    occ = occ.clamp(0.0, 1.0)
    conf = occ.pow(float(max(gamma, 1e-6)))
    if min_conf > 0:
        conf = conf.clamp(min=float(min_conf), max=1.0)
    return conf


def weighted_charbonnier(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """像素级 weighted Charbonnier。"""
    if weight is None:
        weight = torch.ones_like(pred)
    # 支持 (H,W) 或 (C,H,W)，按广播统一。
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(0)
    diff = pred - gt
    val = torch.sqrt(diff * diff + float(epsilon) * float(epsilon))
    num = torch.sum(val * weight)
    den = torch.sum(weight).clamp_min(1e-8)
    return num / den


class OccupancyConfidencePoseLoss(nn.Module):
    """GS occupancy 生成 confidence map，给 PoseRefiner 的配准损失做加权。"""

    def __init__(
        self,
        charbonnier_epsilon: float = 1e-3,
        confidence_gamma: float = 1.0,
        confidence_min: float = 0.05,
        detach_confidence: bool = True,
    ):
        super().__init__()
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.confidence_gamma = float(confidence_gamma)
        self.confidence_min = float(confidence_min)
        self.detach_confidence = bool(detach_confidence)

    def forward(
        self,
        pred_aux: torch.Tensor,
        gt: torch.Tensor,
        occupancy_hw: torch.Tensor,
    ) -> torch.Tensor:
        conf = build_confidence_from_occupancy(
            occupancy_hw,
            gamma=self.confidence_gamma,
            min_conf=self.confidence_min,
            detach=self.detach_confidence,
        )
        return weighted_charbonnier(
            pred=pred_aux,
            gt=gt,
            weight=conf,
            epsilon=self.charbonnier_epsilon,
        )
