"""FiMA 离线 PoseWash：与 ``utils.loss.pose_loss.PoseGtDeltaLoss`` 语义一致的批量版本。"""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseGtDeltaLossBatch(nn.Module):
    """``d_pred = inv(P_coarse) @ P_refined`` 对齐 ``d_gt = inv(P_coarse) @ P_gt``（批次 Frobenius / L2）。"""

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0):
        super().__init__()
        self.rot_weight = float(rot_weight)
        self.trans_weight = float(trans_weight)

    def forward(
        self,
        pose_coarse: torch.Tensor,
        pose_refined: torch.Tensor,
        pose_gt: torch.Tensor,
    ) -> torch.Tensor:
        inv_c = torch.inverse(pose_coarse)
        d_pred = inv_c @ pose_refined
        d_gt = inv_c @ pose_gt
        rot_err = torch.norm(d_pred[..., :3, :3] - d_gt[..., :3, :3], dim=(1, 2))
        trans_err = torch.norm(d_pred[..., :3, 3] - d_gt[..., :3, 3], dim=-1)
        return (self.rot_weight * rot_err + self.trans_weight * trans_err).mean()
