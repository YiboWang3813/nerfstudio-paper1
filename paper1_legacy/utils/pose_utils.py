"""位姿相关的可复用数值工具（噪声采样等）。"""
import math

import torch


def sample_noisy_pose_se3(
    p_gt: torch.Tensor,
    device: torch.device,
    rot_std_deg: float = 1.0,
    trans_std_mm: float = 1.0,
) -> torch.Tensor:
    """
    在 GT 位姿左侧乘以小扰动 delta（轴角 + 平移），用于 Mode B 粗位姿模拟。
    p_gt: (4, 4)
    """
    rot_std = rot_std_deg * math.pi / 180.0
    w = torch.randn((3,), device=device) * rot_std
    t = torch.randn((3,), device=device) * trans_std_mm
    theta = torch.norm(w) + 1e-9
    k = w / theta
    K = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        device=device,
        dtype=p_gt.dtype,
    )
    I = torch.eye(3, device=device, dtype=p_gt.dtype)
    R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    delta = torch.eye(4, device=device, dtype=p_gt.dtype)
    delta[:3, :3] = R
    delta[:3, 3] = t
    return delta @ p_gt
