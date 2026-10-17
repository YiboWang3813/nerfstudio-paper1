"""
相对全局第 0 帧的 SE(3) 与 6DoF（平移 mm + ZYX 欧拉 rad）互转。

旋转约定：``R = Rz(rz) @ Ry(ry) @ Rx(rx)``（与常见 Tait–Bryan ZYX 一致）。
PoseRefiner 侧使用 4×4 矩阵；PEM 侧使用本模块的 6 维向量，二者通过矩阵 compose，禁止混用 twist 与欧拉。
"""
from __future__ import annotations

import torch


def _Rx(a: torch.Tensor) -> torch.Tensor:
    c = torch.cos(a)
    s = torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)
    row0 = torch.stack([o, z, z], dim=-1)
    row1 = torch.stack([z, c, -s], dim=-1)
    row2 = torch.stack([z, s, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _Ry(a: torch.Tensor) -> torch.Tensor:
    c = torch.cos(a)
    s = torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)
    row0 = torch.stack([c, z, s], dim=-1)
    row1 = torch.stack([z, o, z], dim=-1)
    row2 = torch.stack([-s, z, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _Rz(a: torch.Tensor) -> torch.Tensor:
    c = torch.cos(a)
    s = torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)
    row0 = torch.stack([c, -s, z], dim=-1)
    row1 = torch.stack([s, c, z], dim=-1)
    row2 = torch.stack([z, z, o], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def euler_xyz_to_rotation(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
    """标量或批量最后一维为 1；返回 (..., 3, 3)。"""
    return _Rz(rz) @ _Ry(ry) @ _Rx(rx)


def rotation_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """R: (..., 3, 3) -> (..., 3) 欧拉 rx,ry,rz。"""
    eps = 1e-7
    r20 = R[..., 2, 0]
    cy = torch.sqrt(torch.clamp(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2, min=eps))
    singular = cy < 1e-6

    rz = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    ry = torch.atan2(-r20, cy)
    rx = torch.atan2(R[..., 2, 1], R[..., 2, 2])

    # Gimbal lock fallback
    if singular.any():
        rz_g = torch.zeros_like(rz)
        ry_g = torch.atan2(-r20, cy)
        rx_g = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
        rx = torch.where(singular, rx_g, rx)
        ry = torch.where(singular, ry_g, ry)
        rz = torch.where(singular, rz_g, rz)
    return torch.stack([rx, ry, rz], dim=-1)


def se3_from_6dof(xyz_euler: torch.Tensor) -> torch.Tensor:
    """
    xyz_euler: (..., 6) = [tx, ty, tz, rx, ry, rz]，单位为 mm / rad。
    返回 (..., 4, 4)。
    """
    t = xyz_euler[..., :3]
    eul = xyz_euler[..., 3:6]
    R = euler_xyz_to_rotation(eul[..., 0], eul[..., 1], eul[..., 2])
    out = torch.zeros(*xyz_euler.shape[:-1], 4, 4, device=xyz_euler.device, dtype=xyz_euler.dtype)
    out[..., :3, :3] = R
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def six_dof_from_matrix(T: torch.Tensor) -> torch.Tensor:
    """T: (..., 4, 4) -> (..., 6)"""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    eul = rotation_to_euler_xyz(R)
    return torch.cat([t, eul], dim=-1)


def relative_pose_from_world(P_src: torch.Tensor, P_ref: torch.Tensor) -> torch.Tensor:
    """T = P_src @ inv(P_ref)，均为 (..., 4, 4)。"""
    return P_src @ torch.inverse(P_ref)


def adjacent_frame_transform_gt(P_i: torch.Tensor, P_ip1: torch.Tensor) -> torch.Tensor:
    """
    相邻帧 ``i`` 与 ``i+1`` 的相对变换 **T_{i\\to i+1}**（右乘）：
    ``P_{i+1} = P_i @ T``，故 ``T = inv(P_i) @ P_{i+1}``。

    ``P_*`` 为 ``probe \\to world``（或与磁盘 relative 协议一致下的锚世界）齐次阵，shape (..., 4, 4)。
    """
    return torch.linalg.inv(P_i) @ P_ip1


def compose_world_from_rel(T_rel: torch.Tensor, P0_world: torch.Tensor) -> torch.Tensor:
    """P_hat_i = T_rel @ P0_world"""
    return T_rel @ P0_world


def translation_rotation_errors(T_pred: torch.Tensor, T_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    T_pred, T_gt: (..., 4, 4)
    返回平移 L2 (mm)、旋转测地角 (rad) 的逐样本张量，shape (...)。
    """
    dt = torch.norm(T_pred[..., :3, 3] - T_gt[..., :3, 3], dim=-1)
    R_e = T_pred[..., :3, :3].transpose(-1, -2) @ T_gt[..., :3, :3]
    tr = R_e[..., 0, 0] + R_e[..., 1, 1] + R_e[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    ang = torch.acos(cos)
    return dt, ang


def angle_wrap_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """最短弧差 a-b，用于欧拉分量。"""
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))
