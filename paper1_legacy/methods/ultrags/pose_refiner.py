from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class HighDimEncoder(nn.Module):
    """
    对粗 Pose 做 sin-cos 高频编码，风格类似位置编码。

    输入:
        x: (..., D)
    输出:
        y: (..., D * (1 + 2 * num_frequencies))  (include_input=True)
    """

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 6,
        include_input: bool = True,
        log_space: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.log_space = bool(log_space)

        if self.num_frequencies <= 0:
            raise ValueError("num_frequencies must be > 0")

        if self.log_space:
            freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (self.num_frequencies - 1), self.num_frequencies)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.input_dim * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, but got {x.shape[-1]}")

        enc_parts = []
        if self.include_input:
            enc_parts.append(x)

        # (..., D) -> (..., D, 1)
        x_exp = x.unsqueeze(-1)
        # (..., D, F)
        xb = x_exp * self.freq_bands
        enc_parts.append(torch.sin(xb).reshape(*x.shape[:-1], -1))
        enc_parts.append(torch.cos(xb).reshape(*x.shape[:-1], -1))
        return torch.cat(enc_parts, dim=-1)


class PoseRefiner(nn.Module):
    """
    se(3) residual PoseRefiner:
    coarse 4x4 pose -> encoder/decoder -> 6D twist -> exp(se3) -> refined 4x4 pose.
    """

    def __init__(
        self,
        pose_dim: int = 16,
        num_frequencies: int = 6,
        encoder_hidden_dim: int = 256,
        encoder_layers: int = 2,
        bottleneck_dim: int = 128,
        decoder_hidden_dim: int = 256,
        decoder_layers: int = 2,
        max_delta_rot_deg: float = 0.3,
        max_delta_trans_mm: float = 0.05,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.twist_dim = 6
        self.high_dim_encoder = HighDimEncoder(
            input_dim=self.pose_dim,
            num_frequencies=num_frequencies,
            include_input=True,
            log_space=True,
        )
        self.encoder_mlp = self._build_mlp(
            input_dim=self.high_dim_encoder.output_dim,
            hidden_dim=int(encoder_hidden_dim),
            num_layers=int(encoder_layers),
            out_dim=int(bottleneck_dim),
        )
        self.decoder_mlp = self._build_mlp(
            input_dim=int(bottleneck_dim),
            hidden_dim=int(decoder_hidden_dim),
            num_layers=int(decoder_layers),
            out_dim=self.twist_dim,
        )
        self.last_delta_twist = None
        self.max_delta_rot_rad = float(max_delta_rot_deg) * math.pi / 180.0
        self.max_delta_trans_mm = float(max_delta_trans_mm)
        self._init_identity_residual()

    def _limit_twist(self, delta_twist: torch.Tensor) -> torch.Tensor:
        """Hard-limit residual pose magnitude, keeping PoseRefiner as micro-adjuster."""
        omega = delta_twist[:, :3]
        trans = delta_twist[:, 3:]

        # Smoothly squash to bounded range while preserving gradient.
        if self.max_delta_rot_rad > 0:
            omega = self.max_delta_rot_rad * torch.tanh(omega / self.max_delta_rot_rad)
        else:
            omega = torch.zeros_like(omega)

        if self.max_delta_trans_mm > 0:
            trans = self.max_delta_trans_mm * torch.tanh(trans / self.max_delta_trans_mm)
        else:
            trans = torch.zeros_like(trans)

        return torch.cat([omega, trans], dim=-1)

    def _init_identity_residual(self) -> None:
        """Initialize decoder head to zero so delta twist starts from identity."""
        last = self.decoder_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, out_dim: int) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def _flatten_pose(self, coarse_pose: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
        """
        返回:
            flat_pose: (B, D)
            batch_shape: 原 batch 形状
            is_matrix_pose: 是否是 (...,4,4)
        """
        if coarse_pose.dim() >= 2 and coarse_pose.shape[-2:] == (4, 4):
            batch_shape = coarse_pose.shape[:-2]
            flat_pose = coarse_pose.reshape(-1, 16)
            return flat_pose, batch_shape, True

        if coarse_pose.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected last dim {self.pose_dim}, but got {coarse_pose.shape[-1]}")
        batch_shape = coarse_pose.shape[:-1]
        flat_pose = coarse_pose.reshape(-1, self.pose_dim)
        return flat_pose, batch_shape, False

    @staticmethod
    def _hat_3(v: torch.Tensor) -> torch.Tensor:
        """v: (B,3) -> skew matrix (B,3,3)."""
        b = v.shape[0]
        o = torch.zeros((b,), device=v.device, dtype=v.dtype)
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        row0 = torch.stack([o, -vz, vy], dim=-1)
        row1 = torch.stack([vz, o, -vx], dim=-1)
        row2 = torch.stack([-vy, vx, o], dim=-1)
        return torch.stack([row0, row1, row2], dim=1)

    def _exp_se3(self, xi: torch.Tensor) -> torch.Tensor:
        """
        xi: (B,6) [omega, v] -> T: (B,4,4), using exponential map on SE(3).
        """
        omega = xi[:, :3]
        v = xi[:, 3:]
        b = xi.shape[0]
        eps = 1e-8

        theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # (B,1)
        k = omega / theta.clamp_min(eps)
        k_hat = self._hat_3(k)
        eye = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(b, 1, 1)
        theta_m = theta.view(b, 1, 1)
        sin_t = torch.sin(theta_m)
        cos_t = torch.cos(theta_m)

        # Rodrigues for R
        r = eye + sin_t * k_hat + (1.0 - cos_t) * (k_hat @ k_hat)

        # Left Jacobian V for translation
        # V = I + (1-cos(theta))/theta * K + (theta-sin(theta))/theta * K^2
        omega_hat = self._hat_3(omega)
        theta2 = (theta ** 2).view(b, 1, 1).clamp_min(eps)
        theta3 = (theta ** 3).view(b, 1, 1).clamp_min(eps)
        v_mat = (
            eye
            + ((1.0 - torch.cos(theta_m)) / theta2) * omega_hat
            + ((theta_m - torch.sin(theta_m)) / theta3) * (omega_hat @ omega_hat)
        )
        t = (v_mat @ v.unsqueeze(-1)).squeeze(-1)

        # Small-angle fallback
        small = (theta.squeeze(-1) < 1e-4)
        if small.any():
            idx = small.nonzero(as_tuple=False).squeeze(-1)
            om = omega[idx]
            om_hat = self._hat_3(om)
            i_s = eye.index_select(0, idx)
            # 2nd-order approximation
            r_s = i_s + om_hat + 0.5 * (om_hat @ om_hat)
            v_s = i_s + 0.5 * om_hat + (1.0 / 6.0) * (om_hat @ om_hat)
            t_s = (v_s @ v.index_select(0, idx).unsqueeze(-1)).squeeze(-1)
            r[idx] = r_s
            t[idx] = t_s

        t_out = torch.zeros((b, 4, 4), device=xi.device, dtype=xi.dtype)
        t_out[:, :3, :3] = r
        t_out[:, :3, 3] = t
        t_out[:, 3, 3] = 1.0
        return t_out

    def forward(self, coarse_pose: torch.Tensor) -> torch.Tensor:
        flat_pose, batch_shape, is_matrix_pose = self._flatten_pose(coarse_pose)
        high_dim_feat = self.high_dim_encoder(flat_pose)
        latent = self.encoder_mlp(high_dim_feat)
        delta_twist = self.decoder_mlp(latent)
        delta_twist = self._limit_twist(delta_twist)
        self.last_delta_twist = delta_twist.detach()

        if is_matrix_pose:
            coarse_mat = coarse_pose.reshape(-1, 4, 4)
            delta_t = self._exp_se3(delta_twist)
            refined = delta_t @ coarse_mat
            return refined.reshape(*batch_shape, 4, 4)
        raise ValueError("PoseRefiner expects matrix input (...,4,4) in online mode.")

