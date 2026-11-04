"""Native camera optimizer preserving probe-local Euler residuals in mm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from nerfstudio.cameras.camera_optimizers import CameraOptimizer


@dataclass
class UltrasoundPoseConfig:
    enabled: bool = False
    prior_poses_abspath: str = ""
    noise_enabled: bool = False
    noise_rot_deg: float = 0.0
    noise_trans_mm: float = 0.0
    noise_seed: int = 20260526
    trainable: bool = False
    start_iter: int = 0
    end_iter: int = -1
    lr: float = 3.0e-5
    lr_rot: Optional[float] = None
    lr_trans: Optional[float] = None
    optimize_rotation: bool = True
    optimize_translation: bool = True
    lambda_prior: float = 0.0
    lambda_prior_rot: Optional[float] = None
    lambda_prior_trans: Optional[float] = None
    lambda_smooth_rot: float = 0.0
    lambda_smooth_trans: float = 0.0
    residual_bound_rot_deg: float = 0.0
    residual_bound_trans_mm: float = 0.0
    pose_only_start_iter: int = -1
    pose_only_end_iter: int = -1
    log_interval: int = 500
    eval_train_count: int = 64


@dataclass
class UltrasoundUncertaintyConfig:
    enabled: bool = False
    uncertainty_abspath: str = ""
    weight_mode: str = "direct"  # "direct" | "exp"
    min_weight: float = 0.1
    max_weight: float = 1.0
    alpha: float = 1.0
    image_loss_names: tuple = ("l1", "ssim", "mse", "lpips")
    decouple_density_control: bool = True
    log_interval: int = 500
    mc_enabled: bool = False
    mc_samples: int = 2
    mc_trans_sigma_mm: float = 0.0
    mc_rot_sigma_deg: float = 0.0


def _euler_xyz_to_matrix(params: torch.Tensor) -> torch.Tensor:
    rx, ry, rz, tx, ty, tz = params.unbind(dim=-1)
    z0 = torch.zeros_like(rx)
    o1 = torch.ones_like(rx)

    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    row0_x = torch.stack([o1, z0, z0], dim=-1)
    row1_x = torch.stack([z0, cx, -sx], dim=-1)
    row2_x = torch.stack([z0, sx, cx], dim=-1)
    rx_m = torch.stack([row0_x, row1_x, row2_x], dim=-2)

    row0_y = torch.stack([cy, z0, sy], dim=-1)
    row1_y = torch.stack([z0, o1, z0], dim=-1)
    row2_y = torch.stack([-sy, z0, cy], dim=-1)
    ry_m = torch.stack([row0_y, row1_y, row2_y], dim=-2)

    row0_z = torch.stack([cz, -sz, z0], dim=-1)
    row1_z = torch.stack([sz, cz, z0], dim=-1)
    row2_z = torch.stack([z0, z0, o1], dim=-1)
    rz_m = torch.stack([row0_z, row1_z, row2_z], dim=-2)

    R = rz_m @ ry_m @ rx_m
    t = torch.stack([tx, ty, tz], dim=-1).unsqueeze(-1)
    top = torch.cat([R, t], dim=-1)
    bottom = params.new_zeros((*params.shape[:-1], 1, 4))
    bottom[..., 0, 3] = 1.0
    return torch.cat([top, bottom], dim=-2)


class UltrasoundCameraOptimizer(CameraOptimizer):
    """Nerfstudio camera component with named native optimizer parameter groups.

    Unlike stock world-space SO3xR3, this protocol right-multiplies probe-local
    Euler XYZ residuals. Keeping that convention preserves the source method.
    Held-out/viewer cameras receive no learned frame-wise residual.
    """

    def __init__(self, config, frame_ids, poses):
        nn.Module.__init__(self)
        self.config = config
        self.indices = {int(frame): index for index, frame in enumerate(frame_ids)}
        prior = poses.clone()
        if config.enabled and config.prior_poses_abspath:
            path = Path(config.prior_poses_abspath)
            array = np.load(path, allow_pickle=False)
            if array.shape != tuple(poses.shape):
                raise ValueError("Pose prior must match ordered train split")
            prior = torch.as_tensor(array, dtype=poses.dtype, device=poses.device)
        if not torch.isfinite(prior).all():
            raise ValueError("Nonfinite pose prior")
        if config.enabled and config.noise_enabled:
            generator = torch.Generator(device=poses.device).manual_seed(config.noise_seed)
            noise = torch.zeros(len(poses), 6, device=poses.device)
            for offset, scale in [(0, math.radians(config.noise_rot_deg)), (3, config.noise_trans_mm)]:
                if scale > 0:
                    draw = torch.randn(len(poses), 3, generator=generator, device=poses.device)
                    noise[:, offset : offset + 3] = torch.nn.functional.normalize(draw, dim=-1) * scale
            prior = prior @ _euler_xyz_to_matrix(noise)
        self.register_buffer("prior_poses", prior)
        trainable = config.enabled and config.trainable
        self.pose_delta_rot = nn.Parameter(
            torch.zeros(len(poses), 3), requires_grad=trainable and config.optimize_rotation
        )
        self.pose_delta_trans = nn.Parameter(
            torch.zeros(len(poses), 3), requires_grad=trainable and config.optimize_translation
        )
        self.step = 0

    def is_optimizing(self):
        return (
            self.config.enabled
            and self.config.trainable
            and self.step >= self.config.start_iter
            and (self.config.end_iter < 0 or self.step <= self.config.end_iter)
        )

    def forward(self, indices):
        value = torch.cat([self.pose_delta_rot[indices], self.pose_delta_trans[indices]], -1)
        if self.step < self.config.start_iter:
            value = value.detach() * 0
        elif not self.is_optimizing():
            value = value.detach()
        return _euler_xyz_to_matrix(value)[..., :3, :]

    def probe_pose(self, camera, pose):
        if not self.config.enabled or not camera.metadata or "frame_id" not in camera.metadata:
            return pose
        index = self.indices.get(int(camera.metadata["frame_id"].item()))
        if index is None:
            return pose
        residual = torch.eye(4, device=pose.device)
        residual[:3] = self(torch.tensor(index, device=pose.device))
        return self.prior_poses[index] @ residual

    def apply_to_camera(self, camera):
        pose = torch.eye(4, device=camera.device)
        pose[:3] = camera.camera_to_worlds.reshape(3, 4)
        pose[:3, 1:3] *= -1
        result = self.probe_pose(camera, pose)[:3].clone()
        result[:, 1:3] *= -1
        return result[None]

    def get_correction_matrices(self):
        return self(torch.arange(len(self.indices), device=self.prior_poses.device))

    def get_metrics_dict(self, metrics):
        if not self.config.enabled:
            return
        metrics["pose_residual_translation_mm"] = self.pose_delta_trans.detach().norm(dim=-1).mean()
        metrics["pose_residual_rotation_deg"] = torch.rad2deg(self.pose_delta_rot.detach().norm(dim=-1)).mean()

    def get_param_groups(self, groups):
        for name, param in [("pose_rot", self.pose_delta_rot), ("pose_trans", self.pose_delta_trans)]:
            if param.requires_grad:
                groups[name] = [param]

    def get_loss_dict(self, losses, index):
        if not self.is_optimizing():
            return
        for suffix, param in [("rot", self.pose_delta_rot), ("trans", self.pose_delta_trans)]:
            weight = getattr(self.config, "lambda_prior_" + suffix)
            losses["pose_prior_" + suffix] = param[index].square().sum() * (
                self.config.lambda_prior if weight is None else weight
            )
            if len(param) > 2:
                losses["pose_smooth_" + suffix] = (param[2:] - 2 * param[1:-1] + param[:-2]).square().sum(
                    -1
                ).mean() * getattr(self.config, "lambda_smooth_" + suffix)

    @torch.no_grad()
    def clamp_residuals(self):
        for param, bound in [
            (self.pose_delta_rot, math.radians(self.config.residual_bound_rot_deg)),
            (self.pose_delta_trans, self.config.residual_bound_trans_mm),
        ]:
            if bound > 0:
                param.clamp_(-bound, bound)


def frame_weights(config, frame_ids):
    if not config.enabled:
        return torch.ones(len(frame_ids))
    values = torch.from_numpy(np.load(config.uncertainty_abspath, allow_pickle=False).astype(np.float32))
    if values.shape != (len(frame_ids),) or not torch.isfinite(values).all():
        raise ValueError("Uncertainty weights must match ordered train frames and be finite")
    if not 0 < config.min_weight <= config.max_weight:
        raise ValueError("Invalid uncertainty bounds")
    if config.weight_mode == "exp":
        values = config.min_weight + (config.max_weight - config.min_weight) * torch.exp(-config.alpha * values)
    elif config.weight_mode != "direct":
        raise ValueError("Unknown uncertainty weight mode")
    return values.clamp(config.min_weight, config.max_weight)
