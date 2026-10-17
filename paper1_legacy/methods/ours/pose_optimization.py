from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn


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


def _rotation_error_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    rel = gt[..., :3, :3].transpose(-1, -2) @ pred[..., :3, :3]
    cos_theta = ((rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _translation_error_mm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred[..., :3, 3] - gt[..., :3, 3], dim=-1)


def _inverse_transform(t: torch.Tensor) -> torch.Tensor:
    r = t[..., :3, :3]
    trans = t[..., :3, 3]
    out = torch.zeros_like(t)
    out[..., :3, :3] = r.transpose(-1, -2)
    out[..., :3, 3] = -(out[..., :3, :3] @ trans.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def _center_similarity_aligned_error_mm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Similarity-align predicted centers to GT centers and return per-frame center error."""
    x = pred[..., :3, 3]
    y = gt[..., :3, 3]
    if x.shape[0] < 3:
        return _translation_error_mm(pred, gt)
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x0 = x - x_mean
    y0 = y - y_mean
    cov = (y0.transpose(0, 1) @ x0) / float(x.shape[0])
    try:
        u, s, vh = torch.linalg.svd(cov)
    except RuntimeError:
        return _translation_error_mm(pred, gt)
    d = torch.ones(3, device=x.device, dtype=x.dtype)
    if torch.det(u @ vh) < 0:
        d[-1] = -1.0
    rot = u @ torch.diag(d) @ vh
    var_x = (x0.pow(2).sum()) / float(x.shape[0])
    scale = (s * d).sum() / var_x.clamp_min(1e-8)
    trans = y_mean.squeeze(0) - scale * (rot @ x_mean.squeeze(0))
    aligned = scale * (x @ rot.transpose(0, 1)) + trans
    return torch.linalg.norm(aligned - y, dim=-1)


def _relative_pose_errors(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    if pred.shape[0] < 2:
        zero = pred.new_zeros((1,))
        return {"rpe_rot": zero, "rpe_trans": zero, "pred_step": zero, "gt_step": zero}
    pred_rel = _inverse_transform(pred[:-1]) @ pred[1:]
    gt_rel = _inverse_transform(gt[:-1]) @ gt[1:]
    return {
        "rpe_rot": _rotation_error_deg(pred_rel, gt_rel),
        "rpe_trans": _translation_error_mm(pred_rel, gt_rel),
        "pred_step": torch.linalg.norm(pred_rel[..., :3, 3], dim=-1),
        "gt_step": torch.linalg.norm(gt_rel[..., :3, 3], dim=-1),
    }


class OursPoseManager(nn.Module):
    """Noisy PEM-pose prior and optional per-frame residual pose optimizer."""

    def __init__(self, cfg, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.gt_poses: Optional[torch.Tensor] = None
        self.prior_poses: Optional[torch.Tensor] = None
        self.pose_delta_rot: Optional[nn.Parameter] = None
        self.pose_delta_trans: Optional[nn.Parameter] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    @property
    def pcfg(self):
        return self.cfg.model.pose_optimization

    @property
    def pose_delta(self) -> Optional[torch.Tensor]:
        if self.pose_delta_rot is None or self.pose_delta_trans is None:
            return None
        return torch.cat([self.pose_delta_rot, self.pose_delta_trans], dim=-1)

    def setup_from_scene(self, scene) -> None:
        n = len(scene.train_container)
        scene_prior = torch.stack(
            [scene.train_container.get(i).pose.detach().clone() for i in range(n)], dim=0
        ).to(self.device)
        file_poses = getattr(scene.train_container, "file_poses", None)
        if file_poses is not None and tuple(file_poses.shape) == tuple(scene_prior.shape):
            gt = file_poses.detach().clone().to(self.device)
            print("[OursPose] GT diagnostics use original file poses before pose_mode overrides.")
        else:
            gt = scene_prior.detach().clone()
            print("[OursPose] GT diagnostics use current scene poses; original file poses unavailable.")
        self.gt_poses = gt
        base_prior = self._load_external_prior(scene_prior)
        self.prior_poses = self._make_noisy_prior(base_prior)
        has_external_prior = bool(str(getattr(self.pcfg, "prior_poses_abspath", "")).strip())
        if has_external_prior:
            print(
                "[OursPose] Loaded external PEM pose prior:"
                f" path={os.path.abspath(os.path.expanduser(str(self.pcfg.prior_poses_abspath)))},"
                f" shape={tuple(self.prior_poses.shape)}"
            )
        if has_external_prior or bool(self.pcfg.noise_enabled):
            self._replace_scene_poses(scene, self.prior_poses)
        if bool(self.pcfg.noise_enabled):
            print(
                "[OursPose] Applied synthetic residual noise to pose prior:"
                f" rot={float(self.pcfg.noise_rot_deg):.4f} deg,"
                f" trans={float(self.pcfg.noise_trans_mm):.4f} mm,"
                f" seed={int(self.pcfg.noise_seed)}"
            )
        else:
            print("[OursPose] Pose prior uses input prior without added synthetic noise.")

        if bool(self.pcfg.trainable):
            self.pose_delta_rot = nn.Parameter(torch.zeros((n, 3), device=self.device, dtype=gt.dtype))
            self.pose_delta_trans = nn.Parameter(torch.zeros((n, 3), device=self.device, dtype=gt.dtype))
            param_groups = []
            if bool(self.pcfg.optimize_rotation):
                lr_rot = self._cfg_float("lr_rot", float(self.pcfg.lr))
                param_groups.append({"params": [self.pose_delta_rot], "lr": lr_rot, "name": "pose_rot"})
            else:
                self.pose_delta_rot.requires_grad_(False)
                lr_rot = 0.0
            if bool(self.pcfg.optimize_translation):
                lr_trans = self._cfg_float("lr_trans", float(self.pcfg.lr))
                param_groups.append({"params": [self.pose_delta_trans], "lr": lr_trans, "name": "pose_trans"})
            else:
                self.pose_delta_trans.requires_grad_(False)
                lr_trans = 0.0
            self.optimizer = torch.optim.Adam(param_groups, eps=1e-15) if param_groups else None
            print(
                "[OursPose] Trainable residual pose enabled:"
                f" start_iter={int(self.pcfg.start_iter)}, end_iter={int(self.pcfg.end_iter)},"
                f" lr_rot={lr_rot:.3e}, lr_trans={lr_trans:.3e},"
                f" lambda_prior={float(self.pcfg.lambda_prior):.3e},"
                f" lambda_smooth_rot={float(self.pcfg.lambda_smooth_rot):.3e},"
                f" lambda_smooth_trans={float(self.pcfg.lambda_smooth_trans):.3e},"
                f" residual_bound_rot_deg={float(self.pcfg.residual_bound_rot_deg):.4f},"
                f" residual_bound_trans_mm={float(self.pcfg.residual_bound_trans_mm):.4f},"
                f" pose_only=[{int(self.pcfg.pose_only_start_iter)}, {int(self.pcfg.pose_only_end_iter)}]"
            )
        else:
            self.pose_delta_rot = None
            self.pose_delta_trans = None
            self.optimizer = None
            print("[OursPose] Trainable residual pose disabled.")

    def _cfg_float(self, name: str, fallback: float) -> float:
        value = getattr(self.pcfg, name, None)
        return fallback if value is None else float(value)

    def _load_external_prior(self, gt: torch.Tensor) -> torch.Tensor:
        path = str(getattr(self.pcfg, "prior_poses_abspath", "")).strip()
        if not path:
            return gt.detach().clone()
        abspath = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(abspath):
            raise FileNotFoundError(f"External PEM pose prior not found: {abspath}")
        poses_np = np.load(abspath)
        if poses_np.shape != tuple(gt.shape):
            raise ValueError(
                "External PEM pose prior shape must match train poses: "
                f"got {poses_np.shape}, expected {tuple(gt.shape)} from {abspath}"
            )
        poses = torch.as_tensor(poses_np, device=self.device, dtype=gt.dtype)
        return poses.detach().clone()

    def _lambda_prior_rot(self) -> float:
        return self._cfg_float("lambda_prior_rot", float(self.pcfg.lambda_prior))

    def _lambda_prior_trans(self) -> float:
        return self._cfg_float("lambda_prior_trans", float(self.pcfg.lambda_prior))

    def _make_noisy_prior(self, gt: torch.Tensor) -> torch.Tensor:
        if not bool(self.pcfg.noise_enabled):
            return gt.detach().clone()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.pcfg.noise_seed))
        rot_scale = math.radians(float(self.pcfg.noise_rot_deg))
        trans_scale = float(self.pcfg.noise_trans_mm)
        noise = torch.zeros((gt.shape[0], 6), device=self.device, dtype=gt.dtype)
        if rot_scale > 0.0:
            noise[:, :3] = torch.randn((gt.shape[0], 3), device=self.device, generator=gen, dtype=gt.dtype)
            noise[:, :3] = noise[:, :3] / noise[:, :3].norm(dim=-1, keepdim=True).clamp_min(1e-8) * rot_scale
        if trans_scale > 0.0:
            noise[:, 3:] = torch.randn((gt.shape[0], 3), device=self.device, generator=gen, dtype=gt.dtype)
            noise[:, 3:] = noise[:, 3:] / noise[:, 3:].norm(dim=-1, keepdim=True).clamp_min(1e-8) * trans_scale
        # PEM noise is modeled as a local probe-frame residual. Right-multiplying
        # keeps the requested translation perturbation magnitude independent of
        # the frame's distance from the sweep origin.
        return (gt @ _euler_xyz_to_matrix(noise)).detach()

    def _replace_scene_poses(self, scene, poses: torch.Tensor) -> None:
        for i in range(len(scene.train_container)):
            current = scene.train_container.get(i)
            scene.train_container._slice_meta_list[i] = current.replace_pose(poses[i])

    def is_active(self, iteration: int) -> bool:
        return self.optimizer is not None and int(iteration) >= int(self.pcfg.start_iter)

    def is_optimizing(self, iteration: int) -> bool:
        end_iter = int(self.pcfg.end_iter)
        if end_iter >= 0 and int(iteration) > end_iter:
            return False
        return self.is_active(iteration)

    def is_pose_only(self, iteration: int) -> bool:
        start = int(self.pcfg.pose_only_start_iter)
        end = int(self.pcfg.pose_only_end_iter)
        return self.is_optimizing(iteration) and start >= 0 and end >= start and start <= int(iteration) <= end

    def current_pose(self, frame_id: int, iteration: int) -> torch.Tensor:
        if self.prior_poses is None:
            raise RuntimeError("OursPoseManager.setup_from_scene() must run before current_pose().")
        prior = self.prior_poses[int(frame_id)]
        if not self.is_active(iteration):
            return prior
        delta_params = torch.cat(
            [self.pose_delta_rot[int(frame_id)], self.pose_delta_trans[int(frame_id)]],
            dim=-1,
        )
        delta = _euler_xyz_to_matrix(delta_params)
        return prior @ delta

    def replace_slice_meta(self, slice_meta, iteration: int):
        pose = self.current_pose(int(slice_meta.frame_id), iteration)
        return slice_meta.replace_pose_differentiable(pose)

    def prior_loss(self, frame_id: int, iteration: int) -> torch.Tensor:
        if not self.is_optimizing(iteration):
            return torch.zeros((), device=self.device)
        rot = self.pose_delta_rot[int(frame_id)].pow(2).sum()
        trans = self.pose_delta_trans[int(frame_id)].pow(2).sum()
        return rot + trans

    def loss_terms(self, frame_id: int, iteration: int) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=self.device)
        if not self.is_optimizing(iteration):
            return {}
        rot_prior = self.pose_delta_rot[int(frame_id)].pow(2).sum() * self._lambda_prior_rot()
        trans_prior = self.pose_delta_trans[int(frame_id)].pow(2).sum() * self._lambda_prior_trans()
        terms = {
            "pose_prior_rot": rot_prior,
            "pose_prior_trans": trans_prior,
        }
        if float(self.pcfg.lambda_smooth_rot) > 0.0:
            terms["pose_smooth_rot"] = self._smooth_loss(self.pose_delta_rot) * float(self.pcfg.lambda_smooth_rot)
        else:
            terms["pose_smooth_rot"] = zero
        if float(self.pcfg.lambda_smooth_trans) > 0.0:
            terms["pose_smooth_trans"] = self._smooth_loss(self.pose_delta_trans) * float(self.pcfg.lambda_smooth_trans)
        else:
            terms["pose_smooth_trans"] = zero
        return terms

    def _smooth_loss(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] < 3:
            return values.new_zeros(())
        second_diff = values[2:] - 2.0 * values[1:-1] + values[:-2]
        return second_diff.pow(2).sum(dim=-1).mean()

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self, iteration: int) -> None:
        if self.is_optimizing(iteration):
            self.optimizer.step()
            self._clamp_pose_delta()

    def _clamp_pose_delta(self) -> None:
        if self.pose_delta_rot is not None:
            bound_rot = math.radians(float(getattr(self.pcfg, "residual_bound_rot_deg", 0.0) or 0.0))
            if bound_rot > 0.0:
                self.pose_delta_rot.data.clamp_(min=-bound_rot, max=bound_rot)
        if self.pose_delta_trans is not None:
            bound_trans = float(getattr(self.pcfg, "residual_bound_trans_mm", 0.0) or 0.0)
            if bound_trans > 0.0:
                self.pose_delta_trans.data.clamp_(min=-bound_trans, max=bound_trans)

    def metrics(self, frame_ids: Optional[Iterable[int]] = None, iteration: int = 0) -> Dict[str, float]:
        if self.gt_poses is None or self.prior_poses is None:
            return {}
        if frame_ids is None:
            n = self.gt_poses.shape[0]
            count = min(int(self.pcfg.eval_train_count), n)
            ids = torch.linspace(0, n - 1, count, device=self.device).round().long()
        else:
            ids = torch.as_tensor(list(frame_ids), device=self.device, dtype=torch.long)
            if ids.numel() == 0:
                return {}
        current = torch.stack([self.current_pose(int(i), iteration).detach() for i in ids.tolist()], dim=0)
        # Pose diagnostics are scalar logging only. Keep them on CPU to avoid
        # CUDA/NVRTC linalg compatibility issues on newer GPUs with older torch.
        current = current.float().cpu()
        gt = self.gt_poses[ids].detach().float().cpu()
        prior = self.prior_poses[ids].detach().float().cpu()
        cur_rot = _rotation_error_deg(current, gt)
        cur_trans = _translation_error_mm(current, gt)
        prior_rot = _rotation_error_deg(prior, gt)
        prior_trans = _translation_error_mm(prior, gt)
        cur_aligned = _center_similarity_aligned_error_mm(current, gt)
        prior_aligned = _center_similarity_aligned_error_mm(prior, gt)
        cur_rel = _relative_pose_errors(current, gt)
        prior_rel = _relative_pose_errors(prior, gt)
        out = {
            "pose/rot_err_deg_mean": float(cur_rot.mean().detach().cpu()),
            "pose/trans_err_mm_mean": float(cur_trans.mean().detach().cpu()),
            "pose/prior_rot_err_deg_mean": float(prior_rot.mean().detach().cpu()),
            "pose/prior_trans_err_mm_mean": float(prior_trans.mean().detach().cpu()),
            "pose/center_aligned_err_mm_mean": float(cur_aligned.mean().detach().cpu()),
            "pose/center_aligned_err_mm_max": float(cur_aligned.max().detach().cpu()),
            "pose/prior_center_aligned_err_mm_mean": float(prior_aligned.mean().detach().cpu()),
            "pose/prior_center_aligned_err_mm_max": float(prior_aligned.max().detach().cpu()),
            "pose/rpe_rot_deg_mean": float(cur_rel["rpe_rot"].mean().detach().cpu()),
            "pose/rpe_trans_mm_mean": float(cur_rel["rpe_trans"].mean().detach().cpu()),
            "pose/prior_rpe_rot_deg_mean": float(prior_rel["rpe_rot"].mean().detach().cpu()),
            "pose/prior_rpe_trans_mm_mean": float(prior_rel["rpe_trans"].mean().detach().cpu()),
            "pose/pred_step_mm_mean": float(cur_rel["pred_step"].mean().detach().cpu()),
            "pose/gt_step_mm_mean": float(cur_rel["gt_step"].mean().detach().cpu()),
        }
        if self.pose_delta_rot is not None:
            out["pose/residual_rot_deg_abs_max"] = float(
                torch.rad2deg(self.pose_delta_rot.detach().abs().max()).cpu()
            )
            out["pose/residual_rot_deg_abs_mean"] = float(
                torch.rad2deg(self.pose_delta_rot.detach().abs()).mean().cpu()
            )
        if self.pose_delta_trans is not None:
            out["pose/residual_trans_mm_abs_max"] = float(self.pose_delta_trans.detach().abs().max().cpu())
            out["pose/residual_trans_mm_abs_mean"] = float(self.pose_delta_trans.detach().abs().mean().cpu())
        return out

    def save(self, path: str) -> None:
        torch.save(
            {
                "gt_poses": self.gt_poses.detach().cpu() if self.gt_poses is not None else None,
                "prior_poses": self.prior_poses.detach().cpu() if self.prior_poses is not None else None,
                "pose_delta": self.pose_delta.detach().cpu() if self.pose_delta is not None else None,
                "pose_delta_rot": self.pose_delta_rot.detach().cpu() if self.pose_delta_rot is not None else None,
                "pose_delta_trans": self.pose_delta_trans.detach().cpu() if self.pose_delta_trans is not None else None,
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location=self.device)
        if blob.get("gt_poses") is not None:
            self.gt_poses = blob["gt_poses"].to(self.device)
        if blob.get("prior_poses") is not None:
            self.prior_poses = blob["prior_poses"].to(self.device)
        if self.pose_delta_rot is not None and blob.get("pose_delta_rot") is not None:
            self.pose_delta_rot.data.copy_(blob["pose_delta_rot"].to(self.device))
        if self.pose_delta_trans is not None and blob.get("pose_delta_trans") is not None:
            self.pose_delta_trans.data.copy_(blob["pose_delta_trans"].to(self.device))
        if self.pose_delta_rot is not None and self.pose_delta_trans is not None and blob.get("pose_delta") is not None:
            old_delta = blob["pose_delta"].to(self.device)
            self.pose_delta_rot.data.copy_(old_delta[:, :3])
            self.pose_delta_trans.data.copy_(old_delta[:, 3:])
        if self.optimizer is not None and blob.get("optimizer") is not None:
            self.optimizer.load_state_dict(blob["optimizer"])
