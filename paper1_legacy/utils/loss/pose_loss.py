import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .registry import register_loss_criterion
from .image_loss import SSIMLoss


def _rel_pose(p_prev: torch.Tensor, p_curr: torch.Tensor) -> torch.Tensor:
    return p_curr @ torch.inverse(p_prev)


def _to_bchw(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 2:
        return img[None, None, ...]
    if img.dim() == 3:
        return img[None, ...]
    if img.dim() == 4:
        return img
    raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")


def _warp_prev_to_curr(
    prev_img_bchw: torch.Tensor,
    pose_prev: torch.Tensor,
    pose_curr: torch.Tensor,
    delta_lateral_mm: float,
    delta_axial_mm: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rel = torch.linalg.inv(pose_prev) @ pose_curr  # curr local -> prev local
    _, _, h, w = prev_img_bchw.shape
    device, dtype = prev_img_bchw.device, prev_img_bchw.dtype
    xs = (torch.arange(w, device=device, dtype=dtype) - (w - 1) * 0.5) * float(delta_lateral_mm)
    ys = (torch.arange(h, device=device, dtype=dtype) - (h - 1) * 0.5) * float(delta_axial_mm)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    zeros = torch.zeros_like(xx)
    ones = torch.ones_like(xx)
    pts_curr = torch.stack([xx, yy, zeros, ones], dim=-1).reshape(-1, 4).t()  # [4, H*W]
    pts_prev = (rel @ pts_curr).t().reshape(h, w, 4)
    x_prev = pts_prev[..., 0] / float(delta_lateral_mm) + (w - 1) * 0.5
    y_prev = pts_prev[..., 1] / float(delta_axial_mm) + (h - 1) * 0.5
    x_norm = (x_prev / max(w - 1, 1)) * 2.0 - 1.0
    y_norm = (y_prev / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    valid = ((x_norm >= -1.0) & (x_norm <= 1.0) & (y_norm >= -1.0) & (y_norm <= 1.0)).float()
    warped = F.grid_sample(
        prev_img_bchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    valid = valid.unsqueeze(0).unsqueeze(0)
    return warped, valid


@register_loss_criterion("pose_anchor")
class PoseAnchorLoss(nn.Module):
    """直接锚定 refined 不要偏离 coarse_curr 太远（小修模式核心约束）。"""

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0):
        super().__init__()
        self.rot_weight = float(rot_weight)
        self.trans_weight = float(trans_weight)

    def forward(self, p_refined_curr: torch.Tensor, p_coarse_curr: torch.Tensor) -> torch.Tensor:
        d = torch.inverse(p_coarse_curr) @ p_refined_curr
        rot_err = torch.norm(d[:3, :3] - torch.eye(3, device=d.device, dtype=d.dtype), p="fro")
        trans_err = torch.norm(d[:3, 3], p=2)
        return self.rot_weight * rot_err + self.trans_weight * trans_err


@register_loss_criterion("pose_prior")
class PosePriorLoss(nn.Module):
    """
    锚定 refined 相对运动贴近 coarse 相对运动。
    """

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0):
        super().__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def forward(
        self,
        p_refined_prev: torch.Tensor,
        p_refined_curr: torch.Tensor,
        p_coarse_prev: torch.Tensor,
        p_coarse_curr: torch.Tensor,
    ) -> torch.Tensor:
        t_ref = _rel_pose(p_refined_prev, p_refined_curr)
        t_coa = _rel_pose(p_coarse_prev, p_coarse_curr)
        d = torch.inverse(t_coa) @ t_ref
        rot_err = torch.norm(d[:3, :3] - torch.eye(3, device=d.device, dtype=d.dtype), p="fro")
        trans_err = torch.norm(d[:3, 3], p=2)
        return self.rot_weight * rot_err + self.trans_weight * trans_err


@register_loss_criterion("pose_smooth")
class PoseSmoothLoss(nn.Module):
    """
    一阶平滑：惩罚 refined 相邻帧相对位姿幅值。
    """

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0):
        super().__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def forward(self, p_refined_prev: torch.Tensor, p_refined_curr: torch.Tensor) -> torch.Tensor:
        t = _rel_pose(p_refined_prev, p_refined_curr)
        rot_err = torch.norm(t[:3, :3] - torch.eye(3, device=t.device, dtype=t.dtype), p="fro")
        trans_err = torch.norm(t[:3, 3], p=2)
        return self.rot_weight * rot_err + self.trans_weight * trans_err


@register_loss_criterion("pose_acc")
class PoseAccLoss(nn.Module):
    """
    二阶平滑：惩罚相邻相对运动变化。
    """

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0):
        super().__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def forward(
        self,
        p_refined_prevprev: torch.Tensor,
        p_refined_prev: torch.Tensor,
        p_refined_curr: torch.Tensor,
    ) -> torch.Tensor:
        t1 = _rel_pose(p_refined_prevprev, p_refined_prev)
        t2 = _rel_pose(p_refined_prev, p_refined_curr)
        rot_err = torch.norm(t2[:3, :3] - t1[:3, :3], p="fro")
        trans_err = torch.norm(t2[:3, 3] - t1[:3, 3], p=2)
        return self.rot_weight * rot_err + self.trans_weight * trans_err


@register_loss_criterion("pose_aux_rec")
class PoseAuxRecLoss(nn.Module):
    """
    Pose 辅助重建损失：L1 + SSIM（同图像重建形式）。
    """

    def __init__(self, l1_weight: float = 0.8, ssim_weight: float = 0.2, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.ssim = SSIMLoss(window_size=window_size, sigma=sigma)

    def forward(self, pred_aux: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        l1 = torch.mean(torch.abs(pred_aux - gt))
        ssim = self.ssim(pred_aux, gt)
        return self.l1_weight * l1 + self.ssim_weight * ssim


@register_loss_criterion("pose_registration")
class PoseRegistrationLoss(nn.Module):
    """
    Pose 配准主损失（不使用 GT pose）：
    - 光度项：Charbonnier，鲁棒于少量异常像素；
    - 梯度项：约束边缘对齐，提升位姿可观测性。
    """

    def __init__(
        self,
        photo_weight: float = 1.0,
        grad_weight: float = 0.2,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.photo_weight = float(photo_weight)
        self.grad_weight = float(grad_weight)
        self.charbonnier_eps = float(charbonnier_eps)

    def _charbonnier(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x * x + self.charbonnier_eps * self.charbonnier_eps)

    def _image_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    def forward(self, pred_aux: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        photo = self._charbonnier(pred_aux - gt).mean()
        gx_p, gy_p = self._image_grad(pred_aux)
        gx_g, gy_g = self._image_grad(gt)
        grad = 0.5 * (
            self._charbonnier(gx_p - gx_g).mean() +
            self._charbonnier(gy_p - gy_g).mean()
        )
        return self.photo_weight * photo + self.grad_weight * grad


@register_loss_criterion("pose_neighbor_registration")
class PoseNeighborRegistrationLoss(nn.Module):
    """邻帧可微配准：warp I_{i-1} 到 i，与 I_i 做 Charbonnier+SSIM。"""

    def __init__(
        self,
        photo_weight: float = 1.0,
        ssim_weight: float = 0.2,
        charbonnier_eps: float = 1e-3,
        window_size: int = 11,
        sigma: float = 1.5,
        min_overlap_ratio: float = 0.3,
    ):
        super().__init__()
        self.photo_weight = float(photo_weight)
        self.ssim_weight = float(ssim_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.ssim = SSIMLoss(window_size=window_size, sigma=sigma)
        self.min_overlap_ratio = float(min_overlap_ratio)

    def _charbonnier(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.charbonnier_eps
        return torch.sqrt(x * x + eps * eps)

    def forward(
        self,
        prev_image: torch.Tensor,
        curr_image: torch.Tensor,
        pose_prev: torch.Tensor,
        pose_curr: torch.Tensor,
        delta_lateral_mm: float,
        delta_axial_mm: float,
    ) -> torch.Tensor:
        prev_img = _to_bchw(prev_image)
        curr_img = _to_bchw(curr_image)
        curr_hat, valid = _warp_prev_to_curr(
            prev_img,
            pose_prev,
            pose_curr,
            delta_lateral_mm=delta_lateral_mm,
            delta_axial_mm=delta_axial_mm,
        )
        overlap = float(valid.mean().detach().item())
        if overlap < self.min_overlap_ratio:
            return torch.zeros((), device=curr_img.device, dtype=curr_img.dtype)
        photo = (self._charbonnier(curr_hat - curr_img) * valid).sum() / valid.sum().clamp_min(1e-8)
        # SSIM 不支持 mask，采用 masked-image 近似以抑制越界区域伪匹配。
        ssim = self.ssim(curr_hat * valid, curr_img * valid)
        return self.photo_weight * photo + self.ssim_weight * ssim


@register_loss_criterion("pose_cycle")
class PoseCycleLoss(nn.Module):
    """前后向 cycle consistency：prev->curr->prev 与 curr->prev->curr。"""

    def __init__(
        self,
        photo_weight: float = 1.0,
        ssim_weight: float = 0.1,
        charbonnier_eps: float = 1e-3,
        window_size: int = 11,
        sigma: float = 1.5,
        min_overlap_ratio: float = 0.3,
    ):
        super().__init__()
        self.photo_weight = float(photo_weight)
        self.ssim_weight = float(ssim_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.ssim = SSIMLoss(window_size=window_size, sigma=sigma)
        self.min_overlap_ratio = float(min_overlap_ratio)

    def _charbonnier(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.charbonnier_eps
        return torch.sqrt(x * x + eps * eps)

    def _masked_photo_ssim(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        overlap = float(valid.mean().detach().item())
        if overlap < self.min_overlap_ratio:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        photo = (self._charbonnier(pred - gt) * valid).sum() / valid.sum().clamp_min(1e-8)
        ssim = self.ssim(pred * valid, gt * valid)
        return self.photo_weight * photo + self.ssim_weight * ssim

    def forward(
        self,
        prev_image: torch.Tensor,
        curr_image: torch.Tensor,
        pose_prev: torch.Tensor,
        pose_curr: torch.Tensor,
        delta_lateral_mm: float,
        delta_axial_mm: float,
    ) -> torch.Tensor:
        prev_img = _to_bchw(prev_image)
        curr_img = _to_bchw(curr_image)
        curr_hat, valid_f = _warp_prev_to_curr(
            prev_img, pose_prev, pose_curr, delta_lateral_mm, delta_axial_mm
        )
        prev_rec, valid_b = _warp_prev_to_curr(
            curr_hat, pose_curr, pose_prev, delta_lateral_mm, delta_axial_mm
        )
        loss_prev = self._masked_photo_ssim(prev_rec, prev_img, valid_f * valid_b)

        prev_hat, valid_fb = _warp_prev_to_curr(
            curr_img, pose_curr, pose_prev, delta_lateral_mm, delta_axial_mm
        )
        curr_rec, valid_bf = _warp_prev_to_curr(
            prev_hat, pose_prev, pose_curr, delta_lateral_mm, delta_axial_mm
        )
        loss_curr = self._masked_photo_ssim(curr_rec, curr_img, valid_fb * valid_bf)
        return 0.5 * (loss_prev + loss_curr)


@register_loss_criterion("pose_const_vel")
class PoseConstVelLoss(nn.Module):
    """Temporal constant-velocity 约束（与二阶差分类似，但显式命名用于论文叙事）。"""

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0):
        super().__init__()
        self.rot_weight = float(rot_weight)
        self.trans_weight = float(trans_weight)

    def forward(
        self,
        p_refined_prevprev: torch.Tensor,
        p_refined_prev: torch.Tensor,
        p_refined_curr: torch.Tensor,
    ) -> torch.Tensor:
        t_prev = _rel_pose(p_refined_prevprev, p_refined_prev)
        t_curr = _rel_pose(p_refined_prev, p_refined_curr)
        rot_err = torch.norm(t_curr[:3, :3] - t_prev[:3, :3], p="fro")
        trans_err = torch.norm(t_curr[:3, 3] - t_prev[:3, 3], p=2)
        return self.rot_weight * rot_err + self.trans_weight * trans_err


@register_loss_criterion("pose_gt_delta")
class PoseGtDeltaLoss(nn.Module):
    """主损失：约束 coarse->refined 残差接近 coarse->gt 残差。"""

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0):
        super().__init__()
        self.rot_weight = float(rot_weight)
        self.trans_weight = float(trans_weight)

    def forward(
        self,
        pose_coarse_curr: torch.Tensor,
        pose_refined_curr: torch.Tensor,
        pose_gt_curr: torch.Tensor,
    ) -> torch.Tensor:
        d_pred = torch.inverse(pose_coarse_curr) @ pose_refined_curr
        d_gt = torch.inverse(pose_coarse_curr) @ pose_gt_curr
        rot_err = torch.norm(d_pred[:3, :3] - d_gt[:3, :3], p="fro")
        trans_err = torch.norm(d_pred[:3, 3] - d_gt[:3, 3], p=2)
        return self.rot_weight * rot_err + self.trans_weight * trans_err


@register_loss_criterion("pose_occ_conf")
class PoseOccConfidenceLoss(nn.Module):
    """辅助损失：GS occupancy -> confidence map，对图像项加权。"""

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
        occ = occupancy_hw
        if self.detach_confidence:
            occ = occ.detach()
        occ = occ.clamp(0.0, 1.0)
        conf = occ.pow(max(self.confidence_gamma, 1e-6)).clamp(self.confidence_min, 1.0)
        while conf.ndim < pred_aux.ndim:
            conf = conf.unsqueeze(0)
        diff = pred_aux - gt
        val = torch.sqrt(diff * diff + self.charbonnier_epsilon * self.charbonnier_epsilon)
        num = torch.sum(val * conf)
        den = torch.sum(conf).clamp_min(1e-8)
        return num / den
