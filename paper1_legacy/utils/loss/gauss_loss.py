import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss_criterion


@register_loss_criterion("scale_wall")
class ScaleWallLoss(nn.Module): 
    def __init__(self, max_scales_mm=1.0, min_scales_mm=0.1):
        super().__init__()
        self.max_scales_mm = max_scales_mm
        self.min_scales_mm = min_scales_mm

    def forward(self, scales_mm: torch.Tensor):
        excess_small = torch.relu(self.min_scales_mm - scales_mm)
        excess_large = torch.relu(scales_mm - self.max_scales_mm)

        # 【关键修改】：巨物和极小值都使用平方惩罚
        loss_upper = excess_large.pow(2).mean()
        loss_lower = excess_small.pow(2).mean()

        loss = loss_upper + loss_lower

        return loss


@register_loss_criterion("scale_aniso")
class ScaleAnisotropicLoss(nn.Module):
    def __init__(self, max_aniso_ratio=5.0):
        super().__init__()
        self.max_aniso_ratio = max_aniso_ratio

    def forward(self, scales_mm: torch.Tensor):
        """
        计算各向异性惩罚损失 (Anisotropic Scale Penalty)
        强制高斯球在三个局部轴上的尺度差距不能过大，防止退化为“薄饼”或“细针”。

        Args:
            scales_linear: (N, 3) tensor, 经过 exp 激活后的物理尺度
            max_ratio: 允许的最大长短轴比例。例如 5.0 表示最长轴最多只能是最短轴的 5 倍。
        """
        eps = 1e-6  # 防止除以零的极小值
        
        # 获取每个高斯球在局部三轴上的最大尺度和最小尺度
        max_scales, _ = scales_mm.max(dim=-1)
        min_scales, _ = scales_mm.min(dim=-1)
        
        # 计算当前长短轴的比值
        ratios = max_scales / (min_scales + eps)
        
        # 构建巨物墙 (ReLU Barrier)
        # 如果 ratio <= max_ratio，ReLU 输出 0，完全不产生梯度，允许正常的不规则形变
        # 如果 ratio > max_ratio，产生线性增长的惩罚梯度，迫使模型把短轴拉长，或把长轴缩短
        excess_ratio = torch.relu(ratios - self.max_aniso_ratio)
        
        # 【关键修改】：改用 pow(2) 进行指数级惩罚。
        # 越界越严重的高斯球，挨的刀越重，并且在求 mean 时不容易被彻底稀释。
        loss_aniso = excess_ratio.pow(2).mean()
        
        return loss_aniso


@register_loss_criterion("opacity")
class OpacityLoss(nn.Module):
    def __init__(self, target_opacity=0.3):
        super().__init__()
        # 目标是不希望高斯球太实心，鼓励半透明
        self.target_opacity = target_opacity

    def forward(self, opacities: torch.Tensor):
        # 对超出我们期望的半透明阈值的高斯球施加平方惩罚
        excess_opacity = torch.relu(opacities - self.target_opacity)
        return excess_opacity.pow(2).mean()


@register_loss_criterion("attenuation_l1")
class AttenuationL1Loss(nn.Module):
    def forward(self, attenuations: torch.Tensor):
        return attenuations.abs().mean()


@register_loss_criterion("attenuation_tail")
class AttenuationTailLoss(nn.Module):
    def __init__(self, max_attenuation=0.05):
        super().__init__()
        self.max_attenuation = max_attenuation

    def forward(self, attenuations: torch.Tensor):
        excess_attenuation = torch.relu(attenuations - self.max_attenuation)
        return excess_attenuation.pow(2).mean()
