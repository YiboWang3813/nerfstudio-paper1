import torch
from dataclasses import dataclass
from typing import Dict, Tuple

from configs.base.scene import SamplerParams

# =========================================================================
# Part 2: Unified Gaussian Sampler (筛选者)
# 职责：根据亮度、均匀度策略，从全量像素中筛选出用于初始化的高斯
# =========================================================================

class UnifiedGaussianSampler:
    """
    [统一采样器] 
    同时支持 Online (小数据) 和 Offline (大数据) 模式。
    通过内部的 `_safe_multinomial` 自动判断是否启用显存保护。
    """
    
    # CUDA multinomial 算子在处理超过 2^24 (约 1677万) 元素时可能崩溃或极慢
    # 我们设置一个安全阈值，超过此值触发两阶段采样
    SAFE_MULTINOMIAL_LIMIT = 16_000_000 

    def __init__(self, cfg: SamplerParams, device: torch.device):
        """
        Args:
            cfg: SamplerParams (含 uniform_ratio, intensity_threshold 等)
        """
        self.device = device
        self.ratio = cfg.uniform_ratio
        self.threshold = cfg.intensity_threshold
        
    @torch.no_grad()
    def sample(
        self,
        pixel_positions: torch.Tensor, 
        pixel_values: torch.Tensor,
        pixel_quats: torch.Tensor,
        pixel_scales: torch.Tensor,
        num_to_sample: int,
        jitter_radius: float,
    ) -> Dict[str, torch.Tensor]:
        """
        执行混合采样：Uniform (填补介质) + Importance (拟合结构)
        """
        P = pixel_positions.shape[0]
        
        # 1. 分配名额
        num_uniform = int(num_to_sample * self.ratio)
        num_importance = num_to_sample - num_uniform
        
        indices_list = []

        # 2. 均匀采样 (Uniform) - 极速
        # 作用：无视图像内容，随机撒点，保证背景介质有高斯覆盖
        if num_uniform > 0:
            indices_uni = torch.randint(0, P, (num_uniform,), device=self.device)
            indices_list.append(indices_uni)

        # 3. 重要性采样 (Importance) - 智能
        # 作用：根据像素亮度撒点，捕捉组织结构
        if num_importance > 0:
            indices_imp = self._sample_importance(pixel_values, num_importance, P)
            indices_list.append(indices_imp)

        # 4. 聚合与后处理
        final_indices = torch.cat(indices_list)
        
        # 继承属性 (Gather)
        selected_means = pixel_positions[final_indices].to(self.device)
        selected_values = pixel_values[final_indices].to(self.device)
        quats = pixel_quats[final_indices].to(self.device)
        scales = pixel_scales[final_indices].to(self.device)
        
        # 位置扰动 (Jitter)
        # 在初始位置附近微调，防止高斯完全重叠，也有助于梯度流
        noise = (torch.rand_like(selected_means) * 2.0 - 1.0) 
        means = selected_means + (noise * jitter_radius)

        return {
            "means": means,
            "intensities": selected_values,
            "scales": scales, 
            "quats": quats,
            "pixel_indices": final_indices,
        }

    def _sample_importance(self, pixel_values: torch.Tensor, k: int, total_pixels: int) -> torch.Tensor:
        """
        内部函数：安全的重要性采样逻辑
        """
        # A. 计算权重
        if pixel_values.dim() > 1:
            weights = pixel_values.mean(dim=1)
        else:
            weights = pixel_values

        # B. 阈值过滤 (获取有效候选点)
        valid_mask = weights > self.threshold
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        num_valid = valid_indices.numel()

        # C. 兜底逻辑：如果全黑，退化为均匀采样
        if num_valid == 0:
            return torch.randint(0, total_pixels, (k,), device=self.device)
        
        # D. 自动显存保护逻辑
        valid_weights = weights[valid_indices]
        
        if num_valid > self.SAFE_MULTINOMIAL_LIMIT:
            # === 分支 1: 大数据保护模式 (Offline 常用) ===
            # 策略：两阶段采样 (Two-stage sampling)
            # 先用 uniform 降采样到安全范围，保留分布的无偏性
            
            # 1. 随机选出安全数量的候选者索引 (Uniform)
            subset_selection = torch.randint(
                0, num_valid, (self.SAFE_MULTINOMIAL_LIMIT,), device=self.device
            )
            
            # 2. 映射回全局索引和权重
            safe_indices = valid_indices[subset_selection]
            safe_weights = valid_weights[subset_selection]
            
            # 3. 在安全子集中进行 Importance Sampling
            local_idx = torch.multinomial(safe_weights, k, replacement=True)
            return safe_indices[local_idx]
            
        else:
            # === 分支 2: 极速模式 (Online / Small Offline) ===
            # 直接采样，无需额外开销
            local_idx = torch.multinomial(valid_weights, k, replacement=True)
            return valid_indices[local_idx]
