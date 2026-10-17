"""
原子图像指标：手写基础指标 + 基于 pyiqa API 的感知/结构指标。
与 utils/loss 对称，均在 methods/.../computers 中统一装配。
"""
import torch
import torch.nn as nn

from .registry import register_metric_criterion

# 延迟导入，避免未安装 pyiqa 时无法导入本模块
def _get_pyiqa():
    import pyiqa
    return pyiqa


# ---------- 手写原子指标（与 loss 可共享数学定义，此处仅用于评估） ----------

@register_metric_criterion("psnr")
class PSNRMetric(nn.Module):
    """PSNR，越高越好。手写实现，稳定且无需额外依赖。"""
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 0.0, self.max_val)
        gt = torch.clamp(gt, 0.0, self.max_val)
        mse = ((pred - gt) ** 2).mean()
        if mse == 0:
            return torch.tensor(float("inf"), device=pred.device)
        return 20 * torch.log10(self.max_val / torch.sqrt(mse))


@register_metric_criterion("l1")
class L1Metric(nn.Module):
    """L1 / MAE，越低越好。"""
    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        return torch.abs(pred - gt).mean()


# ---------- 基于 pyiqa API 的原子指标 ----------

def _to_4d_bgr(x: torch.Tensor) -> torch.Tensor:
    """将 pred/gt 变为 (1, 3, H, W)，单通道则复制为 3 通道，供 pyiqa 使用。"""
    x = torch.clamp(x, 0.0, 1.0)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


@register_metric_criterion("ssim")
class SSIMMetric(nn.Module):
    """SSIM，越高越好。内部使用 pyiqa。"""
    def __init__(self, device: torch.device):
        super().__init__()
        pyiqa = _get_pyiqa()
        self._metric = pyiqa.create_metric("ssim", device=device)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        p = _to_4d_bgr(pred)
        g = _to_4d_bgr(gt)
        out = self._metric(p, g)
        return out if isinstance(out, torch.Tensor) else torch.tensor(out, device=pred.device)


@register_metric_criterion("lpips")
class LPIPSMetric(nn.Module):
    """LPIPS，越低越好。"""
    def __init__(self, device: torch.device):
        super().__init__()
        pyiqa = _get_pyiqa()
        self._metric = pyiqa.create_metric("lpips", device=device)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        p = _to_4d_bgr(pred)
        g = _to_4d_bgr(gt)
        out = self._metric(p, g)
        return out if isinstance(out, torch.Tensor) else torch.tensor(out, device=pred.device)


@register_metric_criterion("dists")
class DISTSMetric(nn.Module):
    """DISTS，越低越好。"""
    def __init__(self, device: torch.device):
        super().__init__()
        pyiqa = _get_pyiqa()
        self._metric = pyiqa.create_metric("dists", device=device)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        p = _to_4d_bgr(pred)
        g = _to_4d_bgr(gt)
        out = self._metric(p, g)
        return out if isinstance(out, torch.Tensor) else torch.tensor(out, device=pred.device)


@register_metric_criterion("gmsd")
class GMSDMetric(nn.Module):
    """GMSD，越低越好。"""
    def __init__(self, device: torch.device):
        super().__init__()
        pyiqa = _get_pyiqa()
        self._metric = pyiqa.create_metric("gmsd", device=device)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        p = _to_4d_bgr(pred)
        g = _to_4d_bgr(gt)
        out = self._metric(p, g)
        return out if isinstance(out, torch.Tensor) else torch.tensor(out, device=pred.device)
