import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss_criterion


def _to_clamped_4d_rgb(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        if x.shape[0] in (1, 3):
            x = x.unsqueeze(0)
        elif x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.unsqueeze(1)
    elif x.dim() == 4 and x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
        x = x.permute(0, 3, 1, 2)
    if x.dim() != 4 or x.shape[1] not in (1, 3):
        raise ValueError(f"LPIPS expects image tensor convertible to NCHW, got {tuple(x.shape)}")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x

@register_loss_criterion("mse")
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return ((pred - gt) ** 2).mean()


@register_loss_criterion("sobel")
class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        # pred, target: (H,W) or (1,H,W)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        gx_p = F.conv2d(pred, self.sobel_x, padding=1)
        gy_p = F.conv2d(pred, self.sobel_y, padding=1)
        gx_t = F.conv2d(target, self.sobel_x, padding=1)
        gy_t = F.conv2d(target, self.sobel_y, padding=1)

        return (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()


@register_loss_criterion("l1")
class L1Loss(nn.Module):
    def __init__(self, clamp_inputs: bool = False):
        super().__init__()
        self.clamp_inputs = bool(clamp_inputs)

    def forward(self, pred, gt):
        if self.clamp_inputs:
            pred = torch.clamp(pred, 0.0, 1.0)
            gt = torch.clamp(gt, 0.0, 1.0)
        return torch.abs(pred - gt).mean()


@register_loss_criterion("highpass_l1")
class HighPassL1Loss(nn.Module):
    def __init__(self, kernel_size: int = 9, sigma: float = 2.0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        coords = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / max(self.sigma, 1.0e-6)) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-8)
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, self.kernel_size, self.kernel_size)
        self.register_buffer("kernel", kernel_2d)

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4:
            return x
        raise ValueError(f"Expected image tensor with dim 2, 3, or 4, got {tuple(x.shape)}")

    def forward(self, pred, gt):
        pred = self._to_nchw(pred)
        gt = self._to_nchw(gt)
        c = int(pred.shape[1])
        kernel = self.kernel.to(device=pred.device, dtype=pred.dtype).expand(c, 1, -1, -1)
        pad = self.kernel_size // 2
        pred_low = F.conv2d(pred, kernel, padding=pad, groups=c)
        gt_low = F.conv2d(gt, kernel, padding=pad, groups=c)
        return torch.abs((pred - pred_low) - (gt - gt_low)).mean()


@register_loss_criterion("ssim")
class SSIMLoss(nn.Module):
    """
    Return:
        loss = 1 - ssim
    """
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        downsample_enabled: bool = False,
        downsample_factor: int = 1,
        downsample_mode: str = "area",
        clamp_inputs: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.downsample_enabled = bool(downsample_enabled)
        self.downsample_factor = max(1, int(downsample_factor))
        self.downsample_mode = str(downsample_mode).lower()
        self.clamp_inputs = bool(clamp_inputs)
        self.channel = None
        self.window = None

    @staticmethod
    def _gaussian_kernel(window_size, sigma):
        coords = torch.arange(window_size).float()
        gauss = torch.exp(-(coords - window_size//2)**2 / (2*sigma*sigma))
        return gauss / gauss.sum()

    def _create_window(self, channel, device):
        _1D = self._gaussian_kernel(self.window_size, self.sigma).unsqueeze(1)
        _2D = (_1D @ _1D.t()).unsqueeze(0).unsqueeze(0)
        window = _2D.expand(channel, 1, self.window_size, self.window_size)
        return window.to(device)

    def forward(self, img1, img2):
        if self.clamp_inputs:
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)

        # reshape to [B,C,H,W]
        if img1.dim() == 2:
            img1 = img1[None, None]
            img2 = img2[None, None]
        elif img1.dim() == 3:
            img1 = img1[None]
            img2 = img2[None]

        if self.downsample_enabled and self.downsample_factor > 1:
            f = self.downsample_factor
            if self.downsample_mode == "stride":
                # Fastest path: strided spatial subsampling (no pooling kernel launch).
                img1 = img1[..., ::f, ::f]
                img2 = img2[..., ::f, ::f]
            elif self.downsample_mode == "avgpool":
                img1 = F.avg_pool2d(img1, kernel_size=f, stride=f)
                img2 = F.avg_pool2d(img2, kernel_size=f, stride=f)
            else:
                scale = 1.0 / float(f)
                img1 = F.interpolate(
                    img1, scale_factor=scale, mode="area", recompute_scale_factor=False
                )
                img2 = F.interpolate(
                    img2, scale_factor=scale, mode="area", recompute_scale_factor=False
                )

        B, C, H, W = img1.shape

        if self.window is None or C != self.channel:
            self.window = self._create_window(C, img1.device)
            self.channel = C

        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 使用 F.relu 确保方差非负，避免数值不稳定产生的黑点
        sigma1_sq = F.relu(F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=C) - mu1_sq)
        sigma2_sq = F.relu(F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=C) - mu2_sq)
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=C) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = ssim_n / ssim_d
        ssim_score = ssim_map.mean()

        return 1 - ssim_score    # loss


@register_loss_criterion("lpips")
class LPIPSLoss(nn.Module):
    """Differentiable LPIPS loss with clamped [0, 1] grayscale-to-RGB inputs."""

    def __init__(self, metric_name: str = "lpips"):
        super().__init__()
        self.metric_name = str(metric_name)
        self._metric = None
        self._metric_device = None

    def _ensure_metric(self, device: torch.device):
        if self._metric is not None and self._metric_device == device:
            return
        import pyiqa

        metric = pyiqa.create_metric(self.metric_name, device=device)
        metric.eval()
        for p in metric.parameters():
            p.requires_grad_(False)
        self._metric = metric
        self._metric_device = device

    def forward(self, pred, gt):
        self._ensure_metric(pred.device)
        pred = _to_clamped_4d_rgb(pred)
        gt = _to_clamped_4d_rgb(gt)
        return self._metric(pred, gt).mean()
    

@register_loss_criterion("marco_ssim")
class MacroSSIMLoss(nn.Module):
    """
    大结构约束 SSIM Loss
    先对输入图像进行大窗口的高斯低通滤波（滤除高频散斑），
    然后再评估其宏观解剖结构的结构相似度。
    """
    def __init__(self, blur_window_size=21, blur_sigma=3.0, ssim_window_size=11, ssim_sigma=1.5):
        super().__init__()
        # 预滤波（低通）参数，用于抹除散斑
        self.blur_window_size = blur_window_size
        self.blur_sigma = blur_sigma
        
        # SSIM 计算参数
        self.ssim_window_size = ssim_window_size
        self.ssim_sigma = ssim_sigma
        
        self.channel = None
        self.blur_window = None
        self.ssim_window = None

    @staticmethod
    def _gaussian_kernel(window_size, sigma):
        coords = torch.arange(window_size).float()
        gauss = torch.exp(-(coords - window_size//2)**2 / (2*sigma*sigma))
        return gauss / gauss.sum()

    def _create_window(self, window_size, sigma, channel, device):
        _1D = self._gaussian_kernel(window_size, sigma).unsqueeze(1)
        _2D = (_1D @ _1D.t()).unsqueeze(0).unsqueeze(0)
        window = _2D.expand(channel, 1, window_size, window_size)
        return window.to(device)

    def forward(self, img1, img2):
        # reshape to [B,C,H,W]
        if img1.dim() == 2:
            img1, img2 = img1[None, None], img2[None, None]
        elif img1.dim() == 3:
            img1, img2 = img1[None], img2[None]

        B, C, H, W = img1.shape

        # 动态创建滤波器
        if self.ssim_window is None or C != self.channel:
            self.blur_window = self._create_window(self.blur_window_size, self.blur_sigma, C, img1.device)
            self.ssim_window = self._create_window(self.ssim_window_size, self.ssim_sigma, C, img1.device)
            self.channel = C

        # 1. 关键步骤：低通滤波，提取宏观解剖结构
        # 这一步彻底破坏了散斑的锐利度，但保留了血管/组织的轮廓
        macro_img1 = F.conv2d(img1, self.blur_window, padding=self.blur_window_size//2, groups=C)
        macro_img2 = F.conv2d(img2, self.blur_window, padding=self.blur_window_size//2, groups=C)

        # 2. 在宏观结构图上执行标准 SSIM 计算
        mu1 = F.conv2d(macro_img1, self.ssim_window, padding=self.ssim_window_size//2, groups=C)
        mu2 = F.conv2d(macro_img2, self.ssim_window, padding=self.ssim_window_size//2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.relu(F.conv2d(macro_img1 * macro_img1, self.ssim_window, padding=self.ssim_window_size//2, groups=C) - mu1_sq)
        sigma2_sq = F.relu(F.conv2d(macro_img2 * macro_img2, self.ssim_window, padding=self.ssim_window_size//2, groups=C) - mu2_sq)
        sigma12 = F.conv2d(macro_img1 * macro_img2, self.ssim_window, padding=self.ssim_window_size//2, groups=C) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = ssim_n / ssim_d
        ssim_score = ssim_map.mean()

        return 1 - ssim_score


def _tv_loss_impl(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    内部实现：在 [B, C, H, W] 张量上计算带 mask 的 TV 损失。
    """
    # 算水平 TV
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    mask_x = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    sum_mask_x = mask_x.sum() + 1e-6
    tv_x = (dx * mask_x).sum() / sum_mask_x

    # 算垂直 TV
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    mask_y = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    sum_mask_y = mask_y.sum() + 1e-6
    tv_y = (dy * mask_y).sum() / sum_mask_y

    return tv_x + tv_y


@register_loss_criterion("depth_tv")
class DepthImageTVLoss(nn.Module):
    """
    专门用于 Depth Image 的全变分损失 (Total Variation Loss)。
    用于平滑深度图、填充空洞，通常配合有效区域掩码使用。
    """
    def __init__(self):
        super().__init__()

    def forward(self, depth_image: torch.Tensor, valid_mask: torch.Tensor = None):
        """
        Args:
            depth_image: 深度图 [B, C, H, W] 或 [H, W]
            valid_mask: 可选。有效前景的二值掩码 (1 为有效)，例如 (pixel_accum > 0.05).float()
        """
        if depth_image.dim() == 2:
            depth_image = depth_image[None, None, ...]
        elif depth_image.dim() == 3:
            depth_image = depth_image[None, ...]

        if valid_mask is not None:
            if valid_mask.dim() == 2:
                valid_mask = valid_mask[None, None, ...]
            elif valid_mask.dim() == 3:
                valid_mask = valid_mask[None, ...]
        else:
            valid_mask = torch.ones_like(depth_image)

        return _tv_loss_impl(depth_image, valid_mask)


@register_loss_criterion("diff_tv")
class DiffImageTVLoss(nn.Module):
    """
    专门用于 Diff Image (|pred - gt|) 的全变分损失。
    用于匹配一阶梯度分布，通常不传 mask 或传入超声扇形区域 mask。
    """
    def __init__(self):
        super().__init__()

    def forward(self, diff_image: torch.Tensor, valid_mask: torch.Tensor = None):
        """
        Args:
            diff_image: 差异图 |pred - gt| [B, C, H, W] 或 [H, W]
            valid_mask: 可选。有效区域掩码，不传则默认全图有效。
        """
        if diff_image.dim() == 2:
            diff_image = diff_image[None, None, ...]
        elif diff_image.dim() == 3:
            diff_image = diff_image[None, ...]

        if valid_mask is not None:
            if valid_mask.dim() == 2:
                valid_mask = valid_mask[None, None, ...]
            elif valid_mask.dim() == 3:
                valid_mask = valid_mask[None, ...]
        else:
            valid_mask = torch.ones_like(diff_image)

        return _tv_loss_impl(diff_image, valid_mask)
