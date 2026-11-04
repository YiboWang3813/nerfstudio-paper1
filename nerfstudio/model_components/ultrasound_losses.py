"""Ultrasound structural loss preserving the frozen padded-window definition."""

import torch
import torch.nn.functional as F
from torch import nn


class UltrasoundSSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        if window_size <= 0 or window_size % 2 != 1 or sigma <= 0:
            raise ValueError("SSIM requires an odd positive window and positive sigma.")
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        kernel = torch.exp(-coords.square() / (2 * sigma * sigma))
        kernel /= kernel.sum()
        self.register_buffer("window", torch.outer(kernel, kernel)[None, None])

    def forward(self, pred, target):
        if pred.shape != target.shape or pred.ndim != 4:
            raise ValueError("Expected matching BCHW tensors.")
        channels = pred.shape[1]
        window = self.window.expand(channels, 1, -1, -1)

        def blur(image):
            return F.conv2d(image, window, padding=self.window_size // 2, groups=channels)

        mu1, mu2 = blur(pred), blur(target)
        v1 = F.relu(blur(pred.square()) - mu1.square())
        v2 = F.relu(blur(target.square()) - mu2.square())
        covariance = blur(pred * target) - mu1 * mu2
        numerator = (2 * mu1 * mu2 + 0.01**2) * (2 * covariance + 0.03**2)
        denominator = (mu1.square() + mu2.square() + 0.01**2) * (v1 + v2 + 0.03**2)
        return (numerator / denominator).mean()
