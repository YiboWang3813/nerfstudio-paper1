"""时序混合：优先 ``mamba_ssm.Mamba``，不可用时退化为 ``TransformerEncoderLayer``。"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    # Avoid ``from mamba_ssm import Mamba``: v1.x imports LM generation
    # utilities from transformers at package top level, which FiMA does not use.
    from mamba_ssm.modules.mamba_simple import Mamba

    HAS_MAMBA = True
    MAMBA_IMPORT_ERROR = None
except Exception as exc:  # CUDA/ABI 不匹配时可能不是 ImportError。
    Mamba = None  # type: ignore
    HAS_MAMBA = False
    MAMBA_IMPORT_ERROR = exc


class FiMAMixerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        prefer_mamba: bool,
        require_mamba: bool = False,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        if bool(require_mamba) and not HAS_MAMBA:
            raise RuntimeError(
                "FiMA was configured with pem.require_mamba=True, but mamba_ssm.Mamba "
                f"could not be imported: {MAMBA_IMPORT_ERROR!r}"
            )
        self.use_mamba = bool(prefer_mamba and HAS_MAMBA)
        if self.use_mamba:
            self.core = Mamba(
                d_model=d_model,
                d_state=int(mamba_d_state),
                d_conv=int(mamba_d_conv),
                expand=int(mamba_expand),
            )
        else:
            nhead = max(1, min(8, d_model // 16))
            if d_model % nhead != 0:
                nhead = 1
            self.core = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=max(d_model * 4, 256),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.use_mamba:
            y = self.core(x)
        else:
            y = self.core(x)
        y = self.dropout(y)
        return self.norm(x + y)


class FiMASequenceMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        *,
        prefer_mamba: bool,
        require_mamba: bool = False,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FiMAMixerLayer(
                    d_model,
                    prefer_mamba=prefer_mamba,
                    require_mamba=require_mamba,
                    mamba_d_state=mamba_d_state,
                    mamba_d_conv=mamba_d_conv,
                    mamba_expand=mamba_expand,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    @property
    def backend_name(self) -> str:
        if len(self.layers) == 0:
            return "none"
        return "mamba_ssm" if bool(self.layers[0].use_mamba) else "transformer"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SpatialStem(nn.Module):
    """将每帧灰度图编码为 d_model 向量（输入已归一化 0~1）。"""

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        d_model: int,
        *,
        input_downsample_stride: int = 1,
        input_downsample_channels: int = 32,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        ds = int(input_downsample_stride)
        if ds > 1:
            c_ds = int(input_downsample_channels)
            layers.extend(
                [
                    nn.Conv2d(c_in, c_ds, kernel_size=5, stride=ds, padding=2),
                    nn.GroupNorm(min(8, c_ds), c_ds),
                    nn.GELU(),
                ]
            )
            c_in = c_ds
        for c_out in channels:
            layers.extend(
                [
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(min(8, c_out), c_out),
                    nn.GELU(),
                ]
            )
            c_in = c_out
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Conv2d(c_in, d_model, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        z = self.cnn(x)
        z = self.proj(z)
        z = self.pool(z).flatten(1)
        return z
