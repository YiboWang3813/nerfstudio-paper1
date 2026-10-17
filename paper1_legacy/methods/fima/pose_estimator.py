"""FiMA 图像序列 -> 相对 ``P_0`` 的 6DoF（PEM）。"""
from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .configs import FiMAConfig
from .remamba_blocks import FiMASequenceMixer, SpatialStem


class FiMAPoseEstimator(nn.Module):
    def __init__(self, cfg: FiMAConfig):
        super().__init__()
        self.cfg = cfg
        pem = cfg.pem
        self.window = int(pem.window_size)
        self.layout = str(pem.pem_output_layout)
        self.d_model = int(pem.d_model)

        self.stem = SpatialStem(
            1,
            list(pem.stem_channels),
            self.d_model,
            input_downsample_stride=int(getattr(pem, "input_downsample_stride", 1)),
            input_downsample_channels=int(getattr(pem, "input_downsample_channels", 32)),
        )
        self.mixer = FiMASequenceMixer(
            self.d_model,
            int(pem.num_mixer_layers),
            prefer_mamba=bool(pem.prefer_mamba),
            require_mamba=bool(getattr(pem, "require_mamba", False)),
            mamba_d_state=int(pem.mamba_d_state),
            mamba_d_conv=int(pem.mamba_d_conv),
            mamba_expand=int(pem.mamba_expand),
            dropout=float(pem.dropout),
        )
        if self.layout == "last_frame_rel0":
            self.head = nn.Linear(self.d_model, 6)
        elif self.layout in ("window_frames_rel0", "window_edges"):
            self.head = nn.Linear(self.d_model, 6)
        else:
            raise ValueError(f"Unknown pem_output_layout: {self.layout}")
        self.predict_uncertainty = bool(getattr(pem, "predict_uncertainty", False))
        if self.predict_uncertainty:
            self.logvar_head = nn.Linear(self.d_model, 2)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)
        else:
            self.logvar_head = None

    def forward(self, images: torch.Tensor, _valid_window: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """
        images: (B, N, 1, H, W)
        _valid_window: (B, N) optional bool mask（预留；当前不屏蔽 mixer）
        """
        B, N, _, H, W = images.shape
        x = images.reshape(B * N, 1, H, W)
        tok = self.stem(x)  # (B*N, D)
        tok = tok.reshape(B, N, self.d_model)
        feat = self.mixer(tok)

        if self.layout == "last_frame_rel0":
            h = self.head(feat[:, -1, :])
            out = {"pred_pose_6": h}
            if self.logvar_head is not None:
                out["pred_logvar_2"] = self.logvar_head(feat[:, -1, :])
            return out

        # window_frames_rel0 / window_edges：每个时间步单独一头（共享权重）。
        # ``window_edges`` 的第 k 个输出对应局部边 (k -> k+1)，使用右端帧 token。
        h = self.head(feat)  # (B, N, 6)
        # 丢弃局部索引 0，保留 N-1 槽位对应局部 1..N-1
        pred = h[:, 1:, :].contiguous()
        out = {"pred_pose_6": pred}
        if self.logvar_head is not None:
            out["pred_logvar_2"] = self.logvar_head(feat)[:, 1:, :].contiguous()
        return out
