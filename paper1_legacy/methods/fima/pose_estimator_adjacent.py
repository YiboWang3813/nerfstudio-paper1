"""双帧输入 → 相邻帧相对变换 6DoF（与 ``pem.input_mode: adjacent_pair`` 配套，结构轻量）。"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .configs import FiMAConfig
from .remamba_blocks import SpatialStem


class FiMAAdjacentPairEstimator(nn.Module):
    """两帧灰度 → ``T_{i\\to i+1}`` 的 6DoF（mm + ZYX rad）。"""

    def __init__(self, cfg: FiMAConfig):
        super().__init__()
        pem = cfg.pem
        self.d_model = int(pem.d_model)
        fusion = str(getattr(pem, "occ_fusion_mode", "concat")).lower()
        use_concat_ch2 = fusion != "weighted" and (
            str(getattr(pem, "gs_occ_cache_dir", "") or "").strip()
            or bool(getattr(pem, "occ_always_two_channels", False))
        )
        in_ch = 2 if use_concat_ch2 else 1
        self.stem = SpatialStem(in_ch, list(pem.stem_channels), self.d_model)
        hid = max(128, self.d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * self.d_model, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, 6),
        )
        self.predict_uncertainty = bool(getattr(pem, "predict_uncertainty", False))
        if self.predict_uncertainty:
            self.logvar_head = nn.Linear(2 * self.d_model, 2)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)
        else:
            self.logvar_head = None

    def forward(self, images: torch.Tensor, _valid_window=None) -> Dict[str, torch.Tensor]:
        """
        images: (B, 2, C, H, W)，``C=1``（含 weighted OCC）或 ``C=2``（concat）
        """
        B, n, c, h, w = images.shape
        if n != 2:
            raise ValueError(f"adjacent_pair expects N=2, got {n}")
        if c not in (1, 2):
            raise ValueError(f"adjacent_pair expects C=1 or C=2 (gray or gray+occ), got {c}")
        x = images.reshape(B * n, c, h, w)
        tok = self.stem(x).reshape(B, n, self.d_model)
        feat = torch.cat([tok[:, 0], tok[:, 1]], dim=-1)
        pred = self.head(feat)
        out = {"pred_pose_6": pred}
        if self.logvar_head is not None:
            out["pred_logvar_2"] = self.logvar_head(feat)
        return out
