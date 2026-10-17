from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import FieldConfig


class PositionalEncoding(nn.Module):
    def __init__(self, input_dims: int = 3, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.input_dims = int(input_dims)
        self.num_freqs = int(num_freqs)
        self.include_input = bool(include_input)
        self.register_buffer("freq_bands", 2.0 ** torch.arange(self.num_freqs).float(), persistent=False)

        self.out_dim = self.input_dims * self.num_freqs * 2
        if self.include_input:
            self.out_dim += self.input_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x] if self.include_input else []
        xb = x.unsqueeze(-2) * self.freq_bands.view(*([1] * (x.ndim - 1)), -1, 1)
        parts.append(torch.sin(torch.pi * xb).flatten(start_dim=-2))
        parts.append(torch.cos(torch.pi * xb).flatten(start_dim=-2))
        return torch.cat(parts, dim=-1)


class UltraNeRFField(nn.Module):
    """Isotropic Ultra-NeRF field: 3D point -> alpha, beta, rho_b, rho_s, phi."""

    def __init__(self, cfg: Optional[FieldConfig] = None):
        super().__init__()
        self.cfg = cfg or FieldConfig()
        if bool(self.cfg.use_viewdirs):
            raise ValueError("UltraNeRFField currently implements the paper's isotropic field; use_viewdirs must be false.")

        self.encoder = PositionalEncoding(
            input_dims=3,
            num_freqs=int(self.cfg.pe_num_freqs),
            include_input=bool(self.cfg.include_input),
        )
        d_in = self.encoder.out_dim
        depth = int(self.cfg.depth)
        width = int(self.cfg.width)
        skip_layer = int(self.cfg.skip_layer)

        layers = []
        for i in range(depth):
            in_dim = d_in if i == 0 else width
            if i == skip_layer:
                in_dim += d_in
            layers.append(nn.Linear(in_dim, width))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(width, 5)
        self._init_output_bias()

    def _init_output_bias(self) -> None:
        with torch.no_grad():
            self.output.bias.zero_()
            self.output.bias[0] = float(self.cfg.alpha_bias_init)

    def forward_raw(self, points: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(points)
        h = encoded
        for i, layer in enumerate(self.layers):
            if i == int(self.cfg.skip_layer):
                h = torch.cat([h, encoded], dim=-1)
            h = F.relu(layer(h), inplace=False)
        return self.output(h)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.forward_raw(points)
        mode = str(self.cfg.alpha_activation).lower()
        if mode == "abs":
            alpha = torch.abs(raw[..., 0])
        elif mode == "softplus":
            alpha = F.softplus(raw[..., 0])
        else:
            raise ValueError(f"Unsupported alpha_activation='{self.cfg.alpha_activation}'")

        return {
            "alpha": alpha,
            "beta": torch.sigmoid(raw[..., 1]),
            "rho_b": torch.sigmoid(raw[..., 2]),
            "rho_s": torch.sigmoid(raw[..., 3]),
            "phi": torch.sigmoid(raw[..., 4]),
            "raw": raw,
        }
