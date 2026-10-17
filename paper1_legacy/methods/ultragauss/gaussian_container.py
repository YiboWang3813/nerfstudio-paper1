from typing import Dict

import torch
import torch.nn.functional as F

from configs.root import RootConfig
from gaussian.gauss_container import GaussContainer
from utils.math_utils import sigmoid_inv

from .configs import ModelConfig


def _quat_wxyz_to_matrix(quats: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quats, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot = torch.empty((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    rot[..., 0, 0] = ww + xx - yy - zz
    rot[..., 0, 1] = 2.0 * (xy - wz)
    rot[..., 0, 2] = 2.0 * (xz + wy)
    rot[..., 1, 0] = 2.0 * (xy + wz)
    rot[..., 1, 1] = ww - xx + yy - zz
    rot[..., 1, 2] = 2.0 * (yz - wx)
    rot[..., 2, 0] = 2.0 * (xz - wy)
    rot[..., 2, 1] = 2.0 * (yz + wx)
    rot[..., 2, 2] = ww - xx - yy + zz
    return rot


def _precision_tril_raw_from_scales_quats(scales: torch.Tensor, quats: torch.Tensor, beta: float) -> torch.Tensor:
    scales = scales.to(dtype=torch.float32).clamp_min(1e-6)
    quats = quats.to(device=scales.device, dtype=scales.dtype)
    rot = _quat_wxyz_to_matrix(quats)
    precision_diag = 1.0 / (scales * scales)
    precision = rot @ torch.diag_embed(precision_diag) @ rot.transpose(1, 2)
    chol = torch.linalg.cholesky(precision)

    diag_raw = (torch.diagonal(chol, dim1=1, dim2=2) - float(beta)).clamp_min(1e-8).sqrt()
    return torch.stack(
        [
            diag_raw[:, 0],
            chol[:, 1, 0],
            chol[:, 2, 0],
            diag_raw[:, 1],
            chol[:, 2, 1],
            diag_raw[:, 2],
        ],
        dim=1,
    )


class UltraGaussContainer(GaussContainer):
    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def __init__(self, cfg: RootConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        self._define_model_structure()

    def _define_model_structure(self):
        self._register_property("means", lr=self.model_cfg.gauss_container.lr.means, activate=lambda x: x, inverse=lambda x: x)
        self._register_property(
            "precision_tril",
            lr=self.model_cfg.gauss_container.lr.precision_tril,
            activate=lambda x: x,
            inverse=lambda x: x,
        )
        self._register_property(
            "opacities",
            lr=self.model_cfg.gauss_container.lr.opacities,
            activate=torch.sigmoid,
            inverse=sigmoid_inv,
        )
        self._register_property(
            "intensities",
            lr=self.model_cfg.gauss_container.lr.intensities,
            activate=torch.sigmoid,
            inverse=sigmoid_inv,
        )

    def init_from_scene(self, scene_info: Dict[str, torch.Tensor]):
        means = scene_info["means"]
        n = int(means.shape[0])

        if "precision_tril" in scene_info:
            precision_tril = scene_info["precision_tril"]
        elif "scales" in scene_info:
            scales = scene_info["scales"].to(self.device)
            if "quats" in scene_info:
                quats = scene_info["quats"].to(self.device)
            else:
                quats = torch.zeros((n, 4), device=self.device, dtype=torch.float32)
                quats[:, 0] = 1.0
            precision_tril = _precision_tril_raw_from_scales_quats(
                scales,
                quats,
                float(self.model_cfg.render.diag_beta),
            )
        else:
            c = self.model_cfg.gauss_container.init_val
            diag = torch.empty((n, 3), device=self.device).uniform_(float(c.diag_raw_min), float(c.diag_raw_max))
            off = torch.randn((n, 3), device=self.device) * float(c.offdiag_std)
            precision_tril = torch.cat([diag[:, 0:1], off[:, 0:1], off[:, 1:2], diag[:, 1:2], off[:, 2:3], diag[:, 2:3]], dim=1)

        if "opacities" in scene_info:
            opacities = self._specs["opacities"].inverse_fn(scene_info["opacities"])
        else:
            opacity0 = float(self.model_cfg.gauss_container.init_val.opacity)
            opacities = self._specs["opacities"].inverse_fn(
                torch.full((n,), opacity0, device=self.device, dtype=torch.float32)
            )

        if "intensities" in scene_info:
            intensities = self._specs["intensities"].inverse_fn(scene_info["intensities"])
        else:
            intensity0 = float(self.model_cfg.gauss_container.init_val.intensity)
            intensities = self._specs["intensities"].inverse_fn(
                torch.full((n,), intensity0, device=self.device, dtype=torch.float32)
            )

        self._inject_tensors(
            {
                "means": means,
                "precision_tril": precision_tril,
                "opacities": opacities,
                "intensities": intensities,
            }
        )

    @property
    def means(self):
        return self._param_dict["means"].value

    @property
    def precision_tril(self):
        return self._param_dict["precision_tril"].value

    @property
    def opacities(self):
        return self._param_dict["opacities"].value

    @property
    def intensities(self):
        return self._param_dict["intensities"].value
