from typing import Dict

import torch
import torch.nn.functional as F

from configs.root import RootConfig
from gaussian.gauss_container import GaussContainer, GaussParam
from utils.math_utils import sigmoid_inv

from .configs import ModelConfig


SH_C0 = 0.28209479177387814


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1.0e-8))


def _bounded_sigmoid(max_value: float):
    max_value = float(max_value)

    def activate(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * max_value

    def inverse(x: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(x / max_value, min=1.0e-6, max=1.0 - 1.0e-6)
        return sigmoid_inv(normalized)

    return activate, inverse


class UltraGSBaselineContainer(GaussContainer):
    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def __init__(self, cfg: RootConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        self.active_sh_degree = int(cfg.model.render.active_sh_degree_start)
        self.max_sh_degree = int(cfg.model.render.sh_degree)
        self._define_model_structure()

    def _inject_tensors(self, data_dict: Dict[str, torch.Tensor]):
        for name in list(self._param_dict.keys()):
            internal_name = f"_{name}"
            if hasattr(self, internal_name):
                delattr(self, internal_name)
        self._param_dict.clear()

        for name, tensor in data_dict.items():
            if name not in self._specs:
                continue
            tensor = tensor.to(self.device)
            wrapper = GaussParam(name, tensor, self._specs[name])
            self._param_dict[name] = wrapper
            self.register_parameter(f"_{name}", wrapper.tensor)

        print(f"[GaussContainer] Successfully injected {len(self._param_dict)} properties. N={self.num_points}")

    def _define_model_structure(self):
        c = self.model_cfg.gauss_container
        opacity_max = float(getattr(c.constraints, "opacity_max", 0.0))
        if opacity_max > 0.0:
            opacity_activate, opacity_inverse = _bounded_sigmoid(opacity_max)
        else:
            opacity_activate, opacity_inverse = torch.sigmoid, sigmoid_inv
        self._register_property("means", lr=c.lr.means, activate=lambda x: x, inverse=lambda x: x)
        self._register_property("sh_dc", lr=c.lr.sh_dc, activate=lambda x: x, inverse=lambda x: x)
        self._register_property("sh_rest", lr=c.lr.sh_rest, activate=lambda x: x, inverse=lambda x: x)
        self._register_property(
            "opacities",
            lr=c.lr.opacities,
            activate=opacity_activate,
            inverse=opacity_inverse,
        )
        self._register_property("scales_2d", lr=c.lr.scales_2d, activate=torch.exp, inverse=_safe_log)
        self._register_property("quats", lr=c.lr.quats, activate=lambda x: F.normalize(x, dim=-1, eps=1e-8), inverse=lambda x: x)
        self._register_property("fov_x", lr=c.lr.fov_x, activate=lambda x: x, inverse=lambda x: x)
        self._register_property("fov_y", lr=c.lr.fov_y, activate=lambda x: x, inverse=lambda x: x)

    def init_from_scene(self, scene_info: Dict[str, torch.Tensor]):
        means = scene_info["means"].to(self.device)
        n = int(means.shape[0])
        init = self.model_cfg.gauss_container.init_val

        if "scales" in scene_info:
            scales = scene_info["scales"].to(self.device)
            if scales.shape[-1] >= 2:
                scales_2d = scales[..., :2]
            else:
                scales_2d = scales.expand(n, 2)
        else:
            scales_2d = torch.tensor(init.scale_2d, device=self.device, dtype=torch.float32).view(1, 2).expand(n, 2)
        scales_2d_raw = self._specs["scales_2d"].inverse_fn(scales_2d.float())

        if "quats" in scene_info:
            quats = scene_info["quats"].to(self.device).float()
        else:
            quats = torch.zeros((n, 4), device=self.device, dtype=torch.float32)
            quats[:, 0] = 1.0

        sh_coeff_count = (self.max_sh_degree + 1) ** 2
        if "intensities" in scene_info:
            intensities = scene_info["intensities"].to(self.device).float()
            if intensities.dim() == 1:
                intensities = intensities.unsqueeze(-1)
            if intensities.shape[-1] == 1:
                intensities = intensities.expand(n, 3)
            else:
                intensities = intensities[..., :3]
            sh_dc = ((intensities.clamp(0.0, 1.0) - 0.5) / SH_C0).reshape(n, 1, 3)
        else:
            sh_dc = torch.full((n, 1, 3), float(init.sh_dc), device=self.device, dtype=torch.float32)
        sh_rest = torch.zeros((n, sh_coeff_count - 1, 3), device=self.device, dtype=torch.float32)

        op0 = torch.full((n, 1), float(init.opacity), device=self.device, dtype=torch.float32)
        opacities = self._specs["opacities"].inverse_fn(op0)

        fov_x = torch.full((n, 1), float(init.fov_x), device=self.device, dtype=torch.float32)
        fov_y = torch.full((n, 1), float(init.fov_y), device=self.device, dtype=torch.float32)

        self._inject_tensors(
            {
                "means": means.float(),
                "sh_dc": sh_dc,
                "sh_rest": sh_rest,
                "opacities": opacities,
                "scales_2d": scales_2d_raw,
                "quats": quats,
                "fov_x": fov_x,
                "fov_y": fov_y,
            }
        )

    @property
    def means(self):
        return self._param_dict["means"].value

    @property
    def scales_2d(self):
        return self._param_dict["scales_2d"].value

    @property
    def quats(self):
        return self._param_dict["quats"].value

    @property
    def opacities(self):
        return self._param_dict["opacities"].value

    @property
    def sh_coeffs(self):
        return torch.cat([self._param_dict["sh_dc"].value, self._param_dict["sh_rest"].value], dim=1)

    @property
    def fov_x_offsets(self):
        return self._param_dict["fov_x"].value

    @property
    def fov_y_offsets(self):
        return self._param_dict["fov_y"].value

    def oneup_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
