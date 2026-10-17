from typing import Dict

import torch
import torch.nn.functional as F

from configs.root import RootConfig
from gaussian.gauss_container import GaussContainer, GaussParam
from utils.math_utils import sigmoid_inv

from .configs import ModelConfig


class UltraGrayGaussContainer(GaussContainer):
    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def __init__(self, cfg: RootConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
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
            spec = self._specs[name]
            param_wrapper = GaussParam(name, tensor, spec)
            self._param_dict[name] = param_wrapper
            self.register_parameter(f"_{name}", param_wrapper.tensor)

        print(f"[GaussContainer] Successfully injected {len(self._param_dict)} properties. N={self.num_points}")

    def _define_model_structure(self):
        self._register_property("means", lr=self.model_cfg.gauss_container.lr.means, activate=lambda x: x, inverse=lambda x: x)
        self._register_property("scales", lr=self.model_cfg.gauss_container.lr.scales, activate=torch.exp, inverse=torch.log)
        self._register_property("quats", lr=self.model_cfg.gauss_container.lr.quats, activate=F.normalize, inverse=lambda x: x)
        self._register_property(
            "attenuations",
            lr=self.model_cfg.gauss_container.lr.attenuations,
            activate=torch.sigmoid,
            inverse=sigmoid_inv,
        )
        # SH degree-1 split into DC/rest for independent LR (paper setting).
        self._register_property("sh_dc", lr=self.model_cfg.gauss_container.lr.sh_dc, activate=lambda x: x, inverse=lambda x: x)
        self._register_property("sh_rest", lr=self.model_cfg.gauss_container.lr.sh_rest, activate=lambda x: x, inverse=lambda x: x)

    def init_from_scene(self, scene_info: Dict[str, torch.Tensor]):
        means = scene_info["means"]
        n = int(means.shape[0])

        if "scales" in scene_info:
            scales = self._specs["scales"].inverse_fn(scene_info["scales"])
        else:
            s0 = float(self.model_cfg.gauss_container.init_val.scale_mm)
            scales = self._specs["scales"].inverse_fn(torch.full((n, 3), s0, device=self.device, dtype=torch.float32))

        if "quats" in scene_info:
            quats = scene_info["quats"]
        else:
            quats = torch.zeros((n, 4), device=self.device, dtype=torch.float32)
            quats[:, 0] = 1.0

        tau0 = float(self.model_cfg.gauss_container.init_val.tau)
        attenuations = self._specs["attenuations"].inverse_fn(torch.full((n,), tau0, device=self.device, dtype=torch.float32))

        sh_dc0 = float(self.model_cfg.gauss_container.init_val.sh_dc)
        sh_dc = torch.full((n, 1), sh_dc0, device=self.device, dtype=torch.float32)
        sh_rest = torch.zeros((n, 3), device=self.device, dtype=torch.float32)

        self._inject_tensors(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "attenuations": attenuations,
                "sh_dc": sh_dc,
                "sh_rest": sh_rest,
            }
        )

    @property
    def means(self):
        return self._param_dict["means"].value

    @property
    def scales(self):
        return self._param_dict["scales"].value

    @property
    def quats(self):
        return self._param_dict["quats"].value

    @property
    def attenuations(self):
        return self._param_dict["attenuations"].value

    @property
    def sh_coeffs(self):
        return torch.cat([self._param_dict["sh_dc"].value, self._param_dict["sh_rest"].value], dim=1)
