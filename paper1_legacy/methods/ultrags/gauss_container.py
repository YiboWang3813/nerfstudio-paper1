import torch
import torch.nn.functional as F

from typing import Dict

from .config import ModelConfig
from configs.root import RootConfig

from gaussian.gauss_container import GaussContainer
from utils.math_utils import sigmoid_inv


class UltraGSGaussContainer(GaussContainer):
    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    # 强类型依赖注入配置
    def __init__(self, cfg: RootConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        
        # Phase 1: 在初始化时就定义好模型结构
        self._define_model_structure()

    def _define_model_structure(self):
        # Means
        self._register_property(
            "means", 
            lr=self.model_cfg.gauss_container.lr.means, 
            activate=lambda x: x, 
            inverse=lambda x: x
        )
        # Scales
        self._register_property(
            "scales", 
            lr=self.model_cfg.gauss_container.lr.scales, 
            activate=torch.exp, 
            inverse=torch.log
        )
        # Quats
        self._register_property(
            "quats",
            lr=self.model_cfg.gauss_container.lr.quats, 
            activate=F.normalize, 
            inverse=lambda x: x
        )
        # Opacity
        self._register_property(
            "opacities", 
            lr=self.model_cfg.gauss_container.lr.opacities, 
            activate=torch.sigmoid, 
            inverse=sigmoid_inv
        )
        # Intensities
        self._register_property(
            "intensities",
            lr=self.model_cfg.gauss_container.lr.intensities,
            activate=torch.sigmoid,
            inverse=sigmoid_inv
        )

    def init_from_scene(self, scene_info: Dict[str, torch.Tensor]):
        # Means
        means = scene_info["means"]
        N = means.shape[0]
        # Scales
        scales = self._specs["scales"].inverse_fn(scene_info["scales"])
        # Quats
        quats = scene_info["quats"]
        # Opacities
        opac_init = self.model_cfg.gauss_container.init_val.opacities
        opacities = torch.empty((N,), device=self.device).uniform_(opac_init * 0.8, opac_init * 1.2)
        opacities = self._specs["opacities"].inverse_fn(opacities)
        # Intensities
        inten_init = self.model_cfg.gauss_container.init_val.intensities
        intensities = torch.empty((N,), device=self.device).uniform_(inten_init * 0.8, inten_init * 1.2)
        intensities = self._specs["intensities"].inverse_fn(intensities)

        data_pack = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "intensities": intensities
        } 

        self._inject_tensors(data_pack)

    def init_empty(self):
        """无高斯占位：仅注入 0 点张量；在线训练在 stream 各帧 concat 时再装入。"""
        d = self.device
        data_pack = {
            "means": torch.empty((0, 3), device=d, dtype=torch.float32),
            "scales": torch.empty((0, 3), device=d, dtype=torch.float32),
            "quats": torch.empty((0, 4), device=d, dtype=torch.float32),
            "opacities": torch.empty((0,), device=d, dtype=torch.float32),
            "intensities": torch.empty((0,), device=d, dtype=torch.float32),
        }
        self._inject_tensors(data_pack)

    def complete_new_gaussians_for_concat(self, gaussians: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        校验增量高斯输入并补全 concat 所需字段。

        必须包含: ``means/scales/quats``。
        若缺失 ``opacities/intensities``，按容器初始化规则补全到 inverse 空间。
        """
        required = ("means", "scales", "quats")
        missing = [k for k in required if k not in gaussians]
        if missing:
            raise ValueError(f"new_gaussians 缺少必需键: {missing}")

        means = gaussians["means"]
        scales = gaussians["scales"]
        quats = gaussians["quats"]
        n_new = int(means.shape[0])

        if int(scales.shape[0]) != n_new or int(quats.shape[0]) != n_new:
            raise ValueError(
                f"new_gaussians 首维不一致: means={means.shape[0]}, "
                f"scales={scales.shape[0]}, quats={quats.shape[0]}"
            )

        out: Dict[str, torch.Tensor] = dict(gaussians)

        if "opacities" not in out:
            opac_init = self.model_cfg.gauss_container.init_val.opacities
            opacities = torch.empty((n_new,), device=self.device).uniform_(opac_init * 0.8, opac_init * 1.2)
            out["opacities"] = self._specs["opacities"].inverse_fn(opacities)
        if "intensities" not in out:
            inten_init = self.model_cfg.gauss_container.init_val.intensities
            intensities = torch.empty((n_new,), device=self.device).uniform_(inten_init * 0.8, inten_init * 1.2)
            out["intensities"] = self._specs["intensities"].inverse_fn(intensities)

        return out

    # 便捷访问属性 (Optional)
    @property
    def means(self): return self._param_dict["means"].value
    
    @property
    def scales(self): return self._param_dict["scales"].value

    @property
    def quats(self): return self._param_dict["quats"].value

    @property
    def opacities(self): return self._param_dict["opacities"].value

    @property
    def intensities(self): return self._param_dict["intensities"].value