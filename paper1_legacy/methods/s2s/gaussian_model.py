import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Dict 

from common.gaussian.base_model import BaseGaussianModel
from common.utils.math_utils import sigmoid_inv, softplus_inv


class GaussianModel(BaseGaussianModel):
    def __init__(self, args, device=None):
        super().__init__(args, device)
        # 1. 仅添加/覆盖特有属性
        self.param_cfg.update({
            "opacities": dict(
                lr=getattr(self.args, "opacities_lr", 5e-4),
                activate=torch.sigmoid,
                inverse=sigmoid_inv,
                default_init_val=getattr(self.args, "opacities_init_val", 0.1),
            ),
            "intensities": dict(
                lr=getattr(self.args, "intensities_lr", 5e-4),
                activate=F.softplus,
                inverse=softplus_inv,
                default_init_val=getattr(self.args, "intensities_init_val", 1.8),
            ),
        })

    @torch.no_grad()
    def init_from_scene(self, gaussians: Dict[str, torch.Tensor]):
        # 1. 调用基类处理几何初始化
        self._init_geometry(gaussians, break_symmetry=True)
        
        # 2. 处理 Attribute 初始化
        N = self._means.shape[0]
        for name in ["opacities", "intensities"]:
            cfg = self.param_cfg[name]
            raw = cfg["inverse"](torch.full((N,), cfg["default_init_val"], device=self.device))
            setattr(self, f"_{name}", nn.Parameter(raw))

        if self.optimizer: self.optimizer.state.clear()
        print(f"[S2S] Model initialized with {N} points.")