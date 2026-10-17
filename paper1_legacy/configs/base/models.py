from dataclasses import dataclass
from omegaconf import MISSING

from methods.ultrags.config import ModelConfig as UltraGSConfig
from methods.ultragray.configs import ModelConfig as UltraGrayConfig
from methods.ultragauss.configs import ModelConfig as UltraGaussConfig
from methods.ultrags_baseline.configs import ModelConfig as UltraGSBaselineConfig
from methods.slice_adapted_3dgs.configs import ModelConfig as SliceAdapted3DGSConfig
from methods.fima.configs import FiMAConfig
from methods.ours.configs import ModelConfig as OursModelConfig
from methods.gaussianpile.configs import ModelConfig as GaussianPileModelConfig
from methods.ultranerf.configs import ModelConfig as UltraNeRFModelConfig
from methods.nerfus.configs import ModelConfig as NerfusModelConfig

# 在这里注册所有 Model Config
MODEL_CONFIG_REGISTRY = {
    "ultrags": UltraGSConfig,
    "ultragray": UltraGrayConfig,
    "ultragauss": UltraGaussConfig,
    "ultrags_baseline": UltraGSBaselineConfig,
    "slice_adapted_3dgs": SliceAdapted3DGSConfig,
    "fima": FiMAConfig,
    "ours": OursModelConfig,
    "gaussianpile": GaussianPileModelConfig,
    "ultranerf": UltraNeRFModelConfig,
    "nerfus": NerfusModelConfig,
}
