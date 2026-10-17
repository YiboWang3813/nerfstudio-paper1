import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, NamedTuple

# 1. 属性规格说明书 (蓝图)
class PropertySpec(NamedTuple):
    lr: float
    activate_fn: Callable
    inverse_fn: Callable

# 2. 实际的参数封装 (实体)
class GaussParam:
    def __init__(self, name: str, tensor: torch.Tensor, spec: PropertySpec):
        self.name = name
        self.spec = spec
        # 此时才真正创建 Parameter
        self.tensor = nn.Parameter(tensor)

    @property
    def value(self):
        return self.spec.activate_fn(self.tensor)

    def update_tensor(self, new_tensor: nn.Parameter):
        self.tensor = new_tensor

class GaussContainer(nn.Module, ABC):
    def __init__(self, device):
        super().__init__()

        self.device = device
        
        # 存储元数据 (Schema): name -> PropertySpec
        self._specs: Dict[str, PropertySpec] = {}
        
        # 存储实例化对象 (Instance): name -> GaussParam
        self._param_dict: Dict[str, GaussParam] = {} 
        
    @abstractmethod
    def _define_model_structure(self):
        """子类实现：定义有哪些属性 (e.g., means, scales)，但不涉及数据"""
        pass

    def _register_property(self, name, lr, activate, inverse):
        """注册属性蓝图"""
        self._specs[name] = PropertySpec(lr, activate, inverse)

    # --- Phase 3: 核心注入逻辑 ---
    def _inject_tensors(self, data_dict: Dict[str, torch.Tensor]):
        """
        通用注入器：无论数据来自 PCD 还是 Checkpoint，都走这里。
        """
        # 安全清理旧参数：只删我们在 param_dict 里记录过的
        for name in list(self._param_dict.keys()):
            internal_name = f"_{name}"
            if hasattr(self, internal_name):
                delattr(self, internal_name) # 这会自动从 _parameters 中移除
        self._param_dict.clear()

        for name, tensor in data_dict.items():
            if name not in self._specs:
                continue # 忽略不认识的参数
            
            # 确保数据在正确的设备
            tensor = tensor.to(self.device)
            
            # 使用蓝图 + 数据 创建实体
            spec = self._specs[name]
            param_wrapper = GaussParam(name, tensor, spec)
            
            # 存入字典
            self._param_dict[name] = param_wrapper
            
            # 注册到 nn.Module (关键！)
            internal_name = f"_{name}"
            self.register_parameter(internal_name, param_wrapper.tensor)

        print(f"[GaussContainer] Successfully injected {len(self._param_dict)} properties. N={self.num_points}")

    # --- Phase 2: 数据源入口 ---
    
    # 入口 A: 从场景初始化
    @abstractmethod
    def init_from_scene(self, scene_info: Dict):
        pass

    # 入口 B: 从文件加载
    def load(self, path: str):
        state = torch.load(path, map_location="cpu") # 先加载到 CPU 处理
        self._inject_tensors(state)

    def save(self, path: str):
        # 只保存隐状态字典
        state = {
            name: wrapper.tensor.detach().cpu() 
            for name, wrapper in self._param_dict.items()
        }
        torch.save(state, path)

    # --- Utils ---
    @property
    def num_points(self):
        if not self._param_dict: return 0
        return next(iter(self._param_dict.values())).tensor.shape[0]

    def get_param_groups(self):
        groups = []
        for name, wrapper in self._param_dict.items():
            groups.append({
                'params': [wrapper.tensor],
                'lr': wrapper.spec.lr, # 从 spec 取 lr
                'name': name
            })
        return groups
