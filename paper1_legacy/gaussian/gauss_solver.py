import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

from .optimizer_manager import OptimizerTensorManager
from .lr_scheduler_manager import LRSchedulerManager
from .gauss_container import GaussContainer 


class GaussSolver(ABC):
    """
    高斯优化器基类 (The Surgeon)
    负责：
    1. 管理 torch.optim.Adam
    2. 管理 Learning Rate Scheduler
    3. 执行 Prune/Concat 操作，并保持 Optimizer State 与 Container Parameter 的同步
    """
    def __init__(self, container: GaussContainer):
        self.container = container
        
        self.optimizer: torch.optim.Optimizer = None
        self.optim_mgr: OptimizerTensorManager = None
        self.lr_mgr = LRSchedulerManager()
        
        # 标志位：是否已初始化
        self._is_initialized = False

    ###########################################################################
    # 向 Runner 暴露
    ###########################################################################
    # 初始化
    def setup(self, eps: float = 1e-15):
        """
        从 Container 获取参数组，初始化 Adam 和 TensorManager
        """
        # 1. 获取 Container 准备好的 param_groups (包含 lr, name, params)
        param_groups = self.container.get_param_groups()
        
        # 2. 初始化 Adam
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=eps)
        
        # 3. 初始化 Tensor 管理器 (负责处理 state_dict 的维度变化)
        self.optim_mgr = OptimizerTensorManager(self.optimizer)
        
        # 4. 配置 LR 调度器 (由子类实现具体逻辑)
        self._setup_lr_scheduler()
        
        self._is_initialized = True

        print("[GaussSolver] Optimizer, TensorManager, and LR Scheduler setup done.")

    @abstractmethod
    def _setup_lr_scheduler(self):
        """子类需实现具体的 LR 策略注册"""
        pass

    # LR Scheduler 行为
    def step_lr(self, iteration: int):
        if self.optimizer:
            self.lr_mgr.step(iteration, self.optimizer)

    # Optimizer 行为
    def zero_grad(self):
        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        if self.optimizer:
            self.optimizer.step()

    # IO 行为
    def load(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if self.optimizer:
            self.optimizer.load_state_dict(state_dict)

    def save(self, path):
        state_dict = self.optimizer.state_dict() if self.optimizer else {}
        torch.save(state_dict, path)

    ###########################################################################
    # 向 DC Controller 暴露
    ###########################################################################
    def prune(self, mask: torch.Tensor):
        """
        删除 mask 为 True 的点。
        流程: OptimMgr 切割 Tensor & State -> 返回新 Param -> 回写 Container
        """
        if not self._is_initialized: return

        # 1. 委托 Manager 处理底层数据和优化器状态
        # 返回: Dict[str, nn.Parameter] (新的参数对象)
        new_params = self.optim_mgr.prune(mask)
        
        # 2. 将新器官（Parameter）缝合回病人（Container）体内
        self._sync_to_container(new_params)

    def concat(self, tensors: Dict[str, torch.Tensor]):
        """
        增加新的点。
        """
        if not self._is_initialized: return

        # 1. 委托 Manager 处理
        new_params = self.optim_mgr.concat(tensors)
        
        # 2. 回写 Container
        self._sync_to_container(new_params)

    def replace(self, name: str, new_tensor: torch.Tensor):
        """
        用于 Reset Opacity 等操作：保持形状，但重置数值和优化器状态
        """
        if not self._is_initialized: return
        
        # 1. 委托 Manager 替换 Parameter 并重置该参数的 Adam 状态
        new_params = self.optim_mgr.replace(name, new_tensor)
        
        # 2. 回写 Container
        self._sync_to_container(new_params)

    def _sync_to_container(self, new_params: Dict[str, nn.Parameter]):
        """
        辅助函数：将新的 Parameter 注册回 Container
        """
        for name, new_p in new_params.items():
            # 1. 更新 Wrapper 内部引用 (假设 container._param_dict 存在)
            if name in self.container._param_dict:
                self.container._param_dict[name].update_tensor(new_p)
            
            # 2. 关键：更新 nn.Module 的注册表
            # 必须删除旧属性再重新注册，否则 optimizer 和 model 指向不一致
            internal_name = f"_{name}"
            
            if hasattr(self.container, internal_name):
                delattr(self.container, internal_name)

            self.container.register_parameter(internal_name, new_p)
