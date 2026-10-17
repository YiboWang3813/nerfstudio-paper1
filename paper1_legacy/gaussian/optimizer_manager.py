import torch
import torch.nn as nn
from typing import Dict


class OptimizerTensorManager:
    """ 只负责维护 Optimizer 中张量的第1维变化 """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def replace(self, name: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            if group.get("name", "") != name:
                continue

            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param, None)

            new_param = nn.Parameter(tensor.requires_grad_(True))

            if state is not None:
                state["exp_avg"] = torch.zeros_like(tensor)
                state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = state

            group["params"][0] = new_param
            optimizable_tensors[name] = new_param

        return optimizable_tensors

    def prune(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # mask 中 True 被删除 valid_mask 中 True 为保留
        valid_mask = ~mask
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            name = group["name"]
            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param, None)

            new_data = old_param.data[valid_mask].contiguous()
            new_param = nn.Parameter(new_data.requires_grad_(True))

            if state is not None:
                for k in ["exp_avg", "exp_avg_sq"]:
                    if k in state:
                        state[k] = state[k][valid_mask]
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = state

            group["params"][0] = new_param
            optimizable_tensors[name] = new_param

        return optimizable_tensors

    def concat(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            name = group["name"]
            ext = tensors[name]

            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param, None)

            new_data = torch.cat([old_param.data, ext], dim=0).contiguous()
            new_param = nn.Parameter(new_data.requires_grad_(True))

            if state is not None:
                for k in ["exp_avg", "exp_avg_sq"]:
                    if k in state:
                        z = torch.zeros_like(ext)
                        state[k] = torch.cat([state[k], z], dim=0)
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = state

            group["params"][0] = new_param
            optimizable_tensors[name] = new_param

        return optimizable_tensors
