from typing import Optional

import torch
from torch import Tensor

from .config import ModelConfig
from .gauss_container import UltraGSGaussContainer
from gaussian.gauss_solver import GaussSolver
from gaussian.lr_scheduler_manager import get_expon_lr_func
from configs.root import RootConfig


class UltraGSGaussSolver(GaussSolver):
    def __init__(self, cfg: RootConfig, container: UltraGSGaussContainer):
        super().__init__(container)

        self.cfg = cfg
    
    @property
    def model_cfg(self) -> ModelConfig: 
        return self.cfg.model

    def _setup_lr_scheduler(self):
        # Online：不注册调度器，各 param_group 保持 Container 注入的常数 lr。
        # Offline：means 使用与单次长跑一致的指数衰减，其余组本就没有 scheduler。
        if self.cfg.engine.stream == "online":
            return
        means_lr = self.model_cfg.gauss_container.lr.means
        means_scheduler = get_expon_lr_func(
            lr_init=means_lr,
            lr_final=means_lr * 0.1,
            max_steps=int(self.cfg.engine.iterations),
        )
        self.lr_mgr.register("means", means_scheduler)

    def step(self, active_global_indices: Optional[Tensor] = None):
        """
        Online 且传入 render 子集下标时：仅对活跃行做 Adam 更新，避免对整锅 N 做 foreach。
        Offline 或未传下标：与基类一致，全参数 step。
        """
        if not self.optimizer:
            return
        if self.cfg.engine.stream != "online":
            super().step()
            return
        if active_global_indices is None or active_global_indices.numel() == 0:
            super().step()
            return
        self._step_adam_active_rows(active_global_indices)

    @torch.no_grad()
    def _step_adam_active_rows(self, active_global_indices: Tensor) -> None:
        idx = active_global_indices.long().view(-1)
        idx = torch.unique(idx)
        if idx.numel() == 0:
            super().step()
            return

        for group in self.optimizer.param_groups:
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            eps = group.get("eps", 1e-8)
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.optimizer.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"].item())

                g = grad.index_select(0, idx)
                if weight_decay != 0:
                    g = g.add(p.data.index_select(0, idx), alpha=weight_decay)

                ea_old = exp_avg.index_select(0, idx)
                eas_old = exp_avg_sq.index_select(0, idx)
                ea_new = ea_old.mul(beta1).add(g, alpha=1.0 - beta1)
                eas_new = eas_old.mul(beta2).addcmul(g, g, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                m_hat = ea_new / bias_correction1
                v_hat = eas_new / bias_correction2
                upd = lr * m_hat / (v_hat.sqrt().add(eps))
                p_slice = p.data.index_select(0, idx).sub(upd)

                p.data.index_copy_(0, idx, p_slice)
                exp_avg.index_copy_(0, idx, ea_new)
                exp_avg_sq.index_copy_(0, idx, eas_new)
