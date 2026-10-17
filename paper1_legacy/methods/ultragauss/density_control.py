from typing import Dict, Tuple

import torch

from configs.root import RootConfig

from .configs import ModelConfig
from .gaussian_container import UltraGaussContainer
from .gaussian_solver import UltraGaussSolver


def _build_precision_cholesky_from_tril(precision_tril: torch.Tensor, beta: float) -> torch.Tensor:
    n = int(precision_tril.shape[0])
    l = torch.zeros((n, 3, 3), device=precision_tril.device, dtype=precision_tril.dtype)
    l[:, 0, 0] = precision_tril[:, 0] * precision_tril[:, 0] + beta
    l[:, 1, 0] = precision_tril[:, 1]
    l[:, 2, 0] = precision_tril[:, 2]
    l[:, 1, 1] = precision_tril[:, 3] * precision_tril[:, 3] + beta
    l[:, 2, 1] = precision_tril[:, 4]
    l[:, 2, 2] = precision_tril[:, 5] * precision_tril[:, 5] + beta
    return l


def _approx_scale_from_precision_tril(precision_tril: torch.Tensor, beta: float) -> torch.Tensor:
    l11 = (precision_tril[:, 0] * precision_tril[:, 0] + beta).clamp(min=1e-8)
    l12 = precision_tril[:, 1]
    l13 = precision_tril[:, 2]
    l22 = (precision_tril[:, 3] * precision_tril[:, 3] + beta).clamp(min=1e-8)
    l23 = precision_tril[:, 4]
    l33 = (precision_tril[:, 5] * precision_tril[:, 5] + beta).clamp(min=1e-8)

    inv_l00 = 1.0 / l11
    inv_l10 = -l12 / (l11 * l22)
    inv_l11 = 1.0 / l22
    inv_l20 = (l12 * l23 - l13 * l22) / (l11 * l22 * l33)
    inv_l21 = -l23 / (l22 * l33)
    inv_l22 = 1.0 / l33

    var_x = inv_l00.square() + inv_l10.square() + inv_l20.square()
    var_y = inv_l11.square() + inv_l21.square()
    var_z = inv_l22.square()
    return torch.sqrt(torch.stack([var_x, var_y, var_z], dim=1).clamp(min=1e-12))


class UltraGaussOfflineDensityController:
    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        self.ctrl_cfg = self.model_cfg.offline.density_control
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.last_event_stats: Dict[str, float] = {}

    def scalar_stats(self, iteration: int) -> Dict[str, float]:
        if int(self.last_event_stats.get("iteration", -1)) != int(iteration):
            return {}
        return {f"train/dc/{k}": v for k, v in self.last_event_stats.items() if k != "iteration"}

    def should_accumulate_grad_stats(self, iteration: int) -> bool:
        t = self.ctrl_cfg.timing
        return int(t.start_iter) <= iteration <= int(t.end_iter)

    def should_densify_prune(self, iteration: int) -> bool:
        if not (bool(self.ctrl_cfg.enable.densify) or bool(self.ctrl_cfg.enable.prune)):
            return False
        t = self.ctrl_cfg.timing
        if iteration < int(t.start_iter) or iteration > int(t.end_iter):
            return False
        return iteration % int(t.freq) == 0

    def update_grad_stats(self, container: UltraGaussContainer):
        means_param = container._param_dict["means"].tensor
        if means_param.grad is None:
            return
        n = int(means_param.shape[0])
        if self.xyz_grad_accum.shape[0] != n:
            self.xyz_grad_accum = torch.zeros((n, 1), device=container.device)
            self.denom = torch.zeros((n, 1), device=container.device)
        grad_norm = torch.norm(means_param.grad, dim=-1, keepdim=True)
        self.xyz_grad_accum += grad_norm
        self.denom += 1

    @torch.no_grad()
    def prune_and_densify(self, iteration: int, solver: UltraGaussSolver, container: UltraGaussContainer):
        if container.num_points == 0:
            return
        th = self.ctrl_cfg.thresholds
        n_before = int(container.num_points)

        grads = self.xyz_grad_accum / self.denom.clamp(min=1.0)
        grads[grads.isnan()] = 0.0
        grads = grads.squeeze(-1)
        max_grad = float(grads.max().item()) if grads.numel() > 0 else 0.0
        mean_grad = float(grads.mean().item()) if grads.numel() > 0 else 0.0

        scales = _approx_scale_from_precision_tril(
            container._param_dict["precision_tril"].tensor,
            float(container.model_cfg.render.diag_beta),
        )
        max_scale = torch.max(scales, dim=1).values
        high_grad = grads > float(th.densify_grad)
        if bool(self.ctrl_cfg.enable.densify):
            split_mask = high_grad & (max_scale > float(th.split_scale_mm))
            clone_mask = high_grad & (~split_mask)
        else:
            split_mask = torch.zeros_like(high_grad, dtype=torch.bool)
            clone_mask = torch.zeros_like(high_grad, dtype=torch.bool)

        new_tensors, n_clone, n_split = self._generate_new_points(container, clone_mask, split_mask)
        if new_tensors:
            solver.concat(new_tensors)

        scales_new = _approx_scale_from_precision_tril(
            container._param_dict["precision_tril"].tensor,
            float(container.model_cfg.render.diag_beta),
        )
        max_scale_new = torch.max(scales_new, dim=1).values
        if bool(self.ctrl_cfg.enable.prune):
            prune_mask = (max_scale_new < float(th.min_scale_mm)) | (max_scale_new > float(th.max_scale_mm))
            prune_mask = prune_mask | (container.opacities < float(th.prune_opacity))
            prune_mask = prune_mask | (container.intensities < float(th.prune_intensity))
        else:
            prune_mask = torch.zeros((int(container.num_points),), device=container.device, dtype=torch.bool)

        n_after = int(container.num_points)
        n_max = int(self.ctrl_cfg.n_max)
        overflow = 0
        if n_after > n_max:
            overflow = n_after - n_max
            g = self.xyz_grad_accum.squeeze(-1)
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            _, idx_small = torch.topk(g, k=overflow, largest=False)
            prune_mask[idx_small] = True

        n_prune = int(prune_mask.sum().item())
        if prune_mask.any():
            solver.prune(prune_mask)

        n_final = int(container.num_points)
        self.last_event_stats = {
            "iteration": float(iteration),
            "num_gaussians_before": float(n_before),
            "num_gaussians_after": float(n_final),
            "num_gaussians_delta": float(n_final - n_before),
            "max_grad": float(max_grad),
            "mean_grad": float(mean_grad),
            "densify_grad_threshold": float(th.densify_grad),
            "clone": float(n_clone),
            "split": float(n_split),
            "prune": float(n_prune),
            "prune_nmax": float(overflow),
        }
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)

    def _generate_new_points(
        self,
        container: UltraGaussContainer,
        clone_mask: torch.Tensor,
        split_mask: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        n_clone = int(clone_mask.sum().item())
        n_split = int(split_mask.sum().item())
        if n_clone == 0 and n_split == 0:
            return {}, 0, 0

        param_dict = {name: wrapper.tensor for name, wrapper in container._param_dict.items()}
        new_tensors_list = {name: [] for name in param_dict.keys()}

        if n_clone > 0:
            for name, tensor in param_dict.items():
                new_tensors_list[name].append(tensor[clone_mask])

        if n_split > 0:
            for name, tensor in param_dict.items():
                repeats = [2] + [1] * (tensor.dim() - 1)
                t_split = tensor[split_mask].repeat(*repeats)
                new_tensors_list[name].append(t_split)
            split_means = new_tensors_list["means"][-1]
            split_precision = new_tensors_list["precision_tril"][-1]
            beta = float(container.model_cfg.render.diag_beta)
            l = _build_precision_cholesky_from_tril(split_precision, beta)
            # UltraGauss Eq.(12): y = mu + L^{-T} z, z~N(0,I)
            z = torch.randn_like(split_means)
            delta = torch.linalg.solve_triangular(l.transpose(1, 2), z.unsqueeze(-1), upper=True).squeeze(-1)
            split_means.add_(delta)

        final_tensors = {name: torch.cat(t_list, dim=0) for name, t_list in new_tensors_list.items()}
        return final_tensors, n_clone, n_split
