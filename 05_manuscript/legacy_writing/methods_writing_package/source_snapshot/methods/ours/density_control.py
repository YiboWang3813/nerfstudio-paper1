"""Offline density control for ours (clone/split/prune); logic aligned with ``methods/ultrags/density_control``."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from configs.root import RootConfig
from utils.math_utils import sigmoid_inv

from .configs import ModelConfig
from .gaussian_container import OursGaussContainer
from .gaussian_solver import OursGaussSolver


def _normalized_active_indices(
    active_global_indices: Optional[torch.Tensor], n_pts: int
) -> Optional[torch.Tensor]:
    if active_global_indices is None or active_global_indices.numel() == 0 or n_pts <= 0:
        return None
    ia = torch.unique(active_global_indices.long().view(-1))
    ia = ia[(ia >= 0) & (ia < n_pts)]
    if ia.numel() == 0:
        return None
    return ia


def _raw_param_1d(container: OursGaussContainer, name: str) -> torch.Tensor:
    t = container._param_dict[name].tensor
    return t.view(t.shape[0], -1).squeeze(-1)


def _raw_param_2d(container: OursGaussContainer, name: str) -> torch.Tensor:
    return container._param_dict[name].tensor


def _topk_mask(scores: torch.Tensor, k: int, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if k <= 0 or scores.numel() == 0:
        return mask
    if valid is None:
        valid = scores > 0
    else:
        valid = valid & (scores > 0)
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return mask
    k = min(int(k), int(valid_idx.numel()))
    local_scores = scores.index_select(0, valid_idx)
    picked = torch.topk(local_scores, k=k, largest=True).indices
    mask.index_fill_(0, valid_idx.index_select(0, picked), True)
    return mask


class OursOfflineDensityController:
    """观测梯度 → clone/split/prune → 调用 ``OursGaussSolver`` 改拓扑。"""

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        self.thresh = self.model_cfg.offline.density_control.thresholds
        self.switches = self.model_cfg.offline.density_control.enable
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.resp_contribution = torch.empty(0)
        self.resp_residual_contribution = torch.empty(0)
        self.resp_max_residual_hits = torch.empty(0)
        self.resp_support = torch.empty(0)
        self.resp_extra_stats: Dict[str, float] = {}
        self.last_event_stats: Dict[str, float] = {}
        print(
            f"[OursDensityController] Init Densify={self.switches.densify}, Prune={self.switches.prune}"
        )

    @property
    def strategy(self) -> str:
        return str(getattr(self.model_cfg.offline.density_control, "strategy", "gradient")).lower()

    @property
    def responsibility_cfg(self):
        return self.model_cfg.offline.density_control.responsibility

    def uses_responsibility(self) -> bool:
        return self.strategy.startswith("contribution_residual") or bool(
            getattr(self.responsibility_cfg, "enabled", False)
        )

    def should_accumulate_grad_stats(self, iteration: int) -> bool:
        if not (self.switches.densify or self.switches.prune):
            return False
        t = self.model_cfg.offline.density_control.timing
        return int(t.start_iter) <= iteration <= int(t.end_iter)

    def should_densify_prune(self, iteration: int) -> bool:
        if not (self.switches.densify or self.switches.prune):
            return False
        t = self.model_cfg.offline.density_control.timing
        if iteration < int(t.start_iter) or iteration > int(t.end_iter):
            return False
        return iteration % int(t.freq) == 0

    def should_collect_responsibility_stats(self, iteration: int) -> bool:
        return self.uses_responsibility() and self.should_densify_prune(iteration)

    def scalar_stats(self, iteration: int) -> Dict[str, float]:
        if int(self.last_event_stats.get("iteration", -1)) != int(iteration):
            return {}
        return {f"train/dc/{k}": v for k, v in self.last_event_stats.items() if k != "iteration"}

    def update_grad_stats(
        self,
        container: OursGaussContainer,
        active_global_indices: Optional[torch.Tensor] = None,
        grad_scale: float = 1.0,
    ):
        if "means" not in container._param_dict:
            return
        means_param = container._param_dict["means"].tensor
        if means_param.grad is None:
            return
        current_N = means_param.shape[0]
        if self.xyz_grad_accum.shape[0] != current_N:
            self.xyz_grad_accum = torch.zeros((current_N, 1), device=container.device)
            self.denom = torch.zeros((current_N, 1), device=container.device)
        grad = means_param.grad
        if float(grad_scale) != 1.0:
            grad = grad * float(grad_scale)
        if active_global_indices is not None and active_global_indices.numel() > 0:
            idx = torch.unique(active_global_indices.long().view(-1))
            gn = torch.norm(grad.index_select(0, idx), dim=-1, keepdim=True)
            self.xyz_grad_accum.index_add_(0, idx, gn)
            self.denom.index_add_(
                0,
                idx,
                torch.ones((idx.numel(), 1), device=container.device, dtype=self.denom.dtype),
            )
        else:
            grad_norm = torch.norm(grad, dim=-1, keepdim=True)
            self.xyz_grad_accum += grad_norm
            self.denom += 1

    def update_responsibility_stats(
        self,
        container: OursGaussContainer,
        contribution: torch.Tensor,
        residual_contribution: torch.Tensor,
        max_residual_hits: torch.Tensor,
        support: torch.Tensor,
    ) -> None:
        current_N = int(container.num_points)
        if current_N <= 0:
            return
        tensors = [
            contribution.detach().view(-1),
            residual_contribution.detach().view(-1),
            max_residual_hits.detach().view(-1),
            support.detach().view(-1),
        ]
        if any(int(t.numel()) != current_N for t in tensors):
            raise ValueError(
                "responsibility stats length mismatch: "
                f"expected {current_N}, got {[int(t.numel()) for t in tensors]}"
            )
        if self.resp_contribution.shape[0] != current_N:
            self.resp_contribution = torch.zeros((current_N,), device=container.device)
            self.resp_residual_contribution = torch.zeros((current_N,), device=container.device)
            self.resp_max_residual_hits = torch.zeros((current_N,), device=container.device)
            self.resp_support = torch.zeros((current_N,), device=container.device)
        self.resp_contribution += tensors[0].to(device=container.device, dtype=torch.float32)
        self.resp_residual_contribution += tensors[1].to(device=container.device, dtype=torch.float32)
        self.resp_max_residual_hits += tensors[2].to(device=container.device, dtype=torch.float32)
        self.resp_support += tensors[3].to(device=container.device, dtype=torch.float32)

    def update_responsibility_diag_stats(self, stats: Dict[str, float]) -> None:
        for key, value in stats.items():
            self.resp_extra_stats[key] = self.resp_extra_stats.get(key, 0.0) + float(value)

    def _reset_accumulators(self) -> None:
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.resp_contribution = torch.empty(0)
        self.resp_residual_contribution = torch.empty(0)
        self.resp_max_residual_hits = torch.empty(0)
        self.resp_support = torch.empty(0)
        self.resp_extra_stats = {}

    def _responsibility_scores(self, n_pts: int) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if self.resp_contribution.shape[0] != n_pts:
            return None, {"responsibility_available": 0.0}
        rcfg = self.responsibility_cfg
        contribution = self.resp_contribution
        residual_mean = self.resp_residual_contribution / contribution.clamp_min(1e-8)
        score = residual_mean * torch.log1p(contribution)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        positive = score[score > 0]
        if positive.numel() == 0:
            return score, {
                "responsibility_available": 1.0,
                "responsibility_positive": 0.0,
                "responsibility_score_threshold": 0.0,
                "responsibility_score_max": 0.0,
                "responsibility_score_mean": 0.0,
            }
        q = min(max(float(getattr(rcfg, "score_quantile", 0.98)), 0.0), 1.0)
        threshold = torch.quantile(positive, q)
        max_fraction = float(getattr(rcfg, "max_fraction", 0.05))
        if 0.0 < max_fraction < 1.0:
            k = max(1, int(n_pts * max_fraction))
            if positive.numel() > k:
                topk_threshold = torch.topk(score, k=k, largest=True).values[-1]
                threshold = torch.maximum(threshold, topk_threshold)
        min_score = float(getattr(rcfg, "min_score", 1e-6))
        threshold = torch.maximum(threshold, torch.tensor(min_score, device=score.device, dtype=score.dtype))
        return score, {
            "responsibility_available": 1.0,
            "responsibility_positive": float(positive.numel()),
            "responsibility_score_threshold": float(threshold.detach().item()),
            "responsibility_score_max": float(score.max().detach().item()),
            "responsibility_score_mean": float(score.mean().detach().item()),
        }

    def prune_and_densify(
        self,
        iteration: int,
        solver: OursGaussSolver,
        container: OursGaussContainer,
        gsm: Optional[Any] = None,
        birth_frame_id: Optional[int] = None,
        active_global_indices: Optional[torch.Tensor] = None,
    ):
        del gsm, birth_frame_id
        if not self.switches.densify and not self.switches.prune:
            return

        n_pts = int(container.num_points)
        n_before = n_pts
        idx_active = _normalized_active_indices(active_global_indices, n_pts)

        grads_full: Optional[torch.Tensor] = None
        max_grad = 0.0
        mean_grad = 0.0
        if self.switches.densify or self.switches.prune:
            if idx_active is not None:
                den = self.denom.index_select(0, idx_active).clamp(min=1)
                g_sel = (self.xyz_grad_accum.index_select(0, idx_active) / den).squeeze(-1)
                g_sel = torch.nan_to_num(g_sel, nan=0.0)
                if g_sel.numel() > 0:
                    max_grad = float(g_sel.max().item())
                    mean_grad = float(g_sel.mean().item())
            else:
                grads_full = self.xyz_grad_accum / self.denom.clamp(min=1)
                grads_full[grads_full.isnan()] = 0.0
                if grads_full.numel() > 0:
                    max_grad = float(grads_full.max().item())
                    mean_grad = float(grads_full.mean().item())
                grads_full = grads_full.squeeze(-1)

            print(
                f"[Ours DC Check] Iter {iteration} | Max Grad: {max_grad:.2e} | Mean Grad: {mean_grad:.2e} | "
                f"Thresh: {self.thresh.densify_grad}"
                + (" | active_sparse_dc" if idx_active is not None else "")
            )

        scales = container.scales
        intensities = container.intensities
        opacities = container.opacities
        stats = {
            "clone": 0,
            "split": 0,
            "prune_weak": 0,
            "prune_scale": 0,
            "high_grad": 0,
            "split_candidates": 0,
            "clone_candidates": 0,
            "active_count": int(idx_active.numel()) if idx_active is not None else int(n_pts),
            "responsibility_selected": 0,
            "responsibility_gradient_union_selected": 0,
            "responsibility_fallback": 0,
        }
        resp_score: Optional[torch.Tensor] = None
        resp_stats: Dict[str, float] = {"responsibility_available": 0.0}
        if self.uses_responsibility():
            resp_score, resp_stats = self._responsibility_scores(n_pts)

        if self.switches.densify:
            min_scale_limit = 0.01
            use_resp = (
                self.strategy.startswith("contribution_residual")
                and resp_score is not None
                and bool((resp_score > 0).any().item())
            )
            if use_resp:
                threshold = float(resp_stats.get("responsibility_score_threshold", 0.0))
                high_resp_mask = resp_score >= threshold
                mix_mode = str(getattr(self.responsibility_cfg, "mix_mode", "responsibility")).lower()
                if self.strategy == "contribution_residual_mixed" or mix_mode == "union_budget":
                    if grads_full is not None:
                        grad_score_full = grads_full
                    else:
                        grad_score_full = torch.zeros((n_pts,), device=scales.device, dtype=torch.float32)
                        if idx_active is not None and idx_active.numel() > 0:
                            den = self.denom.index_select(0, idx_active).clamp(min=1)
                            g_sel = (self.xyz_grad_accum.index_select(0, idx_active) / den).squeeze(-1)
                            g_sel = torch.nan_to_num(g_sel, nan=0.0)
                            grad_score_full.index_copy_(0, idx_active, g_sel.to(grad_score_full.dtype))
                    resp_count = int(high_resp_mask.sum().item())
                    grad_fraction = max(
                        0.0, float(getattr(self.responsibility_cfg, "union_gradient_fraction", 0.50))
                    )
                    grad_budget = int(round(resp_count * grad_fraction))
                    grad_min_factor = max(
                        0.0, float(getattr(self.responsibility_cfg, "union_gradient_min_factor", 0.25))
                    )
                    grad_valid = (~high_resp_mask) & (grad_score_full > float(self.thresh.densify_grad) * grad_min_factor)
                    high_grad_union = _topk_mask(grad_score_full, grad_budget, grad_valid)
                    high_resp_mask = high_resp_mask | high_grad_union
                    stats["responsibility_gradient_union_selected"] = int(high_grad_union.sum().item())
                stats["responsibility_selected"] = int(high_resp_mask.sum().item())
                max_scale = torch.max(scales, dim=1).values
                split_mask = high_resp_mask & (max_scale > self.thresh.split_scale)
                clone_mask = high_resp_mask & (max_scale <= self.thresh.split_scale) & (max_scale > min_scale_limit)
                stats["split_candidates"] = int(split_mask.sum().item())
                stats["clone_candidates"] = int(clone_mask.sum().item())
            elif idx_active is not None:
                if self.strategy.startswith("contribution_residual"):
                    stats["responsibility_fallback"] = 1
                den = self.denom.index_select(0, idx_active).clamp(min=1)
                g_sel = (self.xyz_grad_accum.index_select(0, idx_active) / den).squeeze(-1)
                g_sel = torch.nan_to_num(g_sel, nan=0.0)
                high = g_sel > self.thresh.densify_grad
                max_scale_sub = torch.max(scales.index_select(0, idx_active), dim=1).values
                split_sel = (high & (max_scale_sub > self.thresh.split_scale)).reshape(-1)
                clone_sel = (
                    high & (max_scale_sub <= self.thresh.split_scale) & (max_scale_sub > min_scale_limit)
                ).reshape(-1)
                stats["high_grad"] = int(high.sum().item())
                stats["split_candidates"] = int(split_sel.sum().item())
                stats["clone_candidates"] = int(clone_sel.sum().item())
                clone_mask = torch.zeros((n_pts,), device=scales.device, dtype=torch.bool)
                split_mask = torch.zeros((n_pts,), device=scales.device, dtype=torch.bool)
                clone_mask.index_copy_(0, idx_active, clone_sel)
                split_mask.index_copy_(0, idx_active, split_sel)
            else:
                high_grad_mask = grads_full > self.thresh.densify_grad
                max_scale = torch.max(scales, dim=1).values
                split_mask = high_grad_mask & (max_scale > self.thresh.split_scale)
                clone_mask = high_grad_mask & (max_scale <= self.thresh.split_scale) & (max_scale > min_scale_limit)
                stats["high_grad"] = int(high_grad_mask.sum().item())
                stats["split_candidates"] = int(split_mask.sum().item())
                stats["clone_candidates"] = int(clone_mask.sum().item())

            new_tensors, n_clone, n_split, _parent_indices = self._generate_new_points(
                container, clone_mask, split_mask
            )
            if n_clone > 0 or n_split > 0:
                old_n_pts = n_pts
                solver.concat(new_tensors)
                stats["clone"] = n_clone
                stats["split"] = n_split
                scales = container.scales
                intensities = container.intensities
                opacities = container.opacities
                n_pts = int(container.num_points)
                if resp_score is not None and resp_score.shape[0] == old_n_pts:
                    pad = torch.zeros((n_pts - old_n_pts,), device=resp_score.device, dtype=resp_score.dtype)
                    resp_score = torch.cat([resp_score, pad], dim=0)

        if self.switches.prune:
            if idx_active is not None:
                op_sub = opacities.index_select(0, idx_active).squeeze()
                int_sub = intensities.index_select(0, idx_active).squeeze()
                is_transparent_sub = op_sub < self.thresh.prune_opacity
                is_dark_sub = int_sub < self.thresh.prune_intensity
                weak_sub = (is_transparent_sub | is_dark_sub).reshape(-1)
                final_prune_mask = torch.zeros((n_pts,), device=opacities.device, dtype=torch.bool)
                final_prune_mask.index_copy_(0, idx_active, weak_sub)
            else:
                is_transparent = (opacities < self.thresh.prune_opacity).squeeze()
                is_dark = (intensities < self.thresh.prune_intensity).squeeze()
                final_prune_mask = is_transparent | is_dark
            if self.uses_responsibility() and self.resp_contribution.shape[0] > 0:
                old_n = int(self.resp_contribution.shape[0])
                if old_n < n_pts:
                    pad = torch.full((n_pts - old_n,), float("inf"), device=opacities.device, dtype=torch.float32)
                    contribution = torch.cat([self.resp_contribution.to(opacities.device), pad], dim=0)
                else:
                    contribution = self.resp_contribution[:n_pts].to(opacities.device)
                rcfg = self.responsibility_cfg
                low_q = min(max(float(getattr(rcfg, "low_contribution_quantile", 0.10)), 0.0), 1.0)
                contrib_threshold = torch.quantile(contribution, low_q) if contribution.numel() > 0 else 0.0
                low_contribution = contribution <= contrib_threshold
                atten = container.attenuations.squeeze()
                if bool(getattr(rcfg, "protect_attenuation_enabled", True)):
                    att_q = min(max(float(getattr(rcfg, "protect_attenuation_quantile", 0.90)), 0.0), 1.0)
                    att_threshold = (
                        torch.quantile(atten, att_q)
                        if atten.numel() > 0
                        else torch.tensor(float("inf"), device=atten.device)
                    )
                    attenuation_protect = atten >= att_threshold
                else:
                    attenuation_protect = torch.zeros_like(final_prune_mask, dtype=torch.bool)
                final_prune_mask = final_prune_mask & low_contribution & (~attenuation_protect)

            max_scale_mm = float(getattr(self.thresh, "max_scale_mm", 0.0))
            if max_scale_mm > 0.0:
                scale_prune_mask = torch.max(scales, dim=1).values > max_scale_mm
                stats["prune_scale"] = int(scale_prune_mask.sum().item())
                final_prune_mask = final_prune_mask | scale_prune_mask

            stats["prune_weak"] = int(final_prune_mask.sum().item())
            if final_prune_mask.sum() > 0:
                solver.prune(final_prune_mask)

        if stats["split"] + stats["clone"] + stats["prune_weak"] > 0:
            print(
                f"[Ours DC] Iter {iteration} | Split: {stats['split']} Clone: {stats['clone']} | "
                f"Prune: {stats['prune_weak']}(Weak+Scale) ScalePrune: {stats['prune_scale']}"
            )
        self.last_event_stats = {
            "iteration": float(iteration),
            "max_grad": float(max_grad),
            "mean_grad": float(mean_grad),
            "densify_grad_threshold": float(self.thresh.densify_grad),
            "num_gaussians_before": float(n_before),
            "num_gaussians_after": float(container.num_points),
            "active_count": float(stats["active_count"]),
            "high_grad": float(stats["high_grad"]),
            "split_candidates": float(stats["split_candidates"]),
            "clone_candidates": float(stats["clone_candidates"]),
            "split": float(stats["split"]),
            "clone": float(stats["clone"]),
            "prune_weak": float(stats["prune_weak"]),
            "prune_scale": float(stats["prune_scale"]),
            "responsibility_selected": float(stats["responsibility_selected"]),
            "responsibility_gradient_union_selected": float(stats["responsibility_gradient_union_selected"]),
            "responsibility_fallback": float(stats["responsibility_fallback"]),
            "echo_alpha_mean": float(container.echo_alphas.mean().detach().item()) if container.num_points > 0 else 0.0,
            "attenuation_mean": float(container.attenuations.mean().detach().item()) if container.num_points > 0 else 0.0,
        }
        self.last_event_stats.update(resp_stats)
        if self.resp_contribution.shape[0] > 0:
            self.last_event_stats.update(
                {
                    "responsibility_contribution_mean": float(self.resp_contribution.mean().detach().item()),
                    "responsibility_support_mean": float(self.resp_support.mean().detach().item()),
                    "responsibility_max_hit_sum": float(self.resp_max_residual_hits.sum().detach().item()),
                }
            )
        self.last_event_stats.update(self.resp_extra_stats)
        self._reset_accumulators()

    def _generate_new_points(
        self,
        container: OursGaussContainer,
        clone_mask: torch.Tensor,
        split_mask: torch.Tensor,
    ) -> Tuple[Dict, int, int, Optional[torch.Tensor]]:
        n_clone = clone_mask.sum().item()
        n_split = split_mask.sum().item()
        if n_clone == 0 and n_split == 0:
            return {}, 0, 0, None
        param_dict = {name: wrapper.tensor for name, wrapper in container._param_dict.items()}
        new_tensors_list = {name: [] for name in param_dict.keys()}
        parent_chunks: List[torch.Tensor] = []
        if n_clone > 0:
            clone_idx = torch.nonzero(clone_mask, as_tuple=False).squeeze(-1)
            for name, tensor in param_dict.items():
                new_tensors_list[name].append(tensor[clone_mask])
            parent_chunks.append(clone_idx)
        if n_split > 0:
            split_idx = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)
            for name, tensor in param_dict.items():
                repeats = [2] + [1] * (tensor.dim() - 1)
                t_split = tensor[split_mask].repeat(*repeats)
                if name == "scales":
                    t_split -= torch.log(torch.tensor(2.0, device=container.device))
                new_tensors_list[name].append(t_split)
            parent_chunks.append(split_idx.repeat_interleave(2))
            split_scales_log = new_tensors_list["scales"][-1]
            split_scales = torch.exp(split_scales_log)
            split_means = new_tensors_list["means"][-1]
            perturb = torch.randn_like(split_means) * split_scales * 0.1
            split_means.add_(perturb)
        final_tensors = {name: torch.cat(t_list, dim=0) for name, t_list in new_tensors_list.items()}
        parent_indices = torch.cat(parent_chunks, dim=0) if parent_chunks else None
        return final_tensors, n_clone, n_split, parent_indices
