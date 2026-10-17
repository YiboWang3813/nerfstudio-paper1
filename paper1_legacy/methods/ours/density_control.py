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


def transmission_from_tau(tau: torch.Tensor, ray_density: torch.Tensor) -> torch.Tensor:
    """Renderer transmission formula used by attenuation-protection tests."""
    return tau + (1.0 - tau) * torch.exp(-ray_density)


def attenuation_protection_mask(
    attenuations: torch.Tensor,
    mode: str,
    quantile: float,
    support: Optional[torch.Tensor] = None,
    path_contribution: Optional[torch.Tensor] = None,
    path_threshold_mode: str = "quantile",
    path_threshold: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return pruning-protection mask for P1/P2/P3 attenuation variants.

    ``attenuations`` is the learned tau-like lower transmission bound. Lower values mean
    stronger possible attenuation. ``support`` is diagnostic ray/Gaussian encounter
    support; when present, ``path_effect`` uses the allowed approximation
    ``(1 - tau) * support``.  ``path_contribution`` is the renderer-consistent
    accumulated ``A_g = sum(1-T_igp)`` statistic used by the production
    path-aware mode.
    """
    atten = attenuations.detach().view(-1)
    if atten.numel() == 0:
        return torch.zeros_like(atten, dtype=torch.bool), {
            "attenuation_protect_count": 0.0,
            "attenuation_protect_threshold": 0.0,
            "path_effect_score_mean": 0.0,
            "path_effect_score_max": 0.0,
        }
    q = min(max(float(quantile), 0.0), 1.0)
    mode_norm = str(mode or "high_tau").lower()
    if mode_norm in ("none", "off", "false"):
        score = torch.zeros_like(atten)
        mask = torch.zeros_like(atten, dtype=torch.bool)
        threshold = torch.tensor(0.0, device=atten.device, dtype=atten.dtype)
    elif mode_norm in ("high_tau", "p1", "legacy"):
        score = atten
        threshold = torch.quantile(score, q)
        mask = score >= threshold
    elif mode_norm in ("low_tau", "p2", "strong_attenuation"):
        score = 1.0 - atten
        threshold = torch.quantile(score, q)
        mask = score >= threshold
    elif mode_norm in ("path_effect", "p3"):
        if support is None or int(support.numel()) != int(atten.numel()):
            support_score = torch.ones_like(atten)
        else:
            support_score = support.detach().to(device=atten.device, dtype=atten.dtype).view(-1).clamp_min(0.0)
        score = (1.0 - atten).clamp_min(0.0) * support_score
        positive = score[score > 0]
        if positive.numel() == 0:
            threshold = torch.tensor(float("inf"), device=atten.device, dtype=atten.dtype)
            mask = torch.zeros_like(atten, dtype=torch.bool)
        else:
            threshold = torch.quantile(positive, q)
            mask = score >= threshold
    elif mode_norm in ("path_contribution", "path_aware", "ag"):
        if path_contribution is None or int(path_contribution.numel()) != int(atten.numel()):
            # Missing/misaligned A_g must fail safe: protect everything instead of
            # silently reverting to tau or deleting Gaussians without path evidence.
            score = torch.zeros_like(atten)
            threshold = torch.tensor(float("nan"), device=atten.device, dtype=atten.dtype)
            mask = torch.ones_like(atten, dtype=torch.bool)
            available = 0.0
        else:
            raw_score = path_contribution.detach().to(device=atten.device, dtype=atten.dtype).view(-1)
            invalid = ~torch.isfinite(raw_score)
            score = torch.nan_to_num(raw_score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            threshold_mode = str(path_threshold_mode or "quantile").lower()
            if threshold_mode == "quantile":
                threshold = torch.quantile(score, q) if score.numel() > 0 else torch.tensor(
                    0.0, device=atten.device, dtype=atten.dtype
                )
            elif threshold_mode == "fixed":
                threshold = torch.tensor(
                    max(0.0, float(path_threshold)), device=atten.device, dtype=atten.dtype
                )
            else:
                raise ValueError(
                    "path_contribution_threshold_mode must be 'quantile' or 'fixed'; got "
                    f"{path_threshold_mode!r}."
                )
            # Invalid statistics are a correctness error recorded by diagnostics. They
            # fail safe for topology: never prune those Gaussians in the affected event.
            mask = (score >= threshold) | invalid
            available = 1.0
    else:
        raise ValueError(
            "model.offline.density_control.responsibility.protect_attenuation_mode must be "
            "one of high_tau, low_tau, path_effect, path_contribution, none; got "
            f"{mode!r}."
        )
    out = {
        "attenuation_protect_count": float(mask.sum().detach().item()),
        "attenuation_protect_threshold": float(threshold.detach().item())
        if torch.isfinite(threshold).detach().item()
        else float("inf"),
        "path_effect_score_mean": float(score.mean().detach().item()) if score.numel() > 0 else 0.0,
        "path_effect_score_max": float(score.max().detach().item()) if score.numel() > 0 else 0.0,
    }
    if mode_norm in ("path_contribution", "path_aware", "ag"):
        out["path_contribution_available"] = available
        out["path_contribution_score_mean"] = out["path_effect_score_mean"]
        out["path_contribution_score_max"] = out["path_effect_score_max"]
    return mask, out


def _distribution_stats(prefix: str, values: torch.Tensor) -> Dict[str, float]:
    """Compact finite-value distribution summary suitable for event scalar logs."""
    x = values.detach().view(-1).to(dtype=torch.float32)
    nan_count = int(torch.isnan(x).sum().item())
    inf_count = int(torch.isinf(x).sum().item())
    finite = x[torch.isfinite(x)]
    out = {
        f"{prefix}_nan_count": float(nan_count),
        f"{prefix}_inf_count": float(inf_count),
        f"{prefix}_nonzero_ratio": float((finite > 0).float().mean().item()) if finite.numel() else 0.0,
    }
    if finite.numel() == 0:
        for name in ("min", "mean", "median", "p10", "p20", "p50", "p80", "p90", "max"):
            out[f"{prefix}_{name}"] = 0.0
        return out
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.10, 0.20, 0.50, 0.80, 0.90], device=finite.device, dtype=finite.dtype),
    )
    out.update(
        {
            f"{prefix}_min": float(finite.min().item()),
            f"{prefix}_mean": float(finite.mean().item()),
            f"{prefix}_median": float(quantiles[2].item()),
            f"{prefix}_p10": float(quantiles[0].item()),
            f"{prefix}_p20": float(quantiles[1].item()),
            f"{prefix}_p50": float(quantiles[2].item()),
            f"{prefix}_p80": float(quantiles[3].item()),
            f"{prefix}_p90": float(quantiles[4].item()),
            f"{prefix}_max": float(finite.max().item()),
        }
    )
    return out


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().view(-1)
    y = y.float().view(-1)
    valid = torch.isfinite(x) & torch.isfinite(y)
    if int(valid.sum().item()) < 2:
        return 0.0
    x = x[valid] - x[valid].mean()
    y = y[valid] - y[valid].mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(denom.item()) <= 0.0:
        return 0.0
    return float(((x * y).sum() / denom).item())


def _average_tie_ranks(x: torch.Tensor) -> torch.Tensor:
    values, inverse, counts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True)
    del values
    before = torch.cumsum(counts, dim=0) - counts
    average = before.to(torch.float32) + (counts.to(torch.float32) - 1.0) * 0.5
    return average.index_select(0, inverse)


def _path_tau_diagnostics(path: torch.Tensor, tau: torch.Tensor, high_q: float = 0.80) -> Dict[str, float]:
    a = path.detach().float().view(-1)
    t = tau.detach().float().view(-1)
    valid = torch.isfinite(a) & torch.isfinite(t)
    a = a[valid]
    t = t[valid]
    if a.numel() == 0:
        return {
            "path_tau_pearson": 0.0,
            "path_tau_spearman": 0.0,
            "high_path_in_high_tau_ratio": 0.0,
            "high_tau_low_path_ratio": 0.0,
        }
    q = min(max(float(high_q), 0.0), 1.0)
    a_thr = torch.quantile(a, q)
    t_thr = torch.quantile(t, q)
    low_a_thr = torch.quantile(a, 1.0 - q)
    high_a = a >= a_thr
    high_t = t >= t_thr
    low_a = a <= low_a_thr
    return {
        "path_tau_pearson": _pearson(a, t),
        "path_tau_spearman": _pearson(_average_tie_ranks(a), _average_tie_ranks(t)),
        "high_path_in_high_tau_ratio": float((high_a & high_t).sum().item()) / max(1, int(high_a.sum().item())),
        "high_tau_low_path_ratio": float((high_t & low_a).sum().item()) / max(1, int(high_t.sum().item())),
        "path_high_threshold_q80": float(a_thr.item()),
        "tau_high_threshold_q80": float(t_thr.item()),
        "path_low_threshold_q20": float(low_a_thr.item()),
    }


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
        self.resp_path_contribution = torch.empty(0)
        self.resp_path_support = torch.empty(0)
        self.path_frames_accumulated = 0
        self.resp_extra_stats: Dict[str, float] = {}
        self.last_event_stats: Dict[str, float] = {}
        print(
            f"[OursDensityController] Init Densify={self.switches.densify}, Prune={self.switches.prune}"
        )

    @property
    def strategy(self) -> str:
        policy = getattr(self.model_cfg.offline.density_control, "densification_policy", None)
        if policy is not None:
            if policy not in ("gradient", "residual_guided"):
                raise ValueError(f"Unknown densification_policy: {policy}")
            return "contribution_residual" if policy == "residual_guided" else "gradient"
        return str(getattr(self.model_cfg.offline.density_control, "strategy", "gradient")).lower()

    def global_path_available(self) -> bool:
        sr = self.model_cfg.slice_render
        return bool(sr.ray_attenuation_enabled) and sr.ray_attenuation_beta > 0 and sr.ray_attenuation_strength > 0

    @property
    def pruning_policy(self) -> str:
        if not self.switches.prune:
            return "none"
        dc = self.model_cfg.offline.density_control
        policy = getattr(dc, "pruning_policy", None)
        if policy is None:
            residual = self.strategy.startswith("contribution_residual") or bool(self.responsibility_cfg.enabled)
            policy = ("path_aware" if self.responsibility_cfg.protect_attenuation_enabled else "local_aware") if residual else "opacity"
        if policy not in ("none", "opacity", "local_aware", "path_aware"):
            raise ValueError(f"Unknown pruning_policy: {policy}")
        if policy == "path_aware" and not self.global_path_available():
            return "local_aware"
        return policy

    @property
    def responsibility_cfg(self):
        return self.model_cfg.offline.density_control.responsibility

    @property
    def protection_mode(self) -> str:
        if getattr(self.model_cfg.offline.density_control, "pruning_policy", None) == "path_aware":
            return "path_contribution"
        return str(getattr(self.responsibility_cfg, "protect_attenuation_mode", "high_tau")).lower()

    def uses_responsibility(self) -> bool:
        dc = self.model_cfg.offline.density_control
        if getattr(dc, "densification_policy", None) is not None or getattr(dc, "pruning_policy", None) is not None:
            return self.strategy.startswith("contribution_residual") or self.pruning_policy in ("local_aware", "path_aware")
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
        # Preserve the production densification/local-pruning protocol: local
        # responsibility is sampled on the adaptation-event frame.
        return self.uses_responsibility() and self.should_densify_prune(iteration)

    def should_collect_path_stats(self, iteration: int) -> bool:
        rcfg = self.responsibility_cfg
        mode = self.protection_mode
        timing = self.model_cfg.offline.density_control.timing
        freq = max(1, int(timing.freq))
        # Do not accumulate an interval after the final adaptation event: those
        # samples can never be consumed and would only add renderer/CSR overhead.
        last_event = (int(timing.end_iter) // freq) * freq
        return (
            self.uses_responsibility()
            and self.pruning_policy == "path_aware"
            and mode in ("path_contribution", "path_aware", "ag")
            and self.should_accumulate_grad_stats(iteration)
            and iteration <= last_event
        )

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

    def update_path_stats(
        self,
        container: OursGaussContainer,
        path_contribution: torch.Tensor,
        path_support: torch.Tensor,
    ) -> None:
        current_N = int(container.num_points)
        if current_N <= 0:
            return
        path = path_contribution.detach().view(-1)
        support = path_support.detach().view(-1)
        if int(path.numel()) != current_N or int(support.numel()) != current_N:
            raise ValueError(
                "path contribution stats length mismatch: "
                f"expected {current_N}, got {[int(path.numel()), int(support.numel())]}"
            )
        if self.resp_path_contribution.numel() == 0:
            self.resp_path_contribution = torch.zeros((current_N,), device=container.device)
            self.resp_path_support = torch.zeros((current_N,), device=container.device)
        elif self.resp_path_contribution.shape[0] != current_N:
            raise RuntimeError(
                "Gaussian population changed inside a path-statistics interval: "
                f"accumulator={self.resp_path_contribution.shape[0]}, current={current_N}. "
                "Population changes must occur only at an adaptation event followed by reset."
            )
        self.resp_path_contribution += path.to(device=container.device, dtype=torch.float32)
        self.resp_path_support += support.to(device=container.device, dtype=torch.float32)
        self.path_frames_accumulated += 1

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
        self.resp_path_contribution = torch.empty(0)
        self.resp_path_support = torch.empty(0)
        self.path_frames_accumulated = 0
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
            "attenuation_protect_count": 0,
            "path_protected_candidate_count": 0,
            "weak_low_local_candidate_count": 0,
        }
        protection_stats: Dict[str, float] = {}
        interval_stats: Dict[str, float] = {
            "path_frames_accumulated": float(self.path_frames_accumulated)
        }
        resp_score: Optional[torch.Tensor] = None
        resp_stats: Dict[str, float] = {"responsibility_available": 0.0}
        if self.uses_responsibility():
            resp_score, resp_stats = self._responsibility_scores(n_pts)
        if self.resp_contribution.shape[0] == n_before:
            interval_stats.update(_distribution_stats("local_contribution", self.resp_contribution))
        if self.resp_path_contribution.shape[0] == n_before:
            interval_stats.update(_distribution_stats("path_contribution", self.resp_path_contribution))
            interval_stats.update(
                _path_tau_diagnostics(
                    self.resp_path_contribution,
                    container.attenuations.detach().view(-1)[:n_before],
                )
            )

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

        if self.pruning_policy != "none":
            if idx_active is not None:
                op_sub = opacities.index_select(0, idx_active).squeeze()
                is_transparent_sub = op_sub < self.thresh.prune_opacity
                weak_sub = is_transparent_sub.reshape(-1)
                final_prune_mask = torch.zeros((n_pts,), device=opacities.device, dtype=torch.bool)
                final_prune_mask.index_copy_(0, idx_active, weak_sub)
            else:
                is_transparent = (opacities < self.thresh.prune_opacity).squeeze()
                final_prune_mask = is_transparent
            if self.pruning_policy in ("local_aware", "path_aware") and self.resp_contribution.shape[0] > 0:
                old_n = int(self.resp_contribution.shape[0])
                if old_n < n_pts:
                    pad = torch.full((n_pts - old_n,), float("inf"), device=opacities.device, dtype=torch.float32)
                    contribution = torch.cat([self.resp_contribution.to(opacities.device), pad], dim=0)
                else:
                    contribution = self.resp_contribution[:n_pts].to(opacities.device)
                rcfg = self.responsibility_cfg
                low_q = min(max(float(getattr(rcfg, "low_contribution_quantile", 0.10)), 0.0), 1.0)
                contrib_threshold = torch.quantile(contribution, low_q) if contribution.numel() > 0 else 0.0
                # Preserve the production bottom-quantile tie policy.  A single-frame
                # C_g distribution contains many exact zeros, so strict ``< Q`` with
                # Q=0 would disable local pruning altogether.  ``<=`` includes the
                # tied lowest-contribution population without changing q_p.
                low_contribution = contribution <= contrib_threshold
                protection_stats["local_contribution_threshold"] = float(
                    contrib_threshold.detach().item()
                    if isinstance(contrib_threshold, torch.Tensor)
                    else contrib_threshold
                )
                atten = container.attenuations.squeeze()
                if self.pruning_policy == "path_aware":
                    protect_mode = self.protection_mode
                    if protect_mode in ("path_contribution", "path_aware", "ag"):
                        path_q = min(
                            max(float(getattr(rcfg, "path_contribution_quantile", 0.80)), 0.0), 1.0
                        )
                        old_path_n = int(self.resp_path_contribution.shape[0])
                        path_original = (
                            self.resp_path_contribution.to(opacities.device)
                            if old_path_n == old_n
                            else None
                        )
                        protect_original, protection_stats = attenuation_protection_mask(
                            atten[:old_n],
                            mode=protect_mode,
                            quantile=path_q,
                            path_contribution=path_original,
                            path_threshold_mode=str(
                                getattr(rcfg, "path_contribution_threshold_mode", "quantile")
                            ),
                            path_threshold=float(getattr(rcfg, "path_contribution_threshold", 0.0)),
                        )
                        protection_stats["local_contribution_threshold"] = float(
                            contrib_threshold.detach().item()
                            if isinstance(contrib_threshold, torch.Tensor)
                            else contrib_threshold
                        )
                        if old_n < n_pts:
                            # Clone/split children have no history for the interval that just
                            # ended. Protect them in this event and start them at zero after reset.
                            attenuation_protect = torch.cat(
                                [
                                    protect_original,
                                    torch.ones(
                                        (n_pts - old_n,), device=opacities.device, dtype=torch.bool
                                    ),
                                ]
                            )
                            path_for_decision = torch.cat(
                                [
                                    path_original
                                    if path_original is not None
                                    else torch.zeros((old_n,), device=opacities.device),
                                    torch.full(
                                        (n_pts - old_n,),
                                        float("inf"),
                                        device=opacities.device,
                                        dtype=torch.float32,
                                    ),
                                ]
                            )
                        else:
                            attenuation_protect = protect_original
                            path_for_decision = (
                                path_original
                                if path_original is not None
                                else torch.zeros((n_pts,), device=opacities.device)
                            )
                    else:
                        att_q = min(
                            max(float(getattr(rcfg, "protect_attenuation_quantile", 0.90)), 0.0), 1.0
                        )
                        support_for_protection = None
                        if self.resp_support.shape[0] > 0:
                            old_s = int(self.resp_support.shape[0])
                            if old_s < n_pts:
                                support_for_protection = torch.cat(
                                    [
                                        self.resp_support.to(opacities.device),
                                        torch.zeros(
                                            (n_pts - old_s,), device=opacities.device, dtype=torch.float32
                                        ),
                                    ],
                                    dim=0,
                                )
                            else:
                                support_for_protection = self.resp_support[:n_pts].to(opacities.device)
                        attenuation_protect, protection_stats = attenuation_protection_mask(
                            atten,
                            mode=protect_mode,
                            quantile=att_q,
                            support=support_for_protection,
                        )
                        path_for_decision = torch.zeros((n_pts,), device=opacities.device)
                else:
                    attenuation_protect = torch.zeros_like(final_prune_mask, dtype=torch.bool)
                    path_for_decision = torch.zeros((n_pts,), device=opacities.device)
                    protection_stats = {
                        "attenuation_protect_count": 0.0,
                        "attenuation_protect_threshold": 0.0,
                        "path_effect_score_mean": 0.0,
                        "path_effect_score_max": 0.0,
                    }
                stats["attenuation_protect_count"] = int(attenuation_protect.sum().item())
                weak_low_local = final_prune_mask & low_contribution
                path_protected_candidates = weak_low_local & attenuation_protect
                stats["weak_low_local_candidate_count"] = int(weak_low_local.sum().item())
                stats["path_protected_candidate_count"] = int(path_protected_candidates.sum().item())
                final_prune_mask = weak_low_local & (~attenuation_protect)
                if str(getattr(rcfg, "protect_attenuation_mode", "high_tau")).lower() in (
                    "path_contribution",
                    "path_aware",
                    "ag",
                ):
                    interval_stats.update(
                        _distribution_stats(
                            "path_protected", path_for_decision[path_protected_candidates]
                        )
                    )
                    interval_stats.update(
                        _distribution_stats("path_pruned", path_for_decision[final_prune_mask])
                    )

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
            "attenuation_protect_count": float(stats["attenuation_protect_count"]),
            "path_protected_candidate_count": float(stats["path_protected_candidate_count"]),
            "weak_low_local_candidate_count": float(stats["weak_low_local_candidate_count"]),
            "echo_alpha_mean": float(container.echo_alphas.mean().detach().item()) if container.num_points > 0 else 0.0,
            "attenuation_mean": float(container.attenuations.mean().detach().item()) if container.num_points > 0 else 0.0,
        }
        self.last_event_stats.update(protection_stats)
        self.last_event_stats.update(resp_stats)
        self.last_event_stats.update(interval_stats)
        if self.resp_contribution.shape[0] > 0:
            self.last_event_stats.update(
                {
                    "responsibility_contribution_mean": float(self.resp_contribution.mean().detach().item()),
                    "responsibility_support_mean": float(self.resp_support.mean().detach().item()),
                    "path_contribution_support_mean": float(self.resp_path_support.mean().detach().item())
                    if self.resp_path_support.shape[0] > 0
                    else 0.0,
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
