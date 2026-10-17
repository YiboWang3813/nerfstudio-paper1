import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Any, List

from configs.root import RootConfig
from utils.math_utils import sigmoid_inv
from .config import ModelConfig
from .gauss_container import UltraGSGaussContainer
from .gauss_solver import UltraGSGaussSolver


def _normalized_active_indices(
    active_global_indices: Optional[torch.Tensor], n_pts: int
) -> Optional[torch.Tensor]:
    """与 render 子集一致的全局行号；非法或空则返回 None（走全量 DC 路径）。"""
    if active_global_indices is None or active_global_indices.numel() == 0 or n_pts <= 0:
        return None
    ia = torch.unique(active_global_indices.long().view(-1))
    ia = ia[(ia >= 0) & (ia < n_pts)]
    if ia.numel() == 0:
        return None
    return ia


def _raw_param_1d(container: UltraGSGaussContainer, name: str) -> torch.Tensor:
    t = container._param_dict[name].tensor
    return t.view(t.shape[0], -1).squeeze(-1)


def _raw_param_2d(container: UltraGSGaussContainer, name: str) -> torch.Tensor:
    return container._param_dict[name].tensor


@torch.no_grad()
def _collect_merge_pair_candidates(
    means: torch.Tensor,
    dist_thresh_mm: float,
    max_pairs: int,
) -> torch.Tensor:
    """
    体素同格 + 三轴排序相邻 并集，距离过滤后按距离升序截断到 max_pairs。
    返回 (M,2) long，列 (i,j) 满足 i<j。
    """
    device = means.device
    n = int(means.shape[0])
    if n < 2:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    dthr = float(dist_thresh_mm)
    cell = max(dthr, 1e-6)
    chunks: List[torch.Tensor] = []

    g = torch.floor(means / cell).long()
    mn = g.min(dim=0).values
    g0 = g - mn
    nx = int(g0[:, 0].max().item()) + 2
    ny = int(g0[:, 1].max().item()) + 2
    hid = g0[:, 0] + g0[:, 1] * nx + g0[:, 2] * nx * ny
    order = torch.argsort(hid)
    hid_s = hid[order]
    chg = torch.cat(
        [torch.ones(1, dtype=torch.bool, device=device), hid_s[1:] != hid_s[:-1]],
        dim=0,
    )
    starts = torch.where(chg)[0]
    ends = torch.cat([starts[1:], torch.tensor([hid_s.numel()], device=device, dtype=torch.long)])
    for s, e in zip(starts.tolist(), ends.tolist()):
        seg_len = e - s
        if seg_len < 2:
            continue
        idx = order[s:e]
        if seg_len > 256:
            perm = torch.randperm(seg_len, device=device)[:256]
            idx = idx[perm]
        L = int(idx.numel())
        if L < 2:
            continue
        ii = idx.unsqueeze(1).expand(L, L)
        jj = idx.unsqueeze(0).expand(L, L)
        tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        pa, pb = ii[tri], jj[tri]
        dist = torch.norm(means[pa] - means[pb], dim=-1)
        keep = dist <= dthr
        if keep.any():
            lo = torch.minimum(pa[keep], pb[keep])
            hi = torch.maximum(pa[keep], pb[keep])
            chunks.append(torch.stack([lo, hi], dim=-1))

    for ax in range(3):
        ord_ax = torch.argsort(means[:, ax])
        a = ord_ax[:-1]
        b = ord_ax[1:]
        dist = torch.norm(means[a] - means[b], dim=-1)
        keep = dist <= dthr
        if keep.any():
            lo = torch.minimum(a[keep], b[keep])
            hi = torch.maximum(a[keep], b[keep])
            chunks.append(torch.stack([lo, hi], dim=-1))

    if not chunks:
        return torch.empty((0, 2), device=device, dtype=torch.long)

    cat = torch.cat(chunks, dim=0)
    cat = torch.unique(cat, dim=0)
    d = torch.norm(means[cat[:, 0]] - means[cat[:, 1]], dim=-1)
    k = min(int(max_pairs), int(cat.shape[0]))
    if k <= 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    if cat.shape[0] > k:
        _, sub = torch.topk(d, k, largest=False)
        cat = cat[sub]
        d = d[sub]
    perm = torch.argsort(d)
    return cat[perm]


@torch.no_grad()
def _fuse_loser_into_winner(
    container: UltraGSGaussContainer,
    w: int,
    l: int,
    eps: float = 1e-8,
) -> None:
    """在 inverse/raw 空间把 loser 的统计量融合进 winner（就地改 Parameter.data）。"""
    means_r = _raw_param_2d(container, "means")
    scales_r = _raw_param_2d(container, "scales")
    quats_r = _raw_param_2d(container, "quats")
    opac_r = _raw_param_1d(container, "opacities")
    inte_r = _raw_param_1d(container, "intensities")

    ow = torch.sigmoid(opac_r[w].clamp(-20.0, 20.0))
    ol = torch.sigmoid(opac_r[l].clamp(-20.0, 20.0))
    iw = torch.sigmoid(inte_r[w].clamp(-20.0, 20.0))
    il = torch.sigmoid(inte_r[l].clamp(-20.0, 20.0))
    mw = (ow * iw).clamp_min(eps)
    ml = (ol * il).clamp_min(eps)
    den_m = mw + ml

    means_r[w] = (means_r[w] * mw + means_r[l] * ml) / den_m

    sw = torch.exp(scales_r[w])
    sl = torch.exp(scales_r[l])
    s_new = torch.sqrt(sw * sw + sl * sl + eps)
    scales_r[w] = torch.log(s_new)

    o_new = (1.0 - (1.0 - ow) * (1.0 - ol)).clamp(eps, 1.0 - eps)
    opac_r[w] = sigmoid_inv(o_new)

    i_new = ((iw * mw + il * ml) / den_m).clamp(eps, 1.0 - eps)
    inte_r[w] = sigmoid_inv(i_new)

    qw = F.normalize(quats_r[w], dim=-1)
    ql = F.normalize(quats_r[l], dim=-1)
    if float((qw * ql).sum().item()) < 0.0:
        ql = -ql
    q_sum = qw + ql
    quats_r[w] = F.normalize(q_sum, dim=-1)


@torch.no_grad()
def online_pair_merge_and_fuse(
    cfg: RootConfig,
    solver: UltraGSGaussSolver,
    container: UltraGSGaussContainer,
    gsm: Optional[Any],
    max_pairs: int,
) -> int:
    """
    Online 近邻合并：候选对 → 按距离贪心 → 质量高者吸收弱者（参数融合）→ 单次 prune。
    返回删除的高斯个数。
    """
    if str(getattr(cfg.engine, "stream", "") or "").lower() != "online":
        return 0
    odc = cfg.model.online.density_control
    dthr = float(odc.merge_dist_thresh_mm)
    op_thr = float(odc.merge_opacity_thresh)
    n = int(container.num_points)
    if n < 2:
        return 0

    means = container.means.detach()
    pairs = _collect_merge_pair_candidates(means, dthr, max_pairs)
    if pairs.numel() == 0:
        return 0

    device = container.device
    removed = torch.zeros((n,), device=device, dtype=torch.bool)

    for k in range(int(pairs.shape[0])):
        i = int(pairs[k, 0].item())
        j = int(pairs[k, 1].item())
        if removed[i] or removed[j]:
            continue
        oi = float(container.opacities[i].squeeze().item())
        oj = float(container.opacities[j].squeeze().item())
        if max(oi, oj) < op_thr:
            continue
        score_i = oi * float(container.intensities[i].squeeze().item())
        score_j = oj * float(container.intensities[j].squeeze().item())
        if score_i >= score_j:
            w, l = i, j
        else:
            w, l = j, i
        _fuse_loser_into_winner(container, w, l)
        removed[l] = True

    n_removed = int(removed.sum().item())
    if n_removed <= 0:
        return 0
    keep = ~removed
    solver.prune(removed)
    if gsm is not None:
        gsm.on_remap(keep)
    return n_removed


class UltraGSOfflineDensityController:
    """
    Density Controller (Strategy Layer)

    职责:
    1. 观测: 累积梯度 (update_grad_stats)
    2. 决策: 计算 Clone/Split/Prune 的 Mask
    3. 指挥: 调用 Solver 执行拓扑变更
    """

    @property
    def model_cfg(self) -> ModelConfig:
        return self.cfg.model

    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        self.thresh = self.model_cfg.offline.density_control.thresholds
        self.switches = self.model_cfg.offline.density_control.enable
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        print(f"[DensityController] Init done. Densify={self.switches.densify}, Prune={self.switches.prune}")

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

    def update_grad_stats(
        self,
        container: UltraGSGaussContainer,
        active_global_indices: Optional[torch.Tensor] = None,
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
        if active_global_indices is not None and active_global_indices.numel() > 0:
            idx = torch.unique(active_global_indices.long().view(-1))
            gn = torch.norm(means_param.grad.index_select(0, idx), dim=-1, keepdim=True)
            self.xyz_grad_accum.index_add_(0, idx, gn)
            self.denom.index_add_(
                0,
                idx,
                torch.ones((idx.numel(), 1), device=container.device, dtype=self.denom.dtype),
            )
        else:
            grad_norm = torch.norm(means_param.grad, dim=-1, keepdim=True)
            self.xyz_grad_accum += grad_norm
            self.denom += 1

    def prune_and_densify(
        self,
        iteration: int,
        solver: UltraGSGaussSolver,
        container: UltraGSGaussContainer,
        gsm: Optional[Any] = None,
        birth_frame_id: Optional[int] = None,
        active_global_indices: Optional[torch.Tensor] = None,
    ):
        """
        gsm / birth_frame_id：仅 Online 传入。
        active_global_indices：若提供且非空，则 densify 的梯度/尺度判断与弱 prune 的阈值判断
        均在活跃子集上完成（O(M)），再 scatter 成整锅 mask 调 solver；否则为全量 O(N) 路径（Offline）。
        Online 且 ``merge_after_dc``：在 prune 之后执行近邻融合合并（见 ``online_pair_merge_and_fuse``）。
        """
        _stream = str(getattr(self.cfg.engine, "stream", "") or "").lower()
        _odc = getattr(self.model_cfg, "online", None)
        _odc = getattr(_odc, "density_control", None) if _odc is not None else None
        _merge_after = bool(getattr(_odc, "merge_after_dc", False)) if _odc is not None else False

        if not self.switches.densify and not self.switches.prune:
            if not (_stream == "online" and _merge_after):
                return

        n_pts = int(container.num_points)
        idx_active = _normalized_active_indices(active_global_indices, n_pts)

        grads_full: Optional[torch.Tensor] = None
        if self.switches.densify or self.switches.prune:
            if idx_active is not None:
                den = self.denom.index_select(0, idx_active).clamp(min=1)
                g_sel = (self.xyz_grad_accum.index_select(0, idx_active) / den).squeeze(-1)
                g_sel = torch.nan_to_num(g_sel, nan=0.0)
                max_grad = float(g_sel.max().item())
                mean_grad = float(g_sel.mean().item())
            else:
                grads_full = self.xyz_grad_accum / self.denom.clamp(min=1)
                grads_full[grads_full.isnan()] = 0.0
                max_grad = float(grads_full.max().item())
                mean_grad = float(grads_full.mean().item())
                grads_full = grads_full.squeeze(-1)

            print(
                f"[DC Check] Iter {iteration} | Max Grad: {max_grad:.2e} | Mean Grad: {mean_grad:.2e} | "
                f"Thresh: {self.thresh.densify_grad}"
                + (" | active_sparse_dc" if idx_active is not None else "")
            )
        else:
            print(f"[DC Check] Iter {iteration} | merge_only (no densify/prune this pass)")

        scales = container.scales
        intensities = container.intensities
        opacities = container.opacities
        stats = {"clone": 0, "split": 0, "prune_weak": 0, "prune_big": 0, "merge_removed": 0}

        # --- 1. Densify ---
        if self.switches.densify:
            min_scale_limit = 0.01
            if idx_active is not None:
                den = self.denom.index_select(0, idx_active).clamp(min=1)
                g_sel = (self.xyz_grad_accum.index_select(0, idx_active) / den).squeeze(-1)
                g_sel = torch.nan_to_num(g_sel, nan=0.0)
                high = g_sel > self.thresh.densify_grad
                max_scale_sub = torch.max(scales.index_select(0, idx_active), dim=1).values
                split_sel = (high & (max_scale_sub > self.thresh.split_scale)).reshape(-1)
                clone_sel = (
                    high & (max_scale_sub <= self.thresh.split_scale) & (max_scale_sub > min_scale_limit)
                ).reshape(-1)
                clone_mask = torch.zeros((n_pts,), device=scales.device, dtype=torch.bool)
                split_mask = torch.zeros((n_pts,), device=scales.device, dtype=torch.bool)
                clone_mask.index_copy_(0, idx_active, clone_sel)
                split_mask.index_copy_(0, idx_active, split_sel)
            else:
                high_grad_mask = grads_full > self.thresh.densify_grad
                max_scale = torch.max(scales, dim=1).values
                split_mask = high_grad_mask & (max_scale > self.thresh.split_scale)
                clone_mask = high_grad_mask & (max_scale <= self.thresh.split_scale) & (max_scale > min_scale_limit)

            new_tensors, n_clone, n_split, parent_indices = self._generate_new_points(
                container, clone_mask, split_mask
            )
            if n_clone > 0 or n_split > 0:
                solver.concat(new_tensors)
                stats["clone"] = n_clone
                stats["split"] = n_split
                n_add = int(new_tensors["means"].shape[0])
                if gsm is not None and n_add > 0:
                    frame_id_for_add = int(birth_frame_id) if birth_frame_id is not None else 0
                    gsm.on_concat(
                        n_add,
                        frame_id_for_add,
                        parent_indices=parent_indices,
                    )
                self.xyz_grad_accum = torch.empty(0)
                self.denom = torch.empty(0)
                scales = container.scales
                intensities = container.intensities
                opacities = container.opacities
                n_pts = int(container.num_points)

        # --- 2. Prune ---
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

            stats["prune_weak"] = int(final_prune_mask.sum().item())
            if final_prune_mask.sum() > 0:
                keep_mask = ~final_prune_mask
                solver.prune(final_prune_mask)
                if gsm is not None:
                    gsm.on_remap(keep_mask)
                self.xyz_grad_accum = torch.empty(0)
                self.denom = torch.empty(0)

        # --- 3. Online：DC 步末近邻融合合并（与帧末 ``run_periodic_pair_merge`` 同一套核心）---
        if _stream == "online" and _merge_after:
            max_pairs = int(getattr(_odc, "merge_max_pairs", 100_000)) if _odc is not None else 100_000
            mr = online_pair_merge_and_fuse(self.cfg, solver, container, gsm, max_pairs)
            stats["merge_removed"] = mr
            if mr > 0:
                self.xyz_grad_accum = torch.empty(0)
                self.denom = torch.empty(0)

        if sum(stats.values()) > 0:
            extra = f" | MergeRemoved: {stats['merge_removed']}" if stats.get("merge_removed", 0) > 0 else ""
            print(
                f"[DC] Iter {iteration} | Split: {stats['split']} Clone: {stats['clone']} | "
                f"Prune: {stats['prune_weak']}(Weak){extra}"
            )

    def _generate_new_points(
        self,
        container: UltraGSGaussContainer,
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


class UltraGSOnlineDensityController(UltraGSOfflineDensityController):
    """Online：全局 GS 步调度；逻辑与 Offline 共用 prune_and_densify。"""

    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        odc = self.model_cfg.online.density_control
        self.thresh = odc.thresholds
        self.switches = odc.enable
        self._online_timing = odc.timing
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        print(
            f"[OnlineDensityController] Init Densify={self.switches.densify}, Prune={self.switches.prune}, "
            f"merge_after_dc={getattr(odc, 'merge_after_dc', False)} | "
            f"start_gs={self._online_timing.start_gs_step}, end_gs={self._online_timing.end_gs_step}, "
            f"freq_gs={self._online_timing.freq_gs_steps} | "
            f"merge_every_frames={int(self.model_cfg.online.density_control.merge_every_frames)}"
        )

    def run_periodic_pair_merge(
        self,
        solver: UltraGSGaussSolver,
        container: UltraGSGaussContainer,
        gsm: Optional[Any],
        birth_frame_id: int,
    ) -> int:
        """帧末 atom_merge：与 DC 末尾合并共用 ``online_pair_merge_and_fuse``。"""
        _ = birth_frame_id
        max_pairs = int(self.model_cfg.online.density_control.merge_max_pairs)
        return online_pair_merge_and_fuse(self.cfg, solver, container, gsm, max_pairs)

    def should_accumulate_grad_stats(self, global_gs_step: int) -> bool:
        if not (self.switches.densify or self.switches.prune):
            return False
        t = self._online_timing
        if global_gs_step < int(t.start_gs_step):
            return False
        end = int(t.end_gs_step)
        if end > 0 and global_gs_step > end:
            return False
        return True

    def should_densify_prune(self, global_gs_step: int) -> bool:
        odc = self.model_cfg.online.density_control
        if not (self.switches.densify or self.switches.prune) and not getattr(odc, "merge_after_dc", False):
            return False
        t = self._online_timing
        freq = int(t.freq_gs_steps)
        if freq <= 0:
            return False
        if global_gs_step < int(t.start_gs_step):
            return False
        end = int(t.end_gs_step)
        if end > 0 and global_gs_step > end:
            return False
        return global_gs_step % freq == 0
