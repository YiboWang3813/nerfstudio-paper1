"""Gaussian State Manager implementations."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


class GaussianStateManager:
    def initialize(self, num_points: int, birth_frame_id: int):
        raise NotImplementedError

    def on_concat(
        self,
        n_new: int,
        frame_id: int,
        parent_indices: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def on_remap(self, keep_mask: torch.Tensor):
        raise NotImplementedError

    def update_membership_from_render(self, frame_id: int, seen_gauss_indices: torch.Tensor):
        return

    def gauss_indices_for_sub(self, i_right: int, W: int) -> Optional[torch.Tensor]:
        return None

    def gauss_indices_for_active_set(
        self,
        i_right: int,
        W: int,
        *,
        means: torch.Tensor,
        scales: torch.Tensor,
        slice_probe_origin: torch.Tensor,
        slice_vec_elevational: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.gauss_indices_for_sub(i_right, W)

    def get_aux_render_indices(self, frame_id: int, num_points: int) -> torch.Tensor:
        return torch.arange(num_points, device=torch.device("cpu"), dtype=torch.long)

    def gauss_indices_for_window(self, l: int, r: int, expand_m: int = 0) -> Optional[torch.Tensor]:
        _ = (l, r, expand_m)
        return None


class MembershipBasedGaussianStateManager(GaussianStateManager):
    def __init__(
        self,
        device: torch.device,
        membership_k: int,
        sub_locality_l: int = 0,
    ):
        self.device = device
        self.membership_k = max(0, int(membership_k))
        self.sub_locality_l = int(sub_locality_l)
        self.membership: Optional[torch.Tensor] = None

    def initialize(self, num_points: int, birth_frame_id: int):
        if self.membership_k <= 0:
            self.membership = None
            return
        mem = torch.full((num_points, self.membership_k), -1, device=self.device, dtype=torch.long)
        if num_points > 0:
            mem[:, 0] = int(birth_frame_id)
        self.membership = mem

    def on_concat(
        self,
        n_new: int,
        frame_id: int,
        parent_indices: Optional[torch.Tensor] = None,
    ):
        if n_new <= 0 or self.membership_k <= 0:
            return
        if self.membership is None:
            raise RuntimeError("GaussianStateManager: membership 未初始化但 membership_k>0")
        add_mem = torch.full((n_new, self.membership_k), -1, device=self.device, dtype=torch.long)
        add_mem[:, 0] = int(frame_id)
        self.membership = torch.cat([self.membership, add_mem], dim=0)

    def on_remap(self, keep_mask: torch.Tensor):
        if self.membership is None:
            return
        self.membership = self.membership[keep_mask]

    def update_membership_from_render(self, frame_id: int, seen_gauss_indices: torch.Tensor):
        if self.membership is None or seen_gauss_indices.numel() == 0:
            return
        idx = seen_gauss_indices.to(dtype=torch.long)
        mem = self.membership
        row = mem[idx]
        empty = row.eq(-1)
        has_empty = empty.any(dim=1)
        if has_empty.any():
            pos = empty.float().argmax(dim=1)
            write_rows = has_empty.nonzero(as_tuple=False).squeeze(-1)
            mem[idx[write_rows], pos[write_rows]] = int(frame_id)
        if (~has_empty).any():
            full_rows = (~has_empty).nonzero(as_tuple=False).squeeze(-1)
            full_idx = idx[full_rows]
            mem[full_idx, :-1] = mem[full_idx, 1:].clone()
            mem[full_idx, -1] = int(frame_id)

    def gauss_indices_for_sub(self, i_right: int, W: int) -> Optional[torch.Tensor]:
        """
        若 sub_locality_l > 0：保留与 [i_right-W+1-l+1, i_right] 时间区间有交的 membership；
        否则返回 None（全量）。
        """
        if self.membership is None:
            return None
        l = self.sub_locality_l
        if l <= 0:
            return None
        lo = i_right - W + 1 - l + 1
        hi = i_right
        m = self.membership
        mask = torch.zeros((m.shape[0],), device=self.device, dtype=torch.bool)
        for k in range(m.shape[1]):
            col = m[:, k]
            ok = (col >= lo) & (col <= hi)
            mask |= ok
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def get_aux_render_indices(self, frame_id: int, num_points: int) -> torch.Tensor:
        """Pose 辅助路径：排除「首槽主归属为 frame_id」的高斯，与 slam §2.2 一致。"""
        if self.membership is None:
            return torch.arange(num_points, device=self.device, dtype=torch.long)
        mask = self.membership[:, 0] != int(frame_id)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)


class SpatialChunkBasedGaussianStateManager(GaussianStateManager):
    def __init__(
        self,
        device: torch.device,
        spatial_chunk_size_mm: float = 1.0,
        spatial_active_half_width_mm: float = 3.0,
        spatial_scale_margin_ratio: float = 1.0,
    ):
        self.device = device
        self.spatial_chunk_size_mm = max(1e-6, float(spatial_chunk_size_mm))
        self.spatial_active_half_width_mm = max(0.0, float(spatial_active_half_width_mm))
        self.spatial_scale_margin_ratio = max(0.0, float(spatial_scale_margin_ratio))
        self._spatial_axis: Optional[torch.Tensor] = None
        self._spatial_origin: Optional[torch.Tensor] = None
        self._proj_on_axis: Optional[torch.Tensor] = None
        self._spatial_bin_ids: Optional[torch.Tensor] = None
        self._bin_to_idx_chunks: Dict[int, torch.Tensor] = {}
        self._indexed_count: int = 0
        self._needs_full_rebuild: bool = True

    def initialize(self, num_points: int, birth_frame_id: int):
        self._invalidate_spatial_index()

    def on_concat(
        self,
        n_new: int,
        frame_id: int,
        parent_indices: Optional[torch.Tensor] = None,
    ):
        return

    def on_remap(self, keep_mask: torch.Tensor):
        self._invalidate_spatial_index()

    def _invalidate_spatial_index(self) -> None:
        self._spatial_axis = None
        self._spatial_origin = None
        self._proj_on_axis = None
        self._spatial_bin_ids = None
        self._bin_to_idx_chunks = {}
        self._indexed_count = 0
        self._needs_full_rebuild = True

    @torch.no_grad()
    def _ensure_spatial_index(
        self,
        means: torch.Tensor,
        slice_probe_origin: torch.Tensor,
        slice_vec_elevational: torch.Tensor,
    ) -> None:
        axis = slice_vec_elevational.to(device=self.device, dtype=means.dtype)
        axis = axis / (axis.norm() + 1e-12)
        if self._spatial_axis is None or self._spatial_origin is None:
            self._spatial_axis = axis.detach().clone()
            self._spatial_origin = slice_probe_origin.to(device=self.device, dtype=means.dtype).detach().clone()
            self._needs_full_rebuild = True
        axis_changed = (self._spatial_axis - axis).abs().max() > 1e-5
        if axis_changed:
            self._spatial_axis = axis.detach().clone()
            self._spatial_origin = slice_probe_origin.to(device=self.device, dtype=means.dtype).detach().clone()
            self._needs_full_rebuild = True
        need_full = (
            self._needs_full_rebuild
            or
            self._proj_on_axis is None
            or self._spatial_bin_ids is None
            or self._indexed_count > means.shape[0]
        )
        origin = self._spatial_origin
        axis_ref = self._spatial_axis
        if need_full:
            proj = (means - origin.unsqueeze(0)).mul(axis_ref.unsqueeze(0)).sum(dim=1)
            bin_ids = torch.floor(proj / self.spatial_chunk_size_mm).to(dtype=torch.long)
            self._proj_on_axis = proj
            self._spatial_bin_ids = bin_ids
            self._bin_to_idx_chunks = {}
            uniq = torch.unique(bin_ids)
            for b in uniq.tolist():
                idx = torch.nonzero(bin_ids == int(b), as_tuple=False).squeeze(-1)
                self._bin_to_idx_chunks[int(b)] = idx
            self._indexed_count = int(means.shape[0])
            self._needs_full_rebuild = False
            return
        if means.shape[0] == self._indexed_count:
            return
        # 增量追加：仅为新增行计算投影和分桶。
        start = int(self._indexed_count)
        end = int(means.shape[0])
        new_means = means[start:end]
        if new_means.numel() == 0:
            self._indexed_count = end
            return
        proj_new = (new_means - origin.unsqueeze(0)).mul(axis_ref.unsqueeze(0)).sum(dim=1)
        bin_new = torch.floor(proj_new / self.spatial_chunk_size_mm).to(dtype=torch.long)
        self._proj_on_axis = torch.cat([self._proj_on_axis, proj_new], dim=0)
        self._spatial_bin_ids = torch.cat([self._spatial_bin_ids, bin_new], dim=0)
        global_idx = torch.arange(start, end, device=self.device, dtype=torch.long)
        for b in torch.unique(bin_new).tolist():
            mask = bin_new == int(b)
            add_idx = global_idx[mask]
            if int(b) in self._bin_to_idx_chunks:
                self._bin_to_idx_chunks[int(b)] = torch.cat([self._bin_to_idx_chunks[int(b)], add_idx], dim=0)
            else:
                self._bin_to_idx_chunks[int(b)] = add_idx
        self._indexed_count = end

    @torch.no_grad()
    def _gauss_indices_for_spatial_chunk(
        self,
        *,
        means: torch.Tensor,
        scales: torch.Tensor,
        slice_probe_origin: torch.Tensor,
        slice_vec_elevational: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if means.numel() == 0:
            return None
        self._ensure_spatial_index(
            means=means,
            slice_probe_origin=slice_probe_origin,
            slice_vec_elevational=slice_vec_elevational,
        )
        assert self._spatial_axis is not None
        assert self._spatial_origin is not None
        assert self._proj_on_axis is not None
        s_curr = ((slice_probe_origin - self._spatial_origin) * self._spatial_axis).sum()
        lo = s_curr - self.spatial_active_half_width_mm
        hi = s_curr + self.spatial_active_half_width_mm
        b_lo = int(torch.floor(lo / self.spatial_chunk_size_mm).item())
        b_hi = int(torch.floor(hi / self.spatial_chunk_size_mm).item())
        picks: List[torch.Tensor] = []
        for b in range(b_lo, b_hi + 1):
            idx = self._bin_to_idx_chunks.get(b)
            if idx is not None and idx.numel() > 0:
                picks.append(idx)
        if not picks:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        candidate = torch.unique(torch.cat(picks, dim=0))
        proj = self._proj_on_axis.index_select(0, candidate)
        if scales.ndim == 1:
            scale_r = scales.index_select(0, candidate).abs()
        else:
            scale_r = scales.index_select(0, candidate).abs().amax(dim=1)
        margin = scale_r * self.spatial_scale_margin_ratio
        keep = (proj + margin >= lo) & (proj - margin <= hi)
        return candidate[keep]

    def gauss_indices_for_active_set(
        self,
        i_right: int,
        W: int,
        *,
        means: torch.Tensor,
        scales: torch.Tensor,
        slice_probe_origin: torch.Tensor,
        slice_vec_elevational: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self._gauss_indices_for_spatial_chunk(
            means=means,
            scales=scales,
            slice_probe_origin=slice_probe_origin,
            slice_vec_elevational=slice_vec_elevational,
        )

    def get_aux_render_indices(self, frame_id: int, num_points: int) -> torch.Tensor:
        return torch.arange(num_points, device=self.device, dtype=torch.long)


class SliceIdentityBasedGaussianStateManager(GaussianStateManager):
    """仅维护高斯所属 slice id，并用帧 id 距离做快速子集筛选。"""

    def __init__(self, device: torch.device, exclude_num_ids: int = 15):
        self.device = device
        self.exclude_num_ids = int(exclude_num_ids)
        self.slice_ids: Optional[torch.Tensor] = None
        self._cache_i_right: Optional[int] = None
        self._cache_exclude: Optional[int] = None
        self._cache_indices: Optional[torch.Tensor] = None

    def _invalidate_cache(self) -> None:
        self._cache_i_right = None
        self._cache_exclude = None
        self._cache_indices = None

    def initialize(self, num_points: int, birth_frame_id: int):
        self.slice_ids = torch.full(
            (int(num_points),),
            int(birth_frame_id),
            device=self.device,
            dtype=torch.long,
        )
        self._invalidate_cache()

    def on_concat(
        self,
        n_new: int,
        frame_id: int,
        parent_indices: Optional[torch.Tensor] = None,
    ):
        if n_new <= 0:
            return
        if self.slice_ids is None:
            self.slice_ids = torch.empty((0,), device=self.device, dtype=torch.long)
        if parent_indices is not None and parent_indices.numel() > 0:
            parent_indices = parent_indices.to(device=self.device, dtype=torch.long).view(-1)
            add_ids = self.slice_ids.index_select(0, parent_indices)
        else:
            add_ids = torch.full((int(n_new),), int(frame_id), device=self.device, dtype=torch.long)
        self.slice_ids = torch.cat([self.slice_ids, add_ids], dim=0)
        self._invalidate_cache()

    def on_remap(self, keep_mask: torch.Tensor):
        if self.slice_ids is None:
            return
        self.slice_ids = self.slice_ids[keep_mask]
        self._invalidate_cache()

    def gauss_indices_for_sub(self, i_right: int, W: int) -> Optional[torch.Tensor]:
        _ = W
        if self.slice_ids is None:
            return None
        ex = int(self.exclude_num_ids)
        if ex <= 0:
            return None
        if (
            self._cache_indices is not None
            and self._cache_i_right == int(i_right)
            and self._cache_exclude == ex
        ):
            return self._cache_indices
        diff = (self.slice_ids - int(i_right)).abs()
        keep = diff < ex
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        self._cache_i_right = int(i_right)
        self._cache_exclude = ex
        self._cache_indices = idx
        return idx

    def get_aux_render_indices(self, frame_id: int, num_points: int) -> torch.Tensor:
        if self.slice_ids is None:
            return torch.arange(num_points, device=self.device, dtype=torch.long)
        keep = self.slice_ids != int(frame_id)
        return torch.nonzero(keep, as_tuple=False).squeeze(-1)


class SliceIdentityBasedNoExcludeGaussianStateManager(GaussianStateManager):
    """仅维护出生 slice id，不做排除逻辑；支持窗口范围筛选。"""

    def __init__(self, device: torch.device):
        self.device = device
        self.slice_ids: Optional[torch.Tensor] = None
        self._cache_l: Optional[int] = None
        self._cache_r: Optional[int] = None
        self._cache_m: Optional[int] = None
        self._cache_indices: Optional[torch.Tensor] = None

    def _invalidate_cache(self) -> None:
        self._cache_l = None
        self._cache_r = None
        self._cache_m = None
        self._cache_indices = None

    def initialize(self, num_points: int, birth_frame_id: int):
        self.slice_ids = torch.full(
            (int(num_points),),
            int(birth_frame_id),
            device=self.device,
            dtype=torch.long,
        )
        self._invalidate_cache()

    def on_concat(
        self,
        n_new: int,
        frame_id: int,
        parent_indices: Optional[torch.Tensor] = None,
    ):
        if n_new <= 0:
            return
        if self.slice_ids is None:
            self.slice_ids = torch.empty((0,), device=self.device, dtype=torch.long)
        if parent_indices is not None and parent_indices.numel() > 0:
            parent_indices = parent_indices.to(device=self.device, dtype=torch.long).view(-1)
            add_ids = self.slice_ids.index_select(0, parent_indices)
        else:
            add_ids = torch.full((int(n_new),), int(frame_id), device=self.device, dtype=torch.long)
        self.slice_ids = torch.cat([self.slice_ids, add_ids], dim=0)
        self._invalidate_cache()

    def on_remap(self, keep_mask: torch.Tensor):
        if self.slice_ids is None:
            return
        self.slice_ids = self.slice_ids[keep_mask]
        self._invalidate_cache()

    def gauss_indices_for_window(self, l: int, r: int, expand_m: int = 0) -> Optional[torch.Tensor]:
        if self.slice_ids is None:
            return None
        ll = int(l) - max(0, int(expand_m))
        rr = int(r) + max(0, int(expand_m))
        if (
            self._cache_indices is not None
            and self._cache_l == ll
            and self._cache_r == rr
            and self._cache_m == int(expand_m)
        ):
            return self._cache_indices
        keep = (self.slice_ids >= ll) & (self.slice_ids <= rr)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        self._cache_l = ll
        self._cache_r = rr
        self._cache_m = int(expand_m)
        self._cache_indices = idx
        return idx

    def get_aux_render_indices(self, frame_id: int, num_points: int) -> torch.Tensor:
        return torch.arange(num_points, device=self.device, dtype=torch.long)


def build_gaussian_state_manager(
    *,
    mode: str,
    device: torch.device,
    membership_k: int,
    sub_locality_l: int,
    slice_identity_exclude_num_ids: int,
    spatial_chunk_size_mm: float,
    spatial_active_half_width_mm: float,
    spatial_scale_margin_ratio: float,
) -> GaussianStateManager:
    m = str(mode).lower()
    if m == "membership":
        return MembershipBasedGaussianStateManager(
            device=device,
            membership_k=membership_k,
            sub_locality_l=sub_locality_l,
        )
    if m == "spatial_chunk":
        return SpatialChunkBasedGaussianStateManager(
            device=device,
            spatial_chunk_size_mm=spatial_chunk_size_mm,
            spatial_active_half_width_mm=spatial_active_half_width_mm,
            spatial_scale_margin_ratio=spatial_scale_margin_ratio,
        )
    if m == "slice_identity":
        return SliceIdentityBasedGaussianStateManager(
            device=device,
            exclude_num_ids=slice_identity_exclude_num_ids,
        )
    if m == "slice_identity_no_exclude":
        return SliceIdentityBasedNoExcludeGaussianStateManager(device=device)
    return MembershipBasedGaussianStateManager(
        device=device,
        membership_k=0,
        sub_locality_l=0,
    )
