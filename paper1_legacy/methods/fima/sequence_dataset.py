"""
FiMA 采样：多帧因果窗 **或** 双帧相邻边 ``(i,i+1)``（``pem.input_mode``）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import torch

from scene.container.single_sweep_container import SingleSweepContainer

from .configs import FiMAConfig
from .pose_geometry import adjacent_frame_transform_gt, relative_pose_from_world, six_dof_from_matrix


@dataclass
class FiMAWindowBatch:
    """单个 batch 的张量包。"""

    images: torch.Tensor  # (B, N, C, H, W)；``adjacent_pair`` 时 N=2，C=1 或 2
    frame_idx_end: torch.Tensor  # (B,) long；相邻模式下为右端全局索引 i+1
    P0: torch.Tensor  # (B, 4, 4)
    #: ``last_frame_rel0`` / 相邻边：均为 (B, 6) 监督向量
    T_gt_vec_last: torch.Tensor  # (B, 6)
    #: window_frames_rel0：(B, N-1, 6)，仅多帧模式
    T_gt_vec_window: Optional[torch.Tensor] = None
    #: window_edges：(B, N-1, 6)，窗口内相邻边 T_{slot-1 -> slot}
    T_gt_vec_edges: Optional[torch.Tensor] = None
    valid_window: torch.Tensor = None  # (B, N) bool

    def __post_init__(self):
        if self.valid_window is None:
            self.valid_window = torch.ones(self.images.shape[0], self.images.shape[1], dtype=torch.bool)


def _load_occ_map(
    occ_dir: str, frame_idx: int, H: int, W: int, device: torch.device
) -> torch.Tensor:
    """``(H,W)`` float32 occupancy(lack)，缺文件或为标量则用 0。"""
    path = os.path.join(occ_dir, f"{int(frame_idx):06d}.npy")
    if not os.path.isfile(path):
        return torch.zeros(H, W, device=device, dtype=torch.float32)
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim == 0:
        return torch.zeros(H, W, device=device, dtype=torch.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D (H,W), got shape {arr.shape}")
    ah, aw = int(arr.shape[0]), int(arr.shape[1])
    if ah != H or aw != W:
        raise ValueError(f"{path}: shape ({ah},{aw}) != meta ({H},{W})")
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _gray_weighted_by_occ(gray_1hw: torch.Tensor, occ_hw: torch.Tensor, pem) -> torch.Tensor:
    """灰度 (1,H,W) 乘 OCC 推导的标量权重场；occ 全 0 时倍率为 1（见公式）。"""
    scale = float(getattr(pem, "occ_weight_scale", 0.6))
    lo = float(getattr(pem, "occ_weight_clamp_lo", 0.35))
    hi = float(getattr(pem, "occ_weight_clamp_hi", 1.35))
    m = 1.0 + scale * occ_hw
    m = torch.clamp(m, lo, hi)
    return gray_1hw * m.unsqueeze(0)


def _build_val_indices(n_frames: int, val_ratio: float, seed: int) -> Set[int]:
    if n_frames <= 1 or val_ratio <= 0:
        return set()
    rng = np.random.RandomState(seed)
    k = max(1, int(round(n_frames * val_ratio)))
    k = min(k, n_frames - 1)
    perm = rng.permutation(n_frames)
    return set(int(x) for x in perm[:k])


def _build_block_holdout_indices(
    n_frames: int,
    num_samples: int,
    block_size: int,
    trim_each: int,
) -> List[int]:
    """Build contiguous held-out blocks on the global one-sweep timeline."""
    if n_frames <= 0:
        raise ValueError("Cannot build block holdout for empty dataset.")
    if num_samples <= 0:
        raise ValueError(f"block_holdout_num_samples must be > 0, got {num_samples}")
    if block_size <= 0:
        raise ValueError(f"block_holdout_block_size must be > 0, got {block_size}")
    if trim_each < 0:
        raise ValueError(f"block_holdout_trim_each must be >= 0, got {trim_each}")

    i_min = int(trim_each)
    i_max = int(n_frames - 1 - trim_each)
    if i_max < i_min:
        raise ValueError(
            f"No valid block holdout interval after trimming: N={n_frames}, trim_each={trim_each}"
        )
    available = i_max - i_min + 1
    if num_samples >= available:
        raise ValueError(
            f"block_holdout_num_samples={num_samples} must be smaller than available={available} "
            f"(N={n_frames}, trim_each={trim_each})"
        )

    block_size = min(int(block_size), int(num_samples))
    n_blocks = int(np.ceil(float(num_samples) / float(block_size)))
    max_start = i_max - block_size + 1
    if max_start < i_min:
        raise ValueError(
            f"block_holdout_block_size={block_size} exceeds available={available} "
            f"(N={n_frames}, trim_each={trim_each})"
        )
    starts = np.rint(np.linspace(i_min, max_start, n_blocks, dtype=np.float64)).astype(np.int64)
    starts = sorted(int(x) for x in np.unique(starts).tolist())

    held: List[int] = []
    seen: Set[int] = set()
    for s in starts:
        for g in range(int(s), min(int(s) + block_size, i_max + 1)):
            if g not in seen:
                held.append(int(g))
                seen.add(int(g))
            if len(held) >= int(num_samples):
                break
        if len(held) >= int(num_samples):
            break
    if len(held) < int(num_samples):
        raise ValueError(
            f"block holdout produced {len(held)} unique frames, need {num_samples}; "
            "use fewer samples or smaller blocks."
        )
    return sorted(held)


class FiMASequenceSampler:
    def __init__(
        self,
        container: SingleSweepContainer,
        cfg: FiMAConfig,
        device: torch.device,
        *,
        train_global_indices: Optional[List[int]] = None,
        test_global_indices: Optional[List[int]] = None,
    ):
        self.container = container
        self.cfg = cfg
        self.device = device
        self.n_frames = len(container)
        self.val_ratio = float(cfg.sequence.val_ratio)
        self.pad_mode = str(cfg.sequence.pad_mode)
        self.input_mode = str(getattr(cfg.pem, "input_mode", "multiframe")).lower()
        self.train_protocol = str(getattr(cfg.pem, "train_protocol", "standard")).lower()
        self.split_protocol = str(getattr(cfg.sequence, "split_protocol", "auto")).lower()

        use_holdout = bool(getattr(cfg.sequence, "respect_scene_holdout", True)) and (
            train_global_indices is not None and test_global_indices is not None
        )
        if use_holdout:
            self.train_indices = sorted(int(i) for i in train_global_indices)
            self.val_indices = set(int(i) for i in test_global_indices)
            self.train_index_set = set(self.train_indices)
            self.test_index_set = set(self.val_indices)
            self._holdout_subseq_train = True
            self._split_description = "scene_holdout"
        elif self.split_protocol == "block_holdout":
            held = _build_block_holdout_indices(
                self.n_frames,
                int(getattr(cfg.sequence, "block_holdout_num_samples", 20)),
                int(getattr(cfg.sequence, "block_holdout_block_size", 5)),
                int(getattr(cfg.sequence, "block_holdout_trim_each", 10)),
            )
            self.val_indices = set(held)
            self.train_indices = [i for i in range(self.n_frames) if i not in self.val_indices]
            self.train_index_set = set(self.train_indices)
            self.test_index_set = set(self.val_indices)
            self._holdout_subseq_train = True
            self._split_description = (
                "block_holdout"
                f"(samples={len(held)}, block_size={int(getattr(cfg.sequence, 'block_holdout_block_size', 5))})"
            )
        else:
            self.val_indices = _build_val_indices(self.n_frames, self.val_ratio, int(cfg.sequence.split_seed))
            self.train_indices = [i for i in range(self.n_frames) if i not in self.val_indices]
            self.train_index_set = set(self.train_indices)
            self.test_index_set = set(self.val_indices)
            self._holdout_subseq_train = False
            self._split_description = "random_frame"

        if self.input_mode == "adjacent_pair":
            self.window = 2
            if self.n_frames < 2:
                raise ValueError("adjacent_pair requires at least 2 frames.")
            self._build_adjacent_structures()
            return

        self.window = int(cfg.pem.window_size)
        if self.n_frames < self.window:
            raise ValueError(
                f"Sequence length {self.n_frames} < pem.window_size={self.window}; "
                "shorten window_size in config."
            )
        if getattr(self, "_holdout_subseq_train", False):
            self._build_multiframe_holdout_windows()

    def _build_adjacent_structures(self) -> None:
        """全局相邻边 ``(k,k+1)`` 两端均在 train 内；以及阶段 A 局部窗（全局连续、全在 train）。"""
        self._global_consecutive_train_edges: List[Tuple[int, int]] = []
        for k in range(self.n_frames - 1):
            if k in self.train_index_set and (k + 1) in self.train_index_set:
                self._global_consecutive_train_edges.append((k, k + 1))
        if not self._global_consecutive_train_edges:
            raise RuntimeError(
                "FiMASequenceSampler(adjacent_pair): no global consecutive train edges; "
                "check holdout/train_indices."
            )
        W = max(2, int(getattr(self.cfg.pem, "phase_a_local_window", 8)))
        S = max(1, int(getattr(self.cfg.pem, "phase_a_local_stride", 4)))
        self._train_global_windows: List[List[int]] = []
        for s in range(0, self.n_frames - W + 1, S):
            if all((s + j) in self.train_index_set for j in range(W)):
                self._train_global_windows.append([s + j for j in range(W)])
        if self.train_protocol == "phase_a_local_window" and not self._train_global_windows:
            raise RuntimeError(
                "FiMASequenceSampler: phase_a_local_window requested but no valid global windows; "
                "try smaller phase_a_local_window / stride or adjust holdout."
            )

    def _build_multiframe_holdout_windows(self) -> None:
        """Build protocol-safe global-contiguous train windows under scene holdout."""
        self._holdout_global_windows: List[List[int]] = []
        for start in range(0, self.n_frames - self.window + 1):
            g_list = [start + j for j in range(self.window)]
            if all(g in self.train_index_set for g in g_list):
                self._holdout_global_windows.append(g_list)
        if not self._holdout_global_windows:
            raise RuntimeError(
                "FiMASequenceSampler: no protocol-safe global-contiguous train windows under holdout; "
                "lower pem.window_size or adjust holdout."
            )

    def _window_indices_fully_in_train_pool(self, frame_end: int) -> bool:
        start_g = frame_end - self.window + 1
        for g in range(max(0, start_g), frame_end + 1):
            if g < 0:
                continue
            if g not in self.train_index_set:
                return False
        return True

    def global_index_from_slot(self, frame_end: int, slot: int) -> int:
        return frame_end - self.window + 1 + slot

    def _build_from_global_list(self, g_list: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(g_list) != self.window:
            raise ValueError(f"g_list length {len(g_list)} != window={self.window}")
        H, W = int(self.container.meta.H), int(self.container.meta.W)
        N = self.window
        imgs = torch.zeros(N, 1, H, W, device=self.device)
        valid = torch.zeros(N, dtype=torch.bool, device=self.device)
        poses = torch.zeros(N, 4, 4, device=self.device)

        P0 = self.container.get(0).pose.to(self.device)

        for slot in range(N):
            g = int(g_list[slot])
            if g < 0:
                if self.pad_mode == "zero":
                    valid[slot] = False
                    imgs[slot, 0] = 0.0
                    poses[slot] = torch.eye(4, device=self.device, dtype=poses.dtype)
                else:
                    sm = self.container.get(0)
                    valid[slot] = False
                    imgs[slot, 0] = sm.image.to(self.device)
                    poses[slot] = sm.pose.to(self.device)
            else:
                sm = self.container.get(g)
                valid[slot] = True
                imgs[slot, 0] = sm.image.to(self.device)
                poses[slot] = sm.pose.to(self.device)

        frame_end = int(g_list[-1])
        sm_end = self.container.get(frame_end)
        P_end = sm_end.pose.to(self.device)
        T_end = relative_pose_from_world(P_end, P0)
        t_gt_last = six_dof_from_matrix(T_end)

        t_win = torch.zeros(N - 1, 6, device=self.device)
        t_edges = torch.zeros(N - 1, 6, device=self.device)
        for k in range(N - 1):
            slot = k + 1
            g = int(g_list[slot])
            if g < 0 or not valid[slot]:
                t_win[k] = 0.0
                t_edges[k] = 0.0
                continue
            Pg = poses[slot]
            Tg = relative_pose_from_world(Pg, P0)
            t_win[k] = six_dof_from_matrix(Tg)
            if valid[slot - 1]:
                T_edge = adjacent_frame_transform_gt(poses[slot - 1], poses[slot])
                t_edges[k] = six_dof_from_matrix(T_edge)

        return imgs, valid, P0, t_gt_last, t_win, t_edges

    def build_window_tensors(self, frame_end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = self.window
        g_list = [frame_end - N + 1 + s for s in range(N)]
        return self._build_from_global_list(g_list)

    def collate_adjacent_edges(self, edges: List[Tuple[int, int]]) -> FiMAWindowBatch:
        """``edges``：全局 ``(i,i+1)`` 相邻帧；监督 ``T = inv(P_i) P_{i+1}`` 的 6D。"""
        if not edges:
            raise ValueError("collate_adjacent_edges: empty edges")
        imgs_l: List[torch.Tensor] = []
        tgt_l: List[torch.Tensor] = []
        ends: List[int] = []
        eye = torch.eye(4, device=self.device, dtype=torch.float32)
        occ_dir = str(getattr(self.cfg.pem, "gs_occ_cache_dir", "") or "").strip()
        if occ_dir:
            occ_dir = os.path.abspath(os.path.expanduser(occ_dir))
        force_dual = bool(getattr(self.cfg.pem, "occ_always_two_channels", False))
        fusion = str(getattr(self.cfg.pem, "occ_fusion_mode", "concat")).lower()
        use_weighted = fusion == "weighted"
        dual_ch = False if use_weighted else (force_dual or bool(occ_dir))

        H, W = int(self.container.meta.H), int(self.container.meta.W)

        for (a, b) in edges:
            if b != a + 1:
                raise ValueError(f"adjacent edge must be global consecutive, got {(a, b)}")
            sm_a = self.container.get(a)
            sm_b = self.container.get(b)
            ia = sm_a.image.to(self.device).unsqueeze(0)
            ib = sm_b.image.to(self.device).unsqueeze(0)
            if use_weighted:
                oa = torch.zeros(H, W, device=self.device, dtype=torch.float32)
                ob = torch.zeros(H, W, device=self.device, dtype=torch.float32)
                if occ_dir:
                    oa = _load_occ_map(occ_dir, a, H, W, self.device)
                    ob = _load_occ_map(occ_dir, b, H, W, self.device)
                ia = _gray_weighted_by_occ(ia, oa, self.cfg.pem)
                ib = _gray_weighted_by_occ(ib, ob, self.cfg.pem)
                pair = torch.stack([ia, ib], dim=0)
            elif dual_ch:
                oa = torch.zeros(1, H, W, device=self.device, dtype=torch.float32)
                ob = torch.zeros(1, H, W, device=self.device, dtype=torch.float32)
                if occ_dir:
                    oa = _load_occ_map(occ_dir, a, H, W, self.device).unsqueeze(0)
                    ob = _load_occ_map(occ_dir, b, H, W, self.device).unsqueeze(0)
                ta = torch.cat([ia, oa], dim=0)
                tb = torch.cat([ib, ob], dim=0)
                pair = torch.stack([ta, tb], dim=0)
            else:
                pair = torch.stack([ia, ib], dim=0)
            Pa = sm_a.pose.to(self.device)
            Pb = sm_b.pose.to(self.device)
            T = adjacent_frame_transform_gt(Pa, Pb)
            imgs_l.append(pair)
            tgt_l.append(six_dof_from_matrix(T))
            ends.append(int(b))

        images = torch.stack(imgs_l, dim=0)
        tgt_b = torch.stack(tgt_l, dim=0)
        B = images.shape[0]
        P0b = eye.unsqueeze(0).expand(B, 4, 4).contiguous()
        ends_t = torch.tensor(ends, device=self.device, dtype=torch.long)
        valid_w = torch.ones(B, 2, dtype=torch.bool, device=self.device)
        return FiMAWindowBatch(
            images=images,
            frame_idx_end=ends_t,
            P0=P0b,
            T_gt_vec_last=tgt_b,
            T_gt_vec_window=None,
            T_gt_vec_edges=None,
            valid_window=valid_w,
        )

    def sample_train_batch(self, batch_size: int, rng: Optional[np.random.RandomState] = None) -> FiMAWindowBatch:
        rng = rng or np.random.RandomState()
        if self.input_mode == "adjacent_pair":
            return self._sample_adjacent_train_batch(batch_size, rng)

        if getattr(self, "_holdout_subseq_train", False):
            windows = getattr(self, "_holdout_global_windows", [])
            if not windows:
                raise RuntimeError(
                    "FiMASequenceSampler: no protocol-safe global-contiguous train windows under holdout."
                )
            picks = [windows[int(rng.randint(len(windows)))] for _ in range(batch_size)]
            return self._collate_global_windows(picks)

        candidates = [i for i in self.train_indices if i >= 1 and i <= self.n_frames - 1]
        min_i = max(1, self.window - 1)
        candidates = [i for i in candidates if i >= min_i and self._window_indices_fully_in_train_pool(i)]
        if not candidates:
            raise RuntimeError("FiMASequenceSampler: no valid train frames for causal window (sequence too short?).")

        ends = [int(rng.choice(candidates)) for _ in range(batch_size)]
        return self._collate_ends(ends)

    def _sample_adjacent_train_batch(self, batch_size: int, rng: np.random.RandomState) -> FiMAWindowBatch:
        proto = self.train_protocol
        if proto == "phase_a_local_window":
            wins = self._train_global_windows
            Bw = max(1, int(getattr(self.cfg.pem, "phase_a_windows_per_batch", 4)))
            picks = [wins[int(rng.randint(len(wins)))] for _ in range(Bw)]
            edges: List[Tuple[int, int]] = []
            for w in picks:
                for j in range(len(w) - 1):
                    edges.append((w[j], w[j + 1]))
            maxp = max(1, int(getattr(self.cfg.pem, "phase_a_max_pair_forward", 56)))
            if len(edges) > maxp:
                perm = rng.permutation(len(edges))
                edges = [edges[int(i)] for i in perm[:maxp]]
            return self.collate_adjacent_edges(edges)

        pool = self._global_consecutive_train_edges
        n = max(1, int(batch_size))
        edges = [pool[int(rng.randint(len(pool)))] for _ in range(n)]
        return self.collate_adjacent_edges(edges)

    def sample_val_batch(self, batch_size: int, rng: Optional[np.random.RandomState] = None) -> FiMAWindowBatch:
        rng = rng or np.random.RandomState()
        if self.input_mode == "adjacent_pair":
            return self._sample_adjacent_train_batch(batch_size, rng)

        candidates = sorted(self.val_indices)
        candidates = [i for i in candidates if i >= max(1, self.window - 1) and i <= self.n_frames - 1]
        if not candidates:
            return self.sample_train_batch(batch_size, rng)

        ends = [int(rng.choice(candidates)) for _ in range(batch_size)]
        return self._collate_ends(ends)

    def all_eval_frames(self, use_val: bool) -> List[int]:
        pool = sorted(self.val_indices) if use_val else [i for i in range(self.n_frames)]
        return [i for i in pool if 1 <= i <= self.n_frames - 1]

    def _collate_global_windows(self, windows: List[List[int]]) -> FiMAWindowBatch:
        imgs_l: List[torch.Tensor] = []
        val_l: List[torch.Tensor] = []
        p0_l: List[torch.Tensor] = []
        last_l: List[torch.Tensor] = []
        win_l: List[torch.Tensor] = []
        edge_l: List[torch.Tensor] = []
        global_ends: List[int] = []

        for g_list in windows:
            if len(g_list) != self.window:
                raise ValueError(f"global window length {len(g_list)} != window={self.window}")
            for a, b in zip(g_list[:-1], g_list[1:]):
                if int(b) != int(a) + 1:
                    raise ValueError(f"global window must be consecutive, got {g_list}")
            imgs, valid, P0, t_last, t_win, t_edges = self._build_from_global_list(g_list)
            global_ends.append(int(g_list[-1]))
            imgs_l.append(imgs)
            val_l.append(valid)
            p0_l.append(P0)
            last_l.append(t_last)
            win_l.append(t_win)
            edge_l.append(t_edges)

        images = torch.stack(imgs_l, dim=0)
        valid_w = torch.stack(val_l, dim=0)
        P0b = torch.stack(p0_l, dim=0)
        last_b = torch.stack(last_l, dim=0)
        win_b = torch.stack(win_l, dim=0)
        edge_b = torch.stack(edge_l, dim=0)
        ends_t = torch.tensor(global_ends, device=self.device, dtype=torch.long)
        return FiMAWindowBatch(
            images=images,
            frame_idx_end=ends_t,
            P0=P0b,
            T_gt_vec_last=last_b,
            T_gt_vec_window=win_b,
            T_gt_vec_edges=edge_b,
            valid_window=valid_w,
        )

    def _collate_train_subsequence_ends(self, ends_local: List[int]) -> FiMAWindowBatch:
        """Legacy helper kept for compatibility; refuses compressed holdout timelines."""
        windows: List[List[int]] = []
        for le in ends_local:
            g_list = self.train_indices[int(le) - self.window + 1 : int(le) + 1]
            windows.append(g_list)
        return self._collate_global_windows(windows)

    def _collate_ends(self, ends: List[int]) -> FiMAWindowBatch:
        imgs_l: List[torch.Tensor] = []
        val_l: List[torch.Tensor] = []
        p0_l: List[torch.Tensor] = []
        last_l: List[torch.Tensor] = []
        win_l: List[torch.Tensor] = []
        edge_l: List[torch.Tensor] = []

        for frame_end in ends:
            imgs, valid, P0, t_last, t_win, t_edges = self.build_window_tensors(frame_end)
            imgs_l.append(imgs)
            val_l.append(valid)
            p0_l.append(P0)
            last_l.append(t_last)
            win_l.append(t_win)
            edge_l.append(t_edges)

        images = torch.stack(imgs_l, dim=0)
        valid_w = torch.stack(val_l, dim=0)
        P0b = torch.stack(p0_l, dim=0)
        last_b = torch.stack(last_l, dim=0)
        win_b = torch.stack(win_l, dim=0)
        edge_b = torch.stack(edge_l, dim=0)
        ends_t = torch.tensor(ends, device=self.device, dtype=torch.long)
        return FiMAWindowBatch(
            images=images,
            frame_idx_end=ends_t,
            P0=P0b,
            T_gt_vec_last=last_b,
            T_gt_vec_window=win_b,
            T_gt_vec_edges=edge_b,
            valid_window=valid_w,
        )

    def target_stats(self, layout: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute train target mean/std in real output units for loss standardization."""
        layout = str(layout)
        values: List[torch.Tensor] = []
        if self.input_mode == "adjacent_pair":
            for edge in self._global_consecutive_train_edges:
                values.append(self.collate_adjacent_edges([edge]).T_gt_vec_last.squeeze(0))
        elif layout == "window_edges":
            if getattr(self, "_holdout_subseq_train", False):
                for w in self._holdout_global_windows:
                    b = self._collate_global_windows([w])
                    values.append(b.T_gt_vec_edges.reshape(-1, 6))
            else:
                for end in self.train_indices:
                    if end < self.window - 1 or not self._window_indices_fully_in_train_pool(int(end)):
                        continue
                    b = self._collate_ends([int(end)])
                    values.append(b.T_gt_vec_edges.reshape(-1, 6))
        elif layout == "window_frames_rel0":
            if getattr(self, "_holdout_subseq_train", False):
                for w in self._holdout_global_windows:
                    b = self._collate_global_windows([w])
                    values.append(b.T_gt_vec_window.reshape(-1, 6))
            else:
                for end in self.train_indices:
                    if end < self.window - 1 or not self._window_indices_fully_in_train_pool(int(end)):
                        continue
                    b = self._collate_ends([int(end)])
                    values.append(b.T_gt_vec_window.reshape(-1, 6))
        else:
            if getattr(self, "_holdout_subseq_train", False):
                for w in self._holdout_global_windows:
                    b = self._collate_global_windows([w])
                    values.append(b.T_gt_vec_last.squeeze(0))
            else:
                for end in self.train_indices:
                    if end < max(1, self.window - 1) or not self._window_indices_fully_in_train_pool(int(end)):
                        continue
                    b = self._collate_ends([int(end)])
                    values.append(b.T_gt_vec_last.squeeze(0))
        if not values:
            raise RuntimeError(f"FiMASequenceSampler.target_stats: no targets for layout={layout}")
        flat = torch.cat([v.reshape(-1, 6) for v in values], dim=0).to(self.device)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(float(self.cfg.pem_loss.target_std_min))
        trans_floor = float(getattr(self.cfg.pem_loss, "target_std_min_trans", 0.0) or 0.0)
        rot_floor = float(getattr(self.cfg.pem_loss, "target_std_min_rot", 0.0) or 0.0)
        if trans_floor > 0.0:
            std[:3] = std[:3].clamp_min(trans_floor)
        if rot_floor > 0.0:
            std[3:] = std[3:].clamp_min(rot_floor)
        return mean, std
