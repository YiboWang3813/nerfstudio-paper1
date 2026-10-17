"""从已训练的 PEM 导出链式位姿 (N,4,4)。"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .callbacks import (
    _infer_global_edge_predictions,
    _infer_multiframe_window_edge_predictions,
    _merge_phase_a_window_edge_predictions,
)
from .pose_geometry import se3_from_6dof


def _compose_chain_from_edges(
    edge_pred: Dict[int, torch.Tensor],
    n_frames: int,
    device: torch.device,
    *,
    initial_pose: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    eye = torch.eye(4, device=device, dtype=torch.float32)
    anchor = eye if initial_pose is None else initial_pose.to(device=device, dtype=torch.float32)
    if anchor.shape != (4, 4):
        raise ValueError(f"initial_pose must be (4,4), got {tuple(anchor.shape)}")
    out = torch.empty((n_frames, 4, 4), device=device, dtype=torch.float32)
    out[0] = anchor
    for i in range(1, n_frames):
        Tk = edge_pred.get(i - 1)
        if Tk is None:
            raise KeyError(f"missing edge prediction for ({i-1}->{i})")
        out[i] = out[i - 1] @ Tk.to(device=device, dtype=out.dtype)
    return out


def _project_rotation(R: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    det = (
        R_proj[0, 0] * (R_proj[1, 1] * R_proj[2, 2] - R_proj[1, 2] * R_proj[2, 1])
        - R_proj[0, 1] * (R_proj[1, 0] * R_proj[2, 2] - R_proj[1, 2] * R_proj[2, 0])
        + R_proj[0, 2] * (R_proj[1, 0] * R_proj[2, 1] - R_proj[1, 1] * R_proj[2, 0])
    )
    if bool(det < 0):
        U = U.clone()
        U[:, -1] *= -1.0
        R_proj = U @ Vh
    return R_proj


def _weighted_pose_average(poses: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if len(poses) != len(weights) or not poses:
        raise ValueError("weighted pose average requires the same nonzero number of poses and weights")
    device = poses[0].device
    dtype = poses[0].dtype
    w = torch.tensor(weights, device=device, dtype=dtype)
    w = w / torch.clamp(w.sum(), min=torch.finfo(dtype).eps)
    out = torch.eye(4, device=device, dtype=dtype)
    t = torch.zeros(3, device=device, dtype=dtype)
    R_acc = torch.zeros(3, 3, device=device, dtype=dtype)
    for wi, Pi in zip(w, poses):
        t = t + wi * Pi[:3, 3]
        R_acc = R_acc + wi * Pi[:3, :3]
    out[:3, :3] = _project_rotation(R_acc)
    out[:3, 3] = t
    return out


def _local_slot_weight(slot: int, window: int) -> float:
    center = 0.5 * float(window - 1)
    dist = abs(float(slot) - center)
    return 1.0 / (1.0 + dist)


def _compose_local_window_poses(pred_6d: torch.Tensor) -> torch.Tensor:
    """Compose one window's predicted adjacent edges into poses relative to slot 0."""
    window = int(pred_6d.shape[0]) + 1
    device = pred_6d.device
    out = torch.empty((window, 4, 4), device=device, dtype=torch.float32)
    out[0] = torch.eye(4, device=device, dtype=torch.float32)
    for slot in range(window - 1):
        out[slot + 1] = out[slot] @ se3_from_6dof(pred_6d[slot].to(dtype=torch.float32))
    return out


def _window_start_indices(n_frames: int, window: int, stride: int) -> list[int]:
    if n_frames < window:
        raise ValueError(f"n_frames={n_frames} < window={window}")
    stride = max(1, int(stride))
    starts = list(range(0, n_frames - window + 1, stride))
    last = n_frames - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _export_local_stitched_window_poses_probe_world(state) -> torch.Tensor:
    """
    Export a full trajectory by composing short overlapping window trajectories
    and aligning each new window to the already fused overlap frames.

    This intentionally avoids a single hundreds-frame adjacent-edge chain. It is
    a diagnostic export mode for checking whether long-chain accumulation is the
    dominant FiMA failure mode.
    """
    sampler = state.sequence_sampler
    pem = state.cfg.model.pem
    n = int(sampler.n_frames)
    window = int(getattr(sampler, "window", getattr(pem, "window_size", 2)))
    stride = int(getattr(pem, "infer_stitch_stride", max(1, window // 4)))
    starts = _window_start_indices(n, window, stride)
    net = state.pose_estimator
    device = state.device

    candidates: list[list[torch.Tensor]] = [[] for _ in range(n)]
    weights: list[list[float]] = [[] for _ in range(n)]
    fused: list[torch.Tensor | None] = [None for _ in range(n)]
    anchor = _export_initial_pose_anchor(state)

    with torch.no_grad():
        for start in starts:
            frame_ids = [start + j for j in range(window)]
            batch = sampler._collate_global_windows([frame_ids])
            out = net(batch.images, batch.valid_window)
            pred = out["pred_pose_6"].squeeze(0)
            local = _compose_local_window_poses(pred)

            overlap_alignments: list[torch.Tensor] = []
            overlap_weights: list[float] = []
            for slot, gidx in enumerate(frame_ids):
                if fused[gidx] is None:
                    continue
                overlap_alignments.append(fused[gidx] @ torch.linalg.inv(local[slot]))
                overlap_weights.append(_local_slot_weight(slot, window))

            if overlap_alignments:
                align = _weighted_pose_average(overlap_alignments, overlap_weights)
            elif start == 0:
                align = anchor
            else:
                prev = start - 1
                if fused[prev] is None:
                    raise RuntimeError(
                        f"Cannot stitch window start={start}: no overlap and previous frame {prev} is missing"
                    )
                align = fused[prev] @ torch.linalg.inv(local[0])

            for slot, gidx in enumerate(frame_ids):
                candidates[gidx].append(align @ local[slot])
                weights[gidx].append(_local_slot_weight(slot, window))
                fused[gidx] = _weighted_pose_average(candidates[gidx], weights[gidx])

    missing = [i for i, Pi in enumerate(fused) if Pi is None]
    if missing:
        raise RuntimeError(f"local_stitch export missed {len(missing)} frames; first missing={missing[:5]}")
    return torch.stack([Pi for Pi in fused if Pi is not None], dim=0).to(device=device, dtype=torch.float32)


def _export_initial_pose_anchor(state) -> torch.Tensor:
    scene_cfg = getattr(state.cfg, "scene", None)
    loading_cfg = getattr(scene_cfg, "loading", None)
    pose_type = str(getattr(loading_cfg, "pose_type", "absolute")).lower()
    device = state.device
    if pose_type == "relative":
        return torch.eye(4, device=device, dtype=torch.float32)
    container = getattr(getattr(state, "sequence_sampler", None), "container", None)
    if container is None or len(container) <= 0:
        raise RuntimeError("FiMA export requires a non-empty sequence sampler container.")
    return container.get(0).pose.to(device=device, dtype=torch.float32)


def export_adjacent_chain_poses_probe_world(state) -> torch.Tensor:
    """
    与 FiMA 相邻帧阶段 A 评测链式一致：``P_{i+1}=P_i T_i``。

    Relative-pose scenes keep ``P_0=I``. Absolute-pose scenes anchor the chain
    at the first loaded sweep pose so exported priors can initialize Ours
    without GT-pose frame-0 leakage beyond the selected pose-estimation prior.

    Returns:
        ``(N, 4, 4)`` float32 tensor，与对应 ``SliceMeta.pose`` 可比（同设备上 probe→world）。
    """
    mc = state.cfg.model
    pem = mc.pem
    device = state.device
    n = int(state.sequence_sampler.n_frames)
    input_mode = str(getattr(pem, "input_mode", "multiframe")).lower()
    export_mode = str(getattr(pem, "export_pose_mode", "chain")).lower()

    merge_windows = str(getattr(pem, "train_protocol", "standard")).lower() == "phase_a_local_window"

    state.pose_estimator.eval()
    with torch.no_grad():
        if export_mode == "local_stitch":
            if input_mode == "adjacent_pair" or str(getattr(pem, "pem_output_layout", "")) != "window_edges":
                raise ValueError("export_pose_mode=local_stitch requires multiframe pem_output_layout=window_edges")
            return _export_local_stitched_window_poses_probe_world(state)
        if export_mode != "chain":
            raise ValueError(f"Unsupported PEM export_pose_mode={export_mode!r}; expected chain|local_stitch")
        if input_mode != "adjacent_pair" and str(getattr(pem, "pem_output_layout", "")) == "window_edges":
            edge_pred = _infer_multiframe_window_edge_predictions(state)
        elif input_mode == "adjacent_pair":
            edge_pred = (
                _merge_phase_a_window_edge_predictions(state)
                if merge_windows
                else _infer_global_edge_predictions(state)
            )
        else:
            raise ValueError(
                "export_adjacent_chain_poses_probe_world supports adjacent_pair or "
                "multiframe pem_output_layout=window_edges"
            )

    anchor = _export_initial_pose_anchor(state)
    return _compose_chain_from_edges(edge_pred, n, device, initial_pose=anchor)


def export_adjacent_chain_poses_numpy(state) -> np.ndarray:
    t = export_adjacent_chain_poses_probe_world(state).detach().cpu().numpy().astype(np.float32)
    if t.ndim != 3 or t.shape[1:] != (4, 4):
        raise RuntimeError(f"expected (N,4,4), got {t.shape}")
    return t
