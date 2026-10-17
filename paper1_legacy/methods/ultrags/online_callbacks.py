"""
Online 训练回调编排。

结构说明（与 Engine 的配合）
----------------------------
``Engine.train()`` 对每个 callback **只调用一次** ``on_train``。因此 **流式 while 循环**
只能放在 **一个** ``UltraGSOnlineStreamCallback`` 内；首窗 / 后续窗无法在 Engine 层拆成
两个独立 ``on_train``，否则第二个永远不会在流中间执行。

为保持与规格文档一致的可读结构，将逻辑拆为：

1. ``UltraGSOnlineSetupCallback`` — 仅初始化组件；``GaussContainer`` 为 **0 点占位**（``init_empty``），高斯在 stream 各帧按 ``concat`` 装入（``seed_every_frames`` + ``scene.sampler.num_new`` 或 ``stream_seed``；增密主要靠 DC）。
2. **窗口本位 pipeline**（首窗 / 后续窗的 ``process_stream_frame_*`` 等）— 与 ``UltraGSOnlineState`` 紧耦合，见本文件前半段。
3. ``UltraGSOnlineWindowZeroStreamPipeline`` — **首窗 $W_0$ 非标**（函数式 ``run``）。
4. ``UltraGSOnlineStandardWindowStreamPipeline`` — **后续窗标准流**（函数式 ``run``）。
5. ``UltraGSOnlineStreamCallback`` — 流式 ``while``，按 ``logical_fid`` 调用 3 或 4，并串联原子操作。

本模块内的 ``atom_*`` 为原 ``online_plugins.py`` 中的原子操作（函数式）。
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from engine.computer import LossComputer, MetricComputer
from engine.registry import register_callback
from scene.factory import SceneFactory
from scene.meta.slice_meta import SliceMeta
from scene.scene import Scene
from utils.image_utils import make_diff
from utils.pose_utils import sample_noisy_pose_se3

from .config import StreamSeedConfig
from .render import occupancy_from_intermediates, render
from .render_integrated import render as render_integrated
from .state import UltraGSOnlineState
from .gauss_container import UltraGSGaussContainer
from .gauss_solver import UltraGSGaussSolver
from .local_window_manager import LocalWindowManager
from .gaussian_state_manager import GaussianStateManager, build_gaussian_state_manager
from .pose_refiner import PoseRefiner
from .density_control import UltraGSOnlineDensityController
from . import online_callbacks_logger as oc_logger
from . import online_callbacks_occupancy_sampler as occ_sampler

_SEED_DEBUG_FILENAME = "debug_new_seed_strategy.txt"


def _stream_eval_should_run(state: UltraGSOnlineState, frame_id: int) -> bool:
    """是否在处理本帧训练前做一次「当前高斯 + 本帧 GT slice」的流式评测。"""
    ec = state.cfg.engine
    interval = max(0, int(getattr(ec, "stream_eval_interval_frames", 0) or 0))
    if interval <= 0:
        return False
    skip_head = max(0, int(getattr(ec, "stream_eval_skip_head_frames", 0) or 0))
    skip_tail = max(0, int(getattr(ec, "stream_eval_skip_tail_frames", 0) or 0))
    n = max(0, int(state.scene.scheduler.dataset_len))
    hi = n - skip_tail
    if hi <= skip_head or frame_id < skip_head or frame_id >= hi:
        return False
    return (frame_id - skip_head) % interval == 0


def _use_render_new(state: UltraGSOnlineState) -> bool:
    online_cfg = getattr(state.model_cfg, "online", None)
    render_cfg = getattr(state.model_cfg, "render", None)
    which = str(getattr(render_cfg, "which_render", "legacy")).lower()
    return which == "integrated"


def _render_dispatch(state: UltraGSOnlineState):
    return render_integrated if _use_render_new(state) else render


def seed_debug_log_path(state: Optional[UltraGSOnlineState] = None) -> str:
    """播种策略调试日志文件（优先写入当前实验 output_dir/logs）。"""
    if state is not None and getattr(state, "cfg", None) is not None:
        logs_dir = os.path.join(state.cfg.system.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, _SEED_DEBUG_FILENAME)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, _SEED_DEBUG_FILENAME)


def _seed_debug_enabled(state: UltraGSOnlineState) -> bool:
    ss = getattr(getattr(state.model_cfg, "online", None), "stream_seed", None)
    return bool(getattr(ss, "debug_enabled", False)) if ss is not None else False


def _append_seed_debug_line(state: UltraGSOnlineState, event: str, **kv: Any) -> None:
    """按行追加播种策略关键决策，便于离线逐帧回放。"""
    if not _seed_debug_enabled(state):
        return
    path = seed_debug_log_path(state)
    fid = int(getattr(state, "stream_frame_id", -1))
    gstep = int(getattr(state, "iteration", -1))
    items = [f"event={event}", f"frame={fid}", f"gs_step={gstep}"]
    for k, v in kv.items():
        if isinstance(v, float):
            items.append(f"{k}={v:.6g}")
        else:
            items.append(f"{k}={v}")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\t".join(items) + "\n")
    except OSError as exc:
        print(f"[SeedDebug] warn: append log failed ({path}): {exc}")


def _seed_pack_size(pack: Optional[Dict[str, torch.Tensor]]) -> int:
    return occ_sampler.seed_pack_size(pack)


@torch.no_grad()
def _sample_stream_pixel_seed(
    scene: Scene,
    slice_meta: SliceMeta,
    num_to_sample: int,
) -> Optional[Dict[str, torch.Tensor]]:
    return occ_sampler.sample_stream_pixel_seed(scene, slice_meta, num_to_sample)


@torch.no_grad()
def _gaussian_blur_hw(diff_hw: torch.Tensor, sigma: float) -> torch.Tensor:
    return occ_sampler.gaussian_blur_hw(diff_hw, sigma)


@torch.no_grad()
def _gather_seed_at_pixel_indices(
    scene: Scene,
    slice_meta: SliceMeta,
    indices: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    return occ_sampler.gather_seed_at_pixel_indices(scene, slice_meta, indices)


@torch.no_grad()
def _select_pixel_indices_occupancy_guided(
    state: UltraGSOnlineState,
    ss: StreamSeedConfig,
) -> torch.Tensor:
    return occ_sampler.select_pixel_indices_occupancy_guided(
        state=state,
        ss=ss,
        get_slice_meta_for_training=get_slice_meta_for_training,
        append_seed_debug_line=_append_seed_debug_line,
    )


def _resolve_stream_new_gaussians(state: UltraGSOnlineState, slice_meta: SliceMeta) -> Optional[Dict[str, torch.Tensor]]:
    """解析本 stream 帧待 ``concat`` 的高斯：legacy 或 occupancy+diff 门控（见 ``StreamSeedConfig``）。"""
    oc = state.model_cfg.online
    ss = getattr(oc, "stream_seed", None)
    mode = str(getattr(ss, "mode", "legacy") if ss is not None else "legacy").lower()
    n_pts = int(state.gauss_container.num_points)
    fid = int(state.stream_frame_id)
    dm = state.scene.dataset_meta
    P = int(dm.H) * int(dm.W)

    if mode == "occupancy_guided" and ss is not None:
        # 池空：首帧可逐像素；否则用 cap 与 num_new 兜底_uniform
        if n_pts == 0:
            if bool(ss.first_frame_dense) and fid == 0:
                idx = torch.arange(P, device=state.device, dtype=torch.long)
                pack = _gather_seed_at_pixel_indices(state.scene, slice_meta, idx)
                _append_seed_debug_line(
                    state,
                    "resolve_seed_first_frame_dense",
                    mode=mode,
                    n_total=n_pts,
                    dense_pixels=P,
                    sampled=_seed_pack_size(pack),
                )
                return pack
            num_fb = min(int(ss.max_new_cap), max(1, int(state.scene.cfg.sampler.num_new)))
            pack = _sample_stream_pixel_seed(state.scene, slice_meta, num_fb)
            _append_seed_debug_line(
                state,
                "resolve_seed_fallback_empty_pool",
                mode=mode,
                n_total=n_pts,
                fallback_num=num_fb,
                sampled=_seed_pack_size(pack),
            )
            return pack

        seed_every = int(oc.seed_every_frames)
        if seed_every > 0 and (fid % seed_every != 0):
            _append_seed_debug_line(
                state,
                "resolve_seed_skip_schedule",
                mode=mode,
                n_total=n_pts,
                seed_every=seed_every,
                frame_mod=(fid % seed_every),
            )
            return None

        idx = _select_pixel_indices_occupancy_guided(state, ss)
        pack = _gather_seed_at_pixel_indices(state.scene, slice_meta, idx)
        _append_seed_debug_line(
            state,
            "resolve_seed_occ_guided",
            mode=mode,
            n_total=n_pts,
            selected_pixels=int(idx.numel()),
            sampled=_seed_pack_size(pack),
        )
        return pack

    # --- legacy：固定帧频 + num_new ---
    seed_every = int(oc.seed_every_frames)
    num_new_cfg = int(state.scene.cfg.sampler.num_new)
    want = (seed_every > 0 and fid % seed_every == 0) or n_pts == 0
    if not want:
        _append_seed_debug_line(
            state,
            "resolve_seed_legacy_skip",
            mode=mode,
            n_total=n_pts,
            seed_every=seed_every,
        )
        return None
    pack = _sample_stream_pixel_seed(state.scene, slice_meta, num_new_cfg)
    _append_seed_debug_line(
        state,
        "resolve_seed_legacy",
        mode=mode,
        n_total=n_pts,
        num_new_cfg=num_new_cfg,
        sampled=_seed_pack_size(pack),
    )
    return pack


# ---------------------------------------------------------------------------
# 窗口本位 pipeline（docs/methods/baselines/ultrags_baseline/online_update_spec.md §4；原 window_pipeline）
# ---------------------------------------------------------------------------


def global_frame_id(frame_index: int) -> int:
    """数据集全局帧下标（流从 0 起步进，与 LWM 逻辑帧一致）。"""
    return int(frame_index)


def _set_gauss_requires_grad(container: UltraGSGaussContainer, requires: bool) -> None:
    for _, g in container._param_dict.items():
        g.tensor.requires_grad_(requires)


def _emit_online_step_logs(
    state: UltraGSOnlineState,
    loss_dict: Dict[str, torch.Tensor],
    stage: str,
    *,
    optim_frame_id: Optional[int] = None,
) -> None:
    """按 online 配置把本步 loss 写入标量管理器并在控制台可选打印。

    optim_frame_id：本步 GS 实际优化的全局帧下标（联合段会轮询窗内多帧）。
    若为 None 则退回 stream_frame_id（仅与「当前 ingest 的流式帧」一致，会误导 joint 日志）。
    """
    oc = state.model_cfg.online
    if not getattr(oc, "log_scalars_every_gs_step", False) and not getattr(
        oc, "console_log_every_gs_step", False
    ):
        return
    floats = {k: float(v.detach().item()) for k, v in loss_dict.items()}
    step = int(state.iteration)
    opt_fid = int(optim_frame_id) if optim_frame_id is not None else int(state.stream_frame_id)
    stream_fid = int(state.stream_frame_id)
    sm = state.scalar_manager
    if getattr(oc, "log_scalars_every_gs_step", False):
        payload = {f"{stage}/{k}": v for k, v in floats.items()}
        payload["optim_frame_id"] = float(opt_fid)
        if stream_fid != opt_fid:
            payload["stream_frame_id"] = float(stream_fid)
        sm.log_scalars(payload, step, prefix="train")
    if getattr(oc, "console_log_every_gs_step", False):
        sm.print_online_gs_step(step, opt_fid, stage, floats, stream_frame_id=stream_fid)


def get_slice_meta_for_training(state: UltraGSOnlineState, global_fid: int) -> SliceMeta:
    """训练用 SliceMeta：模式 A 用 GT；模式 B 用 PoseRefiner(coarse)。"""
    mc = state.model_cfg
    oc = mc.online
    sm = state.scene.container.get(global_fid)
    mode = str(oc.mode).upper()
    if mode == "A":
        return sm
    if mode == "C":
        return sm
    if mode == "B":
        if state.pose_refiner is None:
            return sm
        p_gt = sm.pose.to(state.device)
        if global_fid not in state.pose_carryover.coarse_by_frame:
            if global_fid == 0:
                # 帧 0 作为绝对锚点，保持干净 pose（identity）。
                state.pose_carryover.coarse_by_frame[global_fid] = p_gt.detach().clone()
            else:
                state.pose_carryover.coarse_by_frame[global_fid] = sample_noisy_pose_se3(
                    p_gt,
                    state.device,
                    rot_std_deg=oc.noise_pose.rot_std_deg,
                    trans_std_mm=oc.noise_pose.trans_std_mm,
                )
        p_coarse = state.pose_carryover.coarse_by_frame[global_fid]
        p_refined = state.pose_refiner(p_coarse.unsqueeze(0)).squeeze(0)
        return sm.replace_pose(p_refined)
    raise ValueError(f"Unsupported online mode: {mode}")


def gsm_gauss_indices(
    state: UltraGSOnlineState,
    i_right_global: int,
    slice_meta: SliceMeta,
    window_j: int,
) -> Optional[torch.Tensor]:
    oc = state.model_cfg.online
    W = int(state.lwm.W)
    gsm_mode = str(getattr(oc.gsm, "mode", "off")).lower()
    l = int(oc.gsm.membership.sub_locality_l)
    use_membership = gsm_mode == "membership"
    use_spatial = gsm_mode == "spatial_chunk"
    use_slice_identity = gsm_mode == "slice_identity"
    if not use_membership and not use_spatial and not use_slice_identity:
        return None
    if use_membership:
        if l <= 0:
            return None
        return state.gsm.gauss_indices_for_sub(i_right_global, W)
    if use_slice_identity:
        return state.gsm.gauss_indices_for_sub(i_right_global, W)
    # Spatial-chunk: 用 Local Window 本位激活集合，取窗口内所有成员帧覆盖高斯的并集。
    window_frames = state.lwm.window_frame_ids(window_j)
    if not window_frames:
        window_frames = [i_right_global]
    picks = []
    means = state.gauss_container.means
    scales = state.gauss_container.scales
    for fid in window_frames:
        sm_i = state.scene.container.get(int(fid))
        idx_i = state.gsm.gauss_indices_for_active_set(
            i_right=i_right_global,
            W=W,
            means=means,
            scales=scales,
            slice_probe_origin=sm_i.probe_origin,
            slice_vec_elevational=sm_i.vec_elevational,
        )
        if idx_i is not None and idx_i.numel() > 0:
            picks.append(idx_i)
    if not picks:
        raise RuntimeError(
            "SpatialChunk GSM active set is empty. "
            f"window_j={window_j}, window_frames={window_frames}, "
            f"i_right_global={i_right_global}, "
            f"num_points={int(state.gauss_container.num_points)}. "
            "Please relax spatial_chunk thresholds (chunk_size_mm / active_half_width_mm / scale_margin_ratio) "
            "or check slice pose continuity."
        )
    return torch.unique(torch.cat(picks, dim=0))


def run_pose_wash(state: UltraGSOnlineState, global_fid: int) -> None:
    """POSE_WASH：冻结高斯，仅优化 PoseRefiner，num_pose_refinement 步。"""
    mc = state.model_cfg
    oc = mc.online
    mode = str(oc.mode).upper()
    if mode == "A" or state.pose_refiner is None or state.pose_refiner_optimizer is None:
        return

    n_inner = max(1, int(oc.num_pose_refinement))
    sm = state.scene.container.get(global_fid)
    p_gt = sm.pose.to(state.device)
    if global_fid not in state.pose_carryover.coarse_by_frame:
        if global_fid == 0:
            # 帧 0 不引入噪声，作为在线序列绝对参考位姿。
            state.pose_carryover.coarse_by_frame[global_fid] = p_gt.detach().clone()
        else:
            state.pose_carryover.coarse_by_frame[global_fid] = sample_noisy_pose_se3(
                p_gt,
                state.device,
                rot_std_deg=oc.noise_pose.rot_std_deg,
                trans_std_mm=oc.noise_pose.trans_std_mm,
            )
    p_noise = state.pose_carryover.coarse_by_frame[global_fid]
    sm_prev = state.scene.container.get(global_fid - 1) if global_fid > 0 else None
    img_prev = sm_prev.image.to(state.device) if sm_prev is not None else None
    img_curr = sm.image.to(state.device)
    p_prev_gt = sm_prev.pose.to(state.device) if sm_prev is not None else p_gt
    carry = state.pose_carryover
    p_prev = (carry.P_refined_prev if carry.P_refined_prev is not None else p_prev_gt).to(state.device)
    p_prevprev = (carry.P_refined_prevprev if carry.P_refined_prevprev is not None else p_prev).to(state.device)
    p_coarse_prev = (carry.P_coarse_prev if carry.P_coarse_prev is not None else p_prev_gt).to(state.device)
    p_prev_at_frame_start = p_prev.detach()
    gate_cfg = getattr(oc, "pose_wash_gate", None)
    gate_enable = bool(getattr(gate_cfg, "enable", False)) if gate_cfg is not None else False
    gate_min_improve = float(getattr(gate_cfg, "min_improve", 0.0)) if gate_cfg is not None else 0.0
    gate_use_abs = bool(getattr(gate_cfg, "use_abs_improve", False)) if gate_cfg is not None else False
    gate_rb_model = bool(getattr(gate_cfg, "rollback_model_on_reject", False)) if gate_cfg is not None else False

    selected_pose_losses = list(getattr(oc, "pose_wash_loss_names", []) or [])
    # Pose-wash 不使用旧的 pose_aux_rec/pose_registration，改走可控的新损失组合。
    selected_pose_losses = [
        n for n in selected_pose_losses
        if n not in ("pose_aux_rec", "pose_registration")
    ]
    if not selected_pose_losses:
        if "pose_gt_delta" not in state.loss_computer.loss_items:
            raise KeyError(
                "pose_gt_delta is required for pose-wash stage but not built. "
                "Please add it to loss.needed_losses."
            )
        selected_pose_losses = ["pose_gt_delta"]
    if "pose_gt_delta" not in selected_pose_losses:
        selected_pose_losses = ["pose_gt_delta"] + selected_pose_losses

    primary_pose_losses = [n for n in selected_pose_losses if n != "pose_occ_conf"]
    if not primary_pose_losses:
        primary_pose_losses = ["pose_gt_delta"]

    occ_once_losses = list(primary_pose_losses)
    if "pose_occ_conf" in selected_pose_losses:
        occ_once_losses.append("pose_occ_conf")

    def _compute_pose_loss_dict(
        p_refined_curr: torch.Tensor,
        selected_losses: list[str],
    ) -> Dict[str, torch.Tensor]:
        pred_aux = None
        occupancy_hw = None
        if "pose_occ_conf" in selected_losses:
            refined_slice_meta = sm.replace_pose_differentiable(p_refined_curr)
            exclude_idx = state.gsm.get_aux_render_indices(global_fid, state.gauss_container.num_points)
            pred_aux, _, inter = _render_dispatch(state)(
                mc,
                state.scene.dataset_meta,
                refined_slice_meta,
                state.gauss_container,
                active_global_indices=exclude_idx,
                is_jitter=False,
                return_intermediates=True,
            )
            occupancy_hw = inter.get("occupancy_image", None)
            if occupancy_hw is None:
                occupancy_hw = occupancy_from_intermediates(
                    inter,
                    int(state.scene.dataset_meta.H),
                    int(state.scene.dataset_meta.W),
                )
        return state.loss_computer.compute(
            state.inputs,
            state.outputs,
            selected_losses=selected_losses,
            gt_image=img_curr,
            prev_image=img_prev,
            curr_image=img_curr,
            pose_coarse_prev=p_coarse_prev,
            pose_coarse_curr=p_noise,
            pose_refined_prevprev=p_prevprev,
            pose_refined_prev=p_prev,
            pose_refined_curr=p_refined_curr,
            pose_gt_curr=p_gt,
            pred_aux_image=pred_aux,
            occupancy_image=occupancy_hw,
            delta_lateral_mm=float(state.scene.dataset_meta.delta_lateral_mm),
            delta_axial_mm=float(state.scene.dataset_meta.delta_axial_mm),
        )

    _set_gauss_requires_grad(state.gauss_container, False)
    try:
        pose_refiner_state_at_frame_start = (
            {k: v.detach().clone() for k, v in state.pose_refiner.state_dict().items()}
            if (gate_enable and gate_rb_model)
            else None
        )
        with torch.no_grad():
            coarse_loss_val = float(_compute_pose_loss_dict(p_noise.detach(), primary_pose_losses)["total"].item())
        pose_loss_val = 0.0
        p_refined = None
        grad_norm_sum = 0.0
        grad_norm_count = 0
        delta_twist_norm_sum = 0.0
        delta_twist_norm_count = 0
        for iw in range(n_inner):
            state.pose_refiner_optimizer.zero_grad()
            state.gauss_solver.zero_grad()
            p_refined = state.pose_refiner(p_noise.unsqueeze(0)).squeeze(0)
            step_losses = occ_once_losses if iw == 0 else primary_pose_losses
            pose_loss_dict = _compute_pose_loss_dict(p_refined, step_losses)
            pose_loss_val = float(pose_loss_dict["total"].detach().item())
            _emit_online_step_logs(state, pose_loss_dict, stage=f"pose_wash/{iw}", optim_frame_id=global_fid)
            pose_loss_dict["total"].backward()
            # Debug signal 1: gradient norm on PoseRefiner parameters.
            sq = 0.0
            for p in state.pose_refiner.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                sq += float(torch.sum(g * g).item())
            grad_norm = sq ** 0.5
            grad_norm_sum += grad_norm
            grad_norm_count += 1
            # Debug signal 2: delta twist norm from PoseRefiner output head.
            dt = getattr(state.pose_refiner, "last_delta_twist", None)
            if dt is not None and dt.numel() > 0:
                dt_norm = float(torch.linalg.norm(dt, dim=-1).mean().item())
                delta_twist_norm_sum += dt_norm
                delta_twist_norm_count += 1
            state.pose_refiner_optimizer.step()
            state.gauss_solver.zero_grad()
        if p_refined is not None:
            accepted_refined = True
            if gate_enable:
                with torch.no_grad():
                    refined_loss_val = float(
                        _compute_pose_loss_dict(p_refined.detach(), primary_pose_losses)["total"].item()
                    )
                improve = coarse_loss_val - refined_loss_val
                score = abs(improve) if gate_use_abs else improve
                accepted_refined = bool(score >= gate_min_improve)
                if not accepted_refined:
                    p_refined = p_noise
                    pose_loss_val = coarse_loss_val
                    if pose_refiner_state_at_frame_start is not None:
                        state.pose_refiner.load_state_dict(pose_refiner_state_at_frame_start, strict=True)
                        state.pose_refiner_optimizer.zero_grad()
                print(
                    f"[PoseWashGate] fid={int(global_fid)} accepted={int(accepted_refined)} "
                    f"improve={float(improve):+.6f} score={float(score):.6f} "
                    f"use_abs={int(gate_use_abs)} threshold={float(gate_min_improve):.6f}"
                )
            carry.pose_refine_loss = pose_loss_val
            carry.P_refined_prevprev = p_prev_at_frame_start
            carry.P_refined_prev = p_refined.detach()
            carry.P_coarse_prev = p_noise.detach()
            mean_grad_norm = grad_norm_sum / max(1, grad_norm_count)
            mean_delta_twist_norm = delta_twist_norm_sum / max(1, delta_twist_norm_count)
            oc_logger.log_pose_wash_result(
                state,
                frame_id=int(global_fid),
                gs_step=int(state.iteration),
                pose_wash_steps=int(n_inner),
                pose_wash_loss=float(pose_loss_val),
                pose_grad_norm_mean=float(mean_grad_norm),
                delta_twist_norm_mean=float(mean_delta_twist_norm),
                p_gt=p_gt.detach(),
                p_coarse=p_noise.detach(),
                p_refined=p_refined.detach(),
            )
    finally:
        _set_gauss_requires_grad(state.gauss_container, True)


def run_concat_seed(state: UltraGSOnlineState, global_fid: int) -> None:
    new_gaussians = state.stream_new_gaussians
    if new_gaussians is None:
        return
    tensors = state.gauss_container.complete_new_gaussians_for_concat(new_gaussians)
    n_new = int(tensors["means"].shape[0])
    if n_new <= 0:
        return
    state.gauss_solver.concat(tensors)
    state.gsm.on_concat(n_new=n_new, frame_id=global_fid)


def _update_membership_from_intermediates(
    state: UltraGSOnlineState, global_fid: int, intermediates: Optional[dict]
) -> None:
    if intermediates is None:
        return
    idxs = intermediates.get("seen_gauss_global_indices")
    if idxs is not None and idxs.numel() > 0:
        seen = idxs.to(dtype=torch.long)
        state.gsm.update_membership_from_render(global_fid, seen)


def gs_optim_one_step(
    state: UltraGSOnlineState,
    global_fid: int,
    i_right_global: int,
    window_j: int,
    *,
    stage: str = "gs",
) -> tuple[float, Optional[dict]]:
    """单步 GS_OPTIM：主渲染 + 全量损失（无帧权）。"""
    t0 = time.perf_counter()
    mc = state.model_cfg
    gsm_mode = str(getattr(mc.online.gsm, "mode", "off")).lower()
    slice_meta = get_slice_meta_for_training(state, global_fid)
    t_gsm0 = time.perf_counter()
    gauss_indices = gsm_gauss_indices(state, i_right_global, slice_meta, window_j)
    t_gsm_index_ms = (time.perf_counter() - t_gsm0) * 1000.0

    state.iteration += 1
    state.gauss_solver.step_lr(state.iteration)
    t_render0 = time.perf_counter()
    pred, depth, intermediates = _render_dispatch(state)(
        mc,
        state.scene.dataset_meta,
        slice_meta,
        state.gauss_container,
        active_global_indices=gauss_indices,
        is_jitter=False,
        return_intermediates=True,
    )
    t_render_ms = (time.perf_counter() - t_render0) * 1000.0
    gt = slice_meta.image
    seen_idx = None
    if intermediates is not None:
        seen_idx = intermediates.get("seen_gauss_global_indices")
    t_loss0 = time.perf_counter()
    render_loss_names = list(getattr(mc.online, "render_loss_names", []) or [])
    loss_dict = state.loss_computer.compute(
        state.inputs,
        state.outputs,
        selected_losses=render_loss_names if render_loss_names else None,
        gt_image=gt,
        pred_image=pred,
        depth_image=depth,
        gauss_scales=state.gauss_container.scales,
        gauss_scale_indices=seen_idx,
        gauss_opacities=state.gauss_container.opacities,
    )
    t_loss_ms = (time.perf_counter() - t_loss0) * 1000.0
    _emit_online_step_logs(state, loss_dict, stage=stage, optim_frame_id=global_fid)
    t_bw0 = time.perf_counter()
    loss_dict["total"].backward()
    t_backward_ms = (time.perf_counter() - t_bw0) * 1000.0
    dc = state.dc
    t_dc_acc0 = time.perf_counter()
    if dc is not None and dc.should_accumulate_grad_stats(state.iteration):
        dc.update_grad_stats(state.gauss_container, gauss_indices)
    t_dc_accum_ms = (time.perf_counter() - t_dc_acc0) * 1000.0
    t_opt0 = time.perf_counter()
    state.gauss_solver.step(gauss_indices)
    t_opt_step_ms = (time.perf_counter() - t_opt0) * 1000.0
    t_zero0 = time.perf_counter()
    state.gauss_solver.zero_grad()
    t_zero_grad_ms = (time.perf_counter() - t_zero0) * 1000.0
    t_dc_prune0 = time.perf_counter()
    if dc is not None and dc.should_densify_prune(state.iteration):
        dc.prune_and_densify(
            state.iteration,
            state.gauss_solver,
            state.gauss_container,
            gsm=state.gsm,
            birth_frame_id=state.stream_frame_id,
            active_global_indices=gauss_indices,
        )
    t_dc_prune_ms = (time.perf_counter() - t_dc_prune0) * 1000.0
    raw = float(loss_dict["total"].detach().item())
    _update_membership_from_intermediates(state, global_fid, intermediates)
    t_total_ms = (time.perf_counter() - t0) * 1000.0
    n_total = int(state.gauss_container.num_points)
    n_active = int(gauss_indices.numel()) if gauss_indices is not None else n_total
    oc_logger.append_step_timing_line(
        gs_step=int(state.iteration),
        stage=str(stage),
        optim_frame_id=int(global_fid),
        window_j=int(window_j),
        i_right=int(i_right_global),
        gsm_mode=gsm_mode,
        n_total=n_total,
        n_active=n_active,
        t_gsm_index_ms=t_gsm_index_ms,
        t_render_ms=t_render_ms,
        t_loss_ms=t_loss_ms,
        t_backward_ms=t_backward_ms,
        t_dc_accum_ms=t_dc_accum_ms,
        t_opt_step_ms=t_opt_step_ms,
        t_zero_grad_ms=t_zero_grad_ms,
        t_dc_prune_ms=t_dc_prune_ms,
        t_total_ms=t_total_ms,
    )
    oc_logger.o1_load_accumulate_step(
        state,
        n_total=n_total,
        n_active=n_active,
        t_gsm_index_ms=t_gsm_index_ms,
        t_render_ms=t_render_ms,
        t_loss_ms=t_loss_ms,
        t_backward_ms=t_backward_ms,
        t_opt_step_ms=t_opt_step_ms,
        t_total_ms=t_total_ms,
    )
    return raw, intermediates


def run_independent_steps(
    state: UltraGSOnlineState,
    logical_fid: int,
    window_j: int,
) -> tuple[float, int]:
    """独立段：仅当前帧。返回 (loss 和, 子步数)。"""
    lwm = state.lwm
    gf = global_frame_id(logical_fid)
    k = lwm.k_ind(logical_fid, window_j)
    if k <= 0:
        return 0.0, 0
    total = 0.0
    i_right_global = gf
    for _ in range(k):
        raw, inter = gs_optim_one_step(state, gf, i_right_global, window_j, stage="ind")
        total += raw
        state.outputs["render_intermediates"] = inter
    return total, k


def run_joint_steps(
    state: UltraGSOnlineState,
    window_j: int,
) -> tuple[float, int]:
    """联合段：轮询帧直至各帧 joint 子步耗尽。"""
    lwm = state.lwm
    pool = dict(lwm.joint_pool(window_j))
    if not pool:
        return 0.0, 0
    order = sorted(pool.keys())
    cursor = 0
    total_raw = 0.0
    n_steps = 0
    i_right_global = global_frame_id(window_j + lwm.W - 1)

    while True:
        active = [i for i in order if pool.get(i, 0) > 0]
        if not active:
            break
        logical_i = active[cursor % len(active)]
        pool[logical_i] -= 1
        cursor += 1
        gf = global_frame_id(logical_i)
        raw, inter = gs_optim_one_step(state, gf, i_right_global, window_j, stage="joint")
        total_raw += raw
        n_steps += 1
        state.outputs["render_intermediates"] = inter
    return total_raw, n_steps


def process_stream_frame_first_window(
    state: UltraGSOnlineState,
    logical_fid: int,
) -> None:
    """首窗 W0：帧 0 无 POSE_WASH；帧 1..W-1 有；最后一帧独立段后接 W0 联合段。"""
    W = int(state.lwm.W)
    mode = str(state.model_cfg.online.mode).upper()
    t_pose_wash_ms = 0.0
    t_concat_ms = 0.0
    t_ind_ms = 0.0
    t_joint_ms = 0.0
    n_joint = 0

    gf = global_frame_id(logical_fid)
    if logical_fid > 0 and mode == "B":
        t_pw0 = time.perf_counter()
        run_pose_wash(state, gf)
        t_pose_wash_ms = float((time.perf_counter() - t_pw0) * 1000.0)
    t_concat0 = time.perf_counter()
    run_concat_seed(state, gf)
    t_concat_ms = float((time.perf_counter() - t_concat0) * 1000.0)

    t_ind0 = time.perf_counter()
    ind_sum, ind_n = run_independent_steps(state, logical_fid, window_j=0)
    t_ind_ms = float((time.perf_counter() - t_ind0) * 1000.0)
    if logical_fid == W - 1:
        t_joint0 = time.perf_counter()
        joint_sum, joint_n = run_joint_steps(state, window_j=0)
        t_joint_ms = float((time.perf_counter() - t_joint0) * 1000.0)
        n_joint = int(joint_n)
        state.outputs["frame_avg_loss"] = (ind_sum + joint_sum) / max(1, ind_n + joint_n)
    else:
        state.outputs["frame_avg_loss"] = ind_sum / max(1, ind_n)
    state.outputs["t_pipeline_pose_wash_ms"] = float(t_pose_wash_ms)
    state.outputs["t_pipeline_concat_ms"] = float(t_concat_ms)
    state.outputs["t_pipeline_ind_ms"] = float(t_ind_ms)
    state.outputs["t_pipeline_joint_ms"] = float(t_joint_ms)
    state.outputs["n_ind_steps"] = float(ind_n)
    state.outputs["n_joint_steps"] = float(n_joint)


def process_stream_frame_standard(
    state: UltraGSOnlineState,
    logical_fid: int,
) -> None:
    """后续窗：POSE_WASH → concat → 新帧独立段 → 联合段。"""
    W = int(state.lwm.W)
    mode = str(state.model_cfg.online.mode).upper()
    j = logical_fid - W + 1
    t_pose_wash_ms = 0.0

    gf = global_frame_id(logical_fid)
    if mode == "B":
        t_pw0 = time.perf_counter()
        run_pose_wash(state, gf)
        t_pose_wash_ms = float((time.perf_counter() - t_pw0) * 1000.0)
    t_concat0 = time.perf_counter()
    run_concat_seed(state, gf)
    t_concat_ms = float((time.perf_counter() - t_concat0) * 1000.0)

    t_ind0 = time.perf_counter()
    ind_sum, ind_n = run_independent_steps(state, logical_fid, window_j=j)
    t_ind_ms = float((time.perf_counter() - t_ind0) * 1000.0)
    t_joint0 = time.perf_counter()
    joint_sum, joint_n = run_joint_steps(state, window_j=j)
    t_joint_ms = float((time.perf_counter() - t_joint0) * 1000.0)
    state.outputs["frame_avg_loss"] = (ind_sum + joint_sum) / max(1, ind_n + joint_n)
    state.outputs["t_pipeline_pose_wash_ms"] = float(t_pose_wash_ms)
    state.outputs["t_pipeline_concat_ms"] = float(t_concat_ms)
    state.outputs["t_pipeline_ind_ms"] = float(t_ind_ms)
    state.outputs["t_pipeline_joint_ms"] = float(t_joint_ms)
    state.outputs["n_ind_steps"] = float(ind_n)
    state.outputs["n_joint_steps"] = float(joint_n)


def atom_merge(state: UltraGSOnlineState) -> None:
    oc = state.model_cfg.online
    merge_every = int(oc.density_control.merge_every_frames)
    state.outputs["merged_removed"] = 0
    if merge_every <= 0 or state.stream_frame_count % merge_every != 0:
        return
    dc = state.dc
    if dc is None:
        return
    fid = int(state.stream_frame_id)
    n = dc.run_periodic_pair_merge(
        state.gauss_solver,
        state.gauss_container,
        state.gsm,
        fid,
    )
    state.outputs["merged_removed"] = float(n)


# ---------------------------------------------------------------------------
# Callback 1：Setup
# ---------------------------------------------------------------------------


@register_callback("UltraGSOnlineSetupCallback")
class UltraGSOnlineSetupCallback:
    priority = 10

    def on_setup(self, state: UltraGSOnlineState):
        print(f"[UltraGS-Online] Setting up components for device: {state.device}")
        cfg = state.cfg
        if state.pose_carryover is None:
            state.pose_carryover = UltraGSOnlineState.OnlinePoseCarryover()

        oc_logger.configure_online_log_paths(cfg.system.output_dir)
        state.scene = SceneFactory.build_scene(cfg, state.device)
        state.gauss_container = UltraGSGaussContainer(cfg, device=state.device)
        state.gauss_container.init_empty()

        state.gauss_solver = UltraGSGaussSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()
        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]

        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)

        state.iteration = 0
        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)

        state.model_cfg = state.gauss_container.model_cfg

        online_cfg = state.model_cfg.online
        # todo 如何修改 Scene/SceneScheduler 让基类也有 dataset_len 属性
        dataset_len = int(state.scene.scheduler.dataset_len)
        N_stream = max(0, dataset_len)
        T = int(online_cfg.lwm.total_opt_steps or cfg.engine.iterations)
        W_cfg = int(online_cfg.lwm.window_size_frames)

        state.lwm = LocalWindowManager(
            N=N_stream,
            T=T,
            W=W_cfg,
            ratio_front=float(online_cfg.lwm.ratio_front),
            ratio_rear=float(online_cfg.lwm.ratio_rear),
        )
        state.gsm = build_gaussian_state_manager(
            mode=str(getattr(online_cfg.gsm, "mode", "off")),
            device=state.device,
            membership_k=int(online_cfg.gsm.membership.membership_k),
            sub_locality_l=int(online_cfg.gsm.membership.sub_locality_l),
            slice_identity_exclude_num_ids=int(online_cfg.gsm.slice_identity.exclude_num_ids),
            spatial_chunk_size_mm=float(online_cfg.gsm.spatial_chunk.chunk_size_mm),
            spatial_active_half_width_mm=float(online_cfg.gsm.spatial_chunk.active_half_width_mm),
            spatial_scale_margin_ratio=float(online_cfg.gsm.spatial_chunk.scale_margin_ratio),
        )
        state.gsm.initialize(num_points=state.gauss_container.num_points, birth_frame_id=0)

        odc = online_cfg.density_control
        if (
            odc.enable.densify
            or odc.enable.prune
            or getattr(odc, "merge_after_dc", False)
            or int(online_cfg.density_control.merge_every_frames) > 0
        ):
            state.dc = UltraGSOnlineDensityController(cfg)
        else:
            state.dc = None

        mode = str(online_cfg.mode).upper()
        state.pose_estimator = None
        state.pose_refiner = None
        state.pose_refiner_optimizer = None
        if mode == "B":
            pr_cfg = online_cfg.pose_refiner
            state.pose_refiner = PoseRefiner(
                pose_dim=pr_cfg.pose_dim,
                num_frequencies=pr_cfg.num_frequencies,
                encoder_hidden_dim=pr_cfg.encoder_hidden_dim,
                encoder_layers=pr_cfg.encoder_layers,
                bottleneck_dim=pr_cfg.bottleneck_dim,
                decoder_hidden_dim=pr_cfg.decoder_hidden_dim,
                decoder_layers=pr_cfg.decoder_layers,
                max_delta_rot_deg=pr_cfg.max_delta_rot_deg,
                max_delta_trans_mm=pr_cfg.max_delta_trans_mm,
            ).to(state.device)
            state.pose_refiner_optimizer = torch.optim.Adam(state.pose_refiner.parameters(), lr=pr_cfg.lr)

        oc_logger.reset_step_timing_log_file()
        print(f"[UltraGS-Online] Step-timing TSV cleared: {oc_logger.step_timing_log_path()}")
        oc_logger.reset_o1_load_log_file()
        print(f"[UltraGS-Online] O1-load TSV cleared: {oc_logger.o1_load_log_path()}")
        oc_logger.reset_pose_wash_log_file()
        print(f"[UltraGS-Online] Pose-wash TSV cleared: {oc_logger.pose_wash_log_path()}")
        oc_logger.reset_frame_timing_log_file()
        print(f"[UltraGS-Online] Frame-timing TSV cleared: {oc_logger.frame_timing_log_path()}")
        oc_logger.online_run_init(state)
        if _seed_debug_enabled(state):
            with open(seed_debug_log_path(state), "w", encoding="utf-8") as f:
                f.write(
                    "event\tframe\tgs_step\tmode\tn_total\tseed_every\tselected_before_cap\t"
                    "selected_after_cap\tsampled\toccupancy_mean\tlack_mean\tdiff_mean\n"
                )
            print(f"[UltraGS-Online] Seed-debug log reset: {seed_debug_log_path(state)}")
        print("[UltraGS-Online] Components initialized successfully.")


# ---------------------------------------------------------------------------
# Callback 2（逻辑）：首窗 $W_0$ 非标管道
# ---------------------------------------------------------------------------
# 注册到 CALLBACK_REGISTRY 仅便于按名称发现；Engine 的 needed_callbacks 中只放
# UltraGSOnlineStreamCallback，本类由 Stream 内部调用 ``run``。


@register_callback("UltraGSOnlineWindowZeroStreamPipeline")
class UltraGSOnlineWindowZeroStreamPipeline:
    priority = 49

    @staticmethod
    def run(state: UltraGSOnlineState, logical_fid: int) -> None:
        """首窗填充 + 在 $W-1$ 后接 $W_0$ 联合段。"""
        process_stream_frame_first_window(state, logical_fid)


# ---------------------------------------------------------------------------
# Callback 3（逻辑）：后续窗标准管道
# ---------------------------------------------------------------------------


@register_callback("UltraGSOnlineStandardWindowStreamPipeline")
class UltraGSOnlineStandardWindowStreamPipeline:
    priority = 49

    @staticmethod
    def run(state: UltraGSOnlineState, logical_fid: int) -> None:
        """POSE_WASH → concat → 独立段 → 联合段。"""
        process_stream_frame_standard(state, logical_fid)


# ---------------------------------------------------------------------------
# 流式驱动：按帧选择首窗 / 后续窗管道，并串联原子操作
# ---------------------------------------------------------------------------


@register_callback("UltraGSOnlineStreamCallback")
class UltraGSOnlineStreamCallback:
    priority = 50

    def on_train(self, state: UltraGSOnlineState):
        assert state.model_cfg is not None
        oc = state.model_cfg.online

        hooks: Dict[str, Callable[..., Any]] = {
            "save_checkpoint": _save_checkpoint,
        }

        pbar = tqdm(desc="StreamingFrames")
        prev_tqdm = getattr(state.scalar_manager, "tqdm_bar", None)
        state.scalar_manager.tqdm_bar = pbar
        try:
            while True:
                frame_t0 = time.perf_counter()
                t_fetch0 = time.perf_counter()
                datum = state.scene.next_stream_datum()
                t_fetch_datum_ms = float((time.perf_counter() - t_fetch0) * 1000.0)
                if datum is None:
                    print("[UltraGS-Online] Stream finished.")
                    break

                state.stream_frame_id = int(datum["index"])
                state.stream_slice_meta = datum["slice_meta"]
                logical_fid = int(state.stream_frame_id)
                t_pre_stream_eval_ms = 0.0
                if _stream_eval_should_run(state, logical_fid):
                    t_pe0 = time.perf_counter()
                    _run_online_eval_one(
                        state,
                        frame_id=logical_fid,
                        global_step=int(state.iteration),
                        slice_meta=datum["slice_meta"],
                    )
                    t_pre_stream_eval_ms = float((time.perf_counter() - t_pe0) * 1000.0)
                state.outputs["t_pre_stream_eval_ms"] = t_pre_stream_eval_ms
                t_seed0 = time.perf_counter()
                state.stream_new_gaussians = _resolve_stream_new_gaussians(state, datum["slice_meta"])
                t_resolve_seed_ms = float((time.perf_counter() - t_seed0) * 1000.0)
                _ng = state.stream_new_gaussians
                state.outputs["stream_seed_num"] = (
                    float(_ng["means"].shape[0]) if _ng is not None else 0.0
                )
                state.outputs["render_intermediates"] = None
                state.outputs["frame_avg_loss"] = 0.0
                state.outputs["merged_removed"] = 0
                oc_logger.o1_load_init_frame(state)
                n_total_begin = int(state.gauss_container.num_points)

                W = int(state.lwm.W)
                t_pipe0 = time.perf_counter()
                if logical_fid < W:
                    UltraGSOnlineWindowZeroStreamPipeline.run(state, logical_fid)
                    window_type = "w0"
                else:
                    UltraGSOnlineStandardWindowStreamPipeline.run(state, logical_fid)
                    window_type = "std"
                t_pipeline_ms = float((time.perf_counter() - t_pipe0) * 1000.0)
                t_merge0 = time.perf_counter()
                atom_merge(state)
                t_merge_ms = float((time.perf_counter() - t_merge0) * 1000.0)
                state.outputs["frame_wall_ms"] = float((time.perf_counter() - frame_t0) * 1000.0)
                state.outputs["t_fetch_datum_ms"] = t_fetch_datum_ms
                state.outputs["t_resolve_seed_ms"] = t_resolve_seed_ms
                state.outputs["t_pipeline_ms"] = t_pipeline_ms
                state.outputs["t_merge_ms"] = t_merge_ms
                t_eval0 = time.perf_counter()
                oc_logger.atom_log_eval_save(state, hooks)
                t_eval_save_ms = float((time.perf_counter() - t_eval0) * 1000.0)
                state.outputs["t_eval_save_ms"] = t_eval_save_ms
                state.outputs["run_sum_eval_save_ms"] = float(
                    state.outputs.get("run_sum_eval_save_ms", 0.0) + t_eval_save_ms
                )
                oc_logger.append_frame_timing_line(
                    frame_id=int(state.stream_frame_id),
                    logical_fid=logical_fid,
                    window_type=window_type,
                    stream_seed_num=float(state.outputs.get("stream_seed_num", 0.0)),
                    n_total_begin=n_total_begin,
                    n_total_end=int(state.gauss_container.num_points),
                    n_ind_steps=int(state.outputs.get("n_ind_steps", 0.0)),
                    n_joint_steps=int(state.outputs.get("n_joint_steps", 0.0)),
                    t_fetch_datum_ms=t_fetch_datum_ms,
                    t_pre_stream_eval_ms=float(state.outputs.get("t_pre_stream_eval_ms", 0.0)),
                    t_resolve_seed_ms=t_resolve_seed_ms,
                    t_pipeline_ms=t_pipeline_ms,
                    t_merge_ms=t_merge_ms,
                    t_eval_save_ms=t_eval_save_ms,
                    t_frame_wall_ms=float(state.outputs.get("frame_wall_ms", 0.0)),
                    t_pipeline_pose_wash_ms=float(state.outputs.get("t_pipeline_pose_wash_ms", 0.0)),
                    t_pipeline_concat_ms=float(state.outputs.get("t_pipeline_concat_ms", 0.0)),
                    t_pipeline_ind_ms=float(state.outputs.get("t_pipeline_ind_ms", 0.0)),
                    t_pipeline_joint_ms=float(state.outputs.get("t_pipeline_joint_ms", 0.0)),
                )

                state.stream_frame_count += 1
                pbar.update(1)
                if state.stop_training:
                    break
        finally:
            state.scalar_manager.tqdm_bar = prev_tqdm
            oc_logger.flush_online_debug_logs(force=True)
            oc_logger.write_online_run_summary(state)


def _save_checkpoint(state: UltraGSOnlineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


@torch.no_grad()
def _run_online_eval_one(state: UltraGSOnlineState, frame_id: int, global_step: int, slice_meta):
    if state.gauss_container.num_points <= 0:
        return
    cfg = state.cfg
    vis_dir = os.path.join(cfg.system.output_dir, "vis_stream")
    os.makedirs(vis_dir, exist_ok=True)

    pred, _, _ = _render_dispatch(state)(
        state.model_cfg,
        state.scene.dataset_meta,
        slice_meta,
        state.gauss_container,
        return_intermediates=False,
        is_jitter=False,
    )
    gt = slice_meta.image
    state.inputs["gt_image"] = gt
    state.outputs["pred_image"] = pred
    scores = state.metric_computer.compute(state.inputs, state.outputs)
    state.outputs["run_eval_count"] = float(state.outputs.get("run_eval_count", 0.0) + 1.0)
    if "psnr" in scores:
        state.outputs["last_eval_psnr"] = float(scores["psnr"])
    if "ssim" in scores:
        state.outputs["last_eval_ssim"] = float(scores["ssim"])
    if "l1" in scores:
        state.outputs["last_eval_l1"] = float(scores["l1"])

    item_dir = os.path.join(vis_dir, f"frame_{frame_id}")
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    diff = make_diff(pred_np, gt_np)
    state.io_manager.save_images(item_dir, pred, gt, diff)

    gt_f = np.clip(gt_np, 0.0, 1.0).astype(np.float32)
    pred_f = np.clip(pred_np, 0.0, 1.0).astype(np.float32)
    diff_f = (diff.astype(np.float32) / 255.0)
    compose = np.concatenate([gt_f, pred_f, diff_f], axis=1)
    compose_path = os.path.join(item_dir, "compose.png")
    state.io_manager.save_image(compose_path, compose)

    metrics_log = {f"stream_test/{k}": v for k, v in scores.items()}
    metrics_log["stream_test/frame_id"] = float(frame_id)
    state.scalar_manager.log_scalars(metrics_log, global_step)
    parts = []
    for k in sorted(scores.keys()):
        v = scores[k]
        if isinstance(v, (int, float)):
            parts.append(f"{k}={float(v):.4f}")
    print(
        "[UltraGS-Online] Stream pre-train eval | "
        f"frame={int(frame_id)} | gs_step={int(global_step)} | " + " ".join(parts)
    )
    if getattr(state.model_cfg.online, "console_print_eval_metrics", False):
        state.scalar_manager.print_test(global_step, scores, in_place=False)
