from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

import torch

from .state import UltraGSOnlineState

_STEP_TIMING_FILENAME = "online_step_timing_breakdown.tsv"
_O1_LOAD_FILENAME = "online_o1_load_monitor.tsv"
_POSE_WASH_FILENAME = "online_pose_wash_monitor.tsv"
_FRAME_TIMING_FILENAME = "online_frame_timing_breakdown.tsv"
_LOG_IO_FLUSH_EVERY = 256
_STEP_TIMING_LINE_BUFFER: list[str] = []
_O1_LOAD_LINE_BUFFER: list[str] = []
_POSE_WASH_LINE_BUFFER: list[str] = []
_FRAME_TIMING_LINE_BUFFER: list[str] = []
_STEP_TIMING_PATH: Optional[str] = None
_O1_LOAD_PATH: Optional[str] = None
_POSE_WASH_PATH: Optional[str] = None
_FRAME_TIMING_PATH: Optional[str] = None


def step_timing_log_path() -> str:
    if _STEP_TIMING_PATH is None:
        raise RuntimeError("Online logger path is not configured. Call configure_online_log_paths() first.")
    return _STEP_TIMING_PATH


def o1_load_log_path() -> str:
    if _O1_LOAD_PATH is None:
        raise RuntimeError("Online logger path is not configured. Call configure_online_log_paths() first.")
    return _O1_LOAD_PATH


def pose_wash_log_path() -> str:
    if _POSE_WASH_PATH is None:
        raise RuntimeError("Online logger path is not configured. Call configure_online_log_paths() first.")
    return _POSE_WASH_PATH


def frame_timing_log_path() -> str:
    if _FRAME_TIMING_PATH is None:
        raise RuntimeError("Online logger path is not configured. Call configure_online_log_paths() first.")
    return _FRAME_TIMING_PATH


def configure_online_log_paths(output_dir: str) -> None:
    global _STEP_TIMING_PATH, _O1_LOAD_PATH, _POSE_WASH_PATH, _FRAME_TIMING_PATH
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    _STEP_TIMING_PATH = os.path.join(logs_dir, _STEP_TIMING_FILENAME)
    _O1_LOAD_PATH = os.path.join(logs_dir, _O1_LOAD_FILENAME)
    _POSE_WASH_PATH = os.path.join(logs_dir, _POSE_WASH_FILENAME)
    _FRAME_TIMING_PATH = os.path.join(logs_dir, _FRAME_TIMING_FILENAME)


def reset_step_timing_log_file() -> None:
    path = step_timing_log_path()
    header = (
        "gs_step\tstage\toptim_frame_id\twindow_j\ti_right\tgsm_mode\t"
        "n_total\tn_active\t"
        "t_gsm_index_ms\tt_render_ms\tt_loss_ms\tt_backward_ms\t"
        "t_dc_accum_ms\tt_opt_step_ms\tt_zero_grad_ms\tt_dc_prune_ms\tt_total_ms\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
    _STEP_TIMING_LINE_BUFFER.clear()


def reset_o1_load_log_file() -> None:
    path = o1_load_log_path()
    header = (
        "frame_id\tgsm_mode\tgs_steps\t"
        "n_total_last\tn_active_mean\tn_active_min\tn_active_max\tn_active_ratio_mean\t"
        "t_gsm_index_ms_mean\tt_render_ms_mean\tt_loss_ms_mean\tt_backward_ms_mean\t"
        "t_opt_step_ms_mean\tt_total_ms_mean\tt_total_ms_min\tt_total_ms_max\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
    _O1_LOAD_LINE_BUFFER.clear()


def reset_pose_wash_log_file() -> None:
    path = pose_wash_log_path()
    header = (
        "frame_id\tgs_step\tpose_wash_steps\tpose_wash_loss\t"
        "pose_grad_norm_mean\tdelta_twist_norm_mean\t"
        "coarse_trans_err_mm\trefined_trans_err_mm\ttrans_improve_mm\t"
        "coarse_rot_err_deg\trefined_rot_err_deg\trot_improve_deg\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
    _POSE_WASH_LINE_BUFFER.clear()


def reset_frame_timing_log_file() -> None:
    path = frame_timing_log_path()
    header = (
        "frame_id\tlogical_fid\twindow_type\tstream_seed_num\t"
        "n_total_begin\tn_total_end\tn_ind_steps\tn_joint_steps\t"
        "t_fetch_datum_ms\tt_pre_stream_eval_ms\tt_resolve_seed_ms\tt_pipeline_ms\tt_merge_ms\tt_eval_save_ms\tt_frame_wall_ms\t"
        "t_pipeline_pose_wash_ms\tt_pipeline_concat_ms\tt_pipeline_ind_ms\tt_pipeline_joint_ms\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
    _FRAME_TIMING_LINE_BUFFER.clear()


def _flush_line_buffer(path: str, buffer: list[str], *, force: bool = False) -> None:
    if not buffer:
        return
    if not force and len(buffer) < _LOG_IO_FLUSH_EVERY:
        return
    payload = "".join(buffer)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload)
    except OSError as exc:
        print(f"[OnlineLog] warn: append log failed ({path}): {exc}")
        return
    buffer.clear()


def flush_online_debug_logs(*, force: bool = False) -> None:
    _flush_line_buffer(step_timing_log_path(), _STEP_TIMING_LINE_BUFFER, force=force)
    _flush_line_buffer(o1_load_log_path(), _O1_LOAD_LINE_BUFFER, force=force)
    _flush_line_buffer(pose_wash_log_path(), _POSE_WASH_LINE_BUFFER, force=force)
    _flush_line_buffer(frame_timing_log_path(), _FRAME_TIMING_LINE_BUFFER, force=force)


def append_step_timing_line(
    *,
    gs_step: int,
    stage: str,
    optim_frame_id: int,
    window_j: int,
    i_right: int,
    gsm_mode: str,
    n_total: int,
    n_active: int,
    t_gsm_index_ms: float,
    t_render_ms: float,
    t_loss_ms: float,
    t_backward_ms: float,
    t_dc_accum_ms: float,
    t_opt_step_ms: float,
    t_zero_grad_ms: float,
    t_dc_prune_ms: float,
    t_total_ms: float,
) -> None:
    _STEP_TIMING_LINE_BUFFER.append(
        f"{gs_step}\t{stage}\t{optim_frame_id}\t{window_j}\t{i_right}\t{gsm_mode}\t"
        f"{n_total}\t{n_active}\t"
        f"{t_gsm_index_ms:.6f}\t{t_render_ms:.6f}\t{t_loss_ms:.6f}\t{t_backward_ms:.6f}\t"
        f"{t_dc_accum_ms:.6f}\t{t_opt_step_ms:.6f}\t{t_zero_grad_ms:.6f}\t{t_dc_prune_ms:.6f}\t"
        f"{t_total_ms:.6f}\n"
    )
    _flush_line_buffer(step_timing_log_path(), _STEP_TIMING_LINE_BUFFER)


def append_frame_timing_line(
    *,
    frame_id: int,
    logical_fid: int,
    window_type: str,
    stream_seed_num: float,
    n_total_begin: int,
    n_total_end: int,
    n_ind_steps: int,
    n_joint_steps: int,
    t_fetch_datum_ms: float,
    t_pre_stream_eval_ms: float,
    t_resolve_seed_ms: float,
    t_pipeline_ms: float,
    t_merge_ms: float,
    t_eval_save_ms: float,
    t_frame_wall_ms: float,
    t_pipeline_pose_wash_ms: float,
    t_pipeline_concat_ms: float,
    t_pipeline_ind_ms: float,
    t_pipeline_joint_ms: float,
) -> None:
    _FRAME_TIMING_LINE_BUFFER.append(
        f"{int(frame_id)}\t{int(logical_fid)}\t{str(window_type)}\t{float(stream_seed_num):.3f}\t"
        f"{int(n_total_begin)}\t{int(n_total_end)}\t{int(n_ind_steps)}\t{int(n_joint_steps)}\t"
        f"{float(t_fetch_datum_ms):.6f}\t{float(t_pre_stream_eval_ms):.6f}\t{float(t_resolve_seed_ms):.6f}\t{float(t_pipeline_ms):.6f}\t"
        f"{float(t_merge_ms):.6f}\t{float(t_eval_save_ms):.6f}\t{float(t_frame_wall_ms):.6f}\t"
        f"{float(t_pipeline_pose_wash_ms):.6f}\t{float(t_pipeline_concat_ms):.6f}\t"
        f"{float(t_pipeline_ind_ms):.6f}\t{float(t_pipeline_joint_ms):.6f}\n"
    )
    _flush_line_buffer(frame_timing_log_path(), _FRAME_TIMING_LINE_BUFFER)


def o1_load_init_frame(state: UltraGSOnlineState) -> None:
    state.outputs["o1_gs_steps"] = 0.0
    state.outputs["o1_sum_gsm_ms"] = 0.0
    state.outputs["o1_sum_render_ms"] = 0.0
    state.outputs["o1_sum_loss_ms"] = 0.0
    state.outputs["o1_sum_backward_ms"] = 0.0
    state.outputs["o1_sum_opt_step_ms"] = 0.0
    state.outputs["o1_sum_total_ms"] = 0.0
    state.outputs["o1_sum_active"] = 0.0
    state.outputs["o1_sum_active_ratio"] = 0.0
    state.outputs["o1_min_active"] = float("inf")
    state.outputs["o1_max_active"] = 0.0
    state.outputs["o1_min_total_ms"] = float("inf")
    state.outputs["o1_max_total_ms"] = 0.0
    state.outputs["o1_n_total_last"] = float(state.gauss_container.num_points)


def online_run_init(state: UltraGSOnlineState) -> None:
    state.outputs["run_frame_count"] = 0.0
    state.outputs["run_sum_frame_wall_ms"] = 0.0
    state.outputs["run_min_frame_wall_ms"] = float("inf")
    state.outputs["run_max_frame_wall_ms"] = 0.0
    state.outputs["run_sum_o1_t_total_ms_mean"] = 0.0
    state.outputs["run_sum_active_ratio"] = 0.0
    state.outputs["run_eval_count"] = 0.0
    state.outputs["run_pose_wash_count"] = 0.0
    state.outputs["run_pose_wash_sum_trans_improve_mm"] = 0.0
    state.outputs["run_pose_wash_sum_rot_improve_deg"] = 0.0
    state.outputs["run_sum_fetch_datum_ms"] = 0.0
    state.outputs["run_sum_resolve_seed_ms"] = 0.0
    state.outputs["run_sum_pipeline_ms"] = 0.0
    state.outputs["run_sum_merge_ms"] = 0.0
    state.outputs["run_sum_eval_save_ms"] = 0.0
    state.outputs["run_sum_pipeline_pose_wash_ms"] = 0.0
    state.outputs["run_sum_pipeline_concat_ms"] = 0.0
    state.outputs["run_sum_pipeline_ind_ms"] = 0.0
    state.outputs["run_sum_pipeline_joint_ms"] = 0.0


def o1_load_accumulate_step(
    state: UltraGSOnlineState,
    *,
    n_total: int,
    n_active: int,
    t_gsm_index_ms: float,
    t_render_ms: float,
    t_loss_ms: float,
    t_backward_ms: float,
    t_opt_step_ms: float,
    t_total_ms: float,
) -> None:
    state.outputs["o1_gs_steps"] = float(state.outputs.get("o1_gs_steps", 0.0) + 1.0)
    state.outputs["o1_sum_gsm_ms"] = float(state.outputs.get("o1_sum_gsm_ms", 0.0) + t_gsm_index_ms)
    state.outputs["o1_sum_render_ms"] = float(state.outputs.get("o1_sum_render_ms", 0.0) + t_render_ms)
    state.outputs["o1_sum_loss_ms"] = float(state.outputs.get("o1_sum_loss_ms", 0.0) + t_loss_ms)
    state.outputs["o1_sum_backward_ms"] = float(state.outputs.get("o1_sum_backward_ms", 0.0) + t_backward_ms)
    state.outputs["o1_sum_opt_step_ms"] = float(state.outputs.get("o1_sum_opt_step_ms", 0.0) + t_opt_step_ms)
    state.outputs["o1_sum_total_ms"] = float(state.outputs.get("o1_sum_total_ms", 0.0) + t_total_ms)
    state.outputs["o1_sum_active"] = float(state.outputs.get("o1_sum_active", 0.0) + float(n_active))
    ratio = float(n_active) / max(1.0, float(n_total))
    state.outputs["o1_sum_active_ratio"] = float(state.outputs.get("o1_sum_active_ratio", 0.0) + ratio)
    state.outputs["o1_min_active"] = float(min(state.outputs.get("o1_min_active", float("inf")), float(n_active)))
    state.outputs["o1_max_active"] = float(max(state.outputs.get("o1_max_active", 0.0), float(n_active)))
    state.outputs["o1_min_total_ms"] = float(
        min(state.outputs.get("o1_min_total_ms", float("inf")), float(t_total_ms))
    )
    state.outputs["o1_max_total_ms"] = float(max(state.outputs.get("o1_max_total_ms", 0.0), float(t_total_ms)))
    state.outputs["o1_n_total_last"] = float(n_total)


def _rotation_error_deg(p_est: torch.Tensor, p_ref: torch.Tensor) -> float:
    r = p_est[:3, :3] @ p_ref[:3, :3].transpose(0, 1)
    tr = torch.trace(r)
    cos_th = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos_th)).item())


def _translation_error_mm(p_est: torch.Tensor, p_ref: torch.Tensor) -> float:
    return float(torch.norm(p_est[:3, 3] - p_ref[:3, 3], p=2).item())


def _relative_improve_percent(coarse_err: float, improve_abs: float) -> float:
    """Relative improvement ratio in percent; positive is better."""
    # 对接近零的基线误差不报告夸张百分比，按 0% 处理。
    if abs(float(coarse_err)) < 1e-3:
        return 0.0
    denom = abs(float(coarse_err))
    return float(improve_abs) / denom * 100.0


def log_pose_wash_result(
    state: UltraGSOnlineState,
    *,
    frame_id: int,
    gs_step: int,
    pose_wash_steps: int,
    pose_wash_loss: float,
    pose_grad_norm_mean: float,
    delta_twist_norm_mean: float,
    p_gt: torch.Tensor,
    p_coarse: torch.Tensor,
    p_refined: torch.Tensor,
) -> None:
    coarse_trans = _translation_error_mm(p_coarse, p_gt)
    refined_trans = _translation_error_mm(p_refined, p_gt)
    coarse_rot = _rotation_error_deg(p_coarse, p_gt)
    refined_rot = _rotation_error_deg(p_refined, p_gt)
    trans_improve = coarse_trans - refined_trans
    rot_improve = coarse_rot - refined_rot
    trans_improve_pct = _relative_improve_percent(coarse_trans, trans_improve)
    rot_improve_pct = _relative_improve_percent(coarse_rot, rot_improve)

    state.outputs["pose_wash_last_coarse_trans_err_mm"] = coarse_trans
    state.outputs["pose_wash_last_refined_trans_err_mm"] = refined_trans
    state.outputs["pose_wash_last_trans_improve_mm"] = trans_improve
    state.outputs["pose_wash_last_coarse_rot_err_deg"] = coarse_rot
    state.outputs["pose_wash_last_refined_rot_err_deg"] = refined_rot
    state.outputs["pose_wash_last_rot_improve_deg"] = rot_improve
    state.outputs["run_pose_wash_count"] = float(state.outputs.get("run_pose_wash_count", 0.0) + 1.0)
    state.outputs["run_pose_wash_sum_trans_improve_mm"] = float(
        state.outputs.get("run_pose_wash_sum_trans_improve_mm", 0.0) + trans_improve
    )
    state.outputs["run_pose_wash_sum_rot_improve_deg"] = float(
        state.outputs.get("run_pose_wash_sum_rot_improve_deg", 0.0) + rot_improve
    )

    _POSE_WASH_LINE_BUFFER.append(
        f"{int(frame_id)}\t{int(gs_step)}\t{int(pose_wash_steps)}\t{float(pose_wash_loss):.6f}\t"
        f"{float(pose_grad_norm_mean):.6f}\t{float(delta_twist_norm_mean):.6f}\t"
        f"{coarse_trans:.6f}\t{refined_trans:.6f}\t{trans_improve:.6f}\t"
        f"{coarse_rot:.6f}\t{refined_rot:.6f}\t{rot_improve:.6f}\n"
    )
    _flush_line_buffer(pose_wash_log_path(), _POSE_WASH_LINE_BUFFER)

    print(
        f"[PoseWash] fid={int(frame_id)} step={int(gs_step)} n={int(pose_wash_steps)} "
        f"| grad={float(pose_grad_norm_mean):.3e} | dxi={float(delta_twist_norm_mean):.3e} "
        f"| trans(mm) {coarse_trans:.3f}->{refined_trans:.3f} "
        f"(d={trans_improve:+.3f}, rel={trans_improve_pct:+.1f}%) "
        f"| rot(deg) {coarse_rot:.3f}->{refined_rot:.3f} "
        f"(d={rot_improve:+.3f}, rel={rot_improve_pct:+.1f}%) "
        f"| loss={float(pose_wash_loss):.4f}"
    )


def _append_o1_load_line(state: UltraGSOnlineState) -> None:
    oc = state.model_cfg.online
    gsm_mode = str(getattr(oc.gsm, "mode", "off")).lower()
    frame_id = int(state.stream_frame_id)
    gs_steps = int(state.outputs.get("o1_gs_steps", 0.0))
    if gs_steps <= 0:
        return
    inv = 1.0 / float(gs_steps)
    n_total_last = int(state.outputs.get("o1_n_total_last", float(state.gauss_container.num_points)))
    n_active_mean = float(state.outputs.get("o1_sum_active", 0.0) * inv)
    n_active_min = int(state.outputs.get("o1_min_active", 0.0))
    n_active_max = int(state.outputs.get("o1_max_active", 0.0))
    n_active_ratio_mean = float(state.outputs.get("o1_sum_active_ratio", 0.0) * inv)
    t_gsm_index_ms_mean = float(state.outputs.get("o1_sum_gsm_ms", 0.0) * inv)
    t_render_ms_mean = float(state.outputs.get("o1_sum_render_ms", 0.0) * inv)
    t_loss_ms_mean = float(state.outputs.get("o1_sum_loss_ms", 0.0) * inv)
    t_backward_ms_mean = float(state.outputs.get("o1_sum_backward_ms", 0.0) * inv)
    t_opt_step_ms_mean = float(state.outputs.get("o1_sum_opt_step_ms", 0.0) * inv)
    t_total_ms_mean = float(state.outputs.get("o1_sum_total_ms", 0.0) * inv)
    t_total_ms_min = float(state.outputs.get("o1_min_total_ms", 0.0))
    t_total_ms_max = float(state.outputs.get("o1_max_total_ms", 0.0))
    _O1_LOAD_LINE_BUFFER.append(
        f"{frame_id}\t{gsm_mode}\t{gs_steps}\t"
        f"{n_total_last}\t{n_active_mean:.3f}\t{n_active_min}\t{n_active_max}\t{n_active_ratio_mean:.6f}\t"
        f"{t_gsm_index_ms_mean:.6f}\t{t_render_ms_mean:.6f}\t{t_loss_ms_mean:.6f}\t{t_backward_ms_mean:.6f}\t"
        f"{t_opt_step_ms_mean:.6f}\t{t_total_ms_mean:.6f}\t{t_total_ms_min:.6f}\t{t_total_ms_max:.6f}\n"
    )
    _flush_line_buffer(o1_load_log_path(), _O1_LOAD_LINE_BUFFER)


def atom_log_eval_save(state: UltraGSOnlineState, hooks: Dict[str, Callable[..., Any]]) -> None:
    oc = state.model_cfg.online
    frame_id = state.stream_frame_id
    global_step = state.iteration
    frame_avg_loss = float(state.outputs.get("frame_avg_loss", 0.0))
    merged_removed = float(state.outputs.get("merged_removed", 0))
    stream_seed_num = float(state.outputs.get("stream_seed_num", 0.0))
    gs_steps = float(state.outputs.get("o1_gs_steps", 0.0))
    inv_steps = 1.0 / gs_steps if gs_steps > 0 else 0.0
    o1_t_total_ms_mean = float(state.outputs.get("o1_sum_total_ms", 0.0) * inv_steps)
    o1_active_ratio_mean = float(state.outputs.get("o1_sum_active_ratio", 0.0) * inv_steps)
    frame_wall_ms = float(state.outputs.get("frame_wall_ms", 0.0))
    frame_fps = 1000.0 / frame_wall_ms if frame_wall_ms > 1e-6 else 0.0
    fetch_datum_ms = float(state.outputs.get("t_fetch_datum_ms", 0.0))
    pre_eval_ms = float(state.outputs.get("t_pre_stream_eval_ms", 0.0))
    resolve_seed_ms = float(state.outputs.get("t_resolve_seed_ms", 0.0))
    pipeline_ms = float(state.outputs.get("t_pipeline_ms", 0.0))
    merge_ms = float(state.outputs.get("t_merge_ms", 0.0))
    pipeline_pose_wash_ms = float(state.outputs.get("t_pipeline_pose_wash_ms", 0.0))
    pipeline_concat_ms = float(state.outputs.get("t_pipeline_concat_ms", 0.0))
    pipeline_ind_ms = float(state.outputs.get("t_pipeline_ind_ms", 0.0))
    pipeline_joint_ms = float(state.outputs.get("t_pipeline_joint_ms", 0.0))
    mode = str(oc.mode).upper()
    scalars = {
        "stream/frame_idx": float(frame_id),
        "stream/loss_avg": frame_avg_loss,
        "stream/num_gaussians": float(state.gauss_container.num_points),
        "stream/merged_removed": merged_removed,
        "stream/seed_num": stream_seed_num,
        "stream/gs_steps": gs_steps,
        "stream/o1_t_total_ms_mean": o1_t_total_ms_mean,
        "stream/o1_active_ratio_mean": o1_active_ratio_mean,
        "stream/frame_wall_ms": frame_wall_ms,
        "stream/frame_fps": frame_fps,
        "stream/t_fetch_datum_ms": fetch_datum_ms,
        "stream/t_pre_stream_eval_ms": pre_eval_ms,
        "stream/t_resolve_seed_ms": resolve_seed_ms,
        "stream/t_pipeline_ms": pipeline_ms,
        "stream/t_merge_ms": merge_ms,
        "stream/t_pipeline_pose_wash_ms": pipeline_pose_wash_ms,
        "stream/t_pipeline_concat_ms": pipeline_concat_ms,
        "stream/t_pipeline_ind_ms": pipeline_ind_ms,
        "stream/t_pipeline_joint_ms": pipeline_joint_ms,
    }
    if mode != "A":
        scalars["stream/pose_refine_loss"] = float(state.pose_carryover.pose_refine_loss)
        if "pose_wash_last_trans_improve_mm" in state.outputs:
            scalars["stream/pose_wash/trans_improve_mm"] = float(state.outputs["pose_wash_last_trans_improve_mm"])
            scalars["stream/pose_wash/rot_improve_deg"] = float(state.outputs["pose_wash_last_rot_improve_deg"])
            scalars["stream/pose_wash/refined_trans_err_mm"] = float(
                state.outputs["pose_wash_last_refined_trans_err_mm"]
            )
            scalars["stream/pose_wash/refined_rot_err_deg"] = float(
                state.outputs["pose_wash_last_refined_rot_err_deg"]
            )
    state.scalar_manager.log_scalars(scalars, global_step)
    summary_iv = int(getattr(oc, "console_frame_summary_interval", 10))
    if summary_iv > 0 and (state.stream_frame_count % summary_iv == 0):
        print(
            f"[UltraGS-Online] Frame {frame_id} | step={global_step} | "
            f"loss_avg={frame_avg_loss:.4f} | N={state.gauss_container.num_points}"
        )
    save_every = int(oc.save_every_frames)
    if save_every > 0 and (state.stream_frame_count % save_every == 0):
        hooks["save_checkpoint"](state, global_step)
    _append_o1_load_line(state)
    state.outputs["run_frame_count"] = float(state.outputs.get("run_frame_count", 0.0) + 1.0)
    state.outputs["run_sum_frame_wall_ms"] = float(state.outputs.get("run_sum_frame_wall_ms", 0.0) + frame_wall_ms)
    state.outputs["run_min_frame_wall_ms"] = float(
        min(state.outputs.get("run_min_frame_wall_ms", float("inf")), frame_wall_ms)
    )
    state.outputs["run_max_frame_wall_ms"] = float(
        max(state.outputs.get("run_max_frame_wall_ms", 0.0), frame_wall_ms)
    )
    state.outputs["run_sum_o1_t_total_ms_mean"] = float(
        state.outputs.get("run_sum_o1_t_total_ms_mean", 0.0) + o1_t_total_ms_mean
    )
    state.outputs["run_sum_active_ratio"] = float(
        state.outputs.get("run_sum_active_ratio", 0.0) + o1_active_ratio_mean
    )
    state.outputs["run_sum_fetch_datum_ms"] = float(
        state.outputs.get("run_sum_fetch_datum_ms", 0.0) + fetch_datum_ms
    )
    state.outputs["run_sum_resolve_seed_ms"] = float(
        state.outputs.get("run_sum_resolve_seed_ms", 0.0) + resolve_seed_ms
    )
    state.outputs["run_sum_pipeline_ms"] = float(state.outputs.get("run_sum_pipeline_ms", 0.0) + pipeline_ms)
    state.outputs["run_sum_merge_ms"] = float(state.outputs.get("run_sum_merge_ms", 0.0) + merge_ms)
    state.outputs["run_sum_pipeline_pose_wash_ms"] = float(
        state.outputs.get("run_sum_pipeline_pose_wash_ms", 0.0) + pipeline_pose_wash_ms
    )
    state.outputs["run_sum_pipeline_concat_ms"] = float(
        state.outputs.get("run_sum_pipeline_concat_ms", 0.0) + pipeline_concat_ms
    )
    state.outputs["run_sum_pipeline_ind_ms"] = float(
        state.outputs.get("run_sum_pipeline_ind_ms", 0.0) + pipeline_ind_ms
    )
    state.outputs["run_sum_pipeline_joint_ms"] = float(
        state.outputs.get("run_sum_pipeline_joint_ms", 0.0) + pipeline_joint_ms
    )


def write_online_run_summary(state: UltraGSOnlineState) -> None:
    cfg = state.cfg
    out_path = os.path.join(cfg.system.output_dir, "logs", "online_run_summary.json")
    n_frames = int(state.outputs.get("run_frame_count", 0.0))
    sum_wall = float(state.outputs.get("run_sum_frame_wall_ms", 0.0))
    mean_wall = (sum_wall / n_frames) if n_frames > 0 else 0.0
    mean_o1 = float(state.outputs.get("run_sum_o1_t_total_ms_mean", 0.0) / max(1, n_frames))
    mean_active_ratio = float(state.outputs.get("run_sum_active_ratio", 0.0) / max(1, n_frames))
    mean_fetch = float(state.outputs.get("run_sum_fetch_datum_ms", 0.0) / max(1, n_frames))
    mean_seed = float(state.outputs.get("run_sum_resolve_seed_ms", 0.0) / max(1, n_frames))
    mean_pipeline = float(state.outputs.get("run_sum_pipeline_ms", 0.0) / max(1, n_frames))
    mean_merge = float(state.outputs.get("run_sum_merge_ms", 0.0) / max(1, n_frames))
    mean_eval_save = float(state.outputs.get("run_sum_eval_save_ms", 0.0) / max(1, n_frames))
    mean_pipe_pose_wash = float(state.outputs.get("run_sum_pipeline_pose_wash_ms", 0.0) / max(1, n_frames))
    mean_pipe_concat = float(state.outputs.get("run_sum_pipeline_concat_ms", 0.0) / max(1, n_frames))
    mean_pipe_ind = float(state.outputs.get("run_sum_pipeline_ind_ms", 0.0) / max(1, n_frames))
    mean_pipe_joint = float(state.outputs.get("run_sum_pipeline_joint_ms", 0.0) / max(1, n_frames))
    n_pw = int(state.outputs.get("run_pose_wash_count", 0.0))
    mean_pw_trans_improve = float(state.outputs.get("run_pose_wash_sum_trans_improve_mm", 0.0) / max(1, n_pw))
    mean_pw_rot_improve = float(state.outputs.get("run_pose_wash_sum_rot_improve_deg", 0.0) / max(1, n_pw))
    summary = {
        "exp_tag": str(cfg.system.exp_tag),
        "output_dir": str(cfg.system.output_dir),
        "dataset_name": str(cfg.scene.dataset_name),
        "num_frames_processed": n_frames,
        "final_gs_step": int(state.iteration),
        "final_num_gaussians": int(state.gauss_container.num_points),
        "seed_every_frames": int(state.model_cfg.online.seed_every_frames),
        "num_new": int(cfg.scene.sampler.num_new),
        "total_opt_steps_cfg": int(state.model_cfg.online.lwm.total_opt_steps or cfg.engine.iterations),
        "window_size_frames": int(state.model_cfg.online.lwm.window_size_frames),
        "gsm_mode": str(state.model_cfg.online.gsm.mode),
        "frame_wall_ms_mean": float(mean_wall),
        "frame_wall_ms_min": float(state.outputs.get("run_min_frame_wall_ms", 0.0)),
        "frame_wall_ms_max": float(state.outputs.get("run_max_frame_wall_ms", 0.0)),
        "frame_fps_mean": float(1000.0 / mean_wall) if mean_wall > 1e-6 else 0.0,
        "o1_t_total_ms_mean": float(mean_o1),
        "o1_active_ratio_mean": float(mean_active_ratio),
        "frame_stage_ms_mean": {
            "fetch_datum": float(mean_fetch),
            "resolve_seed": float(mean_seed),
            "pipeline": float(mean_pipeline),
            "merge": float(mean_merge),
            "eval_save": float(mean_eval_save),
        },
        "pipeline_stage_ms_mean": {
            "pose_wash": float(mean_pipe_pose_wash),
            "concat": float(mean_pipe_concat),
            "independent": float(mean_pipe_ind),
            "joint": float(mean_pipe_joint),
        },
        "pose_wash_count": n_pw,
        "pose_wash_trans_improve_mm_mean": float(mean_pw_trans_improve),
        "pose_wash_rot_improve_deg_mean": float(mean_pw_rot_improve),
        "last_eval_psnr": float(state.outputs.get("last_eval_psnr", 0.0)),
        "last_eval_ssim": float(state.outputs.get("last_eval_ssim", 0.0)),
        "last_eval_l1": float(state.outputs.get("last_eval_l1", 0.0)),
        "num_eval_events": int(state.outputs.get("run_eval_count", 0.0)),
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        print(f"[UltraGS-Online] Run summary saved: {out_path}")
    except OSError as exc:
        print(f"[UltraGS-Online] warn: failed to write run summary ({out_path}): {exc}")
