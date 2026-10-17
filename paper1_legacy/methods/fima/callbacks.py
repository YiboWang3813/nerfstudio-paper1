from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from engine.registry import register_callback
from scene.factory import SceneFactory

from .configs import FiMAConfig
from .losses import heteroscedastic_pose_loss, pem_supervision_loss
from .pose_estimator import FiMAPoseEstimator
from .pose_estimator_adjacent import FiMAAdjacentPairEstimator
from .pose_geometry import (
    compose_world_from_rel,
    se3_from_6dof,
    six_dof_from_matrix,
    translation_rotation_errors,
)
from .pose_refiner import PoseRefiner
from .pose_wash import optimize_pose_refiner_batch
from .pose_wash_losses import PoseGtDeltaLossBatch
from .sequence_dataset import FiMASequenceSampler
from .state import FiMAOfflineState


def _mc(state: FiMAOfflineState) -> FiMAConfig:
    return state.cfg.model


def _reverse_window_edge_targets(target: torch.Tensor) -> torch.Tensor:
    mats = se3_from_6dof(target)
    inv = torch.linalg.inv(mats)
    return six_dof_from_matrix(torch.flip(inv, dims=[1]))


def _edge_cycle_loss(forward_pred: torch.Tensor, reverse_pred: torch.Tensor) -> torch.Tensor:
    reverse_aligned = torch.flip(reverse_pred, dims=[1])
    cycle = se3_from_6dof(forward_pred) @ se3_from_6dof(reverse_aligned)
    cycle_vec = six_dof_from_matrix(cycle)
    return cycle_vec[..., :3].abs().mean() + cycle_vec[..., 3:6].abs().mean()


def _load_state_dict(module: torch.nn.Module, path: str, device: torch.device) -> None:
    blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    module.load_state_dict(blob, strict=True)


def _pem_ckpt_items(state: FiMAOfflineState) -> List[Tuple[str, Any]]:
    return [("pose_estimator", state.pose_estimator), ("pem_optimizer", state.pem_optimizer)]


def _refiner_ckpt_items(state: FiMAOfflineState) -> List[Tuple[str, Any]]:
    return [
        ("pose_refiner", state.pose_refiner),
        ("pose_refiner_optimizer", state.pose_refiner_optimizer),
    ]


def _initialize_pem_head_from_target_stats(state: FiMAOfflineState) -> None:
    head = getattr(state.pose_estimator, "head", None)
    mean = getattr(state, "pem_target_mean", None)
    if head is None or mean is None or not hasattr(head, "weight") or not hasattr(head, "bias"):
        return
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(mean.to(device=head.bias.device, dtype=head.bias.dtype))
    print("[FiMA] PEM head initialized to target mean for stable standardized training.")


@register_callback("FiMAOfflineSetupCallback")
class FiMAOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: FiMAOfflineState):
        cfg = state.cfg
        mc = _mc(state)
        state.model_cfg = mc
        print(f"[FiMA] device={state.device} component={mc.fima_component}")

        state.scene = SceneFactory.build_scene(cfg, state.device)
        tr = getattr(state.scene, "train_indices", None)
        te = getattr(state.scene, "test_indices", None)
        split_protocol = str(getattr(mc.sequence, "split_protocol", "auto")).lower()
        use_holdout = (
            split_protocol in ("auto", "scene_holdout")
            and bool(getattr(mc.sequence, "respect_scene_holdout", True))
            and tr is not None
            and te is not None
        )
        state.sequence_sampler = FiMASequenceSampler(
            state.scene.train_container,
            mc,
            state.device,
            train_global_indices=tr if use_holdout else None,
            test_global_indices=te if use_holdout else None,
        )
        if use_holdout:
            print(f"[FiMA] Sequence split aligned with Scene holdout: train={len(tr)}, test={len(te)}")
        else:
            desc = getattr(state.sequence_sampler, "_split_description", split_protocol)
            print(
                f"[FiMA] Sequence split protocol={desc}:"
                f" train={len(state.sequence_sampler.train_indices)},"
                f" val={len(state.sequence_sampler.val_indices)}"
            )
        seed = int(cfg.system.seed) if cfg.system.seed is not None else 0
        state.rng_np = np.random.RandomState(seed)

        if mc.fima_component == "pem":
            if str(getattr(mc.pem, "input_mode", "multiframe")).lower() == "adjacent_pair":
                state.pose_estimator = FiMAAdjacentPairEstimator(mc).to(state.device)
                print("[FiMA] PEM: adjacent_pair (two-frame) estimator")
            else:
                state.pose_estimator = FiMAPoseEstimator(mc).to(state.device)
                backend = getattr(getattr(state.pose_estimator, "mixer", None), "backend_name", "unknown")
                print(f"[FiMA] PEM: multiframe estimator layout={mc.pem.pem_output_layout} mixer={backend}")
            if bool(getattr(mc.pem_loss, "standardize_targets", False)):
                mean, std = state.sequence_sampler.target_stats(str(getattr(mc.pem, "pem_output_layout", "last_frame_rel0")))
                state.pem_target_mean = mean.detach()
                state.pem_target_std = std.detach()
                print(
                    "[FiMA] PEM target standardization:"
                    f" mean={state.pem_target_mean.detach().cpu().tolist()}"
                    f" std={state.pem_target_std.detach().cpu().tolist()}"
                )
                _initialize_pem_head_from_target_stats(state)
            state.pem_optimizer = torch.optim.AdamW(
                state.pose_estimator.parameters(),
                lr=float(mc.pem.lr),
                weight_decay=float(mc.pem.weight_decay),
            )
            decay_steps = max(1, int(getattr(mc.pem, "lr_decay_steps", 0) or 0))
            if decay_steps > 0:
                state.pem_scheduler = torch.optim.lr_scheduler.StepLR(
                    state.pem_optimizer,
                    step_size=decay_steps,
                    gamma=float(getattr(mc.pem, "lr_decay_gamma", 0.5)),
                )
            else:
                state.pem_scheduler = None
            state.ckpt_items = _pem_ckpt_items(state)
            state.pose_refiner = None
            state.pose_refiner_optimizer = None
        else:
            if str(getattr(mc.pem, "input_mode", "multiframe")).lower() == "adjacent_pair":
                state.pose_estimator = FiMAAdjacentPairEstimator(mc).to(state.device)
            else:
                state.pose_estimator = FiMAPoseEstimator(mc).to(state.device)
            pem_path = str(mc.pem_checkpoint_path or "").strip()
            if pem_path:
                if not os.path.isfile(pem_path):
                    raise FileNotFoundError(f"pem_checkpoint_path not found: {pem_path}")
                _load_state_dict(state.pose_estimator, pem_path, state.device)
                print(f"[FiMA] Loaded PEM weights from {pem_path}")
            state.pose_estimator.eval()
            for p in state.pose_estimator.parameters():
                p.requires_grad_(False)

            pr = mc.pose_refiner
            state.pose_refiner = PoseRefiner(
                pose_dim=int(pr.pose_dim),
                num_frequencies=int(pr.num_frequencies),
                encoder_hidden_dim=int(pr.encoder_hidden_dim),
                encoder_layers=int(pr.encoder_layers),
                bottleneck_dim=int(pr.bottleneck_dim),
                decoder_hidden_dim=int(pr.decoder_hidden_dim),
                decoder_layers=int(pr.decoder_layers),
                max_delta_rot_deg=float(pr.max_delta_rot_deg),
                max_delta_trans_mm=float(pr.max_delta_trans_mm),
            ).to(state.device)
            state.pose_refiner_optimizer = torch.optim.Adam(
                state.pose_refiner.parameters(),
                lr=float(pr.lr),
            )
            state.ckpt_items = _refiner_ckpt_items(state)
            state.pem_optimizer = None

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)

        ref_path = str(mc.refiner_checkpoint_path or "").strip()
        if mc.fima_component == "refiner" and ref_path and os.path.isfile(ref_path):
            _load_state_dict(state.pose_refiner, ref_path, state.device)
            print(f"[FiMA] Loaded PoseRefiner weights from {ref_path}")

        if mc.fima_component == "refiner" and str(getattr(mc.pem, "input_mode", "multiframe")).lower() == "adjacent_pair":
            raise NotImplementedError(
                "FiMA refiner stage with pem.input_mode=adjacent_pair is not wired "
                "(PEM output is inter-frame T, not rel-P0). Use multiframe PEM or extend refiner compose."
            )

        print("[FiMA] Setup done.")


@register_callback("FiMAOfflineTrainCallback")
class FiMAOfflineTrainCallback:
    priority = 50

    def on_train(self, state: FiMAOfflineState):
        mc = _mc(state)
        if mc.fima_component == "pem":
            _train_pem(state)
        else:
            _train_refiner(state)


@register_callback("FiMAOfflineTestCallback")
class FiMAOfflineTestCallback:
    priority = 50

    def on_test(self, state: FiMAOfflineState):
        mc = _mc(state)
        if mc.fima_component == "pem":
            _eval_pem(state, use_val=True)
        else:
            _eval_refiner(state, use_val=True)


@register_callback("FiMAOfflineSaveCallback")
class FiMAOfflineSaveCallback:
    priority = 50

    def on_save(self, state: FiMAOfflineState, iteration: int):
        state.io_manager.save_checkpoint(state.ckpt_items, iteration)


@register_callback("FiMAOfflineFinishCallback")
class FiMAOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: FiMAOfflineState):
        state.scalar_manager.close()


def _train_pem(state: FiMAOfflineState):
    cfg = state.cfg
    mc = _mc(state)
    pem = mc.pem
    iterations = int(cfg.engine.iterations)
    accum_steps = max(1, int(cfg.engine.accum_steps))
    intv = cfg.engine.interval
    bs = max(1, int(pem.batch_size))

    state.pose_estimator.train()
    pbar = tqdm(range(state.iteration + 1, iterations + 1), initial=state.iteration, total=iterations)
    state.scalar_manager.tqdm_bar = pbar
    state.pem_optimizer.zero_grad(set_to_none=True)

    for iteration in pbar:
        state.iteration = iteration

        batch = state.sequence_sampler.sample_train_batch(bs, state.rng_np)
        images = batch.images
        pred_pack = state.pose_estimator(images, batch.valid_window)

        if str(getattr(pem, "input_mode", "multiframe")).lower() == "adjacent_pair":
            pred = pred_pack["pred_pose_6"]
            target = batch.T_gt_vec_last
        elif pem.pem_output_layout == "last_frame_rel0":
            pred = pred_pack["pred_pose_6"]
            target = batch.T_gt_vec_last
        elif pem.pem_output_layout == "window_edges":
            pred = pred_pack["pred_pose_6"]
            target = batch.T_gt_vec_edges
        else:
            pred = pred_pack["pred_pose_6"]
            target = batch.T_gt_vec_window

        loss, parts = pem_supervision_loss(
            pred,
            target,
            mc.pem_loss,
            target_mean=state.pem_target_mean,
            target_std=state.pem_target_std,
        )
        if (
            str(getattr(pem, "input_mode", "multiframe")).lower() != "adjacent_pair"
            and str(getattr(pem, "pem_output_layout", "")) == "window_edges"
            and bool(getattr(mc.pem_loss, "use_bidirectional_edges", False))
        ):
            rev_images = torch.flip(images, dims=[1])
            rev_pred = state.pose_estimator(rev_images, torch.flip(batch.valid_window, dims=[1]))["pred_pose_6"]
            rev_target = _reverse_window_edge_targets(target)
            rev_loss, rev_parts = pem_supervision_loss(
                rev_pred,
                rev_target,
                mc.pem_loss,
                target_mean=state.pem_target_mean,
                target_std=state.pem_target_std,
            )
            loss = loss + float(getattr(mc.pem_loss, "lambda_reverse", 1.0)) * rev_loss
            for key, value in rev_parts.items():
                if key != "total":
                    parts[f"reverse_{key}"] = value.detach()
            parts["reverse_total"] = rev_loss.detach()
            if bool(getattr(mc.pem_loss, "use_edge_cycle", False)):
                cycle_loss = _edge_cycle_loss(pred, rev_pred)
                loss = loss + float(getattr(mc.pem_loss, "lambda_cycle", 0.1)) * cycle_loss
                parts["cycle"] = cycle_loss.detach()
            parts["total"] = loss
        if bool(getattr(mc.pem_loss, "use_heteroscedastic_nll", False)):
            logvar = pred_pack.get("pred_logvar_2")
            if logvar is None:
                raise RuntimeError(
                    "model.pem_loss.use_heteroscedastic_nll=true requires "
                    "model.pem.predict_uncertainty=true and pred_logvar_2 output."
                )
            hetero_loss, hetero_parts = heteroscedastic_pose_loss(
                pred,
                logvar,
                target,
                mc.pem_loss,
                target_mean=state.pem_target_mean,
                target_std=state.pem_target_std,
                logvar_min=float(getattr(mc.pem, "uncertainty_logvar_min", -8.0)),
                logvar_max=float(getattr(mc.pem, "uncertainty_logvar_max", 4.0)),
            )
            loss = loss + hetero_loss
            for key, value in hetero_parts.items():
                parts[f"hetero_{key}"] = value.detach()
            parts["total"] = loss
        (loss / accum_steps).backward()

        if iteration % accum_steps == 0:
            state.pem_optimizer.step()
            if getattr(state, "pem_scheduler", None) is not None:
                state.pem_scheduler.step()
            state.pem_optimizer.zero_grad(set_to_none=True)

            if iteration % intv.train.step == 0:
                stats: Dict[str, float] = {"train/pem/total": float(parts["total"].item())}
                for k, v in parts.items():
                    if k != "total":
                        stats[f"train/pem/{k}"] = float(v.item())
                stats["train/lr"] = float(state.pem_optimizer.param_groups[0]["lr"])
                state.scalar_manager.log_scalars(stats, iteration)
                if iteration % intv.train.print == 0:
                    state.scalar_manager.print_train(iteration)

            if intv.test and iteration % intv.test == 0:
                _eval_pem(state, use_val=True, global_step=iteration)
            if intv.save and iteration % intv.save == 0:
                state.io_manager.save_checkpoint(state.ckpt_items, iteration)

        if state.stop_training:
            break


def _infer_global_edge_predictions(
    state: FiMAOfflineState, chunk: int = 32
) -> Dict[int, torch.Tensor]:
    """对每条全局边 ``(k,k+1)`` 跑一次 PEM，得到 ``T_k``（``R_{k+1}=R_k T_k``）。"""
    sampler = state.sequence_sampler
    n = sampler.n_frames
    device = state.device
    net = state.pose_estimator
    edge_pred: Dict[int, torch.Tensor] = {}
    buf: List[Tuple[int, int]] = []
    for k in range(n - 1):
        buf.append((k, k + 1))
        if len(buf) >= chunk or k == n - 2:
            b = sampler.collate_adjacent_edges(buf)
            out = net(b.images, b.valid_window)
            pred = out["pred_pose_6"]
            for bi, (a, _) in enumerate(buf):
                edge_pred[int(a)] = se3_from_6dof(pred[bi])
            buf.clear()
    return edge_pred


def _pem_chain_eval_anchor(state: FiMAOfflineState) -> torch.Tensor:
    scene_cfg = getattr(state.cfg, "scene", None)
    loading_cfg = getattr(scene_cfg, "loading", None)
    pose_type = str(getattr(loading_cfg, "pose_type", "absolute")).lower()
    if pose_type == "relative":
        return torch.eye(4, device=state.device, dtype=torch.float32)
    return state.scene.train_container.get(0).pose.to(device=state.device, dtype=torch.float32)


def _infer_multiframe_window_edge_predictions(state: FiMAOfflineState) -> Dict[int, torch.Tensor]:
    """
    ReMamba-style sequence PEM: one contiguous window predicts all local adjacent
    edges. For every global edge, choose a deterministic window containing it.
    """
    sampler = state.sequence_sampler
    n = sampler.n_frames
    window = int(getattr(sampler, "window", getattr(state.cfg.model.pem, "window_size", 2)))
    net = state.pose_estimator
    aggregation = str(getattr(state.cfg.model.pem, "infer_window_edge_aggregation", "single")).lower()
    if aggregation not in ("single", "mean"):
        raise ValueError(f"Unsupported infer_window_edge_aggregation={aggregation!r}; expected single|mean")
    edge_pred: Dict[int, torch.Tensor] = {}
    if aggregation == "mean":
        accum: Dict[int, List[torch.Tensor]] = {}
        chunk = max(1, int(getattr(state.cfg.model.pem, "batch_size", 1)))
        with torch.no_grad():
            for s in range(0, n, chunk):
                ends = list(range(s, min(n, s + chunk)))
                batch = sampler._collate_ends(ends)
                out = net(batch.images, batch.valid_window)
                pred = out["pred_pose_6"]
                for bi, end in enumerate(ends):
                    start = int(end) - window + 1
                    valid = batch.valid_window[bi]
                    for local_edge in range(window - 1):
                        k = start + local_edge
                        if k < 0 or k >= n - 1:
                            continue
                        if not (bool(valid[local_edge]) and bool(valid[local_edge + 1])):
                            continue
                        accum.setdefault(int(k), []).append(pred[bi, local_edge].detach())
        missing = [k for k in range(n - 1) if k not in accum]
        if missing:
            raise RuntimeError(
                f"infer_window_edge_aggregation=mean missed {len(missing)} edges; "
                f"first missing={missing[:5]}"
            )
        for k, values in accum.items():
            edge_pred[int(k)] = se3_from_6dof(torch.stack(values, dim=0).mean(dim=0))
        return edge_pred

    with torch.no_grad():
        for k in range(n - 1):
            end = min(n - 1, max(window - 1, k + 1))
            start = end - window + 1
            local_edge = k - start
            if local_edge < 0 or local_edge >= window - 1:
                raise RuntimeError(
                    f"invalid local edge for global edge {k}: start={start}, end={end}, window={window}"
                )
            batch = sampler._collate_ends([end])
            out = net(batch.images, batch.valid_window)
            pred = out["pred_pose_6"].squeeze(0)
            edge_pred[int(k)] = se3_from_6dof(pred[local_edge])
    return edge_pred


def _merge_phase_a_window_edge_predictions(state: FiMAOfflineState) -> Dict[int, torch.Tensor]:
    """各训练窗内独立前向，重叠边上的 6D 预测取平均后再 ``se3``。"""
    sampler = state.sequence_sampler
    device = state.device
    net = state.pose_estimator
    accum: Dict[int, List[torch.Tensor]] = {}
    wins = getattr(sampler, "_train_global_windows", [])
    with torch.no_grad():
        for win in wins:
            for j in range(len(win) - 1):
                a, b = int(win[j]), int(win[j + 1])
                if b != a + 1:
                    continue
                bch = sampler.collate_adjacent_edges([(a, b)])
                out = net(bch.images, bch.valid_window)
                v = out["pred_pose_6"].squeeze(0)
                accum.setdefault(a, []).append(v.detach())
    edge_pred: Dict[int, torch.Tensor] = {}
    for kk, lst in accum.items():
        if not lst:
            continue
        vm = torch.stack(lst, dim=0).mean(dim=0)
        edge_pred[int(kk)] = se3_from_6dof(vm)
    for k in range(sampler.n_frames - 1):
        if k in edge_pred:
            continue
        bch = sampler.collate_adjacent_edges([(k, k + 1)])
        out = net(bch.images, bch.valid_window)
        edge_pred[k] = se3_from_6dof(out["pred_pose_6"].squeeze(0))
    return edge_pred


def _eval_pem_adjacent_chain(
    state: FiMAOfflineState,
    *,
    use_val: bool,
    global_step: int | None,
    merge_windows: bool,
) -> None:
    """阶段 A：从第 0 帧链式乘到各 test 帧，与 GT ``R_t``（``SliceMeta.pose``）比 SE(3) 误差。"""
    mc = _mc(state)
    device = state.device
    n = state.sequence_sampler.n_frames
    anchor = _pem_chain_eval_anchor(state)

    with torch.no_grad():
        input_mode = str(getattr(mc.pem, "input_mode", "multiframe")).lower()
        if input_mode != "adjacent_pair" and str(getattr(mc.pem, "pem_output_layout", "")) == "window_edges":
            edge_pred = _infer_multiframe_window_edge_predictions(state)
        elif merge_windows:
            edge_pred = _merge_phase_a_window_edge_predictions(state)
        else:
            edge_pred = _infer_global_edge_predictions(state)

        pool = sorted(state.sequence_sampler.all_eval_frames(use_val=use_val))
        trans_errs: List[float] = []
        rot_errs: List[float] = []
        for t in pool:
            tt = int(t)
            R_pred = anchor.clone()
            ok = True
            for k in range(0, tt):
                Tk = edge_pred.get(k)
                if Tk is None:
                    ok = False
                    break
                R_pred = R_pred @ Tk
            if not ok:
                continue
            R_gt = state.scene.train_container.get(tt).pose.to(device)
            dt, da = translation_rotation_errors(R_pred.unsqueeze(0), R_gt.unsqueeze(0))
            trans_errs.append(float(dt.item()))
            rot_errs.append(float(da.item()))

    if not trans_errs:
        print("[FiMA][eval_pem:chain] No test frames evaluated.")
        return

    tag = "val" if use_val else "full"
    step = global_step if global_step is not None else state.iteration
    mode = "window_merge" if merge_windows else "full_chain"
    metrics = {
        f"eval/pem_chain/{tag}/trans_rmse_mm": float(np.mean(trans_errs)),
        f"eval/pem_chain/{tag}/rot_mean_rad": float(np.mean(rot_errs)),
    }
    state.scalar_manager.log_scalars(metrics, step)
    print(
        f"[FiMA][eval_pem:chain:{tag}:{mode}] trans_rmse_mm={metrics[f'eval/pem_chain/{tag}/trans_rmse_mm']:.4f} "
        f"rot_mean_rad={metrics[f'eval/pem_chain/{tag}/rot_mean_rad']:.4f}"
    )


def _eval_pem(state: FiMAOfflineState, use_val: bool = True, global_step: int | None = None):
    mc = _mc(state)
    pem = mc.pem
    state.pose_estimator.eval()
    if str(getattr(pem, "input_mode", "multiframe")).lower() != "adjacent_pair" and pem.pem_output_layout == "window_edges":
        _eval_pem_adjacent_chain(
            state,
            use_val=use_val,
            global_step=global_step,
            merge_windows=False,
        )
        return
    if str(getattr(pem, "input_mode", "multiframe")).lower() == "adjacent_pair":
        merge = str(getattr(pem, "train_protocol", "standard")).lower() == "phase_a_local_window"
        _eval_pem_adjacent_chain(state, use_val=use_val, global_step=global_step, merge_windows=merge)
        return

    frames = state.sequence_sampler.all_eval_frames(use_val=use_val)
    if not frames:
        print("[FiMA][eval_pem] No frames to evaluate.")
        return

    trans_errs = []
    rot_errs = []
    with torch.no_grad():
        for frame_end in frames:
            batch = state.sequence_sampler._collate_ends([frame_end])
            images = batch.images
            out = state.pose_estimator(images, batch.valid_window)
            pred_vec = out["pred_pose_6"]
            if pem.pem_output_layout != "last_frame_rel0":
                pred_vec = pred_vec[:, -1, :]
            T_pred = se3_from_6dof(pred_vec.squeeze(0))
            T_gt = se3_from_6dof(batch.T_gt_vec_last.squeeze(0))
            dt, da = translation_rotation_errors(T_pred.unsqueeze(0), T_gt.unsqueeze(0))
            trans_errs.append(float(dt.item()))
            rot_errs.append(float(da.item()))

    tag = "val" if use_val else "full"
    step = global_step if global_step is not None else state.iteration
    metrics = {
        f"eval/pem/{tag}/trans_rmse_mm": float(np.mean(trans_errs)),
        f"eval/pem/{tag}/rot_mean_rad": float(np.mean(rot_errs)),
    }
    state.scalar_manager.log_scalars(metrics, step)
    print(f"[FiMA][eval_pem:{tag}] trans_rmse_mm={metrics[f'eval/pem/{tag}/trans_rmse_mm']:.4f} "
          f"rot_mean_rad={metrics[f'eval/pem/{tag}/rot_mean_rad']:.4f}")


def _train_refiner(state: FiMAOfflineState):
    cfg = state.cfg
    mc = _mc(state)
    iterations = int(cfg.engine.iterations)
    intv = cfg.engine.interval
    bs = max(1, int(mc.pem.batch_size))
    wash = mc.pose_wash
    loss_fn = PoseGtDeltaLossBatch(
        rot_weight=float(mc.pose_gt_delta.rot_weight),
        trans_weight=float(mc.pose_gt_delta.trans_weight),
    ).to(state.device)

    state.pose_refiner.train()
    pbar = tqdm(range(state.iteration + 1, iterations + 1), initial=state.iteration, total=iterations)
    state.scalar_manager.tqdm_bar = pbar

    for iteration in pbar:
        state.iteration = iteration
        batch = state.sequence_sampler.sample_train_batch(bs, state.rng_np)
        with torch.no_grad():
            pem_out = state.pose_estimator(batch.images, batch.valid_window)
            pred_vec = pem_out["pred_pose_6"]
            if str(getattr(mc.pem, "input_mode", "multiframe")).lower() != "adjacent_pair":
                if mc.pem.pem_output_layout != "last_frame_rel0":
                    pred_vec = pred_vec[:, -1, :]
            T_hat = se3_from_6dof(pred_vec)
            p_coarse = compose_world_from_rel(T_hat, batch.P0)
            p_gt = torch.stack(
                [state.scene.container.get(int(i)).pose.to(state.device) for i in batch.frame_idx_end.tolist()],
                dim=0,
            )

        inner_steps = max(1, int(wash.num_pose_wash_steps))
        p_ref, inner_mean = optimize_pose_refiner_batch(
            state.pose_refiner,
            state.pose_refiner_optimizer,
            loss_fn,
            p_coarse,
            p_gt,
            inner_steps,
        )

        if iteration % intv.train.step == 0:
            with torch.no_grad():
                dt_c, _ = translation_rotation_errors(p_coarse, p_gt)
                loss_ref = float(loss_fn(p_coarse, p_ref.detach(), p_gt).item())
            stats = {
                "train/refiner/inner_loss": inner_mean,
                "train/refiner/coarse_trans_rmse_mm_batch": float(dt_c.mean().item()),
                "train/refiner/pose_gt_delta_after_inner": loss_ref,
                "train/lr": float(state.pose_refiner_optimizer.param_groups[0]["lr"]),
            }
            state.scalar_manager.log_scalars(stats, iteration)
            if iteration % intv.train.print == 0:
                state.scalar_manager.print_train(iteration)

        if intv.test and iteration % intv.test == 0:
            _eval_refiner(state, use_val=True, global_step=iteration)
        if intv.save and iteration % intv.save == 0:
            state.io_manager.save_checkpoint(state.ckpt_items, iteration)

        if state.stop_training:
            break


def _eval_refiner(state: FiMAOfflineState, use_val: bool = True, global_step: int | None = None):
    mc = _mc(state)
    pem = mc.pem

    state.pose_estimator.eval()
    state.pose_refiner.eval()
    frames = state.sequence_sampler.all_eval_frames(use_val=use_val)
    if not frames:
        print("[FiMA][eval_refiner] No frames.")
        return

    trans_coarse = []
    trans_ref = []
    rot_coarse = []
    rot_ref = []

    with torch.no_grad():
        for frame_end in frames:
            batch = state.sequence_sampler._collate_ends([frame_end])
            pem_out = state.pose_estimator(batch.images, batch.valid_window)
            pred_vec = pem_out["pred_pose_6"]
            if str(getattr(pem, "input_mode", "multiframe")).lower() != "adjacent_pair":
                if pem.pem_output_layout != "last_frame_rel0":
                    pred_vec = pred_vec[:, -1, :]
            T_hat = se3_from_6dof(pred_vec.squeeze(0))
            p_coarse = compose_world_from_rel(T_hat.unsqueeze(0), batch.P0)[0]
            sm = state.scene.container.get(int(frame_end))
            p_gt = sm.pose.to(state.device)

            # 评测只做单次前向，避免在 eval 中改写 Refiner 权重
            p_ref = state.pose_refiner(p_coarse.unsqueeze(0)).squeeze(0)

            dt_c, da_c = translation_rotation_errors(p_coarse.unsqueeze(0), p_gt.unsqueeze(0))
            dt_r, da_r = translation_rotation_errors(p_ref.unsqueeze(0), p_gt.unsqueeze(0))
            trans_coarse.append(float(dt_c.item()))
            rot_coarse.append(float(da_c.item()))
            trans_ref.append(float(dt_r.item()))
            rot_ref.append(float(da_r.item()))

    tag = "val" if use_val else "full"
    step = global_step if global_step is not None else state.iteration
    metrics = {
        f"eval/refiner/{tag}/coarse_trans_rmse_mm": float(np.mean(trans_coarse)),
        f"eval/refiner/{tag}/coarse_rot_mean_rad": float(np.mean(rot_coarse)),
        f"eval/refiner/{tag}/refined_trans_rmse_mm": float(np.mean(trans_ref)),
        f"eval/refiner/{tag}/refined_rot_mean_rad": float(np.mean(rot_ref)),
    }
    state.scalar_manager.log_scalars(metrics, step)
    print(
        f"[FiMA][eval_refiner:{tag}] coarse trans={metrics[f'eval/refiner/{tag}/coarse_trans_rmse_mm']:.4f} "
        f"ref trans={metrics[f'eval/refiner/{tag}/refined_trans_rmse_mm']:.4f}"
    )
