"""Offline training with ``ours_renderer`` (world-mm Gaussians + GT pose)."""

import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from engine.computer import LossComputer, MetricComputer
from engine.registry import register_callback
from scene.factory import SceneFactory
from utils.eval_protocol import clear_vis_dir, frame_metric_row, should_save_vis, test_frame_count, write_per_frame_metrics
from utils.image_utils import make_diff

from .dc_diagnostics import compute_contribution_residual_stats
from .density_control import OursOfflineDensityController
from .gaussian_container import OursGaussContainer
from .gaussian_solver import OursGaussSolver
from .pose_optimization import OursPoseManager
from .render import render
from .state import OursOfflineState


def _console_log(state: OursOfflineState, message: str) -> None:
    """与 tqdm 共存且立即 flush（避免 pipe/tee 全缓冲导致日志滞后）。"""
    pbar = getattr(state.scalar_manager, "tqdm_bar", None)
    if pbar is not None and hasattr(pbar, "write"):
        pbar.write(message)
        fp = getattr(pbar, "fp", None)
        if fp is not None and hasattr(fp, "flush"):
            fp.flush()
        sys.stderr.flush()
    else:
        print(message, flush=True)


def _setup_pose_uncertainty_weights(state: OursOfflineState) -> None:
    pcfg = getattr(state.model_cfg, "pose_uncertainty", None)
    if pcfg is None or not bool(getattr(pcfg, "enabled", False)):
        state.pose_uncertainty_weights = None
        return
    path = os.path.abspath(os.path.expanduser(str(getattr(pcfg, "uncertainty_abspath", "") or "")))
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"pose_uncertainty.uncertainty_abspath not found: {path}")
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 1:
        raise ValueError(f"pose_uncertainty file must be 1D frame weights/scores, got {arr.shape}: {path}")
    n = len(state.scene.train_container)
    if int(arr.shape[0]) != n:
        raise ValueError(f"pose_uncertainty length {arr.shape[0]} != train frames {n}: {path}")
    vals = torch.as_tensor(arr, device=state.device, dtype=torch.float32)
    mode = str(getattr(pcfg, "weight_mode", "direct")).lower()
    w_min = float(getattr(pcfg, "min_weight", 0.1))
    w_max = float(getattr(pcfg, "max_weight", 1.0))
    if mode == "exp":
        alpha = float(getattr(pcfg, "alpha", 1.0))
        weights = w_min + (w_max - w_min) * torch.exp(-alpha * vals)
    elif mode == "direct":
        weights = vals
    else:
        raise ValueError(f"Unknown pose_uncertainty.weight_mode: {mode}")
    weights = weights.clamp(min=w_min, max=w_max).detach()
    state.pose_uncertainty_weights = weights
    print(
        "[OursUncertainty] Loaded frame weights:"
        f" path={path} min={float(weights.min()):.4f} max={float(weights.max()):.4f}"
        f" mean={float(weights.mean()):.4f}"
    )


def _apply_pose_uncertainty_weight(state: OursOfflineState, loss_dict, frame_id: int):
    weights = getattr(state, "pose_uncertainty_weights", None)
    if weights is None:
        return loss_dict, None
    pcfg = state.model_cfg.pose_uncertainty
    w = weights[int(frame_id)].to(device=state.device)
    names = set(str(x) for x in getattr(pcfg, "image_loss_names", ("l1", "ssim", "mse", "lpips")))
    total = loss_dict["total"]
    for name in names:
        if name not in loss_dict:
            continue
        old = loss_dict[name]
        new = old * w
        total = total - old + new
        loss_dict[name] = new
    loss_dict["total"] = total
    return loss_dict, float(w.detach().cpu().item())


def _euler_delta_matrix(rot_xyz: torch.Tensor, trans_xyz: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = rot_xyz.unbind(dim=0)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    z = torch.zeros((), device=rot_xyz.device, dtype=rot_xyz.dtype)
    o = torch.ones((), device=rot_xyz.device, dtype=rot_xyz.dtype)
    Rx = torch.stack(
        [
            torch.stack([o, z, z]),
            torch.stack([z, cx, -sx]),
            torch.stack([z, sx, cx]),
        ]
    )
    Ry = torch.stack(
        [
            torch.stack([cy, z, sy]),
            torch.stack([z, o, z]),
            torch.stack([-sy, z, cy]),
        ]
    )
    Rz = torch.stack(
        [
            torch.stack([cz, -sz, z]),
            torch.stack([sz, cz, z]),
            torch.stack([z, z, o]),
        ]
    )
    T = torch.eye(4, device=rot_xyz.device, dtype=rot_xyz.dtype)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = trans_xyz
    return T


def _sample_pose_marginalization_override(state: OursOfflineState, pose: torch.Tensor, sample_idx: int) -> torch.Tensor:
    if sample_idx == 0:
        return pose
    pcfg = state.model_cfg.pose_uncertainty
    trans_sigma = float(getattr(pcfg, "mc_trans_sigma_mm", 0.0))
    rot_sigma = math.radians(float(getattr(pcfg, "mc_rot_sigma_deg", 0.0)))
    if trans_sigma <= 0.0 and rot_sigma <= 0.0:
        return pose
    dtype = pose.dtype
    device = pose.device
    trans = torch.randn(3, device=device, dtype=dtype) * trans_sigma
    rot = torch.randn(3, device=device, dtype=dtype) * rot_sigma
    return pose @ _euler_delta_matrix(rot, trans)


def _render_with_optional_pose_mc(
    state: OursOfflineState,
    slice_meta,
    datum_index: int,
    collect_path_stats: bool = False,
):
    pcfg = getattr(state.model_cfg, "pose_uncertainty", None)
    if pcfg is None or not bool(getattr(pcfg, "mc_enabled", False)):
        return render(
            state.gauss_container.model_cfg,
            state.scene.dataset_meta,
            slice_meta,
            state.gauss_container,
            is_jitter=True,
            return_intermediates=False,
            collect_path_stats=collect_path_stats,
        )
    samples = max(1, int(getattr(pcfg, "mc_samples", 2)))
    preds = []
    depths = []
    path_contributions = []
    path_supports = []
    for sample_idx in range(samples):
        pose_override = _sample_pose_marginalization_override(state, slice_meta.pose, sample_idx)
        pred, depth, intermediates = render(
            state.gauss_container.model_cfg,
            state.scene.dataset_meta,
            slice_meta,
            state.gauss_container,
            is_jitter=True,
            return_intermediates=False,
            pose_override=pose_override,
            collect_path_stats=collect_path_stats,
        )
        preds.append(pred)
        depths.append(depth)
        if collect_path_stats:
            path_contributions.append(intermediates["path_contribution"])
            path_supports.append(intermediates["path_support"])
    if state.iteration % int(getattr(pcfg, "log_interval", 500)) == 0:
        _console_log(
            state,
            "[OursUncertainty] MC marginalization:"
            f" frame={int(datum_index)} samples={samples}"
            f" trans_sigma_mm={float(getattr(pcfg, 'mc_trans_sigma_mm', 0.0)):.4f}"
            f" rot_sigma_deg={float(getattr(pcfg, 'mc_rot_sigma_deg', 0.0)):.4f}",
        )
    intermediates = {}
    if collect_path_stats:
        intermediates = {
            "path_contribution": torch.stack(path_contributions, dim=0).mean(dim=0),
            "path_support": torch.stack(path_supports, dim=0).mean(dim=0),
        }
    return torch.stack(preds, dim=0).mean(dim=0), torch.stack(depths, dim=0).mean(dim=0), intermediates


def _should_log_scale_tail_diag(iteration: int) -> bool:
    return iteration % 1000 == 0


def _scale_tail_grad_stats(state: OursOfflineState):
    container = state.gauss_container
    if container is None or "scales" not in container._param_dict:
        return {}
    with torch.no_grad():
        scales = container.scales.detach()
        max_scale = scales.max(dim=1).values
        qs = torch.quantile(max_scale.float(), torch.tensor([0.99, 0.999, 0.9999], device=max_scale.device))
        stats = {
            "train/scale_tail/p99_mm": float(qs[0].item()),
            "train/scale_tail/p999_mm": float(qs[1].item()),
            "train/scale_tail/p9999_mm": float(qs[2].item()),
            "train/scale_tail/max_mm": float(max_scale.max().item()),
            "train/scale_tail/count_gt5": float((max_scale > 5.0).sum().item()),
            "train/scale_tail/count_gt10": float((max_scale > 10.0).sum().item()),
            "train/scale_tail/count_gt20": float((max_scale > 20.0).sum().item()),
        }
        n = int(max_scale.numel())
        if n == 0:
            return stats
        k = min(256, n)
        top_idx = torch.topk(max_scale, k=k, largest=True).indices
        stats["train/scale_tail/top256_scale_mean_mm"] = float(max_scale.index_select(0, top_idx).mean().item())
        for prop_name, metric_name in (("echo_alphas", "echo_alpha"), ("attenuations", "attenuation")):
            if prop_name in container._param_dict:
                values = getattr(container, prop_name).detach().reshape(-1)
                stats[f"train/scale_tail/top256_{metric_name}_mean"] = float(
                    values.index_select(0, top_idx).mean().item()
                )

    param = container._param_dict["scales"].tensor
    grad = param.grad
    if grad is None:
        return stats
    with torch.no_grad():
        grad_top = grad.detach().index_select(0, top_idx)
        raw_top = param.detach().index_select(0, top_idx)
        stats["train/scale_tail/top256_scale_grad_mean"] = float(grad_top.mean().item())
        stats["train/scale_tail/top256_scale_grad_abs_mean"] = float(grad_top.abs().mean().item())
        stats["train/scale_tail/top256_scale_grad_pos_frac"] = float((grad_top > 0.0).float().mean().item())
        stats["train/scale_tail/top256_raw_grad_dot_mean"] = float((raw_top * grad_top).mean().item())
    return stats


@register_callback("OursOfflineSetupCallback")
class OursOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: OursOfflineState):
        cfg = state.cfg
        print(f"[Ours] Setting up on device: {state.device}")
        state.scene = SceneFactory.build_scene(cfg, state.device)
        if bool(getattr(state.model_cfg.pose_optimization, "enabled", False)):
            state.pose_manager = OursPoseManager(cfg, state.device)
            state.pose_manager.setup_from_scene(state.scene)
        if state.scene.train_indices is not None:
            train_n = len(state.scene.train_indices)
            test_n = len(state.scene.test_indices or [])
            test_desc = (
                f"holdout_interval trim={cfg.scene.loading.holdout_trim_each}"
                f" samples={cfg.scene.loading.holdout_num_samples}"
            )
        else:
            train_n = len(state.scene.train_container)
            test_n = len(state.scene.test_container)
            test_desc = str(cfg.scene.loading.test_images_name)
        print(
            "[Ours] Offline data split:"
            f" train={train_n} ({cfg.scene.loading.images_name}),"
            f" test={test_n} ({test_desc})"
        )

        gi = state.model_cfg.gauss_init
        n_init = int(gi.num_points)
        if n_init > 0:
            state.cfg.scene.sampler.num_init = n_init
            print(f"[Ours] Gaussian init count from model.gauss_init.num_points: {n_init}")

        state.gauss_container = OursGaussContainer(cfg, device=state.device)
        init_data = state.scene.build_init_gaussians()
        state.gauss_container.init_from_scene(init_data["gaussians"])
        state.scene_radius = float(init_data.get("scene_radius", 0.0))

        with torch.no_grad():
            m = state.gauss_container.means
            print(
                "[Ours] Init Gaussians (world mm): "
                f"N={state.gauss_container.num_points} "
                f"means_min={m.min(dim=0).values.cpu().tolist()} "
                f"means_max={m.max(dim=0).values.cpu().tolist()} "
                f"scene_radius={state.scene_radius:.4f}"
            )

        state.gauss_solver = OursGaussSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()

        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]
        if state.pose_manager is not None:
            state.ckpt_items.append(("pose_manager", state.pose_manager))
        state.dc = OursOfflineDensityController(cfg)
        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)
        _setup_pose_uncertainty_weights(state)

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)
        print("[Ours] Setup done.")


@register_callback("OursOfflineTrainCallback")
class OursOfflineTrainCallback:
    priority = 50

    def on_train(self, state: OursOfflineState):
        cfg = state.cfg
        iterations = cfg.engine.iterations
        accum_steps = cfg.engine.accum_steps
        intv = cfg.engine.interval
        step_intv = intv.train.step
        print_intv = intv.train.print
        test_intv = intv.test
        save_intv = intv.save

        start_iter = state.iteration
        # 进度条走 stderr；训练/测试日志经 tqdm.write 即时刷出，stdout 可安全 tee
        pbar = tqdm(
            range(start_iter + 1, iterations + 1),
            initial=start_iter,
            total=iterations,
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        state.scalar_manager.tqdm_bar = pbar
        state.gauss_solver.zero_grad()
        if state.pose_manager is not None:
            state.pose_manager.zero_grad()

        for iteration in pbar:
            state.iteration = iteration
            state.gauss_solver.step_lr(iteration)

            datum = state.scene.next_train_datum()
            slice_meta = datum["slice_meta"]
            if state.pose_manager is not None:
                slice_meta = state.pose_manager.replace_slice_meta(slice_meta, iteration)
            collect_path_stats = state.dc.should_collect_path_stats(iteration)
            pred_image, depth_image, render_intermediates = _render_with_optional_pose_mc(
                state,
                slice_meta,
                int(datum["index"]),
                collect_path_stats=collect_path_stats,
            )
            gt_image = slice_meta.image

            state.inputs["gt_image"] = gt_image
            state.outputs["pred_image"] = pred_image
            state.outputs["depth_image"] = depth_image
            state.outputs["gauss_scales"] = state.gauss_container.scales
            state.outputs["gauss_opacities"] = state.gauss_container.opacities
            state.outputs["gauss_attenuations"] = state.gauss_container.attenuations
            loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
            uncertainty_weight = None
            if getattr(state, "pose_uncertainty_weights", None) is not None:
                loss_dict, uncertainty_weight = _apply_pose_uncertainty_weight(
                    state, loss_dict, int(datum["index"])
                )
            if state.pose_manager is not None:
                for name, pose_loss in state.pose_manager.loss_terms(int(datum["index"]), iteration).items():
                    loss_dict[name] = pose_loss
                    loss_dict["total"] = loss_dict["total"] + pose_loss
            total_loss = loss_dict["total"] / accum_steps
            total_loss.backward()

            if iteration % accum_steps == 0:
                pose_only = state.pose_manager is not None and state.pose_manager.is_pose_only(iteration)
                if not pose_only and state.dc.should_accumulate_grad_stats(iteration):
                    dc_grad_scale = 1.0
                    if (
                        uncertainty_weight is not None
                        and bool(getattr(state.model_cfg.pose_uncertainty, "decouple_density_control", True))
                    ):
                        dc_grad_scale = 1.0 / max(float(uncertainty_weight), 1.0e-6)
                    state.dc.update_grad_stats(state.gauss_container, grad_scale=dc_grad_scale)
                if not pose_only and state.dc.should_collect_responsibility_stats(iteration):
                    rcfg = state.model_cfg.offline.density_control.responsibility
                    resp_stats = compute_contribution_residual_stats(
                        state.gauss_container.model_cfg,
                        state.scene.dataset_meta,
                        slice_meta,
                        state.gauss_container,
                        pred_image,
                        gt_image,
                        float(getattr(rcfg, "high_residual_quantile", 0.90)),
                    )
                    state.dc.update_responsibility_stats(state.gauss_container, *resp_stats[:4])
                    if len(resp_stats) > 4:
                        state.dc.update_responsibility_diag_stats(resp_stats[4])
                if not pose_only and collect_path_stats:
                    state.dc.update_path_stats(
                        state.gauss_container,
                        render_intermediates["path_contribution"],
                        render_intermediates["path_support"],
                    )
                scale_tail_stats = None
                if not pose_only and _should_log_scale_tail_diag(iteration):
                    scale_tail_stats = _scale_tail_grad_stats(state)
                if not pose_only:
                    state.gauss_solver.step()
                if state.pose_manager is not None:
                    state.pose_manager.step(iteration)
                state.gauss_solver.zero_grad()
                if state.pose_manager is not None:
                    state.pose_manager.zero_grad()
                if not pose_only and state.dc.should_densify_prune(iteration):
                    state.dc.prune_and_densify(iteration, state.gauss_solver, state.gauss_container)

                if iteration % step_intv == 0:
                    lr = state.gauss_solver.optimizer.param_groups[0]["lr"]
                    num_g = state.gauss_container.num_points
                    stats = {"train/loss/total": loss_dict["total"].item()}
                    for k, v in loss_dict.items():
                        if k != "total":
                            stats[f"train/loss/{k}"] = v.item()
                    stats["train/lr"] = lr
                    stats["train/num_gaussians"] = num_g
                    stats.update(state.dc.scalar_stats(iteration))
                    if state.pose_manager is not None:
                        if iteration % int(state.model_cfg.pose_optimization.log_interval) == 0:
                            stats.update(
                                {
                                    f"train/{k}": v
                                    for k, v in state.pose_manager.metrics(iteration=iteration).items()
                                }
                            )
                    if uncertainty_weight is not None and iteration % int(state.model_cfg.pose_uncertainty.log_interval) == 0:
                        stats["train/pose_uncertainty_weight"] = uncertainty_weight
                    if scale_tail_stats is not None:
                        stats.update(scale_tail_stats)
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % print_intv == 0:
                        state.scalar_manager.print_train(iteration, in_place=False)

                if test_intv and iteration % test_intv == 0:
                    pbar.write("")
                    _run_test(state, iteration)
                if save_intv and iteration % save_intv == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("OursOfflineTestCallback")
class OursOfflineTestCallback:
    priority = 50

    def on_test(self, state: OursOfflineState):
        _run_test(state, iteration=None)


@register_callback("OursOfflineSaveCallback")
class OursOfflineSaveCallback:
    priority = 50

    def on_save(self, state: OursOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("OursOfflineFinishCallback")
class OursOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: OursOfflineState):
        state.scalar_manager.close()


def _save_checkpoint(state: OursOfflineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


def _run_test(state: OursOfflineState, iteration=None):
    cfg = state.cfg
    it_str = f"Iter {iteration}" if iteration is not None else "Eval"
    _console_log(state, f"[{it_str}] Ours evaluation — start (test frames)...")
    vis_dir = os.path.join(cfg.system.output_dir, "vis")
    clear_vis_dir(vis_dir)
    count = 0
    metric_sums = {}
    per_frame_rows = []
    test_total = test_frame_count(state.scene)

    with torch.no_grad():
        while True:
            datum = state.scene.next_test_datum()
            if datum is None:
                break
            slice_meta = datum["slice_meta"]
            if state.pose_manager is not None:
                slice_meta = state.pose_manager.replace_slice_meta(slice_meta, iteration or state.iteration)
            pred, _, _ = render(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                slice_meta,
                state.gauss_container,
                is_jitter=False,
                return_intermediates=False,
            )
            gt = slice_meta.image
            state.inputs["gt_image"] = gt
            state.outputs["pred_image"] = pred
            scores = state.metric_computer.compute(state.inputs, state.outputs)
            for k, v in scores.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            count += 1
            per_frame_rows.append(frame_metric_row(count, datum, scores))

            if should_save_vis(cfg, count, test_total):
                item_dir = os.path.join(vis_dir, f"item_{count}")
                pred_np = pred.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()
                diff = make_diff(pred_np, gt_np)
                state.io_manager.save_images(item_dir, pred, gt, diff)

                gt_f = np.clip(gt_np, 0.0, 1.0).astype(np.float32)
                pred_f = np.clip(pred_np, 0.0, 1.0).astype(np.float32)
                diff_f = diff.astype(np.float32) / 255.0
                compose = np.concatenate([gt_f, pred_f, diff_f], axis=1)
                compose_path = os.path.join(item_dir, "compose.png")
                state.io_manager.save_image(compose_path, compose)

    if count == 0:
        _console_log(state, f"[{it_str}] Ours evaluation — no test frames.")
        return
    metrics_log = {f"test/{k}": v / count for k, v in metric_sums.items()}
    metrics_display = {k: v / count for k, v in metric_sums.items()}
    if state.pose_manager is not None:
        pose_metrics = state.pose_manager.metrics(iteration=iteration or state.iteration)
        metrics_log.update({f"test/{k}": v for k, v in pose_metrics.items()})
    write_per_frame_metrics(cfg.system.output_dir, iteration or state.iteration, per_frame_rows, cfg)
    state.scalar_manager.log_scalars(metrics_log, iteration or state.iteration)
    state.scalar_manager.print_test(iteration, metrics_display, in_place=False)
    _console_log(state, f"[{it_str}] Ours evaluation — done ({count} frames).")
