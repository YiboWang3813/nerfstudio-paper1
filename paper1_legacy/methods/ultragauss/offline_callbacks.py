import os
import numpy as np
import torch
from tqdm import tqdm

from engine.registry import register_callback
from engine.computer import LossComputer, MetricComputer
from scene.factory import SceneFactory
from utils.eval_protocol import clear_vis_dir, frame_metric_row, should_save_vis, test_frame_count, write_per_frame_metrics
from utils.image_utils import make_diff

from .density_control import UltraGaussOfflineDensityController
from .gaussian_container import UltraGaussContainer
from .gaussian_solver import UltraGaussSolver
from .render import render
from .state import UltraGaussOfflineState
from .density_control import _approx_scale_from_precision_tril


def _should_log_scale_tail_diag(iteration: int) -> bool:
    return iteration % 1000 == 0


def _scale_tail_grad_stats(state: UltraGaussOfflineState):
    container = state.gauss_container
    if container is None or "precision_tril" not in container._param_dict:
        return {}
    beta = float(container.model_cfg.render.diag_beta)
    with torch.no_grad():
        precision = container._param_dict["precision_tril"].tensor.detach()
        scales = _approx_scale_from_precision_tril(precision, beta)
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
        stats["train/scale_tail/top256_opacity_mean"] = float(
            container.opacities.detach().reshape(-1).index_select(0, top_idx).mean().item()
        )
        stats["train/scale_tail/top256_intensity_mean"] = float(
            container.intensities.detach().reshape(-1).index_select(0, top_idx).mean().item()
        )

    param = container._param_dict["precision_tril"].tensor
    grad = param.grad
    if grad is None:
        return stats
    with torch.no_grad():
        raw_top = param.detach().index_select(0, top_idx)
        grad_top = grad.detach().index_select(0, top_idx)
        diag_raw = raw_top[:, [0, 3, 5]]
        diag_grad = grad_top[:, [0, 3, 5]]
        # Positive value means an SGD step would reduce |diag_raw|, reduce precision, and increase scale.
        scale_growth_drive = diag_grad * torch.sign(diag_raw)
        stats["train/scale_tail/top256_precision_grad_abs_mean"] = float(grad_top.abs().mean().item())
        stats["train/scale_tail/top256_diag_scale_growth_drive_mean"] = float(scale_growth_drive.mean().item())
        stats["train/scale_tail/top256_diag_scale_growth_drive_pos_frac"] = float(
            (scale_growth_drive > 0.0).float().mean().item()
        )
    return stats


@register_callback("UltraGaussOfflineSetupCallback")
class UltraGaussOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: UltraGaussOfflineState):
        cfg = state.cfg
        state.scene = SceneFactory.build_scene(cfg, state.device)

        state.gauss_container = UltraGaussContainer(cfg, device=state.device)
        init_data = state.scene.build_init_gaussians()
        state.gauss_container.init_from_scene(init_data["gaussians"])

        state.gauss_solver = UltraGaussSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()

        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]
        state.dc = UltraGaussOfflineDensityController(cfg)
        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)


@register_callback("UltraGaussOfflineTrainCallback")
class UltraGaussOfflineTrainCallback:
    priority = 50

    def on_train(self, state: UltraGaussOfflineState):
        cfg = state.cfg
        iterations = cfg.engine.iterations
        accum_steps = cfg.engine.accum_steps
        intv = cfg.engine.interval

        pbar = tqdm(range(state.iteration + 1, iterations + 1), initial=state.iteration, total=iterations)
        state.scalar_manager.tqdm_bar = pbar
        state.gauss_solver.zero_grad()

        for iteration in pbar:
            state.iteration = iteration
            state.gauss_solver.step_lr(iteration)

            datum = state.scene.next_train_datum()
            pred_image, depth_image, render_info = render(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
                is_jitter=True,
            )
            gt_image = datum["slice_meta"].image

            state.inputs["gt_image"] = gt_image
            state.outputs["pred_image"] = pred_image
            state.outputs["depth_image"] = depth_image
            state.outputs["gauss_scales"] = _approx_scale_from_precision_tril(
                state.gauss_container.precision_tril, float(state.gauss_container.model_cfg.render.diag_beta)
            )
            loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
            (loss_dict["total"] / accum_steps).backward()

            if iteration % accum_steps == 0:
                if state.dc.should_accumulate_grad_stats(iteration):
                    state.dc.update_grad_stats(state.gauss_container)

                scale_tail_stats = None
                if _should_log_scale_tail_diag(iteration):
                    scale_tail_stats = _scale_tail_grad_stats(state)

                state.gauss_solver.step()
                state.gauss_solver.zero_grad()

                if state.dc.should_densify_prune(iteration):
                    state.dc.prune_and_densify(iteration, state.gauss_solver, state.gauss_container)

                if iteration % intv.train.step == 0:
                    stats = {"train/loss/total": loss_dict["total"].item()}
                    for k, v in loss_dict.items():
                        if k != "total":
                            stats[f"train/loss/{k}"] = v.item()
                    stats["train/lr"] = state.gauss_solver.optimizer.param_groups[0]["lr"]
                    stats["train/num_gaussians"] = state.gauss_container.num_points
                    stats.update(state.dc.scalar_stats(iteration))
                    if scale_tail_stats is not None:
                        stats.update(scale_tail_stats)
                    if _should_log_render_stats(cfg, iteration):
                        stats.update(
                            _collect_render_stats(
                                "train/render",
                                pred_image,
                                gt_image,
                                render_info,
                                state.gauss_container,
                            )
                        )
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % intv.train.print == 0:
                        state.scalar_manager.print_train(iteration)

                if intv.test and iteration % intv.test == 0:
                    _run_test(state, iteration)
                if intv.save and iteration % intv.save == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("UltraGaussOfflineTestCallback")
class UltraGaussOfflineTestCallback:
    priority = 50

    def on_test(self, state: UltraGaussOfflineState):
        _run_test(state, iteration=None)


@register_callback("UltraGaussOfflineSaveCallback")
class UltraGaussOfflineSaveCallback:
    priority = 50

    def on_save(self, state: UltraGaussOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("UltraGaussOfflineFinishCallback")
class UltraGaussOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: UltraGaussOfflineState):
        state.scalar_manager.close()


def _save_checkpoint(state: UltraGaussOfflineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


def _run_test(state: UltraGaussOfflineState, iteration=None):
    cfg = state.cfg
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
            pred, _, render_info = render(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
                is_jitter=False,
            )
            gt = datum["slice_meta"].image
            state.inputs["gt_image"] = gt
            state.outputs["pred_image"] = pred
            scores = state.metric_computer.compute(state.inputs, state.outputs)
            for k, v in scores.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            if bool(cfg.model.offline.diagnostics.enable_render_stats):
                render_stats = _collect_render_stats(
                    "test/render",
                    pred,
                    gt,
                    render_info,
                    state.gauss_container,
                )
                for k, v in render_stats.items():
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
                diff_f = (diff.astype(np.float32) / 255.0)
                compose = np.concatenate([gt_f, pred_f, diff_f], axis=1)
                state.io_manager.save_image(os.path.join(item_dir, "compose.png"), compose)

    if count == 0:
        return
    metrics_log = {}
    metrics_display = {}
    for k, v in metric_sums.items():
        value = v / count
        if k.startswith("test/"):
            metrics_log[k] = value
        else:
            metrics_log[f"test/{k}"] = value
            metrics_display[k] = value
    write_per_frame_metrics(cfg.system.output_dir, iteration or state.iteration, per_frame_rows, cfg)
    state.scalar_manager.log_scalars(metrics_log, iteration or state.iteration)
    state.scalar_manager.print_test(iteration, metrics_display, in_place=False)


def _should_log_render_stats(cfg, iteration: int) -> bool:
    diagnostics = cfg.model.offline.diagnostics
    if not bool(diagnostics.enable_render_stats):
        return False
    step = max(1, int(diagnostics.train_step))
    return iteration % step == 0


def _finite_stats(prefix: str, tensor: torch.Tensor):
    values = tensor.detach()
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        values = values[finite]
    if values.numel() == 0:
        return {
            f"{prefix}/finite_frac": 0.0,
        }
    return {
        f"{prefix}/min": values.min().item(),
        f"{prefix}/max": values.max().item(),
        f"{prefix}/mean": values.mean().item(),
        f"{prefix}/std": values.float().std(unbiased=False).item(),
        f"{prefix}/finite_frac": finite.float().mean().item(),
    }


def _collect_render_stats(prefix: str, pred: torch.Tensor, gt: torch.Tensor, render_info, container: UltraGaussContainer):
    stats = {}
    alpha = render_info.get("alpha_image")
    counts = render_info.get("pixel_gauss_counts")

    stats.update(_finite_stats(f"{prefix}/pred", pred))
    stats.update(_finite_stats(f"{prefix}/gt", gt))
    if alpha is not None:
        stats.update(_finite_stats(f"{prefix}/alpha", alpha))
        stats[f"{prefix}/alpha/covered_frac"] = (alpha.detach() > 1.0e-5).float().mean().item()
    if counts is not None:
        counts_f = counts.detach().float()
        stats.update(_finite_stats(f"{prefix}/pixel_counts", counts_f))
        stats[f"{prefix}/pixel_counts/nonzero_frac"] = (counts > 0).float().mean().item()
        stats[f"{prefix}/pixel_counts/total_pairs"] = counts_f.sum().item()

    with torch.no_grad():
        stats.update(_finite_stats(f"{prefix}/opacity", container.opacities))
        stats.update(_finite_stats(f"{prefix}/intensity", container.intensities))
        scales = _approx_scale_from_precision_tril(
            container.precision_tril,
            float(container.model_cfg.render.diag_beta),
        )
        max_scale = torch.max(scales, dim=1).values
        stats.update(_finite_stats(f"{prefix}/max_scale_mm", max_scale))
    return stats
