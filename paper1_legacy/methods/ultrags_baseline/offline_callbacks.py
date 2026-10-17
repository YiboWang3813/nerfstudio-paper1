import os
import numpy as np
import torch
from tqdm import tqdm

from engine.computer import LossComputer, MetricComputer
from engine.registry import register_callback
from scene.factory import SceneFactory
from utils.eval_protocol import clear_vis_dir, frame_metric_row, should_save_vis, test_frame_count, write_per_frame_metrics
from utils.image_utils import make_diff

from .density_control import UltraGSBaselineDensityController
from .gaussian_container import UltraGSBaselineContainer
from .gaussian_solver import UltraGSBaselineSolver
from .render import render
from .state import UltraGSBaselineOfflineState


@register_callback("UltraGSBaselineOfflineSetupCallback")
class UltraGSBaselineOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: UltraGSBaselineOfflineState):
        cfg = state.cfg
        state.scene = SceneFactory.build_scene(cfg, state.device)
        state.gauss_container = UltraGSBaselineContainer(cfg, state.device)
        init_data = state.scene.build_init_gaussians()
        state.gauss_container.init_from_scene(init_data["gaussians"])

        state.gauss_solver = UltraGSBaselineSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()
        state.dc = UltraGSBaselineDensityController(cfg)
        state.dc.set_scene_radius(float(init_data.get("scene_radius", 1.0)))

        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)
        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]
        if cfg.engine.resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, cfg.engine.resume)


@register_callback("UltraGSBaselineOfflineTrainCallback")
class UltraGSBaselineOfflineTrainCallback:
    priority = 50

    def on_train(self, state: UltraGSBaselineOfflineState):
        cfg = state.cfg
        iterations = int(cfg.engine.iterations)
        accum_steps = max(1, int(cfg.engine.accum_steps))
        intv = cfg.engine.interval

        pbar = tqdm(range(state.iteration + 1, iterations + 1), initial=state.iteration, total=iterations)
        state.scalar_manager.tqdm_bar = pbar
        state.gauss_solver.zero_grad()
        for iteration in pbar:
            state.iteration = iteration
            state.gauss_solver.step_lr(iteration)
            _maybe_update_sh_degree(state, iteration)

            datum = state.scene.next_train_datum()
            pred_image, depth_image, render_info = render(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
            )
            gt_image = datum["slice_meta"].image
            if gt_image.dim() == 2:
                gt_image = gt_image.unsqueeze(0)

            state.inputs["gt_image"] = gt_image
            state.outputs["pred_image"] = pred_image
            state.outputs["depth_image"] = depth_image
            loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
            (loss_dict["total"] / accum_steps).backward()

            if iteration % accum_steps == 0:
                if state.dc.should_accumulate_grad_stats(iteration):
                    state.dc.update_grad_stats(state.gauss_container, render_info)

                if state.dc.should_densify_prune(iteration):
                    state.dc.prune_and_densify(iteration, state.gauss_solver, state.gauss_container)
                if state.dc.should_reset_opacity(iteration):
                    state.dc.reset_opacity(state.gauss_solver, state.gauss_container)

                state.gauss_solver.step()
                state.gauss_solver.zero_grad()

                if iteration % intv.train.step == 0:
                    stats = {"train/loss/total": loss_dict["total"].item()}
                    for key, value in loss_dict.items():
                        if key != "total":
                            stats[f"train/loss/{key}"] = value.item()
                    stats["train/num_gaussians"] = state.gauss_container.num_points
                    stats["train/active_sh_degree"] = state.gauss_container.active_sh_degree
                    stats.update(state.dc.scalar_stats(iteration))
                    if bool(cfg.model.offline.diagnostics.enable_render_stats):
                        stats.update(_collect_render_stats("train/render", pred_image, gt_image, render_info, state.gauss_container))
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % intv.train.print == 0:
                        state.scalar_manager.print_train(iteration)

                if intv.test and iteration % intv.test == 0:
                    _run_test(state, iteration)
                if intv.save and iteration % intv.save == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("UltraGSBaselineOfflineTestCallback")
class UltraGSBaselineOfflineTestCallback:
    priority = 50

    def on_test(self, state: UltraGSBaselineOfflineState):
        _run_test(state, iteration=None)


@register_callback("UltraGSBaselineOfflineSaveCallback")
class UltraGSBaselineOfflineSaveCallback:
    priority = 50

    def on_save(self, state: UltraGSBaselineOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("UltraGSBaselineOfflineFinishCallback")
class UltraGSBaselineOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: UltraGSBaselineOfflineState):
        state.scalar_manager.close()


def _maybe_update_sh_degree(state: UltraGSBaselineOfflineState, iteration: int):
    step = int(state.cfg.model.render.sh_degree_step_iter)
    if step > 0 and iteration % step == 0:
        state.gauss_container.oneup_sh_degree()


def _save_checkpoint(state: UltraGSBaselineOfflineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


def _run_test(state: UltraGSBaselineOfflineState, iteration=None):
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
            )
            gt = datum["slice_meta"].image
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)
            state.inputs["gt_image"] = gt
            state.outputs["pred_image"] = pred
            scores = state.metric_computer.compute(state.inputs, state.outputs)
            for key, value in scores.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            count += 1
            per_frame_rows.append(frame_metric_row(count, datum, scores))

            if should_save_vis(cfg, count, test_total):
                item_dir = os.path.join(vis_dir, f"item_{count}")
                pred_np = pred.detach().cpu().squeeze().numpy()
                gt_np = gt.detach().cpu().squeeze().numpy()
                diff = make_diff(pred_np, gt_np)
                state.io_manager.save_images(item_dir, pred.squeeze(), gt.squeeze(), diff)

                gt_f = np.clip(gt_np, 0.0, 1.0).astype(np.float32)
                pred_f = np.clip(pred_np, 0.0, 1.0).astype(np.float32)
                diff_f = diff.astype(np.float32) / 255.0
                state.io_manager.save_image(os.path.join(item_dir, "compose.png"), np.concatenate([gt_f, pred_f, diff_f], axis=1))

    if count == 0:
        return
    metrics_log = {f"test/{key}": value / count for key, value in metric_sums.items()}
    write_per_frame_metrics(cfg.system.output_dir, iteration or state.iteration, per_frame_rows, cfg)
    state.scalar_manager.log_scalars(metrics_log, iteration or state.iteration)
    state.scalar_manager.print_test(iteration, {k: v / count for k, v in metric_sums.items()}, in_place=False)


def _finite_stats(prefix: str, tensor: torch.Tensor):
    values = tensor.detach()
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        values = values[finite]
    if values.numel() == 0:
        return {f"{prefix}/finite_frac": 0.0}
    return {
        f"{prefix}/min": values.min().item(),
        f"{prefix}/max": values.max().item(),
        f"{prefix}/mean": values.mean().item(),
        f"{prefix}/std": values.float().std(unbiased=False).item(),
        f"{prefix}/finite_frac": finite.float().mean().item(),
    }


def _collect_render_stats(prefix: str, pred: torch.Tensor, gt: torch.Tensor, render_info, container: UltraGSBaselineContainer):
    stats = {}
    stats.update(_finite_stats(f"{prefix}/pred", pred))
    stats.update(_finite_stats(f"{prefix}/gt", gt))
    alpha = render_info.get("alpha_image")
    if alpha is not None:
        stats.update(_finite_stats(f"{prefix}/alpha", alpha))
        stats[f"{prefix}/alpha/covered_frac"] = (alpha.detach() > 1.0e-5).float().mean().item()
    stats.update(_finite_stats(f"{prefix}/opacity", container.opacities))
    stats.update(_finite_stats(f"{prefix}/scales_2d", container.scales_2d))
    return stats
