import os
import numpy as np
import torch
from tqdm import tqdm

from engine.registry import register_callback
from engine.computer import LossComputer, MetricComputer
from scene.factory import SceneFactory
from utils.eval_protocol import clear_vis_dir, frame_metric_row, should_save_vis, test_frame_count, write_per_frame_metrics
from utils.image_utils import make_diff

from .density_control import UltraGrayOfflineDensityController
from .gaussian_container import UltraGrayGaussContainer
from .gaussian_solver import UltraGrayGaussSolver
from .render import render
from .state import UltraGrayOfflineState


@register_callback("UltraGrayOfflineSetupCallback")
class UltraGrayOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: UltraGrayOfflineState):
        cfg = state.cfg
        state.scene = SceneFactory.build_scene(cfg, state.device)

        state.gauss_container = UltraGrayGaussContainer(cfg, device=state.device)
        init_data = state.scene.build_init_gaussians()
        state.gauss_container.init_from_scene(init_data["gaussians"])

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint([("container", state.gauss_container)], resume)

        state.gauss_solver = UltraGrayGaussSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()

        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]
        state.dc = UltraGrayOfflineDensityController(cfg)
        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)

        if resume:
            state.iteration = state.io_manager.load_checkpoint([("solver", state.gauss_solver)], resume)


@register_callback("UltraGrayOfflineTrainCallback")
class UltraGrayOfflineTrainCallback:
    priority = 50

    def on_train(self, state: UltraGrayOfflineState):
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
            pred_image, _, _ = render(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
                is_jitter=True,
                active_sh_degree1=iteration >= int(cfg.model.offline.sh_degree1_start_iter),
            )
            gt_image = datum["slice_meta"].image

            state.inputs["gt_image"] = gt_image
            state.outputs["pred_image"] = pred_image
            state.outputs["gauss_scales"] = state.gauss_container.scales
            loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
            (loss_dict["total"] / accum_steps).backward()

            if iteration % accum_steps == 0:
                if state.dc.should_accumulate_grad_stats(iteration):
                    state.dc.update_grad_stats(state.gauss_container)

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
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % intv.train.print == 0:
                        state.scalar_manager.print_train(iteration)

                if intv.test and iteration % intv.test == 0:
                    _run_test(state, iteration)
                if intv.save and iteration % intv.save == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("UltraGrayOfflineTestCallback")
class UltraGrayOfflineTestCallback:
    priority = 50

    def on_test(self, state: UltraGrayOfflineState):
        _run_test(state, iteration=None)


@register_callback("UltraGrayOfflineSaveCallback")
class UltraGrayOfflineSaveCallback:
    priority = 50

    def on_save(self, state: UltraGrayOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("UltraGrayOfflineFinishCallback")
class UltraGrayOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: UltraGrayOfflineState):
        state.scalar_manager.close()


def _save_checkpoint(state: UltraGrayOfflineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


def _run_test(state: UltraGrayOfflineState, iteration=None):
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
            pred, _, _ = render(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
                is_jitter=False,
                active_sh_degree1=state.iteration >= int(cfg.model.offline.sh_degree1_start_iter),
            )
            gt = datum["slice_meta"].image
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
                diff_f = (diff.astype(np.float32) / 255.0)
                compose = np.concatenate([gt_f, pred_f, diff_f], axis=1)
                state.io_manager.save_image(os.path.join(item_dir, "compose.png"), compose)

    if count == 0:
        return
    metrics_log = {f"test/{k}": v / count for k, v in metric_sums.items()}
    metrics_display = {k: v / count for k, v in metric_sums.items()}
    write_per_frame_metrics(cfg.system.output_dir, iteration or state.iteration, per_frame_rows, cfg)
    state.scalar_manager.log_scalars(metrics_log, iteration or state.iteration)
    state.scalar_manager.print_test(iteration, metrics_display, in_place=False)
