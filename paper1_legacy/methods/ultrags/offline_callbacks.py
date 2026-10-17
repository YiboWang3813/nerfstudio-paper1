import os

import numpy as np
import torch
from tqdm import tqdm

from engine.registry import register_callback
from engine.computer import LossComputer, MetricComputer
from scene.factory import SceneFactory
from utils.image_utils import make_diff

from .render import render
from .render_integrated import render_product as render_integrated_product
from .render_integrated import render_debug as render_integrated_debug
from .render_integrated import render_batch_product as render_integrated_batch_product
from .render_integrated import render_batch_debug as render_integrated_batch_debug
from .state import UltraGSOfflineState
from .gauss_container import UltraGSGaussContainer
from .gauss_solver import UltraGSGaussSolver
from .density_control import UltraGSOfflineDensityController


def _want_debug_intermediates(state: UltraGSOfflineState) -> bool:
    level = str(
        getattr(
            state.gauss_container.model_cfg.render,
            "intermediates_level",
            "none",
        )
    ).lower()
    return level != "none"


def _render_dispatch(state: UltraGSOfflineState):
    which = str(getattr(state.gauss_container.model_cfg.render, "which_render", "legacy")).lower()
    if which == "integrated":
        return render_integrated_debug if _want_debug_intermediates(state) else render_integrated_product
    return render


def _render_dispatch_train_batch(state: UltraGSOfflineState):
    """Integrated batched path when ``render.batched_kernel`` and ``batch_size > 1``."""
    mc = state.gauss_container.model_cfg
    which = str(getattr(mc.render, "which_render", "legacy")).lower()
    if which != "integrated":
        return None
    if not bool(getattr(mc.render, "batched_kernel", False)):
        return None
    if int(getattr(mc.render, "batch_size", 1) or 1) <= 1:
        return None
    return (
        render_integrated_batch_debug
        if _want_debug_intermediates(state)
        else render_integrated_batch_product
    )


@register_callback("UltraGSOfflineSetupCallback")
class UltraGSOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: UltraGSOfflineState):
        print(f"[UltraGS] Setting up components for device: {state.device}")
        cfg = state.cfg
        state.scene = SceneFactory.build_scene(cfg, state.device)
        train_n = len(state.scene.train_container)
        test_n = len(state.scene.test_container)
        print(
            "[UltraGS] Offline data split:"
            f" train={train_n} ({cfg.scene.loading.images_name}),"
            f" test={test_n} ({cfg.scene.loading.test_images_name})"
        )

        state.gauss_container = UltraGSGaussContainer(cfg, device=state.device)
        init_data = state.scene.build_init_gaussians()
        state.gauss_container.init_from_scene(init_data["gaussians"])

        state.gauss_solver = UltraGSGaussSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()

        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]
        state.dc = UltraGSOfflineDensityController(cfg)
        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)
        print("[UltraGS] Components initialized successfully.")


@register_callback("UltraGSOfflineTrainCallback")
class UltraGSOfflineTrainCallback:
    priority = 50

    def on_train(self, state: UltraGSOfflineState):
        cfg = state.cfg
        iterations = cfg.engine.iterations
        accum_steps = cfg.engine.accum_steps
        intv = cfg.engine.interval
        step_intv = intv.train.step
        print_intv = intv.train.print
        test_intv = intv.test
        save_intv = intv.save

        start_iter = state.iteration
        pbar = tqdm(range(start_iter + 1, iterations + 1), initial=start_iter, total=iterations)
        state.scalar_manager.tqdm_bar = pbar
        state.gauss_solver.zero_grad()
        log_batch_shape_once = os.environ.get("ULTRAGS_LOG_BATCH_SHAPE_ONCE", "0") == "1"
        shape_logged = False

        for iteration in pbar:
            state.iteration = iteration
            state.gauss_solver.step_lr(iteration)

            train_batch_fn = _render_dispatch_train_batch(state)
            if train_batch_fn is not None:
                B = int(state.gauss_container.model_cfg.render.batch_size)
                batch = state.scene.next_train_batch(B)
                pred_image, depth_image, _ = train_batch_fn(
                    state.gauss_container.model_cfg,
                    state.scene.dataset_meta,
                    batch,
                    state.gauss_container,
                    is_jitter=True,
                )
                gt_image = batch["stacked"]["image"]
            else:
                datum = state.scene.next_train_datum()
                pred_image, depth_image, _ = _render_dispatch(state)(
                    state.gauss_container.model_cfg,
                    state.scene.dataset_meta,
                    datum["slice_meta"],
                    state.gauss_container,
                    is_jitter=True,
                )
                gt_image = datum["slice_meta"].image

            if log_batch_shape_once and (not shape_logged):
                print(
                    "[UltraGS][BatchShape] "
                    f"iter={iteration} "
                    f"gt_shape={tuple(gt_image.shape)} "
                    f"pred_shape={tuple(pred_image.shape)} "
                    f"batched_kernel={bool(getattr(state.gauss_container.model_cfg.render, 'batched_kernel', False))} "
                    f"batch_size={int(getattr(state.gauss_container.model_cfg.render, 'batch_size', 1))}"
                )
                shape_logged = True

            state.inputs["gt_image"] = gt_image
            state.outputs["pred_image"] = pred_image
            state.outputs["depth_image"] = depth_image
            state.outputs["gauss_scales"] = state.gauss_container.scales
            state.outputs["gauss_opacities"] = state.gauss_container.opacities
            loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
            total_loss = loss_dict["total"] / accum_steps
            total_loss.backward()

            if iteration % accum_steps == 0:
                if state.dc.should_accumulate_grad_stats(iteration):
                    state.dc.update_grad_stats(state.gauss_container)
                state.gauss_solver.step()
                state.gauss_solver.zero_grad()
                if state.dc.should_densify_prune(iteration):
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
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % print_intv == 0:
                        state.scalar_manager.print_train(iteration)

                if test_intv and iteration % test_intv == 0:
                    _run_test(state, iteration)
                if save_intv and iteration % save_intv == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("UltraGSOfflineTestCallback")
class UltraGSOfflineTestCallback:
    priority = 50

    def on_test(self, state: UltraGSOfflineState):
        _run_test(state, iteration=None)


@register_callback("UltraGSOfflineSaveCallback")
class UltraGSOfflineSaveCallback:
    priority = 50

    def on_save(self, state: UltraGSOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("UltraGSOfflineFinishCallback")
class UltraGSOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: UltraGSOfflineState):
        state.scalar_manager.close()


def _save_checkpoint(state: UltraGSOfflineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


def _run_test(state: UltraGSOfflineState, iteration=None):
    cfg = state.cfg
    it_str = f"Iter {iteration}" if iteration is not None else "Eval"
    print(f"\n[{it_str}] Starting Evaluation...")
    print(
        "[UltraGS] Eval source:"
        f" images={cfg.scene.loading.test_images_name},"
        f" poses={cfg.scene.loading.test_poses_name}"
    )
    vis_dir = os.path.join(cfg.system.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    count = 0
    metric_sums = {}

    with torch.no_grad():
        while True:
            datum = state.scene.next_test_datum()
            if datum is None:
                break
            pred, _, _ = _render_dispatch(state)(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
                return_intermediates=False,
                is_jitter=False,
            )
            gt = datum["slice_meta"].image
            state.inputs["gt_image"] = gt
            state.outputs["pred_image"] = pred
            scores = state.metric_computer.compute(state.inputs, state.outputs)
            for k, v in scores.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            count += 1

            item_dir = os.path.join(vis_dir, f"item_{count}")
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

    if count == 0:
        return
    metrics_log = {f"test/{k}": v / count for k, v in metric_sums.items()}
    metrics_display = {k: v / count for k, v in metric_sums.items()}
    state.scalar_manager.log_scalars(metrics_log, iteration or state.iteration)
    state.scalar_manager.print_test(iteration, metrics_display, in_place=False)
