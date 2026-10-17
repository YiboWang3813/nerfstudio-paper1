import os
import time

import numpy as np
import torch
from tqdm import tqdm

from engine.registry import register_callback
from engine.computer import LossComputer, MetricComputer
from scene.factory import SceneFactory
from utils.image_utils import make_diff

from .render import render
from .render_integrated import render as render_integrated
from .state import UltraGSOfflineState
from .gauss_container import UltraGSGaussContainer
from .gauss_solver import UltraGSGaussSolver
from .density_control import UltraGSOfflineDensityController


def _render_dispatch(state: UltraGSOfflineState):
    which = str(getattr(state.gauss_container.model_cfg.render, "which_render", "legacy")).lower()
    if which == "integrated":
        return render_integrated
    return render


def _sync_cuda_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_once_ms(fn) -> float:
    _sync_cuda_if_needed()
    t0 = time.perf_counter()
    fn()
    _sync_cuda_if_needed()
    return (time.perf_counter() - t0) * 1000.0


@torch.no_grad()
def _ab_profile_render_ms(state: UltraGSOfflineState, datum, interval: int) -> None:
    iteration = int(state.iteration)
    if interval <= 0 or (iteration % interval != 0):
        return
    cfg = state.gauss_container.model_cfg
    dataset_meta = state.scene.dataset_meta
    slice_meta = datum["slice_meta"]
    container = state.gauss_container

    def _call_legacy():
        render(
            cfg,
            dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )

    def _call_integrated():
        render_integrated(
            cfg,
            dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )

    t_legacy = _time_once_ms(_call_legacy)
    t_integrated = _time_once_ms(_call_integrated)
    ratio = (t_integrated / t_legacy) if t_legacy > 0 else float("nan")

    state.scalar_manager.log_scalars(
        {
            "debug_ab/render_legacy_ms": float(t_legacy),
            "debug_ab/render_integrated_ms": float(t_integrated),
            "debug_ab/render_ratio_integrated_over_legacy": float(ratio),
        },
        iteration,
    )
    print(
        "[UltraGS-DebugRender-AB] "
        f"iter={iteration} legacy_ms={t_legacy:.3f} "
        f"integrated_ms={t_integrated:.3f} ratio={ratio:.3f}"
    )


@register_callback("UltraGSDebugInRenderOfflineSetupCallback")
class UltraGSDebugInRenderOfflineSetupCallback:
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


@register_callback("UltraGSDebugInRenderOfflineTrainCallback")
class UltraGSDebugInRenderOfflineTrainCallback:
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
        render_ms_sum = 0.0
        render_count = 0
        profile_interval = int(getattr(cfg.engine.interval.train, "print", 100))

        for iteration in pbar:
            state.iteration = iteration
            state.gauss_solver.step_lr(iteration)

            t_fetch0 = time.perf_counter()
            datum = state.scene.next_train_datum()
            t_fetch_ms = (time.perf_counter() - t_fetch0) * 1000.0
            _ab_profile_render_ms(state, datum, interval=profile_interval)
            _sync_cuda_if_needed()
            t_render0 = time.perf_counter()
            pred_image, depth_image, _ = _render_dispatch(state)(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
                is_jitter=True,
                return_intermediates=True,
            )
            _sync_cuda_if_needed()
            t_render_ms = (time.perf_counter() - t_render0) * 1000.0
            render_ms_sum += t_render_ms
            render_count += 1
            gt_image = datum["slice_meta"].image

            state.inputs["gt_image"] = gt_image
            state.outputs["pred_image"] = pred_image
            state.outputs["depth_image"] = depth_image
            state.outputs["gauss_scales"] = state.gauss_container.scales
            state.outputs["gauss_opacities"] = state.gauss_container.opacities
            t_loss0 = time.perf_counter()
            loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
            t_loss_ms = (time.perf_counter() - t_loss0) * 1000.0
            total_loss = loss_dict["total"] / accum_steps
            _sync_cuda_if_needed()
            t_backward0 = time.perf_counter()
            total_loss.backward()
            _sync_cuda_if_needed()
            t_backward_ms = (time.perf_counter() - t_backward0) * 1000.0

            if iteration % accum_steps == 0:
                t_opt0 = time.perf_counter()
                if state.dc.should_accumulate_grad_stats(iteration):
                    state.dc.update_grad_stats(state.gauss_container)
                state.gauss_solver.step()
                state.gauss_solver.zero_grad()
                _sync_cuda_if_needed()
                t_opt_ms = (time.perf_counter() - t_opt0) * 1000.0
                t_dc0 = time.perf_counter()
                if state.dc.should_densify_prune(iteration):
                    state.dc.prune_and_densify(iteration, state.gauss_solver, state.gauss_container)
                _sync_cuda_if_needed()
                t_dc_ms = (time.perf_counter() - t_dc0) * 1000.0

                if iteration % step_intv == 0:
                    lr = state.gauss_solver.optimizer.param_groups[0]["lr"]
                    num_g = state.gauss_container.num_points
                    stats = {"train/loss/total": loss_dict["total"].item()}
                    for k, v in loss_dict.items():
                        if k != "total":
                            stats[f"train/loss/{k}"] = v.item()
                    stats["train/lr"] = lr
                    stats["train/num_gaussians"] = num_g
                    stats["train/time/fetch_ms"] = t_fetch_ms
                    stats["train/time/render_ms"] = t_render_ms
                    stats["train/time/render_ms_avg"] = render_ms_sum / max(1, render_count)
                    stats["train/time/loss_ms"] = t_loss_ms
                    stats["train/time/backward_ms"] = t_backward_ms
                    stats["train/time/opt_ms"] = t_opt_ms
                    stats["train/time/dc_ms"] = t_dc_ms
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % print_intv == 0:
                        state.scalar_manager.print_train(iteration)
                        which = str(
                            getattr(state.gauss_container.model_cfg.render, "which_render", "legacy")
                        ).lower()
                        print(
                            "[UltraGS-DebugRender] "
                            f"iter={iteration} which_render={which} "
                            f"fetch_ms={t_fetch_ms:.3f} render_ms={t_render_ms:.3f} "
                            f"loss_ms={t_loss_ms:.3f} backward_ms={t_backward_ms:.3f} "
                            f"opt_ms={t_opt_ms:.3f} dc_ms={t_dc_ms:.3f} "
                            f"avg_render_ms={render_ms_sum / max(1, render_count):.3f}"
                        )

                if test_intv and iteration % test_intv == 0:
                    _run_test(state, iteration)
                if save_intv and iteration % save_intv == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("UltraGSDebugInRenderOfflineTestCallback")
class UltraGSDebugInRenderOfflineTestCallback:
    priority = 50

    def on_test(self, state: UltraGSOfflineState):
        _run_test(state, iteration=None)


@register_callback("UltraGSDebugInRenderOfflineSaveCallback")
class UltraGSDebugInRenderOfflineSaveCallback:
    priority = 50

    def on_save(self, state: UltraGSOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("UltraGSDebugInRenderOfflineFinishCallback")
class UltraGSDebugInRenderOfflineFinishCallback:
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
    render_ms_sum = 0.0

    with torch.no_grad():
        while True:
            datum = state.scene.next_test_datum()
            if datum is None:
                break
            _sync_cuda_if_needed()
            t_render0 = time.perf_counter()
            pred, _, _ = _render_dispatch(state)(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                datum["slice_meta"],
                state.gauss_container,
                return_intermediates=False,
                is_jitter=False,
            )
            _sync_cuda_if_needed()
            render_ms_sum += (time.perf_counter() - t_render0) * 1000.0
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
    metrics_log["test/render_ms_avg"] = render_ms_sum / count
    state.scalar_manager.log_scalars(metrics_log, iteration or state.iteration)
    state.scalar_manager.print_test(iteration, metrics_display, in_place=False)
    which = str(getattr(state.gauss_container.model_cfg.render, "which_render", "legacy")).lower()
    print(
        f"[UltraGS-DebugRender] eval which_render={which} "
        f"avg_render_ms={render_ms_sum / count:.3f} over {count} slices"
    )
