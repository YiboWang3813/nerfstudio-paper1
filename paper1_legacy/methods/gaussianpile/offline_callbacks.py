"""Offline GaussianPile training callbacks."""

import csv
import os
import shutil
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from engine.computer import LossComputer, MetricComputer
from engine.registry import register_callback
from scene.factory import SceneFactory
from utils.eval_protocol import clear_vis_dir, frame_metric_row, should_save_vis, test_frame_count, write_per_frame_metrics
from utils.image_utils import make_diff

from .density_control import GaussianPileOfflineDensityController
from .gaussian_container import GaussianPileGaussContainer
from .gaussian_solver import GaussianPileGaussSolver
from .render import render
from .state import GaussianPileOfflineState


def _console_log(state: GaussianPileOfflineState, message: str) -> None:
    pbar = getattr(state.scalar_manager, "tqdm_bar", None)
    if pbar is not None and hasattr(pbar, "write"):
        pbar.write(message)
        fp = getattr(pbar, "fp", None)
        if fp is not None and hasattr(fp, "flush"):
            fp.flush()
        sys.stderr.flush()
    else:
        print(message, flush=True)


@register_callback("GaussianPileOfflineSetupCallback")
class GaussianPileOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: GaussianPileOfflineState):
        cfg = state.cfg
        print(f"[GaussianPile] Setting up on device: {state.device}")
        state.scene = SceneFactory.build_scene(cfg, state.device)
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
            "[GaussianPile] Offline data split:"
            f" train={train_n} ({cfg.scene.loading.images_name}),"
            f" test={test_n} ({test_desc})"
        )

        n_init = int(state.model_cfg.gauss_init.num_points)
        if n_init > 0:
            state.cfg.scene.sampler.num_init = n_init
            print(f"[GaussianPile] Gaussian init count: {n_init}")

        state.gauss_container = GaussianPileGaussContainer(cfg, device=state.device)
        init_data = state.scene.build_init_gaussians()
        state.gauss_container.init_from_scene(init_data["gaussians"])
        state.scene_radius = float(init_data.get("scene_radius", 0.0))

        with torch.no_grad():
            m = state.gauss_container.means
            print(
                "[GaussianPile] Init Gaussians (world mm): "
                f"N={state.gauss_container.num_points} "
                f"means_min={m.min(dim=0).values.cpu().tolist()} "
                f"means_max={m.max(dim=0).values.cpu().tolist()} "
                f"scene_radius={state.scene_radius:.4f}"
            )

        state.gauss_solver = GaussianPileGaussSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()
        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]
        state.dc = GaussianPileOfflineDensityController(cfg)
        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)
        print("[GaussianPile] Setup done.")


@register_callback("GaussianPileOfflineTrainCallback")
class GaussianPileOfflineTrainCallback:
    priority = 50

    def on_train(self, state: GaussianPileOfflineState):
        cfg = state.cfg
        iterations = int(cfg.engine.iterations)
        accum_steps = int(cfg.engine.accum_steps)
        intv = cfg.engine.interval
        step_intv = int(intv.train.step)
        print_intv = int(intv.train.print)
        test_intv = int(intv.test)
        save_intv = int(intv.save)

        start_iter = int(state.iteration)
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
        last_speed_iter = start_iter
        last_speed_time = time.perf_counter()

        for iteration in pbar:
            state.iteration = iteration
            state.gauss_solver.step_lr(iteration)

            datum = state.scene.next_train_datum()
            slice_meta = datum["slice_meta"]
            pred_image, depth_image, _ = render(
                state.gauss_container.model_cfg,
                state.scene.dataset_meta,
                slice_meta,
                state.gauss_container,
                is_jitter=True,
                return_intermediates=False,
            )
            gt_image = slice_meta.image

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
                    state.dc.update_grad_stats(state.gauss_container, state.scene.dataset_meta, slice_meta)
                state.gauss_solver.step()
                state.gauss_solver.zero_grad()
                if state.dc.should_densify_prune(iteration):
                    state.dc.prune_and_densify(iteration, state.gauss_solver, state.gauss_container)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if iteration % step_intv == 0:
                    lr = state.gauss_solver.optimizer.param_groups[0]["lr"]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    elapsed = max(1.0e-9, now - last_speed_time)
                    iter_per_s = float((iteration - last_speed_iter) / elapsed)
                    last_speed_iter = iteration
                    last_speed_time = now
                    stats = {"train/loss/total": loss_dict["total"].item()}
                    for k, v in loss_dict.items():
                        if k != "total":
                            stats[f"train/loss/{k}"] = v.item()
                    stats["train/lr"] = lr
                    stats["train/num_gaussians"] = state.gauss_container.num_points
                    stats["train/iter_per_s"] = iter_per_s
                    stats.update(state.dc.scalar_stats(iteration))
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % print_intv == 0:
                        state.scalar_manager.print_train(iteration, in_place=False)

                if test_intv and iteration % test_intv == 0:
                    pbar.write("")
                    metrics = _run_test(state, iteration)
                    _maybe_save_best(state, iteration, metrics)
                if save_intv and iteration % save_intv == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("GaussianPileOfflineTestCallback")
class GaussianPileOfflineTestCallback:
    priority = 50

    def on_test(self, state: GaussianPileOfflineState):
        _run_test(state, iteration=None)


@register_callback("GaussianPileOfflineSaveCallback")
class GaussianPileOfflineSaveCallback:
    priority = 50

    def on_save(self, state: GaussianPileOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("GaussianPileOfflineFinishCallback")
class GaussianPileOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: GaussianPileOfflineState):
        if bool(state.cfg.model.offline.diagnostics.restore_best_on_finish):
            _restore_best_on_finish(state)
        _write_run_manifest(state)
        state.scalar_manager.close()


def _save_checkpoint(state: GaussianPileOfflineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


def _maybe_save_best(state: GaussianPileOfflineState, iteration: int, metrics) -> None:
    if not metrics or "psnr" not in metrics:
        return
    psnr = float(metrics["psnr"])
    best = getattr(state, "best_test_metrics", None)
    if best is not None and psnr <= float(best.get("psnr", float("-inf"))):
        return
    state.best_test_iteration = int(iteration)
    state.best_test_metrics = {k: float(v) for k, v in metrics.items()}
    state.io_manager.save_checkpoint(state.ckpt_items, "best")
    best_vis_dir = _publish_best_visuals(state)
    state.best_test_vis_dir = best_vis_dir
    _console_log(
        state,
        f"[GaussianPile] New best test PSNR {psnr:.4f} at iter {iteration}; checkpoint saved as iteration_best_*.",
    )


def _restore_best_on_finish(state: GaussianPileOfflineState) -> bool:
    best_iteration = getattr(state, "best_test_iteration", None)
    best_metrics = getattr(state, "best_test_metrics", None)
    if best_iteration is None or not best_metrics:
        return False
    if int(best_iteration) == int(state.iteration):
        return False

    if not _load_named_checkpoint(state, "best", load_solver=False):
        return False

    state.gauss_solver.setup()
    state.gauss_solver.step_lr(int(state.iteration))
    state.gauss_solver.zero_grad()
    _console_log(
        state,
        "[GaussianPile] Restored best checkpoint for final artifacts: "
        f"iter={best_iteration}, psnr={float(best_metrics.get('psnr', float('nan'))):.4f}.",
    )
    metrics = _run_test(state, int(state.iteration))
    if metrics is not None:
        state.final_restored_from_best = True
    return True


def _load_named_checkpoint(state: GaussianPileOfflineState, tag: str, load_solver: bool = True) -> bool:
    ckpt_dir = str(getattr(state.cfg.system, "checkpoint_dir", "") or os.path.join(state.cfg.system.output_dir, "checkpoints"))
    items = [(name, obj) for name, obj in state.ckpt_items if load_solver or name != "solver"]
    paths = {name: os.path.join(ckpt_dir, f"iteration_{tag}_{name}.pth") for name, _ in items}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        _console_log(state, f"[GaussianPile] Best checkpoint restore skipped; missing {missing[0]}.")
        return False

    for name, obj in items:
        if name == "solver":
            # Loading the container recreates Parameters, so rebuild the optimizer
            # before loading its saved state.
            obj.setup()
        path = paths[name]
        if hasattr(obj, "load"):
            obj.load(path)
        else:
            blob = torch.load(path, map_location=getattr(state, "device", None))
            if hasattr(obj, "load_state_dict"):
                obj.load_state_dict(blob)
    return True


def _run_test(state: GaussianPileOfflineState, iteration=None):
    cfg = state.cfg
    it_str = f"Iter {iteration}" if iteration is not None else "Eval"
    _console_log(state, f"[{it_str}] GaussianPile evaluation - start (test frames)...")
    vis_dir = _test_vis_dir(cfg.system.output_dir, iteration)
    clear_vis_dir(vis_dir)
    state.last_test_vis_dir = vis_dir
    count = 0
    metric_sums = {}
    per_frame_rows = []
    test_total = test_frame_count(state.scene)
    pred_mean_values = []
    pred_max_values = []

    with torch.no_grad():
        while True:
            datum = state.scene.next_test_datum()
            if datum is None:
                break
            slice_meta = datum["slice_meta"]
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

            pred_mean_values.append(float(pred.detach().mean().item()))
            pred_max_values.append(float(pred.detach().max().item()))
            if should_save_vis(cfg, count, test_total):
                item_dir = os.path.join(vis_dir, f"item_{count}")
                pred_np = pred.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()
                diff = make_diff(pred_np, gt_np)
                state.io_manager.save_images(item_dir, pred, gt, diff)
                compose = np.concatenate(
                    [
                        np.clip(gt_np, 0.0, 1.0).astype(np.float32),
                        np.clip(pred_np, 0.0, 1.0).astype(np.float32),
                        diff.astype(np.float32) / 255.0,
                    ],
                    axis=1,
                )
                state.io_manager.save_image(os.path.join(item_dir, "compose.png"), compose)

    if count == 0:
        _console_log(state, f"[{it_str}] GaussianPile evaluation - no test frames.")
        return None

    health = _prediction_health_summary(
        pred_mean_values,
        pred_max_values,
        float(cfg.model.offline.diagnostics.all_black_eps),
    )
    metrics_log = {f"test/{k}": v / count for k, v in metric_sums.items()}
    metrics_log.update({f"test/{k}": v for k, v in health.items()})
    metrics_display = {k: v / count for k, v in metric_sums.items()}
    metrics_display.update(health)
    write_per_frame_metrics(cfg.system.output_dir, iteration or state.iteration, per_frame_rows, cfg)
    state.scalar_manager.log_scalars(metrics_log, iteration or state.iteration)
    state.scalar_manager.print_test(iteration, metrics_display, in_place=False)
    if (
        bool(cfg.model.offline.diagnostics.warn_on_all_black_test)
        and health["pred_all_black_frac"] >= 1.0
    ):
        msg = (
            f"[{it_str}] GaussianPile warning: all held-out predictions are below "
            f"{float(cfg.model.offline.diagnostics.all_black_eps):.2e}; "
            "this checkpoint is visually collapsed."
        )
        _console_log(state, msg)
        if bool(cfg.model.offline.diagnostics.fail_on_all_black_test):
            raise RuntimeError(msg)
    _console_log(state, f"[{it_str}] GaussianPile evaluation - done ({count} frames).")
    return metrics_display


def _test_vis_dir(output_dir: str, iteration) -> str:
    if iteration is None:
        return os.path.join(output_dir, "vis")
    return os.path.join(output_dir, "vis_history", f"iteration_{int(iteration):06d}")


def _replace_tree(src: str, dst: str) -> None:
    if not src or not os.path.isdir(src):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _publish_best_visuals(state: GaussianPileOfflineState) -> str:
    src = getattr(state, "last_test_vis_dir", "")
    dst = os.path.join(state.cfg.system.output_dir, "vis")
    _replace_tree(src, dst)
    return dst


def _prediction_health_summary(pred_means, pred_maxes, black_eps: float):
    if not pred_maxes:
        return {
            "pred_mean": 0.0,
            "pred_max_mean": 0.0,
            "pred_all_black_frac": 0.0,
        }
    pred_maxes_arr = np.asarray(pred_maxes, dtype=np.float64)
    pred_means_arr = np.asarray(pred_means, dtype=np.float64)
    return {
        "pred_mean": float(pred_means_arr.mean()),
        "pred_max_mean": float(pred_maxes_arr.mean()),
        "pred_all_black_frac": float(np.mean(pred_maxes_arr <= float(black_eps))),
    }


def _last_scalars(output_dir: str):
    path = os.path.join(output_dir, "logs", "scalars.tsv")
    if not os.path.exists(path):
        return {}, path
    series = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                series[row["key"]] = (int(row["step"]), float(row["value"]))
            except (KeyError, ValueError):
                continue
    return series, path


def _write_run_manifest(state: GaussianPileOfflineState) -> None:
    cfg = state.cfg
    series, scalars_path = _last_scalars(cfg.system.output_dir)

    def val(key):
        item = series.get(key)
        return "null" if item is None else f"{item[1]:.10g}"

    manifest_path = os.path.join(cfg.system.output_dir, "run_manifest.yaml")
    stdout_log = f"logs/gaussianpile/{os.environ.get('SLURM_JOB_NAME', 'local')}_{os.environ.get('SLURM_JOB_ID', 'local')}.out"
    stderr_log = f"logs/gaussianpile/{os.environ.get('SLURM_JOB_NAME', 'local')}_{os.environ.get('SLURM_JOB_ID', 'local')}.err"
    best_metrics = getattr(state, "best_test_metrics", None) or {}
    best_iteration = getattr(state, "best_test_iteration", None)

    def best_val(key):
        return "null" if key not in best_metrics else f"{float(best_metrics[key]):.10g}"

    text = f"""method_name: gaussianpile
experiment_name: {cfg.system.exp_tag}
experiment_namespace: gaussianpile
config: {getattr(cfg.system, 'config_path', '')}
overrides: ""
git_commit: dirty-or-unrecorded
slurm_job_id: {os.environ.get('SLURM_JOB_ID', '')}
slurm_script: {os.environ.get('SLURM_SCRIPT', '')}
host: {os.uname().nodename}
conda_env: {os.environ.get('CONDA_DEFAULT_ENV', '')}
cuda_module: {os.environ.get('CUDA_MODULE', 'cuda/11.8.0')}
dataset: {cfg.scene.dataset_name}
split: {cfg.scene.loading.test_mode}
seed: {cfg.system.seed}
resolution_or_scale: "H={state.scene.dataset_meta.H}, W={state.scene.dataset_meta.W}, lateral_mm={state.scene.dataset_meta.delta_lateral_mm}, axial_mm={state.scene.dataset_meta.delta_axial_mm}, elev_mm={state.scene.dataset_meta.delta_elevational_mm}"
train_steps: {cfg.engine.iterations}
initial_gaussian_count: {cfg.model.gauss_init.num_points}
final_gaussian_count: {state.gauss_container.num_points if state.gauss_container is not None else 'null'}
end_time: {time.strftime('%Y-%m-%d %H:%M:%S')}
final_metrics:
  psnr: {val('test/psnr')}
  ssim: {val('test/ssim')}
  lpips: {val('test/lpips')}
  l1: {val('test/l1')}
  speed_it_s: {val('train/iter_per_s')}
  final_gaussian_count: {state.gauss_container.num_points if state.gauss_container is not None else 'null'}
best_metrics:
  iteration: {best_iteration if best_iteration is not None else 'null'}
  psnr: {best_val('psnr')}
  ssim: {best_val('ssim')}
  lpips: {best_val('lpips')}
  l1: {best_val('l1')}
best_visuals: {getattr(state, 'best_test_vis_dir', 'null')}
latest_visuals: {getattr(state, 'last_test_vis_dir', 'null')}
final_restored_from_best: {str(bool(getattr(state, 'final_restored_from_best', False))).lower()}
logs:
  stdout: {stdout_log}
  stderr: {stderr_log}
  scalars: {scalars_path}
notes: "Generated by GaussianPileOfflineFinishCallback."
"""
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(text)
    _console_log(state, f"[GaussianPile] Run manifest written: {manifest_path}")
