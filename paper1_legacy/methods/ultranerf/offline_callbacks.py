import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from engine.computer import LossComputer, MetricComputer
from engine.registry import register_callback
from scene.factory import SceneFactory
from utils.eval_protocol import clear_vis_dir, frame_metric_row, should_save_vis, test_frame_count, write_per_frame_metrics
from utils.image_utils import make_diff

from .field import UltraNeRFField
from .ray_sampler import extract_slice_grid, random_patch_spec
from .render import render_slice
from .solver import UltraNeRFSolver
from .state import UltraNeRFOfflineState


@register_callback("UltraNeRFOfflineSetupCallback")
class UltraNeRFOfflineSetupCallback:
    priority = 10

    def on_setup(self, state: UltraNeRFOfflineState):
        cfg = state.cfg
        state.scene = SceneFactory.build_scene(cfg, state.device)
        state.field = UltraNeRFField(cfg.model.field).to(state.device)
        state.solver = UltraNeRFSolver(cfg.model, state.field)
        state.solver.setup()

        state.ckpt_items = [("field", state.field), ("optimizer", state.solver.optimizer)]
        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)

        train_n = len(state.scene.train_indices) if state.scene.train_indices is not None else len(state.scene.train_container)
        test_n = len(state.scene.test_indices) if state.scene.test_indices is not None else len(state.scene.test_container)
        print(
            "[UltraNeRF] Offline setup:"
            f" train={train_n}, test={test_n},"
            f" resolution=({state.scene.dataset_meta.H},{state.scene.dataset_meta.W}),"
            f" patch=({cfg.model.ray.train_patch_height},{cfg.model.ray.train_patch_width})"
        )


@register_callback("UltraNeRFOfflineTrainCallback")
class UltraNeRFOfflineTrainCallback:
    priority = 50

    def on_train(self, state: UltraNeRFOfflineState):
        cfg = state.cfg
        iterations = int(cfg.engine.iterations)
        accum_steps = max(1, int(cfg.engine.accum_steps))
        intv = cfg.engine.interval

        pbar = tqdm(range(state.iteration + 1, iterations + 1), initial=state.iteration, total=iterations)
        state.scalar_manager.tqdm_bar = pbar
        state.solver.zero_grad()

        for iteration in pbar:
            state.iteration = iteration

            datum = state.scene.next_train_datum()
            slice_meta = datum["slice_meta"]
            gt = slice_meta.image
            patch = random_patch_spec(
                int(gt.shape[0]),
                int(gt.shape[1]),
                int(cfg.model.ray.train_patch_height),
                int(cfg.model.ray.train_patch_width),
            )
            _points, _depths, gt_patch = extract_slice_grid(slice_meta, patch)
            pred_patch, depth_patch, render_info = render_slice(
                cfg.model,
                slice_meta,
                state.field,
                patch=patch,
                chunk_size=int(cfg.model.ray.eval_chunk_size),
            )

            state.inputs["gt_image"] = gt_patch
            state.outputs["pred_image"] = pred_patch
            state.outputs["depth_image"] = depth_patch
            loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
            (loss_dict["total"] / accum_steps).backward()

            if iteration % accum_steps == 0:
                state.solver.step()
                state.solver.step_lr(iteration)
                state.solver.zero_grad()

                if iteration % intv.train.step == 0:
                    stats = {"train/loss/total": loss_dict["total"].item()}
                    for k, v in loss_dict.items():
                        if k != "total":
                            stats[f"train/loss/{k}"] = v.item()
                    stats["train/lr"] = state.solver.optimizer.param_groups[0]["lr"]
                    if _should_log_render_stats(cfg, iteration):
                        stats.update(_collect_render_stats("train/render", pred_patch, gt_patch, render_info))
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % intv.train.print == 0:
                        state.scalar_manager.print_train(iteration, in_place=False)

                if intv.test and iteration % intv.test == 0:
                    _run_test(state, iteration)
                if intv.save and iteration % intv.save == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("UltraNeRFOfflineTestCallback")
class UltraNeRFOfflineTestCallback:
    priority = 50

    def on_test(self, state: UltraNeRFOfflineState):
        _run_test(state, iteration=None)


@register_callback("UltraNeRFOfflineSaveCallback")
class UltraNeRFOfflineSaveCallback:
    priority = 50

    def on_save(self, state: UltraNeRFOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("UltraNeRFOfflineFinishCallback")
class UltraNeRFOfflineFinishCallback:
    priority = 90

    def on_finish(self, state: UltraNeRFOfflineState):
        _write_manifest(state)
        state.scalar_manager.close()


def _save_checkpoint(state: UltraNeRFOfflineState, iteration: int):
    state.io_manager.save_checkpoint(state.ckpt_items, iteration)


def _run_test(state: UltraNeRFOfflineState, iteration=None):
    cfg = state.cfg
    vis_dir = os.path.join(cfg.system.output_dir, "vis")
    clear_vis_dir(vis_dir)

    count = 0
    metric_sums: Dict[str, float] = {}
    per_frame_rows = []
    test_total = test_frame_count(state.scene)
    state.field.eval()
    with torch.no_grad():
        while True:
            datum = state.scene.next_test_datum()
            if datum is None:
                break
            slice_meta = datum["slice_meta"]
            pred, _depth, render_info, stochastic_pred = _render_eval(state, slice_meta, iteration, count)
            gt = slice_meta.image
            state.inputs["gt_image"] = gt
            state.outputs["pred_image"] = pred
            scores = state.metric_computer.compute(state.inputs, state.outputs)
            for k, v in scores.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            if bool(cfg.model.offline.diagnostics.enable_render_stats):
                stats = _collect_render_stats("test/render", pred, gt, render_info)
                for k, v in stats.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v

            count += 1
            per_frame_rows.append(frame_metric_row(count, datum, scores))
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
                if stochastic_pred is not None:
                    stochastic_np = stochastic_pred.detach().cpu().numpy()
                    stochastic_diff = make_diff(stochastic_np, gt_np)
                    state.io_manager.save_image(os.path.join(item_dir, "pred_stochastic.png"), stochastic_pred)
                    stochastic_compose = np.concatenate(
                        [
                            np.clip(gt_np, 0.0, 1.0).astype(np.float32),
                            np.clip(stochastic_np, 0.0, 1.0).astype(np.float32),
                            stochastic_diff.astype(np.float32) / 255.0,
                        ],
                        axis=1,
                    )
                    state.io_manager.save_image(os.path.join(item_dir, "compose_stochastic.png"), stochastic_compose)
                if bool(cfg.model.offline.diagnostics.save_debug_maps):
                    _save_debug_maps(state, item_dir, render_info)

    state.field.train()
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


def _eval_seed(cfg, slice_meta, iteration, sample_idx: int) -> int:
    frame_id = int(slice_meta.frame_id) if getattr(slice_meta, "frame_id", None) is not None else 0
    iter_id = int(iteration) if iteration is not None else 0
    return int(cfg.model.render.eval_seed) + frame_id * 1009 + iter_id * 9176 + int(sample_idx)


def _average_render_infos(infos):
    keys = infos[0].keys()
    out = {}
    for key in keys:
        vals = [info.get(key) for info in infos]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[key] = torch.stack(vals, dim=0).mean(dim=0)
    return out


def _render_eval(state: UltraNeRFOfflineState, slice_meta, iteration, count: int):
    cfg = state.cfg
    mode = str(cfg.model.render.eval_render_mode).lower()
    chunk_size = int(cfg.model.ray.eval_chunk_size)

    if mode == "expectation":
        pred, depth, info = render_slice(
            cfg.model,
            slice_meta,
            state.field,
            patch=None,
            chunk_size=chunk_size,
            boundary_sampling_mode="expectation",
            scatter_sampling_mode="expectation",
        )
        return pred, depth, info, None

    if mode == "bernoulli":
        pred, depth, info = render_slice(
            cfg.model,
            slice_meta,
            state.field,
            patch=None,
            chunk_size=chunk_size,
            sampling_seed=_eval_seed(cfg, slice_meta, iteration, sample_idx=0),
        )
        return pred, depth, info, None

    if mode == "mc_average":
        mc_samples = max(1, int(cfg.model.render.eval_mc_samples))
        preds = []
        infos = []
        depth = None
        for sample_idx in range(mc_samples):
            pred_i, depth_i, info_i = render_slice(
                cfg.model,
                slice_meta,
                state.field,
                patch=None,
                chunk_size=chunk_size,
                sampling_seed=_eval_seed(cfg, slice_meta, iteration, sample_idx=sample_idx),
            )
            preds.append(pred_i)
            infos.append(info_i)
            depth = depth_i
        pred = torch.stack(preds, dim=0).mean(dim=0)
        stochastic_pred = preds[0] if bool(cfg.model.render.eval_save_stochastic) and count < 4 else None
        return pred, depth, _average_render_infos(infos), stochastic_pred

    raise ValueError(f"Unsupported ultranerf render.eval_render_mode='{cfg.model.render.eval_render_mode}'")


def _save_debug_maps(state: UltraNeRFOfflineState, item_dir: str, render_info: Dict[str, torch.Tensor]) -> None:
    for name in ("alpha", "beta", "rho_b", "rho_s", "phi", "transmission", "backscatter", "reflected"):
        tensor = render_info.get(name)
        if tensor is None:
            continue
        arr = tensor.detach()
        arr = arr - arr.min()
        arr = arr / arr.max().clamp_min(1.0e-8)
        state.io_manager.save_image(os.path.join(item_dir, f"{name}.png"), arr)


def _should_log_render_stats(cfg, iteration: int) -> bool:
    diagnostics = cfg.model.offline.diagnostics
    if not bool(diagnostics.enable_render_stats):
        return False
    return iteration % max(1, int(diagnostics.train_step)) == 0


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


def _collect_render_stats(prefix: str, pred: torch.Tensor, gt: torch.Tensor, render_info: Dict[str, torch.Tensor]):
    stats = {}
    stats.update(_finite_stats(f"{prefix}/pred", pred))
    stats.update(_finite_stats(f"{prefix}/gt", gt))
    for name in ("alpha", "beta", "rho_b", "rho_s", "phi", "transmission", "backscatter", "reflected"):
        tensor = render_info.get(name)
        if tensor is not None:
            stats.update(_finite_stats(f"{prefix}/{name}", tensor))
    return stats


def _write_manifest(state: UltraNeRFOfflineState) -> None:
    cfg = state.cfg
    path = os.path.join(cfg.system.output_dir, "run_manifest.yaml")
    train_n = len(state.scene.train_indices) if state.scene and state.scene.train_indices is not None else None
    test_n = len(state.scene.test_indices) if state.scene and state.scene.test_indices is not None else None
    lines = [
        "method_name: ultranerf",
        f"experiment_name: {cfg.system.exp_tag}",
        f"output_dir: {cfg.system.output_dir}",
        f"dataset: {cfg.scene.dataset_name}",
        f"sweep: {cfg.scene.loading.sweep_name}",
        f"split: {cfg.scene.loading.test_mode}",
        f"train_frames: {train_n}",
        f"test_frames: {test_n}",
        f"resolution: {state.scene.dataset_meta.H}x{state.scene.dataset_meta.W}" if state.scene else "resolution: null",
        f"train_steps: {cfg.engine.iterations}",
        f"seed: {cfg.system.seed}",
        f"final_iteration: {state.iteration}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
