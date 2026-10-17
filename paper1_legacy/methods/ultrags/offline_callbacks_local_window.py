import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from engine.registry import register_callback
from engine.computer import LossComputer, MetricComputer
from scene.factory import SceneFactory
from scene.meta.slice_meta.collate import stack_slice_metas
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
from .gaussian_state_manager import build_gaussian_state_manager


def _want_debug_intermediates(state: UltraGSOfflineState) -> bool:
    level = str(
        getattr(
            state.gauss_container.model_cfg.render,
            "intermediates_level",
            "none",
        )
    ).lower()
    return level != "none"


def _render_dispatch_single(state: UltraGSOfflineState):
    which = str(getattr(state.gauss_container.model_cfg.render, "which_render", "legacy")).lower()
    if which == "integrated":
        return render_integrated_debug if _want_debug_intermediates(state) else render_integrated_product
    return render


def _render_dispatch_batch(state: UltraGSOfflineState):
    mc = state.gauss_container.model_cfg
    which = str(getattr(mc.render, "which_render", "legacy")).lower()
    if which != "integrated":
        return None
    if not bool(getattr(mc.render, "batched_kernel", False)):
        return None
    if int(getattr(mc.render, "batch_size", 1) or 1) <= 1:
        return None
    return render_integrated_batch_debug if _want_debug_intermediates(state) else render_integrated_batch_product


def _build_local_windows(num_frames: int, window_size: int, overlap_k: int) -> List[Tuple[int, int]]:
    if num_frames <= 0:
        return []
    w = max(1, min(int(window_size), int(num_frames)))
    k = max(0, min(int(overlap_k), w - 1))
    stride = max(1, w - k)
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + w - 1, num_frames - 1)
        windows.append((start, end))
        if end >= num_frames - 1:
            break
        start += stride
    return windows


def _active_union(active_chunks: List[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if not active_chunks:
        return None
    merged = []
    for t in active_chunks:
        if t is None or t.numel() == 0:
            continue
        merged.append(t.to(device=device, dtype=torch.long).view(-1))
    if not merged:
        return None
    return torch.unique(torch.cat(merged, dim=0))


def _build_active_union_and_mask(
    subsets: List[Optional[torch.Tensor]],
    device: torch.device,
    num_points: int,
    pos_map_buf: Optional[torch.Tensor] = None,
    prev_touched: Optional[torch.Tensor] = None,
    union_mark_buf: Optional[torch.Tensor] = None,
    union_mark_token: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    valid = [s.to(device=device, dtype=torch.long).view(-1) for s in subsets if s is not None and s.numel() > 0]
    if not valid:
        return None, None, pos_map_buf, prev_touched, union_mark_buf, union_mark_token
    if union_mark_buf is None or int(union_mark_buf.numel()) != int(num_points):
        union_mark_buf = torch.zeros((int(num_points),), device=device, dtype=torch.int32)
        union_mark_token = 0
    union_mark_token += 1
    if union_mark_token >= (2**30):
        union_mark_buf.zero_()
        union_mark_token = 1
    union_chunks: List[torch.Tensor] = []
    for s in valid:
        s_local = torch.unique(s)
        marks = union_mark_buf.index_select(0, s_local)
        add = s_local[marks != union_mark_token]
        if add.numel() > 0:
            union_mark_buf.index_fill_(0, add, union_mark_token)
            union_chunks.append(add)
    if not union_chunks:
        return None, None, pos_map_buf, prev_touched, union_mark_buf, union_mark_token
    union = torch.cat(union_chunks, dim=0)
    if union.numel() == 0:
        return None, None, pos_map_buf, prev_touched, union_mark_buf, union_mark_token
    mask = torch.zeros((len(subsets), int(union.numel())), device=device, dtype=torch.uint8)
    if pos_map_buf is None or int(pos_map_buf.numel()) != int(num_points):
        pos_map_buf = torch.full((int(num_points),), -1, device=device, dtype=torch.long)
        prev_touched = None
    if prev_touched is not None and prev_touched.numel() > 0:
        pos_map_buf.index_fill_(0, prev_touched, -1)
    pos_map_buf.index_copy_(
        0,
        union,
        torch.arange(int(union.numel()), device=device, dtype=torch.long),
    )
    prev_touched = union
    for bi, s in enumerate(subsets):
        if s is None or s.numel() == 0:
            # fallback to full union for this sample
            mask[bi].fill_(1)
            continue
        s_local = torch.unique(s.to(device=device, dtype=torch.long).view(-1))
        pos = pos_map_buf.index_select(0, s_local)
        pos = pos[pos >= 0]
        if pos.numel() > 0:
            mask[bi, pos] = 1
    return union, mask, pos_map_buf, prev_touched, union_mark_buf, union_mark_token




def _build_window_subset_cache(
    windows: List[Tuple[int, int]],
    *,
    gsm,
    use_subset: bool,
    expand_m: int,
) -> Dict[Tuple[int, int], Optional[torch.Tensor]]:
    cache: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
    if not use_subset:
        for w in windows:
            cache[w] = None
        return cache
    for l, r in windows:
        idx = gsm.gauss_indices_for_window(int(l), int(r), int(expand_m))
        if idx is not None and idx.numel() == 0:
            idx = None
        cache[(l, r)] = idx
    return cache


@register_callback("UltraGSOfflineLocalWindowSetupCallback")
class UltraGSOfflineLocalWindowSetupCallback:
    priority = 10

    def on_setup(self, state: UltraGSOfflineState):
        print(f"[UltraGS-LW] Setting up components for device: {state.device}")
        cfg = state.cfg
        state.scene = SceneFactory.build_scene(cfg, state.device)
        train_indices = getattr(state.scene, "train_indices", None)
        if train_indices is None:
            state.local_train_indices = list(range(len(state.scene.train_container)))
        else:
            state.local_train_indices = [int(i) for i in train_indices]
        if len(state.local_train_indices) == 0:
            raise RuntimeError("No train indices available for local-window training.")
        train_n = len(state.local_train_indices)
        test_n = len(state.scene.test_container)
        print(
            "[UltraGS-LW] Offline data split:"
            f" train={train_n} ({cfg.scene.loading.images_name}),"
            f" test={test_n} ({cfg.scene.loading.test_images_name})"
        )

        state.gauss_container = UltraGSGaussContainer(cfg, device=state.device)
        init_data = state.scene.build_init_gaussians()
        state.gauss_container.init_from_scene(init_data["gaussians"])
        init_birth_ids = init_data.get("gaussian_birth_frame_ids")

        state.gauss_solver = UltraGSGaussSolver(cfg, state.gauss_container)
        state.gauss_solver.setup()
        state.ckpt_items = [("container", state.gauss_container), ("solver", state.gauss_solver)]

        state.dc = UltraGSOfflineDensityController(cfg)
        state.loss_computer = LossComputer(cfg, state.device)
        state.metric_computer = MetricComputer(cfg, state.device)

        lw_cfg = cfg.model.offline.local_window
        state.local_windows = _build_local_windows(
            num_frames=train_n,
            window_size=int(lw_cfg.window_size_frames),
            overlap_k=int(lw_cfg.overlap_k_frames),
        )
        state.local_window_rng = random.Random(int(cfg.system.seed))
        state.gsm = build_gaussian_state_manager(
            mode="slice_identity_no_exclude",
            device=state.device,
            membership_k=0,
            sub_locality_l=0,
            slice_identity_exclude_num_ids=0,
            spatial_chunk_size_mm=1.0,
            spatial_active_half_width_mm=1.0,
            spatial_scale_margin_ratio=1.0,
        )
        state.gsm.initialize(num_points=state.gauss_container.num_points, birth_frame_id=0)
        if (
            init_birth_ids is not None
            and hasattr(state.gsm, "slice_ids")
            and int(init_birth_ids.numel()) == int(state.gauss_container.num_points)
        ):
            state.gsm.slice_ids = init_birth_ids.to(device=state.device, dtype=torch.long)
            if hasattr(state.gsm, "_invalidate_cache"):
                state.gsm._invalidate_cache()

        resume = cfg.engine.resume
        if resume:
            state.iteration = state.io_manager.load_checkpoint(state.ckpt_items, resume)
        print(
            "[UltraGS-LW] Components initialized successfully."
            f" windows={len(state.local_windows)},"
            f" window_size={int(lw_cfg.window_size_frames)},"
            f" overlap={int(lw_cfg.overlap_k_frames)},"
            f" train_index_mode={'holdout' if train_indices is not None else 'full'}"
        )


@register_callback("UltraGSOfflineLocalWindowTrainCallback")
class UltraGSOfflineLocalWindowTrainCallback:
    priority = 50

    def on_train(self, state: UltraGSOfflineState):
        cfg = state.cfg
        lw_cfg = cfg.model.offline.local_window
        iterations = cfg.engine.iterations
        accum_steps = cfg.engine.accum_steps
        intv = cfg.engine.interval
        step_intv = intv.train.step
        print_intv = intv.train.print
        test_intv = intv.test
        save_intv = intv.save

        windows: List[Tuple[int, int]] = list(state.local_windows)
        if not windows:
            raise RuntimeError("Local window list is empty, cannot train.")
        total_windows = len(windows)
        num_windows_per_iter = int(getattr(lw_cfg, "num_windows_per_iter", 0) or 0)
        subset_cache_rebuild_every = int(getattr(lw_cfg, "subset_cache_rebuild_every", 0) or 0)
        global_refresh_every = int(getattr(lw_cfg, "global_refresh_every", 0) or 0)
        subset_cache = _build_window_subset_cache(
            windows,
            gsm=state.gsm,
            use_subset=bool(lw_cfg.use_gauss_subset),
            expand_m=int(lw_cfg.gauss_expand_m_frames),
        )
        subset_cache_dirty = False

        start_iter = state.iteration
        pbar = tqdm(range(start_iter + 1, iterations + 1), initial=start_iter, total=iterations)
        state.scalar_manager.tqdm_bar = pbar
        state.gauss_solver.zero_grad()
        batched_render_fn = _render_dispatch_batch(state)
        batch_size = int(getattr(state.gauss_container.model_cfg.render, "batch_size", 1) or 1)
        union_pos_map_buf: Optional[torch.Tensor] = None
        union_prev_touched: Optional[torch.Tensor] = None
        union_mark_buf: Optional[torch.Tensor] = None
        union_mark_token: int = 0

        for iteration in pbar:
            state.iteration = iteration
            state.gauss_solver.step_lr(iteration)
            if subset_cache_rebuild_every > 0 and (iteration % subset_cache_rebuild_every == 0):
                subset_cache_dirty = True
            if subset_cache_dirty:
                subset_cache = _build_window_subset_cache(
                    windows,
                    gsm=state.gsm,
                    use_subset=bool(lw_cfg.use_gauss_subset),
                    expand_m=int(lw_cfg.gauss_expand_m_frames),
                )
                subset_cache_dirty = False

            if num_windows_per_iter > 0 and num_windows_per_iter < total_windows:
                active_windows = state.local_window_rng.sample(windows, num_windows_per_iter)
            else:
                active_windows = windows
            n_used_windows = max(1, len(active_windows))

            active_chunks: List[torch.Tensor] = []
            per_window_losses: List[torch.Tensor] = []
            loss_sums: Dict[str, float] = {}
            active_sizes: List[int] = []
            do_global_refresh = bool(global_refresh_every > 0 and iteration % global_refresh_every == 0)

            sampled_slice_metas: List = []
            sampled_subsets: List[Optional[torch.Tensor]] = []
            for l, r in active_windows:
                local_fid = state.local_window_rng.randint(int(l), int(r))
                global_fid = int(state.local_train_indices[local_fid])
                sampled_slice_metas.append(state.scene.train_container.get(global_fid))
                gauss_indices = None if do_global_refresh else subset_cache.get((l, r), None)
                sampled_subsets.append(gauss_indices)
                if gauss_indices is not None and gauss_indices.numel() > 0:
                    active_chunks.append(gauss_indices)
                    active_sizes.append(int(gauss_indices.numel()))
                else:
                    active_sizes.append(int(state.gauss_container.num_points))

            if batched_render_fn is not None and len(sampled_slice_metas) > 0:
                # Batched kernel only supports one active index set for the whole batch.
                # Use union of all sampled local-window subsets in this iteration.
                for i0 in range(0, len(sampled_slice_metas), batch_size):
                    metas_chunk = sampled_slice_metas[i0 : i0 + batch_size]
                    subsets_chunk = sampled_subsets[i0 : i0 + batch_size]
                    (
                        active_union_for_batch,
                        active_mask_for_batch,
                        union_pos_map_buf,
                        union_prev_touched,
                        union_mark_buf,
                        union_mark_token,
                    ) = _build_active_union_and_mask(
                        subsets_chunk,
                        state.device,
                        int(state.gauss_container.num_points),
                        pos_map_buf=union_pos_map_buf,
                        prev_touched=union_prev_touched,
                        union_mark_buf=union_mark_buf,
                        union_mark_token=union_mark_token,
                    )
                    # Do not reuse writable stack buffers across micro-batches in one iteration.
                    # Otherwise, later in-place overwrite may invalidate earlier autograd graph versions.
                    chunk = {"stacked": stack_slice_metas(metas_chunk, buffers=None)}
                    pred_image, depth_image, intermediates = batched_render_fn(
                        state.gauss_container.model_cfg,
                        state.scene.dataset_meta,
                        chunk,
                        state.gauss_container,
                        active_global_indices=active_union_for_batch,
                        active_gauss_mask_batched=active_mask_for_batch,
                        is_jitter=True,
                    )
                    gt_image = chunk["stacked"]["image"]
                    state.inputs["gt_image"] = gt_image
                    state.outputs["pred_image"] = pred_image
                    state.outputs["depth_image"] = depth_image
                    state.outputs["gauss_scales"] = state.gauss_container.scales
                    state.outputs["gauss_opacities"] = state.gauss_container.opacities
                    loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
                    per_window_losses.append(loss_dict["total"])
                    for k, v in loss_dict.items():
                        loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().item())
                    seen = intermediates.get("seen_gauss_global_indices") if intermediates is not None else None
                    if seen is not None and seen.numel() > 0:
                        active_chunks.append(seen)
            else:
                render_single = _render_dispatch_single(state)
                for slice_meta, gauss_indices in zip(sampled_slice_metas, sampled_subsets):
                    pred_image, depth_image, intermediates = render_single(
                        state.gauss_container.model_cfg,
                        state.scene.dataset_meta,
                        slice_meta,
                        state.gauss_container,
                        active_global_indices=gauss_indices,
                        is_jitter=True,
                        return_intermediates=True,
                    )
                    gt_image = slice_meta.image
                    state.inputs["gt_image"] = gt_image
                    state.outputs["pred_image"] = pred_image
                    state.outputs["depth_image"] = depth_image
                    state.outputs["gauss_scales"] = state.gauss_container.scales
                    state.outputs["gauss_opacities"] = state.gauss_container.opacities
                    loss_dict = state.loss_computer.compute(state.inputs, state.outputs)
                    per_window_losses.append(loss_dict["total"])
                    for k, v in loss_dict.items():
                        loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().item())

                    seen = intermediates.get("seen_gauss_global_indices") if intermediates is not None else None
                    if seen is not None and seen.numel() > 0:
                        active_chunks.append(seen)

            recon_loss = torch.stack(per_window_losses, dim=0).mean()
            seam_loss = torch.tensor(0.0, device=state.device)
            if bool(lw_cfg.joint_opt.enable) and len(active_windows) > 1:
                seam_acc = torch.tensor(0.0, device=state.device)
                seam_cnt = 0
                sorted_active_windows = sorted(active_windows, key=lambda x: x[0])
                for wi in range(len(sorted_active_windows) - 1):
                    l0, r0 = sorted_active_windows[wi]
                    l1, r1 = sorted_active_windows[wi + 1]
                    ov_l = max(l0, l1)
                    ov_r = min(r0, r1)
                    if ov_l > ov_r:
                        continue
                    n_samples = max(1, int(lw_cfg.joint_opt.num_overlap_samples))
                    for _ in range(n_samples):
                        local_fid = state.local_window_rng.randint(int(ov_l), int(ov_r))
                        global_fid = int(state.local_train_indices[local_fid])
                        sm = state.scene.train_container.get(global_fid)
                        idx0 = None if do_global_refresh else subset_cache.get((l0, r0), None)
                        idx1 = None if do_global_refresh else subset_cache.get((l1, r1), None)
                        if idx0 is not None and idx0.numel() > 0:
                            active_chunks.append(idx0)
                        if idx1 is not None and idx1.numel() > 0:
                            active_chunks.append(idx1)
                        pred0, _, _ = _render_dispatch_single(state)(
                            state.gauss_container.model_cfg,
                            state.scene.dataset_meta,
                            sm,
                            state.gauss_container,
                            active_global_indices=idx0,
                            is_jitter=False,
                            return_intermediates=False,
                        )
                        pred1, _, _ = _render_dispatch_single(state)(
                            state.gauss_container.model_cfg,
                            state.scene.dataset_meta,
                            sm,
                            state.gauss_container,
                            active_global_indices=idx1,
                            is_jitter=False,
                            return_intermediates=False,
                        )
                        seam_acc = seam_acc + torch.mean(torch.abs(pred0 - pred1))
                        seam_cnt += 1
                if seam_cnt > 0:
                    seam_loss = seam_acc / float(seam_cnt)

            total_loss = recon_loss + float(lw_cfg.joint_opt.lambda_seam) * seam_loss
            (total_loss / accum_steps).backward()

            if iteration % accum_steps == 0:
                active_union = _active_union(active_chunks, state.device)
                if state.dc.should_accumulate_grad_stats(iteration):
                    state.dc.update_grad_stats(state.gauss_container, active_union)
                state.gauss_solver.step()
                state.gauss_solver.zero_grad()
                if state.dc.should_densify_prune(iteration):
                    state.dc.prune_and_densify(
                        iteration,
                        state.gauss_solver,
                        state.gauss_container,
                        gsm=state.gsm,
                        birth_frame_id=None,
                        active_global_indices=active_union,
                    )
                    # DC may change gauss topology; cached subset indices become stale.
                    subset_cache_dirty = True

                if iteration % step_intv == 0:
                    lr = state.gauss_solver.optimizer.param_groups[0]["lr"]
                    num_g = state.gauss_container.num_points
                    avg_losses = {
                        f"train/loss/{k}": float(v) / float(n_used_windows)
                        for k, v in loss_sums.items()
                        if k != "total"
                    }
                    stats = {
                        "train/loss/total": float(total_loss.detach().item()),
                        "train/loss/recon": float(recon_loss.detach().item()),
                        "train/loss/seam": float(seam_loss.detach().item()),
                        "train/lr": lr,
                        "train/num_gaussians": num_g,
                        "train/local_window/num_windows": float(n_used_windows),
                        "train/local_window/total_windows": float(total_windows),
                        "train/local_window/active_gauss_mean": float(np.mean(active_sizes)),
                        "train/local_window/do_global_refresh": 1.0 if do_global_refresh else 0.0,
                    }
                    if active_union is not None and num_g > 0:
                        stats["train/local_window/active_gauss_union_ratio"] = float(
                            active_union.numel() / float(num_g)
                        )
                    stats.update(avg_losses)
                    state.scalar_manager.log_scalars(stats, iteration)
                    if iteration % print_intv == 0:
                        state.scalar_manager.print_train(iteration)

                if test_intv and iteration % test_intv == 0:
                    _run_test(state, iteration=iteration)
                if save_intv and iteration % save_intv == 0:
                    _save_checkpoint(state, iteration)

            if state.stop_training:
                break


@register_callback("UltraGSOfflineLocalWindowTestCallback")
class UltraGSOfflineLocalWindowTestCallback:
    priority = 50

    def on_test(self, state: UltraGSOfflineState):
        _run_test(state, iteration=None)


@register_callback("UltraGSOfflineLocalWindowSaveCallback")
class UltraGSOfflineLocalWindowSaveCallback:
    priority = 50

    def on_save(self, state: UltraGSOfflineState, iteration: int):
        _save_checkpoint(state, iteration)


@register_callback("UltraGSOfflineLocalWindowFinishCallback")
class UltraGSOfflineLocalWindowFinishCallback:
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
        "[UltraGS-LW] Eval source:"
        f" images={cfg.scene.loading.test_images_name},"
        f" poses={cfg.scene.loading.test_poses_name}"
    )
    vis_dir = os.path.join(cfg.system.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    count = 0
    metric_sums: Dict[str, float] = {}

    with torch.no_grad():
        while True:
            datum = state.scene.next_test_datum()
            if datum is None:
                break
            pred, _, _ = _render_dispatch_single(state)(
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
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
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
