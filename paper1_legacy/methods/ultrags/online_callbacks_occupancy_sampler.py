from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from scene.meta.slice_meta import SliceMeta
from scene.scene import Scene

from .config import StreamSeedConfig
from .render import occupancy_from_intermediates, render


def seed_pack_size(pack: Optional[Dict[str, torch.Tensor]]) -> int:
    if pack is None or "means" not in pack:
        return 0
    return int(pack["means"].shape[0])


@torch.no_grad()
def sample_stream_pixel_seed(
    scene: Scene,
    slice_meta: SliceMeta,
    num_to_sample: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """从当前帧像素做混合采样（与旧 Scene.next_stream_datum 内逻辑一致，不改像素世界坐标）。"""
    if num_to_sample <= 0:
        return None
    n_pixels = int(scene.container.meta.H * scene.container.meta.W)
    scales = scene._get_pixel_scales(scene.cfg.sampler.scales_init, n_pixels)
    quats = scene._get_pixel_quats(scene.cfg.sampler.quats_init, n_pixels)
    return scene.sampler.sample(
        pixel_positions=slice_meta.pixel_pos.to(scene.device),
        pixel_values=slice_meta.pixel_int.to(scene.device),
        pixel_quats=quats,
        pixel_scales=scales,
        num_to_sample=int(num_to_sample),
        jitter_radius=float(scene.cfg.sampler.jitter_radius),
    )


@torch.no_grad()
def gaussian_blur_hw(diff_hw: torch.Tensor, sigma: float) -> torch.Tensor:
    """对单通道 (H,W) 图像做可分离高斯平滑；``sigma<=0`` 原样返回。"""
    if sigma <= 0 or diff_hw.numel() == 0:
        return diff_hw
    r = max(1, int(math.ceil(3.0 * sigma)))
    k = 2 * r + 1
    device, dtype = diff_hw.device, diff_hw.dtype
    ax = torch.arange(k, device=device, dtype=dtype) - float(r)
    g = torch.exp(-((ax / sigma) ** 2) * 0.5)
    g = g / (g.sum() + 1e-12)
    g2 = (g[:, None] @ g[None, :]).view(1, 1, k, k)
    x = diff_hw.unsqueeze(0).unsqueeze(0)
    y = F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), g2)
    return y.squeeze(0).squeeze(0)


@torch.no_grad()
def gather_seed_at_pixel_indices(
    scene: Scene,
    slice_meta: SliceMeta,
    indices: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    """在指定像素下标处构造 concat 字典（与 ``UnifiedGaussianSampler`` 抖动规则一致）。"""
    if indices.numel() == 0:
        return None
    indices = indices.to(device=scene.device, dtype=torch.long).contiguous()
    n_pixels = int(scene.container.meta.H * scene.container.meta.W)
    pixel_pos = slice_meta.pixel_pos.to(scene.device)
    scales_full = scene._get_pixel_scales(scene.cfg.sampler.scales_init, n_pixels)
    quats_full = scene._get_pixel_quats(scene.cfg.sampler.quats_init, n_pixels)
    selected_means = pixel_pos[indices]
    quats = quats_full[indices]
    scales = scales_full[indices]
    jr = float(scene.cfg.sampler.jitter_radius)
    noise = (torch.rand_like(selected_means) * 2.0 - 1.0) * jr
    means = selected_means + noise
    return {"means": means, "scales": scales, "quats": quats}


@torch.no_grad()
def select_pixel_indices_occupancy_guided(
    state: Any,
    ss: StreamSeedConfig,
    get_slice_meta_for_training: Callable[[Any, int], SliceMeta],
    append_seed_debug_line: Callable[..., None],
) -> torch.Tensor:
    """全池预渲染：以 occupancy 覆盖度为主、|pred−GT| 为辅，分位 + Cap。"""
    mc = state.model_cfg
    dm = state.scene.dataset_meta
    H, W = int(dm.H), int(dm.W)
    P = H * W
    fid = int(state.stream_frame_id)
    sm = get_slice_meta_for_training(state, fid)
    pred, _, inter = render(
        mc,
        dm,
        sm,
        state.gauss_container,
        active_global_indices=None,
        is_jitter=False,
        return_intermediates=True,
    )
    occupancy = occupancy_from_intermediates(inter, H, W)
    lack = 1.0 - occupancy.clamp(0.0, 1.0)
    gt = sm.image.view(H, W)
    pr = pred.view(H, W)
    diff = (pr - gt).abs()
    diff = gaussian_blur_hw(diff, float(ss.diff_blur_sigma))
    lack_n = lack / (lack.max() + 1e-8)
    diff_n = diff / (diff.max() + 1e-8)
    lw = float(ss.occupancy_lack_weight)
    dw = float(ss.diff_weight)
    ssum = lw + dw
    if ssum <= 1e-12:
        lw, dw = 1.0, 0.0
    else:
        lw, dw = lw / ssum, dw / ssum
    score = lw * lack_n + dw * diff_n
    score_flat = score.reshape(-1)
    cap = max(0, int(ss.max_new_cap))
    if cap <= 0:
        append_seed_debug_line(
            state,
            "occ_select_skip_cap",
            cap=cap,
            quantile=float(ss.score_quantile),
            occupancy_mean=float(occupancy.mean().item()),
            occupancy_min=float(occupancy.min().item()),
            occupancy_max=float(occupancy.max().item()),
            lack_mean=float(lack.mean().item()),
            diff_mean=float(diff.mean().item()),
        )
        return torch.tensor([], dtype=torch.long, device=score_flat.device)
    q = float(ss.score_quantile)
    q = min(max(q, 0.0), 1.0)
    thr = torch.quantile(score_flat, q)
    mask = score_flat >= thr
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    before_cap = int(idx.numel())
    if idx.numel() > cap:
        sub = score_flat[idx]
        k = min(cap, int(sub.numel()))
        pick = torch.topk(sub, k).indices
        idx = idx[pick]
    elif idx.numel() == 0:
        k = min(cap, P)
        idx = torch.topk(score_flat, k).indices.flatten()
    append_seed_debug_line(
        state,
        "occ_select",
        quantile=q,
        threshold=float(thr.item()),
        cap=cap,
        selected_before_cap=before_cap,
        selected_after_cap=int(idx.numel()),
        occupancy_mean=float(occupancy.mean().item()),
        occupancy_min=float(occupancy.min().item()),
        occupancy_max=float(occupancy.max().item()),
        lack_mean=float(lack.mean().item()),
        lack_max=float(lack.max().item()),
        diff_mean=float(diff.mean().item()),
        diff_max=float(diff.max().item()),
        weight_lack=float(lw),
        weight_diff=float(dw),
    )
    return idx
