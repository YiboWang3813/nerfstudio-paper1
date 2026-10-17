#!/usr/bin/env python
"""Profile Ours local/global ray renderer from an existing checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

OPS_DIRS = [
    PROJECT_DIR / "ops" / "ours_renderer",
    PROJECT_DIR / "ops" / "gaussian_renderer",
]
for path in OPS_DIRS:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.setup import prepare_config  # noqa: E402
from methods.ours.gaussian_container import OursGaussContainer  # noqa: E402
import ours_renderer._C as _C  # noqa: E402
from methods.ours.render import build_ours_render_config, build_slice_grid_geometry, render  # noqa: E402
from ours_renderer.autograd import _us_focus_params_vector  # noqa: E402
from ours_renderer.grid_geometry import shader_weight_mode_id  # noqa: E402
from scene.factory import SceneFactory  # noqa: E402


def _holdout_indices(n_frames: int, trim_each: int, num_samples: int) -> list[int]:
    i_min = int(trim_each)
    i_max = int(n_frames - 1 - trim_each)
    picks = np.linspace(i_min, i_max, int(num_samples), dtype=np.float64)
    return [int(x) for x in np.unique(np.rint(picks).astype(np.int64)).tolist()]


def _parse_frames(raw: str | None, fallback: Iterable[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(fallback)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = float(sum(values) / len(values))
    var = float(sum((x - mean) * (x - mean) for x in values) / len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": var ** 0.5,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--variant", choices=("config", "local", "ray"), default="config")
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices. Default: spread holdout frames.")
    parser.add_argument("--num-frames", default=8, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--repeats", default=20, type=int)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--profile-table", default=None, type=Path)
    parser.add_argument("--profile-forward-components", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ray-beta", default=None, type=float)
    parser.add_argument("--ray-strength", default=None, type=float)
    parser.add_argument("--ray-min-transmission", default=None, type=float)
    parser.add_argument("--ray-quadrature-points", default=None, type=int)
    parser.add_argument("--ray-formula", default="integral", choices=("integral", "slice_depth"))
    parser.add_argument("--ray-candidate-k-sigma", default=None, type=float)
    return parser.parse_args()


def _slice_render(cfg_like):
    return cfg_like.model.slice_render if hasattr(cfg_like, "model") else cfg_like.slice_render


def _apply_variant(cfg_like, args: argparse.Namespace) -> None:
    sr = _slice_render(cfg_like)
    if args.variant == "local":
        sr.ray_attenuation_enabled = False
        return
    if args.variant == "ray":
        sr.ray_attenuation_enabled = True
        sr.ray_attenuation_source = "attenuation"
        sr.ray_attenuation_mapper_kind = "ray_csr"
        sr.ray_attenuation_formula = str(args.ray_formula)
    if args.ray_beta is not None:
        sr.ray_attenuation_beta = float(args.ray_beta)
    if args.ray_strength is not None:
        sr.ray_attenuation_strength = float(args.ray_strength)
    if args.ray_min_transmission is not None:
        sr.ray_attenuation_min_transmission = float(args.ray_min_transmission)
    if args.ray_quadrature_points is not None:
        sr.ray_attenuation_quadrature_points = int(args.ray_quadrature_points)
    if args.ray_candidate_k_sigma is not None:
        sr.ray_attenuation_candidate_k_sigma = float(args.ray_candidate_k_sigma)


def _render_loss(model_cfg, scene, container, frame: int) -> torch.Tensor:
    slice_meta = scene.train_container.get(int(frame))
    pred, _, _ = render(
        model_cfg,
        scene.dataset_meta,
        slice_meta,
        container,
        is_jitter=False,
        return_intermediates=False,
    )
    gt = slice_meta.image.to(device=pred.device, dtype=pred.dtype)
    return torch.mean(torch.abs(pred - gt))


def _profile_forward_components(model_cfg, scene, container, frames: list[int], repeats: int) -> dict:
    r_cfg = build_ours_render_config(scene.dataset_meta, model_cfg)
    grid = build_slice_grid_geometry(scene.dataset_meta, container.means.device)
    us_params = _us_focus_params_vector(r_cfg, container.means.device)
    sr = model_cfg.slice_render
    ray_enabled = bool(sr.ray_attenuation_enabled)
    ray_mapper_kind = 1 if str(sr.ray_attenuation_mapper_kind) == "ray_csr" else 0
    ray_beta = float(sr.ray_attenuation_beta) if ray_enabled else 0.0
    ray_strength = float(sr.ray_attenuation_strength) if ray_enabled else 0.0
    rows = []
    stats_rows = []
    for i in range(max(1, int(repeats))):
        frame = frames[i % len(frames)]
        slice_meta = scene.train_container.get(int(frame))
        pose = slice_meta.pose.to(device=container.means.device, dtype=torch.float32).contiguous()
        timings, stats = _C.ProjectedSliceForwardRayTileProfileCUDA(
            container.means.float().contiguous(),
            container.scales.float().contiguous(),
            container.quats.float().contiguous(),
            container.opacities.float().contiguous(),
            container.intensities.float().contiguous(),
            container.attenuations.float().contiguous(),
            pose,
            grid.to(device=container.means.device, dtype=torch.float32).contiguous(),
            us_params,
            int(r_cfg.H),
            int(r_cfg.W),
            float(r_cfg.lateral_spacing_mm_per_pix),
            float(r_cfg.axial_spacing_mm_per_pix),
            float(r_cfg.axial_near_mm),
            int(r_cfg.tile_size_px),
            float(r_cfg.elev_corridor_half_mm),
            float(r_cfg.slab_half_thickness_mm),
            float(r_cfg.k_sigma),
            bool(r_cfg.mapper_sort_by_v),
            0 if r_cfg.compositing_mode == "additive" else 1,
            float(r_cfg.fixed_bg_color),
            int(shader_weight_mode_id(r_cfg.shader_weight_mode)),
            ray_beta,
            ray_strength,
            float(sr.ray_attenuation_min_transmission),
            float(sr.ray_attenuation_weight_min),
            float(sr.ray_attenuation_axial_margin_mm),
            float(sr.ray_attenuation_quadrature_window_mm),
            int(sr.ray_attenuation_quadrature_points),
            ray_mapper_kind,
            1 if str(sr.ray_attenuation_formula) == "slice_depth" else 0,
            float(sr.ray_attenuation_candidate_k_sigma),
        )
        rows.append(timings.cpu().tolist())
        stats_rows.append(stats.cpu().tolist())

    labels = [
        "local_mapper",
        "local_raster",
        "ray_total",
        "compose",
        "total",
        "local_total",
        "ray_count",
        "ray_scan",
        "ray_d2h_sync_wall",
        "ray_emit",
        "ray_sort",
        "ray_ranges",
        "ray_depths",
        "ray_modulation",
        "ray_csr_gpu_total",
        "ray_scan_query_wall",
        "ray_sort_query_wall",
    ]
    stat_labels = [
        "num_gaussians",
        "num_pixels",
        "csr0_numel",
        "csr1_numel",
        "csr2_numel",
        "gauss_rects",
        "ray_candidates",
        "ray_depth_values",
        "ray_profile_num_gaussians",
        "ray_profile_width",
        "ray_profile_columns",
        "ray_profile_candidates",
        "ray_scan_temp_bytes",
        "ray_sort_temp_bytes",
        "ray_overflow_candidates",
        "ray_fixed_capacity_slots",
    ]
    out = {}
    for idx, label in enumerate(labels):
        out[label] = _summarize_ms([float(row[idx]) for row in rows])
    stats_out = {}
    for idx, label in enumerate(stat_labels):
        vals = [float(row[idx]) for row in stats_rows]
        stats_out[label] = _summarize_ms(vals)
    return {"timing_ms": out, "stats": stats_out}


def main() -> None:
    args = parse_args()
    if args.ray_quadrature_points is not None and args.ray_quadrature_points not in (1, 2, 3):
        raise ValueError("--ray-quadrature-points must be 1, 2, or 3.")

    cfg_path = args.run_dir / "resolved_config.yaml"
    ckpt_path = args.run_dir / "checkpoints" / f"iteration_{args.iteration}_container.pth"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    raw_cfg = OmegaConf.load(cfg_path)
    cfg = prepare_config(raw_cfg)
    _apply_variant(cfg, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA profiling requires a CUDA device.")

    scene = SceneFactory.build_scene(cfg, device=device)
    container = OursGaussContainer(cfg, device=device)
    container.load(str(ckpt_path))
    container.train()
    model_cfg = copy.deepcopy(container.model_cfg)
    _apply_variant(model_cfg, args)

    holdout = _holdout_indices(
        len(scene.train_container),
        int(cfg.scene.loading.holdout_trim_each),
        int(cfg.scene.loading.holdout_num_samples),
    )
    if args.frames:
        frames = _parse_frames(args.frames, holdout)
    else:
        pick = np.linspace(0, len(holdout) - 1, int(args.num_frames), dtype=np.int64)
        frames = [holdout[int(i)] for i in pick]
    if not frames:
        raise RuntimeError("No frames selected for profiling.")

    torch.cuda.reset_peak_memory_stats(device)
    for i in range(max(0, int(args.warmup))):
        container.zero_grad(set_to_none=True)
        loss = _render_loss(model_cfg, scene, container, frames[i % len(frames)])
        loss.backward()
    torch.cuda.synchronize(device)

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    total_ms: list[float] = []
    losses: list[float] = []
    for i in range(max(1, int(args.repeats))):
        frame = frames[i % len(frames)]
        container.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        fwd_done = torch.cuda.Event(enable_timing=True)
        bwd_done = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = _render_loss(model_cfg, scene, container, frame)
        fwd_done.record()
        loss.backward()
        bwd_done.record()
        torch.cuda.synchronize(device)
        fwd_ms.append(float(start.elapsed_time(fwd_done)))
        bwd_ms.append(float(fwd_done.elapsed_time(bwd_done)))
        total_ms.append(float(start.elapsed_time(bwd_done)))
        losses.append(float(loss.detach().item()))

    profile_table = ""
    if args.profile_table is not None:
        container.zero_grad(set_to_none=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            loss = _render_loss(model_cfg, scene, container, frames[0])
            loss.backward()
        profile_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        args.profile_table.parent.mkdir(parents=True, exist_ok=True)
        args.profile_table.write_text(profile_table, encoding="utf-8")

    forward_components = {}
    if args.profile_forward_components:
        forward_components = _profile_forward_components(model_cfg, scene, container, frames, args.repeats)

    sr = model_cfg.slice_render
    result = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "variant": args.variant,
        "frames": frames,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "ray_params": {
            "enabled": bool(sr.ray_attenuation_enabled),
            "source": str(sr.ray_attenuation_source),
            "mapper": str(sr.ray_attenuation_mapper_kind),
            "beta": float(sr.ray_attenuation_beta),
            "strength": float(sr.ray_attenuation_strength),
            "min_transmission": float(sr.ray_attenuation_min_transmission),
            "quadrature_window_mm": float(sr.ray_attenuation_quadrature_window_mm),
            "quadrature_points": int(sr.ray_attenuation_quadrature_points),
            "formula": str(sr.ray_attenuation_formula),
            "candidate_k_sigma": float(sr.ray_attenuation_candidate_k_sigma),
        },
        "loss_l1": _summarize_ms(losses),
        "timing_ms": {
            "forward": _summarize_ms(fwd_ms),
            "backward": _summarize_ms(bwd_ms),
            "forward_backward": _summarize_ms(total_ms),
        },
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        "profile_table": str(args.profile_table) if args.profile_table is not None else "",
        "forward_components": forward_components,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if profile_table:
        print(profile_table)


if __name__ == "__main__":
    main()
