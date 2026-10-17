#!/usr/bin/env python
"""Visualize Ours responsibility density-control decisions from checkpoints.

This script is a post-hoc Stage-C diagnostic: it rerenders selected frames,
recomputes the same per-Gaussian residual responsibility score used by DC, and
projects selected/high-score Gaussians back onto the slice image plane.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

OPS_DIR = PROJECT_DIR / "ops" / "ours_renderer"
if OPS_DIR.is_dir() and str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from configs.setup import prepare_config  # noqa: E402
from methods.ours.dc_diagnostics import compute_contribution_residual_stats  # noqa: E402
from methods.ours.gaussian_container import OursGaussContainer  # noqa: E402
from methods.ours.render import render  # noqa: E402
from scene.factory import SceneFactory  # noqa: E402


def _as_np2d(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            arr = arr.mean(axis=0)
        elif arr.shape[-1] in (1, 3):
            arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D image, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _normalize(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(im, mode="L").save(path)


def _heat_color(x: np.ndarray) -> np.ndarray:
    """Small matplotlib-free magma-like heatmap."""
    x = np.clip(x.astype(np.float32, copy=False), 0.0, 1.0)
    stops = np.array(
        [
            [0.000, 0.000, 0.020],
            [0.180, 0.055, 0.330],
            [0.500, 0.080, 0.360],
            [0.820, 0.240, 0.170],
            [0.980, 0.620, 0.120],
            [1.000, 0.960, 0.620],
        ],
        dtype=np.float32,
    )
    pos = x * (len(stops) - 1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, len(stops) - 1)
    t = (pos - i0)[..., None]
    rgb = stops[i0] * (1.0 - t) + stops[i1] * t
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _save_heat(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_heat_color(_normalize(arr)), mode="RGB").save(path)


def _overlay_heat(gray: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> Image.Image:
    base = np.repeat((np.clip(gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[..., None], 3, axis=-1)
    h = _heat_color(_normalize(heat))
    mask = (_normalize(heat) > 0.02)[..., None].astype(np.float32)
    out = base.astype(np.float32) * (1.0 - alpha * mask) + h.astype(np.float32) * (alpha * mask)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.nan_to_num(arr.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1.0)),
        "p05": float(np.percentile(arr, 5.0)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _default_frames(n_frames: int, trim_each: int, count: int) -> list[int]:
    lo = int(max(0, trim_each))
    hi = int(max(lo, n_frames - 1 - trim_each))
    xs = np.linspace(lo, hi, int(count), dtype=np.float64)
    return [int(v) for v in np.unique(np.rint(xs).astype(np.int64)).tolist()]


def _responsibility_score(contribution, residual_contribution, max_hits, support, rcfg):
    del max_hits, support
    residual_mean = residual_contribution / contribution.clamp_min(1e-8)
    score = residual_mean * torch.log1p(contribution)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    positive = score[score > 0]
    if positive.numel() == 0:
        threshold = torch.tensor(float("inf"), device=score.device, dtype=score.dtype)
    else:
        q = min(max(float(getattr(rcfg, "score_quantile", 0.98)), 0.0), 1.0)
        threshold = torch.quantile(positive, q)
        max_fraction = float(getattr(rcfg, "max_fraction", 0.05))
        if 0.0 < max_fraction < 1.0:
            k = max(1, int(score.numel() * max_fraction))
            if positive.numel() > k:
                threshold = torch.maximum(threshold, torch.topk(score, k=k, largest=True).values[-1])
        min_score = float(getattr(rcfg, "min_score", 1e-6))
        threshold = torch.maximum(threshold, torch.tensor(min_score, device=score.device, dtype=score.dtype))
    selected = score >= threshold
    return score, selected, threshold


def _project_to_slice(container: OursGaussContainer, slice_meta, dataset_meta):
    means = container.means.detach()
    pose_inv = torch.linalg.inv(slice_meta.pose.to(device=means.device, dtype=means.dtype))
    ones = torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype)
    local = (pose_inv @ torch.cat([means, ones], dim=1).t()).t()[:, :3]
    w_offset = (int(dataset_meta.W) - 1) * 0.5 if bool(getattr(dataset_meta, "center_lateral", True)) else 0.0
    u = local[:, 0] / float(dataset_meta.delta_lateral_mm) + float(w_offset)
    v = (local[:, 1] - float(dataset_meta.near_mm)) / float(dataset_meta.delta_axial_mm)
    elev = local[:, 2]
    return u.detach().cpu().numpy(), v.detach().cpu().numpy(), elev.detach().cpu().numpy()


def _projected_score_map(
    H: int,
    W: int,
    u: np.ndarray,
    v: np.ndarray,
    score: np.ndarray,
    selected: np.ndarray,
    max_points: int,
) -> np.ndarray:
    valid = selected & np.isfinite(u) & np.isfinite(v) & np.isfinite(score)
    valid &= (u >= -4) & (u < W + 4) & (v >= -4) & (v < H + 4) & (score > 0)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    if idx.size > max_points:
        keep = np.argsort(score[idx])[-max_points:]
        idx = idx[keep]
    out = np.zeros((H, W), dtype=np.float32)
    score_sel = score[idx]
    score_scale = max(float(np.percentile(score_sel, 99.0)), 1e-8)
    for g in idx:
        cu = float(u[g])
        cv = float(v[g])
        amp = float(np.clip(score[g] / score_scale, 0.0, 1.0))
        x0 = int(round(cu))
        y0 = int(round(cv))
        radius = 3
        for yy in range(max(0, y0 - radius), min(H, y0 + radius + 1)):
            dy = yy - cv
            for xx in range(max(0, x0 - radius), min(W, x0 + radius + 1)):
                dx = xx - cu
                w = math.exp(-(dx * dx + dy * dy) / 4.0)
                out[yy, xx] += amp * w
    return out


def _draw_selected_overlay(
    base: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    score: np.ndarray,
    selected: np.ndarray,
    max_points: int,
) -> Image.Image:
    rgb = Image.fromarray(
        np.repeat((np.clip(base, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[..., None], 3, axis=-1),
        mode="RGB",
    )
    draw = ImageDraw.Draw(rgb, mode="RGBA")
    valid = selected & np.isfinite(u) & np.isfinite(v) & np.isfinite(score) & (score > 0)
    valid &= (u >= 0) & (u < base.shape[1]) & (v >= 0) & (v < base.shape[0])
    idx = np.flatnonzero(valid)
    if idx.size > max_points:
        keep = np.argsort(score[idx])[-max_points:]
        idx = idx[keep]
    if idx.size == 0:
        return rgb
    s = score[idx]
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    denom = max(s_max - s_min, 1e-8)
    for g in idx:
        t = float((score[g] - s_min) / denom)
        color = (255, int(70 + 170 * t), 30, 170)
        x = float(u[g])
        y = float(v[g])
        r = 2.0 + 2.0 * t
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return rgb


def _save_hist(path: Path, score: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = score[np.isfinite(score) & (score > 0)]
    W, H = 720, 420
    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle((55, 25, W - 25, H - 55), outline=(20, 20, 20), width=1)
    if positive.size == 0:
        draw.text((70, 50), "no positive scores", fill=(0, 0, 0))
        im.save(path)
        return
    vals = np.log10(positive + 1e-12)
    hist, edges = np.histogram(vals, bins=80)
    ymax = max(int(hist.max()), 1)
    x0, y0, x1, y1 = 55, 25, W - 25, H - 55
    for i, h in enumerate(hist):
        xa = x0 + (x1 - x0) * i / len(hist)
        xb = x0 + (x1 - x0) * (i + 1) / len(hist)
        ya = y1 - (y1 - y0) * int(h) / ymax
        draw.rectangle((xa, ya, xb, y1), fill=(78, 121, 167))
    th_log = math.log10(max(float(threshold), 1e-12))
    if edges[0] <= th_log <= edges[-1]:
        xt = x0 + (x1 - x0) * (th_log - edges[0]) / (edges[-1] - edges[0] + 1e-12)
        draw.line((xt, y0, xt, y1), fill=(220, 80, 20), width=3)
        draw.text((xt + 5, y0 + 8), "threshold", fill=(220, 80, 20))
    draw.text((55, H - 42), "log10(responsibility score)", fill=(0, 0, 0))
    draw.text((60, 8), f"positive={positive.size}  threshold={threshold:.4g}", fill=(0, 0, 0))
    im.save(path)


def _make_panel(path: Path, images: list[tuple[str, Image.Image]]) -> None:
    if not images:
        return
    cell_w = max(im.width for _, im in images)
    cell_h = max(im.height for _, im in images)
    header = 28
    out = Image.new("RGB", (cell_w * len(images), cell_h + header), "white")
    draw = ImageDraw.Draw(out)
    for i, (title, im) in enumerate(images):
        x = i * cell_w
        draw.text((x + 6, 6), title, fill=(0, 0, 0))
        if im.mode != "RGB":
            im = im.convert("RGB")
        out.paste(im, (x, header))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--frames", nargs="*", type=int, default=None)
    parser.add_argument("--num-auto-frames", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-overlay-points", type=int, default=3000)
    parser.add_argument("--max-scoremap-points", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.run_dir / "resolved_config.yaml"
    ckpt_path = args.run_dir / "checkpoints" / f"iteration_{args.iteration}_container.pth"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    cfg = prepare_config(OmegaConf.load(cfg_path))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("responsibility DC visualization requires CUDA for contribution stats")
    torch.set_grad_enabled(False)

    scene = SceneFactory.build_scene(cfg, device=device)
    container = OursGaussContainer(cfg, device=device)
    container.load(str(ckpt_path))
    container.eval()

    frames = args.frames
    if not frames:
        frames = _default_frames(
            len(scene.train_container),
            int(cfg.scene.loading.holdout_trim_each),
            int(args.num_auto_frames),
        )

    rcfg = cfg.model.offline.density_control.responsibility
    out_rows: list[dict[str, object]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        slice_meta = scene.train_container.get(int(frame))
        pred, _, _ = render(
            container.model_cfg,
            scene.dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )
        gt = slice_meta.image
        residual = (pred - gt).abs()
        contribution, residual_contribution, max_hits, support, diag_stats = compute_contribution_residual_stats(
            container.model_cfg,
            scene.dataset_meta,
            slice_meta,
            container,
            pred,
            gt,
            float(getattr(rcfg, "high_residual_quantile", 0.90)),
        )
        score, selected, threshold = _responsibility_score(
            contribution, residual_contribution, max_hits, support, rcfg
        )

        gt_np = _as_np2d(gt)
        pred_np = _as_np2d(pred)
        residual_np = _as_np2d(residual)
        score_np = score.detach().float().cpu().numpy()
        selected_np = selected.detach().cpu().numpy().astype(bool)
        u, v, elev = _project_to_slice(container, slice_meta, scene.dataset_meta)
        score_map = _projected_score_map(
            int(scene.dataset_meta.H),
            int(scene.dataset_meta.W),
            u,
            v,
            score_np,
            selected_np,
            int(args.max_scoremap_points),
        )
        overlay = _draw_selected_overlay(
            gt_np,
            u,
            v,
            score_np,
            selected_np,
            int(args.max_overlay_points),
        )

        frame_dir = args.out_dir / f"frame_{int(frame):04d}_iter_{int(args.iteration)}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        _save_gray(frame_dir / "target.png", gt_np)
        _save_gray(frame_dir / "render.png", pred_np)
        _save_heat(frame_dir / "residual_heat.png", residual_np)
        _save_heat(frame_dir / "responsibility_score_projection.png", score_map)
        _overlay_heat(gt_np, residual_np).save(frame_dir / "target_residual_overlay.png")
        _overlay_heat(gt_np, score_map).save(frame_dir / "target_score_overlay.png")
        overlay.save(frame_dir / "selected_gaussians_overlay.png")
        _save_hist(frame_dir / "score_histogram.png", score_np, float(threshold.detach().item()))

        _make_panel(
            frame_dir / "panel.png",
            [
                ("target", Image.open(frame_dir / "target.png")),
                ("render", Image.open(frame_dir / "render.png")),
                ("residual", Image.open(frame_dir / "residual_heat.png")),
                ("score proj", Image.open(frame_dir / "responsibility_score_projection.png")),
                ("selected", overlay),
            ],
        )

        valid_selected = selected_np & np.isfinite(u) & np.isfinite(v)
        in_fov_selected = valid_selected & (u >= 0) & (u < scene.dataset_meta.W) & (v >= 0) & (v < scene.dataset_meta.H)
        row = {
            "frame": int(frame),
            "iteration": int(args.iteration),
            "num_gaussians": int(score_np.shape[0]),
            "selected": int(selected_np.sum()),
            "selected_in_fov": int(in_fov_selected.sum()),
            "threshold": float(threshold.detach().item()),
            "score_max": float(np.nanmax(score_np)) if score_np.size else 0.0,
            "score_mean": float(np.nanmean(score_np)) if score_np.size else 0.0,
            "contribution_mean": float(contribution.detach().mean().item()),
            "support_mean": float(support.detach().mean().item()),
            "max_hit_sum": float(max_hits.detach().sum().item()),
            "residual_mean": float(residual_np.mean()),
            "residual_p95": float(np.percentile(residual_np, 95.0)),
            "panel": str(frame_dir / "panel.png"),
            "diag_stats": diag_stats,
        }
        out_rows.append(row)
        with (frame_dir / "stats.json").open("w", encoding="utf-8") as f:
            json.dump({**row, "diag_stats": diag_stats, "residual": _stats(residual_np)}, f, indent=2)
        print(
            f"[VIS] frame={frame} selected={row['selected']} in_fov={row['selected_in_fov']} "
            f"threshold={row['threshold']:.4g} panel={frame_dir / 'panel.png'}"
        )

    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "iteration",
                "num_gaussians",
                "selected",
                "selected_in_fov",
                "threshold",
                "score_max",
                "score_mean",
                "contribution_mean",
                "support_mean",
                "max_hit_sum",
                "residual_mean",
                "residual_p95",
                "panel",
            ],
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"[VIS] summary={csv_path}")


if __name__ == "__main__":
    main()
