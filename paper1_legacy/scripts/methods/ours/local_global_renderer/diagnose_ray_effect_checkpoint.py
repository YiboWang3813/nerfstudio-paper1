#!/usr/bin/env python
"""Diagnose ray attenuation effect and gradients from an Ours checkpoint."""

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

PROJECT_DIR = Path(__file__).resolve().parents[4]
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
from methods.ours.render import render  # noqa: E402
from scene.factory import SceneFactory  # noqa: E402


def _stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float().flatten()
    if xf.numel() == 0:
        return {}
    qs = torch.quantile(xf, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=xf.device))
    return {
        "min": float(xf.min().item()),
        "p01": float(qs[0].item()),
        "p05": float(qs[1].item()),
        "mean": float(xf.mean().item()),
        "median": float(qs[2].item()),
        "p95": float(qs[3].item()),
        "p99": float(qs[4].item()),
        "max": float(xf.max().item()),
    }


def _grad_stats(x: torch.Tensor | None) -> dict[str, float]:
    if x is None:
        return {"mean_abs": 0.0, "rms": 0.0, "max_abs": 0.0, "nonzero_frac": 0.0}
    g = x.detach().float().flatten()
    if g.numel() == 0:
        return {"mean_abs": 0.0, "rms": 0.0, "max_abs": 0.0, "nonzero_frac": 0.0}
    abs_g = g.abs()
    return {
        "mean_abs": float(abs_g.mean().item()),
        "rms": float(torch.sqrt((g * g).mean()).item()),
        "max_abs": float(abs_g.max().item()),
        "nonzero_frac": float((abs_g > 0).float().mean().item()),
    }


def _holdout_indices(n_frames: int, trim_each: int, num_samples: int) -> list[int]:
    i_min = int(trim_each)
    i_max = int(n_frames - 1 - trim_each)
    picks = np.linspace(i_min, i_max, int(num_samples), dtype=np.float64)
    return [int(x) for x in np.unique(np.rint(picks).astype(np.int64)).tolist()]


def _parse_frames(raw: str | None, fallback: Iterable[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(fallback)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", default=2000, type=int)
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices. Default: spread holdout frames.")
    parser.add_argument("--num-frames", default=8, type=int)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.run_dir / "resolved_config.yaml"
    ckpt_path = args.run_dir / "checkpoints" / f"iteration_{args.iteration}_container.pth"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    raw_cfg = OmegaConf.load(cfg_path)
    cfg = prepare_config(raw_cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    scene = SceneFactory.build_scene(cfg, device=device)
    container = OursGaussContainer(cfg, device=device)
    container.load(str(ckpt_path))
    container.train()

    local_model_cfg = copy.deepcopy(container.model_cfg)
    local_model_cfg.slice_render.ray_attenuation_enabled = False

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

    modulation_chunks = []
    masked_modulation_chunks = []
    l1_terms = []
    local_l1_terms = []

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
        with torch.no_grad():
            local_pred, _, _ = render(
                local_model_cfg,
                scene.dataset_meta,
                slice_meta,
                container,
                is_jitter=False,
                return_intermediates=False,
            )
        gt = slice_meta.image.to(device=pred.device, dtype=pred.dtype)
        l1_terms.append(torch.mean(torch.abs(pred - gt)))
        local_l1_terms.append(torch.mean(torch.abs(local_pred - gt)).detach())

        with torch.no_grad():
            mod = pred.detach().float() / local_pred.detach().float().clamp_min(1e-6)
            modulation_chunks.append(mod.flatten().cpu())
            mask = local_pred.detach().float() > 1e-4
            masked_modulation_chunks.append(mod[mask].flatten().cpu())

    loss = torch.stack(l1_terms).mean()
    container.zero_grad(set_to_none=True)
    loss.backward()

    prop_stats = {}
    grad_stats = {}
    for name, wrapper in container._param_dict.items():
        prop_stats[name] = _stats(wrapper.value)
        grad_stats[name] = _grad_stats(wrapper.tensor.grad)

    modulation = torch.cat(modulation_chunks)
    masked_modulation = torch.cat(masked_modulation_chunks) if masked_modulation_chunks else modulation
    result = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "frames": frames,
        "ray_params": {
            "enabled": bool(container.model_cfg.slice_render.ray_attenuation_enabled),
            "beta": float(container.model_cfg.slice_render.ray_attenuation_beta),
            "strength": float(container.model_cfg.slice_render.ray_attenuation_strength),
            "min_transmission": float(container.model_cfg.slice_render.ray_attenuation_min_transmission),
            "quadrature_window_mm": float(container.model_cfg.slice_render.ray_attenuation_quadrature_window_mm),
            "quadrature_points": int(container.model_cfg.slice_render.ray_attenuation_quadrature_points),
        },
        "mean_l1_full": float(torch.stack([x.detach() for x in l1_terms]).mean().item()),
        "mean_l1_local_only_same_checkpoint": float(torch.stack(local_l1_terms).mean().item()),
        "modulation_all_pixels": _stats(modulation),
        "modulation_local_gt_1e-4": _stats(masked_modulation),
        "property_stats": prop_stats,
        "raw_parameter_grad_stats_from_l1": grad_stats,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
