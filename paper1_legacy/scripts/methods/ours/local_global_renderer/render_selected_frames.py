#!/usr/bin/env python
"""Render selected frames from an existing Ours checkpoint for diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

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


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(path)


def _save_gray_norm(path: Path, arr: np.ndarray) -> None:
    arr = arr.astype(np.float32, copy=False)
    lo = float(np.nanpercentile(arr, 1.0))
    hi = float(np.nanpercentile(arr, 99.0))
    if hi <= lo + 1.0e-8:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - lo) / (hi - lo)
    _save_gray(path, norm)


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = arr.astype(np.float64, copy=False)
    return {
        "min": float(np.nanmin(arr)),
        "p01": float(np.nanpercentile(arr, 1.0)),
        "p05": float(np.nanpercentile(arr, 5.0)),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p95": float(np.nanpercentile(arr, 95.0)),
        "p99": float(np.nanpercentile(arr, 99.0)),
        "max": float(np.nanmax(arr)),
    }


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = float(np.sqrt(np.sum(af * af) * np.sum(bf * bf)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(af * bf) / denom)


def _build_holdout_indices(n_frames: int, trim_each: int, num_samples: int) -> list[int]:
    i_min = int(trim_each)
    i_max = int(n_frames - 1 - trim_each)
    picks = np.linspace(i_min, i_max, int(num_samples), dtype=np.float64)
    return [int(x) for x in np.unique(np.rint(picks).astype(np.int64)).tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--frames", required=True, nargs="+", type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-ray-diagnostics",
        action="store_true",
        help="Also save local-only render, final/local modulation ratio, and attenuation stats.",
    )
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
    torch.set_grad_enabled(False)

    scene = SceneFactory.build_scene(cfg, device=device)
    container = OursGaussContainer(cfg, device=device)
    container.load(str(ckpt_path))
    container.eval()
    local_model_cfg = copy.deepcopy(container.model_cfg)
    local_model_cfg.slice_render.ray_attenuation_enabled = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_ray_diagnostics and "attenuations" in container._param_dict:
        attenuation_np = container.attenuations.detach().cpu().numpy()
        with (args.out_dir / "attenuation_stats.json").open("w", encoding="utf-8") as f:
            json.dump(_stats(attenuation_np), f, indent=2)

    test_idx = _build_holdout_indices(
        len(scene.train_container),
        int(cfg.scene.loading.holdout_trim_each),
        int(cfg.scene.loading.holdout_num_samples),
    )
    print("test_index_map:")
    for item, frame in enumerate(test_idx, start=1):
        print(f"  item_{item}: frame={frame}")

    print("selected_frame_metrics:")
    for frame in args.frames:
        slice_meta = scene.train_container.get(int(frame))
        pred, _, _ = render(
            container.model_cfg,
            scene.dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )
        if args.save_ray_diagnostics:
            local_pred, _, _ = render(
                local_model_cfg,
                scene.dataset_meta,
                slice_meta,
                container,
                is_jitter=False,
                return_intermediates=False,
            )
        else:
            local_pred = None
        gt_np = slice_meta.image.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        diff_np = np.abs(pred_np - gt_np)
        compose = np.concatenate(
            [
                np.clip(gt_np, 0.0, 1.0),
                np.clip(pred_np, 0.0, 1.0),
                np.clip(diff_np * 4.0, 0.0, 1.0),
            ],
            axis=1,
        )

        frame_dir = args.out_dir / f"frame_{int(frame):04d}"
        _save_gray(frame_dir / "gt.png", gt_np)
        _save_gray(frame_dir / "pred.png", pred_np)
        _save_gray(frame_dir / "diff_x4.png", diff_np * 4.0)
        _save_gray(frame_dir / "compose.png", compose)
        if args.save_ray_diagnostics and local_pred is not None:
            local_np = local_pred.detach().cpu().numpy()
            modulation_np = pred_np / np.clip(local_np, 1.0e-6, None)
            modulation_np = np.clip(modulation_np, 0.0, 1.5)
            _save_gray(frame_dir / "local_only.png", local_np)
            _save_gray(frame_dir / "modulation.png", modulation_np)
            _save_gray_norm(frame_dir / "modulation_norm.png", modulation_np)
            with (frame_dir / "ray_stats.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "local_only": _stats(local_np),
                        "final": _stats(pred_np),
                        "modulation": _stats(modulation_np),
                    },
                    f,
                    indent=2,
                )

        print(
            f"  frame={int(frame)}"
            f" ncc={_ncc(pred_np, gt_np):.6f}"
            f" l1={float(np.mean(diff_np)):.6f}"
            f" out={frame_dir / 'compose.png'}"
        )


if __name__ == "__main__":
    main()
