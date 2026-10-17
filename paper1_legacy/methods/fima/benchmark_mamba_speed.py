"""Small GPU benchmark for FiMA Mamba sequence PEM.

This script intentionally avoids the Engine/Scene stack so it can measure the
model step cost on pooled synthetic_liver windows without checkpoint/log noise.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from methods.fima.configs import FiMAConfig
from methods.fima.losses import pem_supervision_loss
from methods.fima.pose_estimator import FiMAPoseEstimator
from methods.fima.pose_geometry import adjacent_frame_transform_gt, six_dof_from_matrix


SWEEPS = ("l1", "l2", "l3", "r1", "r2", "r3")


def _holdout_indices(n: int, trim_each: int = 10, num_samples: int = 40) -> set[int]:
    picks = np.linspace(trim_each, n - 1 - trim_each, num_samples, dtype=np.float64)
    return set(int(x) for x in np.rint(picks).astype(np.int64).tolist())


def _load_pooled_windows(
    root: str,
    *,
    image_name: str,
    window: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for sweep in SWEEPS:
        d = os.path.join(root, sweep)
        images_np = np.load(os.path.join(d, image_name)).astype(np.float32, copy=False)
        poses_np = np.load(os.path.join(d, "relative_poses.npy")).astype(np.float32, copy=False)
        holdout = _holdout_indices(int(images_np.shape[0]))
        train = [i for i in range(int(images_np.shape[0])) if i not in holdout]
        train_set = set(train)
        # The GS holdout split is interleaved, so strict globally contiguous
        # train-only windows of length 32 do not exist. For this speed benchmark
        # we follow FiMASequenceSampler's train-pool behavior: windows are
        # contiguous in the ordered train frame list, not necessarily in global
        # frame id. This measures model cost without using test frames.
        for pos in range(int(window) - 1, len(train)):
            ids = train[pos - int(window) + 1 : pos + 1]
            img = torch.from_numpy(images_np[ids]).to(device=device, dtype=torch.float32)
            if img.max() > 2.0:
                img = img / 255.0
            img = img.unsqueeze(1)
            poses = torch.from_numpy(poses_np[ids]).to(device=device, dtype=torch.float32)
            edges = []
            for k in range(1, int(window)):
                edge = adjacent_frame_transform_gt(poses[k - 1], poses[k])
                edges.append(six_dof_from_matrix(edge))
            target = torch.stack(edges, dim=0)
            windows.append((img, target))
    if not windows:
        raise RuntimeError("No valid pooled training windows were built.")
    return windows


def _make_cfg(window: int, d_model: int, layers: int) -> FiMAConfig:
    cfg = FiMAConfig()
    cfg.pem.input_mode = "multiframe"
    cfg.pem.pem_output_layout = "window_edges"
    cfg.pem.window_size = int(window)
    cfg.pem.batch_size = 1
    cfg.pem.d_model = int(d_model)
    cfg.pem.num_mixer_layers = int(layers)
    cfg.pem.prefer_mamba = True
    cfg.pem.require_mamba = True
    cfg.pem_loss.use_l1 = True
    cfg.pem_loss.use_mse = False
    cfg.pem_loss.use_pearson = False
    cfg.pem_loss.standardize_targets = False
    return cfg


def _sample_batch(
    windows: List[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    size_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    items = random.choices(windows, k=int(batch_size))
    images = torch.stack([x[0] for x in items], dim=0)
    target = torch.stack([x[1] for x in items], dim=0)
    if tuple(images.shape[-2:]) != tuple(size_hw):
        b, n, c, h, w = images.shape
        images = F.interpolate(
            images.reshape(b * n, c, h, w),
            size=tuple(size_hw),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, n, c, int(size_hw[0]), int(size_hw[1]))
    return images, target


def _bench_one(
    windows: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    size_hw: Tuple[int, int],
    batch_size: int,
    window: int,
    d_model: int,
    layers: int,
    warmup: int,
    steps: int,
    device: torch.device,
) -> None:
    cfg = _make_cfg(window, d_model, layers)
    model = FiMAPoseEstimator(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(int(warmup)):
        images, target = _sample_batch(windows, batch_size, size_hw)
        out = model(images)
        loss, _ = pem_supervision_loss(out["pred_pose_6"], target, cfg.pem_loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_loss = None
    for _ in range(int(steps)):
        images, target = _sample_batch(windows, batch_size, size_hw)
        out = model(images)
        loss, _ = pem_supervision_loss(out["pred_pose_6"], target, cfg.pem_loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().item())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    it_s = float(steps) / max(dt, 1e-9)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024.0**3)
    est_40k_min = 40000.0 / it_s / 60.0
    print(
        "BENCH"
        f"\tsize={size_hw[0]}x{size_hw[1]}"
        f"\tbatch={batch_size}"
        f"\twindow={window}"
        f"\td_model={d_model}"
        f"\tlayers={layers}"
        f"\tsteps={steps}"
        f"\tit_per_s={it_s:.3f}"
        f"\tms_per_iter={1000.0 / it_s:.3f}"
        f"\test_40k_min={est_40k_min:.2f}"
        f"\tpeak_gb={peak_gb:.3f}"
        f"\tlast_loss={last_loss:.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver")
    ap.add_argument("--image-name", default="images_x2.npy")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--sizes", default="256x128,128x64,64x32")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mamba_ssm benchmark.")
    device = torch.device("cuda")
    torch.manual_seed(42)
    random.seed(42)
    windows = _load_pooled_windows(args.root, image_name=args.image_name, window=args.window, device=device)
    print(
        "DATA"
        f"\twindows={len(windows)}"
        f"\tsweeps={','.join(SWEEPS)}"
        f"\timage_name={args.image_name}"
        f"\tdevice={torch.cuda.get_device_name(0)}"
    )
    for spec in str(args.sizes).split(","):
        h, w = [int(x) for x in spec.lower().split("x")]
        _bench_one(
            windows,
            size_hw=(h, w),
            batch_size=args.batch_size,
            window=args.window,
            d_model=args.d_model,
            layers=args.layers,
            warmup=args.warmup,
            steps=args.steps,
            device=device,
        )


if __name__ == "__main__":
    main()
