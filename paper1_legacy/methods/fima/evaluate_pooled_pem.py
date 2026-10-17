"""Evaluate a pooled FiMA PEM checkpoint on synthetic_liver holdout edges."""
from __future__ import annotations

import argparse
from argparse import Namespace

import torch

from methods.fima.pose_estimator import FiMAPoseEstimator
from methods.fima.train_pooled_pem import PooledSyntheticLiverWindows, _evaluate, _make_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-windows", type=int, default=120)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pooled FiMA evaluation.")
    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = Namespace(**ckpt["args"])
    cfg = _make_cfg(train_args)
    sweeps = [x.strip() for x in str(train_args.sweeps).split(",") if x.strip()]
    data = PooledSyntheticLiverWindows(
        train_args.root,
        sweeps,
        image_name=str(train_args.image_name),
        window=int(train_args.window_size),
        trim_each=10,
        holdout_num_samples=40,
        device=device,
    )
    model = FiMAPoseEstimator(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    target_mean = ckpt["target_mean"].to(device)
    target_std = ckpt["target_std"].to(device)
    val = _evaluate(model, data, cfg, target_mean, target_std, int(args.eval_windows))
    print(
        f"checkpoint={args.checkpoint}\n"
        f"eval_windows={int(args.eval_windows)}\n"
        f"val_loss={val['loss']:.8f}\n"
        f"val_edge_trans_mm={val['edge_trans_mm']:.8f}\n"
        f"val_edge_rot_deg={val['edge_rot_deg']:.8f}"
    )


if __name__ == "__main__":
    main()
