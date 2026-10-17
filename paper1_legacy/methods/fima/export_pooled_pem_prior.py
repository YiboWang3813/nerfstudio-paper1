"""Export a sweep-level pose prior and frame uncertainty from a pooled FiMA PEM."""
from __future__ import annotations

import argparse
import os
from argparse import Namespace
from typing import Dict

import numpy as np
import torch

from methods.fima.pose_estimator import FiMAPoseEstimator
from methods.fima.pose_geometry import se3_from_6dof, translation_rotation_errors
from methods.fima.train_pooled_pem import PooledSyntheticLiverWindows, _load_image, _make_cfg


def _predict_edge(
    model: FiMAPoseEstimator,
    images: torch.Tensor,
    edge_idx: int,
    window: int,
) -> Dict[str, torch.Tensor]:
    n = int(images.shape[0])
    right = int(edge_idx + 1)
    end = max(right, window - 1)
    end = min(end, n - 1)
    start = end - window + 1
    if start < 0:
        raise ValueError(f"Invalid window for edge={edge_idx}, n={n}, window={window}")
    local = right - start - 1
    if local < 0 or local >= window - 1:
        raise ValueError(f"Invalid local edge slot: edge={edge_idx} start={start} local={local}")
    batch = images[start : end + 1].unsqueeze(0).unsqueeze(2)
    out = model(batch)
    pred = out["pred_pose_6"][0, local].detach()
    result = {"pose6": pred}
    if "pred_logvar_2" in out:
        result["logvar2"] = out["pred_logvar_2"][0, local].detach()
    return result


def _scores_to_weights(score: np.ndarray, alpha: float, min_weight: float) -> np.ndarray:
    return min_weight + (1.0 - min_weight) * np.exp(-float(alpha) * score.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver")
    ap.add_argument("--sweep", default="r2")
    ap.add_argument("--image-name", default="images.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weight-alpha", type=float, default=2.0)
    ap.add_argument("--min-weight", type=float, default=0.1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FiMA PEM export.")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = Namespace(**ckpt["args"])
    cfg = _make_cfg(train_args)
    model = FiMAPoseEstimator(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sweep_dir = os.path.join(args.root, args.sweep)
    images = _load_image(os.path.join(sweep_dir, args.image_name)).to(device=device, dtype=torch.float32)
    gt = torch.from_numpy(np.load(os.path.join(sweep_dir, "relative_poses.npy")).astype(np.float32, copy=False)).to(
        device=device, dtype=torch.float32
    )
    window = int(cfg.pem.window_size)
    n = int(images.shape[0])
    if n != int(gt.shape[0]):
        raise ValueError(f"image/pose length mismatch: {n} vs {gt.shape[0]}")

    target_std = ckpt.get("target_std", torch.ones(6)).to(device=device, dtype=torch.float32)
    trans_std = float(target_std[:3].mean().detach().cpu().item())
    rot_std = float(target_std[3:6].mean().detach().cpu().item())

    poses = torch.empty_like(gt)
    poses[0] = gt[0]
    edge_pose6 = []
    edge_logvar = []
    with torch.no_grad():
        for edge_idx in range(n - 1):
            pred = _predict_edge(model, images, edge_idx, window)
            T_edge = se3_from_6dof(pred["pose6"])
            poses[edge_idx + 1] = poses[edge_idx] @ T_edge
            edge_pose6.append(pred["pose6"].detach().cpu().numpy())
            if "logvar2" in pred:
                edge_logvar.append(pred["logvar2"].detach().cpu().numpy())

    edge_pose6_np = np.asarray(edge_pose6, dtype=np.float32)
    if edge_logvar:
        edge_logvar_np = np.asarray(edge_logvar, dtype=np.float32)
        edge_sigma_trans = np.sqrt(np.exp(np.clip(edge_logvar_np[:, 0], -8.0, 4.0))).astype(np.float32) * trans_std
        edge_sigma_rot = np.sqrt(np.exp(np.clip(edge_logvar_np[:, 1], -8.0, 4.0))).astype(np.float32) * rot_std
    else:
        edge_logvar_np = np.zeros((n - 1, 2), dtype=np.float32)
        edge_sigma_trans = np.zeros(n - 1, dtype=np.float32)
        edge_sigma_rot = np.zeros(n - 1, dtype=np.float32)

    frame_unc = np.zeros((n, 2), dtype=np.float32)
    if n > 1:
        frame_unc[1:, 0] = np.sqrt(np.cumsum(edge_sigma_trans * edge_sigma_trans))
        frame_unc[1:, 1] = np.sqrt(np.cumsum(edge_sigma_rot * edge_sigma_rot))
    hi = np.percentile(frame_unc, 95, axis=0).astype(np.float32)
    hi = np.maximum(hi, np.asarray([1.0e-6, 1.0e-8], dtype=np.float32))
    score = np.clip(0.5 * (frame_unc[:, 0] / hi[0] + frame_unc[:, 1] / hi[1]), 0.0, 1.0).astype(np.float32)
    weights = _scores_to_weights(score, args.weight_alpha, args.min_weight).astype(np.float32)

    pred_np = poses.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    dt, dr = translation_rotation_errors(poses, gt)
    frame_errors = np.stack(
        [dt.detach().cpu().numpy(), torch.rad2deg(dr).detach().cpu().numpy()],
        axis=1,
    ).astype(np.float32)

    np.save(os.path.join(args.out_dir, "poses.npy"), pred_np)
    np.save(os.path.join(args.out_dir, "gt_poses.npy"), gt_np)
    np.save(os.path.join(args.out_dir, "edge_pose6.npy"), edge_pose6_np)
    np.save(os.path.join(args.out_dir, "edge_logvar.npy"), edge_logvar_np)
    np.save(os.path.join(args.out_dir, "frame_uncertainty.npy"), frame_unc)
    np.save(os.path.join(args.out_dir, "frame_uncertainty_score.npy"), score)
    np.save(os.path.join(args.out_dir, "frame_weights.npy"), weights)
    np.save(os.path.join(args.out_dir, "frame_pose_errors.npy"), frame_errors)

    print(f"checkpoint={args.checkpoint}")
    print(f"sweep={args.sweep} frames={n} window={window}")
    print(f"pose_path={os.path.join(args.out_dir, 'poses.npy')}")
    print(f"weights_path={os.path.join(args.out_dir, 'frame_weights.npy')}")
    print(f"mean_trans_mm={float(frame_errors[:, 0].mean()):.8f}")
    print(f"mean_rot_deg={float(frame_errors[:, 1].mean()):.8f}")
    print(
        "uncertainty_score="
        f"min={float(score.min()):.6f} mean={float(score.mean()):.6f} max={float(score.max()):.6f}"
    )
    print(f"weights=min={float(weights.min()):.6f} mean={float(weights.mean()):.6f} max={float(weights.max()):.6f}")


if __name__ == "__main__":
    main()
