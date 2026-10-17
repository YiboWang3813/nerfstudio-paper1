"""Create corrupted pose priors and oracle frame weights for uncertainty-aware GS."""
from __future__ import annotations

import argparse
import math
import os

import numpy as np


def _euler_xyz_to_matrix(v: np.ndarray) -> np.ndarray:
    rx, ry, rz, tx, ty, tz = [float(x) for x in v]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="oracle_corrupt")
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--trans-mm", type=float, default=0.5)
    ap.add_argument("--rot-deg", type=float, default=0.1)
    ap.add_argument("--clean-weight", type=float, default=1.0)
    ap.add_argument("--corrupt-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260526)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    gt = np.load(args.gt).astype(np.float32, copy=False)
    rng = np.random.RandomState(int(args.seed))
    n = int(gt.shape[0])
    k = max(1, int(round(float(args.ratio) * n)))
    mask = np.zeros(n, dtype=np.bool_)
    # Avoid the first frame so relative origin remains clean.
    choices = np.arange(1, n, dtype=np.int64)
    corrupt_ids = rng.choice(choices, size=min(k, len(choices)), replace=False)
    mask[corrupt_ids] = True
    prior = gt.copy()
    for idx in corrupt_ids.tolist():
        axis_t = rng.randn(3).astype(np.float32)
        axis_t /= max(float(np.linalg.norm(axis_t)), 1e-8)
        axis_r = rng.randn(3).astype(np.float32)
        axis_r /= max(float(np.linalg.norm(axis_r)), 1e-8)
        v = np.zeros(6, dtype=np.float32)
        v[:3] = axis_r * math.radians(float(args.rot_deg))
        v[3:] = axis_t * float(args.trans_mm)
        prior[idx] = prior[idx] @ _euler_xyz_to_matrix(v)
    weights = np.full(n, float(args.clean_weight), dtype=np.float32)
    weights[mask] = float(args.corrupt_weight)
    pose_path = os.path.join(args.out_dir, f"{args.label}_poses.npy")
    weight_path = os.path.join(args.out_dir, f"{args.label}_weights.npy")
    mask_path = os.path.join(args.out_dir, f"{args.label}_mask.npy")
    np.save(pose_path, prior)
    np.save(weight_path, weights)
    np.save(mask_path, mask.astype(np.uint8))
    print(f"poses={pose_path}")
    print(f"weights={weight_path}")
    print(f"mask={mask_path}")
    print(f"num_frames={n} corrupt={int(mask.sum())} ratio={float(mask.mean()):.4f}")


if __name__ == "__main__":
    main()
