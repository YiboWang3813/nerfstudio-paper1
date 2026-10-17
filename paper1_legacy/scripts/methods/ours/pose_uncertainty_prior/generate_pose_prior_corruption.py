"""Generate noisy pose priors plus oracle uncertainty/weights for US-GS ablations."""
from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import numpy as np


def _euler_xyz_to_matrix(rot_xyz: np.ndarray, trans_xyz: np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(x) for x in rot_xyz]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = trans_xyz.astype(np.float32, copy=False)
    return T


def _unit(rng: np.random.RandomState) -> np.ndarray:
    v = rng.randn(3).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1.0e-8)


def _delta_from_magnitudes(rng: np.random.RandomState, trans_mm: float, rot_deg: float) -> np.ndarray:
    return _euler_xyz_to_matrix(
        _unit(rng) * math.radians(float(rot_deg)),
        _unit(rng) * float(trans_mm),
    )


def _make_offsets(args: argparse.Namespace, n: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    mode = str(args.mode).lower()
    rot = np.zeros((n, 3), dtype=np.float32)
    trans = np.zeros((n, 3), dtype=np.float32)
    mask = np.zeros(n, dtype=np.bool_)
    if mode == "random_outlier":
        k = max(1, int(round(float(args.corrupt_ratio) * n)))
        choices = np.arange(1, n, dtype=np.int64)
        ids = rng.choice(choices, size=min(k, len(choices)), replace=False)
        mask[ids] = True
        for i in ids.tolist():
            rot[i] = _unit(rng) * math.radians(float(args.corrupt_rot_deg))
            trans[i] = _unit(rng) * float(args.corrupt_trans_mm)
    elif mode == "translation_bias":
        mask[1:] = True
        trans[1:] = _unit(rng) * float(args.corrupt_trans_mm)
    elif mode == "rotation_bias":
        mask[1:] = True
        rot[1:] = _unit(rng) * math.radians(float(args.corrupt_rot_deg))
    elif mode == "smooth_drift":
        mask[1:] = True
        t_step = _unit(rng) * float(args.corrupt_trans_mm) / max(n - 1, 1)
        r_step = _unit(rng) * math.radians(float(args.corrupt_rot_deg)) / max(n - 1, 1)
        ramp = np.arange(n, dtype=np.float32).reshape(-1, 1)
        trans = ramp * t_step.reshape(1, 3)
        rot = ramp * r_step.reshape(1, 3)
    elif mode == "sinusoidal_drift":
        mask[1:] = True
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        cycles = float(args.cycles)
        amp_t = _unit(rng) * float(args.corrupt_trans_mm)
        amp_r = _unit(rng) * math.radians(float(args.corrupt_rot_deg))
        x = np.arange(n, dtype=np.float32)
        s = np.sin(2.0 * math.pi * cycles * x / max(n - 1, 1) + phase).reshape(-1, 1)
        trans = s * amp_t.reshape(1, 3)
        rot = s * amp_r.reshape(1, 3)
    elif mode == "random_walk":
        mask[1:] = True
        trans_step = float(args.corrupt_trans_mm) / max(math.sqrt(n), 1.0)
        rot_step = math.radians(float(args.corrupt_rot_deg)) / max(math.sqrt(n), 1.0)
        for i in range(1, n):
            trans[i] = trans[i - 1] + _unit(rng) * trans_step
            rot[i] = rot[i - 1] + _unit(rng) * rot_step
    else:
        raise ValueError(f"Unknown corruption mode: {args.mode}")
    return rot, trans, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-path", "--gt", dest="pose_path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="pose_corrupt")
    ap.add_argument(
        "--mode",
        default="random_outlier",
        choices=["random_outlier", "translation_bias", "rotation_bias", "smooth_drift", "sinusoidal_drift", "random_walk"],
    )
    ap.add_argument("--corrupt-ratio", "--ratio", dest="corrupt_ratio", type=float, default=0.2)
    ap.add_argument("--corrupt-trans-mm", "--trans-mm", dest="corrupt_trans_mm", type=float, default=0.5)
    ap.add_argument("--corrupt-rot-deg", "--rot-deg", dest="corrupt_rot_deg", type=float, default=0.1)
    ap.add_argument("--clean-weight", type=float, default=1.0)
    ap.add_argument("--corrupt-weight", type=float, default=0.1)
    ap.add_argument("--cycles", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260526)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    poses = np.load(args.pose_path).astype(np.float32, copy=False)
    rng = np.random.RandomState(int(args.seed))
    rot, trans, mask = _make_offsets(args, int(poses.shape[0]), rng)
    prior = poses.copy()
    for i in range(1, int(poses.shape[0])):
        if np.any(rot[i] != 0.0) or np.any(trans[i] != 0.0):
            prior[i] = prior[i] @ _euler_xyz_to_matrix(rot[i], trans[i])

    weights = np.full(int(poses.shape[0]), float(args.clean_weight), dtype=np.float32)
    weights[mask] = float(args.corrupt_weight)
    uncertainty = np.zeros((int(poses.shape[0]), 2), dtype=np.float32)
    uncertainty[:, 0] = np.linalg.norm(rot, axis=1).astype(np.float32)
    uncertainty[:, 1] = np.linalg.norm(trans, axis=1).astype(np.float32)

    paths = {
        "poses": os.path.join(args.out_dir, f"{args.label}_poses.npy"),
        "weights": os.path.join(args.out_dir, f"{args.label}_weights.npy"),
        "uncertainty": os.path.join(args.out_dir, f"{args.label}_uncertainty.npy"),
        "mask": os.path.join(args.out_dir, f"{args.label}_mask.npy"),
    }
    np.save(paths["poses"], prior)
    np.save(paths["weights"], weights)
    np.save(paths["uncertainty"], uncertainty)
    np.save(paths["mask"], mask.astype(np.uint8))
    for k, v in paths.items():
        print(f"{k}={v}")
    print(f"num_frames={poses.shape[0]} corrupt={int(mask.sum())} ratio={float(mask.mean()):.4f} mode={args.mode}")


if __name__ == "__main__":
    main()
