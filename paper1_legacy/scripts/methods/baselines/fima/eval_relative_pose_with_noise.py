#!/usr/bin/env python3
"""
Quick checker for relative-pose noise scale on synthetic_liver/l1.

Default data path follows scene loading in `single_sweep_container.py`:
  /home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/l1/relative_poses.npy
"""

from __future__ import annotations

import argparse
import os
import numpy as np


def rotation_angle_deg(r: np.ndarray) -> np.ndarray:
    """r: (..., 3, 3) -> angle in degrees."""
    tr = np.trace(r, axis1=-2, axis2=-1)
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def rel_between_neighbors(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step translation(mm) and rotation(deg) magnitudes."""
    p_prev = poses[:-1]
    p_curr = poses[1:]
    p_prev_inv = np.linalg.inv(p_prev)
    rel = p_prev_inv @ p_curr
    t_step = np.linalg.norm(rel[:, :3, 3], axis=-1)
    r_step = rotation_angle_deg(rel[:, :3, :3])
    return t_step, r_step


def stats_line(x: np.ndarray, name: str, unit: str) -> str:
    return (
        f"{name}: mean={x.mean():.6f} {unit}, "
        f"median={np.median(x):.6f} {unit}, "
        f"p95={np.percentile(x, 95):.6f} {unit}, "
        f"max={x.max():.6f} {unit}"
    )


def recommend_noise_levels(t_step: np.ndarray, r_step: np.ndarray) -> dict:
    # Use p95 as robust "typical upper bound" scale.
    t95 = float(np.percentile(t_step, 95))
    r95 = float(np.percentile(r_step, 95))
    return {
        "conservative": {
            "trans_std_mm": max(1e-6, 0.5 * t95),
            "rot_std_deg": max(1e-6, 0.5 * r95),
        },
        "balanced": {
            "trans_std_mm": max(1e-6, 1.0 * t95),
            "rot_std_deg": max(1e-6, 1.0 * r95),
        },
        "challenging": {
            "trans_std_mm": max(1e-6, 2.0 * t95),
            "rot_std_deg": max(1e-6, 2.0 * r95),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses",
        type=str,
        default="/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/l1/relative_poses.npy",
        help="Path to relative_poses.npy",
    )
    parser.add_argument("--current-trans-std-mm", type=float, default=0.3)
    parser.add_argument("--current-rot-std-deg", type=float, default=0.1)
    parser.add_argument(
        "--target-percent-of-p95",
        type=float,
        default=10.0,
        help="Target noise std as percentage of neighbor-step p95 (e.g., 10 for 10%%).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.poses):
        raise FileNotFoundError(f"Missing file: {args.poses}")

    poses = np.load(args.poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected pose shape: {poses.shape}, expected (N,4,4)")

    t_step, r_step = rel_between_neighbors(poses)
    rec = recommend_noise_levels(t_step, r_step)
    t95 = float(np.percentile(t_step, 95))
    r95 = float(np.percentile(r_step, 95))
    ratio = max(0.0, float(args.target_percent_of_p95)) / 100.0
    target_trans = ratio * t95
    target_rot = ratio * r95

    print(f"[PoseNoiseEval] poses={args.poses}")
    print(f"[PoseNoiseEval] shape={poses.shape}")
    print(stats_line(t_step, "Neighbor translation step", "mm"))
    print(stats_line(r_step, "Neighbor rotation step", "deg"))
    print("")
    print("[PoseNoiseEval] Recommended noise std (for Mode-B coarse pose):")
    for k in ("conservative", "balanced", "challenging"):
        print(
            f"  - {k:12s}: trans_std_mm={rec[k]['trans_std_mm']:.6f}, "
            f"rot_std_deg={rec[k]['rot_std_deg']:.6f}"
        )
    print("")
    print(
        f"[PoseNoiseEval] Target by percentage: {args.target_percent_of_p95:.2f}% of p95\n"
        f"  - trans_std_mm={target_trans:.6f}\n"
        f"  - rot_std_deg={target_rot:.6f}"
    )
    print("[PoseNoiseEval] YAML snippet:")
    print("noise_pose:")
    print(f"  rot_std_deg: {target_rot:.6f}")
    print(f"  trans_std_mm: {target_trans:.6f}")
    print("")
    print(
        "[PoseNoiseEval] Current config (debug_online_pose_wash.yaml): "
        f"trans_std_mm={args.current_trans_std_mm:.6f}, rot_std_deg={args.current_rot_std_deg:.6f}"
    )


if __name__ == "__main__":
    main()
