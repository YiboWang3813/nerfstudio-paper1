#!/usr/bin/env python3
"""
从 sweep 中相邻帧对 (image_i, pose_i), (image_{i+1}, pose_{i+1}) 在半时刻插值，生成
test_images.npy / test_poses.npy，与 images.npy / poses.npy 同目录。

约定（与 configs_hydra/scene/liver.yaml 一致）：
  默认 sweep 目录 = baseroot/dataset/sweep_name
  -> .../data/synthetic_liver/l1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 与 liver.yaml 一致的默认路径
_DEFAULT_LIVER_L1 = os.path.join(
    "/home/boywang25/ultrasound-scene-reconstruction/data",
    "synthetic_liver",
    "l1",
)


def _quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> 单位四元数 (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """单位四元数 (w,x,y,z) -> 旋转矩阵。"""
    w, x, y, z = q / np.linalg.norm(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    th = theta_0 * t
    s0 = np.sin(theta_0 - th) / sin_0
    s1 = np.sin(th) / sin_0
    return s0 * q0 + s1 * q1


def interpolate_pose_se3(T0: np.ndarray, T1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """对 4x4 齐次变换做 SE(3) 插值：平移线性，旋转四元数 Slerp（纯 NumPy）。"""
    R0 = T0[:3, :3].astype(np.float64)
    R1 = T1[:3, :3].astype(np.float64)
    p0 = T0[:3, 3].astype(np.float64)
    p1 = T1[:3, 3].astype(np.float64)
    q0 = _quaternion_from_matrix(R0)
    q1 = _quaternion_from_matrix(R1)
    q = _quaternion_slerp(q0, q1, t)
    Ri = _quaternion_to_matrix(q)
    pi = (1.0 - t) * p0 + t * p1
    out = np.eye(4, dtype=T0.dtype)
    out[:3, :3] = Ri.astype(T0.dtype, copy=False)
    out[:3, 3] = pi.astype(T0.dtype, copy=False)
    return out


def interpolate_image_pair(img0: np.ndarray, img1: np.ndarray, t: float = 0.5) -> np.ndarray:
    """逐像素线性插值，输出 dtype 与输入一致。"""
    a = img0.astype(np.float64)
    b = img1.astype(np.float64)
    out = (1.0 - t) * a + t * b
    if np.issubdtype(img0.dtype, np.integer):
        return np.clip(np.rint(out), np.iinfo(img0.dtype).min, np.iinfo(img0.dtype).max).astype(
            img0.dtype
        )
    return out.astype(img0.dtype, copy=False)


def select_pair_start_indices(
    n_frames: int,
    trim_each: int,
    num_samples: int,
) -> np.ndarray:
    """
    掐头去尾：去掉各自 trim_each 帧后，在中间区间 [trim_each, N-2-trim_each] 内
    均匀选取 num_samples 个相邻对的左端索引 i（对 (i, i+1) 插值）。
    """
    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames, got N={n_frames}")
    if trim_each < 0:
        raise ValueError(f"trim_each must be >= 0, got {trim_each}")
    i_min = trim_each
    i_max = n_frames - 2 - trim_each
    if i_max < i_min:
        raise ValueError(
            f"After trimming {trim_each} from each end, no valid pairs: "
            f"N={n_frames}, need N >= {2 * trim_each + 2}"
        )
    n_avail = i_max - i_min + 1
    if n_avail < num_samples:
        raise ValueError(
            f"Not enough pair slots after trim: available={n_avail}, "
            f"requested num_samples={num_samples}. "
            f"Lower --num-samples or --trim-each (N={n_frames}, trim_each={trim_each})."
        )
    # 包含端点的均匀网格，避免全挤在一边
    pair_starts = np.linspace(i_min, i_max, num_samples, dtype=np.float64)
    pair_starts = np.rint(pair_starts).astype(np.int64)
    # 去重后若变短（极少发生在 n_avail >> num_samples），报错
    uniques = np.unique(pair_starts)
    if uniques.size < num_samples:
        raise ValueError(
            f"Rounded linspace collapsed to {uniques.size} unique pair indices "
            f"(need {num_samples}); widen the middle interval or reduce --num-samples."
        )
    return pair_starts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate test_images.npy and test_poses.npy by midpoint interpolation "
            "between consecutive frames (after symmetric head/tail trim)."
        )
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=_DEFAULT_LIVER_L1,
        help=(
            "含 images.npy / poses.npy 的 sweep 目录；默认对齐 liver.yaml："
            f"{_DEFAULT_LIVER_L1}"
        ),
    )
    parser.add_argument(
        "--images-name",
        type=str,
        default="images.npy",
        help="训练图像文件名。",
    )
    parser.add_argument(
        "--poses-name",
        type=str,
        default="poses.npy",
        help="训练姿态文件名（与 SingleSweepContainer pose_type 一致）。",
    )
    parser.add_argument(
        "--out-images",
        type=str,
        default="test_images.npy",
        help="输出插值图像堆叠文件名。",
    )
    parser.add_argument(
        "--out-poses",
        type=str,
        default="test_poses.npy",
        help="输出插值姿态堆叠文件名。",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=40,
        help="中间插值样本条数（默认 40）。",
    )
    parser.add_argument(
        "--trim-each",
        type=int,
        default=1,
        help="在序列首尾各丢弃的帧数；有效相邻对仅在中间段选取。默认 1。",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.5,
        help="相邻帧之间的插值参数，0.5 即 image_{i+t}（默认 0.5）。",
    )
    args = parser.parse_args()

    if not (0.0 < args.t < 1.0):
        print("--t must be in (0, 1)", file=sys.stderr)
        sys.exit(1)

    sweep_dir = os.path.abspath(os.path.expanduser(args.sweep_dir))
    images_path = os.path.join(sweep_dir, args.images_name)
    poses_path = os.path.join(sweep_dir, args.poses_name)
    if not os.path.isfile(images_path):
        raise FileNotFoundError(images_path)
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(poses_path)

    images = np.load(images_path)
    poses = np.load(poses_path)
    if images.ndim != 3:
        raise ValueError(f"images must be (N,H,W), got shape {images.shape}")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N,4,4), got shape {poses.shape}")
    n = images.shape[0]
    if poses.shape[0] != n:
        raise ValueError(f"images N={n} != poses N={poses.shape[0]}")

    pair_starts = select_pair_start_indices(n, args.trim_each, args.num_samples)

    out_images = []
    out_poses = []
    for i in pair_starts:
        img_mid = interpolate_image_pair(images[i], images[i + 1], args.t)
        pose_mid = interpolate_pose_se3(poses[i], poses[i + 1], args.t)
        out_images.append(img_mid)
        out_poses.append(pose_mid)

    stack_img = np.stack(out_images, axis=0)
    stack_pose = np.stack(out_poses, axis=0)

    out_i = os.path.join(sweep_dir, args.out_images)
    out_p = os.path.join(sweep_dir, args.out_poses)
    np.save(out_i, stack_img)
    np.save(out_p, stack_pose)

    print(f"[gen_syn_data_for_test] sweep_dir: {sweep_dir}")
    print(f"  loaded: {args.images_name} {images.shape}, {args.poses_name} {poses.shape}")
    print(f"  trim_each={args.trim_each}, t={args.t}, num_samples={args.num_samples}")
    print(f"  pair start indices (first/last): {pair_starts[0]} .. {pair_starts[-1]}")
    print(f"  wrote: {out_i}  {stack_img.shape}")
    print(f"  wrote: {out_p}  {stack_pose.shape}")


if __name__ == "__main__":
    main()
