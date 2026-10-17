#!/usr/bin/env python3
"""与 synthetic_liver/gen_relative.py 同式，对 test_poses.npy 生成 test_relative_poses.npy。

约定：第 0 帧为 I，第 i 帧存 inv(pose[0]) @ pose[i]（4×4）。
用法:
  python scripts/datasets/synthetic_liver/generate_test_relative_poses.py --dir /path/to/synthetic_liver/l1
  # 或显式路径:
  python scripts/datasets/synthetic_liver/generate_test_relative_poses.py --in test_poses.npy --out test_relative_poses.npy
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="", help="含 test_poses.npy 的目录，写出 test_relative_poses.npy")
    p.add_argument("--in", dest="inp", type=str, default="", help="输入 npy 路径")
    p.add_argument("--out", dest="outp", type=str, default="", help="输出 npy 路径")
    args = p.parse_args()

    if args.dir:
        in_path = os.path.join(args.dir, "test_poses.npy")
        out_path = os.path.join(args.dir, "test_relative_poses.npy")
    else:
        if not args.inp or not args.outp:
            raise SystemExit("需要 --dir 或同时提供 --in / --out")
        in_path, out_path = args.inp, args.outp

    pose = np.load(in_path)
    if pose.ndim != 3 or pose.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N,4,4), got {pose.shape}")
    p0_inv = np.linalg.inv(pose[0])
    rel = np.stack([p0_inv @ pose[i] for i in range(pose.shape[0])], axis=0)
    np.save(out_path, rel.astype(np.float32))
    print(f"Wrote {out_path} shape={rel.shape} from {in_path}")


if __name__ == "__main__":
    main()
