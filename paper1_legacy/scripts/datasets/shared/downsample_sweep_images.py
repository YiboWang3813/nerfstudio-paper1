#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import numpy as np


def parse_hw(hw: str) -> Tuple[int, int]:
    parts = hw.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid --target-hw '{hw}', expected format like 256x128")
    h, w = int(parts[0]), int(parts[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid target size ({h}, {w}), both must be > 0")
    return h, w


def downsample_integer_factor(images: np.ndarray, target_h: int, target_w: int, mode: str) -> np.ndarray:
    _, src_h, src_w = images.shape
    if src_h % target_h != 0 or src_w % target_w != 0:
        raise ValueError(
            f"Target size ({target_h}, {target_w}) must evenly divide source ({src_h}, {src_w})"
        )
    fh = src_h // target_h
    fw = src_w // target_w
    if mode == "nearest":
        return images[:, ::fh, ::fw]
    if mode != "area":
        raise ValueError(f"Unsupported mode '{mode}', expected 'area' or 'nearest'")

    # Area-like downsampling via block average (for integer factors).
    images_f = images.astype(np.float32, copy=False)
    pooled = images_f.reshape(images.shape[0], target_h, fh, target_w, fw).mean(axis=(2, 4))
    return np.clip(np.rint(pooled), 0.0, 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample sweep images.npy and save alongside source files."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/boywang25/ultrasound-scene-reconstruction/data",
        help="Dataset root directory.",
    )
    parser.add_argument("--src-dataset", type=str, default="synthetic_liver")
    parser.add_argument("--sweep-name", type=str, default="l1")
    parser.add_argument(
        "--output-name",
        type=str,
        default="images_x2.npy",
        help="Output filename written under the source sweep directory.",
    )
    parser.add_argument(
        "--target-hw",
        type=str,
        default="256x128",
        help="Output size in HxW format, e.g. 256x128",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="area",
        choices=["area", "nearest"],
        help="Downsampling mode (integer factor only).",
    )
    args = parser.parse_args()

    target_h, target_w = parse_hw(args.target_hw)

    src_dir = os.path.join(args.data_root, args.src_dataset, args.sweep_name)

    src_images_path = os.path.join(src_dir, "images.npy")
    dst_images_path = os.path.join(src_dir, args.output_name)

    if not os.path.isfile(src_images_path):
        raise FileNotFoundError(f"Missing source file: {src_images_path}")

    images = np.load(src_images_path)  # (N, H, W)
    if images.ndim != 3:
        raise ValueError(f"Expected images shape (N,H,W), got {images.shape}")
    resized = downsample_integer_factor(images, target_h=target_h, target_w=target_w, mode=args.mode)

    np.save(dst_images_path, resized)

    print(f"Saved downsampled images to: {dst_images_path}")
    print(f"New image shape: {resized.shape}")


if __name__ == "__main__":
    main()
