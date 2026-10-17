#!/usr/bin/env python3
"""Build p100k processed datasets from raw OmniUltra dataset directories.

The output keeps only dataloader-standard files and resizes each image slice to
the agreed working resolution while preserving the physical field of view in
DatasetMeta/configs:

- synthetic_liver: 448 x 224
- spine_phantom: 896 x 112
- tus_rec_2024: 288 x 384
- tdsc_abus_2023: 280 x 356
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TARGETS = {
    "synthetic_liver": (448, 224),
    "spine_phantom": (896, 112),
    "tus_rec_2024": (288, 384),
    "tdsc_abus_2023": (280, 356),
}

SYNTHETIC_SWEEPS = ("l1", "l2", "l3", "r1", "r2", "r3")


def compute_relative_poses(poses: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(poses[0]).astype(np.float32)
    relative = np.matmul(first_inv[None, :, :], poses).astype(np.float32)
    relative[0] = np.eye(4, dtype=np.float32)
    return relative


def normalize_images(images: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uint8_255":
        out = np.nan_to_num(images, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(out, 0.0, 255.0).astype(np.float32, copy=False)
    if mode == "spine_phantom_shifted_float":
        out = np.nan_to_num(images.astype(np.float32, copy=False), nan=-0.125, posinf=1.0, neginf=-0.125)
        return (np.clip(out + 0.125, 0.0, 1.0) * 255.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported mode: {mode}")


def resize_stack_uint8(images: np.ndarray, target_hw: tuple[int, int], mode: str, batch_size: int) -> np.ndarray:
    target_h, target_w = target_hw
    images_f = normalize_images(images, mode=mode)
    out_chunks = []
    for start in range(0, images_f.shape[0], batch_size):
        batch = images_f[start : start + batch_size]
        tensor = torch.from_numpy(batch).unsqueeze(1).float()
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        resized = resized.squeeze(1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        out_chunks.append(resized)
    return np.concatenate(out_chunks, axis=0)


def resize_mask_uint8(mask: np.ndarray, target_hw: tuple[int, int], batch_size: int) -> np.ndarray:
    target_h, target_w = target_hw
    out_chunks = []
    for start in range(0, mask.shape[0], batch_size):
        batch = mask[start : start + batch_size].astype(np.float32, copy=False)
        tensor = torch.from_numpy(batch).unsqueeze(1)
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
        out_chunks.append(resized.squeeze(1).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def save_metadata(out_dir: Path, metadata: dict) -> None:
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def copy_pose_files(src_dir: Path, out_dir: Path, pose_type: str = "absolute") -> None:
    poses_path = src_dir / "poses.npy"
    if poses_path.exists():
        poses = np.load(poses_path).astype(np.float32, copy=False)
        np.save(out_dir / "poses.npy", poses)
        rel_path = src_dir / "relative_poses.npy"
        relative = np.load(rel_path).astype(np.float32, copy=False) if rel_path.exists() else compute_relative_poses(poses)
        np.save(out_dir / "relative_poses.npy", relative)
    if pose_type == "synthetic_test":
        test_poses_path = src_dir / "test_poses.npy"
        if test_poses_path.exists():
            test_poses = np.load(test_poses_path).astype(np.float32, copy=False)
            np.save(out_dir / "test_poses.npy", test_poses)
            test_rel_path = src_dir / "test_relative_poses.npy"
            if test_rel_path.exists():
                np.save(out_dir / "test_relative_poses.npy", np.load(test_rel_path).astype(np.float32, copy=False))


def build_sweep(
    src_dir: Path,
    out_dir: Path,
    target_hw: tuple[int, int],
    image_mode: str,
    batch_size: int,
    include_test: bool = False,
    include_mask: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = np.load(src_dir / "images.npy")
    images_out = resize_stack_uint8(images, target_hw=target_hw, mode=image_mode, batch_size=batch_size)
    np.save(out_dir / "images.npy", images_out)

    copy_pose_files(src_dir, out_dir, pose_type="synthetic_test" if include_test else "absolute")

    test_shape = None
    if include_test and (src_dir / "test_images.npy").exists():
        test_images = np.load(src_dir / "test_images.npy")
        test_out = resize_stack_uint8(test_images, target_hw=target_hw, mode=image_mode, batch_size=batch_size)
        np.save(out_dir / "test_images.npy", test_out)
        test_shape = list(test_out.shape)

    if include_mask and (src_dir / "masks.npy").exists():
        mask = np.load(src_dir / "masks.npy")
        mask_out = resize_mask_uint8(mask, target_hw=target_hw, batch_size=batch_size)
        np.save(out_dir / "masks.npy", mask_out)

    metadata = {
        "processed_version": "p100k",
        "source_dir": str(src_dir),
        "source_image_shape": list(images.shape),
        "target_image_shape": list(images_out.shape),
        "target_hw": list(target_hw),
        "image_dtype": "uint8",
        "image_value_mode": "uint8_255",
        "resize": {
            "method": "bilinear",
            "preserve_physical_fov": True,
        },
    }
    if test_shape is not None:
        metadata["target_test_image_shape"] = test_shape

    existing_meta = src_dir / "metadata.json"
    if existing_meta.exists():
        with existing_meta.open() as f:
            metadata["source_metadata"] = json.load(f)

    save_metadata(out_dir, metadata)
    return metadata


def build_synthetic(src_root: Path, out_root: Path, batch_size: int) -> None:
    for sweep in SYNTHETIC_SWEEPS:
        meta = build_sweep(
            src_root / sweep,
            out_root / sweep,
            target_hw=TARGETS["synthetic_liver"],
            image_mode="uint8_255",
            batch_size=batch_size,
            include_test=True,
        )
        print(json.dumps({"dataset": "synthetic_liver", "sweep": sweep, "shape": meta["target_image_shape"]}), flush=True)


def build_spine(src_root: Path, out_root: Path, batch_size: int) -> None:
    for src_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and (p / "images.npy").exists()):
        meta = build_sweep(
            src_dir,
            out_root / src_dir.name,
            target_hw=TARGETS["spine_phantom"],
            image_mode="spine_phantom_shifted_float",
            batch_size=batch_size,
        )
        print(json.dumps({"dataset": "spine_phantom", "sweep": src_dir.name, "shape": meta["target_image_shape"]}), flush=True)


def build_tus(src_root: Path, out_root: Path, batch_size: int) -> None:
    for src_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and (p / "images.npy").exists()):
        meta = build_sweep(
            src_dir,
            out_root / src_dir.name,
            target_hw=TARGETS["tus_rec_2024"],
            image_mode="uint8_255",
            batch_size=batch_size,
        )
        print(json.dumps({"dataset": "tus_rec_2024", "sweep": src_dir.name, "shape": meta["target_image_shape"]}), flush=True)


def build_tdsc(src_root: Path, out_root: Path, batch_size: int, include_masks: bool) -> None:
    for src_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and (p / "images.npy").exists()):
        meta = build_sweep(
            src_dir,
            out_root / src_dir.name,
            target_hw=TARGETS["tdsc_abus_2023"],
            image_mode="uint8_255",
            batch_size=batch_size,
            include_mask=include_masks,
        )
        print(json.dumps({"dataset": "tdsc_abus_2023", "sweep": src_dir.name, "shape": meta["target_image_shape"]}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(TARGETS), default=list(TARGETS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tdsc-include-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if "synthetic_liver" in args.datasets:
        build_synthetic(raw_root / "synthetic_liver", out_root / "synthetic_liver", args.batch_size)
    if "spine_phantom" in args.datasets:
        build_spine(raw_root / "spine_phantom", out_root / "spine_phantom", args.batch_size)
    if "tus_rec_2024" in args.datasets:
        build_tus(raw_root / "tus_rec_2024", out_root / "tus_rec_2024", args.batch_size)
    if "tdsc_abus_2023" in args.datasets:
        build_tdsc(raw_root / "tdsc_abus_2023", out_root / "tdsc_abus_2023", args.batch_size, args.tdsc_include_masks)


if __name__ == "__main__":
    main()
