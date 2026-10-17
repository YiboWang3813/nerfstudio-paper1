#!/usr/bin/env python3
"""Export TDSC-ABUS2023 volumes into OmniUltra sweep directories.

TDSC-ABUS2023 provides calibrated 3D ABUS volumes, not tracked freehand image
sequences. We treat each volume as one sweep by slicing along the volume Z axis:

    source volume[x, y, z] -> images[z, y, x]

The generated poses are parallel image planes translated by the documented
between-slice spacing. This keeps project coordinates consistent with the
linear image-plane convention: lateral=X, axial=Y, elevational=Z.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nrrd
import numpy as np
import pandas as pd


DEFAULT_RAW_ROOT = Path("/tmp/tdsc_abus_2023_raw")
DEFAULT_OUT_ROOT = Path("/tmp/tdsc_abus_2023_standard_fullres")

# README values from tdsc-abus2023-pytorch.
DELTA_LATERAL_MM = 0.200
DELTA_AXIAL_MM = 0.073
DELTA_ELEVATIONAL_MM = 0.475674
TARGET_X = 865
TARGET_Y = 682


def compute_relative_poses(poses: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(poses[0]).astype(np.float32)
    relative = np.matmul(first_inv[None, :, :], poses).astype(np.float32)
    relative[0] = np.eye(4, dtype=np.float32)
    return relative


def make_parallel_poses(n_slices: int) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float32), (n_slices, 1, 1))
    poses[:, 2, 3] = np.arange(n_slices, dtype=np.float32) * float(DELTA_ELEVATIONAL_MM)
    return poses


def load_split_table(split_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(split_dir / "labels.csv")
    bbox = pd.read_csv(split_dir / "bbx_labels.csv")
    return labels.merge(bbox, left_on="case_id", right_on="id", how="left")


def iter_cases(table: pd.DataFrame, case_ids: set[int] | None) -> Iterable[pd.Series]:
    for _, row in table.sort_values("case_id").iterrows():
        case_id = int(row["case_id"])
        if case_ids is not None and case_id not in case_ids:
            continue
        yield row


def export_case(
    split_dir: Path,
    out_root: Path,
    split: str,
    row: pd.Series,
    skip_non_target_shape: bool,
    pad_to_target_shape: bool,
    include_mask: bool,
    dry_run: bool,
) -> dict:
    case_id = int(row["case_id"])
    data_rel = str(row["data_path"]).replace("\\", "/")
    mask_rel = str(row["mask_path"]).replace("\\", "/")
    data_path = split_dir / data_rel
    mask_path = split_dir / mask_rel

    volume, header = nrrd.read(str(data_path))
    if volume.ndim != 3:
        raise ValueError(f"{data_path}: expected 3D volume, got shape={volume.shape}")
    x_size, y_size, z_size = (int(v) for v in volume.shape)

    pad_y_before = 0
    pad_y_after = 0
    if skip_non_target_shape and (x_size != TARGET_X or y_size != TARGET_Y):
        return {
            "case_id": case_id,
            "status": "skipped_shape",
            "shape": [x_size, y_size, z_size],
        }
    if (x_size, y_size) != (TARGET_X, TARGET_Y):
        if not pad_to_target_shape:
            raise ValueError(
                f"{data_path}: shape {(x_size, y_size)} does not match target "
                f"{(TARGET_X, TARGET_Y)}; pass --pad-to-target-shape to pad smaller axial fields"
            )
        if x_size != TARGET_X or y_size > TARGET_Y:
            raise ValueError(
                f"{data_path}: cannot pad shape {(x_size, y_size)} to target {(TARGET_X, TARGET_Y)}"
            )
        pad_y_after = TARGET_Y - y_size
        volume = np.pad(volume, ((0, 0), (pad_y_before, pad_y_after), (0, 0)), mode="constant")
        x_size, y_size, z_size = (int(v) for v in volume.shape)

    images = np.ascontiguousarray(volume.transpose(2, 1, 0))
    poses = make_parallel_poses(images.shape[0])
    relative_poses = compute_relative_poses(poses)

    sweep_name = f"{split.lower()}_{case_id:03d}"
    out_dir = out_root / sweep_name

    metadata = {
        "source_dataset": "TDSC-ABUS2023",
        "source_split": split,
        "source_case_id": case_id,
        "source_data_path": str(data_path),
        "source_mask_path": str(mask_path),
        "source_volume_shape_xyz": [x_size, y_size, z_size],
        "source_volume_shape_xyz_before_padding": [int(v) for v in header.get("sizes", [])],
        "padding_xyz_vox": {
            "x": [0, 0],
            "y": [pad_y_before, pad_y_after],
            "z": [0, 0],
        },
        "export_axis_mapping": "images[z, y, x] = volume[x, y, z]",
        "image_shape_h_w": [int(images.shape[1]), int(images.shape[2])],
        "num_slices": int(images.shape[0]),
        "label": str(row["label"]),
        "bbox_xyz_vox": {
            "center": [float(row["c_x"]), float(row["c_y"]), float(row["c_z"])],
            "length": [float(row["len_x"]), float(row["len_y"]), float(row["len_z"])],
        },
        "spacing_mm": {
            "lateral_x": DELTA_LATERAL_MM,
            "axial_y": DELTA_AXIAL_MM,
            "elevational_z": DELTA_ELEVATIONAL_MM,
        },
        "nrrd_header": {
            "type": str(header.get("type")),
            "dimension": int(header.get("dimension", 0)),
            "sizes": [int(v) for v in header.get("sizes", [])],
            "space_directions": np.asarray(header.get("space directions", [])).tolist(),
            "space_origin": np.asarray(header.get("space origin", [])).tolist(),
            "encoding": str(header.get("encoding")),
        },
        "pose_convention": "parallel image plane, probe/image-mm -> world",
        "pose_formula": "pose[i] = identity with translation z=i*delta_elevational_mm",
        "center_lateral": True,
    }

    if dry_run:
        return {
            "case_id": case_id,
            "status": "dry_run",
            "sweep_name": sweep_name,
            "shape": [x_size, y_size, z_size],
            "images_shape": list(images.shape),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "images.npy", images.astype(np.uint8, copy=False))
    np.save(out_dir / "poses.npy", poses)
    np.save(out_dir / "relative_poses.npy", relative_poses)
    if include_mask:
        mask, _ = nrrd.read(str(mask_path))
        if pad_y_after:
            mask = np.pad(mask, ((0, 0), (pad_y_before, pad_y_after), (0, 0)), mode="constant")
        if mask.shape != volume.shape:
            raise ValueError(f"{mask_path}: mask shape {mask.shape} != volume shape {volume.shape}")
        masks = np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.uint8, copy=False)
        np.save(out_dir / "masks.npy", masks)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "case_id": case_id,
        "status": "exported",
        "sweep_name": sweep_name,
        "shape": [x_size, y_size, z_size],
        "images_shape": list(images.shape),
        "out_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--split", choices=("Train", "Validation", "Test"), default="Validation")
    parser.add_argument("--case-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--include-mask", action="store_true")
    parser.add_argument("--allow-variable-shape", action="store_true")
    parser.add_argument(
        "--pad-to-target-shape",
        action="store_true",
        help="Pad smaller axial fields to the canonical 865x682 image plane before exporting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    split_dir = raw_root / args.split
    case_ids = set(args.case_ids) if args.case_ids else None

    table = load_split_table(split_dir)
    rows = list(iter_cases(table, case_ids))
    if args.max_cases is not None:
        rows = rows[: args.max_cases]

    total = 0
    for row in rows:
        result = export_case(
            split_dir=split_dir,
            out_root=out_root,
            split=args.split,
            row=row,
            skip_non_target_shape=not args.allow_variable_shape,
            pad_to_target_shape=bool(args.pad_to_target_shape),
            include_mask=bool(args.include_mask),
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(result, sort_keys=True))
        total += int(result["status"] in {"exported", "dry_run"})
    print(f"processed={len(rows)} usable={total} split={args.split} out_root={out_root}")


if __name__ == "__main__":
    main()
