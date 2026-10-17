#!/usr/bin/env python3
"""Export local TUS-REC HDF5 scans into OmniUltra sweep directories.

Each input .h5 file contains:
  frames: uint8 [N,H,W]
  tforms: tracker-tool -> optical-camera, [N,4,4]

The output pose convention is project probe/image-mm -> world:
  poses = tforms @ spatial_calibration_from_image_mm_to_tool

Do not multiply scaling_from_pixel_to_mm into poses. DatasetMeta supplies the
pixel-to-mm image-plane spacing, so including the scale here would apply it
twice during rendering.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_RAW_ROOT = Path("/home/boywang25/ultrasound-scene-reconstruction/data/TUS_REC_2024")
DEFAULT_OUT_ROOT = Path("/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024")


@dataclass(frozen=True)
class Calibration:
    scaling_from_pixel_to_mm: np.ndarray
    spatial_image_mm_to_tool: np.ndarray


def _read_matrix(rows: list[list[str]], start: int) -> np.ndarray:
    mat_rows = rows[start + 1 : start + 5]
    if len(mat_rows) != 4:
        raise ValueError(f"Expected 4 matrix rows after CSV row {start}")
    return np.asarray([[float(x) for x in row] for row in mat_rows], dtype=np.float32)


def read_calibration(path: Path) -> Calibration:
    with path.open(newline="") as f:
        rows = [row for row in csv.reader(f) if row]

    labels = {row[0]: i for i, row in enumerate(rows)}
    try:
        scaling_i = labels["scaling_from_pixel_to_mm"]
        spatial_i = labels["spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system"]
    except KeyError as exc:
        raise KeyError(f"Missing calibration block in {path}: {exc}") from exc

    return Calibration(
        scaling_from_pixel_to_mm=_read_matrix(rows, scaling_i),
        spatial_image_mm_to_tool=_read_matrix(rows, spatial_i),
    )


def iter_h5_files(raw_root: Path, subjects: set[str] | None, scan_glob: str) -> Iterable[Path]:
    for subject_dir in sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.isdigit()):
        if subjects is not None and subject_dir.name not in subjects:
            continue
        yield from sorted(subject_dir.glob(scan_glob))


def compute_relative_poses(poses: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(poses[0]).astype(np.float32)
    relative = np.matmul(first_inv[None, :, :], poses).astype(np.float32)
    relative[0] = np.eye(4, dtype=np.float32)
    return relative


def write_export(
    out_dir: Path,
    frames: np.ndarray,
    poses: np.ndarray,
    metadata: dict,
) -> None:
    relative = compute_relative_poses(poses)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "images.npy", frames)
    np.save(out_dir / "poses.npy", poses)
    np.save(out_dir / "relative_poses.npy", relative)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def export_scan(
    path: Path,
    raw_root: Path,
    out_root: Path,
    calib: Calibration,
    dry_run: bool,
) -> dict:
    subject = path.parent.name
    scan_name = path.stem
    sweep_name = f"{subject}_{scan_name}"
    out_dir = out_root / sweep_name

    with h5py.File(path, "r") as f:
        frames_ds = f["frames"]
        tforms_ds = f["tforms"]
        frames_shape = tuple(int(x) for x in frames_ds.shape)
        frames_dtype = str(frames_ds.dtype)
        tforms_shape = tuple(int(x) for x in tforms_ds.shape)

        if len(frames_shape) != 3:
            raise ValueError(f"{path}: expected frames [N,H,W], got {frames_shape}")
        if tforms_shape[1:] != (4, 4):
            raise ValueError(f"{path}: expected tforms [N,4,4], got {tforms_shape}")
        if frames_shape[0] != tforms_shape[0]:
            raise ValueError(f"{path}: frame count mismatch frames={frames_shape[0]} tforms={tforms_shape[0]}")

        if dry_run:
            frames = None
            poses = None
        else:
            frames = frames_ds[:]
            tforms = tforms_ds[:].astype(np.float32)
            poses = np.matmul(tforms, calib.spatial_image_mm_to_tool[None, :, :]).astype(np.float32)

    metadata = {
        "source_h5": str(path),
        "subject": subject,
        "scan_name": scan_name,
        "sweep_name": sweep_name,
        "frames_shape": list(frames_shape),
        "frames_dtype": frames_dtype,
        "pose_convention": "image-mm/probe -> camera/world",
        "pose_formula": "tforms @ spatial_calibration_from_image_mm_to_tracking_tool",
        "pixel_scale_mm": {
            "lateral_x": float(calib.scaling_from_pixel_to_mm[0, 0]),
            "axial_y": float(calib.scaling_from_pixel_to_mm[1, 1]),
        },
        "relative_pose_formula": "inv(poses[0]) @ poses[i]",
        "center_lateral": False,
    }

    if not dry_run:
        assert frames is not None and poses is not None
        write_export(out_dir, frames, poses, metadata)

    return {
        "sweep_name": sweep_name,
        "source": str(path.relative_to(raw_root)),
        "output": str(out_dir),
        "frames": int(frames_shape[0]),
        "height": int(frames_shape[1]),
        "width": int(frames_shape[2]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs to export, e.g. --subjects 010 011. Defaults to all numeric folders.",
    )
    parser.add_argument(
        "--scan-glob",
        type=str,
        default="*.h5",
        help="Filename glob within each subject directory, e.g. LH_Par_C_DtP.h5.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect and print planned exports without writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    subjects = set(args.subjects) if args.subjects else None

    calib = read_calibration(raw_root / "calib_matrix.csv")
    rows = []
    for path in iter_h5_files(raw_root, subjects, args.scan_glob):
        rows.append(
            export_scan(
                path,
                raw_root=raw_root,
                out_root=out_root,
                calib=calib,
                dry_run=args.dry_run,
            )
        )

    total_frames = sum(row["frames"] for row in rows)
    print(f"scans={len(rows)} total_frames={total_frames} out_root={out_root}")
    for row in rows:
        print(
            f"{row['sweep_name']}: frames={row['frames']} shape={row['height']}x{row['width']} "
            f"source={row['source']} -> {row['output']}"
        )


if __name__ == "__main__":
    main()
