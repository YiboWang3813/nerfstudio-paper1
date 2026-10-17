#!/usr/bin/env python3
"""Pose-QC and crop TUS-REC-2024 sweeps into production-ready segments.

The QC rule keeps continuous frame ranges whose adjacent probe motion is
primarily along the local elevational axis. This removes in-plane sliding
segments that violate the current 3D freehand reconstruction assumption.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SRC_ROOT = Path("/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024")
DEFAULT_OUT_ROOT = Path("/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc")


@dataclass(frozen=True)
class QcParams:
    min_elevational_step_mm: float = 0.10
    elevational_over_lateral_ratio: float = 1.0
    elevational_over_axial_ratio: float = 0.5
    max_step_norm_mm: float = 1.5
    bridge_bad_steps: int = 5
    trim_boundary_frames: int = 5
    min_segment_frames: int = 80


@dataclass(frozen=True)
class Segment:
    source_sweep: str
    output_sweep: str
    start_frame: int
    end_frame_exclusive: int
    num_frames: int
    median_abs_lateral_step_mm: float
    median_abs_axial_step_mm: float
    median_abs_elevational_step_mm: float
    qc_good_step_fraction: float


def compute_relative_poses(poses: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(poses[0]).astype(np.float32)
    relative = np.matmul(first_inv[None, :, :], poses).astype(np.float32)
    relative[0] = np.eye(4, dtype=np.float32)
    return relative


def iter_sweeps(root: Path) -> Iterable[Path]:
    yield from sorted(p for p in root.iterdir() if p.is_dir() and (p / "images.npy").exists() and (p / "poses.npy").exists())


def local_step_components(poses: np.ndarray) -> np.ndarray:
    translations = poses[:, :3, 3].astype(np.float64, copy=False)
    rotations = poses[:-1, :3, :3].astype(np.float64, copy=False)
    delta_world = translations[1:] - translations[:-1]
    # R columns are local lateral/axial/elevational axes in world coordinates.
    return np.einsum("nji,nj->ni", rotations, delta_world)


def build_good_step_mask(local_steps: np.ndarray, params: QcParams) -> np.ndarray:
    abs_steps = np.abs(local_steps)
    lat = abs_steps[:, 0]
    ax = abs_steps[:, 1]
    ele = abs_steps[:, 2]
    norm = np.linalg.norm(local_steps, axis=1)
    return (
        (ele >= params.min_elevational_step_mm)
        & (ele >= params.elevational_over_lateral_ratio * lat)
        & (ele >= params.elevational_over_axial_ratio * ax)
        & (norm <= params.max_step_norm_mm)
    )


def bridge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return mask.copy()
    out = mask.copy()
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        start = i
        while i < len(out) and not out[i]:
            i += 1
        end = i
        if start > 0 and end < len(out) and (end - start) <= max_gap:
            out[start:end] = True
    return out


def mask_to_frame_ranges(mask: np.ndarray, params: QcParams) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        step_start = i
        while i < len(mask) and mask[i]:
            i += 1
        step_end = i - 1

        frame_start = step_start + params.trim_boundary_frames
        frame_end_exclusive = step_end + 2 - params.trim_boundary_frames
        if frame_end_exclusive - frame_start >= params.min_segment_frames:
            ranges.append((int(frame_start), int(frame_end_exclusive)))
    return ranges


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")


def analyze_sweep(sweep_dir: Path, params: QcParams) -> tuple[list[Segment], dict]:
    poses = np.load(sweep_dir / "poses.npy").astype(np.float64, copy=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{sweep_dir}: expected poses.npy shape (N,4,4), got {poses.shape}")
    if poses.shape[0] < 2:
        return [], {"source_sweep": sweep_dir.name, "num_frames": int(poses.shape[0]), "segments": 0}

    local_steps = local_step_components(poses)
    raw_good = build_good_step_mask(local_steps, params)
    good = bridge_small_gaps(raw_good, params.bridge_bad_steps)
    ranges = mask_to_frame_ranges(good, params)
    abs_steps = np.abs(local_steps)

    segments: list[Segment] = []
    for idx, (start, end_excl) in enumerate(ranges, start=1):
        step_slice = abs_steps[start : max(start, end_excl - 1)]
        if len(step_slice) == 0:
            continue
        output_name = f"{sweep_dir.name}_qc_{start:04d}_{end_excl:04d}"
        if len(ranges) > 1:
            output_name = f"{output_name}_s{idx:02d}"
        segments.append(
            Segment(
                source_sweep=sweep_dir.name,
                output_sweep=output_name,
                start_frame=int(start),
                end_frame_exclusive=int(end_excl),
                num_frames=int(end_excl - start),
                median_abs_lateral_step_mm=float(np.median(step_slice[:, 0])),
                median_abs_axial_step_mm=float(np.median(step_slice[:, 1])),
                median_abs_elevational_step_mm=float(np.median(step_slice[:, 2])),
                qc_good_step_fraction=float(np.mean(good[start : max(start, end_excl - 1)])),
            )
        )

    summary = {
        "source_sweep": sweep_dir.name,
        "num_frames": int(poses.shape[0]),
        "raw_good_step_fraction": float(np.mean(raw_good)),
        "bridged_good_step_fraction": float(np.mean(good)),
        "median_abs_lateral_step_mm": float(np.median(abs_steps[:, 0])),
        "median_abs_axial_step_mm": float(np.median(abs_steps[:, 1])),
        "median_abs_elevational_step_mm": float(np.median(abs_steps[:, 2])),
        "segments": len(segments),
        "kept_frames": int(sum(seg.num_frames for seg in segments)),
    }
    return segments, summary


def export_segment(src_dir: Path, out_root: Path, segment: Segment, params: QcParams, overwrite: bool) -> None:
    out_dir = out_root / segment.output_sweep
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = np.load(src_dir / "images.npy", mmap_mode="r")
    poses = np.load(src_dir / "poses.npy").astype(np.float32, copy=False)
    start = segment.start_frame
    end = segment.end_frame_exclusive
    if images.shape[0] != poses.shape[0]:
        raise ValueError(f"{src_dir}: image/pose frame mismatch: {images.shape[0]} vs {poses.shape[0]}")

    images_out = np.asarray(images[start:end])
    poses_out = poses[start:end].astype(np.float32, copy=False)
    np.save(out_dir / "images.npy", images_out)
    np.save(out_dir / "poses.npy", poses_out)
    np.save(out_dir / "relative_poses.npy", compute_relative_poses(poses_out))

    src_meta = load_json(src_dir / "metadata.json")
    metadata = {
        "processed_version": "p100k_tus_pose_qc_v1",
        "source_sweep": segment.source_sweep,
        "source_dir": str(src_dir),
        "frame_start_inclusive": start,
        "frame_end_exclusive": end,
        "num_frames": segment.num_frames,
        "image_shape": list(images_out.shape),
        "pose_shape": list(poses_out.shape),
        "qc_reason": "keep continuous segments whose adjacent probe motion is primarily elevational; remove in-plane sliding frames",
        "qc_params": asdict(params),
        "qc_segment_stats": asdict(segment),
        "source_metadata": src_meta,
    }
    write_json(out_dir / "metadata.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--export", action="store_true", help="Write cropped sweep directories.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output directory contents.")
    parser.add_argument("--min-elevational-step-mm", type=float, default=QcParams.min_elevational_step_mm)
    parser.add_argument("--elevational-over-lateral-ratio", type=float, default=QcParams.elevational_over_lateral_ratio)
    parser.add_argument("--elevational-over-axial-ratio", type=float, default=QcParams.elevational_over_axial_ratio)
    parser.add_argument("--max-step-norm-mm", type=float, default=QcParams.max_step_norm_mm)
    parser.add_argument("--bridge-bad-steps", type=int, default=QcParams.bridge_bad_steps)
    parser.add_argument("--trim-boundary-frames", type=int, default=QcParams.trim_boundary_frames)
    parser.add_argument("--min-segment-frames", type=int, default=QcParams.min_segment_frames)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = QcParams(
        min_elevational_step_mm=float(args.min_elevational_step_mm),
        elevational_over_lateral_ratio=float(args.elevational_over_lateral_ratio),
        elevational_over_axial_ratio=float(args.elevational_over_axial_ratio),
        max_step_norm_mm=float(args.max_step_norm_mm),
        bridge_bad_steps=int(args.bridge_bad_steps),
        trim_boundary_frames=int(args.trim_boundary_frames),
        min_segment_frames=int(args.min_segment_frames),
    )
    src_root = args.src_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    all_segments: list[Segment] = []
    summaries: list[dict] = []
    for sweep_dir in iter_sweeps(src_root):
        segments, summary = analyze_sweep(sweep_dir, params)
        all_segments.extend(segments)
        summaries.append(summary)

    report = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "exported": bool(args.export),
        "qc_params": asdict(params),
        "num_source_sweeps": len(summaries),
        "num_output_segments": len(all_segments),
        "total_source_frames": int(sum(row["num_frames"] for row in summaries)),
        "total_kept_frames": int(sum(seg.num_frames for seg in all_segments)),
        "sweep_summaries": summaries,
        "segments": [asdict(seg) for seg in all_segments],
    }

    if args.export:
        out_root.mkdir(parents=True, exist_ok=True)
        for segment in all_segments:
            export_segment(src_root / segment.source_sweep, out_root, segment, params, overwrite=bool(args.overwrite))
        write_json(out_root / "_qc_report.json", report)
        write_tsv(out_root / "_qc_segments.tsv", [asdict(seg) for seg in all_segments])
        write_tsv(out_root / "_qc_sweeps.tsv", summaries)
    else:
        print(json.dumps(report, indent=2))

    print(
        f"source_sweeps={report['num_source_sweeps']} output_segments={report['num_output_segments']} "
        f"source_frames={report['total_source_frames']} kept_frames={report['total_kept_frames']} "
        f"export={args.export} out_root={out_root}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
