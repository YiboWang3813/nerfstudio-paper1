#!/usr/bin/env python3
"""Analyze adjacent GT pose motion scale for sweep-level noise calibration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def rotation_angle_deg(r: np.ndarray) -> np.ndarray:
    tr = np.trace(r, axis1=-2, axis2=-1)
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def neighbor_motion(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected poses shape (N,4,4), got {poses.shape}")
    rel = np.linalg.inv(poses[:-1]) @ poses[1:]
    trans = np.linalg.norm(rel[:, :3, 3], axis=-1)
    rot = rotation_angle_deg(rel[:, :3, :3])
    return trans, rot


def summarize(x: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p05": float(np.percentile(x, 5)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_max": float(np.max(x)),
    }


def format_float(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.2f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}"


def analyze_file(path: Path, sweep: str, split: str, ref_trans_mm: float, ref_rot_deg: float) -> dict[str, object]:
    poses = np.load(path)
    trans, rot = neighbor_motion(poses)
    out: dict[str, object] = {
        "sweep": sweep,
        "split": split,
        "path": str(path),
        "num_poses": int(poses.shape[0]),
        "num_steps": int(max(0, poses.shape[0] - 1)),
    }
    out.update(summarize(trans, "trans_mm"))
    out.update(summarize(rot, "rot_deg"))
    trans_p95 = max(float(out["trans_mm_p95"]), 1e-12)
    rot_p95 = max(float(out["rot_deg_p95"]), 1e-12)
    out["ref_trans_over_p95"] = float(ref_trans_mm / trans_p95)
    out["ref_rot_over_p95"] = float(ref_rot_deg / rot_p95)
    for pct in (5.0, 10.0, 25.0, 50.0):
        ratio = pct / 100.0
        out[f"noise_{int(pct)}pct_trans_mm"] = trans_p95 * ratio
        out[f"noise_{int(pct)}pct_rot_deg"] = rot_p95 * ratio
    return out


def iter_pose_files(data_root: Path, dataset: str, sweeps: Iterable[str]) -> Iterable[tuple[str, str, Path]]:
    for sweep in sweeps:
        sweep_dir = data_root / dataset / sweep
        for split, name in (("train", "relative_poses.npy"), ("test", "test_relative_poses.npy")):
            path = sweep_dir / name
            if path.exists():
                yield sweep, split, path
            elif split == "test":
                fallback = sweep_dir / "test_poses.npy"
                if fallback.exists():
                    yield sweep, "test_absolute", fallback


def print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "sweep",
        "split",
        "N",
        "trans mean",
        "trans p95",
        "rot mean",
        "rot p95",
        "0.5mm / p95",
        "3deg / p95",
        "5% trans",
        "5% rot",
        "10% trans",
        "10% rot",
    ]
    print("\t".join(headers))
    for row in rows:
        values = [
            str(row["sweep"]),
            str(row["split"]),
            str(row["num_poses"]),
            format_float(float(row["trans_mm_mean"])),
            format_float(float(row["trans_mm_p95"])),
            format_float(float(row["rot_deg_mean"])),
            format_float(float(row["rot_deg_p95"])),
            format_float(float(row["ref_trans_over_p95"])),
            format_float(float(row["ref_rot_over_p95"])),
            format_float(float(row["noise_5pct_trans_mm"])),
            format_float(float(row["noise_5pct_rot_deg"])),
            format_float(float(row["noise_10pct_trans_mm"])),
            format_float(float(row["noise_10pct_rot_deg"])),
        ]
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/boywang25/ultrasound-scene-reconstruction/data"))
    parser.add_argument("--dataset", type=str, default="synthetic_liver")
    parser.add_argument("--sweeps", nargs="+", default=["l1", "l2", "l3", "r1", "r2", "r3"])
    parser.add_argument("--ref-trans-mm", type=float, default=0.5)
    parser.add_argument("--ref-rot-deg", type=float, default=3.0)
    parser.add_argument("--out-tsv", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        analyze_file(path, sweep=sweep, split=split, ref_trans_mm=args.ref_trans_mm, ref_rot_deg=args.ref_rot_deg)
        for sweep, split, path in iter_pose_files(args.data_root, args.dataset, args.sweeps)
    ]
    if not rows:
        raise SystemExit("no pose files found")

    print_table(rows)
    if args.out_tsv:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[written] {args.out_tsv}")


if __name__ == "__main__":
    main()
