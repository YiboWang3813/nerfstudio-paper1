#!/usr/bin/env python3
"""Select real frames nearest to a uniform pose-arc-length grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def select_nearest_uniform_indices(centers: np.ndarray, target_count: int, start: int, end: int) -> tuple[np.ndarray, dict]:
    if target_count < 2:
        raise ValueError("--target-count must be >= 2")
    if start < 0 or end > centers.shape[0] or end <= start:
        raise ValueError(f"Invalid frame range [{start}, {end}) for N={centers.shape[0]}")

    frame_ids = np.arange(start, end, dtype=np.int64)
    sub_centers = centers[start:end]
    if sub_centers.shape[0] < target_count:
        raise ValueError(f"Range has {sub_centers.shape[0]} frames, smaller than target_count={target_count}")

    steps = np.linalg.norm(sub_centers[1:] - sub_centers[:-1], axis=1)
    arc = np.zeros(sub_centers.shape[0], dtype=np.float64)
    arc[1:] = np.cumsum(steps)
    if arc[-1] <= 0.0:
        raise ValueError("Pose centers have zero arc length")

    targets = np.linspace(0.0, arc[-1], target_count, dtype=np.float64)
    chosen_local: list[int] = []
    prev = -1
    for target in targets:
        pos = int(np.searchsorted(arc, target, side="left"))
        candidates = []
        for cand in (pos - 1, pos, pos + 1):
            if prev < cand < arc.shape[0]:
                candidates.append(cand)
        if not candidates:
            cand = prev + 1
            if cand >= arc.shape[0]:
                break
            candidates.append(cand)
        best = min(candidates, key=lambda idx: abs(float(arc[idx] - target)))
        chosen_local.append(best)
        prev = best

    # Fill any missing tail samples while keeping strict monotonic order.
    if len(chosen_local) < target_count:
        remaining = [idx for idx in range(chosen_local[-1] + 1, arc.shape[0])]
        chosen_local.extend(remaining[: target_count - len(chosen_local)])
    chosen_local = chosen_local[:target_count]
    if len(chosen_local) != target_count:
        raise ValueError(f"Could only select {len(chosen_local)} frames, requested {target_count}")

    chosen = frame_ids[np.asarray(chosen_local, dtype=np.int64)]
    chosen_arc = arc[np.asarray(chosen_local, dtype=np.int64)]
    chosen_steps = np.diff(chosen_arc)
    original_steps = steps
    stats = {
        "source_frame_start": int(start),
        "source_frame_end": int(end),
        "source_frame_count": int(end - start),
        "target_count": int(target_count),
        "arc_length_mm": float(arc[-1]),
        "selected_first": int(chosen[0]),
        "selected_last": int(chosen[-1]),
        "selected_index_step_min": int(np.diff(chosen).min()) if chosen.size > 1 else 0,
        "selected_index_step_max": int(np.diff(chosen).max()) if chosen.size > 1 else 0,
        "original_step_mean_mm": float(original_steps.mean()),
        "original_step_median_mm": float(np.median(original_steps)),
        "original_step_cv": float(original_steps.std() / max(original_steps.mean(), 1e-12)),
        "selected_step_mean_mm": float(chosen_steps.mean()),
        "selected_step_median_mm": float(np.median(chosen_steps)),
        "selected_step_cv": float(chosen_steps.std() / max(chosen_steps.mean(), 1e-12)),
        "selected_step_min_mm": float(chosen_steps.min()),
        "selected_step_max_mm": float(chosen_steps.max()),
    }
    return chosen, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1, help="Exclusive; negative means full sweep")
    args = parser.parse_args()

    poses = np.load(args.poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape (N,4,4), got {poses.shape}")
    end = poses.shape[0] if args.frame_end < 0 else min(args.frame_end, poses.shape[0])
    centers = poses[:, :3, 3].astype(np.float64, copy=False)
    indices, stats = select_nearest_uniform_indices(
        centers=centers,
        target_count=int(args.target_count),
        start=int(args.frame_start),
        end=int(end),
    )

    payload = {
        "description": "Real frame indices selected by nearest-neighbor uniform pose arc length. Original images/poses are not modified.",
        "poses": str(args.poses),
        "indices": [int(x) for x in indices.tolist()],
        "stats": stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(json.dumps(payload["stats"], indent=2))


if __name__ == "__main__":
    main()
