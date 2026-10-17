from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np


METRIC_ORDER = ("psnr", "l1", "ssim", "lpips", "dists", "gmsd")


def max_vis_frames(cfg: Any) -> int:
    return max(0, int(getattr(cfg.metric, "max_vis_frames", 10) or 0))


def selected_vis_items(cfg: Any, total_frames: int) -> Set[int]:
    limit = max_vis_frames(cfg)
    if limit <= 0 or total_frames <= 0:
        return set()
    count = min(limit, int(total_frames))
    picks = np.linspace(1, int(total_frames), count, dtype=np.float64)
    return {int(x) for x in np.rint(picks).astype(np.int64).tolist()}


def should_save_vis(cfg: Any, item_index: int, total_frames: Optional[int] = None) -> bool:
    if total_frames is None:
        return int(item_index) <= max_vis_frames(cfg)
    return int(item_index) in selected_vis_items(cfg, total_frames)


def test_frame_count(scene: Any) -> int:
    indices = getattr(scene, "test_indices", None)
    if indices is not None:
        return len(indices)
    return len(getattr(scene, "test_container"))


def frame_metric_row(item_index: int, datum: Mapping[str, Any], scores: Mapping[str, float]) -> Dict[str, Any]:
    slice_meta = datum.get("slice_meta")
    frame_id = getattr(slice_meta, "frame_id", datum.get("index", ""))
    row: Dict[str, Any] = {
        "item": int(item_index),
        "index": datum.get("index", ""),
        "frame_id": frame_id,
    }
    for key in METRIC_ORDER:
        if key in scores:
            row[key] = float(scores[key])
    for key, value in scores.items():
        if key not in row:
            row[key] = float(value)
    return row


def write_per_frame_metrics(
    output_dir: str,
    iteration: Optional[int],
    rows: List[Dict[str, Any]],
    cfg: Any,
) -> None:
    if not rows or not bool(getattr(cfg.metric, "save_per_frame_metrics", True)):
        return
    eval_dir = Path(output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    iter_name = "eval" if iteration is None else f"iteration_{int(iteration):06d}"
    path = eval_dir / f"per_frame_metrics_{iter_name}.tsv"
    fields = ["item", "index", "frame_id"]
    for key in METRIC_ORDER:
        if any(key in row for row in rows):
            fields.append(key)
    extras = sorted({key for row in rows for key in row.keys()} - set(fields))
    fields.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(path, eval_dir / "per_frame_metrics_latest.tsv")


def clear_vis_dir(vis_dir: str) -> None:
    os.makedirs(vis_dir, exist_ok=True)
    for name in os.listdir(vis_dir):
        path = os.path.join(vis_dir, name)
        if name.startswith("item_") and os.path.isdir(path):
            shutil.rmtree(path)
