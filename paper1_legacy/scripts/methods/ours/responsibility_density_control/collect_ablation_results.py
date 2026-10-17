#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def collect_summary_files(root_dir: str) -> List[str]:
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "online_run_summary.json" in filenames:
            hits.append(os.path.join(dirpath, "online_run_summary.json"))
    return sorted(hits)


def load_rows(paths: List[str], exp_prefix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        exp_tag = str(d.get("exp_tag", ""))
        if exp_prefix and not exp_tag.startswith(exp_prefix):
            continue
        row: Dict[str, Any] = {
            "exp_tag": exp_tag,
            "output_dir": str(d.get("output_dir", "")),
            "dataset_name": str(d.get("dataset_name", "")),
            "num_new": _safe_int(d.get("num_new", 0)),
            "total_opt_steps_cfg": _safe_int(d.get("total_opt_steps_cfg", 0)),
            "window_size_frames": _safe_int(d.get("window_size_frames", 0)),
            "num_frames_processed": _safe_int(d.get("num_frames_processed", 0)),
            "final_gs_step": _safe_int(d.get("final_gs_step", 0)),
            "final_num_gaussians": _safe_int(d.get("final_num_gaussians", 0)),
            "last_eval_psnr": _safe_float(d.get("last_eval_psnr", 0.0)),
            "last_eval_ssim": _safe_float(d.get("last_eval_ssim", 0.0)),
            "last_eval_l1": _safe_float(d.get("last_eval_l1", 0.0)),
            "frame_wall_ms_mean": _safe_float(d.get("frame_wall_ms_mean", 0.0)),
            "frame_wall_ms_min": _safe_float(d.get("frame_wall_ms_min", 0.0)),
            "frame_wall_ms_max": _safe_float(d.get("frame_wall_ms_max", 0.0)),
            "frame_fps_mean": _safe_float(d.get("frame_fps_mean", 0.0)),
            "o1_t_total_ms_mean": _safe_float(d.get("o1_t_total_ms_mean", 0.0)),
            "o1_active_ratio_mean": _safe_float(d.get("o1_active_ratio_mean", 0.0)),
            "num_eval_events": _safe_int(d.get("num_eval_events", 0)),
            "summary_path": p,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect online ablation run summaries into one CSV.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/boywang25/ultrasound-scene-reconstruction/omniultra/checkpoints/ours",
        help="Experiment output root directory to scan.",
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        default="ablation_n_total_n_new",
        help="Only keep runs whose exp_tag starts with this prefix. Empty means keep all.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="ablation_n_total_n_new_results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    summary_paths = collect_summary_files(args.root_dir)
    rows = load_rows(summary_paths, args.exp_prefix)
    rows.sort(
        key=lambda r: (
            r.get("num_new", 0),
            r.get("total_opt_steps_cfg", 0),
            -r.get("last_eval_psnr", 0.0),
        )
    )

    fields = [
        "exp_tag",
        "dataset_name",
        "num_new",
        "total_opt_steps_cfg",
        "window_size_frames",
        "num_frames_processed",
        "final_gs_step",
        "final_num_gaussians",
        "last_eval_psnr",
        "last_eval_ssim",
        "last_eval_l1",
        "frame_wall_ms_mean",
        "frame_wall_ms_min",
        "frame_wall_ms_max",
        "frame_fps_mean",
        "o1_t_total_ms_mean",
        "o1_active_ratio_mean",
        "num_eval_events",
        "output_dir",
        "summary_path",
    ]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[Collect] scanned summaries: {len(summary_paths)}")
    print(f"[Collect] matched rows: {len(rows)}")
    print(f"[Collect] wrote csv: {args.out_csv}")


if __name__ == "__main__":
    main()
