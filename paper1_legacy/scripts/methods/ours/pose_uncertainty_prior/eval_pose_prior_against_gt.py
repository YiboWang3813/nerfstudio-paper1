#!/usr/bin/env python3
"""Evaluate exported pose-prior trajectories against GT relative poses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def rotation_angle_deg(r: np.ndarray) -> np.ndarray:
    tr = np.trace(r, axis1=-2, axis2=-1)
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def pose_errors(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = np.linalg.inv(gt) @ pred
    trans = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=-1)
    rot = rotation_angle_deg(rel[:, :3, :3])
    return trans, rot


def adjacent_motion(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = np.linalg.inv(poses[:-1]) @ poses[1:]
    trans = np.linalg.norm(rel[:, :3, 3], axis=-1)
    rot = rotation_angle_deg(rel[:, :3, :3])
    return trans, rot


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def format_float(value: float) -> str:
    if abs(value) >= 1.0:
        return f"{value:.6f}"
    return f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Predicted prior poses .npy, shape (N,4,4)")
    parser.add_argument("--gt", type=Path, required=True, help="GT relative poses .npy, shape (N,4,4)")
    parser.add_argument("--label", type=str, default="", help="Experiment label written to TSV")
    parser.add_argument("--out-tsv", type=Path, default=None, help="Append one summary row to this TSV")
    parser.add_argument(
        "--align-first",
        dest="align_first",
        action="store_true",
        default=True,
        help="Left-align pred[0] to gt[0] before measuring absolute trajectory error.",
    )
    parser.add_argument("--no-align-first", dest="align_first", action="store_false")
    args = parser.parse_args()

    pred = np.load(args.pred).astype(np.float64, copy=False)
    gt = np.load(args.gt).astype(np.float64, copy=False)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (4, 4):
        raise ValueError(f"pose shape mismatch: pred={pred.shape}, gt={gt.shape}")

    if args.align_first:
        align = gt[0] @ np.linalg.inv(pred[0])
        pred = align[None, ...] @ pred

    frame_slice = slice(1, None) if pred.shape[0] > 1 else slice(None)
    trans, rot = pose_errors(pred[frame_slice], gt[frame_slice])
    gt_step_trans, gt_step_rot = adjacent_motion(gt)
    pred_step_trans, pred_step_rot = adjacent_motion(pred)

    row: dict[str, object] = {
        "label": args.label,
        "pred": str(args.pred),
        "gt": str(args.gt),
        "num_frames": int(pred.shape[0]),
        "align_first": bool(args.align_first),
    }
    row.update(summarize(trans, "abs_trans_mm"))
    row.update(summarize(rot, "abs_rot_deg"))
    row.update(summarize(np.abs(pred_step_trans - gt_step_trans), "step_trans_absdiff_mm"))
    row.update(summarize(np.abs(pred_step_rot - gt_step_rot), "step_rot_absdiff_deg"))
    row.update(summarize(gt_step_trans, "gt_step_trans_mm"))
    row.update(summarize(gt_step_rot, "gt_step_rot_deg"))
    row["abs_trans_mean_over_gt_step_p95"] = float(row["abs_trans_mm_mean"]) / max(float(row["gt_step_trans_mm_p95"]), 1e-12)
    row["abs_rot_mean_over_gt_step_p95"] = float(row["abs_rot_deg_mean"]) / max(float(row["gt_step_rot_deg_p95"]), 1e-12)
    row["step_trans_absdiff_mean_over_gt_step_p95"] = float(row["step_trans_absdiff_mm_mean"]) / max(
        float(row["gt_step_trans_mm_p95"]), 1e-12
    )
    row["step_rot_absdiff_mean_over_gt_step_p95"] = float(row["step_rot_absdiff_deg_mean"]) / max(
        float(row["gt_step_rot_deg_p95"]), 1e-12
    )

    print("pose prior summary")
    for key in [
        "label",
        "num_frames",
        "abs_trans_mm_mean",
        "abs_trans_mm_p95",
        "abs_rot_deg_mean",
        "abs_rot_deg_p95",
        "step_trans_absdiff_mm_mean",
        "step_rot_absdiff_deg_mean",
        "gt_step_trans_mm_p95",
        "gt_step_rot_deg_p95",
        "abs_trans_mean_over_gt_step_p95",
        "abs_rot_mean_over_gt_step_p95",
        "step_trans_absdiff_mean_over_gt_step_p95",
        "step_rot_absdiff_mean_over_gt_step_p95",
    ]:
        value = row[key]
        if isinstance(value, float):
            value = format_float(value)
        print(f"{key}\t{value}")

    if args.out_tsv is not None:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.out_tsv.exists()
        with args.out_tsv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"[written] {args.out_tsv}")


if __name__ == "__main__":
    main()
