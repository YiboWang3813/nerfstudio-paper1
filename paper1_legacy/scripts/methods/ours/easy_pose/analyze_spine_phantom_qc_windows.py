#!/usr/bin/env python3
"""Scan contiguous spine_phantom windows for canonical-sweep suitability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _angle_from_rotation(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _window_metrics(poses: np.ndarray, start: int, end: int) -> Dict[str, float]:
    sub = poses[start:end].astype(np.float64, copy=False)
    centers = sub[:, :3, 3]
    rots = sub[:, :3, :3]
    steps = centers[1:] - centers[:-1]
    step_norm = np.linalg.norm(steps, axis=1)
    local_steps = np.einsum("nij,nj->ni", np.swapaxes(rots[:-1], -1, -2), steps)
    abs_local = np.abs(local_steps)
    elev = abs_local[:, 2]
    non_elev = np.linalg.norm(abs_local[:, :2], axis=1)
    rel_rots = np.einsum("nij,njk->nik", np.swapaxes(rots[:-1], -1, -2), rots[1:])
    rel_rot_deg = _angle_from_rotation(rel_rots)

    idx = np.arange(end - start, dtype=np.float64)
    idx -= float(np.mean(idx))
    center_mean = np.mean(centers, axis=0)
    direction = (idx[:, None] * (centers - center_mean)).sum(axis=0) / max(float(np.sum(idx * idx)), 1e-12)
    fitted = center_mean[None, :] + idx[:, None] * direction[None, :]
    residual = np.linalg.norm(centers - fitted, axis=1)

    centered = centers - center_mean[None, :]
    eig = np.linalg.eigvalsh(centered.T @ centered / max(1, centered.shape[0] - 1))
    eig = np.sort(np.maximum(eig, 0.0))[::-1]
    line_ratio = float(eig[0] / max(float(np.sum(eig)), 1e-12))

    step_mean = float(np.mean(step_norm))
    step_cv = float(np.std(step_norm) / max(step_mean, 1e-12))
    non_elev_over_elev = float(np.mean(non_elev / np.maximum(elev, 1e-12)))
    res_mean = float(np.mean(residual))
    return {
        "start": float(start),
        "end": float(end),
        "n_frames": float(end - start),
        "step_mean": step_mean,
        "step_median": float(np.median(step_norm)),
        "step_cv": step_cv,
        "elev_fraction": float(np.mean(elev / np.maximum(step_norm, 1e-12))),
        "non_elev_over_elev": non_elev_over_elev,
        "bad_step_frac": float(np.mean(non_elev > elev)),
        "rot_mean_deg": float(np.mean(rel_rot_deg)),
        "rot_max_deg": float(np.max(rel_rot_deg)),
        "linear_residual_mean": res_mean,
        "linear_residual_max": float(np.max(residual)),
        "pca_line_ratio": line_ratio,
    }


def _score(metrics: Dict[str, float]) -> float:
    n = metrics["n_frames"]
    return (
        metrics["step_cv"]
        + 0.6 * metrics["non_elev_over_elev"]
        + 0.015 * metrics["linear_residual_mean"]
        + 0.02 * metrics["rot_mean_deg"]
        - 0.001 * n
    )


def _scan(poses: np.ndarray, lengths: Iterable[int], top_k: int) -> Dict[int, List[Dict[str, float]]]:
    out: Dict[int, List[Dict[str, float]]] = {}
    for length in lengths:
        candidates = []
        for start in range(0, poses.shape[0] - int(length) + 1):
            metrics = _window_metrics(poses, start, start + int(length))
            metrics["score"] = _score(metrics)
            candidates.append(metrics)
        out[int(length)] = sorted(candidates, key=lambda item: item["score"])[:top_k]
    return out


def _fmt_row(row: Dict[str, float]) -> str:
    return (
        f"{int(row['start']):3d}-{int(row['end']) - 1:3d}"
        f" n={int(row['n_frames']):3d}"
        f" score={row['score']:.3f}"
        f" step_cv={row['step_cv']:.3f}"
        f" med={row['step_median']:.3f}"
        f" elev={row['elev_fraction']:.3f}"
        f" ne/e={row['non_elev_over_elev']:.3f}"
        f" rot={row['rot_mean_deg']:.3f}/{row['rot_max_deg']:.3f}"
        f" res={row['linear_residual_mean']:.3f}/{row['linear_residual_max']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom/left1/poses.npy"),
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[50, 60, 70, 80, 90, 100, 110, 120])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-md", type=Path, default=Path("docs/datasets/spine_phantom/qc_window_analysis.md"))
    args = parser.parse_args()

    poses = np.load(args.poses)
    scanned = _scan(poses, args.lengths, top_k=int(args.top_k))

    lines = [
        "# Spine Phantom QC Window Analysis",
        "",
        f"Pose file: `{args.poses}`",
        "",
        "The score prefers lower adjacent-step CV, lower non-elevational motion, lower line-fit residual, lower rotation, and longer windows.",
        "",
    ]
    for length in args.lengths:
        lines.append(f"## Length {int(length)}")
        lines.append("")
        lines.append("```text")
        for row in scanned[int(length)]:
            lines.append(_fmt_row(row))
        lines.append("```")
        lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
