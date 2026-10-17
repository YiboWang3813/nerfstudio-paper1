#!/usr/bin/env python3
"""Diagnose when a canonical straight-sweep pose prior is geometrically valid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


DATASETS = [
    {
        "dataset": "synthetic_liver",
        "sweep": "l1",
        "root": "synthetic_liver/l1",
        "pose_file": "relative_poses.npy",
        "test_pose_file": "test_relative_poses.npy",
        "delta_mm": 0.2685,
        "stage_e_file_psnr": 32.1386809,
        "stage_e_best_canonical_psnr": 22.9935927,
    },
    {
        "dataset": "spine_phantom",
        "sweep": "left1",
        "root": "spine_phantom/left1",
        "pose_file": "poses.npy",
        "test_pose_file": None,
        "delta_mm": 0.79,
        "stage_e_file_psnr": 30.6422475,
        "stage_e_best_canonical_psnr": 29.328903,
    },
    {
        "dataset": "tus_rec_2024_qc",
        "sweep": "010_LH_Per_L_DtP_qc_0005_0370",
        "root": "tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370",
        "pose_file": "poses.npy",
        "test_pose_file": None,
        "delta_mm": 0.404,
        "stage_e_file_psnr": 32.3016905,
        "stage_e_best_canonical_psnr": 32.4745731,
    },
    {
        "dataset": "tdsc_abus_2023",
        "sweep": "validation_100",
        "root": "tdsc_abus_2023/validation_100",
        "pose_file": "poses.npy",
        "test_pose_file": None,
        "delta_mm": 0.475674,
        "stage_e_file_psnr": 27.4488984,
        "stage_e_best_canonical_psnr": 27.4626612,
    },
]


def _angle_from_rotation(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _safe_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {k: float("nan") for k in ("mean", "median", "std", "min", "max", "p90")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _linear_fit_residual(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(centers.shape[0], dtype=np.float64)
    idx0 = idx - float(np.mean(idx))
    c0 = np.mean(centers, axis=0)
    denom = float(np.sum(idx0 * idx0))
    direction_per_frame = (idx0[:, None] * (centers - c0)).sum(axis=0) / max(denom, 1e-12)
    fitted = c0[None, :] + idx0[:, None] * direction_per_frame[None, :]
    residual = np.linalg.norm(centers - fitted, axis=1)
    return fitted, direction_per_frame, residual


def _pca_linearity(centers: np.ndarray) -> Tuple[float, float]:
    centered = centers - np.mean(centers, axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, centers.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.sort(np.maximum(eig, 0.0))[::-1]
    total = float(np.sum(eig))
    if total <= 1e-12:
        return 1.0, 0.0
    line_ratio = float(eig[0] / total)
    orth_rms = float(math.sqrt(max(eig[1] + eig[2], 0.0)))
    return line_ratio, orth_rms


def _canonical_aligned_center_error(centers: np.ndarray, delta_mm: float) -> Dict[str, float]:
    """Fit scale-free rotation/translation from a canonical index line to centers."""
    n = centers.shape[0]
    src = np.zeros((n, 3), dtype=np.float64)
    src[:, 2] = np.arange(n, dtype=np.float64) * float(delta_mm)
    src0 = src - src.mean(axis=0, keepdims=True)
    dst0 = centers - centers.mean(axis=0, keepdims=True)
    denom = float(np.sum(src0 * src0))
    if denom <= 1e-12:
        err = np.linalg.norm(dst0, axis=1)
        return {
            "mean": float(np.mean(err)),
            "max": float(np.max(err)),
            "scale": 0.0,
        }
    h = src0.T @ dst0
    u, s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    scale = float(np.sum(s) / denom)
    pred = scale * (src0 @ rot.T) + centers.mean(axis=0, keepdims=True)
    err = np.linalg.norm(pred - centers, axis=1)
    return {
        "mean": float(np.mean(err)),
        "max": float(np.max(err)),
        "scale": scale,
    }


def analyze_poses(poses: np.ndarray, delta_mm: float) -> Dict[str, object]:
    poses = np.asarray(poses, dtype=np.float64)
    centers = poses[:, :3, 3]
    rots = poses[:, :3, :3]
    steps = centers[1:] - centers[:-1]
    step_norm = np.linalg.norm(steps, axis=1)
    local_steps = np.einsum("nij,nj->ni", np.swapaxes(rots[:-1], -1, -2), steps)
    abs_local = np.abs(local_steps)
    lateral = abs_local[:, 0]
    axial = abs_local[:, 1]
    elev = abs_local[:, 2]
    non_elev = np.sqrt(lateral * lateral + axial * axial)
    denom_norm = np.maximum(step_norm, 1e-12)
    denom_elev = np.maximum(elev, 1e-12)

    rel_rots = np.einsum("nij,njk->nik", np.swapaxes(rots[:-1], -1, -2), rots[1:])
    rel_rot_deg = _angle_from_rotation(rel_rots)
    normals = rots[:, :, 2]
    normal_mean = np.mean(normals, axis=0)
    normal_mean /= max(float(np.linalg.norm(normal_mean)), 1e-12)
    normal_dev = np.degrees(np.arccos(np.clip(normals @ normal_mean, -1.0, 1.0)))

    _, direction_per_frame, residual = _linear_fit_residual(centers)
    line_ratio, orth_rms = _pca_linearity(centers)
    aligned = _canonical_aligned_center_error(centers, delta_mm)

    return {
        "n_frames": int(poses.shape[0]),
        "span_mm": float(np.linalg.norm(centers[-1] - centers[0])) if poses.shape[0] > 1 else 0.0,
        "center_bbox_mm": [float(x) for x in (centers.max(axis=0) - centers.min(axis=0))],
        "step_norm": _safe_stats(step_norm),
        "step_cv": float(np.std(step_norm) / max(float(np.mean(step_norm)), 1e-12)),
        "meta_delta_mm": float(delta_mm),
        "step_median_over_meta_delta": float(np.median(step_norm) / max(float(delta_mm), 1e-12)),
        "local_abs_lateral": _safe_stats(lateral),
        "local_abs_axial": _safe_stats(axial),
        "local_abs_elev": _safe_stats(elev),
        "elev_fraction_mean": float(np.mean(elev / denom_norm)),
        "non_elev_over_elev_mean": float(np.mean(non_elev / denom_elev)),
        "bad_step_non_elev_gt_elev_frac": float(np.mean(non_elev > elev)),
        "rel_rot_deg": _safe_stats(rel_rot_deg),
        "normal_dev_deg": _safe_stats(normal_dev),
        "linear_fit_step_mm": float(np.linalg.norm(direction_per_frame)),
        "linear_fit_residual_mm": _safe_stats(residual),
        "pca_linearity_ratio": float(line_ratio),
        "pca_orth_rms_mm": float(orth_rms),
        "canonical_similarity_center_err_mm": aligned,
    }


def _fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def _record_markdown(record: Dict[str, object]) -> str:
    m = record["metrics"]
    train_test = record.get("test_metrics")
    delta_psnr = record["stage_e_best_canonical_psnr"] - record["stage_e_file_psnr"]
    lines = [
        f"### {record['dataset']} / {record['sweep']}",
        "",
        f"- Stage E best canonical minus file-pose PSNR: `{_fmt(delta_psnr)}` dB.",
        f"- Frames: `{m['n_frames']}`; pose file: `{record['pose_path']}`.",
        f"- Step norm median/mean/std: `{_fmt(m['step_norm']['median'])}` / `{_fmt(m['step_norm']['mean'])}` / `{_fmt(m['step_norm']['std'])}` mm; CV `{_fmt(m['step_cv'])}`; median/meta-delta `{_fmt(m['step_median_over_meta_delta'])}`.",
        f"- Local motion mean abs lateral/axial/elevational: `{_fmt(m['local_abs_lateral']['mean'])}` / `{_fmt(m['local_abs_axial']['mean'])}` / `{_fmt(m['local_abs_elev']['mean'])}` mm.",
        f"- Elevational fraction mean: `{_fmt(m['elev_fraction_mean'])}`; non-elev/elev mean: `{_fmt(m['non_elev_over_elev_mean'])}`; bad-step fraction non-elev > elev: `{_fmt(m['bad_step_non_elev_gt_elev_frac'])}`.",
        f"- Adjacent rotation mean/max: `{_fmt(m['rel_rot_deg']['mean'])}` / `{_fmt(m['rel_rot_deg']['max'])}` deg; normal deviation mean/max: `{_fmt(m['normal_dev_deg']['mean'])}` / `{_fmt(m['normal_dev_deg']['max'])}` deg.",
        f"- Linear trajectory fit residual mean/max: `{_fmt(m['linear_fit_residual_mm']['mean'])}` / `{_fmt(m['linear_fit_residual_mm']['max'])}` mm; PCA line ratio `{_fmt(m['pca_linearity_ratio'])}`; orthogonal RMS `{_fmt(m['pca_orth_rms_mm'])}` mm.",
        f"- Similarity-aligned canonical center error mean/max: `{_fmt(m['canonical_similarity_center_err_mm']['mean'])}` / `{_fmt(m['canonical_similarity_center_err_mm']['max'])}` mm; fitted scale `{_fmt(m['canonical_similarity_center_err_mm']['scale'])}`.",
    ]
    if train_test is not None:
        train_step = float(m["step_norm"]["median"])
        test_step = float(train_test["step_norm"]["median"])
        test_train_ratio = test_step / max(train_step, 1e-12)
        lines.extend(
            [
                f"- Test pose file: `{record['test_pose_path']}`.",
                f"- Test step norm median: `{_fmt(test_step)}` mm; test/train step ratio: `{_fmt(test_train_ratio)}`; test linear residual mean/max: `{_fmt(train_test['linear_fit_residual_mm']['mean'])}` / `{_fmt(train_test['linear_fit_residual_mm']['max'])}` mm.",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/home/boywang25/ultrasound-scene-reconstruction/data",
        type=Path,
    )
    parser.add_argument("--json-out", default="", type=Path)
    parser.add_argument("--md-out", default="", type=Path)
    args = parser.parse_args()

    records = []
    for spec in DATASETS:
        root = args.data_root / spec["root"]
        pose_path = root / spec["pose_file"]
        poses = np.load(pose_path)
        metrics = analyze_poses(poses, float(spec["delta_mm"]))
        record = {
            **spec,
            "pose_path": str(pose_path),
            "metrics": metrics,
        }
        test_pose_file = spec.get("test_pose_file")
        if test_pose_file:
            test_pose_path = root / test_pose_file
            if test_pose_path.exists():
                test_poses = np.load(test_pose_path)
                record["test_pose_path"] = str(test_pose_path)
                record["test_metrics"] = analyze_poses(test_poses, float(spec["delta_mm"]))
        records.append(record)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    md = [
        "# Canonical Sweep Pose Geometry Diagnosis",
        "",
        "This report diagnoses why the tracking-free canonical sweep prior works on some one-sweep ultrasound datasets and fails on others.",
        "",
        "The key validity assumption is a near-regular ordered sweep: adjacent motion should be dominated by the local elevational axis, rotations should be small, and frame centers should be well approximated by a uniform straight line.",
        "",
        "## Per-Dataset Metrics",
        "",
    ]
    md.extend(_record_markdown(record) + "\n" for record in records)

    md.extend(
        [
            "## Summary Interpretation",
            "",
            "- `tdsc_abus_2023` is the ideal canonical case because its exported poses are generated parallel image-plane poses.",
            "- `tus_rec_2024_qc` is a realistic positive case: the QC crop retained segments whose local motion is mostly elevational, so replacing noisy tracked poses by a straight canonical sweep can act as pose denoising.",
            "- `spine_phantom` is almost a straight line in PCA terms, but its frame-to-frame spacing is highly non-uniform. A fixed-index canonical sweep erases this variable spacing, so tracked file poses remain better.",
            "- `synthetic_liver` is not mainly a bad train-trajectory case. The train step is regular, but the external `test_relative_poses.npy` sequence is sampled at about 5x the train step. Current canonical test loading rebuilds a 40-frame canonical sequence with one-step spacing, which compresses test poses and makes the Stage E canonical result a protocol-mismatch failure.",
            "",
            "## Method Implication",
            "",
            "The pose-free contribution should be framed as a tracking-free canonical sweep prior for regular one-sweep scans, with geometry diagnostics or low-dimensional sweep calibration as a validity/fallback mechanism. For external interpolated-file test protocols, canonical pose generation must preserve original/global frame spacing instead of rebuilding test poses from compressed test indices. It should not be claimed as universal pose-free reconstruction.",
            "",
        ]
    )
    md_text = "\n".join(md)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md_text, encoding="utf-8")
    print(md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
