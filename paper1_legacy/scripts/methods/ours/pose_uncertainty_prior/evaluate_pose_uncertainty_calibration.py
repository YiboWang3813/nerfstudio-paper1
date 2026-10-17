"""Evaluate whether pose uncertainty ranks actual pose-prior error."""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


def _rotation_error_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    r = np.swapaxes(pred[:, :3, :3], 1, 2) @ gt[:, :3, :3]
    tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(c)).astype(np.float64)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1.0e-12:
        return float("nan")
    return float(np.dot(x, y) / den)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    recall = tp / max(pos, 1)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def _load_uncertainty(path: str, score_col: int | None) -> np.ndarray:
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if score_col is not None:
            return arr[:, int(score_col)]
        scale = np.percentile(arr, 95, axis=0)
        scale = np.maximum(scale, 1.0e-12)
        return np.mean(np.clip(arr / scale, 0.0, 1.0), axis=1)
    raise ValueError(f"Unsupported uncertainty shape: {arr.shape}")


def _calibration_bins(score: np.ndarray, err: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qs = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.quantile(score, qs)
    means_s, means_e, counts = [], [], []
    for i in range(int(bins)):
        lo, hi = edges[i], edges[i + 1]
        if i + 1 == int(bins):
            m = (score >= lo) & (score <= hi)
        else:
            m = (score >= lo) & (score < hi)
        counts.append(int(m.sum()))
        means_s.append(float(score[m].mean()) if m.any() else float("nan"))
        means_e.append(float(err[m].mean()) if m.any() else float("nan"))
    return np.asarray(means_s), np.asarray(means_e), np.asarray(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-poses", required=True)
    ap.add_argument("--gt-poses", required=True)
    ap.add_argument("--uncertainty", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-col", type=int, default=None)
    ap.add_argument("--bad-quantile", type=float, default=0.8)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    pred = np.load(args.pred_poses).astype(np.float64, copy=False)
    gt = np.load(args.gt_poses).astype(np.float64, copy=False)
    score = _load_uncertainty(args.uncertainty, args.score_col)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (4, 4):
        raise ValueError(f"Expected pred/gt pose arrays of shape (N,4,4), got {pred.shape}, {gt.shape}")
    if int(score.shape[0]) != int(pred.shape[0]):
        raise ValueError(f"uncertainty length {score.shape[0]} != poses length {pred.shape[0]}")

    trans = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1).astype(np.float64)
    rot = _rotation_error_deg(pred, gt)
    err = trans + rot
    bad_thr = float(np.quantile(err, float(args.bad_quantile)))
    bad = err >= bad_thr
    pearson = _pearson(score, err)
    spearman = _pearson(_rankdata(score), _rankdata(err))
    auroc = _auroc(score, bad)
    auprc = _auprc(score, bad)
    bin_s, bin_e, bin_n = _calibration_bins(score, err, int(args.bins))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Pose Uncertainty Calibration\n\n")
        f.write(f"- pred_poses: `{args.pred_poses}`\n")
        f.write(f"- gt_poses: `{args.gt_poses}`\n")
        f.write(f"- uncertainty: `{args.uncertainty}`\n")
        f.write(f"- frames: {int(pred.shape[0])}\n")
        f.write(f"- mean_trans_mm: {float(trans.mean()):.8f}\n")
        f.write(f"- mean_rot_deg: {float(rot.mean()):.8f}\n")
        f.write(f"- mean_joint_error: {float(err.mean()):.8f}\n")
        f.write(f"- bad_threshold_q{float(args.bad_quantile):.2f}: {bad_thr:.8f}\n")
        f.write(f"- pearson_score_error: {pearson:.8f}\n")
        f.write(f"- spearman_score_error: {spearman:.8f}\n")
        f.write(f"- auroc_bad_pose: {auroc:.8f}\n")
        f.write(f"- auprc_bad_pose: {auprc:.8f}\n\n")
        f.write("| bin | count | mean_score | mean_error |\n")
        f.write("|---:|---:|---:|---:|\n")
        for i in range(int(args.bins)):
            f.write(f"| {i} | {int(bin_n[i])} | {float(bin_s[i]):.8f} | {float(bin_e[i]):.8f} |\n")

    print(f"out={args.out}")
    print(f"pearson={pearson:.8f} spearman={spearman:.8f} auroc={auroc:.8f} auprc={auprc:.8f}")
    print(f"mean_trans_mm={float(trans.mean()):.8f} mean_rot_deg={float(rot.mean()):.8f}")


if __name__ == "__main__":
    main()
