#!/usr/bin/env python3
"""Fit a train-edge translation calibration for exported FiMA pose priors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Set

import numpy as np


def _load_poses(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4), got {arr.shape}: {path}")
    return arr


def _interval_holdout_indices(n_frames: int, trim_each: int, num_samples: int) -> List[int]:
    i_min = int(trim_each)
    i_max = int(n_frames - 1 - trim_each)
    if i_max < i_min:
        raise ValueError(f"invalid interval holdout: N={n_frames}, trim_each={trim_each}")
    picks = np.rint(np.linspace(i_min, i_max, int(num_samples), dtype=np.float64)).astype(np.int64)
    uniq = np.unique(picks)
    if int(uniq.size) != int(num_samples):
        raise ValueError(f"interval holdout collapsed to {uniq.size} unique frames")
    return [int(x) for x in uniq.tolist()]


def _block_holdout_indices(n_frames: int, trim_each: int, num_samples: int, block_size: int) -> List[int]:
    i_min = int(trim_each)
    i_max = int(n_frames - 1 - trim_each)
    if i_max < i_min:
        raise ValueError(f"invalid block holdout: N={n_frames}, trim_each={trim_each}")
    block_size = min(int(block_size), int(num_samples))
    n_blocks = int(np.ceil(float(num_samples) / float(block_size)))
    max_start = i_max - block_size + 1
    starts = sorted(int(x) for x in np.unique(np.rint(np.linspace(i_min, max_start, n_blocks))).tolist())
    out: List[int] = []
    seen: Set[int] = set()
    for s in starts:
        for g in range(s, min(s + block_size, i_max + 1)):
            if g not in seen:
                out.append(int(g))
                seen.add(int(g))
            if len(out) >= int(num_samples):
                break
        if len(out) >= int(num_samples):
            break
    if len(out) != int(num_samples):
        raise ValueError(f"block holdout produced {len(out)} frames, expected {num_samples}")
    return sorted(out)


def _relative_edges(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(poses[:-1]) @ poses[1:]


def _compose_zero_rotation(anchor: np.ndarray, trans_edges: np.ndarray) -> np.ndarray:
    edges = np.repeat(np.eye(4, dtype=np.float64)[None, ...], trans_edges.shape[0], axis=0)
    edges[:, :3, 3] = trans_edges
    out = np.empty((trans_edges.shape[0] + 1, 4, 4), dtype=np.float64)
    out[0] = anchor
    for i in range(trans_edges.shape[0]):
        out[i + 1] = out[i] @ edges[i]
    return out


def _rotation_angle_deg(rot: np.ndarray) -> np.ndarray:
    tr = np.trace(rot, axis1=-2, axis2=-1)
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def _pose_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    trans = np.linalg.norm(pred[1:, :3, 3] - gt[1:, :3, 3], axis=-1)
    rel = np.linalg.inv(gt[1:]) @ pred[1:]
    rot = _rotation_angle_deg(rel[:, :3, :3])
    return {
        "abs_trans_mean_mm": float(trans.mean()),
        "abs_trans_p95_mm": float(np.percentile(trans, 95)),
        "abs_rot_mean_deg": float(rot.mean()),
        "abs_rot_p95_deg": float(np.percentile(rot, 95)),
    }


def _edge_metrics(pred_edges: np.ndarray, gt_edges: np.ndarray, edge_ids: List[int], prefix: str) -> dict[str, float]:
    if not edge_ids:
        return {
            f"{prefix}_edge_trans_mean_mm": float("nan"),
            f"{prefix}_edge_trans_p95_mm": float("nan"),
        }
    idx = np.asarray(edge_ids, dtype=np.int64)
    err = np.linalg.norm(pred_edges[idx, :3, 3] - gt_edges[idx, :3, 3], axis=-1)
    return {
        f"{prefix}_edge_trans_mean_mm": float(err.mean()),
        f"{prefix}_edge_trans_p95_mm": float(np.percentile(err, 95)),
    }


def _fit_affine(src: np.ndarray, dst: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float64)], axis=1)
    if ridge > 0.0:
        reg = float(ridge) * np.eye(4, dtype=np.float64)
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(x.T @ x + reg, x.T @ dst)
    else:
        beta, *_ = np.linalg.lstsq(x, dst, rcond=None)
    a = beta[:3, :].T
    b = beta[3, :]
    return a, b


def _fit_diag_bias(src: np.ndarray, dst: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    scales = []
    bias = []
    for j in range(3):
        x = np.stack([src[:, j], np.ones(src.shape[0], dtype=np.float64)], axis=1)
        if ridge > 0.0:
            reg = float(ridge) * np.eye(2, dtype=np.float64)
            reg[-1, -1] = 0.0
            beta = np.linalg.solve(x.T @ x + reg, x.T @ dst[:, j])
        else:
            beta, *_ = np.linalg.lstsq(x, dst[:, j], rcond=None)
        scales.append(float(beta[0]))
        bias.append(float(beta[1]))
    return np.diag(scales), np.asarray(bias, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, default=None)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--split-protocol", choices=("interval", "block"), default="interval")
    parser.add_argument("--holdout-num-samples", type=int, default=20)
    parser.add_argument("--holdout-trim-each", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--fit-mode", choices=("affine", "diag_bias"), default="diag_bias")
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    args = parser.parse_args()

    pred = _load_poses(args.pred)
    gt = _load_poses(args.gt)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    n = int(pred.shape[0])
    if args.split_protocol == "interval":
        test_idx = _interval_holdout_indices(n, args.holdout_trim_each, args.holdout_num_samples)
    else:
        test_idx = _block_holdout_indices(n, args.holdout_trim_each, args.holdout_num_samples, args.block_size)
    test_set = set(test_idx)
    train_edges = [i for i in range(n - 1) if i not in test_set and (i + 1) not in test_set]
    held_edges = [i for i in range(n - 1) if i in test_set or (i + 1) in test_set]
    if len(train_edges) < 8:
        raise ValueError(f"too few train edges for calibration: {len(train_edges)}")

    pred_edges = _relative_edges(pred)
    gt_edges = _relative_edges(gt)
    src = pred_edges[train_edges, :3, 3]
    dst = gt_edges[train_edges, :3, 3]
    if args.fit_mode == "affine":
        a, b = _fit_affine(src, dst, args.ridge)
    else:
        a, b = _fit_diag_bias(src, dst, args.ridge)

    corrected_trans = pred_edges[:, :3, 3] @ a.T + b[None, :]
    corrected = _compose_zero_rotation(pred[0], corrected_trans)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, corrected.astype(np.float32))

    corr_edges = _relative_edges(corrected)
    row: dict[str, object] = {
        "label": args.label,
        "fit_mode": args.fit_mode,
        "split_protocol": args.split_protocol,
        "num_train_edges": len(train_edges),
        "num_held_edges": len(held_edges),
        "pred": str(args.pred),
        "out": str(args.out),
    }
    row.update(_pose_metrics(corrected, gt))
    row.update(_edge_metrics(pred_edges, gt_edges, train_edges, "raw_train"))
    row.update(_edge_metrics(pred_edges, gt_edges, held_edges, "raw_held"))
    row.update(_edge_metrics(corr_edges, gt_edges, train_edges, "corr_train"))
    row.update(_edge_metrics(corr_edges, gt_edges, held_edges, "corr_held"))
    row["matrix"] = np.array2string(a, precision=8, separator=",")
    row["bias"] = np.array2string(b, precision=8, separator=",")

    print("translation edge calibration")
    for k, v in row.items():
        if isinstance(v, float):
            print(f"{k}\t{v:.8f}")
        else:
            print(f"{k}\t{v}")

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
