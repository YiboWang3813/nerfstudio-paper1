#!/usr/bin/env python3
"""Post-process chained pose priors for constrained FiMA/Ours experiments."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _load_poses(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4), got {arr.shape} from {path}")
    return arr


def _relative_edges(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(poses[:-1]) @ poses[1:]


def _compose_edges(anchor: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.empty((edges.shape[0] + 1, 4, 4), dtype=np.float64)
    out[0] = anchor
    for i, edge in enumerate(edges, start=1):
        out[i] = out[i - 1] @ edge
    return out


def _clamp_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm < 0.0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")
    if max_norm <= 0.0:
        return np.zeros_like(vec)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1.0e-12))
    return vec * scale


def _rotation_z_angle(rot: np.ndarray) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12 or abs(angle) <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _clamp_rotation(rot: np.ndarray, max_deg: float) -> np.ndarray:
    if max_deg <= 0.0:
        return np.eye(3, dtype=np.float64)
    tr = float(np.trace(rot))
    cos_th = float(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_th)
    max_rad = math.radians(max_deg)
    if angle <= max_rad:
        return rot
    skew = np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(skew))
    if axis_norm <= 1.0e-12:
        # Near-pi rotations are not expected in constrained ultrasound sweeps;
        # fall back to a z-axis clamp to keep the output well defined.
        return _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.copysign(max_rad, _rotation_z_angle(rot)))
    axis = skew / axis_norm
    return _rotation_from_axis_angle(axis, max_rad)


def _zero_rotation_edges(poses: np.ndarray) -> np.ndarray:
    edges = _relative_edges(poses)
    out_edges = np.repeat(np.eye(4, dtype=np.float64)[None, ...], edges.shape[0], axis=0)
    out_edges[:, :3, 3] = edges[:, :3, 3]
    return _compose_edges(poses[0], out_edges)


def _base_translation(
    edges: np.ndarray,
    *,
    source: str,
    step_mm: float,
    manual: list[float] | None,
    gt_edges: np.ndarray | None,
) -> np.ndarray:
    if source == "manual":
        if manual is None or len(manual) != 3:
            raise ValueError("--base-source manual requires exactly three --base-trans values")
        base = np.asarray(manual, dtype=np.float64)
    else:
        src_edges = gt_edges if source.startswith("gt_") else edges
        if src_edges is None:
            raise ValueError(f"--base-source {source} requires --gt")
        trans = src_edges[:, :3, 3]
        if source.endswith("median"):
            base = np.median(trans, axis=0)
        elif source.endswith("mean"):
            base = np.mean(trans, axis=0)
        else:
            raise ValueError(f"unsupported base source: {source}")

    if step_mm > 0.0:
        norm = float(np.linalg.norm(base))
        if norm <= 1.0e-12:
            raise ValueError("cannot normalize zero base translation to --step-mm")
        base = base / norm * float(step_mm)
    return base


def _physical_edges(
    poses: np.ndarray,
    *,
    mode: str,
    base_trans: np.ndarray,
    residual_clamp_mm: float,
    residual_scale: float,
    rotation_mode: str,
    rot_clamp_deg: float,
) -> np.ndarray:
    edges = _relative_edges(poses)
    out = np.repeat(np.eye(4, dtype=np.float64)[None, ...], edges.shape[0], axis=0)

    if mode == "physical_step":
        trans = np.repeat(base_trans[None, :], edges.shape[0], axis=0)
    elif mode == "physical_residual":
        residual = edges[:, :3, 3] - base_trans[None, :]
        residual = _clamp_vector_norm(residual, float(residual_clamp_mm))
        trans = base_trans[None, :] + float(residual_scale) * residual
    else:
        raise ValueError(f"unsupported physical mode: {mode}")
    out[:, :3, 3] = trans

    if rotation_mode == "zero":
        pass
    elif rotation_mode == "keep":
        out[:, :3, :3] = edges[:, :3, :3]
    elif rotation_mode == "clamp":
        for i in range(edges.shape[0]):
            out[i, :3, :3] = _clamp_rotation(edges[i, :3, :3], float(rot_clamp_deg))
    else:
        raise ValueError(f"unsupported rotation mode: {rotation_mode}")
    return out


def _rotation_angle_deg(rot: np.ndarray) -> np.ndarray:
    tr = np.trace(rot, axis1=-2, axis2=-1)
    cos_th = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def _pose_error_summary(name: str, pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"{name}: pred shape {pred.shape} != gt shape {gt.shape}")
    trans = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=-1)
    rel = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rot = _rotation_angle_deg(rel)
    summary = {
        f"{name}_trans_mean_mm": float(trans.mean()),
        f"{name}_trans_median_mm": float(np.median(trans)),
        f"{name}_trans_p95_mm": float(np.percentile(trans, 95)),
        f"{name}_rot_mean_deg": float(rot.mean()),
        f"{name}_rot_median_deg": float(np.median(rot)),
        f"{name}_rot_p95_deg": float(np.percentile(rot, 95)),
    }
    print(
        f"[{name}] trans_mean={summary[f'{name}_trans_mean_mm']:.6f} "
        f"trans_p95={summary[f'{name}_trans_p95_mm']:.6f} "
        f"rot_mean_deg={summary[f'{name}_rot_mean_deg']:.6f} "
        f"rot_p95_deg={summary[f'{name}_rot_p95_deg']:.6f}"
    )
    return summary


def _print_edge_summary(name: str, poses: np.ndarray) -> None:
    edge = _relative_edges(poses)
    trans = np.linalg.norm(edge[:, :3, 3], axis=-1)
    rot = _rotation_angle_deg(edge[:, :3, :3])
    print(
        f"[{name}] edge_trans_mean={trans.mean():.6f} edge_trans_p95={np.percentile(trans, 95):.6f} "
        f"edge_rot_mean_deg={rot.mean():.6f} edge_rot_p95_deg={np.percentile(rot, 95):.6f}"
    )


def _frame_scores_from_physical_deviation(
    poses: np.ndarray,
    *,
    base_trans: np.ndarray,
    trans_scale_mm: float,
    rot_scale_deg: float,
) -> np.ndarray:
    edges = _relative_edges(poses)
    trans_dev = np.linalg.norm(edges[:, :3, 3] - base_trans[None, :], axis=-1) / max(float(trans_scale_mm), 1.0e-12)
    rot_dev = _rotation_angle_deg(edges[:, :3, :3]) / max(float(rot_scale_deg), 1.0e-12)
    edge_scores = trans_dev + rot_dev
    out = np.empty((poses.shape[0],), dtype=np.float64)
    if edge_scores.size == 0:
        out.fill(0.0)
    else:
        out[0] = edge_scores[0]
        out[1:] = edge_scores
    return out


def _weights_from_scores(scores: np.ndarray, *, alpha: float, min_weight: float, max_weight: float) -> np.ndarray:
    weights = min_weight + (max_weight - min_weight) * np.exp(-float(alpha) * scores)
    return np.clip(weights, min_weight, max_weight)


def _write_report(path: Path, *, mode: str, base_trans: np.ndarray | None, summaries: dict[str, float]) -> None:
    lines = [
        "# Physical Uncertain Pose Prior Postprocess Report",
        "",
        f"- mode: `{mode}`",
    ]
    if base_trans is not None:
        lines.append(
            "- base translation: "
            f"`[{base_trans[0]:.6f}, {base_trans[1]:.6f}, {base_trans[2]:.6f}]` mm "
            f"(norm `{np.linalg.norm(base_trans):.6f}` mm)"
        )
    if summaries:
        lines.extend(["", "## Metrics", ""])
        for key in sorted(summaries):
            lines.append(f"- `{key}`: {summaries[key]:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input chained pose prior, shape (N,4,4)")
    parser.add_argument("--output", type=Path, required=True, help="Output chained pose prior, shape (N,4,4)")
    parser.add_argument("--gt", type=Path, default=None, help="Optional GT/reference poses for diagnostics, shape (N,4,4)")
    parser.add_argument(
        "--mode",
        choices=("zero_rotation", "physical_step", "physical_residual"),
        required=True,
        help=(
            "zero_rotation keeps local translations and removes adjacent rotations; "
            "physical_step uses a constant physical step; physical_residual keeps a clamped FiMA residual around it."
        ),
    )
    parser.add_argument(
        "--base-source",
        choices=("input_mean", "input_median", "gt_mean", "gt_median", "manual"),
        default="input_median",
        help="How to estimate the physical adjacent translation base for physical_* modes.",
    )
    parser.add_argument("--base-trans", type=float, nargs=3, default=None, help="Manual base translation vector in mm.")
    parser.add_argument("--step-mm", type=float, default=0.0, help="If >0, normalize the base direction to this magnitude.")
    parser.add_argument(
        "--residual-clamp-mm",
        type=float,
        default=0.15,
        help="Max FiMA translation residual norm retained around the physical step for physical_residual.",
    )
    parser.add_argument("--residual-scale", type=float, default=1.0, help="Scale retained residual after clamping.")
    parser.add_argument(
        "--rotation-mode",
        choices=("zero", "keep", "clamp"),
        default="zero",
        help="Adjacent rotation policy for physical_* modes.",
    )
    parser.add_argument("--rot-clamp-deg", type=float, default=0.1, help="Rotation clamp used with --rotation-mode clamp.")
    parser.add_argument("--out-frame-scores", type=Path, default=None, help="Optional frame uncertainty score .npy output.")
    parser.add_argument("--out-frame-weights", type=Path, default=None, help="Optional frame weight .npy output for Ours.")
    parser.add_argument("--uncertainty-trans-scale-mm", type=float, default=0.15)
    parser.add_argument("--uncertainty-rot-scale-deg", type=float, default=0.1)
    parser.add_argument("--weight-alpha", type=float, default=1.0)
    parser.add_argument("--min-weight", type=float, default=0.1)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=None, help="Optional markdown report path.")
    args = parser.parse_args()

    poses = _load_poses(args.input)
    gt = _load_poses(args.gt) if args.gt is not None else None
    gt_edges = _relative_edges(gt) if gt is not None else None
    _print_edge_summary("input", poses)
    summaries: dict[str, float] = {}
    base_trans = None
    if gt is not None:
        summaries.update(_pose_error_summary("input_abs", poses, gt))

    if args.mode == "zero_rotation":
        out = _zero_rotation_edges(poses)
    else:
        base_trans = _base_translation(
            _relative_edges(poses),
            source=str(args.base_source),
            step_mm=float(args.step_mm),
            manual=args.base_trans,
            gt_edges=gt_edges,
        )
        print(
            "[physical_base] "
            f"source={args.base_source} base_trans={base_trans.tolist()} norm={np.linalg.norm(base_trans):.6f}"
        )
        out_edges = _physical_edges(
            poses,
            mode=args.mode,
            base_trans=base_trans,
            residual_clamp_mm=float(args.residual_clamp_mm),
            residual_scale=float(args.residual_scale),
            rotation_mode=str(args.rotation_mode),
            rot_clamp_deg=float(args.rot_clamp_deg),
        )
        out = _compose_edges(poses[0], out_edges)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out.astype(np.float32))
    _print_edge_summary(args.mode, out)
    if gt is not None:
        summaries.update(_pose_error_summary("output_abs", out, gt))

    if args.out_frame_scores is not None or args.out_frame_weights is not None:
        if base_trans is None:
            base_trans = _base_translation(
                _relative_edges(poses),
                source=str(args.base_source),
                step_mm=float(args.step_mm),
                manual=args.base_trans,
                gt_edges=gt_edges,
            )
        scores = _frame_scores_from_physical_deviation(
            poses,
            base_trans=base_trans,
            trans_scale_mm=float(args.uncertainty_trans_scale_mm),
            rot_scale_deg=float(args.uncertainty_rot_scale_deg),
        )
        weights = _weights_from_scores(
            scores,
            alpha=float(args.weight_alpha),
            min_weight=float(args.min_weight),
            max_weight=float(args.max_weight),
        )
        summaries.update(
            {
                "frame_score_mean": float(scores.mean()),
                "frame_score_p95": float(np.percentile(scores, 95)),
                "frame_weight_mean": float(weights.mean()),
                "frame_weight_min": float(weights.min()),
            }
        )
        if args.out_frame_scores is not None:
            args.out_frame_scores.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.out_frame_scores, scores.astype(np.float32))
            print(f"[postprocess_pose_prior] Saved frame scores: {args.out_frame_scores}")
        if args.out_frame_weights is not None:
            args.out_frame_weights.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.out_frame_weights, weights.astype(np.float32))
            print(f"[postprocess_pose_prior] Saved frame weights: {args.out_frame_weights}")

    if args.report is not None:
        _write_report(args.report, mode=args.mode, base_trans=base_trans, summaries=summaries)
        print(f"[postprocess_pose_prior] Saved report: {args.report}")
    print(f"[postprocess_pose_prior] Saved {args.mode} prior: {args.output}")


if __name__ == "__main__":
    main()
