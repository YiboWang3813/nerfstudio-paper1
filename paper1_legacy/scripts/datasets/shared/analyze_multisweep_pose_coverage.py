#!/usr/bin/env python3
"""Analyze cross-sweep geometric coverage from ultrasound image-plane poses.

The script samples each frame's image plane using DatasetMeta physical extents,
transforms those samples by ``poses.npy`` into world coordinates, voxelizes the
resulting point cloud, and reports how well a held-out sweep is covered by the
union of the remaining sweeps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - exercised only on minimal envs.
    cKDTree = None


@dataclass(frozen=True)
class LinearMeta:
    name: str
    h: int
    w: int
    near_mm: float
    far_mm: float
    width_mm: float
    center_lateral: bool = True


DATASETS: Dict[str, dict] = {
    "synthetic_liver": {
        "root": Path("/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver"),
        "meta": LinearMeta(
            name="synthetic_liver",
            h=448,
            w=224,
            near_mm=0.0,
            far_mm=140.0,
            width_mm=80.0,
            center_lateral=True,
        ),
        "sweeps": ["l1", "l2", "l3", "r1", "r2", "r3"],
    },
    "spine_phantom": {
        "root": Path("/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom"),
        "meta": LinearMeta(
            name="spine_phantom",
            h=896,
            w=112,
            near_mm=0.0,
            far_mm=120.0,
            width_mm=40.0,
            center_lateral=True,
        ),
        "sweeps": [
            "left1",
            "left1_1",
            "left2",
            "left3",
            "left3_2",
            "right1",
            "right1_1",
            "right3",
            "right3_2",
        ],
    },
}


def _sample_local_points(meta: LinearMeta, sample_h: int, sample_w: int) -> np.ndarray:
    hs = np.linspace(0.0, float(meta.h - 1), int(sample_h), dtype=np.float64)
    ws = np.linspace(0.0, float(meta.w - 1), int(sample_w), dtype=np.float64)
    arr_h, arr_w = np.meshgrid(hs, ws, indexing="ij")

    axial_spacing = (float(meta.far_mm) - float(meta.near_mm)) / float(meta.h)
    lateral_spacing = float(meta.width_mm) / float(meta.w)
    w_offset = 0.5 * float(meta.w - 1) if meta.center_lateral else 0.0

    local_x = (arr_w - w_offset) * lateral_spacing
    local_y = arr_h * axial_spacing + float(meta.near_mm)
    local_z = np.zeros_like(local_x)
    ones = np.ones_like(local_x)
    return np.stack([local_x, local_y, local_z, ones], axis=-1).reshape(-1, 4)


def _load_sweep_points(
    root: Path,
    sweep: str,
    local_points_h: np.ndarray,
    frame_stride: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    poses_path = root / sweep / "poses.npy"
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)
    poses = np.asarray(np.load(poses_path, mmap_mode="r"), dtype=np.float64)[:: int(frame_stride)]
    points = np.einsum("nij,pj->npi", poses, local_points_h, optimize=True)[..., :3]
    translations = poses[:, :3, 3]
    return points.reshape(-1, 3), int(poses.shape[0]), translations


def _voxelize(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    vox = np.floor(points / float(voxel_mm)).astype(np.int64)
    return np.unique(vox, axis=0)


def _intersection_size(a: np.ndarray, b: np.ndarray) -> int:
    if a.size == 0 or b.size == 0:
        return 0
    dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    av = np.ascontiguousarray(a).view(dtype).reshape(-1)
    bv = np.ascontiguousarray(b).view(dtype).reshape(-1)
    return int(np.intersect1d(av, bv, assume_unique=True).shape[0])


def _voxel_centers(voxels: np.ndarray, voxel_mm: float) -> np.ndarray:
    return (voxels.astype(np.float64) + 0.5) * float(voxel_mm)


def _union_voxels(voxel_sets: Iterable[np.ndarray]) -> np.ndarray:
    all_voxels = np.concatenate(list(voxel_sets), axis=0)
    return np.unique(all_voxels, axis=0)


def _query_distances_mm(train_voxels: np.ndarray, test_voxels: np.ndarray, voxel_mm: float) -> np.ndarray:
    if train_voxels.size == 0 or test_voxels.size == 0:
        return np.full((int(test_voxels.shape[0]),), np.inf, dtype=np.float64)
    train_centers = _voxel_centers(train_voxels, voxel_mm)
    test_centers = _voxel_centers(test_voxels, voxel_mm)
    if cKDTree is not None:
        tree = cKDTree(train_centers)
        dist, _ = tree.query(test_centers, k=1, workers=-1)
        return np.asarray(dist, dtype=np.float64)

    # Slow fallback for environments without scipy. Keep memory bounded by chunks.
    out = np.empty((test_centers.shape[0],), dtype=np.float64)
    chunk = 2048
    for start in range(0, test_centers.shape[0], chunk):
        end = min(start + chunk, test_centers.shape[0])
        d2 = np.sum((test_centers[start:end, None, :] - train_centers[None, :, :]) ** 2, axis=-1)
        out[start:end] = np.sqrt(np.min(d2, axis=1))
    return out


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def _bounds(points: np.ndarray) -> str:
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    span = hi - lo
    return (
        f"x[{_fmt(lo[0])},{_fmt(hi[0])}] "
        f"y[{_fmt(lo[1])},{_fmt(hi[1])}] "
        f"z[{_fmt(lo[2])},{_fmt(hi[2])}] "
        f"span[{_fmt(span[0])},{_fmt(span[1])},{_fmt(span[2])}]"
    )


def analyze_dataset(
    dataset: str,
    sweeps: Sequence[str],
    sample_h: int,
    sample_w: int,
    frame_stride: int,
    voxel_mm: float,
    tolerances_mm: Sequence[float],
    skip_pairwise: bool,
    fast_cell_coverage: bool,
) -> str:
    spec = DATASETS[dataset]
    root: Path = spec["root"]
    meta: LinearMeta = spec["meta"]
    local_points = _sample_local_points(meta, sample_h, sample_w)

    sweep_points: Dict[str, np.ndarray] = {}
    sweep_voxels: Dict[str, np.ndarray] = {}
    sweep_frames: Dict[str, int] = {}
    sweep_centers: Dict[str, np.ndarray] = {}
    for sweep in sweeps:
        points, n_frames, translations = _load_sweep_points(root, sweep, local_points, frame_stride)
        sweep_points[sweep] = points
        sweep_voxels[sweep] = _voxelize(points, voxel_mm)
        sweep_frames[sweep] = n_frames
        sweep_centers[sweep] = translations

    all_union = _union_voxels(sweep_voxels.values())
    sum_voxels = sum(int(v.shape[0]) for v in sweep_voxels.values())
    redundancy = float(sum_voxels) / max(1.0, float(all_union.shape[0]))

    lines: List[str] = []
    lines.append(f"# Multi-Sweep Pose Coverage: {dataset}")
    lines.append("")
    lines.append(f"- root: `{root}`")
    lines.append(f"- sweeps: `{', '.join(sweeps)}`")
    lines.append(f"- sampled image grid per frame: `{sample_h}x{sample_w}`")
    lines.append(f"- frame stride: `{frame_stride}`")
    lines.append(f"- voxel size: `{voxel_mm:g} mm`")
    lines.append(f"- total union voxels: `{all_union.shape[0]}`")
    lines.append(f"- sum per-sweep voxels / union voxels: `{redundancy:.3f}`")
    lines.append("  - `1.0` means nearly no overlap; larger means more repeated coverage.")
    lines.append("")
    lines.append("## Sweep Footprints")
    lines.append("")
    lines.append("| sweep | frames used | voxels | point bounds mm | pose-center span mm |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for sweep in sweeps:
        centers = sweep_centers[sweep]
        center_span = centers.max(axis=0) - centers.min(axis=0)
        lines.append(
            f"| `{sweep}` | {sweep_frames[sweep]} | {sweep_voxels[sweep].shape[0]} | "
            f"{_bounds(sweep_points[sweep])} | "
            f"[{_fmt(center_span[0])}, {_fmt(center_span[1])}, {_fmt(center_span[2])}] |"
        )

    lines.append("")
    lines.append("## Leave-One-Sweep-Out Coverage")
    lines.append("")
    header = [
        "held-out",
        "held-out voxels",
        "train union voxels",
        "median nn mm",
        "p95 nn mm",
        "max nn mm",
    ] + [f"covered <= {tol:g}mm" for tol in tolerances_mm]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    coverage_rows = []
    for heldout in sweeps:
        train_sweeps = [s for s in sweeps if s != heldout]
        train_union = _union_voxels(sweep_voxels[s] for s in train_sweeps)
        test_voxels = sweep_voxels[heldout]
        if fast_cell_coverage:
            row_coverages = []
            for tol in tolerances_mm:
                train_cells = _union_voxels(_voxelize(sweep_points[s], tol) for s in train_sweeps)
                test_cells = _voxelize(sweep_points[heldout], tol)
                row_coverages.append(_intersection_size(train_cells, test_cells) / max(1, int(test_cells.shape[0])))
            fields = [
                f"`{heldout}`",
                str(test_voxels.shape[0]),
                str(train_union.shape[0]),
                "n/a",
                "n/a",
                "n/a",
            ] + [f"{100.0 * c:.1f}%" for c in row_coverages]
        else:
            dist = _query_distances_mm(train_union, test_voxels, voxel_mm)
            row_coverages = [float(np.mean(dist <= float(tol))) for tol in tolerances_mm]
            coverage_rows.append((heldout, row_coverages, dist))
            fields = [
                f"`{heldout}`",
                str(test_voxels.shape[0]),
                str(train_union.shape[0]),
                _fmt(float(np.percentile(dist, 50))),
                _fmt(float(np.percentile(dist, 95))),
                _fmt(float(np.max(dist))),
            ] + [f"{100.0 * c:.1f}%" for c in row_coverages]
        lines.append("| " + " | ".join(fields) + " |")

    if not skip_pairwise:
        lines.append("")
        lines.append("## Pairwise Symmetric Coverage")
        lines.append("")
        tol_for_pair = float(tolerances_mm[-1])
        lines.append(f"Coverage uses nearest-neighbor distance <= `{tol_for_pair:g} mm`.")
        lines.append("")
        lines.append("| sweep A | sweep B | A covered by B | B covered by A |")
        lines.append("| --- | --- | ---: | ---: |")
        for i, a in enumerate(sweeps):
            for b in sweeps[i + 1 :]:
                dist_a = _query_distances_mm(sweep_voxels[b], sweep_voxels[a], voxel_mm)
                dist_b = _query_distances_mm(sweep_voxels[a], sweep_voxels[b], voxel_mm)
                lines.append(
                    f"| `{a}` | `{b}` | "
                    f"{100.0 * float(np.mean(dist_a <= tol_for_pair)):.1f}% | "
                    f"{100.0 * float(np.mean(dist_b <= tol_for_pair)):.1f}% |"
                )

    lines.append("")
    lines.append("## Interpretation Aid")
    lines.append("")
    lines.append("- Low leave-one-sweep-out coverage means the held-out sweep observes spatial regions not sampled by the training sweeps.")
    lines.append("- High median/p95 nearest-neighbor distance means the held-out geometry is physically far from the train union, so novel-view testing becomes extrapolation rather than interpolation.")
    lines.append("- With `--fast-cell-coverage`, coverage is same-cell overlap after re-voxelizing at each tolerance, so distance columns are intentionally disabled.")
    lines.append("- If redundancy is close to 1, combining sweeps mostly expands the scene footprint instead of adding repeated constraints over the same tissue.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--sweeps", nargs="*", default=None, help="Sweep names. Defaults to all known sweeps.")
    parser.add_argument("--sample-h", type=int, default=32)
    parser.add_argument("--sample-w", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--tolerances-mm", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    parser.add_argument("--skip-pairwise", action="store_true", help="Skip the slower pairwise sweep table.")
    parser.add_argument("--fast-cell-coverage", action="store_true", help="Use fast tolerance-cell overlap instead of nearest-neighbor distances.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweeps = args.sweeps or DATASETS[args.dataset]["sweeps"]
    report = analyze_dataset(
        dataset=args.dataset,
        sweeps=sweeps,
        sample_h=args.sample_h,
        sample_w=args.sample_w,
        frame_stride=args.frame_stride,
        voxel_mm=args.voxel_mm,
        tolerances_mm=args.tolerances_mm,
        skip_pairwise=bool(args.skip_pairwise),
        fast_cell_coverage=bool(args.fast_cell_coverage),
    )
    print(report)


if __name__ == "__main__":
    main()
