#!/usr/bin/env python3
"""Analyze interval holdout splits for one-sweep ultrasound reconstruction."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import numpy as np


def build_holdout_indices(n_frames: int, trim_each: int, num_samples: int) -> List[int]:
    """Match ``Scene._build_holdout_indices`` exactly."""
    if n_frames <= 0:
        raise ValueError("Cannot build holdout split for empty dataset.")
    if trim_each < 0:
        raise ValueError(f"holdout_trim_each must be >= 0, got {trim_each}")
    if num_samples <= 0:
        raise ValueError(f"holdout_num_samples must be > 0, got {num_samples}")
    i_min = int(trim_each)
    i_max = int(n_frames - 1 - trim_each)
    if i_max < i_min:
        raise ValueError(f"No valid holdout interval after trimming: N={n_frames}, trim_each={trim_each}")
    n_avail = i_max - i_min + 1
    if num_samples > n_avail:
        raise ValueError(
            f"holdout_num_samples={num_samples} exceeds available={n_avail} "
            f"(N={n_frames}, trim_each={trim_each})"
        )
    picks = np.linspace(i_min, i_max, num_samples, dtype=np.float64)
    picks = np.rint(picks).astype(np.int64)
    uniq = np.unique(picks)
    if int(uniq.size) < int(num_samples):
        raise ValueError(
            f"Rounded interval sampling collapsed to {uniq.size} unique indices; "
            "reduce holdout_num_samples."
        )
    return [int(x) for x in uniq.tolist()]


def _runs(mask: np.ndarray) -> List[int]:
    out: List[int] = []
    cur = 0
    for v in mask.tolist():
        if bool(v):
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _window_pattern(train_mask: np.ndarray, center: int, radius: int) -> str:
    chars: List[str] = []
    for i in range(center - radius, center + radius + 1):
        if i < 0 or i >= int(train_mask.size):
            chars.append("x")
        else:
            chars.append("1" if bool(train_mask[i]) else "0")
    return "".join(chars)


def _percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _fmt_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def _valid_sweep_dirs(root: Path) -> List[Path]:
    if (root / "images.npy").is_file() and (root / "poses.npy").is_file():
        return [root]
    dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "images.npy").is_file() and (p / "poses.npy").is_file():
            dirs.append(p)
    return dirs


def analyze_sweep(sweep_dir: Path, trim_each: int, samples: int, window_sizes: List[int]) -> dict:
    images = np.load(sweep_dir / "images.npy", mmap_mode="r")
    poses = np.load(sweep_dir / "poses.npy", mmap_mode="r")
    n = int(images.shape[0])
    if int(poses.shape[0]) != n:
        raise ValueError(f"{sweep_dir}: images N={n} but poses N={poses.shape[0]}")

    test_idx = build_holdout_indices(n, trim_each, samples)
    test_set = set(test_idx)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False
    train_runs = _runs(train_mask)

    translations = np.asarray(poses[:, :3, 3], dtype=np.float64)
    step_mm = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    path = np.zeros(n, dtype=np.float64)
    if n > 1:
        path[1:] = np.cumsum(step_mm)

    nearest_idx_gaps: List[int] = []
    nearest_mm_gaps: List[float] = []
    bracketing = 0
    for t in test_idx:
        left = t - 1
        while left >= 0 and left in test_set:
            left -= 1
        right = t + 1
        while right < n and right in test_set:
            right += 1
        candidates = []
        if left >= 0:
            candidates.append((abs(t - left), abs(float(path[t] - path[left]))))
        if right < n:
            candidates.append((abs(right - t), abs(float(path[right] - path[t]))))
        if candidates:
            nearest_idx_gaps.append(int(min(x[0] for x in candidates)))
            nearest_mm_gaps.append(float(min(x[1] for x in candidates)))
        if left >= 0 and right < n:
            bracketing += 1

    adjacent_train_edges = sum(1 for i in range(n - 1) if train_mask[i] and train_mask[i + 1])
    total_edges = max(0, n - 1)
    test_gaps = np.diff(np.asarray(test_idx, dtype=np.int64)) if len(test_idx) >= 2 else np.asarray([])
    patterns5 = Counter(_window_pattern(train_mask, t, 2) for t in test_idx)
    patterns7 = Counter(_window_pattern(train_mask, t, 3) for t in test_idx)

    contiguous_windows = {}
    for w in window_sizes:
        total = max(0, n - w + 1)
        count = 0
        for s in range(total):
            if bool(train_mask[s : s + w].all()):
                count += 1
        contiguous_windows[int(w)] = (count, total)

    phase_a_w = 8
    phase_a_s = 4
    phase_a_total = len(range(0, max(0, n - phase_a_w + 1), phase_a_s))
    phase_a_count = 0
    for s in range(0, max(0, n - phase_a_w + 1), phase_a_s):
        if bool(train_mask[s : s + phase_a_w].all()):
            phase_a_count += 1

    return {
        "name": sweep_dir.name,
        "path": str(sweep_dir),
        "n": n,
        "test_idx": test_idx,
        "train_n": int(train_mask.sum()),
        "test_n": int(len(test_idx)),
        "test_gap_min": int(test_gaps.min()) if test_gaps.size else 0,
        "test_gap_med": _percentile(test_gaps, 50) if test_gaps.size else 0.0,
        "test_gap_max": int(test_gaps.max()) if test_gaps.size else 0,
        "train_run_min": int(min(train_runs)) if train_runs else 0,
        "train_run_med": _percentile(train_runs, 50),
        "train_run_max": int(max(train_runs)) if train_runs else 0,
        "nearest_idx_med": _percentile(nearest_idx_gaps, 50),
        "nearest_idx_max": int(max(nearest_idx_gaps)) if nearest_idx_gaps else 0,
        "nearest_mm_med": _percentile(nearest_mm_gaps, 50),
        "nearest_mm_p95": _percentile(nearest_mm_gaps, 95),
        "nearest_mm_max": float(max(nearest_mm_gaps)) if nearest_mm_gaps else 0.0,
        "bracketing_frac": bracketing / max(1, len(test_idx)),
        "step_mm_med": _percentile(step_mm, 50),
        "step_mm_p95": _percentile(step_mm, 95),
        "step_mm_max": float(step_mm.max()) if step_mm.size else 0.0,
        "path_mm": float(path[-1]) if path.size else 0.0,
        "adjacent_train_edges": int(adjacent_train_edges),
        "total_edges": int(total_edges),
        "edge_keep_frac": adjacent_train_edges / max(1, total_edges),
        "edge_removed": int(total_edges - adjacent_train_edges),
        "patterns5": patterns5,
        "patterns7": patterns7,
        "contiguous_windows": contiguous_windows,
        "phase_a_count": int(phase_a_count),
        "phase_a_total": int(phase_a_total),
    }


def _table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def render_report(results_by_samples: dict, trim_each: int, window_sizes: List[int]) -> str:
    lines: List[str] = []
    lines.append("# Holdout Protocol Analysis")
    lines.append("")
    lines.append("This report analyzes `scene.loading.test_mode=holdout_interval` for one-sweep offline reconstruction.")
    lines.append("The split logic matches `Scene._build_holdout_indices`: interval sampling by `np.linspace(trim_each, N-1-trim_each, holdout_num_samples)`, rounded to unique integer frame indices.")
    lines.append("")
    lines.append(f"- `holdout_trim_each`: {trim_each}")
    lines.append(f"- FiMA contiguous window sizes checked: {', '.join(str(x) for x in window_sizes)}")
    lines.append("- `1` in local patterns means train frame; `0` means held-out test frame.")
    lines.append("")

    for samples, results in results_by_samples.items():
        lines.append(f"## holdout_num_samples={samples}")
        lines.append("")
        rows = [[
            "sweep",
            "N",
            "train/test",
            "test gap idx med",
            "train run med",
            "nearest train idx med/max",
            "nearest train mm med/p95/max",
            "bracketed",
            "FiMA edges kept",
            "FiMA W8 kept",
        ]]
        for r in results:
            w8_count, w8_total = r["contiguous_windows"].get(8, (0, 0))
            rows.append([
                r["name"],
                str(r["n"]),
                f"{r['train_n']}/{r['test_n']}",
                f"{_fmt_float(r['test_gap_med'])} [{r['test_gap_min']},{r['test_gap_max']}]",
                f"{_fmt_float(r['train_run_med'])} [{r['train_run_min']},{r['train_run_max']}]",
                f"{_fmt_float(r['nearest_idx_med'])}/{r['nearest_idx_max']}",
                f"{_fmt_float(r['nearest_mm_med'])}/{_fmt_float(r['nearest_mm_p95'])}/{_fmt_float(r['nearest_mm_max'])}",
                f"{100.0 * r['bracketing_frac']:.1f}%",
                f"{r['adjacent_train_edges']}/{r['total_edges']} ({100.0 * r['edge_keep_frac']:.1f}%)",
                f"{w8_count}/{w8_total} ({100.0 * w8_count / max(1, w8_total):.1f}%)",
            ])
        lines.append(_table(rows))
        lines.append("")

        primary = results[0]
        lines.append(f"### Primary Sweep Detail: `{primary['name']}`")
        lines.append("")
        lines.append(f"- Path: `{primary['path']}`")
        lines.append(f"- Held-out indices: `{primary['test_idx']}`")
        lines.append(
            "- Adjacent physical frame step from poses: "
            f"median={_fmt_float(primary['step_mm_med'])} mm, "
            f"p95={_fmt_float(primary['step_mm_p95'])} mm, "
            f"max={_fmt_float(primary['step_mm_max'])} mm, "
            f"path length={_fmt_float(primary['path_mm'])} mm."
        )
        lines.append(
            "- Centered length-5 patterns over test frames: "
            + ", ".join(f"`{k}` x{v}" for k, v in primary["patterns5"].most_common())
        )
        lines.append(
            "- Centered length-7 patterns over test frames: "
            + ", ".join(f"`{k}` x{v}" for k, v in primary["patterns7"].most_common())
        )
        for w in window_sizes:
            c, total = primary["contiguous_windows"][w]
            lines.append(f"- Protocol-safe contiguous train windows W={w}: {c}/{total} ({100.0 * c / max(1, total):.1f}%).")
        lines.append(
            f"- FiMA phase-A local windows W=8 stride=4: {primary['phase_a_count']}/{primary['phase_a_total']} "
            f"({100.0 * primary['phase_a_count'] / max(1, primary['phase_a_total']):.1f}%)."
        )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. The current interval holdout is deterministic and evenly spaced, not random. It does not create frequent local patterns like `10101`; for the primary sweep, each test frame is isolated and bracketed by train frames.")
    lines.append("2. For GS reconstruction metrics, the protocol mostly measures single-slice interpolation inside a dense one-sweep trajectory. Test images are not far from the nearest train slice in physical distance, so a failed idea is unlikely to be explained only by missing immediate neighboring train coverage.")
    lines.append("3. For FiMA pose learning, the same protocol is harsher: every isolated test frame removes the two adjacent supervised train edges crossing that frame, and protocol-safe global-contiguous windows are cut into repeated train runs. With 40 holdouts, W=8 windows can become a limiting factor on shorter sweeps.")
    lines.append("4. Because `holdout_interval` reuses `train_container` as `test_container`, any method that overwrites the full container pose list before GS evaluation also changes holdout test-frame poses. Reports must state whether evaluation used GT poses or a predicted/replaced pose list.")
    lines.append("")
    lines.append("## Recommended Next Protocol Checks")
    lines.append("")
    lines.append("1. Keep interval holdout for comparable interpolation PSNR/SSIM, but report it explicitly as dense one-sweep interpolation.")
    lines.append("2. Add a block or gap holdout sanity split for innovation screening: hold out contiguous chunks so test frames are not always immediate train-neighbor interpolations.")
    lines.append("3. For FiMA, decouple pose-network validation from GS interpolation split: train pose edges on contiguous train blocks and evaluate pose on held-out edges/chunks with explicit mm error.")
    lines.append("4. Re-read pose-prior experiment conclusions after checking whether replaced test poses were used in holdout evaluation.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-sweep",
        type=Path,
        default=Path("/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370"),
        help="Primary sweep directory with images.npy and poses.npy.",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=Path("/home/boywang25/ultrasound-scene-reconstruction/data/tus_rec_2024_qc"),
        help="Optional root whose direct child sweeps are summarized after the primary sweep.",
    )
    parser.add_argument("--trim-each", type=int, default=10)
    parser.add_argument("--samples", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/holdout_protocol.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = args.primary_sweep.resolve()
    sweep_dirs = [primary]
    if args.scan_root:
        for p in _valid_sweep_dirs(args.scan_root.resolve()):
            if p.resolve() != primary:
                sweep_dirs.append(p.resolve())

    results_by_samples = {}
    for samples in args.samples:
        results = []
        for sweep_dir in sweep_dirs:
            try:
                results.append(analyze_sweep(sweep_dir, args.trim_each, int(samples), args.window_sizes))
            except ValueError as exc:
                print(f"[WARN] skipped {sweep_dir}: {exc}")
        results_by_samples[int(samples)] = results

    report = render_report(results_by_samples, args.trim_each, args.window_sizes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
