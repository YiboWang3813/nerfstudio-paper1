#!/usr/bin/env python3
"""Estimate useful axial crop ranges for the Ultra-NeRF spine phantom sweeps.

The spine phantom arrays contain NaNs and a dark background near -0.125, so this
script uses robust row-wise statistics instead of image-library dependencies.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SweepStats:
    name: str
    shape: tuple[int, int, int]
    valid_fraction: np.ndarray
    energy: np.ndarray
    signal_fractions: dict[float, np.ndarray]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def discover_sweeps(root: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "images.npy")
        if os.path.isfile(path):
            out.append(name)
    return out


def interval_from_energy(row_energy: np.ndarray, keep: float) -> tuple[int, int] | None:
    total = float(np.sum(row_energy))
    if total <= 0.0:
        return None
    lo_mass = (1.0 - keep) * 0.5 * total
    hi_mass = (1.0 - (1.0 - keep) * 0.5) * total
    cum = np.cumsum(row_energy)
    lo = int(np.searchsorted(cum, lo_mass, side="left"))
    hi = int(np.searchsorted(cum, hi_mass, side="left"))
    return lo, min(hi, row_energy.shape[0] - 1)


def interval_from_mask(mask: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def pad_interval(interval: tuple[int, int] | None, pad: int, height: int) -> tuple[int, int] | None:
    if interval is None:
        return None
    lo, hi = interval
    return max(0, lo - pad), min(height - 1, hi + pad)


def interval_to_slice(interval: tuple[int, int] | None) -> str:
    if interval is None:
        return "none"
    lo, hi = interval
    return f"{lo}:{hi + 1}"


def merge_union(intervals: Iterable[tuple[int, int] | None]) -> tuple[int, int] | None:
    vals = [x for x in intervals if x is not None]
    if not vals:
        return None
    return min(x[0] for x in vals), max(x[1] for x in vals)


def merge_intersection(intervals: Iterable[tuple[int, int] | None]) -> tuple[int, int] | None:
    vals = [x for x in intervals if x is not None]
    if not vals:
        return None
    lo = max(x[0] for x in vals)
    hi = min(x[1] for x in vals)
    if lo > hi:
        return None
    return lo, hi


def analyze_sweep(
    path: str,
    thresholds: list[float],
    background: float,
    sane_min: float,
    sane_max: float,
    chunk_frames: int,
    max_frames: int | None,
) -> SweepStats:
    images = np.load(path, mmap_mode="r")
    if images.ndim != 3:
        raise ValueError(f"Expected images shape (N,H,W), got {images.shape} from {path}")

    n, h, w = images.shape
    n_use = min(n, max_frames) if max_frames is not None else n
    denom = float(n_use * w)
    valid_count = np.zeros(h, dtype=np.float64)
    energy = np.zeros(h, dtype=np.float64)
    signal_counts = {thr: np.zeros(h, dtype=np.float64) for thr in thresholds}

    for start in range(0, n_use, chunk_frames):
        stop = min(start + chunk_frames, n_use)
        chunk = np.asarray(images[start:stop], dtype=np.float32)
        sane = np.isfinite(chunk) & (chunk >= sane_min) & (chunk <= sane_max)
        valid_count += sane.sum(axis=(0, 2))
        energy += np.where(sane, np.maximum(chunk - background, 0.0), 0.0).sum(axis=(0, 2))
        for thr in thresholds:
            signal_counts[thr] += (sane & (chunk > thr)).sum(axis=(0, 2))

    return SweepStats(
        name=os.path.basename(os.path.dirname(path)),
        shape=(n, h, w),
        valid_fraction=valid_count / denom,
        energy=energy,
        signal_fractions={thr: counts / denom for thr, counts in signal_counts.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/home/boywang25/ultrasound-scene-reconstruction/data/spine_phantom",
        help="spine_phantom root containing per-sweep subdirectories.",
    )
    parser.add_argument(
        "--sweeps",
        default="",
        help="Comma-separated sweep names. Defaults to all directories with images.npy.",
    )
    parser.add_argument("--background", type=float, default=-0.125)
    parser.add_argument("--sane-min", type=float, default=-1.0)
    parser.add_argument("--sane-max", type=float, default=1.0)
    parser.add_argument(
        "--thresholds",
        default="-0.105,-0.075,-0.025,0.0,0.05,0.1",
        help="Comma-separated intensity thresholds for row signal fractions.",
    )
    parser.add_argument("--min-signal-frac", type=float, default=0.05)
    parser.add_argument("--energy-keep", type=float, default=0.99)
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0, help="Use only first K frames; 0 means all.")
    args = parser.parse_args()

    sweeps = [x.strip() for x in args.sweeps.split(",") if x.strip()] if args.sweeps else discover_sweeps(args.root)
    thresholds = parse_csv_floats(args.thresholds)
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else None

    if not sweeps:
        raise SystemExit(f"No sweeps with images.npy found under {args.root}")
    if not (0.0 < args.energy_keep < 1.0):
        raise SystemExit("--energy-keep must be in (0, 1)")

    all_energy_intervals: list[tuple[int, int] | None] = []
    h_ref: int | None = None

    print(f"root: {args.root}")
    print(f"sweeps: {', '.join(sweeps)}")
    print(
        "params:"
        f" background={args.background}, sane=[{args.sane_min},{args.sane_max}],"
        f" energy_keep={args.energy_keep}, pad={args.pad},"
        f" min_signal_frac={args.min_signal_frac}"
    )
    print()

    for sweep in sweeps:
        path = os.path.join(args.root, sweep, "images.npy")
        stats = analyze_sweep(
            path,
            thresholds=thresholds,
            background=args.background,
            sane_min=args.sane_min,
            sane_max=args.sane_max,
            chunk_frames=args.chunk_frames,
            max_frames=max_frames,
        )
        _, h, _ = stats.shape
        if h_ref is None:
            h_ref = h
        elif h != h_ref:
            raise ValueError(f"Height mismatch: {sweep} has H={h}, expected {h_ref}")

        energy_interval = interval_from_energy(stats.energy, args.energy_keep)
        energy_interval_padded = pad_interval(energy_interval, args.pad, h)
        all_energy_intervals.append(energy_interval_padded)

        valid_interval = interval_from_mask(stats.valid_fraction >= 0.95)
        print(f"[{sweep}] shape={stats.shape}")
        print(
            f"  valid>=95% rows: {interval_to_slice(valid_interval)}"
            f"; energy {args.energy_keep:.3f}: {interval_to_slice(energy_interval)}"
            f"; padded: {interval_to_slice(energy_interval_padded)}"
        )
        for thr in thresholds:
            sig_interval = interval_from_mask(stats.signal_fractions[thr] >= args.min_signal_frac)
            peak = float(np.max(stats.signal_fractions[thr]))
            print(
                f"  signal>{thr:g} frac>={args.min_signal_frac:g}:"
                f" {interval_to_slice(sig_interval)} (peak={peak:.3f})"
            )
        print()

    union = merge_union(all_energy_intervals)
    inter = merge_intersection(all_energy_intervals)
    print("recommendation based on padded per-sweep energy intervals:")
    print(f"  conservative union crop rows: {interval_to_slice(union)}")
    print(f"  common intersection crop rows: {interval_to_slice(inter)}")
    if union is not None:
        lo, hi = union
        print(f"  height after conservative crop: {hi - lo + 1}")


if __name__ == "__main__":
    main()
