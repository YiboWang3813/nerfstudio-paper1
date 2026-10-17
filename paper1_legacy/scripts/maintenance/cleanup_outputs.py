#!/usr/bin/env python3
"""Trim OmniUltra output runs to the canonical retention policy.

The script is intentionally conservative:
- It only treats directories with run metadata as run directories.
- It keeps checkpoint files for the best metric step and the last step.
- It keeps non-iteration checkpoint files because their semantics are unknown.
- It trims ``vis/item_*`` folders to a fixed representative subset.

Run with ``--dry-run`` first. Use ``--apply`` to delete files/directories and
write manifests.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_RE = re.compile(r"^iteration_(\d+)(?:_(.+))?\.(pth|pt|ckpt)$")
VIS_ITEM_RE = re.compile(r"^item_(\d+)$")
RUN_MARKERS = (
    "run_manifest.yaml",
    "resolved_config.yaml",
    "config_dump.yaml",
)


@dataclass
class CheckpointPlan:
    best_step: int | None
    last_step: int | None
    keep_files: list[Path]
    delete_files: list[Path]
    unknown_keep_files: list[Path]
    metric_key: str


@dataclass
class VisPlan:
    keep_dirs: list[Path]
    delete_dirs: list[Path]
    keep_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"), help="Output tree root to scan.")
    parser.add_argument("--apply", action="store_true", help="Apply deletions and write manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without modifying files.")
    parser.add_argument("--metric-key", default="test/psnr", help="Scalar key used to choose best checkpoint.")
    parser.add_argument("--vis-keep", type=int, default=10, help="Number of vis/item_* folders to keep per run.")
    parser.add_argument("--include-preproduction", action="store_true", help="Also clean outputs/preproduction runs.")
    parser.add_argument("--keep-all-file", default=".keep_all_outputs", help="Marker filename that disables cleanup in a run.")
    args = parser.parse_args()
    if args.apply == args.dry_run:
        parser.error("choose exactly one of --dry-run or --apply")
    return args


def is_run_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in RUN_MARKERS)


def iter_run_dirs(root: Path, include_preproduction: bool) -> list[Path]:
    run_dirs: list[Path] = []
    if not root.exists():
        return run_dirs
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if not include_preproduction and "preproduction" in path.parts:
            continue
        if is_run_dir(path):
            run_dirs.append(path)
    return sorted(run_dirs)


def read_best_step(run_dir: Path, metric_key: str) -> int | None:
    scalars = run_dir / "logs" / "scalars.tsv"
    if not scalars.exists():
        return None
    best_step: int | None = None
    best_value: float | None = None
    try:
        with scalars.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("key") != metric_key:
                    continue
                try:
                    step = int(row["step"])
                    value = float(row["value"])
                except (KeyError, TypeError, ValueError):
                    continue
                if best_value is None or value > best_value:
                    best_step = step
                    best_value = value
    except UnicodeDecodeError:
        return None
    return best_step


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def plan_checkpoints(run_dir: Path, metric_key: str) -> CheckpointPlan | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    checkpoint_files = sorted(
        p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix in {".pth", ".pt", ".ckpt"}
    )
    if not checkpoint_files:
        return None

    step_files: dict[int, list[Path]] = {}
    unknown_keep: list[Path] = []
    for path in checkpoint_files:
        step = checkpoint_step(path)
        if step is None:
            unknown_keep.append(path)
        else:
            step_files.setdefault(step, []).append(path)

    if not step_files:
        return CheckpointPlan(None, None, [], [], unknown_keep, metric_key)

    last_step = max(step_files)
    best_step = read_best_step(run_dir, metric_key)
    if best_step not in step_files:
        best_step = last_step

    keep_steps = {best_step, last_step}
    keep_files: list[Path] = []
    delete_files: list[Path] = []
    for step, files in step_files.items():
        if step in keep_steps:
            keep_files.extend(files)
        else:
            delete_files.extend(files)

    return CheckpointPlan(best_step, last_step, sorted(keep_files), sorted(delete_files), unknown_keep, metric_key)


def select_evenly(paths: list[Path], keep_count: int) -> set[Path]:
    if len(paths) <= keep_count:
        return set(paths)
    if keep_count <= 1:
        return {paths[0]}
    selected: set[Path] = set()
    for idx in range(keep_count):
        pos = round(idx * (len(paths) - 1) / (keep_count - 1))
        selected.add(paths[pos])
    return selected


def plan_vis(run_dir: Path, keep_count: int) -> VisPlan | None:
    vis_dir = run_dir / "vis"
    if not vis_dir.is_dir():
        return None
    item_dirs: list[tuple[int, Path]] = []
    for path in vis_dir.iterdir():
        if not path.is_dir():
            continue
        match = VIS_ITEM_RE.match(path.name)
        if not match:
            continue
        item_dirs.append((int(match.group(1)), path))
    if not item_dirs:
        return None
    sorted_dirs = [path for _, path in sorted(item_dirs)]
    keep = select_evenly(sorted_dirs, keep_count)
    delete = [path for path in sorted_dirs if path not in keep]
    return VisPlan(sorted(keep), delete, keep_count)


def rel_list(paths: list[Path], base: Path) -> list[str]:
    return [str(path.relative_to(base)) for path in paths]


def write_checkpoint_manifest(run_dir: Path, plan: CheckpointPlan, apply: bool) -> None:
    manifest = run_dir / "checkpoints" / "checkpoint_manifest.yaml"
    lines = [
        "policy: best_and_last",
        f"metric_key: {plan.metric_key}",
        f"best_step: {plan.best_step if plan.best_step is not None else ''}",
        f"last_step: {plan.last_step if plan.last_step is not None else ''}",
        "kept_files:",
    ]
    lines.extend(f"  - {item}" for item in rel_list(plan.keep_files + plan.unknown_keep_files, run_dir))
    lines.append("deleted_files:")
    lines.extend(f"  - {item}" for item in rel_list(plan.delete_files, run_dir))
    lines.append(f"deleted_count: {len(plan.delete_files)}")
    if apply:
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vis_manifest(run_dir: Path, plan: VisPlan, apply: bool) -> None:
    manifest = run_dir / "vis" / "vis_manifest.yaml"
    lines = [
        "policy: selected_items",
        f"requested_keep_count: {plan.keep_count}",
        "kept_items:",
    ]
    lines.extend(f"  - {item}" for item in rel_list(plan.keep_dirs, run_dir))
    lines.append("deleted_items:")
    lines.extend(f"  - {item}" for item in rel_list(plan.delete_dirs, run_dir))
    lines.append(f"deleted_count: {len(plan.delete_dirs)}")
    if apply:
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bytes_for(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return total


def format_bytes(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    value = num_bytes / (1024**idx)
    return f"{value:.2f} {units[idx]}"


def remove_files(paths: list[Path]) -> None:
    for path in paths:
        path.unlink()


def remove_dirs(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    root = args.root
    run_dirs = iter_run_dirs(root, args.include_preproduction)

    total_delete_bytes = 0
    cleaned_runs = 0
    for run_dir in run_dirs:
        if (run_dir / args.keep_all_file).exists():
            continue

        ckpt_plan = plan_checkpoints(run_dir, args.metric_key)
        vis_plan = plan_vis(run_dir, args.vis_keep)
        delete_files = ckpt_plan.delete_files if ckpt_plan else []
        delete_dirs = vis_plan.delete_dirs if vis_plan else []
        if not delete_files and not delete_dirs:
            continue

        cleaned_runs += 1
        delete_bytes = bytes_for(delete_files) + bytes_for(delete_dirs)
        total_delete_bytes += delete_bytes

        print(f"[RUN] {run_dir}")
        if ckpt_plan:
            print(
                "  checkpoints: "
                f"best={ckpt_plan.best_step} last={ckpt_plan.last_step} "
                f"delete={len(ckpt_plan.delete_files)} keep={len(ckpt_plan.keep_files) + len(ckpt_plan.unknown_keep_files)}"
            )
        if vis_plan:
            print(f"  vis: delete={len(vis_plan.delete_dirs)} keep={len(vis_plan.keep_dirs)}")
        print(f"  reclaim: {format_bytes(delete_bytes)}")

        if args.apply:
            if ckpt_plan:
                remove_files(delete_files)
                write_checkpoint_manifest(run_dir, ckpt_plan, apply=True)
            if vis_plan:
                remove_dirs(delete_dirs)
                write_vis_manifest(run_dir, vis_plan, apply=True)

    action = "Applied" if args.apply else "Dry-run"
    print(f"[{action}] runs_with_changes={cleaned_runs} reclaim={format_bytes(total_delete_bytes)}")


if __name__ == "__main__":
    main()
