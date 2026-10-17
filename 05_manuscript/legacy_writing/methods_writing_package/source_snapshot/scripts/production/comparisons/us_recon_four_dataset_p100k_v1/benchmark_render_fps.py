#!/usr/bin/env python3
"""Renderer-only FPS benchmark for the paper comparison tables.

The benchmark intentionally excludes checkpoint loading, scene construction,
metric computation, and image writing from timed sections. It loads completed
production runs, warms up the renderer, synchronizes CUDA, then times only
forward rendering on holdout test frames.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PYTHONPATH_ROOTS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "ops" / "ours_renderer",
    PROJECT_ROOT / "ops" / "ultragauss_renderer",
    PROJECT_ROOT / "ops" / "ultragray_renderer",
    PROJECT_ROOT / "ops" / "gaussianplie_renderer",
    PROJECT_ROOT / "ops" / "ultrags_baseline_renderer",
    PROJECT_ROOT / "ops" / "slice_adapted_3dgs_renderer",
]
for path in reversed(PYTHONPATH_ROOTS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from scene.factory import SceneFactory


GS_METHODS = {
    "ours": ("methods.ours.gaussian_container", "OursGaussContainer", "methods.ours.render", "render"),
    "ultragauss": (
        "methods.ultragauss.gaussian_container",
        "UltraGaussContainer",
        "methods.ultragauss.render",
        "render",
    ),
    "ultragray": (
        "methods.ultragray.gaussian_container",
        "UltraGrayGaussContainer",
        "methods.ultragray.render",
        "render",
    ),
    "gaussianpile": (
        "methods.gaussianpile.gaussian_container",
        "GaussianPileGaussContainer",
        "methods.gaussianpile.render",
        "render",
    ),
    "slice_adapted_3dgs": (
        "methods.slice_adapted_3dgs.gaussian_container",
        "SliceAdapted3DGSContainer",
        "methods.slice_adapted_3dgs.render",
        "render",
    ),
    "ultrags_baseline": (
        "methods.ultrags_baseline.gaussian_container",
        "UltraGSBaselineContainer",
        "methods.ultrags_baseline.render",
        "render",
    ),
}

IMPLICIT_METHODS = {
    "ultranerf": ("methods.ultranerf.field", "UltraNeRFField", "methods.ultranerf.render", "render_slice"),
    "nerfus_no_prior": ("methods.nerfus.field", "NerfusField", "methods.nerfus.render", "render_slice"),
    "nerfus": ("methods.nerfus.field", "NerfusField", "methods.nerfus.render", "render_slice"),
}


@dataclass
class RunSpec:
    benchmark: str
    stage: str
    dataset: str
    sweep: str
    method: str
    final_step: str
    final_num_gaussians: str
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("outputs/production/comparisons/us_recon_four_dataset_p100k_v1"),
    )
    parser.add_argument("--source-tsv", type=Path, action="append", default=[])
    parser.add_argument("--output-per-run", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-iteration", type=int, default=30000)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--timed-frames", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--run-index", type=int, default=-1)
    parser.add_argument("--isolate-subprocesses", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_runs(path: Path) -> List[RunSpec]:
    runs: List[RunSpec] = []
    if not path.exists():
        return runs
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            output_dir = row.get("output_dir", "")
            if not output_dir:
                continue
            runs.append(
                RunSpec(
                    benchmark=row.get("benchmark", ""),
                    stage=row.get("stage", ""),
                    dataset=row.get("dataset", ""),
                    sweep=row.get("sweep", ""),
                    method=row.get("method", ""),
                    final_step=row.get("final_step", ""),
                    final_num_gaussians=row.get("final_num_gaussians", ""),
                    output_dir=Path(output_dir),
                )
            )
    return runs


def split_filter(value: str) -> Optional[set[str]]:
    if not value:
        return None
    return {part for part in value.replace(":", ",").split(",") if part}


def load_config(run_dir: Path):
    for rel in ("resolved_config.yaml", ".hydra/config.yaml"):
        path = run_dir / rel
        if path.exists():
            return OmegaConf.load(path)
    raise FileNotFoundError(f"No resolved_config.yaml or .hydra/config.yaml in {run_dir}")


def select_evenly(items: Sequence, count: int) -> List:
    if count <= 0 or len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    selected = []
    for i in range(count):
        idx = int(round(i * (len(items) - 1) / (count - 1)))
        selected.append(items[idx])
    return selected


def collect_test_slice_metas(scene, count: int) -> List:
    all_metas = []
    while True:
        datum = scene.next_test_datum()
        if datum is None:
            break
        all_metas.append(datum["slice_meta"])
    return select_evenly(all_metas, count)


def import_attr(module_name: str, attr_name: str):
    return getattr(importlib.import_module(module_name), attr_name)


def load_gs_renderer(run: RunSpec, cfg, device: torch.device, iteration: int, dataset_meta):
    if run.method not in GS_METHODS:
        return None, None, "", "unsupported_method"
    container_mod, container_name, render_mod, render_name = GS_METHODS[run.method]
    container_cls = import_attr(container_mod, container_name)
    render_fn = import_attr(render_mod, render_name)
    checkpoint = run.output_dir / "checkpoints" / f"iteration_{iteration}_container.pth"
    if not checkpoint.exists():
        return None, None, str(checkpoint), "missing_checkpoint"
    container = container_cls(cfg, device)
    container.load(str(checkpoint))
    container.eval()

    if run.method == "ours":
        renderer = container.ensure_ours_renderer(dataset_meta, cfg.model)
        ray_attr = (
            container.attenuations
            if str(cfg.model.slice_render.ray_attenuation_source) == "attenuation"
            else None
        )

        def forward(slice_meta):
            pose_w = slice_meta.pose.to(device=container.means.device, dtype=container.means.dtype)
            pred = renderer(
                container.means,
                container.scales,
                container.quats,
                container.opacities,
                container.intensities,
                pose_w,
                ray_attr,
            )
            return pred, None, {}

        return forward, int(container.num_points), str(checkpoint), "ready"

    def forward(slice_meta):
        return render_fn(cfg.model, dataset_meta, slice_meta, container)

    return forward, int(container.num_points), str(checkpoint), "ready"


def load_implicit_renderer(run: RunSpec, cfg, device: torch.device, iteration: int):
    if run.method not in IMPLICIT_METHODS:
        return None, None, "", "unsupported_method"
    field_mod, field_name, render_mod, render_name = IMPLICIT_METHODS[run.method]
    field_cls = import_attr(field_mod, field_name)
    render_fn = import_attr(render_mod, render_name)
    checkpoint = run.output_dir / "checkpoints" / f"iteration_{iteration}_field.pth"
    if not checkpoint.exists():
        return None, None, str(checkpoint), "missing_checkpoint"
    field = field_cls(cfg.model.field).to(device)
    field.load_state_dict(torch.load(checkpoint, map_location=device))
    field.eval()

    def forward(slice_meta):
        return render_fn(
            cfg.model,
            slice_meta,
            field,
            patch=None,
            chunk_size=int(cfg.model.ray.eval_chunk_size),
        )

    return forward, "", str(checkpoint), "ready"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_forward(
    forward: Callable,
    frames: Sequence,
    device: torch.device,
    warmup_frames: int,
    repeats: int,
) -> Dict[str, float]:
    if not frames:
        raise ValueError("No test frames available for FPS benchmark.")

    output_shape = ""
    output_numel = 0
    with torch.no_grad():
        for slice_meta in select_evenly(frames, min(warmup_frames, len(frames))):
            out = forward(slice_meta)
            if isinstance(out, tuple) and out:
                first = out[0]
                if not output_shape and hasattr(first, "shape"):
                    output_shape = "x".join(str(int(v)) for v in first.shape)
                    output_numel = int(first.numel())
        synchronize(device)

        timings = []
        rendered = 0
        for _repeat in range(max(1, int(repeats))):
            synchronize(device)
            start = time.perf_counter()
            for slice_meta in frames:
                out = forward(slice_meta)
                if isinstance(out, tuple) and out:
                    first = out[0]
                    if not output_shape and hasattr(first, "shape"):
                        output_shape = "x".join(str(int(v)) for v in first.shape)
                        output_numel = int(first.numel())
            synchronize(device)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            rendered += len(frames)

    total = sum(timings)
    fps = rendered / total if total > 0 else math.nan
    mean_ms = (total / rendered * 1000.0) if rendered else math.nan
    return {
        "fps": fps,
        "mean_ms_per_frame": mean_ms,
        "timed_passes": float(len(timings)),
        "timed_rendered_frames": float(rendered),
        "timed_total_sec": total,
        "output_shape": output_shape,
        "output_numel": float(output_numel),
    }


def write_tsv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[float]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        try:
            fps = float(row["render_fps"])
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault((row["dataset"], row["method"]), []).append(fps)

    dataset_rows = []
    method_values: Dict[str, List[float]] = {}
    for (dataset, method), values in sorted(groups.items()):
        method_values.setdefault(str(method), []).append(statistics.mean(values))
        dataset_rows.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "method": method,
                "num_runs": len(values),
                "render_fps_mean": statistics.mean(values),
                "render_fps_median": statistics.median(values),
                "render_fps_min": min(values),
                "render_fps_max": max(values),
            }
        )

    paper_rows = []
    for method, values in sorted(method_values.items()):
        paper_rows.append(
            {
                "scope": "paper_tables",
                "dataset": "all_available_dataset_means",
                "method": method,
                "num_dataset_means": len(values),
                "render_fps_mean": statistics.mean(values),
                "render_fps_median": statistics.median(values),
                "render_fps_min": min(values),
                "render_fps_max": max(values),
            }
        )
    return dataset_rows + paper_rows


def run_isolated_children(args: argparse.Namespace, runs: Sequence[RunSpec]) -> None:
    output_per_run = args.output_per_run or args.benchmark_root.resolve() / "tables" / "render_fps_per_run.tsv"
    output_summary = args.output_summary or args.benchmark_root.resolve() / "tables" / "render_fps_method_summary.tsv"
    child_dir = output_per_run.parent / ".render_fps_child_rows"
    child_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for idx, run in enumerate(runs):
        child_out = child_dir / f"row_{idx:04d}_{run.dataset}_{run.sweep}_{run.method}.tsv"
        child_summary = child_dir / f"summary_{idx:04d}.tsv"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--benchmark-root",
            str(args.benchmark_root),
            "--output-per-run",
            str(child_out),
            "--output-summary",
            str(child_summary),
            "--device",
            str(args.device),
            "--checkpoint-iteration",
            str(args.checkpoint_iteration),
            "--warmup-frames",
            str(args.warmup_frames),
            "--timed-frames",
            str(args.timed_frames),
            "--repeats",
            str(args.repeats),
            "--run-index",
            str(idx),
        ]
        for source in args.source_tsv:
            cmd.extend(["--source-tsv", str(source)])
        if args.methods:
            cmd.extend(["--methods", str(args.methods)])
        if args.datasets:
            cmd.extend(["--datasets", str(args.datasets)])
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[ISOLATE] {idx + 1}/{len(runs)} {run.dataset}/{run.sweep}/{run.method}", flush=True)
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
        if child_out.exists():
            with child_out.open("r", encoding="utf-8", newline="") as f:
                child_rows = list(csv.DictReader(f, delimiter="\t"))
            rows.extend(child_rows)
        else:
            rows.append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "benchmark": run.benchmark,
                    "source_stage": run.stage,
                    "dataset": run.dataset,
                    "sweep": run.sweep,
                    "method": run.method,
                    "status": "child_process_failed",
                    "error": f"exit_code={proc.returncode}",
                    "output_dir": str(run.output_dir),
                }
            )
        write_tsv(output_per_run, rows)
        write_tsv(output_summary, summarize(rows))

    print(f"[DONE] Wrote {output_per_run}")
    print(f"[DONE] Wrote {output_summary}")


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    source_tsvs = args.source_tsv or [benchmark_root / "tables" / "gs_paper_3d_checkpoint_per_run.tsv"]
    output_per_run = args.output_per_run or benchmark_root / "tables" / "render_fps_per_run.tsv"
    output_summary = args.output_summary or benchmark_root / "tables" / "render_fps_method_summary.tsv"
    method_filter = split_filter(args.methods)
    dataset_filter = split_filter(args.datasets)

    runs: List[RunSpec] = []
    for path in source_tsvs:
        runs.extend(read_runs(path if path.is_absolute() else Path(path)))
    if method_filter is not None:
        runs = [run for run in runs if run.method in method_filter]
    if dataset_filter is not None:
        runs = [run for run in runs if run.dataset in dataset_filter]
    if args.run_index >= 0:
        runs = [runs[args.run_index]] if args.run_index < len(runs) else []
    if args.max_runs > 0:
        runs = runs[: args.max_runs]

    if args.isolate_subprocesses and args.run_index < 0:
        run_isolated_children(args, runs)
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows: List[Dict[str, object]] = []
    for idx, run in enumerate(runs, start=1):
        row: Dict[str, object] = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark": run.benchmark,
            "source_stage": run.stage,
            "dataset": run.dataset,
            "sweep": run.sweep,
            "method": run.method,
            "final_step": run.final_step,
            "final_num_gaussians": run.final_num_gaussians,
            "checkpoint_iteration": args.checkpoint_iteration,
            "warmup_frames": args.warmup_frames,
            "requested_timed_frames": args.timed_frames,
            "repeats": args.repeats,
            "device": str(device),
            "output_dir": str(run.output_dir),
        }
        print(f"[{idx}/{len(runs)}] {run.dataset}/{run.sweep}/{run.method}", flush=True)
        try:
            cfg = load_config(run.output_dir)
            scene = SceneFactory.build_scene(cfg, device)
            if run.method in GS_METHODS:
                forward, num_model_items, checkpoint, status = load_gs_renderer(
                    run, cfg, device, args.checkpoint_iteration, scene.dataset_meta
                )
            else:
                forward, num_model_items, checkpoint, status = load_implicit_renderer(
                    run, cfg, device, args.checkpoint_iteration
                )
            row["checkpoint"] = checkpoint
            row["num_model_items"] = num_model_items
            if status != "ready":
                row["status"] = status
                rows.append(row)
                continue
            frames = collect_test_slice_metas(scene, args.timed_frames)
            row["timed_frames"] = len(frames)
            if args.dry_run:
                row["status"] = "dry_run_ready"
                rows.append(row)
                continue
            stats = benchmark_forward(forward, frames, device, args.warmup_frames, args.repeats)
            row.update(
                {
                    "status": "success",
                    "render_fps": stats["fps"],
                    "mean_ms_per_frame": stats["mean_ms_per_frame"],
                    "output_shape": stats["output_shape"],
                    "output_numel": int(stats["output_numel"]),
                    "timed_passes": int(stats["timed_passes"]),
                    "timed_rendered_frames": int(stats["timed_rendered_frames"]),
                    "timed_total_sec": stats["timed_total_sec"],
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        rows.append(row)
        write_tsv(output_per_run, rows)
        write_tsv(output_summary, summarize(rows))

    write_tsv(output_per_run, rows)
    write_tsv(output_summary, summarize(rows))
    print(f"[DONE] Wrote {output_per_run}")
    print(f"[DONE] Wrote {output_summary}")


if __name__ == "__main__":
    main()
