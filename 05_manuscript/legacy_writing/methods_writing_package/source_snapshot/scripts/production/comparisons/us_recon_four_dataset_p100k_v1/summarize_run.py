#!/usr/bin/env python3
import csv
import fcntl
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_series(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    series = {}
    if not path.exists():
        return series
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                key = row["key"]
                step = int(row["step"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            series.setdefault(key, []).append((step, value))
    return series


def value_at(series: Dict[str, List[Tuple[int, float]]], key: str, step: Optional[int]):
    if step is None:
        return ""
    for item_step, value in reversed(series.get(key, [])):
        if item_step == step:
            return value
    return ""


def last_value(series: Dict[str, List[Tuple[int, float]]], key: str):
    values = series.get(key)
    if not values:
        return "", ""
    step, value = values[-1]
    return step, value


def numeric_values(series: Dict[str, List[Tuple[int, float]]], key: str) -> List[float]:
    return [float(value) for _, value in series.get(key, [])]


def load_key_value_tsv(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = row.get("key", "")
            if not key:
                continue
            data[key] = row.get("value", "")
    return data


def file_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def latest_mtime(path: Path) -> str:
    if not path.exists():
        return ""
    latest = path.stat().st_mtime
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                latest = max(latest, child.stat().st_mtime)
            except OSError:
                pass
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest))


def append_tsv_with_union(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            return

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            old_fields = list(reader.fieldnames or [])
            old_rows = list(reader)

        fieldnames = list(old_fields)
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for old_row in old_rows:
                writer.writerow(old_row)
            writer.writerow(row)


def write_simple_yaml(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, value in data.items():
            text = "" if value is None else str(value)
            text = text.replace("\n", "\\n")
            f.write(f"{key}: {text}\n")


def main() -> None:
    output_dir = Path(os.environ["OUTPUT_DIR"])
    summary_path = Path(os.environ["SUMMARY_PATH"])
    scalars = output_dir / "logs" / "scalars.tsv"
    efficiency_path = Path(os.environ.get("EFFICIENCY_PATH", str(output_dir / "logs" / "efficiency.tsv")))
    efficiency_summary_path = output_dir / "logs" / "efficiency_summary.yaml"
    efficiency_summary_tsv = Path(os.environ.get("EFFICIENCY_SUMMARY_PATH", str(summary_path.parent / "efficiency_per_run.tsv")))
    per_frame_metrics = output_dir / "eval" / "per_frame_metrics_latest.tsv"
    series = load_series(scalars)
    efficiency = load_key_value_tsv(efficiency_path)

    psnr_values = series.get("test/psnr", [])
    if psnr_values:
        best_step_i, best_psnr = max(psnr_values, key=lambda item: item[1])
    else:
        best_step_i, best_psnr = None, ""
    best_per_frame_metrics = (
        output_dir / "eval" / f"per_frame_metrics_iteration_{int(best_step_i):06d}.tsv"
        if best_step_i is not None
        else None
    )

    final_step, final_psnr = last_value(series, "test/psnr")
    _, final_l1 = last_value(series, "test/l1")
    _, final_ssim = last_value(series, "test/ssim")
    _, final_lpips = last_value(series, "test/lpips")
    _, final_dists = last_value(series, "test/dists")
    _, final_gmsd = last_value(series, "test/gmsd")
    _, final_num_gaussians = last_value(series, "train/num_gaussians")
    num_gaussian_values = numeric_values(series, "train/num_gaussians")
    dc_event_steps = {step for step, _ in series.get("train/dc/num_gaussians_before", [])}
    checkpoint_dir = output_dir / "checkpoints"
    vis_dir = output_dir / "vis"
    eval_dir = output_dir / "eval"
    train_wall_time_sec = efficiency.get("train_wall_time_sec", "")
    train_iterations = ""
    if final_step != "":
        train_iterations = final_step
    elif best_step_i is not None:
        train_iterations = best_step_i
    train_iter_per_sec = ""
    try:
        if train_iterations != "" and train_wall_time_sec not in ("", "0"):
            train_iter_per_sec = float(train_iterations) / float(train_wall_time_sec)
    except (TypeError, ValueError, ZeroDivisionError):
        train_iter_per_sec = ""

    efficiency_row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "benchmark": os.environ.get("BENCHMARK", ""),
        "stage": os.environ.get("STAGE", ""),
        "dataset": os.environ.get("DATASET", ""),
        "sweep": os.environ.get("SWEEP", ""),
        "method": os.environ.get("METHOD", ""),
        "exp_name": os.environ.get("EXP_NAME", ""),
        "train_start_time": efficiency.get("train_start_time", ""),
        "train_end_time": efficiency.get("train_end_time", ""),
        "train_wall_time_sec": train_wall_time_sec,
        "train_iterations": train_iterations,
        "train_iter_per_sec": train_iter_per_sec,
        "train_start_gpu_mem_mb": efficiency.get("train_start_gpu_mem_mb", ""),
        "train_end_gpu_mem_mb": efficiency.get("train_end_gpu_mem_mb", ""),
        "train_peak_gpu_mem_mb": efficiency.get("train_peak_gpu_mem_mb", ""),
        "train_exit_code": efficiency.get("train_exit_code", ""),
        "best_num_gaussians": value_at(series, "train/num_gaussians", best_step_i),
        "final_num_gaussians": final_num_gaussians,
        "min_num_gaussians": min(num_gaussian_values) if num_gaussian_values else "",
        "max_num_gaussians": max(num_gaussian_values) if num_gaussian_values else "",
        "dc_event_count": len(dc_event_steps),
        "checkpoint_bytes": file_size_bytes(checkpoint_dir),
        "vis_bytes": file_size_bytes(vis_dir),
        "eval_bytes": file_size_bytes(eval_dir),
        "output_bytes": file_size_bytes(output_dir),
        "latest_output_mtime": latest_mtime(output_dir),
        "efficiency_tsv": str(efficiency_path) if efficiency_path.exists() else "",
        "output_dir": str(output_dir),
    }

    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "benchmark": os.environ.get("BENCHMARK", ""),
        "stage": os.environ.get("STAGE", ""),
        "dataset": os.environ.get("DATASET", ""),
        "sweep": os.environ.get("SWEEP", ""),
        "method": os.environ.get("METHOD", ""),
        "exp_name": os.environ.get("EXP_NAME", ""),
        "selection_metric": "final checkpoint",
        "best_step": "" if best_step_i is None else best_step_i,
        "best_psnr": best_psnr,
        "best_l1": value_at(series, "test/l1", best_step_i),
        "best_ssim": value_at(series, "test/ssim", best_step_i),
        "best_lpips": value_at(series, "test/lpips", best_step_i),
        "best_dists": value_at(series, "test/dists", best_step_i),
        "best_gmsd": value_at(series, "test/gmsd", best_step_i),
        "best_num_gaussians": value_at(series, "train/num_gaussians", best_step_i),
        "final_step": final_step,
        "final_psnr": final_psnr,
        "final_l1": final_l1,
        "final_ssim": final_ssim,
        "final_lpips": final_lpips,
        "final_dists": final_dists,
        "final_gmsd": final_gmsd,
        "final_num_gaussians": final_num_gaussians,
        "min_num_gaussians": min(num_gaussian_values) if num_gaussian_values else "",
        "max_num_gaussians": max(num_gaussian_values) if num_gaussian_values else "",
        "dc_event_count": len(dc_event_steps),
        "train_wall_time_sec": train_wall_time_sec,
        "train_iter_per_sec": train_iter_per_sec,
        "train_peak_gpu_mem_mb": efficiency.get("train_peak_gpu_mem_mb", ""),
        "checkpoint_bytes": efficiency_row["checkpoint_bytes"],
        "output_bytes": efficiency_row["output_bytes"],
        "status": "success" if scalars.exists() and psnr_values else "missing_metrics",
        "output_dir": str(output_dir),
        "scalars": str(scalars),
        "efficiency": str(efficiency_path) if efficiency_path.exists() else "",
        "efficiency_summary": str(efficiency_summary_path),
        "best_per_frame_metrics": str(best_per_frame_metrics) if best_per_frame_metrics and best_per_frame_metrics.exists() else "",
        "per_frame_metrics": str(per_frame_metrics) if per_frame_metrics.exists() else "",
    }

    write_simple_yaml(efficiency_summary_path, efficiency_row)
    append_tsv_with_union(summary_path, row)
    append_tsv_with_union(efficiency_summary_tsv, efficiency_row)

    print(f"[INFO] summary={summary_path}")
    print(f"[INFO] efficiency_summary={efficiency_summary_tsv}")
    print(f"[INFO] status={row['status']} final_step={row['final_step']} final_psnr={row['final_psnr']}")


if __name__ == "__main__":
    main()
