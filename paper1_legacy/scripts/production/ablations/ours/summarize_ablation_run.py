#!/usr/bin/env python3
import csv
import fcntl
import os
from pathlib import Path


def read_series(path):
    series = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                step = int(row["step"])
                value = float(row["value"])
                key = row["key"]
            except (KeyError, TypeError, ValueError):
                continue
            series.setdefault(key, []).append((step, value))
    return series


def last(series, key):
    values = series.get(key)
    if not values:
        return "", ""
    step, value = values[-1]
    return step, value


def value_at(series, key, target_step):
    if target_step == "":
        return ""
    for step, value in series.get(key, []):
        if step == target_step:
            return value
    return ""


def main() -> None:
    run_dir = Path(os.environ["RUN_DIR"])
    summary_path = Path(os.environ["SUMMARY_PATH"])
    scalars = run_dir / "logs" / "scalars.tsv"
    if not scalars.exists():
        raise SystemExit(f"missing scalars.tsv: {scalars}")

    series = read_series(scalars)
    final_step, final_psnr = last(series, "test/psnr")
    metrics = {
        "psnr": final_psnr,
        "l1": value_at(series, "test/l1", final_step),
        "ssim": value_at(series, "test/ssim", final_step),
        "lpips": value_at(series, "test/lpips", final_step),
        "dists": value_at(series, "test/dists", final_step),
        "gmsd": value_at(series, "test/gmsd", final_step),
        "num_gaussians": value_at(series, "train/num_gaussians", final_step),
    }
    start_epoch = int(os.environ.get("START_EPOCH", "0") or "0")
    end_epoch = int(os.environ.get("END_EPOCH", "0") or "0")
    wall_seconds = max(0, end_epoch - start_epoch) if start_epoch and end_epoch else ""

    row = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "table": os.environ["TABLE_LABEL"],
        "dataset": os.environ["DATASET_LABEL"],
        "sweep": os.environ["SWEEP_LABEL"],
        "variant": os.environ["VARIANT_LABEL"],
        "exp_name": os.environ["EXP_NAME"],
        "wall_seconds": wall_seconds,
        "final_step": final_step,
        "final_psnr": metrics["psnr"],
        "final_l1": metrics["l1"],
        "final_ssim": metrics["ssim"],
        "final_lpips": metrics["lpips"],
        "final_dists": metrics["dists"],
        "final_gmsd": metrics["gmsd"],
        "final_num_gaussians": metrics["num_gaussians"],
        "run_dir": str(run_dir),
        "scalars": str(scalars),
    }
    fields = list(row)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        write_header = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
