#!/usr/bin/env python3
"""Regenerate synthetic_liver benchmark compact and markdown tables."""

from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
from pathlib import Path


ROOT = Path("outputs/production/comparisons/us_synthetic_recon_benchmark_v1")
PER_RUN = ROOT / "tables" / "per_run_metrics.tsv"
COMPACT = ROOT / "tables" / "metrics_compact.csv"
BY_SWEEP = ROOT / "tables" / "metrics_by_sweep.md"

SWEEPS = ["l1", "l2", "l3", "r1", "r2", "r3"]
METHODS = [
    ("ultranerf", "ultra-nerf"),
    ("nerfus_no_prior", "nerfus w/o prior"),
    ("ultragray", "ultragray"),
    ("ultragauss", "ultragauss"),
    ("gaussianpile", "gaussianpile"),
    ("ultrags_baseline", "ultrags baseline"),
    ("slice_adapted_3dgs", "slice-adapted 3DGS"),
    ("ours", "ours"),
]
METHOD_LABEL = dict(METHODS)
METHOD_ALIASES = {
    "nerfus w/o prior": "nerfus_no_prior",
}


def _float_text(value: str, digits: int = 4) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def _latest_rows() -> OrderedDict[tuple[str, str], dict[str, str]]:
    latest: OrderedDict[tuple[str, str], dict[str, str]] = OrderedDict()
    with PER_RUN.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["method"] = METHOD_ALIASES.get(row["method"], row["method"])
            key = (row["sweep"], row["method"])
            latest[key] = row
    return latest


def write_compact(rows: OrderedDict[tuple[str, str], dict[str, str]]) -> None:
    COMPACT.parent.mkdir(parents=True, exist_ok=True)
    with COMPACT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep", "method", "best_step", "psnr", "ssim", "lpips", "l1", "dists", "gmsd", "num_gaussians"])
        for sweep in SWEEPS:
            for method, label in METHODS:
                row = rows.get((sweep, method))
                if row is None:
                    continue
                writer.writerow([
                    sweep,
                    label,
                    row["best_step"],
                    _float_text(row["best_psnr"]),
                    _float_text(row["best_ssim"]),
                    _float_text(row["best_lpips"]),
                    _float_text(row["best_l1"]),
                    _float_text(row["best_dists"]),
                    _float_text(row["best_gmsd"]),
                    row["num_gaussians"],
                ])


def _mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def write_markdown(rows: OrderedDict[tuple[str, str], dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# synthetic_liver 8-method comparison best metrics")
    lines.append("")
    lines.append("Selection metric: max held-out test PSNR. All rows use 30k iterations, seed 42, holdout_interval with trim_each=10 and 40 held-out samples. Gaussian methods use 1M production initialization where applicable.")
    lines.append("")
    lines.append("NeRF-US row is `nerfus w/o prior`: diffusion prior disabled, trained with the UltraNeRF-matched no-prior protocol. DISTS/GMSD are blank for runs whose configs did not enable those metrics.")
    lines.append("")

    metric_sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sweep in SWEEPS:
        lines.append(f"## {sweep}")
        lines.append("")
        lines.append("| method | best_step | PSNR | SSIM | LPIPS | L1 | DISTS | GMSD | Gaussians |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        winner_label = ""
        winner_psnr = float("-inf")
        for method, label in METHODS:
            row = rows.get((sweep, method))
            if row is None:
                continue
            psnr = row["best_psnr"]
            if psnr:
                psnr_value = float(psnr)
                if psnr_value > winner_psnr:
                    winner_psnr = psnr_value
                    winner_label = label
            lines.append(
                "| {method} | {step} | {psnr} | {ssim} | {lpips} | {l1} | {dists} | {gmsd} | {gaussians} |".format(
                    method=label,
                    step=row["best_step"],
                    psnr=_float_text(row["best_psnr"]),
                    ssim=_float_text(row["best_ssim"]),
                    lpips=_float_text(row["best_lpips"]),
                    l1=_float_text(row["best_l1"]),
                    dists=_float_text(row["best_dists"]),
                    gmsd=_float_text(row["best_gmsd"]),
                    gaussians=row["num_gaussians"],
                )
            )
            for key in ["best_psnr", "best_ssim", "best_lpips", "best_l1", "best_dists", "best_gmsd"]:
                if row[key] != "":
                    metric_sums[label][key].append(float(row[key]))
        lines.append("")
        lines.append(f"PSNR winner: {winner_label} ({winner_psnr:.4f} dB).")
        lines.append("")

    lines.append("## Mean by method")
    lines.append("")
    lines.append("| method | PSNR | SSIM | LPIPS | L1 | DISTS | GMSD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _method, label in METHODS:
        metrics = metric_sums[label]
        lines.append(
            "| {method} | {psnr} | {ssim} | {lpips} | {l1} | {dists} | {gmsd} |".format(
                method=label,
                psnr=_mean(metrics["best_psnr"]),
                ssim=_mean(metrics["best_ssim"]),
                lpips=_mean(metrics["best_lpips"]),
                l1=_mean(metrics["best_l1"]),
                dists=_mean(metrics["best_dists"]),
                gmsd=_mean(metrics["best_gmsd"]),
            )
        )
    lines.append("")
    BY_SWEEP.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = _latest_rows()
    write_compact(rows)
    write_markdown(rows)


if __name__ == "__main__":
    main()
