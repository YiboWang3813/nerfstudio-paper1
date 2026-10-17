#!/usr/bin/env python3
"""Generate staged NVS aggregate tables from the per-run summary TSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


DATASETS = [
    ("synthetic_liver", 6),
    ("spine_phantom", 9),
    ("tus_rec_2024", 48),
    ("tdsc_abus_2023", 30),
]

METHODS = [
    "ours",
    "ultragauss",
    "ultragray",
    "gaussianpile",
    "ultrags_baseline",
    "slice_adapted_3dgs",
    "ultranerf",
    "nerfus_no_prior",
]

EXPECTED_CASES = sum(count for _dataset, count in DATASETS)
EXPECTED_ROWS = EXPECTED_CASES * len(METHODS)

METRICS = [
    ("final_psnr", "PSNR", "higher"),
    ("final_l1", "L1", "lower"),
    ("final_ssim", "SSIM", "higher"),
    ("final_lpips", "LPIPS", "lower"),
    ("final_dists", "DISTS", "lower"),
    ("final_gmsd", "GMSD", "lower"),
]

AGG_METRICS = [
    ("psnr", "final_psnr", "PSNR", "higher"),
    ("l1", "final_l1", "L1", "lower"),
    ("ssim", "final_ssim", "SSIM", "higher"),
    ("lpips", "final_lpips", "LPIPS", "lower"),
    ("dists", "final_dists", "DISTS", "lower"),
    ("gmsd", "final_gmsd", "GMSD", "lower"),
]

METHOD_CONFIGS = {
    "ours": {
        "alias": "production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ours",
        "doc": "docs/methods/ours/responsibility_density_control/",
        "init": "1M",
        "max": "1M init, DC-pruned final",
        "iters": "30k",
        "mode": "local-global renderer, file/physical pose",
        "dc": "responsibility DC, timing 1000/25000/2000",
        "status": "production",
        "notes": "Renderer config is fixed; only responsibility DC hyperparameters are method-native.",
    },
    "ultragauss": {
        "alias": "production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ultragauss",
        "doc": "docs/methods/baselines/ultragauss/",
        "init": "1M",
        "max": "2M",
        "iters": "30k",
        "mode": "Gaussian, file/physical pose",
        "dc": "UltraGauss thresholds, timing 1000/25000/2000",
        "status": "production-locked",
        "notes": "GS-strict promotion alias overrides the older 10k reproduction alias at comparison-level resources.",
    },
    "ultragray": {
        "alias": "production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ultragray",
        "doc": "docs/methods/baselines/ultragray/",
        "init": "1M",
        "max": "2M",
        "iters": "30k",
        "mode": "Gaussian grayscale, file/physical pose",
        "dc": "UltraGray thresholds, timing 1000/25000/2000",
        "status": "production-locked",
        "notes": "Uses fast ray settings and n_max 2M under GS-strict timing.",
    },
    "gaussianpile": {
        "alias": "production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/gaussianpile",
        "doc": "docs/methods/baselines/gaussianpile/",
        "init": "1M",
        "max": "2M",
        "iters": "30k",
        "mode": "GaussianPile, file/physical pose",
        "dc": "focus-aware GaussianPile thresholds, timing 1000/25000/2000",
        "status": "production-locked",
        "notes": "Requires collapse diagnostics because historical production timing was denser.",
    },
    "ultrags_baseline": {
        "alias": "production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/ultrags_baseline",
        "doc": "docs/methods/baselines/ultrags_baseline/reproduction_complete_report.md",
        "init": "1M",
        "max": "1M",
        "iters": "30k",
        "mode": "physical slice camera",
        "dc": "baseline thresholds, timing start=1000/end=25000/interval=2000",
        "status": "production-locked",
        "notes": "Uses physical slice camera; does not use 100k released-camera or 300k lightweight settings.",
    },
    "slice_adapted_3dgs": {
        "alias": "production/comparisons/us_recon_four_dataset_p100k_v1/gs_strict/slice_adapted_3dgs",
        "doc": "docs/methods/baselines/slice_adapted_3dgs/",
        "init": "1M",
        "max": "2M",
        "iters": "30k",
        "mode": "slice-adapted 3DGS",
        "dc": "slice-adapted thresholds, timing 1000/25000/2000",
        "status": "production-locked",
        "notes": "Physical slice camera and finite-thickness additive rendering.",
    },
    "ultranerf": {
        "alias": "preproduction/baseline_reproduction/ultranerf/fullres_30k_mc_eval",
        "doc": "docs/methods/baselines/ultranerf/",
        "init": "N/A",
        "max": "N/A",
        "iters": "30k",
        "mode": "implicit UltraNeRF field",
        "dc": "N/A",
        "status": "deferred",
        "notes": "Implicit row is excluded from GS-first execution because runtime is much slower.",
    },
    "nerfus_no_prior": {
        "alias": "preproduction/baseline_reproduction/nerfus/fullres_30k_ultranerf_matched",
        "doc": "docs/methods/baselines/nerfus/",
        "init": "N/A",
        "max": "N/A",
        "iters": "30k",
        "mode": "implicit NeRF-US without prior",
        "dc": "N/A",
        "status": "deferred",
        "notes": "Implicit row is added only after GS protocol stabilizes.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/production/comparisons/us_recon_four_dataset_p100k_v1"),
    )
    parser.add_argument("--per-run", type=Path, default=None)
    parser.add_argument("--doc", type=Path, default=Path("docs/production/multi_comparisons.md"))
    parser.add_argument("--update-doc", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing {key} in row: {row}")
    return float(value)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def aggregate(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "success":
            by_group[(row["dataset"], row["method"])].append(row)

    dataset_rows: list[dict[str, object]] = []
    for dataset, expected_cases in DATASETS:
        for method in METHODS:
            group = by_group[(dataset, method)]
            if not group:
                continue
            row = {
                "dataset": dataset,
                "method": method,
                "cases": len(group),
                "expected_cases": expected_cases,
            }
            for out_key, source_key, _label, _direction in AGG_METRICS:
                row[out_key] = mean(as_float(r, source_key) for r in group)
            dataset_rows.append(row)

    mean_rows = []
    for method in METHODS:
        group = [r for r in rows if r.get("status") == "success" and r["method"] == method]
        if not group:
            continue
        row = {
            "dataset": "Mean",
            "method": method,
            "cases": len(group),
            "expected_cases": EXPECTED_CASES,
        }
        for out_key, source_key, _label, _direction in AGG_METRICS:
            row[out_key] = mean(as_float(r, source_key) for r in group)
        mean_rows.append(row)
    return dataset_rows, mean_rows


def ranks_for(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, int]]:
    ranks: dict[tuple[str, str], dict[str, int]] = {}
    for dataset in [d for d, _ in DATASETS] + ["Mean"]:
        subset = [r for r in rows if r["dataset"] == dataset]
        for key, _source_key, _label, direction in AGG_METRICS:
            ordered = sorted(subset, key=lambda r: float(r[key]), reverse=(direction == "higher"))
            for rank, row in enumerate(ordered, 1):
                ranks.setdefault((str(row["dataset"]), str(row["method"])), {})[key] = rank
    return ranks


def rank_note(row: dict[str, object], ranks: dict[tuple[str, str], dict[str, int]]) -> str:
    r = ranks[(str(row["dataset"]), str(row["method"]))]
    return ", ".join(f"{label} #{r[key]}" for key, _source_key, label, _direction in AGG_METRICS)


def markdown_table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"


def decorated(value: float, rank: int, lower_is_better: bool = False) -> str:
    text = fmt(value)
    if rank == 1:
        return f"**{text}**"
    if rank == 2:
        return f"<u>{text}</u>"
    return text


def write_by_dataset_csv(path: Path, rows: list[dict[str, object]], ranks: dict[tuple[str, str], dict[str, int]]) -> None:
    fields = ["dataset", "method", "psnr", "l1", "ssim", "lpips", "dists", "gmsd", "cases", "rank_note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    **{key: fmt(float(row[key]), 6) for key, _source_key, _label, _direction in AGG_METRICS},
                    "cases": row["cases"],
                    "rank_note": rank_note(row, ranks),
                }
            )


def write_paper_table(path: Path, rows: list[dict[str, object]], ranks: dict[tuple[str, str], dict[str, int]]) -> str:
    lines = [
        ["Dataset", "Method", "PSNR ↑", "L1 ↓", "SSIM ↑", "LPIPS ↓", "DISTS ↓", "GMSD ↓", "Cases", "Notes"],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
    ]
    for row in rows:
        r = ranks[(str(row["dataset"]), str(row["method"]))]
        lines.append(
            [
                str(row["dataset"]),
                str(row["method"]),
                *[
                    decorated(float(row[key]), r[key], lower_is_better=(direction == "lower"))
                    for key, _source_key, _label, direction in AGG_METRICS
                ],
                str(row["cases"]),
                rank_note(row, ranks),
            ]
        )
    table = markdown_table(lines)
    path.write_text(table, encoding="utf-8")
    return table


def write_rank_summary(path: Path, rows: list[dict[str, object]], ranks: dict[tuple[str, str], dict[str, int]]) -> str:
    mean_rows = [r for r in rows if r["dataset"] == "Mean"]
    lines = [
        "# Main NVS Rank Summary",
        "",
        f"Mean rows rank each method across all completed NVS cases. PSNR/SSIM are higher-is-better; L1/LPIPS/DISTS/GMSD are lower-is-better.",
        "",
        "| Method | PSNR rank | L1 rank | SSIM rank | LPIPS rank | DISTS rank | GMSD rank | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(mean_rows, key=lambda r: ranks[("Mean", str(r["method"]))]["psnr"]):
        r = ranks[("Mean", str(row["method"]))]
        lines.append(
            f"| {row['method']} | "
            + " | ".join(str(r[key]) for key, _source_key, _label, _direction in AGG_METRICS)
            + " | "
            + " | ".join(fmt(float(row[key])) for key, _source_key, _label, _direction in AGG_METRICS)
            + " |"
        )

    lines.extend(
        [
            "",
            "Dataset PSNR winners:",
            "",
        ]
    )
    for dataset, _ in DATASETS:
        winner = min([r for r in rows if r["dataset"] == dataset], key=lambda row: ranks[(dataset, str(row["method"]))]["psnr"])
        lines.append(f"- {dataset}: {winner['method']} ({fmt(float(winner['psnr']))} PSNR).")
    lines.append("")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def write_method_config(path: Path) -> str:
    lines = [
        "| Method | Source alias | Method doc source | Init budget | Max budget | Iterations | Camera/render mode | DC/refinement policy | Checkpoint rule | Production status | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for method in METHODS:
        c = METHOD_CONFIGS[method]
        lines.append(
            f"| {method} | `{c['alias']}` | `{c['doc']}` | {c['init']} | {c['max']} | {c['iters']} | "
            f"{c['mode']} | {c['dc']} | final 30k checkpoint | {c['status']} | {c['notes']} |"
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def write_budget_table(path: Path, rows: list[dict[str, object]], ranks: dict[tuple[str, str], dict[str, int]]) -> str:
    table = write_paper_table(path, rows, ranks)
    caption = (
        "\nCaption: staged aggregate comparison generated from the current per-run summary. "
        "Interpret this table only after the protocol section and method-configuration "
        "launch gate have been reviewed for the current production attempt.\n"
    )
    path.write_text(table + caption, encoding="utf-8")
    return table + caption


def write_diagnostics(path: Path, source_rows: list[dict[str, str]], rows: list[dict[str, object]]) -> str:
    failures = [r for r in source_rows if r.get("status") != "success"]
    gp = [r for r in source_rows if r.get("method") == "gaussianpile" and r.get("status") == "success"]
    gp_min_psnr = min((as_float(r, "final_psnr") for r in gp), default=float("nan"))
    gp_min_gauss = min((float(r.get("final_num_gaussians") or 0.0) for r in gp), default=float("nan"))
    lines = [
        "# Main NVS Diagnostics",
        "",
        f"- Expected rows: {EXPECTED_ROWS}",
        f"- Successful rows: {len([r for r in source_rows if r.get('status') == 'success'])}",
        f"- Failed/non-success rows: {len(failures)}",
        f"- GaussianPile minimum final PSNR: {fmt(gp_min_psnr)}",
        f"- GaussianPile minimum final Gaussian count: {gp_min_gauss:.0f}",
        "- GaussianPile collapse/black-output diagnostic: inspect the minimum PSNR, final Gaussian count, and qualitative grids before accepting the run.",
        "- Stage gate: manual review required before scaling to the next stage.",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_idx = text.index(start)
    body_start = start_idx + len(start)
    end_idx = text.index(end, body_start)
    return text[:body_start] + "\n\n" + replacement.rstrip() + "\n\n" + text[end_idx:]


def replace_section(text: str, heading: str, next_heading: str, replacement: str) -> str:
    start_idx = text.index(heading)
    end_idx = text.index(next_heading, start_idx + len(heading))
    return text[:start_idx] + replacement.rstrip() + "\n\n" + text[end_idx:]


def update_doc(doc: Path, primary_table: str, paper_table: str, method_table: str, rows: list[dict[str, object]]) -> None:
    text = doc.read_text(encoding="utf-8")
    primary = primary_table.replace("PSNR ↑", "PSNR").replace("SSIM ↑", "SSIM").replace("LPIPS ↓", "LPIPS").replace("L1 ↓", "L1")
    text = replace_between(
        text,
        "Primary table:\n",
        "Required generated files:",
        primary,
    )
    paper_section = (
        "### 9.2 Main NVS Paper Table\n\n"
        "Target file:\n\n"
        "```text\n"
        "tables/main_nvs_paper_table.md\n"
        "```\n\n"
        "Generated staged table:\n\n"
        f"{paper_table}"
    )
    text = replace_section(
        text,
        "### 9.2 Main NVS Paper Table",
        "### 9.3 Method-Locked Configuration Summary",
        paper_section,
    )
    method_section = (
        "### 9.3 Method-Locked Configuration Summary\n\n"
        "Target file:\n\n"
        "```text\n"
        "tables/method_locked_config_summary.md\n"
        "```\n\n"
        "Generated staged launch-gate table:\n\n"
        f"{method_table}\n"
        "This table is a hard launch gate for staged production and must match the frozen protocol before results are accepted."
    )
    text = replace_section(
        text,
        "### 9.3 Method-Locked Configuration Summary",
        "### 9.4 Full Sweep NVS Per-Run Table",
        method_section,
    )

    doc.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables = args.root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_run = args.per_run or (tables / "gs_full_sweep_per_run.tsv")
    source_rows = read_rows(per_run)
    success_rows = [r for r in source_rows if r.get("status") == "success"]
    if args.require_complete and len(success_rows) != EXPECTED_ROWS:
        raise SystemExit(f"Expected {EXPECTED_ROWS} successful rows, found {len(success_rows)}")

    dataset_rows, mean_rows = aggregate(source_rows)
    all_rows = dataset_rows + mean_rows
    ranks = ranks_for(all_rows)

    write_by_dataset_csv(tables / "main_nvs_by_dataset.csv", all_rows, ranks)
    paper_table = write_paper_table(tables / "main_nvs_paper_table.md", all_rows, ranks)
    write_rank_summary(tables / "main_nvs_rank_summary.md", all_rows, ranks)
    method_table = write_method_config(tables / "method_locked_config_summary.md")
    write_budget_table(tables / "main_nvs_gs_strict_protocol_table.md", all_rows, ranks)
    write_diagnostics(tables / "main_nvs_diagnostics.md", source_rows, all_rows)

    if args.update_doc:
        update_doc(args.doc, paper_table, paper_table, method_table, all_rows)


if __name__ == "__main__":
    main()
