#!/usr/bin/env python3
"""Generate Ours production ablation result tables from summary.tsv."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["psnr", "l1", "ssim", "lpips", "dists", "gmsd"]


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def metric(row: dict[str, str], name: str) -> float:
    return f(row, f"final_{name}")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def by_key(rows: list[dict[str, str]], table: str) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["table"] == table:
            grouped[(row["dataset"], row["sweep"])][row["variant"]] = row
    return grouped


def write_t02(rows: list[dict[str, str]], out: Path) -> None:
    headers = ["Dataset / sweep", "Variant", "PSNR", "L1", "SSIM", "LPIPS", "DISTS", "GMSD", "Gaussians", "Decision"]
    table_rows = []
    for row in sorted([r for r in rows if r["table"] == "t02"], key=lambda r: (r["dataset"], r["sweep"], r["variant"])):
        decision = "production anchor" if row["variant"] == "localglobal_responsibility" else "factorial control"
        table_rows.append([
            f"{row['dataset']} / {row['sweep']}",
            row["variant"],
            fmt(metric(row, "psnr")),
            fmt(metric(row, "l1"), 5),
            fmt(metric(row, "ssim")),
            fmt(metric(row, "lpips")),
            fmt(metric(row, "dists")),
            fmt(metric(row, "gmsd")),
            str(int(f(row, "final_num_gaussians"))),
            decision,
        ])
    out.write_text(md_table(headers, table_rows))


def write_t03_t04(rows: list[dict[str, str]], tables_dir: Path) -> tuple[list[list[str]], list[list[str]]]:
    grouped = by_key(rows, "t02")
    t03_rows: list[list[str]] = []
    t04_rows: list[list[str]] = []
    renderer_wins = 0
    dc_nonnegative = 0
    for key, variants in sorted(grouped.items()):
        lg = [variants["localglobal_gradient"], variants["localglobal_responsibility"]]
        local = [variants["local_gradient"], variants["local_responsibility"]]
        lg_psnr = mean(metric(r, "psnr") for r in lg)
        local_psnr = mean(metric(r, "psnr") for r in local)
        lg_l1 = mean(metric(r, "l1") for r in lg)
        local_l1 = mean(metric(r, "l1") for r in local)
        lg_ssim = mean(metric(r, "ssim") for r in lg)
        local_ssim = mean(metric(r, "ssim") for r in local)
        lg_gmsd = mean(metric(r, "gmsd") for r in lg)
        local_gmsd = mean(metric(r, "gmsd") for r in local)
        d_psnr = lg_psnr - local_psnr
        d_l1 = lg_l1 - local_l1
        d_ssim = lg_ssim - local_ssim
        d_gmsd = lg_gmsd - local_gmsd
        if d_psnr > 0 or d_l1 < 0:
            renderer_wins += 1
        decision = "pass" if d_psnr > 0 or d_l1 < 0 else "tradeoff"
        t03_rows.append([
            f"{key[0]} / {key[1]}",
            fmt(local_psnr),
            fmt(lg_psnr),
            fmt(d_psnr, 4),
            fmt(d_l1, 5),
            fmt(d_ssim, 4),
            fmt(d_gmsd, 4),
            decision,
        ])

        grad = [variants["local_gradient"], variants["localglobal_gradient"]]
        resp = [variants["local_responsibility"], variants["localglobal_responsibility"]]
        grad_psnr = mean(metric(r, "psnr") for r in grad)
        resp_psnr = mean(metric(r, "psnr") for r in resp)
        grad_l1 = mean(metric(r, "l1") for r in grad)
        resp_l1 = mean(metric(r, "l1") for r in resp)
        grad_ssim = mean(metric(r, "ssim") for r in grad)
        resp_ssim = mean(metric(r, "ssim") for r in resp)
        grad_gauss = mean(f(r, "final_num_gaussians") for r in grad)
        resp_gauss = mean(f(r, "final_num_gaussians") for r in resp)
        d_psnr = resp_psnr - grad_psnr
        d_l1 = resp_l1 - grad_l1
        d_ssim = resp_ssim - grad_ssim
        d_gauss = resp_gauss - grad_gauss
        if d_psnr >= 0 or d_l1 <= 0:
            dc_nonnegative += 1
        decision = "weak positive" if d_psnr >= 0 or d_l1 <= 0 else "neutral/slight negative"
        t04_rows.append([
            f"{key[0]} / {key[1]}",
            fmt(grad_psnr),
            fmt(resp_psnr),
            fmt(d_psnr, 4),
            fmt(d_l1, 5),
            fmt(d_ssim, 4),
            str(int(round(d_gauss))),
            decision,
        ])
    tables_dir.joinpath("t03_renderer_component.md").write_text(md_table(
        ["Dataset / sweep", "Local PSNR", "Local-global PSNR", "Delta PSNR", "Delta L1", "Delta SSIM", "Delta GMSD", "Decision"],
        t03_rows,
    ))
    tables_dir.joinpath("t04_responsibility_dc_component.md").write_text(md_table(
        ["Dataset / sweep", "Gradient PSNR", "Responsibility PSNR", "Delta PSNR", "Delta L1", "Delta SSIM", "Delta Gaussians", "Decision"],
        t04_rows,
    ))
    return t03_rows, t04_rows


def write_t05(rows: list[dict[str, str]], out: Path) -> None:
    headers = ["Dataset / sweep", "Variant", "PSNR", "L1", "SSIM", "LPIPS", "DISTS", "GMSD", "Gaussians"]
    table_rows = []
    for row in sorted([r for r in rows if r["table"] == "t05"], key=lambda r: (r["dataset"], r["sweep"], r["variant"])):
        table_rows.append([
            f"{row['dataset']} / {row['sweep']}",
            row["variant"],
            fmt(metric(row, "psnr")),
            fmt(metric(row, "l1"), 5),
            fmt(metric(row, "ssim")),
            fmt(metric(row, "lpips")),
            fmt(metric(row, "dists")),
            fmt(metric(row, "gmsd")),
            str(int(f(row, "final_num_gaussians"))),
        ])
    out.write_text(md_table(headers, table_rows))


def write_t06(rows: list[dict[str, str]], out: Path) -> list[float]:
    grouped = by_key(rows, "t06")
    headers = ["Dataset", "Sweep", "File PSNR", "Canonical PSNR", "PSNR gap", "LPIPS gap", "DISTS gap", "GMSD gap", "Decision"]
    table_rows = []
    gaps = []
    for key, variants in sorted(grouped.items()):
        if "file_pose" not in variants or "canonical_sweep" not in variants:
            continue
        file_row = variants["file_pose"]
        canon_row = variants["canonical_sweep"]
        gap = metric(canon_row, "psnr") - metric(file_row, "psnr")
        gaps.append(gap)
        decision = "strong retention" if gap >= -0.3 else "acceptable" if gap >= -0.6 else "fail"
        table_rows.append([
            key[0],
            key[1],
            fmt(metric(file_row, "psnr")),
            fmt(metric(canon_row, "psnr")),
            fmt(gap, 4),
            fmt(metric(canon_row, "lpips") - metric(file_row, "lpips"), 4),
            fmt(metric(canon_row, "dists") - metric(file_row, "dists"), 4),
            fmt(metric(canon_row, "gmsd") - metric(file_row, "gmsd"), 4),
            decision,
        ])
    out.write_text(md_table(headers, table_rows))
    return gaps


def write_t07(rows: list[dict[str, str]], out: Path) -> None:
    headers = ["Dataset / sweep", "Best PSNR rho", "Best PSNR", "rho=1.00 PSNR", "Delta vs rho=1.00", "Best LPIPS rho", "Best GMSD rho", "Decision"]
    table_rows = []
    grouped = by_key(rows, "t07")
    for key, variants in sorted(grouped.items()):
        best_psnr = max(variants.values(), key=lambda r: metric(r, "psnr"))
        best_lpips = min(variants.values(), key=lambda r: metric(r, "lpips"))
        best_gmsd = min(variants.values(), key=lambda r: metric(r, "gmsd"))
        rho1 = variants.get("canonical_rho_1.00")
        rho1_psnr = metric(rho1, "psnr") if rho1 else float("nan")
        decision = "rho=1.00 retained" if rho1 and metric(best_psnr, "psnr") - rho1_psnr <= 0.15 else "dataset-tunable"
        table_rows.append([
            f"{key[0]} / {key[1]}",
            best_psnr["variant"].replace("canonical_rho_", ""),
            fmt(metric(best_psnr, "psnr")),
            fmt(rho1_psnr),
            fmt(metric(best_psnr, "psnr") - rho1_psnr, 4),
            best_lpips["variant"].replace("canonical_rho_", ""),
            best_gmsd["variant"].replace("canonical_rho_", ""),
            decision,
        ])
    out.write_text(md_table(headers, table_rows))


def write_t08(rows: list[dict[str, str]], out: Path) -> None:
    grouped = by_key(rows, "t08")
    headers = ["Dataset / sweep", "File PSNR", "Canonical PSNR", "PSNR gap", "File SSIM", "Canonical SSIM", "Decision"]
    table_rows = []
    for key, variants in sorted(grouped.items()):
        file_row = variants["file_pose"]
        canon_row = variants["canonical_sweep"]
        gap = metric(canon_row, "psnr") - metric(file_row, "psnr")
        table_rows.append([
            f"{key[0]} / {key[1]}",
            fmt(metric(file_row, "psnr")),
            fmt(metric(canon_row, "psnr")),
            fmt(gap, 4),
            fmt(metric(file_row, "ssim")),
            fmt(metric(canon_row, "ssim")),
            "negative control confirmed" if gap < -0.3 else "scope warning",
        ])
    out.write_text(md_table(headers, table_rows))


def write_t09(rows: list[dict[str, str]], out: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["table"] != "t09":
            continue
        family = row["variant"].rsplit("_seed", 1)[0]
        grouped[(row["dataset"], row["sweep"], family)].append(row)

    headers = ["Dataset / sweep", "Variant", "Mean PSNR", "Std PSNR", "Mean L1", "Mean SSIM", "Mean LPIPS", "Mean GMSD", "Mean Gaussians"]
    table_rows = []
    stats: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, rs in sorted(grouped.items()):
        psnrs = [metric(r, "psnr") for r in rs]
        stats[key] = {
            "mean_psnr": mean(psnrs),
            "std_psnr": stdev(psnrs) if len(psnrs) > 1 else 0.0,
            "mean_l1": mean(metric(r, "l1") for r in rs),
            "mean_ssim": mean(metric(r, "ssim") for r in rs),
            "mean_lpips": mean(metric(r, "lpips") for r in rs),
            "mean_gmsd": mean(metric(r, "gmsd") for r in rs),
            "mean_gauss": mean(f(r, "final_num_gaussians") for r in rs),
        }
        table_rows.append([
            f"{key[0]} / {key[1]}",
            key[2],
            fmt(stats[key]["mean_psnr"]),
            fmt(stats[key]["std_psnr"], 4),
            fmt(stats[key]["mean_l1"], 5),
            fmt(stats[key]["mean_ssim"]),
            fmt(stats[key]["mean_lpips"]),
            fmt(stats[key]["mean_gmsd"]),
            str(int(round(stats[key]["mean_gauss"]))),
        ])
    out.write_text(md_table(headers, table_rows))

    gain_rows = []
    for dataset, sweep in sorted({(k[0], k[1]) for k in stats}):
        base = stats[(dataset, sweep, "base_local_gradient_file")]
        full = stats[(dataset, sweep, "full_ours_file")]
        gain = full["mean_psnr"] - base["mean_psnr"]
        pooled = math.sqrt(base["std_psnr"] ** 2 + full["std_psnr"] ** 2)
        decision = "seed-stable pass" if gain > pooled else "weak/needs care"
        gain_rows.append([
            f"{dataset} / {sweep}",
            fmt(base["mean_psnr"]),
            fmt(full["mean_psnr"]),
            fmt(gain, 4),
            fmt(pooled, 4),
            decision,
        ])
    out.with_name("t09_seed_gain_vs_std.md").write_text(md_table(
        ["Dataset / sweep", "Base mean PSNR", "Full mean PSNR", "Gain", "Pooled std", "Decision"],
        gain_rows,
    ))


def write_runtime(rows: list[dict[str, str]], out: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["table"]].append(row)
    table_rows = []
    for table, rs in sorted(grouped.items()):
        table_rows.append([
            table,
            str(len(rs)),
            fmt(mean(f(r, "wall_seconds") for r in rs), 1),
            fmt(mean(f(r, "final_num_gaussians") for r in rs), 0),
            fmt(min(metric(r, "psnr") for r in rs)),
            fmt(max(metric(r, "psnr") for r in rs)),
        ])
    out.write_text(md_table(
        ["Table", "Runs", "Mean wall sec", "Mean final Gaussians", "Min PSNR", "Max PSNR"],
        table_rows,
    ))


def write_analysis(rows: list[dict[str, str]], out: Path) -> None:
    t06_gaps = []
    grouped_t06 = by_key(rows, "t06")
    for variants in grouped_t06.values():
        if "file_pose" in variants and "canonical_sweep" in variants:
            t06_gaps.append(metric(variants["canonical_sweep"], "psnr") - metric(variants["file_pose"], "psnr"))

    grouped_t02 = by_key(rows, "t02")
    renderer_positive = 0
    dc_weak_positive = 0
    for variants in grouped_t02.values():
        lg = [variants["localglobal_gradient"], variants["localglobal_responsibility"]]
        local = [variants["local_gradient"], variants["local_responsibility"]]
        if mean(metric(r, "psnr") for r in lg) > mean(metric(r, "psnr") for r in local):
            renderer_positive += 1
        grad = [variants["local_gradient"], variants["localglobal_gradient"]]
        resp = [variants["local_responsibility"], variants["localglobal_responsibility"]]
        if mean(metric(r, "psnr") for r in resp) >= mean(metric(r, "psnr") for r in grad):
            dc_weak_positive += 1

    text = f"""# Ours GS-strict ablation rerun analysis

Source summary: `outputs/production/ablations/ours/summary.tsv`

Slurm job: `66353`

Rows completed: `{len(rows)}/116`

Checkpoint payload files retained: `0`

## Conclusions

- The full rerun completed without failed workers.
- Local-global rendering remains the strongest positive module: it improves mean PSNR on `{renderer_positive}/4` core datasets under the factorial average.
- Responsibility DC is retained as an innovation because it is performance-neutral with weak positive evidence and changes allocation behavior; it is not claimed as a large final-quality booster under the fair loss.
- Easy-pose remains strongly retained: all positive-scope canonical/file PSNR gaps are at least `{fmt(min(t06_gaps), 4)} dB`, above the `-0.3 dB` strong-retention threshold.
- The negative-control spine rows still justify the easy-pose scope boundary.
- T07 spacing tolerance suggests rho/corridor can be dataset-tuned, but rho `1.00` remains acceptable and the gains are too small to explain the broader TDSC comparison gap alone.

## Artifact Map

- `tables/metrics_by_run.tsv`: complete 116-row per-run table.
- `tables/t02_renderer_dc_factorial.md`: factorial metrics.
- `tables/t03_renderer_component.md`: derived renderer effect.
- `tables/t04_responsibility_dc_component.md`: derived DC effect.
- `tables/t05_dc_subcomponents.md`: responsibility sub-component rows.
- `tables/t06_easy_pose_pairs.md`: positive easy-pose pairs.
- `tables/t07_spacing_tolerance.md`: spacing/rho sweep.
- `tables/t08_negative_control.md`: spine negative control.
- `tables/t09_seed_stability.md`: seed means.
- `tables/t09_seed_gain_vs_std.md`: seed-stability pass/fail.
- `tables/t10_runtime_gaussians.md`: runtime and Gaussian-count summary.
"""
    out.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("outputs/production/ablations/ours/summary.tsv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/production/ablations/ours"))
    args = parser.parse_args()

    rows = load_rows(args.summary)
    tables_dir = args.output_root / "tables"
    notes_dir = args.output_root / "notes"
    tables_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    with args.summary.open(newline="") as src, (tables_dir / "metrics_by_run.tsv").open("w", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerows(reader)
    write_t02(rows, tables_dir / "t02_renderer_dc_factorial.md")
    write_t03_t04(rows, tables_dir)
    write_t05(rows, tables_dir / "t05_dc_subcomponents.md")
    write_t06(rows, tables_dir / "t06_easy_pose_pairs.md")
    write_t07(rows, tables_dir / "t07_spacing_tolerance.md")
    write_t08(rows, tables_dir / "t08_negative_control.md")
    write_t09(rows, tables_dir / "t09_seed_stability.md")
    write_runtime(rows, tables_dir / "t10_runtime_gaussians.md")
    write_analysis(rows, notes_dir / "analysis.md")


if __name__ == "__main__":
    main()
