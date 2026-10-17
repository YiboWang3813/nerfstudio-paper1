#!/usr/bin/env python3
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


OUTPUT_ROOT = Path("outputs/preproduction/diagnostics/ours_gs_strict_diagnostics")
SUMMARY_PATH = OUTPUT_ROOT / "tables/a1_corridor_sweep_per_run.tsv"
SUMMARY_TSV = OUTPUT_ROOT / "tables/a1_corridor_sweep_summary.tsv"
SUMMARY_MD = OUTPUT_ROOT / "tables/a1_corridor_sweep_summary.md"
NOTES_PATH = OUTPUT_ROOT / "notes/a1_analysis.md"
README_PATH = OUTPUT_ROOT / "README.md"
MANIFEST_PATH = OUTPUT_ROOT / "MANIFEST.yaml"

RATIO_RE = re.compile(r"_(r\d{3})$")
RATIO_VALUES = {
    "r050": 0.50,
    "r075": 0.75,
    "r100": 1.00,
    "r125": 1.25,
    "r150": 1.50,
    "r200": 2.00,
}
DELTAS = {
    "tus_rec_2024_qc": 0.404000,
    "tdsc_abus_2023": 0.475674,
}
BASELINE_TARGETS = {
    "tus_rec_2024_qc": {
        "current_psnr": 32.7664,
        "current_ssim": 0.8661,
        "target": "slice-adapted r150",
        "target_psnr": 33.1485,
        "target_ssim": 0.8711,
    },
    "tdsc_abus_2023": {
        "current_psnr": 27.7341,
        "current_ssim": 0.7859,
        "target": "GaussianPile r200 / slice r125",
        "target_psnr": 29.0897,
        "target_ssim": 0.8014,
    },
}


@dataclass
class Row:
    dataset: str
    sweep: str
    ratio_label: str
    ratio_value: float
    half_corridor_mm: float
    final_psnr: float
    final_l1: float
    final_ssim: float
    final_lpips: float
    final_dists: float
    final_gmsd: float
    final_num_gaussians: str
    train_wall_time_sec: str
    output_dir: str


def as_float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def ratio_label(row: Dict[str, str]) -> Optional[str]:
    match = RATIO_RE.search(row.get("exp_name", ""))
    if match:
        return match.group(1)
    parts = Path(row.get("output_dir", "")).parts
    for part in reversed(parts):
        if part in RATIO_VALUES:
            return part
    return None


def read_rows(path: Path) -> List[Row]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            if raw.get("status") != "success":
                continue
            label = ratio_label(raw)
            if label is None:
                continue
            dataset = raw["dataset"]
            ratio_value = RATIO_VALUES[label]
            rows.append(
                Row(
                    dataset=dataset,
                    sweep=raw["sweep"],
                    ratio_label=label,
                    ratio_value=ratio_value,
                    half_corridor_mm=DELTAS[dataset] * ratio_value,
                    final_psnr=as_float(raw.get("final_psnr", "")),
                    final_l1=as_float(raw.get("final_l1", "")),
                    final_ssim=as_float(raw.get("final_ssim", "")),
                    final_lpips=as_float(raw.get("final_lpips", "")),
                    final_dists=as_float(raw.get("final_dists", "")),
                    final_gmsd=as_float(raw.get("final_gmsd", "")),
                    final_num_gaussians=raw.get("final_num_gaussians", ""),
                    train_wall_time_sec=raw.get("train_wall_time_sec", ""),
                    output_dir=raw.get("output_dir", ""),
                )
            )
    rows.sort(key=lambda r: (r.dataset, r.ratio_value))
    return rows


def best_by_dataset(rows: Iterable[Row]) -> Dict[str, Row]:
    best: Dict[str, Row] = {}
    for row in rows:
        current = best.get(row.dataset)
        if current is None or row.final_psnr > current.final_psnr:
            best[row.dataset] = row
    return best


def write_tsv(rows: List[Row]) -> None:
    SUMMARY_TSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "sweep",
        "ratio_label",
        "ratio_value",
        "half_corridor_mm",
        "final_psnr",
        "final_l1",
        "final_ssim",
        "final_lpips",
        "final_dists",
        "final_gmsd",
        "final_num_gaussians",
        "train_wall_time_sec",
        "output_dir",
    ]
    with SUMMARY_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_table(rows: List[Row]) -> str:
    lines = [
        "| Dataset | Ratio | Half corridor mm | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    row.ratio_label,
                    fmt(row.half_corridor_mm, 6),
                    fmt(row.final_psnr),
                    fmt(row.final_l1, 5),
                    fmt(row.final_ssim),
                    fmt(row.final_lpips),
                    fmt(row.final_dists),
                    fmt(row.final_gmsd),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_outputs(rows: List[Row]) -> None:
    best = best_by_dataset(rows)
    complete = len(rows) == 12
    positive = False
    notes: List[str] = []
    for dataset, best_row in best.items():
        target = BASELINE_TARGETS[dataset]
        psnr_gain = best_row.final_psnr - target["current_psnr"]
        ssim_gain = best_row.final_ssim - target["current_ssim"]
        closes_psnr = best_row.final_psnr >= target["target_psnr"] - 0.15
        if best_row.ratio_label != "r050" and (psnr_gain >= 0.10 or ssim_gain >= 0.002 or closes_psnr):
            positive = True
        notes.append(
            f"- {dataset}: best {best_row.ratio_label} PSNR {fmt(best_row.final_psnr)} "
            f"(delta vs current {psnr_gain:+.4f}), SSIM {fmt(best_row.final_ssim)} "
            f"(delta vs current {ssim_gain:+.4f}); target {target['target']} "
            f"PSNR {target['target_psnr']:.4f} / SSIM {target['target_ssim']:.4f}."
        )

    decision = (
        "A1 positive: proceed to A2 focus-shape sweep around the best corridor settings."
        if complete and positive
        else "A1 not yet positive or incomplete: pause before A2."
    )

    SUMMARY_MD.write_text(markdown_table(rows) + "\n", encoding="utf-8")
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(
        "# A1 Corridor Sweep Analysis\n\n"
        f"Completed rows: {len(rows)} / 12\n\n"
        + "\n".join(notes)
        + "\n\n"
        + decision
        + "\n",
        encoding="utf-8",
    )
    README_PATH.write_text(
        "# Ours GS-Strict Diagnostics\n\n"
        "This output bundle stores diagnostics for the Ours GS-strict endpoint.\n\n"
        f"- A1 per-run table: `{SUMMARY_PATH}`\n"
        f"- A1 summary table: `{SUMMARY_TSV}`\n"
        f"- A1 analysis: `{NOTES_PATH}`\n"
        "- Checkpoints are externalized through `system.checkpoint_root`.\n",
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        "name: ours_gs_strict_diagnostics\n"
        "stage: a1_corridor_sweep\n"
        f"completed_rows: {len(rows)}\n"
        f"expected_rows: 12\n"
        f"decision: {decision}\n"
        "checkpoint_policy: externalized\n",
        encoding="utf-8",
    )
    print(decision)


def main() -> int:
    rows = read_rows(SUMMARY_PATH)
    write_tsv(rows)
    write_outputs(rows)
    print(f"Wrote {SUMMARY_TSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {NOTES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
