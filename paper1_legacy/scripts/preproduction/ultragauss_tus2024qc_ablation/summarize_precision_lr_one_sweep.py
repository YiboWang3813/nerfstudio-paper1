#!/usr/bin/env python3
import csv
from pathlib import Path


SUMMARY = Path("outputs/preproduction/ultragauss_tus2024qc_ablation/tables/precision_lr_one_sweep.tsv")
REPORT = Path("docs/production/problems_during_production/ultragauss_tus2024qc_ablation_report.md")


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None


def main():
    rows = read_rows(SUMMARY)
    rows = [r for r in rows if r.get("dataset") == "tus_rec_2024_qc"]
    rows.sort(key=lambda r: r.get("method", ""))

    print(f"summary={SUMMARY}")
    print(f"rows={len(rows)}")
    for r in rows:
        psnr = f(r, "final_psnr")
        lpips = f(r, "final_lpips")
        ssim = f(r, "final_ssim")
        gauss = f(r, "final_num_gaussians")
        print(
            r.get("method", ""),
            "psnr=", "" if psnr is None else f"{psnr:.4f}",
            "lpips=", "" if lpips is None else f"{lpips:.4f}",
            "ssim=", "" if ssim is None else f"{ssim:.4f}",
            "gaussians=", "" if gauss is None else f"{gauss:.0f}",
            "output=", r.get("output_dir", ""),
        )


if __name__ == "__main__":
    main()
