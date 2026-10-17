#!/usr/bin/env python3
import csv
from pathlib import Path


OUTPUT_ROOT = Path("outputs/preproduction/diagnostics/ours_gs_strict_diagnostics")
B2 = OUTPUT_ROOT / "tables/b2_scale_activation_summary.tsv"
B3 = OUTPUT_ROOT / "tables/b3_soft_attenuation_per_run.tsv"
SUMMARY_MD = OUTPUT_ROOT / "tables/b3_soft_attenuation_summary.md"
NOTES = OUTPUT_ROOT / "notes/b3_soft_attenuation_analysis.md"


def load_b2_bounded():
    rows = {}
    with B2.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row["variant"] == "bounded":
                rows[row["dataset"]] = row
    return rows


def load_b3():
    rows = {}
    with B3.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("status") == "success":
                rows[row["dataset"]] = row
    return rows


def val(row, key):
    return float(row[key])


def main():
    baseline = load_b2_bounded()
    b3 = load_b3()
    lines = [
        "| Dataset | Baseline PSNR | Soft PSNR | Delta PSNR | Baseline SSIM | Soft SSIM | Delta SSIM | Baseline LPIPS | Soft LPIPS | Delta LPIPS | Baseline GMSD | Soft GMSD | Delta GMSD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    notes = ["# B3 Soft Attenuation Analysis", "", f"Completed rows: {len(b3)} / 2", ""]
    positive = False
    for dataset in sorted(b3):
        base = baseline[dataset]
        soft = b3[dataset]
        d_psnr = val(soft, "final_psnr") - val(base, "psnr")
        d_ssim = val(soft, "final_ssim") - val(base, "ssim")
        d_lpips = val(soft, "final_lpips") - val(base, "lpips")
        d_gmsd = val(soft, "final_gmsd") - val(base, "gmsd")
        lines.append(
            f"| {dataset} | {val(base, 'psnr'):.4f} | {val(soft, 'final_psnr'):.4f} | {d_psnr:+.4f} | "
            f"{val(base, 'ssim'):.4f} | {val(soft, 'final_ssim'):.4f} | {d_ssim:+.4f} | "
            f"{val(base, 'lpips'):.4f} | {val(soft, 'final_lpips'):.4f} | {d_lpips:+.4f} | "
            f"{val(base, 'gmsd'):.4f} | {val(soft, 'final_gmsd'):.4f} | {d_gmsd:+.4f} |"
        )
        notes.append(
            f"- {dataset}: soft slice-depth vs bounded-integral delta PSNR {d_psnr:+.4f}, "
            f"SSIM {d_ssim:+.4f}, LPIPS {d_lpips:+.4f}, GMSD {d_gmsd:+.4f}."
        )
        if d_psnr >= 0.10 or d_ssim >= 0.002 or d_lpips <= -0.005 or d_gmsd <= -0.002:
            positive = True

    decision = (
        "B3 positive: soft slice-depth attenuation improves the bounded Ours diagnostic endpoint."
        if len(b3) == 2 and positive
        else "B3 not positive: soft slice-depth attenuation does not materially improve bounded Ours."
    )
    notes.extend(["", decision, ""])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    NOTES.write_text("\n".join(notes), encoding="utf-8")
    print(decision)
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {NOTES}")


if __name__ == "__main__":
    main()
