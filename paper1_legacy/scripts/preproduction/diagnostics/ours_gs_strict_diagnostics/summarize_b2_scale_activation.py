#!/usr/bin/env python3
import csv
from pathlib import Path


OUTPUT_ROOT = Path("outputs/preproduction/diagnostics/ours_gs_strict_diagnostics")
INPUT = OUTPUT_ROOT / "tables/b2_scale_activation_per_run.tsv"
SUMMARY_MD = OUTPUT_ROOT / "tables/b2_scale_activation_summary.md"
SUMMARY_TSV = OUTPUT_ROOT / "tables/b2_scale_activation_summary.tsv"
NOTES = OUTPUT_ROOT / "notes/b2_scale_activation_analysis.md"


def f(value: str) -> float:
    return float(value)


def variant(exp_name: str) -> str:
    if exp_name.endswith("_bounded"):
        return "bounded"
    if exp_name.endswith("_exp"):
        return "exp"
    return exp_name.rsplit("_", 1)[-1]


def main() -> None:
    rows = []
    with INPUT.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("status") != "success":
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "sweep": row["sweep"],
                    "variant": variant(row["exp_name"]),
                    "psnr": f(row["final_psnr"]),
                    "l1": f(row["final_l1"]),
                    "ssim": f(row["final_ssim"]),
                    "lpips": f(row["final_lpips"]),
                    "dists": f(row["final_dists"]),
                    "gmsd": f(row["final_gmsd"]),
                    "gaussians": row["final_num_gaussians"],
                    "output_dir": row["output_dir"],
                }
            )
    rows.sort(key=lambda r: (r["dataset"], r["variant"]))

    SUMMARY_TSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "sweep", "variant", "psnr", "l1", "ssim", "lpips", "dists", "gmsd", "gaussians", "output_dir"]
    with SUMMARY_TSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "| Dataset | Variant | PSNR | L1 | SSIM | LPIPS | DISTS | GMSD | Final Gaussians |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['dataset']} | {row['variant']} | {row['psnr']:.4f} | {row['l1']:.5f} | "
            f"{row['ssim']:.4f} | {row['lpips']:.4f} | {row['dists']:.4f} | {row['gmsd']:.4f} | {row['gaussians']} |"
        )
    SUMMARY_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["variant"]] = row

    notes = ["# B2 Scale Activation Analysis", "", f"Completed rows: {len(rows)} / 4", ""]
    positive = False
    for dataset, group in sorted(by_dataset.items()):
        exp = group.get("exp")
        bounded = group.get("bounded")
        if not exp or not bounded:
            notes.append(f"- {dataset}: incomplete pair.")
            continue
        d_psnr = bounded["psnr"] - exp["psnr"]
        d_ssim = bounded["ssim"] - exp["ssim"]
        d_gmsd = bounded["gmsd"] - exp["gmsd"]
        notes.append(
            f"- {dataset}: bounded vs exp delta PSNR {d_psnr:+.4f}, "
            f"SSIM {d_ssim:+.4f}, GMSD {d_gmsd:+.4f}."
        )
        if d_psnr >= 0.10 or d_ssim >= 0.002 or d_gmsd <= -0.002:
            positive = True

    decision = (
        "B2 positive: bounded scale activation should be promoted into the GS-strict Ours diagnostic endpoint."
        if len(rows) == 4 and positive
        else "B2 not positive: bounded scale activation alone does not explain the Ours gap."
    )
    notes.extend(["", decision, ""])
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    NOTES.write_text("\n".join(notes), encoding="utf-8")
    print(decision)
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {NOTES}")


if __name__ == "__main__":
    main()
