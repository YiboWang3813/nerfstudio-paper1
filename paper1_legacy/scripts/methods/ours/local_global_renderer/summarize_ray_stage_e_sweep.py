#!/usr/bin/env python
"""Summarize Stage-E ray attenuation sweep metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf


def _read_final_metrics(scalars_path: Path) -> dict[str, float]:
    rows: dict[int, dict[str, float]] = {}
    with scalars_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for step_s, key, value_s in reader:
            if not key.startswith("test/"):
                continue
            step = int(step_s)
            rows.setdefault(step, {})[key[len("test/"):]] = float(value_s)
    if not rows:
        raise ValueError(f"No test metrics found in {scalars_path}")
    final_step = max(rows)
    out = dict(rows[final_step])
    out["iteration"] = float(final_step)
    return out


def _read_ray_params(config_path: Path) -> dict[str, float]:
    cfg = OmegaConf.load(config_path)
    sr = cfg.model.slice_render
    return {
        "beta": float(sr.ray_attenuation_beta),
        "strength": float(sr.ray_attenuation_strength),
        "min_transmission": float(sr.ray_attenuation_min_transmission),
        "quadrature_points": int(getattr(sr, "ray_attenuation_quadrature_points", 3)),
        "formula": str(getattr(sr, "ray_attenuation_formula", "integral")),
        "candidate_k_sigma": float(getattr(sr, "ray_attenuation_candidate_k_sigma", -1.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        scalars = run_dir / "logs" / "scalars.tsv"
        config = run_dir / "resolved_config.yaml"
        if not scalars.is_file() or not config.is_file():
            continue
        metrics = _read_final_metrics(scalars)
        params = _read_ray_params(config)
        runs.append({"name": run_dir.name, **params, **metrics})

    runs.sort(key=lambda r: (-r["psnr"], r["l1"], -r["ssim"], r["lpips"]))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    lines = [
        "| Rank | Run | formula | cand k | beta | strength | minT | qpts | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(runs, start=1):
        lines.append(
            "| {rank} | {name} | {formula} | {cand_k:.2f} | {beta:.2f} | {strength:.2f} | {min_t:.2f} | {qpts:d} | "
            "{l1:.6f} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} |".format(
                rank=rank,
                name=row["name"],
                formula=row["formula"],
                cand_k=row["candidate_k_sigma"],
                beta=row["beta"],
                strength=row["strength"],
                min_t=row["min_transmission"],
                qpts=int(row["quadrature_points"]),
                l1=row["l1"],
                psnr=row["psnr"],
                ssim=row["ssim"],
                lpips=row["lpips"],
            )
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)
    if runs:
        print(json.dumps(runs[0], indent=2))


if __name__ == "__main__":
    main()
