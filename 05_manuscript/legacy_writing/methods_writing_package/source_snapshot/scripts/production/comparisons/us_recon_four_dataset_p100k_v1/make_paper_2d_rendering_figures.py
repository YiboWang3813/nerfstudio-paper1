#!/usr/bin/env python3
"""Build paper-facing 2D rendering comparison figures.

The figure policy is defined in docs/production/multi_comparisons.md:
two Ours-winning sweeps per dataset, fixed paper baseline columns, then Ours
and GT. Each panel shows the full rendering with an automatically selected
zoom region where Ours has a local error advantage over the fixed baselines.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image


BENCHMARK_ROOT = Path("outputs/production/comparisons/us_recon_four_dataset_p100k_v1")
TABLE_ROOT = BENCHMARK_ROOT / "tables"
OUT_DIR = Path("docs/production/comparison_final_outputs/figures")
MANIFEST_DIR = Path("docs/production/comparison_final_outputs/manifests")

SUMMARY_FILES = [
    "gs_stage2_fixed_subset_per_run.tsv",
    "gs_full_sweep_per_run.tsv",
    "gs_ultragauss_tus_qc_chi2_5500_replacement.tsv",
]
SUMMARY_PRIORITY = {name: i for i, name in enumerate(SUMMARY_FILES)}

METHODS = [
    "ours",
    "ultragauss",
    "ultragray",
    "gaussianpile",
    "ultrags_baseline",
    "slice_adapted_3dgs",
]

PAPER_METHODS = [
    "ultragauss",
    "ultragray",
    "slice_adapted_3dgs",
    "gaussianpile",
    "ours",
    "gt",
]

METHOD_LABEL = {
    "ours": "Ours",
    "ultragauss": "UltraGauss",
    "ultragray": "UltraGray",
    "gaussianpile": "GaussianPile",
    "ultrags_baseline": "UltraGS",
    "slice_adapted_3dgs": "Slice-3DGS",
    "gt": "GT",
}

GROUPS = {
    "simulated": ["synthetic_liver", "spine_phantom"],
    "real": ["tus_rec_2024_qc", "tdsc_abus_2023"],
}


@dataclass
class PanelSpec:
    dataset: str
    sweep: str
    item: int
    crop_xywh: tuple[int, int, int, int]
    methods: list[str]


def _read_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for name in SUMMARY_FILES:
        path = TABLE_ROOT / name
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row["dataset"], row["sweep"], row["method"])
                if key not in rows or SUMMARY_PRIORITY[name] > SUMMARY_PRIORITY[rows[key]["_summary_file"]]:
                    row["_summary_file"] = name
                    rows[key] = row
    return rows


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def _select_sweeps(rows: dict[tuple[str, str, str], dict[str, str]], dataset: str) -> list[str]:
    sweeps = sorted({sweep for ds, sweep, _method in rows if ds == dataset})
    candidates: list[tuple[float, str]] = []
    for sweep in sweeps:
        ranked = sorted(
            (
                (_float(rows[(dataset, sweep, method)], "final_psnr"), method)
                for method in METHODS
                if (dataset, sweep, method) in rows
            ),
            reverse=True,
        )
        if ranked and ranked[0][1] == "ours":
            candidates.append((ranked[0][0], sweep))
    if len(candidates) < 2:
        raise RuntimeError(f"Need at least two Ours-winning sweeps for {dataset}, got {len(candidates)}.")
    return [sweep for _psnr, sweep in sorted(candidates, reverse=True)[:2]]


def _rank_methods(rows: dict[tuple[str, str, str], dict[str, str]], dataset: str, sweep: str) -> list[str]:
    missing = [method for method in PAPER_METHODS if method != "gt" and (dataset, sweep, method) not in rows]
    if missing:
        raise RuntimeError(f"Missing fixed paper methods for {dataset}/{sweep}: {missing}")
    return list(PAPER_METHODS)


def _run_dir(rows: dict[tuple[str, str, str], dict[str, str]], dataset: str, sweep: str, method: str) -> Path:
    return Path(rows[(dataset, sweep, method)]["output_dir"])


def _available_items(run_dir: Path) -> set[int]:
    items = set()
    for path in (run_dir / "vis").glob("item_*"):
        if path.is_dir() and (path / "pred.png").exists():
            try:
                items.add(int(path.name.split("_", 1)[1]))
            except ValueError:
                continue
    return items


def _read_per_frame(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "eval" / "per_frame_metrics_latest.tsv"
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            item = int(row["item"])
            out[item] = {
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
            }
    return out


def _image_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return arr


def _select_item(
    rows: dict[tuple[str, str, str], dict[str, str]],
    dataset: str,
    sweep: str,
    methods: list[str],
) -> int:
    real_methods = [m for m in methods if m != "gt"]
    common: set[int] | None = None
    metrics: dict[str, dict[int, dict[str, float]]] = {}
    for method in real_methods:
        rd = _run_dir(rows, dataset, sweep, method)
        items = _available_items(rd)
        common = items if common is None else common & items
        metrics[method] = _read_per_frame(rd)
    if not common:
        raise RuntimeError(f"No common vis items for {dataset}/{sweep}/{methods}")

    best_item = sorted(common)[0]
    best_score = -float("inf")
    baselines = [m for m in real_methods if m != "ours"]
    for item in sorted(common):
        ours = metrics["ours"].get(item)
        if ours is None:
            continue
        baseline_psnr = [metrics[m][item]["psnr"] for m in baselines if item in metrics[m]]
        baseline_ssim = [metrics[m][item]["ssim"] for m in baselines if item in metrics[m]]
        if not baseline_psnr:
            continue
        # Prefer frames where Ours separates clearly but avoid using PSNR alone.
        score = (ours["psnr"] - max(baseline_psnr)) + 5.0 * (ours["ssim"] - max(baseline_ssim))
        if score > best_score:
            best_score = score
            best_item = item
    return best_item


def _select_crop(
    rows: dict[tuple[str, str, str], dict[str, str]],
    dataset: str,
    sweep: str,
    item: int,
    methods: list[str],
) -> tuple[int, int, int, int]:
    ours_dir = _run_dir(rows, dataset, sweep, "ours")
    gt = _image_array(ours_dir / "vis" / f"item_{item}" / "gt.png")
    ours = _image_array(ours_dir / "vis" / f"item_{item}" / "pred.png")
    h, w = gt.shape
    crop_w = max(36, int(w * 0.24))
    crop_h = max(36, int(h * 0.24))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    baselines = [m for m in methods if m not in {"ours", "gt"}]
    base_errors = []
    for method in baselines:
        pred = _image_array(_run_dir(rows, dataset, sweep, method) / "vis" / f"item_{item}" / "pred.png")
        base_errors.append(np.abs(pred - gt))
    mean_base_error = np.mean(np.stack(base_errors, axis=0), axis=0)
    ours_error = np.abs(ours - gt)
    advantage = mean_base_error - ours_error
    texture = np.abs(gt - gt.mean())
    score_map = advantage + 0.15 * texture

    best = (-float("inf"), 0, 0)
    nx = 10
    ny = 10
    for yy in np.linspace(0, max(0, h - crop_h), ny).astype(int):
        for xx in np.linspace(0, max(0, w - crop_w), nx).astype(int):
            patch = score_map[yy : yy + crop_h, xx : xx + crop_w]
            score = float(np.mean(patch))
            if score > best[0]:
                best = (score, xx, yy)
    return best[1], best[2], crop_w, crop_h


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _crop_array(img: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = crop
    return img[y : y + h, x : x + w]


def _metric_text(run_dir: Path, item: int) -> str:
    metrics = _read_per_frame(run_dir)
    m = metrics[item]
    return f"PSNR {m['psnr']:.2f}\nSSIM {m['ssim']:.3f}"


def _short_row_label(dataset: str, sweep: str) -> str:
    if dataset == "synthetic_liver":
        return {
            "l3": "Liver_L1",
            "l1": "Liver_R1",
        }.get(sweep, f"Liver_{sweep.upper()}")
    if dataset == "spine_phantom":
        return {
            "left3_2": "Spine_L1",
            "left1_1": "Spine_L11",
        }.get(sweep, f"Spine_{sweep.replace('left', 'L').replace('_', '')}")
    if dataset == "tdsc_abus_2023":
        return sweep.replace("validation_", "TDSC_")
    if dataset == "tus_rec_2024_qc":
        parts = sweep.split("_qc_", 1)
        if len(parts) == 2:
            suffix_parts = parts[1].split("_")
            suffix = suffix_parts[-1] if suffix_parts[-1].startswith("s") else suffix_parts[-1]
            return f"TUS_{suffix.upper()}"
        return sweep
    return sweep


def _inset_bounds(crop: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = crop
    ih, iw = image_shape
    crop_box = np.array([x / iw, y / ih, (x + w) / iw, (y + h) / ih], dtype=np.float32)
    candidates = [
        (0.56, 0.05, 0.40, 0.40),
        (0.04, 0.05, 0.40, 0.40),
        (0.56, 0.56, 0.40, 0.40),
        (0.04, 0.56, 0.40, 0.40),
    ]

    def overlap(candidate: tuple[float, float, float, float]) -> float:
        cx, cy, cw, ch = candidate
        # Axes coordinates have origin at lower left; image coordinates use upper left.
        cand_box = np.array([cx, 1.0 - cy - ch, cx + cw, 1.0 - cy], dtype=np.float32)
        ix0 = max(crop_box[0], cand_box[0])
        iy0 = max(crop_box[1], cand_box[1])
        ix1 = min(crop_box[2], cand_box[2])
        iy1 = min(crop_box[3], cand_box[3])
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    return min(candidates, key=overlap)


def _axes_crop_box(crop: tuple[int, int, int, int], image_shape: tuple[int, int]) -> np.ndarray:
    x, y, w, h = crop
    ih, iw = image_shape
    return np.array([x / iw, 1.0 - (y + h) / ih, (x + w) / iw, 1.0 - y / ih], dtype=np.float32)


def _box_overlap(a: np.ndarray, b: np.ndarray) -> float:
    ix0 = max(float(a[0]), float(b[0]))
    iy0 = max(float(a[1]), float(b[1]))
    ix1 = min(float(a[2]), float(b[2]))
    iy1 = min(float(a[3]), float(b[3]))
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _text_anchor(
    crop: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    inset: tuple[float, float, float, float],
) -> tuple[float, float, str, str]:
    crop_box = _axes_crop_box(crop, image_shape)
    ix, iy, iw, ih = inset
    inset_box = np.array([ix, iy, ix + iw, iy + ih], dtype=np.float32)
    candidates = [
        (0.02, 0.98, "left", "top", np.array([0.00, 0.80, 0.38, 1.00], dtype=np.float32)),
        (0.02, 0.02, "left", "bottom", np.array([0.00, 0.00, 0.38, 0.20], dtype=np.float32)),
        (0.98, 0.98, "right", "top", np.array([0.62, 0.80, 1.00, 1.00], dtype=np.float32)),
        (0.98, 0.02, "right", "bottom", np.array([0.62, 0.00, 1.00, 0.20], dtype=np.float32)),
    ]
    best = min(candidates, key=lambda c: _box_overlap(crop_box, c[4]) + 1.5 * _box_overlap(inset_box, c[4]))
    return best[0], best[1], best[2], best[3]


def _draw_full_with_zoom(
    ax: plt.Axes,
    img: np.ndarray,
    crop: tuple[int, int, int, int],
    label: str,
    metric: str,
) -> None:
    x, y, w, h = crop
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")

    rect_color = "#ffd84d"
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=rect_color, linewidth=0.85))

    inset_bounds = _inset_bounds(crop, img.shape)
    ix, iy, iw, ih = inset_bounds
    inset = ax.inset_axes(inset_bounds)
    inset.imshow(_crop_array(img, crop), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(rect_color)
        spine.set_linewidth(0.9)

    rect_corners = [(x, y), (x + w, y + h)]
    inset_corners = [(0.0, 1.0), (1.0, 0.0)]
    if ix < 0.5:
        rect_corners = [(x, y + h), (x + w, y)]
        inset_corners = [(1.0, 1.0), (0.0, 0.0)]
    for xy_a, xy_b in zip(rect_corners, inset_corners):
        con = ConnectionPatch(
            xyA=xy_a,
            coordsA=ax.transData,
            xyB=xy_b,
            coordsB=inset.transAxes,
            color=rect_color,
            linewidth=0.55,
            alpha=0.95,
            clip_on=False,
        )
        ax.add_artist(con)

    tx, ty, tha, tva = _text_anchor(crop, img.shape, inset_bounds)
    if metric:
        ax.text(
            tx,
            ty,
            metric,
            transform=ax.transAxes,
            ha=tha,
            va=tva,
            fontsize=8.0,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.58, "pad": 1.4, "edgecolor": "none"},
        )


def _build_specs(rows: dict[tuple[str, str, str], dict[str, str]], datasets: Iterable[str]) -> list[PanelSpec]:
    specs: list[PanelSpec] = []
    for dataset in datasets:
        for sweep in _select_sweeps(rows, dataset):
            methods = _rank_methods(rows, dataset, sweep)
            item = _select_item(rows, dataset, sweep, methods)
            crop = _select_crop(rows, dataset, sweep, item, methods)
            specs.append(PanelSpec(dataset=dataset, sweep=sweep, item=item, crop_xywh=crop, methods=methods))
    return specs


def _draw_group(group: str, specs: list[PanelSpec], rows: dict[tuple[str, str, str], dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
        }
    )

    nrows = len(specs)
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 7.2), dpi=300)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_headers = [METHOD_LABEL[method] for method in PAPER_METHODS]
    for c, header in enumerate(col_headers):
        axes[0, c].set_title(header, fontsize=16.0, pad=5)

    manifest: list[dict[str, object]] = []
    for r, spec in enumerate(specs):
        row_label = _short_row_label(spec.dataset, spec.sweep)
        axes[r, 0].set_ylabel(row_label, fontsize=16.0, rotation=90, labelpad=8, va="center")
        for c, method in enumerate(spec.methods):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("0.35")

            if method == "gt":
                img_path = _run_dir(rows, spec.dataset, spec.sweep, "ours") / "vis" / f"item_{spec.item}" / "gt.png"
                metric = ""
            else:
                img_path = _run_dir(rows, spec.dataset, spec.sweep, method) / "vis" / f"item_{spec.item}" / "pred.png"
                metric = _metric_text(_run_dir(rows, spec.dataset, spec.sweep, method), spec.item)
            img = _load_image(img_path)
            _draw_full_with_zoom(ax, img, spec.crop_xywh, METHOD_LABEL[method], metric)

        manifest.append(
            {
                "dataset": spec.dataset,
                "sweep": spec.sweep,
                "item": int(spec.item),
                "crop_xywh": [int(v) for v in spec.crop_xywh],
                "columns": spec.methods,
            }
        )

    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.025, wspace=0.035, hspace=0.08)
    pdf_path = OUT_DIR / f"paper_{group}_2d_rendering.pdf"
    png_path = OUT_DIR / f"paper_{group}_2d_rendering.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)

    manifest_path = MANIFEST_DIR / f"paper_{group}_2d_rendering_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {pdf_path}")
    print(f"[OK] wrote {png_path}")
    print(f"[OK] wrote {manifest_path}")


def main() -> None:
    rows = _read_rows()
    for group, datasets in GROUPS.items():
        specs = _build_specs(rows, datasets)
        _draw_group(group, specs, rows)


if __name__ == "__main__":
    main()
