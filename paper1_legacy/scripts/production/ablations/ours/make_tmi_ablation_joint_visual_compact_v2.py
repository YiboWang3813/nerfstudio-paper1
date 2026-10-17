#!/usr/bin/env python3
"""Build the compact TMI v2 qualitative component ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = PROJECT_ROOT / "docs/production/ablation_final_outputs"
FIGURE_DIR = OUT_ROOT / "figures"
MANIFEST_DIR = OUT_ROOT / "manifests"
SOURCE_MANIFEST = MANIFEST_DIR / "paper_ablation_joint_visual_manifest.json"
YELLOW = "#FFD54F"

VARIANTS = [
    ("Base", "Base"),
    ("+LGR", "+LG"),
    ("+LGR+CGA", "+LG+DC"),
    ("+LGR+CGA+PP", "+LG+DC+EP"),
]
DATASET_LABELS = {
    "synthetic_liver": "Synthetic Liver",
    "tus_rec_2024_qc": "TUS-REC-2024",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def _load_gray(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _read_metric(run_dir: Path, item: int) -> str:
    path = run_dir / "eval/per_frame_metrics_latest.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["item"]) == item:
                return f"PSNR {float(row['psnr']):.2f}\nSSIM {float(row['ssim']):.3f}"
    raise KeyError(f"Missing item {item} in {path}")


def _inset_bounds(crop: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = crop
    image_h, image_w = shape
    crop_box = np.array([x / image_w, y / image_h, (x + w) / image_w, (y + h) / image_h])
    candidates = [
        (0.57, 0.04, 0.40, 0.40),
        (0.03, 0.04, 0.40, 0.40),
        (0.57, 0.56, 0.40, 0.40),
        (0.03, 0.56, 0.40, 0.40),
    ]

    def overlap(candidate: tuple[float, float, float, float]) -> float:
        cx, cy, cw, ch = candidate
        candidate_box = np.array([cx, 1.0 - cy - ch, cx + cw, 1.0 - cy])
        ix0, iy0 = np.maximum(crop_box[:2], candidate_box[:2])
        ix1, iy1 = np.minimum(crop_box[2:], candidate_box[2:])
        return max(0.0, float(ix1 - ix0)) * max(0.0, float(iy1 - iy0))

    return min(candidates, key=overlap)


def _draw_panel(
    ax: plt.Axes,
    image: np.ndarray,
    crop: tuple[int, int, int, int],
    inset_bounds: tuple[float, float, float, float],
    metric: str,
) -> None:
    x, y, w, h = crop
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=YELLOW, linewidth=1.45))
    inset = ax.inset_axes(inset_bounds)
    inset.imshow(image[y : y + h, x : x + w], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(YELLOW)
        spine.set_linewidth(1.45)

    source_corners = [(x, y), (x + w, y + h)]
    inset_corners = [(0.0, 1.0), (1.0, 0.0)]
    if inset_bounds[0] < 0.5:
        source_corners = [(x, y + h), (x + w, y)]
        inset_corners = [(1.0, 1.0), (0.0, 0.0)]
    for source, target in zip(source_corners, inset_corners):
        ax.add_artist(
            ConnectionPatch(
                xyA=source,
                coordsA=ax.transData,
                xyB=target,
                coordsB=inset.transAxes,
                color=YELLOW,
                linewidth=1.05,
                clip_on=False,
            )
        )
    if metric:
        ax.text(
            0.018,
            0.018,
            metric,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.58, "pad": 1.4, "edgecolor": "none"},
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.30")
        spine.set_linewidth(0.8)


def main() -> None:
    _setup_style()
    source_rows = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if len(source_rows) != 2:
        raise ValueError(f"Expected two source rows in {SOURCE_MANIFEST}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 4.6), dpi=300)
    column_labels = [name for name, _ in VARIANTS] + ["GT"]
    for column, label in enumerate(column_labels):
        axes[0, column].set_title(label, fontsize=14.5, pad=4.0)

    panels: list[dict[str, object]] = []
    for row_index, source_row in enumerate(source_rows):
        dataset = source_row["dataset"]
        sweep = source_row["sweep"]
        item = int(source_row["item"])
        crop = tuple(int(v) for v in source_row["crop_xywh"])
        dataset_label = DATASET_LABELS[dataset]
        axes[row_index, 0].set_ylabel(dataset_label, fontsize=14.0, rotation=90, labelpad=7, va="center")
        final_old_label = VARIANTS[-1][1]
        gt_run_dir = Path(source_row["run_dirs"][final_old_label])
        gt_path = gt_run_dir / "vis" / f"item_{item}" / "gt.png"
        reference = _load_gray(gt_path)
        shared_inset = _inset_bounds(crop, reference.shape)

        for column, (new_label, old_label) in enumerate(VARIANTS):
            run_dir = Path(source_row["run_dirs"][old_label])
            source = run_dir / "vis" / f"item_{item}" / "pred.png"
            image = _load_gray(source)
            if image.shape != reference.shape:
                raise ValueError(f"Image shape mismatch: {source}")
            metric = _read_metric(run_dir, item)
            _draw_panel(axes[row_index, column], image, crop, shared_inset, metric)
            panels.append(
                {
                    "row": row_index,
                    "column": column,
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "sweep": sweep,
                    "item": item,
                    "variant": new_label,
                    "run_directory": str(run_dir.resolve()),
                    "source_image": str(source.resolve()),
                    "crop_xywh": list(crop),
                    "inset_axes_bounds": list(shared_inset),
                    "display_range": [0.0, 1.0],
                    "metric_text": metric,
                }
            )

        gt_column = 4
        _draw_panel(axes[row_index, gt_column], reference, crop, shared_inset, "")
        panels.append(
            {
                "row": row_index,
                "column": gt_column,
                "dataset": dataset,
                "dataset_label": dataset_label,
                "sweep": sweep,
                "item": item,
                "variant": "GT",
                "run_directory": str(gt_run_dir.resolve()),
                "source_image": str(gt_path.resolve()),
                "crop_xywh": list(crop),
                "inset_axes_bounds": list(shared_inset),
                "display_range": [0.0, 1.0],
                "metric_text": "",
            }
        )

    fig.subplots_adjust(left=0.09, right=0.998, top=0.91, bottom=0.02, wspace=0.025, hspace=0.07)
    pdf_path = FIGURE_DIR / "paper_ablation_joint_visual_compact.pdf"
    png_path = FIGURE_DIR / "paper_ablation_joint_visual_compact.png"
    manifest_path = MANIFEST_DIR / "paper_ablation_joint_visual_compact_manifest.json"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.015, dpi=300)
    plt.close(fig)
    manifest = {
        "version": "TMI v2",
        "source_manifest": str(SOURCE_MANIFEST.resolve()),
        "layout": {"rows": 2, "columns": 5, "figsize_inches": [12.5, 4.6], "dpi": 300},
        "abbreviations": {
            "LGR": "physics-aware local-global Gaussian renderer",
            "CGA": "contribution-aware Gaussian adaptation",
            "PP": "acquisition-guided pseudo-pose parameterization",
        },
        "panels": panels,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in (pdf_path, png_path, manifest_path):
        print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
