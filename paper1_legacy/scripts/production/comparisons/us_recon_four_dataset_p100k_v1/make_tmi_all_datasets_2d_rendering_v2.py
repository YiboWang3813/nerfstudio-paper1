#!/usr/bin/env python3
"""Build the frozen-sample TMI v2 four-dataset qualitative comparison."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_ROOT = PROJECT_ROOT / "outputs/production/comparisons/us_recon_four_dataset_p100k_v1"
TABLE_ROOT = BENCHMARK_ROOT / "tables"
OUT_ROOT = PROJECT_ROOT / "docs/production/comparison_final_outputs"
FIGURE_DIR = OUT_ROOT / "figures"
MANIFEST_DIR = OUT_ROOT / "manifests"
CONFIG_PATH = OUT_ROOT / "2d_rendering_figure_config_tmi_v2.md"

YELLOW = "#FFD54F"
SUMMARY_FILES = [
    "gs_stage2_fixed_subset_per_run.tsv",
    "gs_full_sweep_per_run.tsv",
    "gs_ultragauss_tus_qc_chi2_5500_replacement.tsv",
    "implicit_full_sweep_per_run.tsv",
]
METHODS = [
    ("ultranerf", "Ultra-NeRF"),
    ("nerfus_no_prior", "NeRF-US"),
    ("ultragauss", "UltraGauss"),
    ("ultragray", "UltraG-Ray"),
    ("ultrags_baseline", "UltraGS"),
    ("slice_adapted_3dgs", "3DGS"),
    ("gaussianpile", "GaussianPile"),
    ("ours", "ULGSplat"),
    ("gt", "GT"),
]


@dataclass(frozen=True)
class RowSpec:
    label: str
    dataset: str
    sweep: str
    item: int
    crop: tuple[int, int, int, int]


ROWS = [
    RowSpec("Liver", "synthetic_liver", "l3", 27, (19, 0, 53, 107)),
    RowSpec("Spine", "spine_phantom", "left3_2", 17, (59, 0, 36, 215)),
    RowSpec(
        "Arm",
        "tus_rec_2024_qc",
        "010_RH_Per_S_PtD_qc_0165_0364_s02",
        18,
        (129, 24, 92, 69),
    ),
    RowSpec("Breast", "tdsc_abus_2023", "validation_116", 39, (150, 23, 85, 67)),
]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
        }
    )


def _read_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for name in SUMMARY_FILES:
        with (TABLE_ROOT / name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows[(row["dataset"], row["sweep"], row["method"])] = row
    return rows


def _load_gray(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


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
) -> None:
    x, y, w, h = crop
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=YELLOW, linewidth=2.0))
    inset = ax.inset_axes(inset_bounds)
    inset.imshow(image[y : y + h, x : x + w], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(YELLOW)
        spine.set_linewidth(2.0)

    ix = inset_bounds[0]
    source_corners = [(x, y), (x + w, y + h)]
    inset_corners = [(0.0, 1.0), (1.0, 0.0)]
    if ix < 0.5:
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
                linewidth=1.55,
                clip_on=False,
            )
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.30")
        spine.set_linewidth(0.9)


def main() -> None:
    _setup_style()
    summary = _read_rows()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 9, figsize=(18.0, 8.0), dpi=300)
    manifest_panels: list[dict[str, object]] = []

    for column, (_, label) in enumerate(METHODS):
        axes[0, column].set_title(label, fontsize=18.5, pad=5.0)

    for row_index, spec in enumerate(ROWS):
        axes[row_index, 0].set_ylabel(spec.label, fontsize=18.0, rotation=90, labelpad=8, va="center")
        ours_row = summary[(spec.dataset, spec.sweep, "ours")]
        ours_dir = Path(ours_row["output_dir"])
        gt_path = ours_dir / "vis" / f"item_{spec.item}" / "gt.png"
        reference = _load_gray(gt_path)
        shared_inset = _inset_bounds(spec.crop, reference.shape)

        for column, (method, label) in enumerate(METHODS):
            if method == "gt":
                run_dir = ours_dir
                source = gt_path
            else:
                row = summary.get((spec.dataset, spec.sweep, method))
                if row is None:
                    raise KeyError(f"Missing summary row for {spec.dataset}/{spec.sweep}/{method}")
                run_dir = Path(row["output_dir"])
                source = run_dir / "vis" / f"item_{spec.item}" / "pred.png"
            image = _load_gray(source)
            if image.shape != reference.shape:
                raise ValueError(f"Image shape mismatch in {spec.dataset}: {source} has {image.shape}, expected {reference.shape}")
            _draw_panel(axes[row_index, column], image, spec.crop, shared_inset)
            manifest_panels.append(
                {
                    "row": row_index,
                    "column": column,
                    "dataset": spec.dataset,
                    "dataset_label": spec.label,
                    "sweep": spec.sweep,
                    "item": spec.item,
                    "method": method,
                    "method_label": label,
                    "run_directory": str(run_dir.resolve()),
                    "source_image": str(source.resolve()),
                    "crop_xywh": list(spec.crop),
                    "inset_axes_bounds": list(shared_inset),
                    "display_range": [0.0, 1.0],
                }
            )

    fig.subplots_adjust(left=0.066, right=0.998, top=0.945, bottom=0.012, wspace=0.025, hspace=0.06)
    pdf_path = FIGURE_DIR / "paper_all_datasets_2d_rendering.pdf"
    png_path = FIGURE_DIR / "paper_all_datasets_2d_rendering.png"
    manifest_path = MANIFEST_DIR / "paper_all_datasets_2d_rendering_manifest.json"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    manifest = {
        "version": "TMI v2",
        "layout": {"rows": 4, "columns": 9, "figsize_inches": [18.0, 8.0], "dpi": 300},
        "summary_files": [str((TABLE_ROOT / name).resolve()) for name in SUMMARY_FILES],
        "panels": manifest_panels,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    CONFIG_PATH.write_text(
        "# TMI v2 Four-Dataset 2D Rendering Figure\n\n"
        "Status: generated and frozen-sample.\n\n"
        "This version preserves the original frozen figures and uses the four fixed samples specified in "
        "`docs/production/new_comparison_ablation_figures/readme.md`.\n\n"
        "## Regeneration\n\n"
        "```bash\n"
        "MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python "
        "scripts/production/comparisons/us_recon_four_dataset_p100k_v1/make_tmi_all_datasets_2d_rendering_v2.py\n"
        "```\n\n"
        "## Outputs\n\n"
        "- `figures/paper_all_datasets_2d_rendering.pdf`\n"
        "- `figures/paper_all_datasets_2d_rendering.png`\n"
        "- `manifests/paper_all_datasets_2d_rendering_manifest.json`\n\n"
        "Layout: 4 datasets by 8 methods plus GT. Images use a shared `[0, 1]` grayscale display range and "
        "one shared crop/inset placement within each row. No metrics or sweep identifiers are overlaid. "
        "Compact row labels are `Liver`, `Spine`, `Arm`, and `Breast`; full dataset ids remain "
        "recorded in the manifest.\n",
        encoding="utf-8",
    )
    for path in (pdf_path, png_path, manifest_path, CONFIG_PATH):
        print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
