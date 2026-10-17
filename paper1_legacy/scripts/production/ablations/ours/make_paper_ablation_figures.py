#!/usr/bin/env python3
"""Generate paper-facing Ours ablation figures.

This script intentionally consumes only completed ablation outputs: rendered
2D frames, per-frame metrics, and summary tables. It does not require or write
training checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path("outputs/production/ablations/ours")
EXPANDED = ROOT / "expanded_paper_plan_20260702"
OUT_ROOT = Path("docs/production/ablation_final_outputs")
FIG_DIR = OUT_ROOT / "figures"
MANIFEST_DIR = OUT_ROOT / "manifests"
CONFIG_PATH = OUT_ROOT / "2d_ablation_figure_config.md"

STD_SYN = ("synthetic_liver", "l1")
STD_REAL = ("tus_rec_2024_qc", "010_LH_Per_L_DtP_qc_0005_0370")

YELLOW = "#ffd84d"
BLUE = "#4c78a8"
ORANGE = "#f58518"
GREEN = "#54a24b"
RED = "#e45756"
PURPLE = "#b279a2"


@dataclass(frozen=True)
class Variant:
    label: str
    run_dir: Path


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _read_summary(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[(row["table"], row["dataset"], row["sweep"], row["variant"])] = row
    return rows


SUMMARY = _read_summary(ROOT / "summary.tsv")
EXP_SUMMARY = _read_summary(EXPANDED / "summary.tsv")


def _summary_run(table: str, dataset: str, sweep: str, variant: str, expanded: bool = False) -> Path:
    source = EXP_SUMMARY if expanded else SUMMARY
    key = (table, dataset, sweep, variant)
    if key not in source:
        raise KeyError(f"Missing summary row: {key}")
    return Path(source[key]["run_dir"])


def _exp_run(table_dir: str, dataset: str, sweep: str, variant: str) -> Path:
    return EXPANDED / "runs" / table_dir / dataset / sweep / variant


def _available_items(run_dir: Path) -> set[int]:
    items: set[int] = set()
    for path in (run_dir / "vis").glob("item_*"):
        if not path.is_dir() or not (path / "pred.png").exists():
            continue
        try:
            items.add(int(path.name.split("_", 1)[1]))
        except ValueError:
            pass
    return items


def _read_per_frame(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "eval" / "per_frame_metrics_latest.tsv"
    out: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item = int(row["item"])
            out[item] = {"psnr": float(row["psnr"]), "ssim": float(row["ssim"])}
    return out


def _metric_text(run_dir: Path, item: int) -> str:
    metric = _read_per_frame(run_dir)[item]
    return f"PSNR {metric['psnr']:.2f}\nSSIM {metric['ssim']:.3f}"


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _diff_image(run_dir: Path, item: int) -> np.ndarray:
    diff = run_dir / "vis" / f"item_{item}" / "diff.png"
    if diff.exists():
        return _load_gray(diff)
    pred = _load_gray(run_dir / "vis" / f"item_{item}" / "pred.png")
    gt = _load_gray(run_dir / "vis" / f"item_{item}" / "gt.png")
    return np.abs(pred - gt)


def _crop_array(img: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = crop
    return img[y : y + h, x : x + w]


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
        cand_box = np.array([cx, 1.0 - cy - ch, cx + cw, 1.0 - cy], dtype=np.float32)
        ix0 = max(float(crop_box[0]), float(cand_box[0]))
        iy0 = max(float(crop_box[1]), float(cand_box[1]))
        ix1 = min(float(crop_box[2]), float(cand_box[2]))
        iy1 = min(float(crop_box[3]), float(cand_box[3]))
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
    metric: str = "",
    cmap: str = "gray",
) -> None:
    x, y, w, h = crop
    ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=YELLOW, linewidth=0.85))

    ix, iy, iw, ih = _inset_bounds(crop, img.shape)
    inset = ax.inset_axes((ix, iy, iw, ih))
    inset.imshow(_crop_array(img, crop), cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(YELLOW)
        spine.set_linewidth(0.9)

    rect_corners = [(x, y), (x + w, y + h)]
    inset_corners = [(0.0, 1.0), (1.0, 0.0)]
    if ix < 0.5:
        rect_corners = [(x, y + h), (x + w, y)]
        inset_corners = [(1.0, 1.0), (0.0, 0.0)]
    for xy_a, xy_b in zip(rect_corners, inset_corners):
        ax.add_artist(
            ConnectionPatch(
                xyA=xy_a,
                coordsA=ax.transData,
                xyB=xy_b,
                coordsB=inset.transAxes,
                color=YELLOW,
                linewidth=0.55,
                alpha=0.95,
                clip_on=False,
            )
        )

    if metric:
        tx, ty, tha, tva = _text_anchor(crop, img.shape, (ix, iy, iw, ih))
        ax.text(
            tx,
            ty,
            metric,
            transform=ax.transAxes,
            ha=tha,
            va=tva,
            fontsize=7.6,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.58, "pad": 1.25, "edgecolor": "none"},
        )


def _select_item(variants: list[Variant]) -> int:
    common: set[int] | None = None
    metrics: dict[str, dict[int, dict[str, float]]] = {}
    for variant in variants:
        items = _available_items(variant.run_dir)
        common = items if common is None else common & items
        metrics[variant.label] = _read_per_frame(variant.run_dir)
    if not common:
        raise RuntimeError(f"No common visual frames for {[str(v.run_dir) for v in variants]}")

    best_item = sorted(common)[0]
    best_score = -float("inf")
    final = variants[-1].label
    for item in sorted(common):
        final_metrics = metrics[final].get(item)
        if final_metrics is None:
            continue
        other_psnr = [metrics[v.label][item]["psnr"] for v in variants[:-1] if item in metrics[v.label]]
        other_ssim = [metrics[v.label][item]["ssim"] for v in variants[:-1] if item in metrics[v.label]]
        if not other_psnr:
            continue
        score = (final_metrics["psnr"] - max(other_psnr)) + 5.0 * (final_metrics["ssim"] - max(other_ssim))
        if score > best_score:
            best_item = item
            best_score = score
    return best_item


def _select_crop(variants: list[Variant], item: int) -> tuple[int, int, int, int]:
    gt = _load_gray(variants[-1].run_dir / "vis" / f"item_{item}" / "gt.png")
    h, w = gt.shape
    crop_w = min(max(36, int(w * 0.24)), w)
    crop_h = min(max(36, int(h * 0.24)), h)
    final = _load_gray(variants[-1].run_dir / "vis" / f"item_{item}" / "pred.png")
    base_errors = [
        np.abs(_load_gray(v.run_dir / "vis" / f"item_{item}" / "pred.png") - gt)
        for v in variants[:-1]
    ]
    mean_base_error = np.mean(np.stack(base_errors, axis=0), axis=0)
    final_error = np.abs(final - gt)
    texture = np.abs(gt - gt.mean())
    score_map = mean_base_error - final_error + 0.15 * texture

    best = (-float("inf"), 0, 0)
    for yy in np.linspace(0, max(0, h - crop_h), 10).astype(int):
        for xx in np.linspace(0, max(0, w - crop_w), 10).astype(int):
            score = float(np.mean(score_map[yy : yy + crop_h, xx : xx + crop_w]))
            if score > best[0]:
                best = (score, xx, yy)
    return best[1], best[2], crop_w, crop_h


def _prepare_axes(axes: np.ndarray) -> None:
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("0.35")


def _joint_variants(dataset: str, sweep: str) -> list[Variant]:
    return [
        Variant("Base", _summary_run("t02", dataset, sweep, "local_gradient")),
        Variant("+LG", _summary_run("t02", dataset, sweep, "localglobal_gradient")),
        Variant("+LG+DC", _summary_run("t02", dataset, sweep, "localglobal_responsibility")),
        Variant("+LG+DC+EP", _summary_run("t06", dataset, sweep, "canonical_sweep")),
    ]


def _make_joint_visual() -> None:
    rows = [(STD_SYN, "Synthetic"), (STD_REAL, "TUS-QC")]
    col_labels = ["Base", "+LG", "+LG+DC", "+LG+DC+EP", "GT"]
    fig, axes = plt.subplots(4, 5, figsize=(10.5, 7.2), dpi=300)
    _prepare_axes(axes)
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=15, pad=5)

    manifest: list[dict[str, object]] = []
    for group_idx, ((dataset, sweep), row_label) in enumerate(rows):
        variants = _joint_variants(dataset, sweep)
        item = _select_item(variants)
        crop = _select_crop(variants, item)
        pred_row = group_idx * 2
        err_row = pred_row + 1
        axes[pred_row, 0].set_ylabel(f"{row_label}\nPred", fontsize=13, rotation=90, labelpad=7, va="center")
        axes[err_row, 0].set_ylabel(f"{row_label}\nError", fontsize=13, rotation=90, labelpad=7, va="center")

        for c, variant in enumerate(variants):
            pred = _load_gray(variant.run_dir / "vis" / f"item_{item}" / "pred.png")
            _draw_full_with_zoom(axes[pred_row, c], pred, crop, _metric_text(variant.run_dir, item))
            _draw_full_with_zoom(axes[err_row, c], _diff_image(variant.run_dir, item), crop, "")

        gt = _load_gray(variants[-1].run_dir / "vis" / f"item_{item}" / "gt.png")
        _draw_full_with_zoom(axes[pred_row, 4], gt, crop, "")
        _draw_full_with_zoom(axes[err_row, 4], gt, crop, "")
        manifest.append(
            {
                "dataset": dataset,
                "sweep": sweep,
                "item": item,
                "crop_xywh": list(map(int, crop)),
                "columns": col_labels,
                "run_dirs": {v.label: str(v.run_dir) for v in variants},
            }
        )

    fig.subplots_adjust(left=0.08, right=0.995, top=0.935, bottom=0.025, wspace=0.035, hspace=0.08)
    _save_figure(fig, "paper_ablation_joint_visual", manifest)


def _internal_specs(dataset: str, sweep: str) -> list[tuple[str, list[Variant]]]:
    if dataset == "tus_rec_2024_qc":
        resp_densify = _summary_run("t05", dataset, sweep, "responsibility_densify_only")
        resp_prune = _summary_run("t05", dataset, sweep, "responsibility_no_attenuation_protect")
    else:
        resp_densify = _exp_run("t03_internal", dataset, sweep, "responsibility_densify_only")
        resp_prune = _exp_run("t03_internal", dataset, sweep, "responsibility_no_attenuation_protect")
    return [
        (
            "Renderer",
            [
                Variant("Local", _summary_run("t02", dataset, sweep, "local_responsibility")),
                Variant("Simple", _exp_run("t02_internal", dataset, sweep, "simple_global_attenuation")),
                Variant("Integral-1", _exp_run("t02_internal", dataset, sweep, "integral_attenuation_q1")),
                Variant("Integral-3", _summary_run("t02", dataset, sweep, "localglobal_responsibility")),
            ],
        ),
        (
            "DC",
            [
                Variant("Grad", _summary_run("t02", dataset, sweep, "localglobal_gradient")),
                Variant("Resp-D", resp_densify),
                Variant("Resp-P", resp_prune),
                Variant("Full DC", _summary_run("t02", dataset, sweep, "localglobal_responsibility")),
            ],
        ),
        (
            "Easy-Pose",
            [
                Variant("File", _summary_run("t06", dataset, sweep, "file_pose")),
                Variant("Step", _exp_run("t04_internal", dataset, sweep, "canonical_file_step_norm")),
                Variant("Axis", _exp_run("t04_internal", dataset, sweep, "canonical_file_axis_projection")),
                Variant("Uniform", _summary_run("t06", dataset, sweep, "canonical_sweep")),
            ],
        ),
    ]


def _make_internal_visual() -> None:
    row_specs: list[tuple[str, str, str, list[Variant]]] = []
    for group, variants in _internal_specs(*STD_SYN):
        row_specs.append((group, "Synthetic", STD_SYN[0], variants))
    for group, variants in _internal_specs(*STD_REAL):
        row_specs.append((group, "TUS-QC", STD_REAL[0], variants))
    # Interleave by module as documented: renderer syn/real, DC syn/real, EP syn/real.
    row_specs = [
        ("Renderer", "Synthetic", STD_SYN[0], _internal_specs(*STD_SYN)[0][1]),
        ("Renderer", "TUS-QC", STD_REAL[0], _internal_specs(*STD_REAL)[0][1]),
        ("DC", "Synthetic", STD_SYN[0], _internal_specs(*STD_SYN)[1][1]),
        ("DC", "TUS-QC", STD_REAL[0], _internal_specs(*STD_REAL)[1][1]),
        ("Easy-Pose", "Synthetic", STD_SYN[0], _internal_specs(*STD_SYN)[2][1]),
        ("Easy-Pose", "TUS-QC", STD_REAL[0], _internal_specs(*STD_REAL)[2][1]),
    ]

    fig, axes = plt.subplots(6, 5, figsize=(10.5, 10.8), dpi=300)
    _prepare_axes(axes)
    for c, label in enumerate(["A", "B", "C", "D", "GT"]):
        axes[0, c].set_title(label, fontsize=15, pad=5)

    manifest: list[dict[str, object]] = []
    for r, (group, dataset_label, _dataset_key, variants) in enumerate(row_specs):
        axes[r, 0].set_ylabel(f"{group}\n{dataset_label}", fontsize=12, rotation=90, labelpad=7, va="center")
        item = _select_item(variants)
        crop = _select_crop(variants, item)
        for c, variant in enumerate(variants):
            if r == 0:
                axes[r, c].set_title(variant.label, fontsize=12.5, pad=4)
            pred = _load_gray(variant.run_dir / "vis" / f"item_{item}" / "pred.png")
            _draw_full_with_zoom(axes[r, c], pred, crop, _metric_text(variant.run_dir, item))
            if r in {2, 4}:
                axes[r, c].text(
                    0.5,
                    0.985,
                    variant.label,
                    transform=axes[r, c].transAxes,
                    ha="center",
                    va="top",
                    fontsize=9.0,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.0, "edgecolor": "none"},
                )
        gt = _load_gray(variants[-1].run_dir / "vis" / f"item_{item}" / "gt.png")
        _draw_full_with_zoom(axes[r, 4], gt, crop, "")
        manifest.append(
            {
                "group": group,
                "dataset_label": dataset_label,
                "item": item,
                "crop_xywh": list(map(int, crop)),
                "columns": [v.label for v in variants] + ["GT"],
                "run_dirs": {v.label: str(v.run_dir) for v in variants},
            }
        )

    fig.subplots_adjust(left=0.085, right=0.995, top=0.955, bottom=0.025, wspace=0.035, hspace=0.08)
    _save_figure(fig, "paper_ablation_internal_visual", manifest)


def _save_figure(fig: plt.Figure, stem: str, manifest: object) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    manifest_path = MANIFEST_DIR / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {pdf}")
    print(f"[OK] wrote {png}")
    print(f"[OK] wrote {manifest_path}")


def _parse_md_table(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("|")]
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        values = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(headers, values)))
    return rows


def _make_rho_curve() -> None:
    data = {
        "Synthetic": [
            (0.60, 30.022, 0.273),
            (0.75, 29.936, 0.260),
            (0.90, 29.826, 0.250),
            (1.00, 29.747, 0.244),
            (1.10, 29.677, 0.241),
            (1.25, 29.554, 0.234),
            (1.50, 29.304, 0.230),
        ],
        "TUS-QC": [
            (0.60, 32.771, 0.153),
            (0.75, 32.884, 0.147),
            (0.90, 32.950, 0.143),
            (1.00, 33.001, 0.141),
            (1.10, 33.037, 0.141),
            (1.25, 33.068, 0.140),
            (1.50, 33.058, 0.141),
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), dpi=300, sharex=True)
    for label, values in data.items():
        xs = [v[0] for v in values]
        psnr = [v[1] for v in values]
        lpips = [v[2] for v in values]
        color = BLUE if label == "Synthetic" else ORANGE
        axes[0].plot(xs, psnr, marker="o", linewidth=1.7, markersize=4.0, label=label, color=color)
        axes[1].plot(xs, lpips, marker="o", linewidth=1.7, markersize=4.0, label=label, color=color)
        best_idx = int(np.argmax(psnr))
        axes[0].scatter([xs[best_idx]], [psnr[best_idx]], s=58, facecolors="none", edgecolors=color, linewidths=1.6)
    for ax in axes:
        ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.set_xlabel("rho")
    axes[0].set_ylabel("PSNR")
    axes[1].set_ylabel("LPIPS")
    axes[0].set_title("A. Reconstruction fidelity")
    axes[1].set_title("B. Perceptual error")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.6, w_pad=1.5)
    _save_figure(fig, "paper_easy_pose_rho_sensitivity", data)


def _make_renderer_sensitivity() -> None:
    rows = _parse_md_table(EXPANDED / "tables" / "t14_renderer_sensitivity_delta_psnr.md")
    params = sorted({r["Parameter"] for r in rows})
    order = sorted(
        params,
        key=lambda p: max(abs(float(r["Real dPSNR"])) for r in rows if r["Parameter"] == p),
        reverse=True,
    )
    colors = {
        "strength": BLUE,
        "beta": ORANGE,
        "min_transmission": GREEN,
        "quadrature_points": RED,
        "candidate_k_sigma": PURPLE,
    }
    defaults = {
        "strength": "0.55",
        "beta": "0.80",
        "min_transmission": "0.35",
        "quadrature_points": "3",
        "candidate_k_sigma": "2.0",
    }
    fig = plt.figure(figsize=(9.6, 3.6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.95], wspace=0.36)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    summary_ax = fig.add_subplot(gs[0, 2])
    summary_ax.axis("off")

    for ax, metric, title in [(axes[0], "Syn dPSNR", "A. Synthetic"), (axes[1], "Real dPSNR", "B. TUS-QC")]:
        for y, param in enumerate(order):
            vals = [(r["Value"], float(r[metric])) for r in rows if r["Parameter"] == param]
            ds = [v[1] for v in vals]
            color = colors[param]
            ax.hlines(y, min(ds), max(ds), color=color, linewidth=2.0, alpha=0.75)
            ax.scatter(ds, [y] * len(ds), s=22, color=color, zorder=3)
            default_val = defaults[param]
            for value, delta in vals:
                if value == default_val:
                    ax.scatter([delta], [y], marker="*", s=110, color="black", zorder=4)
        ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_pretty_param(p) for p in order])
        ax.invert_yaxis()
        ax.grid(True, axis="x", color="0.88", linewidth=0.55)
        ax.set_xlabel("Delta PSNR vs default")
        ax.set_title(title)

    summary_lines = ["C. Default and best", ""]
    for param in order:
        vals = [r for r in rows if r["Parameter"] == param]
        best = max(vals, key=lambda r: float(r["Real dPSNR"]))
        span = max(abs(float(r["Real dPSNR"])) for r in vals)
        summary_lines.append(f"{_pretty_param(param)}")
        summary_lines.append(f"  default {defaults[param]}, best {best['Value']}")
        summary_lines.append(f"  max |delta| {span:.3f} dB")
    summary_ax.text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", fontsize=8.3)
    fig.tight_layout(pad=0.5)
    _save_figure(fig, "paper_renderer_parameter_sensitivity", rows)


def _pretty_param(name: str) -> str:
    return {
        "strength": "strength",
        "beta": "beta",
        "min_transmission": "min trans.",
        "quadrature_points": "quad. points",
        "candidate_k_sigma": "candidate k",
    }.get(name, name)


def _make_dc_sensitivity() -> None:
    rows = _parse_md_table(EXPANDED / "tables" / "t15_dc_sensitivity_delta_psnr.md")
    all_delta = [float(r["Syn dPSNR"]) for r in rows] + [float(r["Real dPSNR"]) for r in rows]
    vmax = max(0.03, max(abs(v) for v in all_delta))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(10.4, 5.8), dpi=300)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.9], height_ratios=[1.0, 1.2], wspace=0.36, hspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    im = _plot_densify_heatmap(ax_a, rows, "Syn dPSNR", norm, "A. Densify, synthetic")
    _plot_densify_heatmap(ax_b, rows, "Real dPSNR", norm, "B. Densify, TUS-QC")
    _plot_opacity_panel(ax_c, rows, norm)
    _plot_prune_heatmap(ax_d, rows, "Syn dPSNR", norm, "D. Prune protect, synthetic")
    _plot_prune_heatmap(ax_e, rows, "Real dPSNR", norm, "E. Prune protect, TUS-QC")

    fig.subplots_adjust(left=0.075, right=0.84, top=0.94, bottom=0.12, wspace=0.36, hspace=0.42)
    cbar_ax = fig.add_axes((0.875, 0.16, 0.025, 0.74))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Delta PSNR vs default")
    ax_f.text(
        0.0,
        1.0,
        "F. Reading guide\n\n"
        "Black star/box: production default.\n"
        "Yellow box: best cell in panel.\n"
        "Shared color: delta PSNR.\n\n"
        "Defaults:\n"
        "score=0.98, max_fraction=0.05\n"
        "low_q=0.10, protect_q=0.80\n"
        "prune_opacity=1e-4",
        va="top",
        ha="left",
        fontsize=8.2,
    )
    _save_figure(fig, "paper_dc_parameter_sensitivity", rows)


def _plot_densify_heatmap(ax: plt.Axes, rows: list[dict[str, str]], metric: str, norm: TwoSlopeNorm, title: str):
    subset = [r for r in rows if r["Group"] == "densify"]
    scores = [0.94, 0.96, 0.98, 0.99]
    fracs = [0.02, 0.05, 0.08, 0.12]
    grid = np.full((len(scores), len(fracs)), np.nan)
    for row in subset:
        match = re.match(r"score=([0-9.]+), max_fraction=([0-9.]+)", row["Setting"])
        if not match:
            continue
        score = float(match.group(1))
        frac = float(match.group(2))
        grid[scores.index(score), fracs.index(frac)] = float(row[metric])
    im = ax.imshow(grid, cmap="coolwarm", norm=norm, aspect="auto")
    _mark_heatmap(ax, grid, scores.index(0.98), fracs.index(0.05))
    ax.set_xticks(range(len(fracs)), [f"{v:.2f}" for v in fracs])
    ax.set_yticks(range(len(scores)), [f"{v:.2f}" for v in scores])
    ax.set_xlabel("max fraction")
    ax.set_ylabel("score quantile")
    ax.set_title(title)
    return im


def _plot_prune_heatmap(ax: plt.Axes, rows: list[dict[str, str]], metric: str, norm: TwoSlopeNorm, title: str):
    subset = [r for r in rows if r["Group"] == "prune_protect"]
    lows = [0.02, 0.05, 0.10, 0.15, 0.20]
    protects = [0.60, 0.70, 0.80, 0.90]
    grid = np.full((len(lows), len(protects)), np.nan)
    for row in subset:
        match = re.match(r"low_q=([0-9.]+), protect_q=([0-9.]+)", row["Setting"])
        if not match:
            continue
        low = float(match.group(1))
        protect = float(match.group(2))
        grid[lows.index(low), protects.index(protect)] = float(row[metric])
    im = ax.imshow(grid, cmap="coolwarm", norm=norm, aspect="auto")
    _mark_heatmap(ax, grid, lows.index(0.10), protects.index(0.80))
    ax.set_xticks(range(len(protects)), [f"{v:.2f}" for v in protects])
    ax.set_yticks(range(len(lows)), [f"{v:.2f}" for v in lows])
    ax.set_xlabel("protect attenuation q")
    ax.set_ylabel("low contribution q")
    ax.set_title(title)
    return im


def _mark_heatmap(ax: plt.Axes, grid: np.ndarray, default_y: int, default_x: int) -> None:
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.add_patch(Rectangle((default_x - 0.5, default_y - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.6))
    ax.scatter([default_x], [default_y], marker="*", s=80, color="black", zorder=4)
    ax.add_patch(Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=YELLOW, linewidth=1.8))
    for y, x in [(default_y, default_x), (best[0], best[1])]:
        ax.text(x, y, f"{grid[y, x]:+.3f}", ha="center", va="center", fontsize=7.4, color="white")


def _plot_opacity_panel(ax: plt.Axes, rows: list[dict[str, str]], norm: TwoSlopeNorm) -> None:
    subset = [r for r in rows if r["Group"] == "prune_opacity"]
    xs = [float(r["Setting"]) for r in subset]
    syn = [float(r["Syn dPSNR"]) for r in subset]
    real = [float(r["Real dPSNR"]) for r in subset]
    ax.scatter(xs, syn, c=syn, cmap="coolwarm", norm=norm, marker="o", s=45, label="Synthetic")
    ax.scatter(xs, real, c=real, cmap="coolwarm", norm=norm, marker="s", s=45, label="TUS-QC")
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    ax.axvline(1e-4, color="black", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.grid(True, color="0.88", linewidth=0.55)
    ax.set_xlabel("prune opacity")
    ax.set_ylabel("Delta PSNR")
    ax.set_title("C. Opacity pruning")
    ax.legend(frameon=False, loc="best")


def _write_config() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        "# Ours Ablation Figure Generation Config\n\n"
        "Status: generated\n\n"
        "Script:\n\n"
        "```text\n"
        "scripts/production/ablations/ours/make_paper_ablation_figures.py\n"
        "```\n\n"
        "Regeneration command:\n\n"
        "```bash\n"
        "MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python "
        "scripts/production/ablations/ours/make_paper_ablation_figures.py\n"
        "```\n\n"
        "Output root:\n\n"
        "```text\n"
        "docs/production/ablation_final_outputs\n"
        "```\n\n"
        "Generated figures: T11-A, T11-B, T12, T14, and T15. T13 3D visualization is intentionally deferred.\n",
        encoding="utf-8",
    )


def main() -> None:
    _setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    _make_joint_visual()
    _make_internal_visual()
    _make_rho_curve()
    _make_renderer_sensitivity()
    _make_dc_sensitivity()
    _write_config()
    print(f"[OK] wrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()
