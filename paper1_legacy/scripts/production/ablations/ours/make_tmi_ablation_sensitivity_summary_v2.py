#!/usr/bin/env python3
"""Build the unified TMI v2 parameter-sensitivity figure from frozen manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = PROJECT_ROOT / "docs/production/ablation_final_outputs"
FIGURE_DIR = OUT_ROOT / "figures"
MANIFEST_DIR = OUT_ROOT / "manifests"
RHO_SOURCE = MANIFEST_DIR / "paper_easy_pose_rho_sensitivity_manifest.json"
RENDERER_SOURCE = MANIFEST_DIR / "paper_renderer_parameter_sensitivity_manifest.json"
ADAPT_SOURCE = MANIFEST_DIR / "paper_dc_parameter_sensitivity_manifest.json"
CONFIG_PATH = OUT_ROOT / "ablation_figure_config_tmi_v2.md"

BLUE = "#4C78A8"
ORANGE = "#F28E2B"
YELLOW = "#FFD54F"
CMAP = LinearSegmentedColormap.from_list("blue_gray_orange", ["#3B6FB6", "#F0F0F0", "#E9803A"])
PDF_METADATA = {"Creator": "ULGSplat TMI figure generator", "CreationDate": None, "ModDate": None}

PARAMETERS = [
    ("strength", r"$\lambda_{\mathrm{ray}}$", "0.55"),
    ("beta", r"$\beta$", "0.80"),
    ("min_transmission", r"$T_{\min}$", "0.35"),
    ("quadrature_points", r"$N_q$", "3"),
    ("candidate_k_sigma", r"$\kappa_{\mathrm{glo}}$", "2.0"),
]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 15.5,
            "axes.labelsize": 14.8,
            "xtick.labelsize": 14.8,
            "ytick.labelsize": 14.8,
            "legend.fontsize": 14.8,
            "axes.linewidth": 0.8,
        }
    )


def _signed_cell(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def _plot_spacing(ax: plt.Axes, source: dict[str, list[list[float]]]) -> dict[str, object]:
    plotted: dict[str, object] = {}
    styles = [
        ("Synthetic", "Synthetic Liver", BLUE, "o", "-"),
        ("TUS-QC", "TUS-REC-2024", ORANGE, "s", "--"),
    ]
    for source_key, label, color, marker, linestyle in styles:
        values = source[source_key]
        rho = np.asarray([row[0] for row in values], dtype=float)
        psnr = np.asarray([row[1] for row in values], dtype=float)
        default_index = int(np.flatnonzero(np.isclose(rho, 1.0))[0])
        delta = psnr - psnr[default_index]
        ax.plot(rho, delta, color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=5.5, label=label)
        best = int(np.argmax(psnr))
        ax.scatter([rho[best]], [delta[best]], s=105, facecolors="none", edgecolors=color, linewidths=1.8, zorder=5)
        plotted[label] = {
            "rho": rho.tolist(),
            "absolute_psnr": psnr.tolist(),
            "delta_psnr": delta.tolist(),
            "default_rho": 1.0,
            "best_rho": float(rho[best]),
        }
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.1)
    ax.text(1.0, 0.97, "Default", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=14.8, color="0.25")
    ax.axhline(0.0, color="0.55", linewidth=0.7)
    ax.grid(True, color="0.88", linewidth=0.55)
    ax.set_xlabel(r"Spacing scale $\rho$")
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("(a) Pseudo-pose spacing")
    ax.legend(frameon=False, loc="lower left", handlelength=1.2, handletextpad=0.45, labelspacing=0.35)
    return plotted


def _plot_renderer(ax: plt.Axes, rows: list[dict[str, str]]) -> dict[str, object]:
    plotted: dict[str, object] = {}
    for y, (field, label, default) in enumerate(PARAMETERS):
        subset = [row for row in rows if row["Parameter"] == field]
        if not subset:
            raise KeyError(f"Missing renderer parameter {field}")
        syn = np.asarray([float(row["Syn dPSNR"]) for row in subset])
        real = np.asarray([float(row["Real dPSNR"]) for row in subset])
        for values, offset, color, marker in [(syn, -0.12, BLUE, "o"), (real, 0.12, ORANGE, "s")]:
            yy = y + offset
            ax.hlines(yy, float(values.min()), float(values.max()), color=color, linewidth=1.6, alpha=0.8)
            ax.scatter(values, np.full_like(values, yy), color=color, marker=marker, s=31, zorder=3)
        plotted[field] = {
            "display_label": label,
            "tested_values": [row["Value"] for row in subset],
            "default_value": default,
            "synthetic_liver_delta_psnr": syn.tolist(),
            "tus_rec_2024_delta_psnr": real.tolist(),
        }
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.1)
    ax.grid(True, axis="x", color="0.88", linewidth=0.55)
    ax.set_yticks(range(len(PARAMETERS)), [label for _, label, _ in PARAMETERS])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$PSNR (dB)")
    ax.set_title("(b) Local-global renderer")
    return plotted


def _adaptation_grid(rows: list[dict[str, str]], metric: str) -> tuple[np.ndarray, list[float], list[float]]:
    q_values = [0.94, 0.96, 0.98, 0.99]
    budgets = [0.02, 0.05, 0.08, 0.12]
    grid = np.full((len(q_values), len(budgets)), np.nan)
    for row in rows:
        if row["Group"] != "densify":
            continue
        match = re.fullmatch(r"score=([0-9.]+), max_fraction=([0-9.]+)", row["Setting"])
        if not match:
            raise ValueError(f"Unrecognized densification setting: {row['Setting']}")
        q_d, budget = map(float, match.groups())
        grid[q_values.index(q_d), budgets.index(budget)] = float(row[metric])
    if np.isnan(grid).any():
        raise ValueError(f"Incomplete densification grid for {metric}")
    return grid, q_values, budgets


def _plot_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    q_values: list[float],
    budgets: list[float],
    title: str,
    norm: TwoSlopeNorm,
) -> None:
    ax.imshow(grid, cmap=CMAP, norm=norm, aspect="auto")
    default_y = q_values.index(0.98)
    default_x = budgets.index(0.05)
    best_y, best_x = np.unravel_index(np.argmax(grid), grid.shape)
    ax.scatter([default_x - 0.31], [default_y - 0.31], marker="*", s=72, color="black", zorder=4)
    ax.add_patch(Rectangle((best_x - 0.5, best_y - 0.5), 1, 1, fill=False, edgecolor=YELLOW, linewidth=2.0))
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            color = "white" if abs(grid[y, x]) >= 0.013 else "black"
            ax.text(x, y, _signed_cell(float(grid[y, x])), ha="center", va="center", fontsize=14.8, color=color)
    ax.set_xticks(range(len(budgets)), [f"{value:.2f}" for value in budgets])
    ax.set_yticks(range(len(q_values)), [f"{value:.2f}" for value in q_values])
    ax.set_xlabel(r"$B$")
    ax.set_ylabel(r"$q_d$")
    ax.set_title(title, fontsize=14.8, pad=4)


def _scale_axis_fonts(ax: plt.Axes, factor: float) -> None:
    """Scale every text element on one axes for the reduced single-column figure."""
    ax.title.set_fontsize(ax.title.get_fontsize() * factor)
    ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * factor)
    ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * factor)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontsize(label.get_fontsize() * factor)
    for text in ax.texts:
        text.set_fontsize(text.get_fontsize() * factor)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(text.get_fontsize() * factor)


def main() -> None:
    _setup_style()
    rho_source = json.loads(RHO_SOURCE.read_text(encoding="utf-8"))
    renderer_source = json.loads(RENDERER_SOURCE.read_text(encoding="utf-8"))
    adapt_source = json.loads(ADAPT_SOURCE.read_text(encoding="utf-8"))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.8, 4.4), dpi=300)
    outer = fig.add_gridspec(1, 3, width_ratios=[0.70, 0.78, 2.52], wspace=0.30)
    spacing_ax = fig.add_subplot(outer[0, 0])
    renderer_ax = fig.add_subplot(outer[0, 1])
    adaptation = outer[0, 2].subgridspec(2, 3, height_ratios=[0.16, 1.0], width_ratios=[1.0, 1.0, 0.065], hspace=0.02, wspace=0.14)
    adaptation_title_ax = fig.add_subplot(adaptation[0, :])
    adaptation_title_ax.axis("off")
    adaptation_title_ax.set_title("(c) Contribution-aware adaptation", fontsize=15.5, pad=0)
    syn_ax = fig.add_subplot(adaptation[1, 0])
    real_ax = fig.add_subplot(adaptation[1, 1])
    colorbar_ax = fig.add_subplot(adaptation[1, 2])

    spacing_manifest = _plot_spacing(spacing_ax, rho_source)
    renderer_manifest = _plot_renderer(renderer_ax, renderer_source)
    syn_grid, q_values, budgets = _adaptation_grid(adapt_source, "Syn dPSNR")
    real_grid, _, _ = _adaptation_grid(adapt_source, "Real dPSNR")
    norm = TwoSlopeNorm(vmin=-0.03, vcenter=0.0, vmax=0.03)
    _plot_heatmap(syn_ax, syn_grid, q_values, budgets, "Synthetic Liver", norm)
    _plot_heatmap(real_ax, real_grid, q_values, budgets, "TUS-REC-2024", norm)
    real_ax.set_ylabel("")
    real_ax.set_yticklabels([])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=colorbar_ax)
    colorbar.set_label(r"$\Delta$PSNR (dB)", fontsize=14.8)
    colorbar.ax.tick_params(labelsize=14.8)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.18)

    pdf_path = FIGURE_DIR / "paper_ablation_parameter_sensitivity.pdf"
    png_path = FIGURE_DIR / "paper_ablation_parameter_sensitivity.png"
    manifest_path = MANIFEST_DIR / "paper_ablation_parameter_sensitivity_manifest.json"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025, metadata=PDF_METADATA)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.025, dpi=300)
    plt.close(fig)

    manifest = {
        "version": "TMI v2",
        "layout": {"main_panels": 3, "figsize_inches": [13.8, 4.4], "dpi": 300},
        "source_manifests": {
            "pseudo_pose_spacing": str(RHO_SOURCE.resolve()),
            "local_global_renderer": str(RENDERER_SOURCE.resolve()),
            "contribution_aware_adaptation": str(ADAPT_SOURCE.resolve()),
        },
        "panels": {
            "a_pseudo_pose_spacing": spacing_manifest,
            "b_local_global_renderer": renderer_manifest,
            "c_contribution_aware_adaptation": {
                "q_d": q_values,
                "B": budgets,
                "default": {"q_d": 0.98, "B": 0.05},
                "color_range_delta_psnr_db": [-0.03, 0.03],
                "synthetic_liver_delta_psnr": syn_grid.tolist(),
                "tus_rec_2024_delta_psnr": real_grid.tolist(),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Single-column layout: (a)/(b) on the first row and the two (c)
    # heatmaps spanning the second row.
    single_fig = plt.figure(figsize=(7.2, 7.0), dpi=300)
    single_outer = single_fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.42)
    single_top = single_outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.48)
    single_spacing_ax = single_fig.add_subplot(single_top[0, 0])
    single_renderer_ax = single_fig.add_subplot(single_top[0, 1])
    single_adaptation = single_outer[1, 0].subgridspec(
        2,
        3,
        height_ratios=[0.15, 1.0],
        width_ratios=[1.0, 1.0, 0.07],
        hspace=0.02,
        wspace=0.16,
    )
    single_adaptation_title_ax = single_fig.add_subplot(single_adaptation[0, :])
    single_adaptation_title_ax.axis("off")
    single_adaptation_title_ax.set_title("(c) Contribution-aware adaptation", fontsize=15.5, pad=0)
    single_syn_ax = single_fig.add_subplot(single_adaptation[1, 0])
    single_real_ax = single_fig.add_subplot(single_adaptation[1, 1])
    single_colorbar_ax = single_fig.add_subplot(single_adaptation[1, 2])

    _plot_spacing(single_spacing_ax, rho_source)
    _plot_renderer(single_renderer_ax, renderer_source)
    _plot_heatmap(single_syn_ax, syn_grid, q_values, budgets, "Synthetic Liver", norm)
    _plot_heatmap(single_real_ax, real_grid, q_values, budgets, "TUS-REC-2024", norm)
    single_real_ax.set_ylabel("")
    single_real_ax.set_yticklabels([])
    single_colorbar = single_fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
        cax=single_colorbar_ax,
    )
    single_colorbar.set_label(r"$\Delta$PSNR (dB)", fontsize=14.8)
    single_colorbar.ax.tick_params(labelsize=14.8)
    for axis in (
        single_spacing_ax,
        single_renderer_ax,
        single_adaptation_title_ax,
        single_syn_ax,
        single_real_ax,
        single_colorbar_ax,
    ):
        _scale_axis_fonts(axis, 1.10)
    single_fig.subplots_adjust(left=0.11, right=0.965, top=0.965, bottom=0.075)

    single_pdf_path = FIGURE_DIR / "paper_ablation_parameter_sensitivity_single_column.pdf"
    single_png_path = FIGURE_DIR / "paper_ablation_parameter_sensitivity_single_column.png"
    single_manifest_path = MANIFEST_DIR / "paper_ablation_parameter_sensitivity_single_column_manifest.json"
    single_fig.savefig(single_pdf_path, bbox_inches="tight", pad_inches=0.025, metadata=PDF_METADATA)
    single_fig.savefig(single_png_path, bbox_inches="tight", pad_inches=0.025, dpi=300)
    plt.close(single_fig)
    single_manifest = {
        **manifest,
        "layout": {
            "target": "IEEE TMI single column",
            "arrangement": "(a) and (b) on top; (c) across the bottom",
            "main_panels": 3,
            "figsize_inches": [7.2, 7.0],
            "dpi": 300,
        },
        "wide_layout_manifest": str(manifest_path.resolve()),
    }
    single_manifest_path.write_text(json.dumps(single_manifest, indent=2), encoding="utf-8")

    CONFIG_PATH.write_text(
        "# TMI v2 Ablation Figure Configuration\n\n"
        "Status: generated from frozen production manifests. No training results were changed.\n\n"
        "## Regeneration\n\n"
        "```bash\n"
        "MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python "
        "scripts/production/ablations/ours/make_tmi_ablation_joint_visual_compact_v2.py\n"
        "MPLCONFIGDIR=/tmp/mplconfig /home/boywang25/miniconda3/envs/gaussian_splatting/bin/python "
        "scripts/production/ablations/ours/make_tmi_ablation_sensitivity_summary_v2.py\n"
        "```\n\n"
        "## Outputs\n\n"
        "- `figures/paper_ablation_joint_visual_compact.pdf` and `.png`\n"
        "- `manifests/paper_ablation_joint_visual_compact_manifest.json`\n"
        "- `figures/paper_ablation_parameter_sensitivity.pdf` and `.png`\n"
        "- `manifests/paper_ablation_parameter_sensitivity_manifest.json`\n"
        "- `figures/paper_ablation_parameter_sensitivity_single_column.pdf` and `.png`\n"
        "- `manifests/paper_ablation_parameter_sensitivity_single_column_manifest.json`\n\n"
        "Terminology: LGR is the physics-aware local-global Gaussian renderer; CGA is contribution-aware "
        "Gaussian adaptation; PP is acquisition-guided pseudo-pose parameterization. The sensitivity figure "
        "uses differences relative to the corresponding production defaults.\n",
        encoding="utf-8",
    )
    for path in (
        pdf_path,
        png_path,
        manifest_path,
        single_pdf_path,
        single_png_path,
        single_manifest_path,
        CONFIG_PATH,
    ):
        print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
