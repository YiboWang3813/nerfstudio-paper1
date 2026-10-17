#!/usr/bin/env python3
"""Generate population-control supplementary data and figures from Stage 2 logs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[4]
STAGE2_ROOT = (
    PROJECT_ROOT
    / "outputs/production/ablations/ours/path_aware_pruning_update_20260831/stage2_t03_t01"
)
RUN_ROOT = STAGE2_ROOT / "runs/t03"
TABLE_ROOT = STAGE2_ROOT / "tables"
FIGURE_ROOT = PROJECT_ROOT / "docs/production/ablation_final_outputs/figures"

DATASETS = (
    ("synthetic_liver", "l1", "Synthetic Liver"),
    ("tus_rec_2024_qc", "010_LH_Per_L_DtP_qc_0005_0370", "TUS-REC-2024"),
)
VARIANTS = (
    ("a_gradient_density_control", "GD"),
    ("b_contribution_densification_only", "CD"),
    ("c_contribution_local_pruning", "CD+LP"),
    ("d_full_path_aware_pruning", "CD+LP+GP"),
)
COLORS = {
    "GD": "#4C78A8",
    "CD": "#F58518",
    "CD+LP": "#54A24B",
    "CD+LP+GP": "#B279A2",
}
EVENT_KEYS = {
    "train/dc/num_gaussians_before": "num_gaussians_before",
    "train/dc/num_gaussians_after": "num_gaussians_after",
    "train/dc/clone": "clone",
    "train/dc/split": "split",
    "train/dc/prune_weak": "prune",
    "train/dc/path_protected_candidate_count": "path_protected_candidate_decisions",
    "train/dc/weak_low_local_candidate_count": "weak_low_local_candidate_decisions",
    "train/dc/path_frames_accumulated": "path_frames_accumulated",
    "train/dc/attenuation_protect_threshold": "path_contribution_threshold",
}


def read_scalars(path: Path) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append((int(float(row["step"])), row["key"], float(row["value"])))
    return rows


def collect_data():
    population_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    plot_population: dict[tuple[str, str], list[tuple[int, float]]] = {}
    plot_events: dict[tuple[str, str], dict[int, dict[str, float]]] = {}

    for dataset, sweep, dataset_label in DATASETS:
        for variant, variant_label in VARIANTS:
            scalar_path = RUN_ROOT / dataset / sweep / variant / "logs/scalars.tsv"
            if not scalar_path.is_file():
                raise FileNotFoundError(f"Missing Stage 2 scalar log: {scalar_path}")
            scalars = read_scalars(scalar_path)

            population = [
                (step, value) for step, key, value in scalars if key == "train/num_gaussians"
            ]
            if not population:
                raise RuntimeError(f"No train/num_gaussians values in {scalar_path}")
            plot_population[(dataset, variant_label)] = population
            for step, value in population:
                population_rows.append(
                    {
                        "dataset": dataset,
                        "sweep": sweep,
                        "dataset_label": dataset_label,
                        "variant": variant_label,
                        "iteration": step,
                        "num_gaussians": int(round(value)),
                    }
                )

            events: dict[int, dict[str, float]] = defaultdict(dict)
            for step, key, value in scalars:
                if key in EVENT_KEYS:
                    events[step][EVENT_KEYS[key]] = value
            plot_events[(dataset, variant_label)] = events
            for step in sorted(events):
                values = events[step]
                event_rows.append(
                    {
                        "dataset": dataset,
                        "sweep": sweep,
                        "dataset_label": dataset_label,
                        "variant": variant_label,
                        "iteration": step,
                        "num_gaussians_before": int(round(values.get("num_gaussians_before", 0))),
                        "num_gaussians_after": int(round(values.get("num_gaussians_after", 0))),
                        "clone": int(round(values.get("clone", 0))),
                        "split": int(round(values.get("split", 0))),
                        "prune": int(round(values.get("prune", 0))),
                        "path_protected_candidate_decisions": int(
                            round(values.get("path_protected_candidate_decisions", 0))
                        ),
                        "weak_low_local_candidate_decisions": int(
                            round(values.get("weak_low_local_candidate_decisions", 0))
                        ),
                        "path_frames_accumulated": int(
                            round(values.get("path_frames_accumulated", 0))
                        ),
                        "path_contribution_threshold": values.get(
                            "path_contribution_threshold", 0.0
                        ),
                    }
                )
    return population_rows, event_rows, plot_population, plot_events


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def thousands(value, _position):
    return f"{value / 1000:.0f}k"


def plot_population_trajectory(plot_population) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (dataset, _sweep, dataset_label) in zip(axes, DATASETS):
        for _variant, variant_label in VARIANTS:
            values = plot_population[(dataset, variant_label)]
            ax.plot(
                [step / 1000 for step, _ in values],
                [count for _, count in values],
                label=variant_label,
                color=COLORS[variant_label],
                linewidth=1.7,
            )
        ax.set_title(dataset_label, fontsize=9)
        ax.set_xlabel("Iteration (k)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    axes[0].set_ylabel("Gaussian population")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_ROOT / "paper_adaptation_population_trajectory_supp.pdf"
    png = FIGURE_ROOT / "paper_adaptation_population_trajectory_supp.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    return pdf, png


def plot_event_statistics(plot_events) -> tuple[Path, Path]:
    metrics = (
        ("clone", "Cloned Gaussians"),
        ("prune", "Pruned Gaussians"),
        ("path_protected_candidate_decisions", "Path-protected candidate decisions"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.8), sharex=True)
    for row_index, (dataset, _sweep, dataset_label) in enumerate(DATASETS):
        for column_index, (metric, title) in enumerate(metrics):
            ax = axes[row_index, column_index]
            for _variant, variant_label in VARIANTS:
                events = plot_events[(dataset, variant_label)]
                steps = sorted(events)
                ax.plot(
                    [step / 1000 for step in steps],
                    [events[step].get(metric, 0.0) for step in steps],
                    marker="o",
                    markersize=2.7,
                    linewidth=1.2,
                    label=variant_label,
                    color=COLORS[variant_label],
                )
            if row_index == 0:
                ax.set_title(title, fontsize=8.5)
            if column_index == 0:
                ax.set_ylabel(dataset_label, fontsize=8.5)
            if row_index == 1:
                ax.set_xlabel("Adaptation iteration (k)")
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.0, w_pad=0.8)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_ROOT / "paper_adaptation_event_statistics_supp.pdf"
    png = FIGURE_ROOT / "paper_adaptation_event_statistics_supp.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    return pdf, png


def main() -> None:
    population_rows, event_rows, plot_population, plot_events = collect_data()
    population_tsv = TABLE_ROOT / "adaptation_population_by_iteration.tsv"
    event_tsv = TABLE_ROOT / "adaptation_event_statistics.tsv"
    write_tsv(population_tsv, population_rows)
    write_tsv(event_tsv, event_rows)
    population_figures = plot_population_trajectory(plot_population)
    event_figures = plot_event_statistics(plot_events)

    print(f"population_data={population_tsv}")
    print(f"event_data={event_tsv}")
    print(f"population_figures={','.join(map(str, population_figures))}")
    print(f"event_figures={','.join(map(str, event_figures))}")
    print(
        "note=path_protected_candidate_decisions counts candidate-event decisions, "
        "not unique Gaussians"
    )


if __name__ == "__main__":
    main()
