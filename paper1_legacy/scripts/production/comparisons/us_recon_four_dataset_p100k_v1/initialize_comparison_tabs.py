#!/usr/bin/env python3
"""Initialize per-dataset comparison ledgers for the production benchmark."""

from __future__ import annotations

from pathlib import Path


METHODS = [
    "ours",
    "ultragauss",
    "ultragray",
    "gaussianpile",
    "ultrags_baseline",
    "slice_adapted_3dgs",
    "ultranerf",
    "nerfus_no_prior",
]

METRIC_COLUMNS = ["PSNR ↑", "L1 ↓", "SSIM ↑", "LPIPS ↓", "DISTS ↓", "GMSD ↓"]

DATASETS = {
    "synthetic_liver": {
        "title": "Synthetic Liver",
        "stage1": ["l1"],
        "stage2": ["l1", "l2"],
        "sweeps": ["l1", "l2", "l3", "r1", "r2", "r3"],
    },
    "spine_phantom": {
        "title": "Spine Phantom",
        "stage1": ["left1"],
        "stage2": ["left1", "left1_1", "left2"],
        "sweeps": ["left1", "left1_1", "left2", "left3", "left3_2", "right1", "right1_1", "right3", "right3_2"],
    },
    "tus_rec_2024": {
        "title": "TUS-REC 2024",
        "stage1": ["010_LH_Par_C_DtP"],
        "stage2": [
            "010_LH_Par_C_DtP",
            "010_LH_Par_C_PtD",
            "010_LH_Par_L_DtP",
            "010_LH_Par_L_PtD",
            "010_LH_Par_S_DtP",
            "010_LH_Par_S_PtD",
            "010_LH_Per_C_DtP",
            "010_LH_Per_C_PtD",
            "010_LH_Per_L_DtP",
            "010_LH_Per_L_PtD",
            "010_LH_Per_S_DtP",
            "010_LH_Per_S_PtD",
        ],
        "sweeps": [
            "010_LH_Par_C_DtP",
            "010_LH_Par_C_PtD",
            "010_LH_Par_L_DtP",
            "010_LH_Par_L_PtD",
            "010_LH_Par_S_DtP",
            "010_LH_Par_S_PtD",
            "010_LH_Per_C_DtP",
            "010_LH_Per_C_PtD",
            "010_LH_Per_L_DtP",
            "010_LH_Per_L_PtD",
            "010_LH_Per_S_DtP",
            "010_LH_Per_S_PtD",
            "010_RH_Par_C_DtP",
            "010_RH_Par_C_PtD",
            "010_RH_Par_L_DtP",
            "010_RH_Par_L_PtD",
            "010_RH_Par_S_DtP",
            "010_RH_Par_S_PtD",
            "010_RH_Per_C_DtP",
            "010_RH_Per_C_PtD",
            "010_RH_Per_L_DtP",
            "010_RH_Per_L_PtD",
            "010_RH_Per_S_DtP",
            "010_RH_Per_S_PtD",
            "011_LH_Par_C_DtP",
            "011_LH_Par_C_PtD",
            "011_LH_Par_L_DtP",
            "011_LH_Par_L_PtD",
            "011_LH_Par_S_DtP",
            "011_LH_Par_S_PtD",
            "011_LH_Per_C_DtP",
            "011_LH_Per_C_PtD",
            "011_LH_Per_L_DtP",
            "011_LH_Per_L_PtD",
            "011_LH_Per_S_DtP",
            "011_LH_Per_S_PtD",
            "011_RH_Par_C_DtP",
            "011_RH_Par_C_PtD",
            "011_RH_Par_L_DtP",
            "011_RH_Par_L_PtD",
            "011_RH_Par_S_DtP",
            "011_RH_Par_S_PtD",
            "011_RH_Per_C_DtP",
            "011_RH_Per_C_PtD",
            "011_RH_Per_L_DtP",
            "011_RH_Per_L_PtD",
            "011_RH_Per_S_DtP",
            "011_RH_Per_S_PtD",
        ],
    },
    "tdsc_abus_2023": {
        "title": "TDSC ABUS 2023",
        "stage1": ["validation_100"],
        "stage2": [f"validation_{i}" for i in range(100, 109)],
        "sweeps": [f"validation_{i}" for i in range(100, 130)],
    },
}


def stage_label(dataset: dict[str, list[str]], sweep: str) -> str:
    if sweep in dataset["stage1"]:
        return "stage1_calibration"
    if sweep in dataset["stage2"]:
        return "stage2_30pct"
    return "stage3_full_sweep"


def table_for_sweep() -> list[str]:
    lines = [
        "| Method | " + " | ".join(METRIC_COLUMNS) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        lines.append(f"| `{method}` | TODO | TODO | TODO | TODO | TODO | TODO |")
    return lines


def dataset_doc(name: str, spec: dict[str, object]) -> str:
    sweeps = list(spec["sweeps"])
    lines = [
        f"# {spec['title']} Comparison Ledger",
        "",
        f"Dataset key: `{name}`",
        f"Expected sweeps: `{len(sweeps)}`",
        "",
        "Rows are the eight comparison methods. Columns are the six production metrics.",
        "Fill rows as runs complete; stages 1, 2, and 3 share this same ledger.",
        "",
        "| Stage | Sweep policy | Sweep count | Purpose |",
        "| --- | --- | ---: | --- |",
        "| `stage1_calibration` | one representative sweep | 1 | Validate fair production configs and confirm Ours remains best before scaling. |",
        "| `stage2_30pct` | first 30% subset including stage 1 | "
        + str(len(spec["stage2"]))
        + " | Test ranking stability before full launch. |",
        "| `stage3_full_sweep` | all sweeps | "
        + str(len(sweeps))
        + " | Complete dataset evidence for the final average table. |",
        "",
    ]
    for index, sweep in enumerate(sweeps, 1):
        lines.extend(
            [
                f"## {index:02d}. `{sweep}`",
                "",
                f"Stage: `{stage_label(spec, sweep)}`",
                "",
                *table_for_sweep(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    out_dir = Path("docs/production/comparison_tabs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in DATASETS.items():
        (out_dir / f"{name}.md").write_text(dataset_doc(name, spec), encoding="utf-8")


if __name__ == "__main__":
    main()
