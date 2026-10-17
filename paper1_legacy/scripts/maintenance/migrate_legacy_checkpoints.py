#!/usr/bin/env python3
"""Migrate selected legacy checkpoints into the Hydra-era output layout.

The legacy ``checkpoints/`` tree is ignored by git and mostly contains
debug/intermediate training state. This script keeps only explicit production
artifacts, copies small result tables into ``outputs/imported/legacy_tables``, moves the
kept run directories under ``outputs/imported/legacy/<date>/``, then removes the
remaining legacy tree when run with ``--execute``.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Tuple


KEEP_DIRS = {
    # Ours: final renderer profiles and representative pose-prior runs.
    "ours/synthetic_liver_ours_us_focus_phase10_fast_seed42": "ours/us_focus_fast_seed42",
    "ours/synthetic_liver_ours_us_focus_phase10_fast_seed43": "ours/us_focus_fast_seed43",
    "ours/synthetic_liver_ours_us_focus_phase10_fast_seed44_retry": "ours/us_focus_fast_seed44",
    "ours/synthetic_liver_ours_us_focus_phase16_fast_l1_065_ssim_035": "ours/us_focus_fast_fidelity_seed42",
    "ours/synthetic_liver_ours_us_focus_phase17_fast_l1_065_ssim_035_seed43": "ours/us_focus_fast_fidelity_seed43",
    "ours/synthetic_liver_ours_us_focus_phase17_fast_l1_065_ssim_035_seed44": "ours/us_focus_fast_fidelity_seed44",
    "ours/synthetic_liver_ours_us_focus_phase12_perceptual_seed42_env": "ours/us_focus_perceptual_seed42",
    "ours/synthetic_liver_ours_us_focus_phase12_perceptual_seed43_env": "ours/us_focus_perceptual_seed43",
    "ours/synthetic_liver_ours_us_focus_phase12_perceptual_seed44_env": "ours/us_focus_perceptual_seed44",
    "ours/synthetic_liver_x2_ours_us_focus_phase11_x2_fast_seed42": "ours/us_focus_x2_fast_seed42",
    "ours/synthetic_liver_x2_ours_us_focus_phase11_x2_fast_seed43": "ours/us_focus_x2_fast_seed43",
    "ours/synthetic_liver_x2_ours_us_focus_phase11_x2_fast_seed44": "ours/us_focus_x2_fast_seed44",
    "ours/synthetic_liver_ours_us_focus_phase22_r2_gt_pose_30k": "ours/pose_prior_gt_pose_r2_30k",
    "ours/synthetic_liver_ours_us_focus_phase24_r2_noise5_fixed_30k": "ours/pose_prior_noise5_fixed_r2_30k",
    "ours/synthetic_liver_ours_us_focus_phase24_r2_noise10_fixed_30k": "ours/pose_prior_noise10_fixed_r2_30k",
    "ours/synthetic_liver_ours_us_focus_phase25b_r2_transonly_poseonly15_17_30k": "ours/pose_prior_transonly_refine_r2_30k",
    "ours/synthetic_liver_ours_us_focus_phase26_r2_trans50_fixed_30k": "ours/pose_prior_trans50_fixed_r2_30k",
    "ours/synthetic_liver_ours_us_focus_phase26_r2_trans50_poseonly15_20_lr3e4_lam001_30k": "ours/pose_prior_trans50_refine_r2_30k",
    "ours/synthetic_liver_ours_us_focus_phase28c_r2_fima_prior_fixed_30k": "ours/fima_prior_phase28c_fixed_r2_30k",
    "ours/synthetic_liver_ours_us_focus_phase28c_r2_fima_prior_transonly_poseonly15_20_30k": "ours/fima_prior_phase28c_transonly_r2_30k",

    # Comparison baselines.
    "gaussianpile/synthetic_liver_gaussianpile_gaussianpile_liver_1m_30000_dc_single_alpha_bestckpt_gpu1": "gaussianpile/liver_1m_30k_bestckpt_seed42",
    "gaussianpile/synthetic_liver_gaussianpile_gaussianpile_liver_1m_30000_dc_rerun_20260530_seed42": "gaussianpile/liver_1m_30k_rerun_seed42",
    "ultragauss/synthetic_liver_ultragauss_ultragauss_liver_1m_10000_dc": "ultragauss/liver_1m_10k_dc",
    "ultragray/synthetic_liver_ultragray_ultragray_liver_1m_10k_no_dc_fast4_4080": "ultragray/liver_1m_10k_no_dc_fast4_4080",
    "ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage": "ultranerf/fullres_30k_bernoulli_fullimage",
    "ultranerf/synthetic_liver_ultranerf_ultranerf_liver_fullres_30k_holdout_paper_bernoulli_fullimage_mc_eval4": "ultranerf/fullres_30k_bernoulli_fullimage_mc_eval4",

    # FiMA artifacts consumed by downstream reports/pose-prior experiments.
    "fima/fima_pooled_fullres_mamba_unc_supervised_liver_6sweep_40000": "fima/pooled_unc_supervised_6sweep_40k",
    "fima/fima_pooled_fullres_mamba_mask_liver_6sweep_40000": "fima/pooled_mask_6sweep_40k",
    "fima/phase27_fima_priors": "fima/phase27_priors",
    "fima/phase28_fima_priors": "fima/phase28_priors",
    "fima/synthetic_liver_x2_fima_phase28c_fima_paper_edges_r2_40000": "fima/phase28c_paper_edges_r2_40k",
}


KEEP_FILES = {
    "comparison/synthetic_liver_1m30k_dc/synthetic_liver_1m30k_dc_results.tsv",
    "ours/us_focus_ablation_results.tsv",
    "ours/us_focus_pose_prior_results.tsv",
    "ours/pose_uncertainty_stageA_250k_dc9k_results.tsv",
    "ours/pose_uncertainty_stageA_250k_results.tsv",
    "ours/pose_uncertainty_stageA_dcdecouple_results.tsv",
    "ours/pose_uncertainty_stageA_oracle_250k_dcdecouple_results.tsv",
    "ours/pose_uncertainty_stageB_predicted_250k_dcdecouple_extra_results.tsv",
    "ours/pose_uncertainty_stageB_predicted_250k_dcdecouple_results.tsv",
    "ours/pose_uncertainty_stageB_predicted_dc7000_results.tsv",
    "ours/pose_uncertainty_stageC_inflation_250k_dcdecouple_results.tsv",
    "ours/pose_uncertainty_stageD_mc_250k_dcdecouple_results.tsv",
    "ultragray_dc_sanity/ultragray_dc_sanity_10k.tsv",
    "fima/phase27_fima_prior_errors.tsv",
    "fima/phase28_fima_prior_errors.tsv",
}


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "K", "M", "G", "T"]:
        if value < 1024.0 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


def iter_legacy_entries(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.iterdir())


def write_import_manifest(dst: Path, src_rel: str, label: str) -> None:
    meta = dst / ".omniultra"
    meta.mkdir(parents=True, exist_ok=True)
    path = meta / "legacy_import.yaml"
    body = "\n".join(
        [
            f"imported_at: {datetime.now().isoformat(timespec='seconds')}",
            f"legacy_path: checkpoints/{src_rel}",
            f"import_label: {label}",
            "status: production_legacy_import",
            "notes: Imported during legacy checkpoint cleanup; DVC/data management intentionally excluded.",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def copy_result_file(src: Path, rel: str, tables_root: Path, execute: bool, actions: List[str]) -> None:
    dst = tables_root / rel
    actions.append(f"COPY file {src} -> {dst}")
    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def migrate_dir(src: Path, rel: str, label: str, imported_root: Path, execute: bool, actions: List[str]) -> None:
    dst = imported_root / label
    actions.append(f"MOVE dir  {src} -> {dst}")
    if execute:
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.move(str(src), str(dst))
        write_import_manifest(dst, rel, label)


def remove_leftovers(root: Path, execute: bool, actions: List[str]) -> None:
    if not root.exists():
        return
    for entry in sorted(root.iterdir()):
        actions.append(f"DELETE    {entry}")
        if execute:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="actually move and delete files")
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument("--import-root", default=f"outputs/imported/legacy/{date.today().isoformat()}")
    parser.add_argument("--tables-root", default="outputs/imported/legacy_tables")
    parser.add_argument("--plan-output", default="outputs/imported/legacy_cleanup_plan.md")
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoints_root)
    import_root = Path(args.import_root)
    tables_root = Path(args.tables_root)
    plan_output = Path(args.plan_output)

    original_size = tree_size(ckpt_root)
    keep_size = 0
    missing: List[str] = []
    actions: List[str] = []

    for rel, label in sorted(KEEP_DIRS.items()):
        src = ckpt_root / rel
        if not src.exists():
            missing.append(rel)
            continue
        keep_size += tree_size(src)
        migrate_dir(src, rel, label, import_root, args.execute, actions)

    for rel in sorted(KEEP_FILES):
        src = ckpt_root / rel
        if not src.exists():
            missing.append(rel)
            continue
        copy_result_file(src, rel, tables_root, args.execute, actions)

    # After production dirs have been moved and small result tables copied, the
    # remaining legacy checkpoint tree is intentionally removed.
    remove_leftovers(ckpt_root, args.execute, actions)

    plan_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "EXECUTE" if args.execute else "DRY RUN"
    plan = [
        f"# Legacy Checkpoint Cleanup Plan",
        "",
        f"Mode: `{mode}`",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        f"- Original checkpoint tree size: `{format_bytes(original_size)}`",
        f"- Production directories selected for migration: `{len(KEEP_DIRS)}`",
        f"- Approximate migrated directory size: `{format_bytes(keep_size)}`",
        f"- Small result files copied to docs: `{len(KEEP_FILES)}`",
        f"- Estimated freed space after migration: `{format_bytes(max(original_size - keep_size, 0))}`",
        f"- Import root: `{import_root}`",
        f"- Tables root: `{tables_root}`",
        "",
        "## Missing Allowlist Entries",
        "",
    ]
    if missing:
        plan.extend(f"- `{item}`" for item in missing)
    else:
        plan.append("- none")
    plan.extend(["", "## Actions", ""])
    plan.extend(f"- `{action}`" for action in actions)
    plan_output.write_text("\n".join(plan) + "\n", encoding="utf-8")

    print(f"mode={mode}")
    print(f"original={format_bytes(original_size)}")
    print(f"migrated={format_bytes(keep_size)}")
    print(f"estimated_freed={format_bytes(max(original_size - keep_size, 0))}")
    print(f"actions={len(actions)}")
    print(f"missing={len(missing)}")
    print(f"plan={plan_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
