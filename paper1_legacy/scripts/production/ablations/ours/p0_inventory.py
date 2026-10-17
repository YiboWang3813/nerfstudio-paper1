"""Read-only experiment inventory and exact config comparison; never submits jobs."""
import csv
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "outputs/production/ablations/ours"
OUT = ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908"


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, prefix + str(key) + "."))
        return result
    return {prefix[:-1]: value}


def load(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def main():
    sources = [BASE / p for p in (
        "summary.tsv", "expanded_paper_plan_20260702/summary.tsv",
        "multisweep_mean_validation_20260702/summary.tsv",
        "sensorless_overnight_202runs_20260717/summary.tsv",
        "path_aware_pruning_update_20260831/stage1_qA/summary.tsv",
        "path_aware_pruning_update_20260831/stage2_t03_t01/summary.tsv")]
    inventory = []
    for summary in sources:
        with summary.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                run = Path(row["run_dir"])
                cp, mp = run / "resolved_config.yaml", run / "run_manifest.yaml"
                if not cp.exists():
                    continue
                config = load(cp)
                manifest = load(mp) if mp.exists() else {}
                dc = config["model"]["offline"]["density_control"]
                rc = dc["responsibility"]
                row.update(seed=manifest.get("seed", config["system"]["seed"]),
                    git_commit=manifest.get("git_commit"), git_dirty=manifest.get("git_dirty"),
                    config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                    protection=rc.get("protect_attenuation_mode", "legacy_high_tau"),
                    pose=config["scene"]["loading"]["pose_mode"], source=str(summary))
                inventory.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "existing_run_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    ref = BASE / "path_aware_pruning_update_20260831/stage2_t03_t01/runs/t03/synthetic_liver/l1/d_full_path_aware_pruning/resolved_config.yaml"
    ref_cfg = flatten(load(ref))
    comparison = {}
    for ds, sweep in (("synthetic_liver", "l1"), ("tus_rec_2024_qc", "010_LH_Per_L_DtP_qc_0005_0370"), ("tdsc_abus_2023", "validation_100")):
        main_cfg = BASE / "path_aware_pruning_update_20260831/main_ours_gpu3/runs/stage3_full_sweep/nvs" / ds / sweep / "ours/resolved_config.yaml"
        if not main_cfg.exists():
            comparison[ds] = {"missing": str(main_cfg)}
            continue
        actual = flatten(load(main_cfg))
        differences = {k: [ref_cfg.get(k), actual.get(k)] for k in set(ref_cfg) | set(actual)
                       if k.startswith(("model.", "scene.", "loss.", "engine.", "system.seed")) and ref_cfg.get(k) != actual.get(k)}
        comparison[ds] = differences
    (OUT / "main_vs_stage2_config_differences.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print("Inventoried", len(inventory), "runs; main comparison differences:")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
