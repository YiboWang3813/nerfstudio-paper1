"""Prove config-level equivalence for latest Stage-1/2 and selected MP sources."""
import copy
import csv
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from configs.setup import load_hydra_config
from scripts.production.ablations.ours.p0_plan import BASE, OUT, STAGE, common


def semantic(cfg):
    cfg = copy.deepcopy(cfg)
    result = {k: cfg[k] for k in ("model", "scene", "loss", "engine")}
    result["seed"] = cfg["system"]["seed"]
    sr = result["model"]["slice_render"]
    sr.setdefault("local_conditioning_mode", "adaptive")
    dc = result["model"]["offline"]["density_control"]
    rc = dc["responsibility"]
    dense = dc.pop("densification_policy", None)
    if dense is None:
        dense = "residual_guided" if dc["strategy"].startswith("contribution_residual") else "gradient"
    prune = dc.pop("pruning_policy", None)
    if not dc["enable"]["prune"]:
        prune = "none"
    elif prune is None:
        if dense == "residual_guided" or rc["enabled"]:
            prune = "path_aware" if rc["protect_attenuation_enabled"] else "local_aware"
        else:
            prune = "opacity"
    if not sr["ray_attenuation_enabled"] or sr["ray_attenuation_beta"] == 0 or sr["ray_attenuation_strength"] == 0:
        if prune == "path_aware":
            prune = "local_aware"
    if not sr["ray_attenuation_enabled"]:
        sr["ray_attenuation_beta"] = sr["ray_attenuation_strength"] = 0.
    dc["strategy"] = dense
    dc["effective_pruning_policy"] = prune
    dc["enable"]["prune"] = prune != "none"
    rc["enabled"] = dense == "residual_guided" or prune in ("local_aware", "path_aware")
    rc["protect_attenuation_enabled"] = prune == "path_aware"
    # This legacy tau-quantile knob is not read by the A_g branch.
    if rc.get("protect_attenuation_mode") == "path_contribution":
        rc.pop("protect_attenuation_quantile", None)
    return result


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flatten(v, prefix + k + "."))
        return out
    return {prefix[:-1]: value}


def compare(row):
    path = Path(row["run_dir"]) / "resolved_config.yaml"
    old = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    normalized = semantic(old)
    dc = normalized["model"]["offline"]["density_control"]
    sr = old["model"]["slice_render"]
    pose = old["scene"]["loading"]["pose_mode"]
    prefix = "model.offline.density_control."
    overrides = common(row["dataset"], row["sweep"], pose) + [
        prefix + "densification_policy=" + dc["strategy"],
        prefix + "pruning_policy=" + dc["effective_pruning_policy"],
        "model.slice_render.ray_attenuation_enabled=" + str(sr["ray_attenuation_enabled"]).lower(),
        prefix + "responsibility.path_contribution_quantile=" + str(old["model"]["offline"]["density_control"]["responsibility"].get("path_contribution_quantile", .8))]
    new = OmegaConf.to_container(load_hydra_config("production/ablations/ours/paper_semantic_freeze", overrides), resolve=True)
    a, b = flatten(normalized), flatten(semantic(new))
    differences = {k: [a.get(k), b.get(k)] for k in set(a) | set(b) if a.get(k) != b.get(k)}
    return dict(dataset=row["dataset"], sweep=row["sweep"], variant=row.get("variant", "MP"),
                source=str(path.parent), decision="REUSE" if not differences else "REVIEW",
                differences=differences, normalized_config_sha256=hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest(),
                post_overrides=overrides,
                qualification="Conditional on P0 default-path tests; historical dirty commits lack their own archived patch.")


def main():
    rows = []
    for group in ("stage1_qA", "stage2_t03_t01"):
        with (STAGE / group / "summary.tsv").open() as f:
            rows.extend(csv.DictReader(f, delimiter="\t"))
    with (BASE / "summary.tsv").open() as f:
        selections = sorted({(r["dataset"], r["sweep"]) for r in csv.DictReader(f, delimiter="\t") if r["table"] == "t06"})
    for ds, sw in selections:
        rows.append(dict(dataset=ds, sweep=sw, run_dir=str(STAGE / "main_ours_gpu3/runs/stage3_full_sweep/nvs" / ds / sw / "ours")))
    audited = [compare(row) for row in rows]
    (OUT / "reuse_matrix.json").write_text(json.dumps(audited, indent=2) + "\n")
    for row in audited:
        if row["differences"]:
            print(row["source"], row["differences"])
    print("REUSE", sum(r["decision"] == "REUSE" for r in audited), "of", len(audited))


if __name__ == "__main__":
    main()
