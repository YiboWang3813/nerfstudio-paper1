"""Corrected-covariance evidence inventory and plans only. No job submission."""
import csv
import itertools
import json
import yaml
from collections import Counter
from pathlib import Path
from omegaconf import OmegaConf
from scripts.production.ablations.ours.p0_plan import common, REP, BASE, STAGE, SR, DC

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/preproduction/regression/p05_corrected_covariance_20260908"


def read_tsv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def plan(minimal):
    rows, seen = [], {}
    def add(group, variant, ds, sw, changes=None, targets=None, legacy=None):
        changes = changes or {}
        settings = dict(pose="file", spacing="uniform", rho=1., local="adaptive", global_mode="integral",
                        dense="residual_guided", prune="path_aware")
        settings.update(changes)
        if settings["global_mode"] == "none" and settings["prune"] == "path_aware":
            settings["prune"] = "local_aware"
        identity = (ds, sw, json.dumps(settings, sort_keys=True))
        if identity in seen:
            seen[identity]["consumers"].extend(targets or [group + "/" + variant])
            return
        scene = dict(synthetic_liver="liver", spine_phantom="spine_phantom", tus_rec_2024_qc="tus_rec_2024_qc", tdsc_abus_2023="tdsc_abus_2023")[ds]
        if ds == "spine_phantom":
            # Exact legacy main scene config is retained separately below.
            overrides = ["scene=" + scene, "scene.loading.sweep_name=" + sw,
                         "scene.loading.pose_mode=file", "experiment.sensorless_required=false", "system.seed=42"]
        else:
            overrides = common(ds, sw, settings["pose"])
        overrides += ["scene.loading.canonical_sweep_spacing_mode=" + settings["spacing"],
                      "scene.loading.canonical_sweep_delta_scale=" + str(settings["rho"]),
                      SR + "local_conditioning_mode=" + settings["local"],
                      SR + "ray_attenuation_enabled=" + str(settings["global_mode"] != "none").lower(),
                      SR + "ray_attenuation_formula=" + (settings["global_mode"] if settings["global_mode"] != "none" else "integral"),
                      DC + "densification_policy=" + settings["dense"], DC + "pruning_policy=" + settings["prune"]]
        if settings["spacing"] != "uniform":
            overrides += ["experiment.sensorless_required=false"]
        for k, v in changes.items():
            if "." in k:
                overrides.append(k + "=" + str(v))
        row = dict(group=group, variant=variant, dataset=ds, sweep=sw, settings=settings,
                   overrides=overrides, consumers=targets or [group + "/" + variant],
                   execution_authorized=group == "main" and (ds, sw) in REP,
                   legacy_status="LEGACY-REFERENCE" if legacy else None,
                   source=legacy, target_status=("NEW" if settings["local"] in ("none", "fixed") or
                       (settings["dense"] == "residual_guided" and settings["prune"] == "opacity") or
                       any(k in changes for k in (SR + "us_focus_det_gamma", SR + "ray_attenuation_quadrature_window_mm", DC + "thresholds.split_scale")) else "RERUN"))
        seen[identity] = row
        rows.append(row)
    mains = read_tsv(STAGE / "main_ours_gpu3/tables/ours_per_run.tsv")
    mains = {(r["dataset"], r["sweep"]): r for r in mains}
    assert len(mains) == 78
    for (ds, sw), r in sorted(mains.items()):
        add("main", "MP_Full", ds, sw, legacy=r["output_dir"])
        # Preserve complete actual dataset/loading/metric config, not a guessed alias.
        cfg = OmegaConf.to_container(OmegaConf.load(Path(r["output_dir"]) / "resolved_config.yaml"), resolve=False)
        rows[-1]["exact_main_scene_config"] = cfg["scene"]
        rows[-1]["exact_main_metric_config"] = cfg["metric"]
    def pair(group, variant, changes, targets=None):
        for ds, sw in REP:
            add(group, variant, ds, sw, changes, targets)
    for ppp, lgr, cga in itertools.product((False, True), repeat=3):
        pair("structural", "Joint_{}{}{}".format(int(ppp), int(lgr), int(cga)),
             dict(pose="canonical_sweep" if ppp else "file", global_mode="integral" if lgr else "none",
                  dense="residual_guided" if cga else "gradient", prune="path_aware" if cga else "opacity"))
    for spacing in ("uniform", "file_step_norm", "file_axis_projection"):
        pair("structural", "PPP_" + spacing, dict(pose="canonical_sweep", spacing=spacing))
    for local, glob in (("none", "none"), ("fixed", "none"), ("adaptive", "none"), ("adaptive", "slice_depth"), ("adaptive", "integral")):
        pair("structural", "LGR_" + local + "_" + glob, dict(local=local, global_mode=glob))
    pair("structural", "Adaptive_zero_global", dict(global_mode="none"), ["hyper_LGR/beta=0", "hyper_LGR/lambda=0"])
    for dense, prune in (("gradient", "opacity"), ("residual_guided", "none"), ("residual_guided", "opacity"), ("residual_guided", "local_aware"), ("residual_guided", "path_aware")):
        label = dense + "_" + prune
        pair("structural", "CGA_" + label, dict(dense=dense, prune=prune), ["CGA/" + label, "population_dynamics/" + label])
    selections = sorted({(r["dataset"], r["sweep"]) for r in read_tsv(BASE / "summary.tsv") if r["table"] == "t06"})
    assert len(selections) == 16
    for ds, sw in selections:
        for pose in ("file", "canonical_sweep"):
            add("multi_sweep", pose, ds, sw, dict(pose=pose))
    for rho in (.6, .75, .9, 1., 1.1, 1.25, 1.5):
        pair("hyper_PPP", "rho_" + str(rho), dict(pose="canonical_sweep", rho=rho))
    choices = {
        "ray_attenuation_beta": [.4, 1.] if minimal else [.4, .6, 1.],
        "ray_attenuation_strength": [.2, 1.] if minimal else [.2, .35, .75, 1.],
        "ray_attenuation_quadrature_points": [1, 2],
        "ray_attenuation_quadrature_window_mm": [1, 4] if minimal else [1, 3, 4],
    }
    if not minimal:
        choices.update(ray_attenuation_min_transmission=[.2, .5, .65, .8], us_focus_det_gamma=[0, .125, .5])
    for key, vals in choices.items():
        for v in vals:
            pair("hyper_LGR", key + "_" + str(v), {SR + key: v})
    for q, b in itertools.product([.94, .98, .99] if minimal else [.94, .96, .98, .99], [.02, .05, .12] if minimal else [.02, .05, .08, .12]):
        if (q, b) != (.98, .05):
            pair("hyper_CGA", "qd{}_B{}".format(q, b), {DC + "responsibility.score_quantile": q, DC + "responsibility.max_fraction": b})
    for qp, qa in itertools.product([.05, .1, .2] if minimal else [.02, .05, .1, .15, .2], [.6, .8, .9] if minimal else [.6, .7, .8, .9]):
        if (qp, qa) != (.1, .8):
            pair("hyper_CGA", "qp{}_qA{}".format(qp, qa), {DC + "responsibility.low_contribution_quantile": qp, DC + "responsibility.path_contribution_quantile": qa})
    for v in ([1e-5, 1e-3] if minimal else [1e-5, 5e-5, 5e-4, 1e-3]):
        pair("hyper_CGA", "alpha_" + str(v), {DC + "thresholds.prune_opacity": v})
    for v in ([.4, 1.] if minimal else [.4, .8, 1.]):
        pair("hyper_CGA", "split_" + str(v), {DC + "thresholds.split_scale": v})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, minimal in (("full", False), ("minimal", True)):
        rows = plan(minimal)
        counts = dict(Counter(r["group"] for r in rows))
        summary[label] = dict(total=len(rows), remaining_after_anchors=len(rows)-2, groups=counts,
                              optional_multisweep_rho=24 if not minimal else 0)
        (OUT / (label + "_run_matrix.json")).write_text(json.dumps(rows, indent=2) + "\n")
        print(label, summary[label])
    inventory = json.loads((ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908/existing_run_inventory.json").read_text())
    for row in inventory:
        with (Path(row["run_dir"]) / "resolved_config.yaml").open() as stream:
            cfg = yaml.safe_load(stream)
        row["local_shader"] = cfg["model"]["slice_render"].get("shader_weight_mode")
        row["faulty_covariance_path"] = row["local_shader"] in ("us_focus", "plane_cond2d")
        row.update(final_paper_status="LEGACY-REFERENCE", target_action="RERUN if selected by corrected plan",
                   reason="No demonstrated equivalence to corrected covariance training; old metrics cannot be final corrected Ours evidence")
    (OUT / "legacy_ablation_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    summary["baselines"] = "FINAL-REUSE: baseline code/data/protocol unchanged; no baseline training authorized or needed"
    summary["old_ours"] = "LEGACY-REFERENCE; target corrected rows RERUN; newly introduced controls NEW"
    summary["legacy_shader_counts"] = dict(Counter(str(r["local_shader"]) for r in inventory))
    main_inventory = read_tsv(STAGE / "main_ours_gpu3/tables/ours_per_run.tsv")
    for row in main_inventory:
        row.update(final_paper_status="LEGACY-REFERENCE", corrected_target="RERUN")
    (OUT / "legacy_main_inventory.json").write_text(json.dumps(main_inventory, indent=2) + "\n")
    summary["main_sweeps"] = dict(Counter(r["dataset"] for r in plan(True) if r["group"] == "main"))
    rho_sweeps = sorted({(r["dataset"], r["sweep"]) for r in read_tsv(BASE / "summary.tsv") if r["table"] == "t07"})
    optional = [dict(dataset=ds, sweep=sw, rho=rho, execution_authorized=False,
                     overrides=common(ds, sw, "canonical_sweep") + ["scene.loading.canonical_sweep_delta_scale=" + str(rho)])
                for ds, sw in rho_sweeps if (ds, sw) not in REP for rho in (.6, .75, .9, 1.1, 1.25, 1.5)]
    assert len(optional) == 24
    (OUT / "optional_full_multisweep_rho.json").write_text(json.dumps(optional, indent=2) + "\n")
    (OUT / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
