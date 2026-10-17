"""Generate reviewable, deduplicated plans/config snapshots. NEVER launch training."""
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

from configs.setup import load_hydra_config
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/preproduction/regression/p0_semantic_freeze_20260908"
DC = "model.offline.density_control."
SR = "model.slice_render."
REP = [("synthetic_liver", "l1"), ("tus_rec_2024_qc", "010_LH_Per_L_DtP_qc_0005_0370")]
BASE = ROOT / "outputs/production/ablations/ours"
STAGE = BASE / "path_aware_pruning_update_20260831"


def common(ds, sweep, pose="file"):
    scene = {"synthetic_liver": "liver", "tus_rec_2024_qc": "tus_rec_2024_qc", "tdsc_abus_2023": "tdsc_abus_2023"}[ds]
    ov = ["scene=" + scene, "scene.loading.sweep_name=" + sweep,
          "scene.loading.test_mode=holdout_interval", "scene.loading.holdout_trim_each=10",
          "scene.loading.holdout_num_samples=0", "scene.loading.holdout_test_ratio=0.2",
          "scene.loading.pose_mode=" + pose, "experiment.sensorless_required=" + str(pose == "canonical_sweep").lower(),
          "system.seed=42"]
    if ds != "synthetic_liver":
        ov += [SR + "elev_corridor_half_mm=" + ("0.202" if ds == "tus_rec_2024_qc" else "0.238")]
    return ov


def plan(minimal=False):
    rows = []
    def pair(group, name, overrides, status="NEW", approval="matrix"):
        for ds, sw in REP:
            rows.append(dict(group=group, variant=name, dataset=ds, sweep=sw, status=status,
                             approval=approval, execution_authorized=False,
                             implementation_ready=approval == "matrix", overrides=common(ds, sw) + overrides))
    pair("structural", "Plane", [SR + "local_conditioning_mode=none", SR + "ray_attenuation_enabled=false"])
    pair("structural", "Fixed", [SR + "local_conditioning_mode=fixed", SR + "ray_attenuation_enabled=false"])
    pair("structural", "Simple", [SR + "ray_attenuation_formula=slice_depth"], "RERUN")
    # B is a recommendation awaiting the user's decision, not an implemented switch.
    pair("structural", "FrozenTau_PROPOSED_B", [], approval="acoustic_control_B_and_matrix")
    pair("structural", "RGD_OP", [DC + "pruning_policy=opacity"])
    for mode in ("file_step_norm", "file_axis_projection"):
        pair("structural", mode, ["scene.loading.pose_mode=canonical_sweep",
             "scene.loading.canonical_sweep_spacing_mode=" + mode, "experiment.sensorless_required=false"], "RERUN")

    # Match existing T06 selection; MP sources use the latest A_g NVS main runs.
    with (BASE / "summary.tsv").open() as f:
        selections = sorted({(r["dataset"], r["sweep"]) for r in csv.DictReader(f, delimiter="\t") if r["table"] == "t06"})
    assert len(selections) == 16
    for ds, sw in selections:
        mp = STAGE / "main_ours_gpu3/runs/stage3_full_sweep/nvs" / ds / sw / "ours/resolved_config.yaml"
        if not mp.exists():
            raise RuntimeError("Missing prospective MP reuse source: " + str(mp))
        if (ds, sw) not in REP:
            rows.append(dict(group="multi_sweep", variant="PPP", dataset=ds, sweep=sw,
                status="RERUN", approval="matrix", overrides=common(ds, sw, "canonical_sweep")))
    for rho in (.6, .75, .9, 1.1, 1.25, 1.5):
        pair("hyper_PPP", "rho_" + str(rho), ["scene.loading.pose_mode=canonical_sweep",
             "experiment.sensorless_required=true", "scene.loading.canonical_sweep_delta_scale=" + str(rho)], "RERUN")

    values = {
      "ray_attenuation_beta": [.4, 1.] if minimal else [.4, .6, 1.],
      "ray_attenuation_strength": [.2, 1.] if minimal else [.2, .35, .75, 1.],
      "ray_attenuation_quadrature_points": [1, 2],
      "ray_attenuation_min_transmission": [.2, .65] if minimal else [.2, .5, .65, .8],
      "ray_attenuation_quadrature_window_mm": [1, 4] if minimal else [1, 3, 4],
      "us_focus_det_gamma": [0, .5] if minimal else [0, .125, .5],
    }
    # beta=0 and strength=0 reuse Adaptive local+LP IF regression equivalence passes.
    for key, choices in values.items():
        for v in choices:
            pair("hyper_LGR", key + "_" + str(v), [SR + key + "=" + str(v)],
                 "NEW" if key in ("ray_attenuation_quadrature_window_mm", "us_focus_det_gamma") else "RERUN")
    for q, b in itertools.product([.94, .98, .99] if minimal else [.94, .96, .98, .99],
                                 [.02, .05, .12] if minimal else [.02, .05, .08, .12]):
        if (q, b) != (.98, .05):
            pair("hyper_CGA", "qd{}_B{}".format(q, b), [DC + "responsibility.score_quantile=" + str(q), DC + "responsibility.max_fraction=" + str(b)], "RERUN")
    for qp, qa in itertools.product([.05, .1, .2] if minimal else [.02, .05, .1, .15, .2],
                                    [.6, .8, .9] if minimal else [.6, .7, .8, .9]):
        if qp != .1:
            pair("hyper_CGA", "qp{}_qA{}".format(qp, qa), [DC + "responsibility.low_contribution_quantile=" + str(qp), DC + "responsibility.path_contribution_quantile=" + str(qa)], "RERUN")
    for v in ([1e-5, 1e-3] if minimal else [1e-5, 5e-5, 5e-4, 1e-3]):
        pair("hyper_CGA", "alpha_" + str(v), [DC + "thresholds.prune_opacity=" + str(v)], "RERUN")
    for v in ([.4, 1.] if minimal else [.4, .8, 1.]):
        pair("hyper_CGA", "split_" + str(v), [DC + "thresholds.split_scale=" + str(v)])
    # Overrides deliberately list only differences from the versioned Hydra anchor.
    identities = [(r["dataset"], r["sweep"], tuple(r["overrides"])) for r in rows]
    assert len(identities) == len(set(identities)), "Duplicate proposed run"
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"execution_authorized": False, "canonical_paper_confirmed": False,
               "covariance_semantic_freeze": "BLOCKED_PRE_EXISTING_DISCREPANCY",
               "acoustic_control": "B proposed, not implemented; 2 conditional runs included",
               "multi_sweep_rho": "not included; optional +24 non-default runs using paired validation defaults"}
    for label, minimal in (("full", False), ("minimal", True)):
        rows = plan(minimal)
        (OUT / (label + "_run_matrix.json")).write_text(json.dumps(rows, indent=2) + "\n")
        summary[label] = dict(total=len(rows), conditional_acoustic=2,
                              groups=dict(Counter(r["group"] for r in rows)))
        print(label, len(rows), summary[label]["groups"])
    (OUT / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for ds, sw in REP:
        cfg = load_hydra_config("production/ablations/ours/paper_semantic_freeze", common(ds, sw, "canonical_sweep"))
        OmegaConf.save(cfg, OUT / ("anchor_" + ds + ".yaml"), resolve=True)


if __name__ == "__main__":
    main()
