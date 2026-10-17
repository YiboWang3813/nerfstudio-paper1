"""Read corrected anchor artifacts; prepare impact deltas and FPS input only."""
import csv
import hashlib
import json
from pathlib import Path

from scripts.production.ablations.ours.p0_plan import REP, STAGE
from scripts.production.ablations.ours.summarize_ablation_run import read_series, value_at

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/preproduction/regression/p05_corrected_covariance_20260908"


def main():
    with (STAGE / "stage2_t03_t01/summary.tsv").open() as stream:
        legacy_rows = list(csv.DictReader(stream, delimiter="\t"))
    impacts, fps_rows = [], []
    for name, (ds, sw) in zip(("liver", "arm"), REP):
        run = OUT / "anchors" / name
        stats = json.loads((run / "anchor_stats.json").read_text())
        assert stats.get("success")
        old = next(r for r in legacy_rows if r["dataset"] == ds and r["table"] == "t03" and r["variant"] == "d_full_path_aware_pruning")
        series = read_series(run / "logs/scalars.tsv")
        metrics = {k: value_at(series, "test/" + k, 30000) for k in ("psnr", "ssim", "lpips", "l1", "dists", "gmsd")}
        metrics["num_gaussians"] = value_at(series, "train/num_gaussians", 30000)
        assert all(v is not None for v in metrics.values())
        old_metrics = {k: float(old["final_" + k]) for k in metrics}
        delta = {k: float(v) - old_metrics[k] for k, v in metrics.items()}
        legacy_run = Path(old["run_dir"])
        common_images = sorted(set(p.parent.name for p in (run / "vis").glob("*/pred.png")) &
                               set(p.parent.name for p in (legacy_run / "vis").glob("*/pred.png")),
                               key=lambda n: int(n.split("_")[-1]))
        chosen = common_images[len(common_images) // 2]
        same_gt = hashlib.sha256((run / "vis" / chosen / "gt.png").read_bytes()).hexdigest() == hashlib.sha256((legacy_run / "vis" / chosen / "gt.png").read_bytes()).hexdigest()
        assert same_gt, "Qualitative pair must have identical GT"
        impact = dict(dataset=ds, sweep=sw, corrected=metrics, legacy=old_metrics, delta=delta,
                      corrected_run=str(run), legacy_run=str(legacy_run), stats=stats,
                      legacy_shared_worker_wall_seconds=float(old["wall_seconds"]),
                      qualitative_selection="middle saved holdout case, chosen by index not metric", qualitative_case=chosen,
                      qualitative_gt_identical=same_gt,
                      timing_caveat="corrected uses one worker/GPU; legacy used two/GPU; no isolated speedup claim",
                      statistics_caveat="single-seed impact comparison, legacy binary not strictly reconstructible")
        impacts.append(impact)
        fps_rows.append(dict(benchmark="p05_corrected", stage="anchor", dataset=ds, sweep=sw,
                             method="ours", final_step=30000, final_num_gaussians=metrics["num_gaussians"], output_dir=str(run)))
        print(name, json.dumps(impact))
    (OUT / "anchor_impact.json").write_text(json.dumps(impacts, indent=2) + "\n")
    with (OUT / "anchor_fps_input.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fps_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(fps_rows)


if __name__ == "__main__":
    main()
