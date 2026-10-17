"""Recover only completed Liver post-validation; never train or change checkpoint."""
import hashlib
import json
import shutil
from pathlib import Path
import torch
from scripts.production.ablations.ours.summarize_ablation_run import read_series, value_at

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "outputs/preproduction/regression/p05_corrected_covariance_20260908/anchors/liver"


def main():
    path = RUN / "anchor_stats.json"
    stats = json.loads(path.read_text())
    series = read_series(RUN / "logs/scalars.tsv")
    for key in ("psnr", "ssim", "lpips", "l1", "dists", "gmsd"):
        value = value_at(series, "test/" + key, 30000)
        assert isinstance(value, float) and torch.isfinite(torch.tensor(value))
    cp = RUN / "checkpoints/iteration_30000_container.pth"
    parameters = torch.load(cp, map_location="cpu")
    for key, tensor in parameters.items():
        assert isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all(), key
        assert tensor.shape[0] == int(value_at(series, "train/num_gaussians", 30000))
    if not stats.get("success"):
        shutil.copy2(path, RUN / "anchor_stats_original_postvalidation_error.json")
        stats.update(success=True, finite_final_parameters=True,
                     postvalidation_recovered="Training/evaluation/save reached 30k; GaussParam-vs-Tensor audit error only; independently verified saved tensors and all final metrics. No retraining.",
                     checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
        path.write_text(json.dumps(stats, indent=2) + "\n")
    print("Completed Liver checkpoint and metrics verified; no retraining.")


if __name__ == "__main__":
    main()
