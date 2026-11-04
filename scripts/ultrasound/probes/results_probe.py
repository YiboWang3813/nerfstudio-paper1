"""Representative 5k native runs compared with retained historical results.

Runs only the new implementation. Historical metrics are evidence, not a legacy
runtime dependency. One frozen sweep, original initialization size and schedules,
a bounded training prefix, and the complete matching held-out frame set.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/home/yibo/paper1")
ARCHIVE = ROOT / "results/archive/eehpc-20261017/evidence/legacy"
METRICS = ("psnr", "l1", "ssim", "lpips", "dists", "gmsd")


def historical_run(method):
    if method in ("nerfus", "ultranerf"):
        stage = "production/comparisons/us_recon_four_dataset_p100k_v1/runs/implicit_stage3_full_sweep"
        label = "nerfus_no_prior" if method == "nerfus" else method
    else:
        stage = "preproduction/comparisons/us_recon_four_dataset_p100k_v1/runs/stage1_calibration"
        label = method
    return ARCHIVE / stage / "nvs/synthetic_liver/l1" / label


def main():
    from nerfstudio.scripts.ultrasound_resources import resources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument(
        "--historical-ultrags",
        action="store_true",
        help="Diagnostic: replay the historical UltraGS parameter variant using native code",
    )
    args = parser.parse_args()
    if args.historical_ultrags and args.method != ["ultrags_baseline"]:
        parser.error("--historical-ultrags requires only --method ultrags_baseline")
    with resources(gpu=True):
        import torch
        import yaml
        from torchvision.utils import save_image

        from nerfstudio.configs.ultrasound_protocol import METHODS, production_config
        from nerfstudio.scripts.train import _set_random_seed
        from nerfstudio.utils.ultrasound_provenance import execution_identity

        for method in args.method:
            if method not in METHODS:
                raise ValueError(method)
            namespace = "native-5k-liver-l1-historical-parameters" if args.historical_ultrags else "native-5k-liver-l1"
            dest = ROOT / "results/probes" / namespace / method
            dest.mkdir(parents=True, exist_ok=True)
            receipt = dest / "receipt.json"
            if receipt.exists():
                raise RuntimeError(f"Existing probe evidence must not be overwritten: {receipt}")
            source = historical_run(method)
            table = source / "eval/per_frame_metrics_iteration_005000.tsv"
            rows = list(csv.DictReader(table.open(), delimiter="\t"))
            historical = {key: sum(float(row[key]) for row in rows) / len(rows) for key in METRICS}
            old_config = yaml.safe_load((source / "resolved_config.yaml").read_text())
            manifest = Path("/home/yibo/shared/datasets/synthetic_liver/p100k-native-v2/l1/manifest.json")
            _set_random_seed(42)
            cfg = production_config(method, manifest, ROOT / "outputs/results-probes")
            if args.historical_ultrags:
                from nerfstudio.configs.ultrasound_baselines import baseline_config
                from nerfstudio.configs.ultrasound_protocol import update_dataclass
                from nerfstudio.model_components.ultrasound_baselines.ultrags_baseline.configs import ModelConfig

                algorithm = update_dataclass(ModelConfig(), old_config["model"])
                # The recorded early implementation predates physical slice culling.
                algorithm.render.enable_slice_cull = False
                cfg.pipeline.model.algorithm = algorithm
                cfg.optimizers = baseline_config(method, algorithm).optimizers
                sampler = old_config["scene"]["sampler"]
                for name, key in [
                    ("uniform_ratio", "uniform_ratio"),
                    ("intensity_threshold", "intensity_threshold"),
                    ("jitter_radius_mm", "jitter_radius"),
                    ("initial_scales_mm", "scales_init"),
                    ("initial_quats_wxyz", "quats_init"),
                ]:
                    setattr(cfg.pipeline, name, sampler[key])
            cfg.timestamp = "probe5k-" + str(time.time_ns())
            cfg.max_num_iterations = 5000
            cfg.steps_per_eval_all_images = 4999
            cfg.steps_per_eval_batch = cfg.steps_per_eval_image = 0
            cfg.steps_per_save = 5000
            cfg.logging.local_writer.enable = False
            cfg.pipeline.evaluation_output = dest / "evaluation"
            if method in ("nerfus", "ultranerf"):
                cfg.pipeline.model.eval_seed_iteration = 5000
            cfg.save_config()
            record = {
                "status": "running",
                "method": method,
                "dataset": "synthetic_liver",
                "sweep": "l1",
                "updates": 5000,
                "formal_matrix_started": False,
                "initial_gaussians": getattr(cfg.pipeline.model, "num_gaussians", None),
                "comparison": "practical historical quality comparison; no old implementation executed",
                "parameter_variant": "historical_stage1_without_slice_culling"
                if args.historical_ultrags
                else "current_frozen_preset",
                "historical_metrics": historical,
                "historical_frames": [int(row["frame_id"]) for row in rows],
                "historical_table": str(table),
                "historical_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                "historical_config_sha256": hashlib.sha256((source / "resolved_config.yaml").read_bytes()).hexdigest(),
                "historical_source_commit": "80f15957ede5e48a37ebe5227c1a209404a53cb0",
                "native_config": str(cfg.get_base_dir() / "config.yml"),
                "native_execution": execution_identity(),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "historical_config": {
                    "iterations": old_config["engine"]["iterations"],
                    "num_init": old_config["scene"]["sampler"]["num_init"],
                },
                "limits": "Historical references include older parameter variants; parameter_variant and native_config identify the actual native settings. This is a 5k prefix, not a full-budget result.",
            }
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            start = time.monotonic()
            try:
                trainer = cfg.setup()
                trainer.setup()
                actual_frames = trainer.pipeline.datamanager.eval_dataset.metadata["frame_ids"]
                if list(actual_frames) != record["historical_frames"]:
                    raise ValueError("Historical and native held-out frame sets/order differ")
                trainer.train()
                eval_receipt = json.loads((dest / "evaluation/step-4999/receipt.json").read_text())
                native = {key: eval_receipt["metrics"][key] for key in METRICS}
                if not all(torch.isfinite(torch.tensor(value)) for value in native.values()):
                    raise ValueError("Nonfinite evaluation metrics")
                with trainer.pipeline.evaluation_context(), torch.no_grad():
                    manager = trainer.pipeline.datamanager
                    for index in sorted({0, len(manager.eval_dataset) // 2, len(manager.eval_dataset) - 1}):
                        camera, batch = manager._get(manager.eval_dataset, index)
                        prediction = trainer.pipeline.model.get_outputs_for_camera(camera)["ultrasound"]
                        image = torch.cat([batch["image"], prediction.clamp(0, 1)], dim=1)
                        save_image(
                            image.permute(2, 0, 1),
                            dest / f"frame-{int(camera.metadata['frame_id'].item()):04d}-target-prediction.png",
                        )
                record.update(
                    status="completed_for_review",
                    native_metrics=native,
                    deltas={key: native[key] - historical[key] for key in METRICS},
                    seconds=time.monotonic() - start,
                    final_gaussians=len(trainer.pipeline.model.gaussians["means"])
                    if hasattr(trainer.pipeline.model, "gaussians")
                    else None,
                )
                print("PROBE_COMPLETE", method, json.dumps(record["deltas"]), flush=True)
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException as error:
                record.update(status="failed", error=repr(error), seconds=time.monotonic() - start)
                raise
            finally:
                receipt.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
