"""Bounded native acceptance: 256 points / 3 updates, never the formal budget."""

import copy
import gc
import json
import time
from pathlib import Path

from nerfstudio.scripts.ultrasound_resources import resources


def main():
    with resources(gpu=True):
        import torch

        from nerfstudio.configs.ultrasound_protocol import METHODS, production_config
        from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location
        from nerfstudio.scripts.train import _set_random_seed
        from nerfstudio.utils.ultrasound_provenance import execution_identity

        root = Path("/home/yibo/paper1")
        output = root / "outputs/preformal-smoke"
        receipt = root / "migration/native-matrix-smoke.json"
        results = []
        identity = execution_identity()
        for dataset, sweep in [
            ("synthetic_liver", "l1"),
            ("spine_phantom", "left1"),
            ("tus_rec_2024_qc", "010_LH_Per_L_DtP_qc_0005_0370"),
            ("tdsc_abus_2023", "validation_100"),
        ]:
            manifest = Path("/home/yibo/shared/datasets") / dataset / "p100k-native-v2" / sweep / "manifest.json"
            for method in METHODS:
                started = time.monotonic()
                _set_random_seed(42)
                cfg = production_config(method, manifest, output)
                cfg.timestamp = "engineering-" + str(time.time_ns())
                cfg.pipeline.evaluation_output = cfg.get_base_dir() / "evaluation"
                cfg.max_num_iterations = 3
                cfg.steps_per_eval_all_images = cfg.steps_per_eval_image = cfg.steps_per_eval_batch = 0
                cfg.logging.local_writer.enable = False
                if hasattr(cfg.pipeline.model, "num_gaussians"):
                    cfg.pipeline.model.num_gaussians = 256
                cfg.save_config()
                trainer = cfg.setup()
                trainer.setup()
                before = {name: p.detach().clone() for name, p in trainer.pipeline.model.named_parameters()}
                trainer.train()
                changed = any(not torch.equal(p, before[name]) for name, p in trainer.pipeline.model.named_parameters())
                if not changed:
                    raise AssertionError(f"{dataset}/{method} did not update any parameters")
                if not all(torch.isfinite(p).all() for p in trainer.pipeline.model.parameters()):
                    raise ValueError("Nonfinite trained parameter")
                checkpoint = next(trainer.checkpoint_dir.glob("*.ckpt"))
                # Evaluate one explicitly identified held-out frame, keeping the
                # frozen split unchanged. This is an engineering test, not NVS acceptance.
                batch = trainer.pipeline.datamanager.next_eval_image(3)
                trainer.pipeline.datamanager.fixed_indices_eval_dataloader = [batch]
                metrics = trainer.pipeline.get_average_eval_image_metrics(step=3)
                if not all(
                    torch.isfinite(torch.tensor(metrics[name]))
                    for name in ("psnr", "l1", "ssim", "lpips", "dists", "gmsd")
                ):
                    raise ValueError("Nonfinite full metrics")

                def advance(current):
                    for callback in current.callbacks:
                        callback.run_callback_at_location(3, Location.BEFORE_TRAIN_ITERATION)
                    current.train_iteration(3)
                    for callback in current.callbacks:
                        callback.run_callback_at_location(3, Location.AFTER_TRAIN_ITERATION)

                advance(trainer)
                expected = {name: p.detach().clone() for name, p in trainer.pipeline.model.named_parameters()}
                loaded = copy.deepcopy(cfg)
                loaded.load_checkpoint = checkpoint
                loaded.timestamp += "-restored"
                restored = loaded.setup()
                restored.setup()
                advance(restored)
                for name, p in restored.pipeline.model.named_parameters():
                    torch.testing.assert_close(p, expected[name], rtol=1e-5, atol=1e-7)
                results.append(
                    {
                        "dataset": dataset,
                        "sweep": sweep,
                        "method": method,
                        "updates": 3,
                        "continuation_update": 4,
                        "initial_gaussians": getattr(cfg.pipeline.model, "num_gaussians", None),
                        "eval_frame_id": int(batch[0].metadata["frame_id"].item()),
                        "metrics": metrics,
                        "seconds": time.monotonic() - started,
                        "manifest_sha256": trainer.pipeline.datamanager.eval_dataset.metadata["manifest_sha256"],
                        "config": str(cfg.get_base_dir() / "config.yml"),
                        "checkpoint": str(checkpoint),
                    }
                )
                receipt.write_text(
                    json.dumps(
                        {
                            "status": "engineering_only",
                            "formal_training_started": False,
                            "execution_at_start": identity,
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print("PASSED", dataset, method, flush=True)
                del trainer, restored, before, expected
                gc.collect()
                torch.cuda.empty_cache()
        print("COMPLETE", len(results), "bounded native cases", flush=True)


if __name__ == "__main__":
    main()
