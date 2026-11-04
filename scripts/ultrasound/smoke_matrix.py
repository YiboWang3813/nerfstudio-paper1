"""Bounded native acceptance: 256 points / 3 updates, never the formal budget."""

import copy
import argparse
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

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--method", choices=METHODS, action="append")
        parser.add_argument(
            "--dataset",
            choices=("synthetic_liver", "spine_phantom", "tus_rec_2024_qc", "tdsc_abus_2023"),
            action="append",
        )
        args = parser.parse_args()

        root = Path("/home/yibo/paper1")
        output = root / "outputs/preformal-smoke"
        receipt = root / "migration/native-matrix-smoke.json"
        if args.dataset or args.method:
            receipt = receipt.with_name(
                "native-matrix-smoke-"
                + "-".join(sorted(args.dataset or ["all-data"]))
                + "-"
                + "-".join(sorted(args.method or ["all-methods"]))
                + ".json"
            )
        results = []
        identity = execution_identity()
        for dataset, sweep in [
            ("synthetic_liver", "l1"),
            ("spine_phantom", "left1"),
            ("tus_rec_2024_qc", "010_LH_Per_L_DtP_qc_0005_0370"),
            ("tdsc_abus_2023", "validation_100"),
        ]:
            if args.dataset and dataset not in args.dataset:
                continue
            manifest = Path("/home/yibo/shared/datasets") / dataset / "p100k-native-v2" / sweep / "manifest.json"
            for method in METHODS:
                if args.method and method not in args.method:
                    continue
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
                metrics = trainer.pipeline.get_average_eval_image_metrics(step=2)
                if not all(
                    torch.isfinite(torch.tensor(metrics[name]))
                    for name in ("psnr", "l1", "ssim", "lpips", "dists", "gmsd")
                ):
                    raise ValueError("Nonfinite full metrics")
                pre_parameters = {name: p.detach().clone() for name, p in trainer.pipeline.model.named_parameters()}
                pre_optimizers = {
                    name: copy.deepcopy(opt.state_dict()) for name, opt in trainer.optimizers.optimizers.items()
                }
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone()
                manager = trainer.pipeline.datamanager
                cursor, order = manager._cursor, manager._order.clone()

                def advance(current):
                    for callback in current.callbacks:
                        callback.run_callback_at_location(3, Location.BEFORE_TRAIN_ITERATION)
                    loss = current.train_iteration(3)[0]
                    for callback in current.callbacks:
                        callback.run_callback_at_location(3, Location.AFTER_TRAIN_ITERATION)
                    return loss.detach()

                expected_loss = advance(trainer)
                expected = {name: p.detach().clone() for name, p in trainer.pipeline.model.named_parameters()}
                expected_grad = {
                    name: p.grad.detach().clone()
                    for name, p in trainer.pipeline.model.named_parameters()
                    if p.grad is not None
                }
                loaded = copy.deepcopy(cfg)
                loaded.load_checkpoint = checkpoint
                loaded.timestamp += "-restored"
                restored = loaded.setup()
                restored.setup()
                assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
                    cuda_rng, torch.cuda.get_rng_state()
                ), "Training RNG restoration failed"
                assert restored.pipeline.datamanager._cursor == cursor and torch.equal(
                    restored.pipeline.datamanager._order, order
                )
                for name, p in restored.pipeline.model.named_parameters():
                    torch.testing.assert_close(p, pre_parameters[name], rtol=0, atol=0)

                def equal_state(first, second):
                    if isinstance(first, torch.Tensor):
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                    elif isinstance(first, dict):
                        assert first.keys() == second.keys()
                        for key in first:
                            equal_state(first[key], second[key])
                    elif isinstance(first, (list, tuple)):
                        assert len(first) == len(second)
                        for one, two in zip(first, second):
                            equal_state(one, two)
                    else:
                        assert first == second

                for name, opt in restored.optimizers.optimizers.items():
                    equal_state(opt.state_dict(), pre_optimizers[name])
                actual_loss = advance(restored)
                with trainer.pipeline.evaluation_context(), restored.pipeline.evaluation_context(), torch.no_grad():
                    expected_image = trainer.pipeline.model.get_outputs_for_camera(batch[0])["ultrasound"]
                    actual_image = restored.pipeline.model.get_outputs_for_camera(batch[0])["ultrasound"]
                differences = {
                    name: {
                        "raw_max": float((p - expected[name]).abs().max()) if p.numel() else 0.0,
                        "gradient_max": float(expected_grad[name].abs().max()) if name in expected_grad else None,
                        "gradient_difference_max": float((p.grad - expected_grad[name]).abs().max())
                        if p.grad is not None and name in expected_grad
                        else None,
                    }
                    for name, p in restored.pipeline.model.named_parameters()
                }
                diagnostics = {
                    "dataset": dataset,
                    "method": method,
                    "parameters": differences,
                    "loss_difference": float((expected_loss - actual_loss).abs()),
                    "render_abs_max": float((expected_image - actual_image).abs().max()),
                }
                print("CONTINUATION_DIAGNOSTICS", json.dumps(diagnostics), flush=True)
                torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(actual_image, expected_image, rtol=1e-5, atol=1e-7)
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
                        "continuation_validation": diagnostics,
                        "rng_optimizer_sampler_restored_exact": True,
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
