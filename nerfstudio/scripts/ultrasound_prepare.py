"""Prepare and check native formal configurations; never launch training."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from nerfstudio.configs.ultrasound_protocol import METHODS, production_config
from nerfstudio.utils.ultrasound_provenance import execution_identity


def prepare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Native experiment output root")
    parser.add_argument("--config", type=Path, required=True, help="Prepared config YAML destination")
    parser.add_argument("--pose-mode", choices=("file", "canonical"), default="file")
    parser.add_argument("--canonical-spacing-mm", type=float)
    parser.add_argument("--local-conditioning", choices=("none", "fixed", "adaptive"))
    parser.add_argument("--densification", choices=("gradient", "residual_guided"))
    parser.add_argument("--pruning", choices=("none", "opacity", "local_aware", "path_aware"))
    parser.add_argument("--global-attenuation", choices=("off", "on"))
    args = parser.parse_args()
    if args.config.exists():
        raise FileExistsError("Refusing to overwrite a prepared experiment configuration")
    cfg = production_config(args.method, args.manifest, args.output)
    if args.pose_mode == "canonical":
        config = cfg.pipeline.datamanager.dataparser
        config.pose_mode = "canonical"
        config.canonical_spacing_mm = (
            args.canonical_spacing_mm
            if args.canonical_spacing_mm is not None
            else json.loads(args.manifest.read_text())["geometry"]["elevational_spacing_mm"]
        )
    options = (args.local_conditioning, args.densification, args.pruning, args.global_attenuation)
    if any(value is not None for value in options):
        if args.method != "ours":
            raise ValueError("Ours ablation options require method ours")
        model = cfg.pipeline.model
        if args.local_conditioning:
            model.renderer.local_conditioning_mode = args.local_conditioning
        if args.densification:
            model.density.densification = args.densification
        if args.pruning:
            model.density.pruning = args.pruning
        if args.global_attenuation:
            model.renderer.ray_attenuation_enabled = args.global_attenuation == "on"
        model.renderer.__post_init__()
        if model.density.pruning == "path_aware" and not model.renderer.ray_attenuation_enabled:
            raise ValueError(
                "Path-aware pruning requires global attenuation; choose an explicit local pruning ablation"
            )
    # Verify actual NPY hashes, poses and frozen splits without allocating a model.
    data = cfg.pipeline.datamanager.dataparser.setup()
    train, test = data.get_dataparser_outputs("train"), data.get_dataparser_outputs("test")
    from nerfstudio.model_components.ultrasound_metrics import verify_assets

    assets = verify_assets()
    text = yaml.dump(cfg)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(text)
    receipt = {
        "status": "prepared_not_started",
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "method": args.method,
        "iterations": cfg.max_num_iterations,
        "gaussians": getattr(cfg.pipeline.model, "num_gaussians", None),
        "train_frames": len(train.cameras),
        "test_frames": len(test.cameras),
        "pose_mode": args.pose_mode,
        "ablation_overrides": dict(
            zip(("local_conditioning", "densification", "pruning", "global_attenuation"), options)
        ),
        "manifest_sha256": train.metadata["manifest_sha256"],
        "dataset_version": train.metadata["dataset_version"],
        "metric_assets": assets,
        "execution": execution_identity(),
    }
    args.config.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def preflight():
    parser = argparse.ArgumentParser(
        description="Read-only validation of all 78 sweeps and eight native formal methods"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for dataset in ("synthetic_liver", "spine_phantom", "tus_rec_2024_qc", "tdsc_abus_2023"):
        root = Path("/home/yibo/shared/datasets") / dataset / "p100k-native-v2"
        catalog = json.loads((root / "dataset.json").read_text())
        publication = json.loads((root / "publication.json").read_text())
        if publication["status"] != "published_verified":
            raise ValueError("Formal dataset version is not published and verified")
        for package in catalog["packages"]:
            archive = Path(package["package"])
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != package["sha256"]:
                    raise ValueError("Dataset package cache checksum mismatch")
            manifest = root / package["sweep"] / "manifest.json"
            for name, expected in package["files"].items():
                with (manifest.parent / name).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError("Frozen dataset member differs")
            parser_config = None
            for method in METHODS:
                cfg = production_config(method, manifest, Path("/home/yibo/paper1/outputs/formal"))
                # Round-trip official Nerfstudio serialization for every recipe.
                restored = yaml.load(yaml.dump(cfg), Loader=yaml.Loader)
                assert restored.pipeline.model._target is cfg.pipeline.model._target
                assert cfg.max_num_iterations == 30000
                assert getattr(cfg.pipeline.model, "num_gaussians", 1000000) == 1000000
                parser_config = cfg.pipeline.datamanager.dataparser
            data = parser_config.setup()
            train, test = data.get_dataparser_outputs("train"), data.get_dataparser_outputs("test")
            records.append(
                {
                    "dataset": dataset,
                    "sweep": package["sweep"],
                    "manifest_sha256": train.metadata["manifest_sha256"],
                    "train_frames": len(train.cameras),
                    "test_frames": len(test.cameras),
                    "methods": list(METHODS),
                    "publication_manifest_sha256": publication["manifest_sha256"],
                }
            )
    from nerfstudio.model_components.ultrasound_metrics import verify_assets

    result = {
        "status": "configuration_and_data_verified",
        "formal_training_started": False,
        "sweeps": records,
        "native_recipes": len(records) * len(METHODS),
        "metric_assets": verify_assets(),
        "execution": execution_identity(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {len(records)} sweeps and {result['native_recipes']} native recipes; training not started.")
