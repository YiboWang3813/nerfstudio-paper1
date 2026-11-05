"""Translate retained paper1 experiment recipes to native Nerfstudio configs.

Frozen source YAML is provenance only. No legacy Python code or environment is
imported, no dataset is modified, and preparing a config never starts training.
"""

import hashlib
import json
from pathlib import Path

import yaml

from nerfstudio.configs.ultrasound_protocol import ours_config, production_config


def native_frozen_config(source: Path, output: Path):
    frozen = yaml.safe_load(source.read_text())
    scene, loading = frozen["scene"], frozen["scene"]["loading"]
    if frozen["model"]["name"] != "ours":
        raise ValueError(
            "This converter handles retained Ours ablation protocols; baseline presets use production_config"
        )
    if loading["test_mode"] != "holdout_interval" or (
        loading["frame_start"],
        loading["frame_end"],
        loading["frame_stride"],
    ) != (0, -1, 1):
        raise ValueError("Unsupported source frame selection; an explicit frozen dataset version is required")
    dataset, sweep = loading.get("dataset_dirname") or scene["dataset_name"], loading["sweep_name"]
    manifest = Path("/home/yibo/shared/datasets") / dataset / "p100k-native-v2" / sweep / "manifest.json"
    data = json.loads(manifest.read_text())
    cfg = production_config("ours", manifest, output)
    converted = ours_config(frozen["model"])
    cfg.pipeline.model, cfg.optimizers = converted.pipeline.model, converted.optimizers
    cfg.pipeline.model.full_metrics = True
    if cfg.pipeline.model.renderer.elev_corridor_half_mm is None:
        cfg.pipeline.model.renderer.elev_corridor_half_mm = (
            float(frozen["model"]["slice_render"]["num_slices_elevational"])
            * data["geometry"]["elevational_spacing_mm"]
            * 0.5
        )
    sampler = scene["sampler"]
    cfg.pipeline.model.num_gaussians = int(sampler["num_init"])
    for attr, key in [
        ("uniform_ratio", "uniform_ratio"),
        ("intensity_threshold", "intensity_threshold"),
        ("jitter_radius_mm", "jitter_radius"),
        ("initial_scales_mm", "scales_init"),
        ("initial_quats_wxyz", "quats_init"),
    ]:
        setattr(cfg.pipeline, attr, sampler[key])
    losses = frozen["loss"]
    if set(losses["needed_losses"]) != {"l1", "ssim"}:
        raise ValueError("Retained Ours protocol has unsupported loss terms")
    cfg.pipeline.model.l1_weight = losses["lambdas"]["l1"]
    cfg.pipeline.model.ssim_weight = losses["lambdas"]["ssim"]
    cfg.machine.seed = cfg.pipeline.datamanager.seed = int(frozen["system"]["seed"])
    cfg.max_num_iterations = int(frozen["engine"]["iterations"])
    cfg.steps_per_eval_all_images = cfg.max_num_iterations - 1
    cfg.steps_per_save = cfg.max_num_iterations
    parser = cfg.pipeline.datamanager.dataparser
    if loading["pose_mode"] == "canonical_sweep":
        parser.pose_mode = "canonical"
        parser.canonical_axis = loading["canonical_sweep_axis"]
        parser.canonical_spacing_mode = loading["canonical_sweep_spacing_mode"]
        parser.canonical_spacing_scale = float(loading["canonical_sweep_delta_scale"])
        delta = float(loading["canonical_sweep_delta_mm"])
        parser.canonical_spacing_mm = (
            delta if delta > 0 else data["geometry"]["elevational_spacing_mm"]
        ) * parser.canonical_spacing_scale
        ids = {
            index: frame
            for split in data["splits"].values()
            for index, frame in zip(split["indices"], split["frame_ids"])
        }
        parser.canonical_reference_frame_id = ids[max(0, int(loading["canonical_sweep_ref_frame"]))]
    elif loading["pose_mode"] != "file":
        raise ValueError("Unsupported source pose mode")
    return cfg, {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_version": data["dataset_version"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "native_converter": "nerfstudio.configs.paper1_frozen_protocol.native_frozen_config",
        "training_started": False,
        "historical_results_relabelled": False,
    }
