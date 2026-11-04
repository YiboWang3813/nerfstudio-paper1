"""Frozen scientific presets mapped to native configs without a legacy runtime.

The snapshots are resolved once by the offline converter. Unsupported options
raise rather than silently reverting to a development default.
"""

import hashlib
import importlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

from gsplat.strategy.ultrasound import UltrasoundStrategy
from gsplat.ultrasound import UltrasoundRenderConfig

from nerfstudio.configs.ultrasound import ultrasound_config
from nerfstudio.configs.ultrasound_baselines import GAUSSIAN_METHODS, baseline_config, field_config
from nerfstudio.engine.optimizers import AdamOptimizerConfig

METHODS = ("ours",) + GAUSSIAN_METHODS + ("nerfus", "ultranerf")


def update_dataclass(instance, values, path="config"):
    """Recursive allow-list from native dataclasses; reject unknown keys."""
    allowed = {item.name for item in fields(instance)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unsupported {path} options: {sorted(unknown)}")
    for name, value in values.items():
        old = getattr(instance, name)
        if is_dataclass(old):
            if not isinstance(value, dict):
                raise TypeError(f"{path}.{name} must be an object")
            update_dataclass(old, value, path + "." + name)
        else:
            setattr(instance, name, tuple(value) if isinstance(old, tuple) else value)
    return instance


def ours_config(model):
    cfg = ultrasound_config()
    cfg.method_name = "ultrasound-ours"
    allowed = {
        "name",
        "render",
        "gauss_init",
        "scene_norm",
        "slice_render",
        "gauss_container",
        "offline",
        "pose_optimization",
        "pose_uncertainty",
    }
    if set(model) - allowed:
        raise ValueError("Unsupported Ours model components")
    native = cfg.pipeline.model
    render = dict(model["slice_render"])
    render.pop("num_slices_elevational", None)
    render.update(model["render"])
    native.renderer = update_dataclass(UltrasoundRenderConfig(1, 1, 1.0, 1.0, 0.0), render, "renderer")
    constraints = model["gauss_container"]["constraints"]
    initial = model["gauss_container"]["init_val"]
    if constraints["scale_activation"] != "bounded_sigmoid":
        raise ValueError("This frozen native protocol requires bounded_sigmoid scales")
    for name in ("scale_min_mm", "scale_max_mm", "echo_alpha_max", "attenuation_max"):
        setattr(native, name, constraints[name])
    native.initial_echo_alpha = initial["echo_alphas"]
    native.initial_attenuation = initial["attenuations"]
    dc = model["offline"]["density_control"]
    resp = dc["responsibility"]
    if resp["mix_mode"] != "responsibility" or resp["structure_weight_enabled"]:
        raise ValueError("Unconverted experimental density heuristic; use an explicit native strategy")
    densification = dc.get("densification_policy") or (
        "residual_guided" if dc["strategy"] == "contribution_residual" else "gradient"
    )
    if densification not in ("residual_guided", "gradient"):
        raise ValueError("Unsupported densification")
    pruning = dc.get("pruning_policy") or (
        ("path_aware" if resp["protect_attenuation_enabled"] else "local_aware")
        if dc["strategy"] == "contribution_residual" or resp["enabled"]
        else "opacity"
    )
    if not dc["enable"]["prune"]:
        pruning = "none"
    if pruning == "path_aware" and (
        not native.renderer.ray_attenuation_enabled
        or native.renderer.ray_attenuation_beta <= 0
        or native.renderer.ray_attenuation_strength <= 0
    ):
        pruning = "local_aware"
    if pruning == "path_aware" and resp["protect_attenuation_mode"] != "path_contribution":
        raise ValueError("Historical tau heuristics require a separate explicit native strategy")
    t, th = dc["timing"], dc["thresholds"]
    native.density = UltrasoundStrategy(
        enabled=dc["enable"]["densify"] or dc["enable"]["prune"],
        refine_start=t["start_iter"],
        refine_stop=t["end_iter"],
        refine_every=t["freq"],
        densify=dc["enable"]["densify"],
        densification=densification,
        pruning=pruning,
        gradient_threshold=th["densify_grad"],
        split_scale_mm=th["split_scale"],
        prune_echo_alpha=th["prune_opacity"],
        max_scale_mm=th.get("max_scale_mm", 0.0),
        score_quantile=resp["score_quantile"],
        min_score=resp["min_score"],
        max_fraction=resp["max_fraction"],
        low_contribution_quantile=resp["low_contribution_quantile"],
        high_residual_quantile=resp["high_residual_quantile"],
        path_contribution_quantile=resp["path_contribution_quantile"],
        path_threshold_mode=resp["path_contribution_threshold_mode"],
        path_threshold=resp["path_contribution_threshold"],
    )
    rates = model["gauss_container"]["lr"]
    for name in cfg.optimizers:
        cfg.optimizers[name]["optimizer"].lr = rates[name]
    from nerfstudio.cameras.ultrasound_poses import UltrasoundPoseConfig, UltrasoundUncertaintyConfig

    native.pose = update_dataclass(UltrasoundPoseConfig(), model.get("pose_optimization", {}), "pose")
    native.uncertainty = update_dataclass(
        UltrasoundUncertaintyConfig(), model.get("pose_uncertainty", {}), "uncertainty"
    )
    if native.pose.enabled and native.pose.trainable:
        for name, lr, enabled in [
            ("pose_rot", native.pose.lr_rot, native.pose.optimize_rotation),
            ("pose_trans", native.pose.lr_trans, native.pose.optimize_translation),
        ]:
            if enabled:
                cfg.optimizers[name] = {
                    "optimizer": AdamOptimizerConfig(lr=native.pose.lr if lr is None else lr, eps=1e-15),
                    "scheduler": None,
                }
    return cfg


def production_config(method, manifest: Path, output: Path):
    if method not in METHODS:
        raise ValueError(method)
    source = json.loads((Path(__file__).parent / "ultrasound_protocols" / f"{method}.json").read_text())
    model = source["model"]
    if method == "ours":
        cfg = ours_config(model)
    else:
        algorithm = importlib.import_module(
            "nerfstudio.model_components.ultrasound_baselines." + method + ".configs"
        ).ModelConfig()
        update_dataclass(algorithm, model, "model")
        cfg = baseline_config(method, algorithm) if method in GAUSSIAN_METHODS else field_config(method, algorithm)
    data = json.loads(manifest.read_text())
    catalog = json.loads((manifest.parent.parent / "dataset.json").read_text())
    package = next(item for item in catalog["packages"] if item["sweep"] == manifest.parent.name)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != package["files"]["manifest.json"]:
        raise ValueError("Frozen manifest differs from the versioned package identity")
    dataset = manifest.parents[2].name
    if data["dataset_version"] != "p100k-native-v2":
        raise ValueError("Formal native protocol requires frozen v2 data")
    if (
        data["split_protocol"]["trim_each"] != (8 if dataset == "spine_phantom" else 10)
        or data["split_protocol"]["ratio"] != 0.2
    ):
        raise ValueError("Dataset split differs from formal protocol")
    cfg.pipeline.datamanager.dataparser.data = manifest.resolve()
    if dataset == "synthetic_liver":
        # Frozen l2/l3 contain about 1% nonorthogonality. Preserve source affine
        # geometry explicitly instead of projecting it to a different experiment.
        cfg.pipeline.datamanager.dataparser.pose_validation = "preserve_source_affine"
    cfg.pipeline.datamanager.seed = source["seed"]
    cfg.machine.seed = source["seed"]
    cfg.output_dir = output.resolve()
    cfg.experiment_name = f"{dataset}-{manifest.parent.name}"
    cfg.set_timestamp()
    cfg.max_num_iterations = source["iterations"]
    cfg.steps_per_eval_batch = cfg.steps_per_eval_image = 0
    # Native Trainer iterates 0..N-1 and step_check tests step % interval.
    # Evaluate after the Nth optimizer update, before its automatic final save.
    cfg.steps_per_eval_all_images = source["iterations"] - 1
    cfg.steps_per_save = source["iterations"]
    cfg.save_only_latest_checkpoint = True
    cfg.pipeline.evaluation_output = cfg.get_base_dir() / "evaluation"
    cfg.pipeline.model.full_metrics = True
    if dataset == "spine_phantom" and method == "ultragray":
        cfg.pipeline.model.algorithm.render.local_transmission_max_window_mm = 0.25
    if dataset == "tus_rec_2024_qc" and method == "slice_adapted_3dgs":
        cfg.pipeline.model.algorithm.render.slice_thickness_ratio = 0.15
    if method == "ours":
        cfg.pipeline.model.renderer.elev_corridor_half_mm = {
            "synthetic_liver": 0.395,
            "spine_phantom": 0.395,
            "tus_rec_2024_qc": 0.202,
            "tdsc_abus_2023": 0.238,
        }[dataset]
    if method in ("nerfus", "ultranerf"):
        cfg.pipeline.model.eval_seed_iteration = source["iterations"]
    else:
        cfg.pipeline.model.num_gaussians = source["sampler"]["num_init"]
    sampler = source["sampler"]
    for name, key in [
        ("uniform_ratio", "uniform_ratio"),
        ("intensity_threshold", "intensity_threshold"),
        ("jitter_radius_mm", "jitter_radius"),
        ("initial_scales_mm", "scales_init"),
        ("initial_quats_wxyz", "quats_init"),
    ]:
        setattr(cfg.pipeline, name, tuple(sampler[key]) if isinstance(sampler[key], list) else sampler[key])
    needed = set(source["loss"]["needed_losses"])
    cfg.pipeline.model.l1_weight = source["loss"]["lambdas"]["l1"] if "l1" in needed else 0.0
    cfg.pipeline.model.ssim_weight = source["loss"]["lambdas"]["ssim"]
    if method in ("nerfus", "ultranerf"):
        if needed != {"mse", "ssim"}:
            raise ValueError("Unsupported implicit losses")
        cfg.pipeline.model.mse_weight = source["loss"]["lambdas"]["mse"]
    elif needed != {"l1", "ssim"}:
        raise ValueError("Unsupported Gaussian losses")
    return cfg
