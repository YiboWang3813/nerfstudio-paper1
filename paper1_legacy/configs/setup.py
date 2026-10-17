import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from .root import RootConfig
from .base.models import MODEL_CONFIG_REGISTRY
from .base.scene import DATASET_META_REGISTRY
from utils.experiment_management import (
    build_run_manifest,
    default_run_dir,
    ensure_run_subdirs,
    save_experiment_snapshot,
    short_run_id,
    slugify,
    write_run_manifest,
)


def _bind_root_schema(raw_cfg) -> RootConfig:
    if "model" not in raw_cfg or "name" not in raw_cfg.model:
        raise ValueError("Config must specify 'model.name'")

    model_name = raw_cfg.model.name
    if model_name not in MODEL_CONFIG_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    schema = OmegaConf.structured(RootConfig)
    schema.model = OmegaConf.structured(MODEL_CONFIG_REGISTRY[model_name])
    cfg = OmegaConf.merge(schema, raw_cfg)
    return cfg


def load_root_config(yaml_path: str) -> RootConfig:
    raise RuntimeError(
        "Legacy YAML loading has been removed. "
        "Use Hydra aliases such as `python run.py "
        "experiment=preproduction/baseline_reproduction/ultragray/smoke`, "
        "or compose configs through configs.setup.load_hydra_config()."
    )


def load_hydra_config(experiment: str, overrides: list = None) -> RootConfig:
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs_hydra"))
    hydra_overrides = [f"experiment={experiment}"]
    hydra_overrides.extend(overrides or [])
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=hydra_overrides)
    return prepare_config(cfg)


def _has_run_artifacts(path: str) -> bool:
    return (
        os.path.isdir(os.path.join(path, "checkpoints"))
        or os.path.exists(os.path.join(path, "config_dump.yaml"))
        or os.path.exists(os.path.join(path, "run_manifest.yaml"))
    )


def _resolve_output_dir(cfg: RootConfig, hydra_output_dir: str = None) -> None:
    run_id = str(cfg.system.run_id or "").strip()
    if not run_id and hydra_output_dir:
        tail = os.path.basename(os.path.abspath(str(hydra_output_dir)))
        if "_" in tail:
            run_id = tail.rsplit("_", 1)[-1]
    run_id = run_id or short_run_id()
    model_slug = slugify(cfg.model.name)
    family = slugify(getattr(cfg.experiment, "family", ""), "")
    name = slugify(getattr(cfg.experiment, "name", ""), "")
    exp_slug = "_".join(part for part in [family, name] if part)
    run_name = str(cfg.system.run_name or "").strip() or "_".join(
        part for part in [exp_slug, model_slug, run_id] if part
    )

    cfg.system.run_id = run_id
    cfg.system.run_name = run_name

    if hydra_output_dir:
        cfg.system.hydra_output_dir = os.path.abspath(os.path.expanduser(str(hydra_output_dir)))
        cfg.system.output_dir = cfg.system.hydra_output_dir
        return

    explicit_out = str(cfg.system.output_dir or "").strip()
    if explicit_out:
        base_out = os.path.abspath(os.path.expanduser(explicit_out))
        if _has_run_artifacts(base_out):
            cfg.system.output_dir = base_out
            return
        if cfg.system.use_legacy_output_naming:
            exp_name = f"{cfg.scene.dataset_name}_{cfg.model.name}"
            if cfg.system.exp_tag:
                exp_name += f"_{cfg.system.exp_tag}"
            cfg.system.output_dir = os.path.join(base_out, exp_name)
            return
        cfg.system.output_dir = base_out
        return

    cfg.system.output_dir = default_run_dir(cfg.system.output_root, cfg.model.name, run_id)


def _resolve_checkpoint_dir(cfg: RootConfig) -> None:
    explicit_ckpt = str(getattr(cfg.system, "checkpoint_dir", "") or "").strip()
    if explicit_ckpt:
        cfg.system.checkpoint_dir = os.path.abspath(os.path.expanduser(explicit_ckpt))
        return

    checkpoint_root = str(getattr(cfg.system, "checkpoint_root", "") or "").strip()
    if not checkpoint_root:
        cfg.system.checkpoint_dir = os.path.join(cfg.system.output_dir, "checkpoints")
        return

    checkpoint_root = os.path.abspath(os.path.expanduser(os.path.expandvars(checkpoint_root)))
    output_dir = os.path.abspath(os.path.expanduser(str(cfg.system.output_dir)))
    output_root = os.path.abspath(os.path.expanduser(os.path.expandvars(str(cfg.system.output_root))))
    try:
        if os.path.commonpath([output_dir, output_root]) == output_root:
            rel_output = os.path.relpath(output_dir, output_root)
        else:
            rel_output = str(cfg.system.run_name)
    except ValueError:
        rel_output = str(cfg.system.run_name)
    cfg.system.checkpoint_dir = os.path.join(checkpoint_root, rel_output, "checkpoints")


def prepare_config(raw_cfg) -> RootConfig:
    return _bind_root_schema(raw_cfg)


def setup_experiment(
    hydra_cfg,
    cli_args: list = None,
    hydra_output_dir: str = None,
) -> RootConfig:
    """
    Bind a Hydra-composed config to the structured schema and set up a run.
    """
    cfg = prepare_config(hydra_cfg)

    if cfg.scene.dataset_name not in DATASET_META_REGISTRY:
         raise ValueError(f"Dataset '{cfg.scene.dataset_name}' not found in registry.")

    if cfg.system.seed is not None:
        seed = cfg.system.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cfg.system.config_path = f"hydra:experiment={cfg.experiment.family}/{cfg.experiment.name}"
    _resolve_output_dir(cfg, hydra_output_dir=hydra_output_dir)
    _resolve_checkpoint_dir(cfg)
    ensure_run_subdirs(
        cfg.system.output_dir,
        checkpoint_dir=cfg.system.checkpoint_dir,
        link_checkpoint_dir=bool(getattr(cfg.system, "link_checkpoint_dir", True)),
    )

    save_experiment_snapshot(
        cfg,
        cfg.system.output_dir,
        cfg.system.config_path,
        cli_args or [],
        hydra_output_dir=hydra_output_dir,
    )
    write_run_manifest(
        cfg.system.output_dir,
        build_run_manifest(
            cfg,
            cfg.system.config_path,
            cli_args or [],
            hydra_output_dir=hydra_output_dir,
        ),
    )
        
    return cfg
