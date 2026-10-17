import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from omegaconf import OmegaConf


def short_run_id() -> str:
    return uuid.uuid4().hex[:7]


def slugify(value: Any, default: str = "run") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or default


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def ensure_run_subdirs(
    output_dir: str,
    checkpoint_dir: Optional[str] = None,
    link_checkpoint_dir: bool = True,
) -> None:
    for subdir in ["logs", "vis"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir or os.path.join(output_dir, "checkpoints")))
    os.makedirs(checkpoint_dir, exist_ok=True)

    local_ckpt_dir = os.path.join(output_dir, "checkpoints")
    if os.path.abspath(local_ckpt_dir) == checkpoint_dir:
        os.makedirs(local_ckpt_dir, exist_ok=True)
        return

    if not link_checkpoint_dir:
        return

    if os.path.islink(local_ckpt_dir):
        current_target = os.readlink(local_ckpt_dir)
        current_abs = os.path.abspath(os.path.join(os.path.dirname(local_ckpt_dir), current_target))
        if current_abs != checkpoint_dir:
            os.unlink(local_ckpt_dir)
            os.symlink(checkpoint_dir, local_ckpt_dir)
        return

    if os.path.isdir(local_ckpt_dir):
        if os.listdir(local_ckpt_dir):
            print(
                "[Experiment] Existing non-empty local checkpoints directory left in place: "
                f"{local_ckpt_dir}. New checkpoints will be written to {checkpoint_dir}."
            )
            return
        os.rmdir(local_ckpt_dir)
    elif os.path.exists(local_ckpt_dir):
        print(
            "[Experiment] Existing checkpoints path is not a directory and was left in place: "
            f"{local_ckpt_dir}. New checkpoints will be written to {checkpoint_dir}."
        )
        return

    os.symlink(checkpoint_dir, local_ckpt_dir)


def default_run_dir(output_root: str, model_name: str, run_id: str) -> str:
    now = datetime.now()
    date_part = now.strftime("%Y-%m-%d")
    time_part = now.strftime("%H-%M-%S")
    name = f"{time_part}_{slugify(model_name)}_{slugify(run_id)}"
    return os.path.abspath(os.path.join(os.path.expanduser(output_root), date_part, name))


def copy_source_config(config_path: str, output_dir: str) -> Optional[str]:
    if not config_path:
        return None
    if not os.path.exists(config_path):
        return None
    dst = os.path.join(output_dir, "source_config.yaml")
    shutil.copyfile(config_path, dst)
    return dst


def save_text_yaml(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
            if data and not data.endswith("\n"):
                f.write("\n")
        else:
            OmegaConf.save(config=OmegaConf.create(data), f=f)


def save_experiment_snapshot(
    cfg: Any,
    output_dir: str,
    config_path: str,
    cli_overrides: Iterable[str],
    hydra_output_dir: Optional[str] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_dir = os.path.join(output_dir, ".omniultra")
    os.makedirs(meta_dir, exist_ok=True)

    OmegaConf.save(cfg, os.path.join(output_dir, "config_dump.yaml"))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), os.path.join(output_dir, "resolved_config.yaml"))

    overrides = list(cli_overrides or [])
    save_text_yaml(os.path.join(output_dir, "overrides.yaml"), {"overrides": overrides})
    save_text_yaml(os.path.join(meta_dir, "overrides.yaml"), {"overrides": overrides})
    save_text_yaml(
        os.path.join(meta_dir, "runtime.yaml"),
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": output_dir,
            "checkpoint_dir": str(getattr(cfg.system, "checkpoint_dir", "")),
            "hydra_output_dir": hydra_output_dir or "",
            "config_path": config_path,
        },
    )
    copy_source_config(config_path, output_dir)


def write_run_manifest(output_dir: str, manifest: Dict[str, Any]) -> None:
    save_text_yaml(os.path.join(output_dir, "run_manifest.yaml"), manifest)


def update_run_manifest(output_dir: str, patch: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, "run_manifest.yaml")
    current: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            current = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
        except Exception:
            current = {}
    current.update(patch)
    write_run_manifest(output_dir, current)


def build_run_manifest(
    cfg: Any,
    config_path: str,
    cli_overrides: Iterable[str],
    hydra_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_id": str(cfg.system.run_id),
        "run_name": str(cfg.system.run_name),
        "output_dir": str(cfg.system.output_dir),
        "checkpoint_dir": str(getattr(cfg.system, "checkpoint_dir", "")),
        "checkpoint_root": str(getattr(cfg.system, "checkpoint_root", "")),
        "hydra_output_dir": str(hydra_output_dir or ""),
        "config_path": str(config_path),
        "overrides": list(cli_overrides or []),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dataset": str(cfg.scene.dataset_name),
        "model": str(cfg.model.name),
        "exp_tag": str(cfg.system.exp_tag or ""),
        "seed": cfg.system.seed,
        "wandb": {
            "enabled": bool(getattr(cfg.wandb, "enable", False)),
            "project": str(getattr(cfg.wandb, "project", "")),
            "mode": str(getattr(cfg.wandb, "mode", "")),
            "run_id": "",
        },
    }


def env_json(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default
