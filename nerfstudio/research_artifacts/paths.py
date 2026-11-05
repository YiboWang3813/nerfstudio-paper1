"""Resolve frozen evidence and historical run paths without a documentation mirror."""

import fcntl
import hashlib
import json
import shutil
import zipfile
from functools import lru_cache
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
RUNS = WORKSPACE / "outputs/eehpc"
REGISTRY = WORKSPACE / "nerfstudio/experiments/paper1/artifacts.json"
DOCS = WORKSPACE / "docs"
VIEW = WORKSPACE / "migration/complete-assets-20261105/compat-workspace"
_OBJECT_ALIASES = {}


@lru_cache(maxsize=1)
def evidence_root(archive_path="outputs/frozen-inputs/paper-evidence-v1.zip"):
    """Validate the immutable input package and expand it once into shared cache."""
    archive = WORKSPACE / archive_path
    manifest = json.loads(archive.with_suffix(".json").read_text())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(f"Frozen evidence checksum mismatch: {archive}")
    cache = WORKSPACE.parent / "shared/cache/paper1-artifacts/evidence" / digest
    cache.mkdir(parents=True, exist_ok=True)
    payload = cache / "payload"
    with (cache / "extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache / "complete").exists():
            shutil.rmtree(payload, ignore_errors=True)
            payload.mkdir()
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if not (payload / member.filename).resolve().is_relative_to(payload.resolve()):
                        raise ValueError(f"Unsafe evidence member: {member.filename}")
                source.extractall(payload)
            (cache / "complete").write_text(digest + "\n")
    return payload


def load_artifact_config(object_id=None, config_path=None):
    """Resolve a versioned figure/table project from docs or an explicit config."""
    if config_path is not None:
        path = Path(config_path).resolve()
    elif object_id:
        registry = json.loads(REGISTRY.read_text())
        if object_id not in registry:
            raise ValueError(f"Unknown artifact object: {object_id}")
        path = WORKSPACE / registry[object_id]
    else:
        return {}, None
    return json.loads(path.read_text()), path


def configure_object(definition):
    global _OBJECT_ALIASES
    _OBJECT_ALIASES = {key: WORKSPACE / value for key, value in definition.get("input_aliases", {}).items()}


def mapped_path(value=".", *parts):
    value = str(value)
    for original in sorted(_OBJECT_ALIASES, key=len, reverse=True):
        if value == original or value.startswith(original + "/"):
            return Path(_OBJECT_ALIASES[original], value[len(original) :].lstrip("/"), *parts)
    marker = "final_paper_evidence_20260910"
    if marker in value.split("/"):
        suffix = value.split(marker, 1)[1].lstrip("/")
        return evidence_root().joinpath(suffix, *parts)
    roots = {
        "/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs": RUNS,
        "/home/boywang25/ultrasound-scene-reconstruction/omniultra": VIEW,
        "/home/boywang25/paper1/results/legacy/outputs": RUNS,
        "/home/yibo/paper1/results/archive/eehpc-20261105/legacy/outputs": RUNS,
        "/home/boywang25/paper1/data": Path("/home/yibo/shared/datasets/paper1-unpublished/eehpc-20261017"),
        "outputs": RUNS,
    }
    for original, target in roots.items():
        if value == original or value.startswith(original + "/"):
            return Path(target, value[len(original) :].lstrip("/"), *parts)
    return Path(value, *parts)


def prepare_view():
    """Keep historical run resolution local; documentation has no compatibility tree."""
    VIEW.mkdir(parents=True, exist_ok=True)
    old_docs = VIEW / "docs"
    if old_docs.is_dir() and all(p.is_symlink() for p in old_docs.iterdir()):
        for link in old_docs.iterdir():
            link.unlink()
        old_docs.rmdir()
    link = VIEW / "outputs"
    if link.is_symlink() and link.readlink() != RUNS:
        link.unlink()
    if not link.exists():
        link.symlink_to(RUNS, target_is_directory=True)
    return VIEW
