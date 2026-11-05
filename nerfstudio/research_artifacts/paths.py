"""Resolve historical source references while preserving stored experiment IDs."""

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
RESULTS = WORKSPACE / "results/archive/eehpc-20261105"
DOCS = WORKSPACE / "docs/archive/eehpc-20261105/supplement"
VIEW = WORKSPACE / "migration/complete-assets-20261105/compat-workspace"


_OBJECT_ALIASES = {}


def configure_object(object_id=None):
    global _OBJECT_ALIASES
    _OBJECT_ALIASES = {}
    if object_id:
        root = WORKSPACE / "results/objects" / object_id
        definition = json.loads((root / "object.json").read_text())
        _OBJECT_ALIASES = {key: root / value for key, value in definition.get("input_aliases", {}).items()}


def mapped_path(value=".", *parts):
    value = str(value)
    for original in sorted(_OBJECT_ALIASES, key=len, reverse=True):
        if value == original or value.startswith(original + "/"):
            return Path(_OBJECT_ALIASES[original], value[len(original) :].lstrip("/"), *parts)
    roots = {
        "/home/boywang25/ultrasound-scene-reconstruction/omniultra/outputs": RESULTS / "legacy/outputs",
        "/home/boywang25/ultrasound-scene-reconstruction/omniultra/docs": VIEW / "docs",
        "/home/boywang25/ultrasound-scene-reconstruction/omniultra": VIEW,
        "/home/boywang25/paper1/results": RESULTS,
        "/home/boywang25/paper1/docs": DOCS,
        "/home/boywang25/paper1/data": Path("/home/yibo/shared/datasets/paper1-unpublished/eehpc-20261017"),
        "outputs": RESULTS / "legacy/outputs",
        "docs": VIEW / "docs",
    }
    for original, target in roots.items():
        if value == original or value.startswith(original + "/"):
            return Path(target, value[len(original) :].lstrip("/"), *parts)
    return Path(value, *parts)


def prepare_view():
    """Local path compatibility only; never exposes a source checkpoint directory."""
    (VIEW / "docs").mkdir(parents=True, exist_ok=True)
    for old, new in {
        "production": "04_experiments/production",
        "datasets": "02_datasets",
        "experiments": "04_experiments/protocol_notes",
        "methods": "01_problem_and_methods",
        "literature": "06_literature",
        "paper": "05_manuscript/legacy_writing",
    }.items():
        link = VIEW / "docs" / old
        target = DOCS / new
        if not target.exists():
            target = WORKSPACE / "docs/archive/eehpc-20261017" / new
        if link.is_symlink() and link.readlink() != target:
            link.unlink()
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
    for name in ("configs_hydra", "configs", "scripts"):
        link = VIEW / name
        target = (
            WORKSPACE
            / "nerfstudio/experiments/paper1/source_reference/eehpc-20261105/source_workspace/omniultra"
            / name
        )
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
    link = VIEW / "outputs"
    if not link.is_symlink():
        link.symlink_to(RESULTS / "legacy/outputs", target_is_directory=True)
    return VIEW
