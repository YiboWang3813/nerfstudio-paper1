"""Build paper1 figures/tables from frozen local evidence; never starts training."""

import argparse
import hashlib
import importlib
import json
import sys

from nerfstudio.research_artifacts.paths import (
    RUNS,
    WORKSPACE,
    configure_object,
    evidence_root,
    load_artifact_config,
    mapped_path,
    prepare_view,
)
from nerfstudio.scripts.ultrasound_resources import resources

GENERATORS = (
    "ablation_tables",
    "main_table",
    "hyperparameters_tmi",
    "hyperparameters_original",
    "hyperparameters_endo_h12",
    "hyperparameters_endo_h3",
    "comparison_2d_tmi",
    "comparison_2d_original",
)


def build(name, object_id=None, config_path=None):
    """Each invocation gets a new output directory; frozen inputs are never overwritten."""
    import time  # noqa: PLC0415 - import under the shared-runtime lock

    definition, config_source = load_artifact_config(object_id, config_path)
    definition.setdefault("generator", name)
    configure_object(definition)
    workspace = prepare_view()
    output = WORKSPACE / "outputs/paper-builds" / name / str(time.time_ns())
    output.mkdir(parents=True)
    module = importlib.import_module("nerfstudio.research_artifacts.generators." + name)
    source = evidence_root(definition.get("frozen_inputs", {}).get("archive", "outputs/paper-evidence/inputs-v1.zip"))
    for loaded_name, loaded in list(sys.modules.items()):
        if not loaded_name.startswith("nerfstudio.research_artifacts.generators.") or loaded is None:
            continue
        loaded.Path = mapped_path
        loaded.ROOT = loaded.PROJECT_ROOT = workspace
        loaded.SOURCE = (
            source / "cga/hyperparameter_data.tsv" if loaded_name.endswith("hyperparameters_endo_h3") else source
        )
        loaded.SRC = loaded.P = source
        loaded.OUT = loaded.OUT_DIR = loaded.OUT_ROOT = output
        loaded.BENCHMARK_ROOT = RUNS / "production/comparisons/us_recon_four_dataset_p100k_v1"
        selected_tables = (
            WORKSPACE / definition["asset_root"] / "inputs/selected-summary-tables"
            if definition.get("asset_root")
            else None
        )
        loaded.TABLE_ROOT = (
            selected_tables if selected_tables and selected_tables.is_dir() else loaded.BENCHMARK_ROOT / "tables"
        )
        loaded.FIGURE_DIR = output / "figures"
        loaded.MANIFEST_DIR = output / "manifests"
        loaded.CONFIG_PATH = output / "2d_rendering_figure_config_tmi_v2.md"
        loaded.FROZEN_MANIFEST = (
            source / definition["qualitative_manifest"] if definition.get("qualitative_manifest") else None
        )
    allowed = {
        "METHODS",
        "PAPER_METHODS",
        "METHOD_LABEL",
        "SUMMARY_FILES",
        "GROUPS",
        "PANELS",
        "DATASETS",
        "DS",
        "DATA",
        "MET",
        "METRICS",
        "COLORS",
        "MARKERS",
        "YELLOW",
        "BLUE",
        "ORANGE",
        "GRAY",
        "BLACK",
        "LIVER",
        "ARM",
        "GRID",
        "REFERENCE",
        "SPINE",
        "ROWS",
    }
    for key, value in definition.get("parameters", {}).items():
        if key not in allowed or not hasattr(module, key):
            raise ValueError(f"Unsupported {name} figure/table parameter: {key}")
        if key == "ROWS":
            value = [module.RowSpec(**row) for row in value]
        setattr(module, key, value)
    if hasattr(module, "SUMMARY_PRIORITY"):
        module.SUMMARY_PRIORITY = {value: i for i, value in enumerate(module.SUMMARY_FILES)}
    if name == "main_table":
        module.NAMES = dict(module.METHODS)
    (output / "artifact-config.json").write_text(json.dumps(definition, indent=2) + "\n")
    from PIL import Image  # noqa: PLC0415 - plotting runtime import under the shared lock

    opened_images = {}
    original_open = Image.open

    def tracked_open(file, *args, **kwargs):
        if isinstance(file, (str, type(output))):
            path = type(output)(file)
            if path.is_file():
                opened_images[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return original_open(file, *args, **kwargs)

    source_reads = set()
    recording = [True]

    def audit_source_reads(event, args):
        if recording[0] and event == "open" and isinstance(args[0], (str, bytes)):
            value = args[0].decode() if isinstance(args[0], bytes) else args[0]
            if value.startswith(str(source) + "/"):
                source_reads.add(value)

    sys.addaudithook(audit_source_reads)
    Image.open = tracked_open
    try:
        module.main()
    finally:
        recording[0] = False
        Image.open = original_open
    files = [
        {
            "path": str(p.relative_to(output)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    reads = {}
    for field in ("INPUTS", "HASHES"):
        value = getattr(module, field, {})
        if isinstance(value, dict):
            reads.update({str(k): str(v) for k, v in value.items()})
    receipt = {
        "generator": name,
        "object": definition.get("id", object_id),
        "config_source": str(config_source) if config_source else None,
        "config_sha256": hashlib.sha256(config_source.read_bytes()).hexdigest() if config_source else None,
        "effective_config": "artifact-config.json",
        "frozen_reads": reads,
        "snapshot_inputs": {
            str(type(output)(p).relative_to(source)): hashlib.sha256(type(output)(p).read_bytes()).hexdigest()
            for p in sorted(source_reads)
        },
        "image_inputs": opened_images,
        "generator_sha256": hashlib.sha256(type(output)(module.__file__).read_bytes()).hexdigest(),
        "training_started": False,
        "source_snapshots_mutated": False,
        "output": str(output),
        "files": files,
    }
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("ARTIFACT_BUILD_COMPLETE", name, str(output), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generator", choices=GENERATORS, nargs="?")
    parser.add_argument("--object", dest="object_id")
    parser.add_argument("--config", help="Figure/table project JSON in docs")
    args = parser.parse_args()
    if args.object_id and args.config:
        parser.error("Choose --object or --config")
    generator = args.generator
    if args.object_id or args.config:
        definition, _ = load_artifact_config(args.object_id, args.config)
        generator = definition.get("generator")
    if generator not in GENERATORS:
        parser.error("Select a supported generator, --object, or executable --config")
    with resources(gpu=False):
        build(generator, args.object_id, args.config)


if __name__ == "__main__":
    main()
