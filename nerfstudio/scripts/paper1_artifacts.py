"""Build paper1 figures/tables from frozen local evidence; never starts training."""

import argparse
import hashlib
import importlib
import json
import sys

from nerfstudio.research_artifacts.paths import (
    RESULTS,
    WORKSPACE,
    configure_object,
    evidence_root,
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


def build(name, object_id=None):
    """Each invocation gets a new output directory; frozen inputs are never overwritten."""
    import time  # noqa: PLC0415 - import under the shared-runtime lock

    configure_object(object_id)
    workspace = prepare_view()
    output = WORKSPACE / "results/artifact-builds" / name / str(time.time_ns())
    output.mkdir(parents=True)
    module = importlib.import_module("nerfstudio.research_artifacts.generators." + name)
    source = evidence_root()
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
        loaded.BENCHMARK_ROOT = RESULTS / "legacy/outputs/production/comparisons/us_recon_four_dataset_p100k_v1"
        selected_tables = (
            WORKSPACE / "results/objects" / object_id / "inputs/selected-summary-tables" if object_id else None
        )
        loaded.TABLE_ROOT = (
            selected_tables if selected_tables and selected_tables.is_dir() else loaded.BENCHMARK_ROOT / "tables"
        )
        loaded.FIGURE_DIR = output / "figures"
        loaded.MANIFEST_DIR = output / "manifests"
        loaded.CONFIG_PATH = output / "2d_rendering_figure_config_tmi_v2.md"
    from PIL import Image  # noqa: PLC0415 - plotting runtime import under the shared lock

    opened_images = {}
    original_open = Image.open

    def tracked_open(file, *args, **kwargs):
        if isinstance(file, (str, type(output))):
            path = type(output)(file)
            if path.is_file():
                opened_images[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return original_open(file, *args, **kwargs)

    Image.open = tracked_open
    try:
        module.main()
    finally:
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
        "object": object_id,
        "frozen_reads": reads,
        "image_inputs": opened_images,
        "generator_sha256": hashlib.sha256(type(output)(module.__file__).read_bytes()).hexdigest(),
        "training_started": False,
        "source_snapshots_mutated": False,
        "output": str(output),
        "files": files,
    }
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if object_id:
        obj = WORKSPACE / "results/objects" / object_id
        target = obj / "builds" / output.name
        target.parent.mkdir(exist_ok=True)
        definition = json.loads((obj / "object.json").read_text())
        subset = definition.get("output_subset")
        if subset and (output / subset).is_dir():
            target.symlink_to(output / subset, target_is_directory=True)
        elif subset:
            target.mkdir()
            for path in output.iterdir():
                if path.name.startswith(subset) or path.name == "build-receipt.json":
                    (target / path.name).symlink_to(path, target_is_directory=path.is_dir())
        else:
            target.symlink_to(output, target_is_directory=True)
        link = obj / "outputs"
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(target.relative_to(obj), target_is_directory=True)
    print("ARTIFACT_BUILD_COMPLETE", name, str(output), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generator", choices=GENERATORS, nargs="?")
    parser.add_argument("--object", dest="object_id")
    args = parser.parse_args()
    generator = args.generator
    if args.object_id:
        if "/" in args.object_id or args.object_id.startswith("."):
            parser.error("Invalid object id")
        definition = json.loads((WORKSPACE / "results/objects" / args.object_id / "object.json").read_text())
        generator = definition["generator"]
    if generator not in GENERATORS:
        parser.error("Select a supported generator or --object")
    with resources(gpu=False):
        build(generator, args.object_id)


if __name__ == "__main__":
    main()
