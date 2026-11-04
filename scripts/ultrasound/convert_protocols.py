"""Offline-only Hydra resolution into version-controlled native snapshots.

This command reads reference configuration files; it never imports or executes
the legacy Engine, Scene, methods, solvers, or callbacks. Native runs use JSON
snapshots and native dataclasses exclusively.
"""

import argparse
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2] / "paper1_legacy/configs_hydra"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(args.source.resolve()), version_base=None):
        for method in (
            "ours",
            "ultragauss",
            "ultragray",
            "gaussianpile",
            "slice_adapted_3dgs",
            "ultrags_baseline",
            "nerfus",
            "ultranerf",
        ):
            alias = (
                "gs_strict/" + method
                if method not in ("nerfus", "ultranerf")
                else "implicit/" + ("nerfus_no_prior" if method == "nerfus" else method)
            )
            c = OmegaConf.to_container(
                compose(
                    config_name="config",
                    overrides=["experiment=production/comparisons/us_recon_four_dataset_p100k_v1/" + alias],
                ),
                resolve=True,
            )
            content = (json.dumps(c, indent=2) + "\n").encode()
            native = {
                "schema_version": 1,
                "source": {
                    "commit": "fa2006d8ad82149bb89ea50f554e8b49edfe77bb",
                    "resolved_sha256": hashlib.sha256(content).hexdigest(),
                    "experiment": c["experiment"],
                },
                "model": c["model"],
                "sampler": c["scene"]["sampler"],
                "grid": c["scene"]["grid"],
                "loss": c["loss"],
                "iterations": c["engine"]["iterations"],
                "metric": c["metric"],
                "seed": c["system"]["seed"],
                "excluded_infrastructure": [
                    "legacy Engine/State/callback registry",
                    "Slurm and legacy output paths",
                    "historical tracking/visualization settings",
                ],
                "runtime": "native Nerfstudio Trainer; gsplat Strategy; no Hydra or legacy imports",
            }
            target = args.output / (method + ".json")
            if target.exists() and json.loads(target.read_text()) != native:
                raise ValueError("Existing snapshot differs; use a new review directory")
            target.write_text(json.dumps(native, indent=2) + "\n")


if __name__ == "__main__":
    main()
