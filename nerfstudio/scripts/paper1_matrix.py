"""Loop over frozen native experiment axes; generate a config only when running.

Without --train this prints a compact plan and performs no training or config
expansion. Every actual run freezes its resolved config through native Trainer.
"""

import argparse
import gc
import json
from pathlib import Path

from nerfstudio.scripts.ultrasound_resources import resources

ROOT = Path("/home/yibo/paper1")


def cases(matrix, methods=None, datasets=None, sweeps=None):
    for dataset in matrix["datasets"]:
        if datasets and dataset not in datasets:
            continue
        base = Path(matrix["dataset_root"]) / dataset / matrix["dataset_version"]
        catalog = json.loads((base / "dataset.json").read_text())
        for package in catalog["packages"]:
            sweep = package["sweep"]
            if sweeps and sweep not in sweeps:
                continue
            for method in matrix["methods"]:
                if not methods or method in methods:
                    yield method, dataset, sweep, base / sweep / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "nerfstudio/experiments/paper1/matrices/main.json")
    parser.add_argument("--method", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--sweep", action="append")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    selected = list(cases(matrix, args.method, args.dataset, args.sweep))
    if not selected:
        parser.error("No matching experiment")
    print(
        json.dumps(
            {
                "experiments": len(selected),
                "methods": sorted({r[0] for r in selected}),
                "datasets": sorted({r[1] for r in selected}),
                "dataset_version": matrix["dataset_version"],
                "training_started": args.train,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.train:
        return
    with resources(gpu=True):
        import torch  # noqa: PLC0415 - shared runtime and GPU lease acquired

        from nerfstudio.configs.ultrasound_protocol import production_config  # noqa: PLC0415
        from nerfstudio.scripts.train import _set_random_seed  # noqa: PLC0415

        for method, dataset, sweep, manifest in selected:
            _set_random_seed(matrix["seed"])
            config = production_config(method, manifest, Path(matrix["output_root"]))
            config.save_config()
            trainer = config.setup()
            trainer.setup()
            trainer.train()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
