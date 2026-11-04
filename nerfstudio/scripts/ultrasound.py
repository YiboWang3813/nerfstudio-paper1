"""Project-scoped native Nerfstudio commands with strict source identity checks."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def check_sources() -> dict[str, str]:
    workspace = Path(__file__).resolve().parents[3]
    expected = {"nerfstudio": workspace / "nerfstudio" / "nerfstudio", "gsplat": workspace / "gsplat" / "gsplat"}
    actual = {}
    for name, root in expected.items():
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        if location.parent != root:
            raise RuntimeError(f"{name} resolved to {location}; expected {root}. Use paper1/bin/paper1.")
        actual[name] = str(location)
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["identity", "train", "eval", "render", "viewer", "build"])
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sources = check_sources()
    if args.command == "identity":
        import torch

        print(
            json.dumps(
                {
                    "sources": sources,
                    "python": sys.executable,
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                },
                indent=2,
            )
        )
        return
    if args.command == "build":
        from gsplat.ultrasound._backend import load_backend

        load_backend(verbose=True)
        return
    commands = {"train": "train", "eval": "eval", "render": "render", "viewer": "viewer.run_viewer"}
    module = importlib.import_module("nerfstudio.scripts." + commands[args.command])
    sys.argv = [f"paper1 {args.command}"] + remaining
    module.entrypoint()


if __name__ == "__main__":
    main()
