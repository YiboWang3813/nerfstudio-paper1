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
    from nerfstudio.scripts.ultrasound_resources import resources

    command = sys.argv[1] if len(sys.argv) > 1 else "identity"
    with resources(gpu=command in ("train", "eval", "render", "viewer", "build", "test") and "--help" not in sys.argv):
        dispatch()


def dispatch():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["identity", "train", "eval", "render", "viewer", "build", "prepare", "preflight", "test", "artifacts"]
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sources = check_sources()
    if args.command == "artifacts":
        from nerfstudio.scripts import paper1_artifacts

        sys.argv = ["paper1 artifacts"] + remaining
        paper1_artifacts.main()
        return
    if args.command == "test":
        import pytest

        raise SystemExit(
            pytest.main(
                [
                    "-o",
                    "addopts=",
                    str(Path(__file__).resolve().parents[2] / "tests/ultrasound"),
                    str(Path(__file__).resolve().parents[3] / "gsplat/tests/test_ultrasound.py"),
                    *remaining,
                ]
            )
        )
    if args.command in ("prepare", "preflight"):
        from nerfstudio.scripts import ultrasound_prepare

        sys.argv = [f"paper1 {args.command}"] + remaining
        getattr(ultrasound_prepare, args.command)()
        return
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

        load_backend(backend=remaining[0] if remaining else "ours", verbose=True)
        return
    commands = {"train": "train", "eval": "eval", "render": "render", "viewer": "viewer.run_viewer"}
    module = importlib.import_module("nerfstudio.scripts." + commands[args.command])
    sys.argv = [f"paper1 {args.command}"] + remaining
    module.entrypoint()


if __name__ == "__main__":
    main()
