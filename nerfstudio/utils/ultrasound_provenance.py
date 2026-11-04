"""Capture the actual source/runtime used by a native experiment."""

import hashlib
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def execution_identity():
    import gsplat
    import torch

    import nerfstudio

    roots = {
        name: Path(module.__file__).resolve().parents[1]
        for name, module in [("nerfstudio", nerfstudio), ("gsplat", gsplat)]
    }
    sources = {}
    for name, root in roots.items():

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args])

        status = git("status", "--porcelain", "--untracked-files=all").decode()
        # Include untracked native source bytes: a HEAD or diff alone is not an
        # execution identity while developing a migration.
        files = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
        digest = hashlib.sha256()
        for file in sorted(set(filter(None, files))):
            path = root / file
            if path.is_file():
                digest.update(file.encode() + b"\0" + path.read_bytes() + b"\0")
        sources[name] = {
            "root": str(root),
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(status),
            "tree_sha256": digest.hexdigest(),
        }
    toolchain = {}
    for name, executable in [
        ("nvcc", str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.4")) / "bin/nvcc")),
        ("cc", os.environ.get("CC", "gcc")),
        ("cxx", os.environ.get("CXX", "g++")),
    ]:
        path = shutil.which(executable)
        toolchain[name] = {
            "path": str(Path(path).resolve()) if path else None,
            "version": subprocess.check_output([path, "--version"], text=True) if path else None,
        }
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "cuda_runtime": torch.version.cuda,
        "sources": sources,
        "toolchain": toolchain,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {name: version(name) for name in ("numpy", "torchvision", "pyiqa")},
    }
