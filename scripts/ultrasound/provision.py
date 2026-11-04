"""Install project source overlays without changing the shared dependency stack."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path


def main():
    workspace = Path(__file__).resolve().parents[3]
    shared = Path("/home/yibo/shared/envs/torch251-cu124-py311")
    registry_path = shared.parent / "registry.json"
    local = workspace / ".venv"
    with (shared.parent / ".provision.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = json.loads(registry_path.read_text())
        entry = next(item for item in registry["environments"] if item["name"] == shared.name)
        if entry["status"] not in ("installed_unvalidated", "validating", "ready"):
            raise RuntimeError("Shared runtime is not installed; continue code work until it is available.")
        subprocess.run(
            [str(shared / "bin/python"), "-c", "import torch; assert torch.__version__ == '2.5.1+cu124'"], check=True
        )
        if not (local / "bin/python").exists():
            subprocess.run([str(shared / "bin/python"), "-m", "venv", "--without-pip", str(local)], check=True)
        site = local / "lib/python3.11/site-packages"
        (site / "readonly-shared-runtime.pth").write_text(
            'import site; site.addsitedir("/home/yibo/shared/envs/overlays/iqa017-torch251")\n'
            f'import site; site.addsitedir("{shared}/lib/python3.11/site-packages")\n'
        )
        environment = dict(os.environ, BUILD_NO_CUDA="1", UV_CACHE_DIR="/home/yibo/shared/cache/uv")
        subprocess.run(
            [
                "/home/yibo/.local/bin/uv",
                "pip",
                "install",
                "--python",
                str(local / "bin/python"),
                "--no-deps",
                "--no-build-isolation",
                "--config-settings",
                "editable_mode=compat",
                "-e",
                str(workspace / "nerfstudio"),
                "-e",
                str(workspace / "gsplat"),
            ],
            env=environment,
            check=True,
        )
        bindir = workspace / "bin"
        bindir.mkdir(exist_ok=True)
        launcher = bindir / "paper1"
        launcher.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nexec "$(dirname -- "${BASH_SOURCE[0]}")/../nerfstudio/scripts/ultrasound/paper1" "$@"\n'
        )
        launcher.chmod(0o755)
        subprocess.run([str(launcher), "identity"], cwd="/tmp", check=True)
        entry.setdefault("source_overlays", {})["paper1"] = {
            "python": str(local / "bin/python"),
            "shared_dependency_pth": str(site / "readonly-shared-runtime.pth"),
            "sources": {"nerfstudio": str(workspace / "nerfstudio"), "gsplat": str(workspace / "gsplat")},
            "copies_of_torch": 0,
            "owner_task": "paper1-native-ultrasound-refactor",
            "status": "source_routing_verified; native_method_acceptance_pending",
        }
        if "paper1" not in entry["consumer_projects"]:
            entry["consumer_projects"].append("paper1")
        temporary = registry_path.with_suffix(".paper1.tmp")
        temporary.write_text(json.dumps(registry, indent=2) + "\n")
        temporary.replace(registry_path)


if __name__ == "__main__":
    main()
