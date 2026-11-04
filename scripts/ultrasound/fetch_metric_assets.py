"""Fetch only four official metric assets, sequentially with bounded retries."""

import fcntl
import hashlib
import json
import shutil
import urllib.request

from nerfstudio.model_components.ultrasound_metrics import ASSETS, MODEL_ROOT, verify_assets


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with (MODEL_ROOT / ".assets.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = MODEL_ROOT / "hub/checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 2 * 1024**3:
            raise OSError("Need 2 GiB free to stage metric assets")
        for url, prefix in ASSETS:
            dest = root / url.rsplit("/", 1)[1]
            if dest.exists():
                with dest.open("rb") as stream:
                    sha = hashlib.file_digest(stream, "sha256").hexdigest()
                if not sha.startswith(prefix):
                    raise ValueError(f"Existing asset differs: {dest}")
                continue
            partial = dest.with_suffix(".partial")
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as stream:
                        shutil.copyfileobj(response, stream, 1024 * 1024)
                    with partial.open("rb") as stream:
                        sha = hashlib.file_digest(stream, "sha256").hexdigest()
                    if not sha.startswith(prefix):
                        raise ValueError("Downloaded hash differs")
                    partial.replace(dest)
                    print(dest.name, sha, flush=True)
                    break
                except Exception:
                    if attempt == 1:
                        raise
        (MODEL_ROOT / "metric-assets.json").write_text(
            json.dumps(
                {"assets": verify_assets(), "purpose": "pyiqa 0.1.7 frozen metrics; official public assets; no R2"},
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
