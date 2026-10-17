"""Capture the complete code/config state without committing user changes."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def capture(destination):
    destination.mkdir(parents=True, exist_ok=False)
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    names = set(git("ls-files", "-z").decode().split("\0"))
    names.update(git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0"))
    files = sorted(n for n in names if n and (ROOT / n).is_file()
                   and not n.startswith(("outputs/", "logs/", "artifacts/", "wandb/")))
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in files}
    (destination / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (destination / "git_status.txt").write_bytes(git("status", "--short"))
    (destination / "base_commit.txt").write_bytes(git("rev-parse", "HEAD"))
    (destination / "worktree.patch").write_bytes(git("diff", "HEAD", "--binary"))
    with tarfile.open(destination / "source.tar.gz", "w:gz") as archive:
        for name in files:
            archive.add(ROOT / name, arcname=name, recursive=False)
    (destination / "snapshot.sha256").write_text(
        hashlib.sha256((destination / "source.tar.gz").read_bytes()).hexdigest() + "\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    capture(parser.parse_args().destination)
