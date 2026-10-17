"""Move run checkpoint directories to external storage and leave symlinks.

By default this script prints the planned moves. Pass --apply to move files.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def has_checkpoint_files(path: Path) -> bool:
    return any(path.glob("iteration_*.pth"))


def iter_checkpoint_dirs(outputs_root: Path):
    for path in sorted(outputs_root.rglob("checkpoints")):
        if path.is_symlink() or not path.is_dir():
            continue
        if has_checkpoint_files(path):
            yield path


def rel_run_path(checkpoint_dir: Path, outputs_root: Path) -> Path:
    run_dir = checkpoint_dir.parent
    return run_dir.relative_to(outputs_root)


def externalize_one(checkpoint_dir: Path, outputs_root: Path, checkpoint_root: Path, apply: bool) -> None:
    target = checkpoint_root / rel_run_path(checkpoint_dir, outputs_root) / "checkpoints"
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")

    print(f"{checkpoint_dir} -> {target}")
    if not apply:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(checkpoint_dir), str(target))
    os.symlink(str(target), str(checkpoint_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs", help="Local outputs root to scan.")
    parser.add_argument("--checkpoint-root", required=True, help="External checkpoint root.")
    parser.add_argument("--apply", action="store_true", help="Actually move files; otherwise only print the plan.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    count = 0
    for checkpoint_dir in iter_checkpoint_dirs(outputs_root):
        externalize_one(checkpoint_dir, outputs_root, checkpoint_root, args.apply)
        count += 1
    mode = "moved" if args.apply else "planned"
    print(f"{mode} {count} checkpoint directories")


if __name__ == "__main__":
    main()
