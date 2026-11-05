"""Read immutable historical evidence from Git without thousands of working files."""

import argparse
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]


def read_evidence(path):
    """Read a frozen source member without restoring a results worktree."""
    archive = json.loads((ROOT / "docs/resources/provenance/results-history.json").read_text())
    member = PurePosixPath(path)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Select a relative historical evidence path")
    spec = archive["commit"] + ":" + archive["tree"] + "/" + str(member)
    return subprocess.check_output(["git", "--git-dir=" + str(ROOT / archive["repository"]), "show", spec])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Path relative to archived evidence/legacy")
    args = parser.parse_args()
    archive = json.loads((ROOT / "docs/resources/provenance/results-history.json").read_text())
    repo = ROOT / archive["repository"]
    if not args.path:
        print(json.dumps(archive, indent=2))
        return
    path = PurePosixPath(args.path)
    if path.is_absolute() or ".." in path.parts:
        parser.error("Select a relative historical evidence path")
    spec = archive["commit"] + ":" + archive["tree"] + "/" + str(path)
    result = subprocess.run(["git", "-C", str(repo), "show", spec], stdout=sys.stdout.buffer, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
