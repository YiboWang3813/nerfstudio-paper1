"""Read retained Mac-paper run evidence without restoring old working trees."""

import argparse
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "docs/resources/provenance/paper-evidence.json"


def read_evidence(path):
    """Return an original run file retained by the manuscript source audit."""
    member = PurePosixPath(path)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Select a relative retained evidence path")
    index = json.loads(INDEX.read_text())
    with zipfile.ZipFile(ROOT / index["archive"]) as archive:
        try:
            return archive.read(str(member))
        except KeyError as error:
            raise FileNotFoundError(f"Not retained by the Mac manuscript audit: {member}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Original output-relative run evidence member")
    args = parser.parse_args()
    if not args.path:
        print(INDEX.read_text(), end="")
        return
    try:
        data = read_evidence(args.path)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(data)


if __name__ == "__main__":
    main()
