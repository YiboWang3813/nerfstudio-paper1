#!/usr/bin/env python3
"""Download and extract the TDSC-ABUS2023 dataset.

The third-party package exposes the same Google Drive file IDs, but its package
uses Python syntax newer than the project's older experiment environment. This
standalone script keeps the download step reproducible without importing that
package.
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import gdown


DEFAULT_OUT_ROOT = Path("/tmp/tdsc_abus_2023_raw")

GDRIVE_FILE_IDS = {
    "Train": "1i0nLWX2PXiJnE_pZd79XyV1zshETCgWz",
    "Validation": "1--ldK9wb-DGKuKPO0ro-Ka6pRtndIekh",
    "Test": "1z7gGTR2w45VySccvstZkgycnQx3qeTVa",
}


def download_split(out_root: Path, split: str, file_id: str, keep_zip: bool) -> None:
    split_dir = out_root / split
    zip_path = out_root / f"{split}.zip"
    if split_dir.is_dir():
        print(f"{split}: exists at {split_dir}, skip")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {split}: {url} -> {zip_path}", flush=True)
        result = gdown.download(url, str(zip_path), quiet=False)
        if result is None or not zip_path.exists():
            raise RuntimeError(f"Failed to download {split} from {url}")
    else:
        print(f"{split}: using existing zip {zip_path}")

    print(f"Extracting {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_root)

    if not keep_zip:
        os.remove(zip_path)
    print(f"{split}: done", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(GDRIVE_FILE_IDS),
        default=list(GDRIVE_FILE_IDS),
    )
    parser.add_argument("--keep-zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_root.expanduser().resolve()
    for split in args.splits:
        download_split(out_root, split, GDRIVE_FILE_IDS[split], keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
