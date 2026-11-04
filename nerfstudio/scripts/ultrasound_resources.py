"""Cooperate with shared-runtime installers and physical GPU users."""

import fcntl
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def resources(gpu=False):
    with open("/home/yibo/shared/envs/.provision.lock", "a") as environment:
        fcntl.flock(environment, fcntl.LOCK_SH)
        if not gpu:
            yield
            return
        requested = os.environ.get("CUDA_VISIBLE_DEVICES")
        if requested is not None and not requested.isdecimal():
            raise ValueError("Select one physical GPU number with CUDA_VISIBLE_DEVICES.")
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
        )
        locks = Path("/home/yibo/shared/cache/locks")
        locks.mkdir(parents=True, exist_ok=True)
        for row in rows.splitlines():
            index, memory, utilization = map(int, row.split(","))
            if requested is not None and index != int(requested):
                continue
            if memory > 256 or utilization > 5:
                continue
            handle = (locks / f"gpu-{index}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                current = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--id=" + str(index),
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                memory, utilization = map(int, current.strip().split(","))
                if memory > 256 or utilization > 5:
                    continue
                os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
                yield
            finally:
                handle.close()
            return
        raise RuntimeError("No requested idle GPU is available; existing GPU jobs were left untouched.")
