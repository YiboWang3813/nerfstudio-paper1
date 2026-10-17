"""将 stdout/stderr 同时写入实验目录下的 logs/（与 shell tee 等价，供 run.py 使用）。"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import BinaryIO, Optional, TextIO


class _TeeStream:
    """转发写入到终端与日志文件；保留 isatty 以便 tqdm 等库正常工作。"""

    def __init__(self, stream: TextIO, log_file: TextIO):
        self._stream = stream
        self._log = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        self._stream.flush()
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self._stream, "encoding", None)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def attach_run_stdio_log(
    output_dir: str,
    *,
    log_basename: Optional[str] = None,
) -> str:
    """
    将 sys.stdout / sys.stderr 重定向为 tee，日志写入 ``{output_dir}/logs/``。

    Returns:
        实际日志文件绝对路径。
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    if log_basename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_basename = f"run_{stamp}.log"
    log_path = os.path.join(log_dir, log_basename)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    sys.stdout = _TeeStream(sys.__stdout__, log_file)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.__stderr__, log_file)  # type: ignore[assignment]
    return os.path.abspath(log_path)
