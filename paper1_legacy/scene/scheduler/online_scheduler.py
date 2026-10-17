from typing import List, Optional

from . import SceneScheduler


class OnlineScheduler(SceneScheduler):
    """
    Online：流从数据集帧 0 起顺序步进至末尾（一次性，不循环）。
    初始化高斯仅用第 0 帧参与 build_init_gaussians（与流起点一致，无 warmup 分段）。
    """

    def __init__(self, dataset_len: int):
        self.dataset_len = dataset_len
        self._stream_ptr = 0

    def get_init_indices(self) -> List[int]:
        """用于初始化的帧索引（当前为单帧 0，与流式首帧对齐）。"""
        if self.dataset_len <= 0:
            return []
        return [0]

    def next_stream_index(self) -> Optional[int]:
        if self._stream_ptr >= self.dataset_len:
            return None
        idx = self._stream_ptr
        self._stream_ptr += 1
        return idx
