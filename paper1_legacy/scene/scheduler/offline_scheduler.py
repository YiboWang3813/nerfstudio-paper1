import numpy as np
from typing import List, Optional

from . import SceneScheduler


class OfflineScheduler(SceneScheduler):
    """
    Offline 模式：
    - Train: 无限循环，Epoch 结束自动 Shuffle。
    - Test: 有限序列，每次调用 reset_test() 重置，跑完返回 None。
    """
    def __init__(
        self,
        train_len: int,
        test_len: int,
        seed: int = 42,
    ):
        self.train_len = train_len
        self.test_len = test_len
        self._rng = np.random.RandomState(seed)
        
        # 1. 离线模式显式分离 train/test 数据源
        self._train_indices = list(range(train_len))
        self._test_indices = list(range(test_len))
        
        # 2. 状态指针
        self._train_ptr = 0
        self._test_ptr = 0
        
        # 初始 Shuffle
        self._rng.shuffle(self._train_indices)

    def get_init_indices(self) -> List[int]:
        # Offline 初始化使用训练集全量
        return list(range(self.train_len))

    def next_train_index(self) -> int:
        """无限产生训练索引"""
        if not self._train_indices: return 0 # 防御性编程

        idx = self._train_indices[self._train_ptr]
        self._train_ptr += 1

        # Epoch 结束，重置并 Shuffle
        if self._train_ptr >= len(self._train_indices):
            self._train_ptr = 0
            self._rng.shuffle(self._train_indices)
        
        return idx

    def next_test_index(self) -> Optional[int]:
        """有限产生测试索引，耗尽返回 None并复位内部test_ptr"""
        if self._test_ptr >= len(self._test_indices):
            self._test_ptr = 0
            return None # 信号：测试结束
        
        idx = self._test_indices[self._test_ptr]
        self._test_ptr += 1
        return idx