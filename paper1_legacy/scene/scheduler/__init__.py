from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import numpy as np

class SceneScheduler(ABC):
    @abstractmethod
    def get_init_indices(self) -> List[int]:
        """仅用于 build_init_gaussians，返回一堆索引用于初始化"""
        pass
