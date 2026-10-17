from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from configs.base.scene import SceneConfig

from .meta.dataset_meta import DatasetMeta
from .container.single_sweep_container import SingleSweepContainer
from .scheduler import SceneScheduler
from .bbox import SceneBBoxMeta, SceneBBoxComputer 
from .sampler.unified_sampler import UnifiedGaussianSampler

from typing import List

from .meta.slice_meta.collate import stack_slice_metas


class Scene:
    def __init__(self,
                 cfg: SceneConfig,            # 依然持有配置，用于流式更新策略
                 train_container: SingleSweepContainer, 
                 scheduler: SceneScheduler, 
                 bbox_computer: SceneBBoxComputer, 
                 gauss_sampler: UnifiedGaussianSampler, 
                 device: torch.device,
                 test_container: Optional[SingleSweepContainer] = None):
        
        self.cfg = cfg
        self.device = device
        
        # 组件注入 (Injection)
        self.container = train_container
        self.train_container = train_container
        self.test_container = test_container or train_container
        self.scheduler = scheduler
        self.bbox_computer = bbox_computer
        self.sampler = gauss_sampler
        
        # 状态
        self._bbox_meta: Optional[SceneBBoxMeta] = None 
        self._train_batch_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self._train_indices: Optional[List[int]] = None
        self._test_indices: Optional[List[int]] = None
        self._train_ptr: int = 0
        self._test_ptr: int = 0
        self._rng = np.random.RandomState(42)
        self._configure_offline_eval_mode()

    # === 对外接口 ===
    @property
    def dataset_meta(self) -> DatasetMeta:
        return self.train_container.meta

    @property
    def bounding_box(self) -> Optional[SceneBBoxMeta]:
        return self._bbox_meta

    @property
    def train_indices(self) -> Optional[List[int]]:
        return self._train_indices

    @property
    def test_indices(self) -> Optional[List[int]]:
        return self._test_indices

    def _configure_offline_eval_mode(self) -> None:
        if not hasattr(self.scheduler, "next_test_index"):
            return
        mode = str(getattr(self.cfg.loading, "test_mode", "interpolated_file")).lower()
        if mode in ("interpolated_file", "file", "external"):
            # Default path: use dedicated test_container loaded from test_images/test_poses files.
            return
        if mode != "holdout_interval":
            raise ValueError(
                f"Unsupported scene.loading.test_mode='{mode}'. "
                "Expected one of: interpolated_file, holdout_interval."
            )

        n = int(len(self.train_container))
        trim_each = int(getattr(self.cfg.loading, "holdout_trim_each", 10) or 0)
        num_samples = int(getattr(self.cfg.loading, "holdout_num_samples", 40) or 0)
        if num_samples <= 0:
            ratio = float(getattr(self.cfg.loading, "holdout_test_ratio", 0.2) or 0.2)
            num_samples = self._resolve_holdout_num_samples(n, trim_each=trim_each, ratio=ratio)
        test_idx = self._build_holdout_indices(n, trim_each=trim_each, num_samples=num_samples)
        test_set = set(test_idx)
        train_idx = [i for i in range(n) if i not in test_set]
        if len(train_idx) <= 0:
            raise ValueError("Holdout split left zero training frames. Lower holdout_num_samples/trim_each.")

        self._train_indices = train_idx
        self._test_indices = test_idx
        self._train_ptr = 0
        self._test_ptr = 0
        self._rng.shuffle(self._train_indices)
        # Holdout mode reuses train container for test by index lookup.
        self.test_container = self.train_container
        print(
            "[Scene] Holdout eval enabled:"
            f" total={n}, train={len(self._train_indices)}, test={len(self._test_indices)},"
            f" trim_each={trim_each}"
        )

    @staticmethod
    def _resolve_holdout_num_samples(n_frames: int, trim_each: int, ratio: float) -> int:
        if n_frames <= 1:
            raise ValueError("Need at least two frames for holdout split.")
        if trim_each < 0:
            raise ValueError(f"holdout_trim_each must be >= 0, got {trim_each}")
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"holdout_test_ratio must be in (0, 1), got {ratio}")
        n_avail = int(n_frames) - 2 * int(trim_each)
        if n_avail <= 1:
            raise ValueError(
                f"No valid holdout interval after trimming: N={n_frames}, trim_each={trim_each}"
            )
        return max(1, min(n_avail - 1, int(round(float(n_frames) * float(ratio)))))

    @staticmethod
    def _build_holdout_indices(n_frames: int, trim_each: int, num_samples: int) -> List[int]:
        if n_frames <= 0:
            raise ValueError("Cannot build holdout split for empty dataset.")
        if trim_each < 0:
            raise ValueError(f"holdout_trim_each must be >= 0, got {trim_each}")
        if num_samples <= 0:
            raise ValueError(f"holdout_num_samples must be > 0, got {num_samples}")

        i_min = int(trim_each)
        i_max = int(n_frames - 1 - trim_each)
        if i_max < i_min:
            raise ValueError(
                f"No valid holdout interval after trimming: N={n_frames}, trim_each={trim_each}"
            )
        n_avail = i_max - i_min + 1
        if num_samples > n_avail:
            raise ValueError(
                f"holdout_num_samples={num_samples} exceeds available={n_avail} "
                f"(N={n_frames}, trim_each={trim_each})"
            )
        picks = np.linspace(i_min, i_max, num_samples, dtype=np.float64)
        picks = np.rint(picks).astype(np.int64)
        uniq = np.unique(picks)
        if int(uniq.size) < int(num_samples):
            raise ValueError(
                f"Rounded interval sampling collapsed to {uniq.size} unique indices "
                f"(need {num_samples}); reduce holdout_num_samples."
            )
        return [int(x) for x in uniq.tolist()]

    def _next_train_index_override(self) -> int:
        if self._train_indices is None:
            return int(self.scheduler.next_train_index())
        idx = int(self._train_indices[self._train_ptr])
        self._train_ptr += 1
        if self._train_ptr >= len(self._train_indices):
            self._train_ptr = 0
            self._rng.shuffle(self._train_indices)
        return idx

    def _next_test_index_override(self) -> Optional[int]:
        if self._test_indices is None:
            return self.scheduler.next_test_index()
        if self._test_ptr >= len(self._test_indices):
            self._test_ptr = 0
            return None
        idx = int(self._test_indices[self._test_ptr])
        self._test_ptr += 1
        return idx
    
    # =========================================================================
    # 1. 初始化阶段 (Static Initialization)
    # 只使用 cfg.init 和 cfg.stream.jitter
    # =========================================================================
    # 辅助函数 通过cfg拿到一张slice上所有pixel的scales和quats
    def _get_pixel_scales(self, scales_init: List[float], n_pixels: int) -> torch.Tensor:
        return torch.tensor(scales_init, device=self.device).unsqueeze(0).expand(n_pixels, -1).float()
    
    def _get_pixel_quats(self, quats_init: List[float], n_pixels: int) -> torch.Tensor:
        quats = torch.tensor(quats_init, device=self.device).unsqueeze(0).expand(n_pixels, -1).float()
        quats = F.normalize(quats, dim=-1)
        return quats

    @torch.no_grad()
    def build_init_gaussians(self):
        # 1. 从 Config 获取策略
        num_init = self.cfg.sampler.num_init
        jitter_radius = self.cfg.sampler.jitter_radius
        
        # 2. 获取数据索引
        indices = self.scheduler.get_init_indices()
        if self._train_indices is not None:
            indices = list(self._train_indices)
        print(f"[Scene] Initializing with {len(indices)} frames...")

        # 获得单张图像像素总数
        n_pixels = self.train_container.meta.H * self.train_container.meta.W
        
        # 3. 收集所有候选点数据
        c_pos, c_int, c_rot, c_scale = [], [], [], []

        for idx in indices:
            slice_meta = self.train_container.get(idx)
            
            # [物理事实] 直接从 slice_meta 获取
            c_pos.append(slice_meta.pixel_pos.to(self.device))
            c_int.append(slice_meta.pixel_int.to(self.device))
            
            # [读取配置] 通过配置直接拿到初始的scale和quat
            # quats 需要添加微小的抖动以防它不优化
            c_rot.append(
                self._get_pixel_quats(self.cfg.sampler.quats_init, n_pixels)
            )
            c_scale.append(
                self._get_pixel_scales(self.cfg.sampler.scales_init, n_pixels)
            )
        
        full_pos = torch.cat(c_pos, dim=0)
        full_int = torch.cat(c_int, dim=0)
        full_rot = torch.cat(c_rot, dim=0)
        full_scale = torch.cat(c_scale, dim=0)

        # 4. 计算场景包围盒 (BBox)
        self._bbox_meta = self.bbox_computer.compute_init(full_pos)
        print(f"[Scene]: BBox Computed {self._bbox_meta}")

        # 5. 执行采样 (Sampler)
        # 内部会自动根据点云数量(Online/Offline)决定是否开启显存保护
        gaussians = self.sampler.sample(
            pixel_positions=full_pos,
            pixel_values=full_int,
            pixel_quats=full_rot,
            pixel_scales=full_scale,
            num_to_sample=num_init,
            jitter_radius=jitter_radius,
        )
        sampled_pixel_indices = gaussians.pop("pixel_indices", None)
        birth_frame_ids = None
        if sampled_pixel_indices is not None:
            birth_frame_ids = torch.div(
                sampled_pixel_indices.to(device=self.device, dtype=torch.long),
                int(n_pixels),
                rounding_mode="floor",
            )
        
        # 清理显存
        del full_pos, full_int, full_rot, full_scale
        torch.cuda.empty_cache()

        return {
            "gaussians": gaussians,
            "scene_radius": self._bbox_meta.scene_radius,
            "gaussian_birth_frame_ids": birth_frame_ids,
        }
    
    # =========================================================================
    # 2. 离线训练/测试 (Offline Loop)
    # =========================================================================
    def next_train_datum(self) -> Dict[str, Any]:
        """Offline Train: 纯数据获取，不涉及高斯更新"""
        idx = self._next_train_index_override()
        slice_meta = self.train_container.get(idx)
        return {"slice_meta": slice_meta, "index": idx}

    def next_train_batch(self, batch_size: int) -> Dict[str, Any]:
        """Offline Train: 连续取 ``batch_size`` 帧并 stack 为 batched 张量。"""
        if batch_size <= 0:
            raise ValueError(f"next_train_batch: batch_size must be positive, got {batch_size}")
        slice_metas = []
        indices: List[int] = []
        for _ in range(batch_size):
            idx = self._next_train_index_override()
            indices.append(idx)
            slice_metas.append(self.train_container.get(idx))
        buffers = self._train_batch_buffers.setdefault(batch_size, {})
        stacked = stack_slice_metas(slice_metas, buffers=buffers)
        return {"slice_metas": slice_metas, "indices": indices, "stacked": stacked}

    def next_test_datum(self) -> Optional[Dict[str, Any]]:
        idx = self._next_test_index_override()
        if idx is None: return None
        slice_meta = self.test_container.get(idx)
        return {"slice_meta": slice_meta, "index": idx}
    
    # =========================================================================
    # 3. 在线流式更新 (Online Stream)
    # 只使用 cfg.stream
    # =========================================================================
    def next_stream_datum(self):
        """
        Online Stream: 仅取下一帧 ``slice_meta`` 与索引；像素增量采样由 ``online_callbacks`` 中
        ``seed_every_frames`` + ``scene.sampler.num_new`` 决定，不在 Scene 内改像素几何。
        """
        idx = self.scheduler.next_stream_index()
        if idx is None:
            return None

        slice_meta = self.container.get(idx)

        # 无离线 init 点云时：用当前帧像素位置建立首包盒
        if self._bbox_meta is None:
            pp0 = slice_meta.pixel_pos.to(self.device)
            if pp0.shape[0] > 0:
                self._bbox_meta = self.bbox_computer.compute_init(pp0)

        if self._bbox_meta is not None:
            self.bbox_computer.update(self._bbox_meta, slice_meta.pixel_pos.to(self.device))

        return {"slice_meta": slice_meta, "index": idx}
