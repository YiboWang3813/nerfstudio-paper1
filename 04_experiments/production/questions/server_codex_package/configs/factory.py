import torch

from configs.root import RootConfig

# 组件引入
from .meta.dataset_meta.registry import DATASET_META_REGISTRY
from .container.single_sweep_container import SingleSweepContainer
from .scheduler.offline_scheduler import OfflineScheduler
from .scheduler.online_scheduler import OnlineScheduler
from .sampler.unified_sampler import UnifiedGaussianSampler
from .bbox import SceneBBoxComputer
from .scene import Scene


class SceneFactory:
    """
    [组装车间]
    负责实例化 Scene 所需的所有零件，并将其组装在一起。
    遵循依赖注入原则。
    """

    @staticmethod
    def _resolve_test_poses_name(scene_cfg) -> str:
        """与 ``loading.pose_type`` 对齐的测试位姿文件名（train 由 ``SingleSweepContainer`` 内解析）。"""
        name = str(scene_cfg.loading.test_poses_name)
        pt = str(getattr(scene_cfg.loading, "pose_type", "absolute")).lower()
        if pt == "relative" and name == "test_poses.npy":
            return "test_relative_poses.npy"
        return name

    @staticmethod
    def build_scene(cfg: RootConfig, device: torch.device) -> Scene:

        # 1. 准备 Configs
        scene_cfg = cfg.scene
        
        # 2. 获取 DatasetMeta (物理硬配置)
        if scene_cfg.dataset_name not in DATASET_META_REGISTRY:
             raise ValueError(f"DatasetMeta '{scene_cfg.dataset_name}' not found.")
        dataset_meta = DATASET_META_REGISTRY[scene_cfg.dataset_name]

        # =========================================================
        # Part A: 构建基础设施 (Dataset, BBox)
        # =========================================================
        train_container = SingleSweepContainer(
            meta=dataset_meta, 
            cfg=scene_cfg, 
            device=device
        )
        sensorless_required = bool(getattr(cfg.experiment, "sensorless_required", False))
        if sensorless_required and bool(getattr(train_container, "uses_file_poses_for_geometry", True)):
            raise RuntimeError(
                "Sensorless experiment requires canonical generated poses for train SliceMeta, "
                "but train_container reports file poses reach geometry."
            )
        
        bbox_computer = SceneBBoxComputer(
            tile_size=scene_cfg.grid.tile_size,
            margin_ratio=scene_cfg.grid.bbox_margin 
        )

        # =========================================================
        # Part B: 构建统一采样器 (Selection)
        # =========================================================
        gauss_sampler = UnifiedGaussianSampler(
            cfg=scene_cfg.sampler, 
            device=device
        )

        # =========================================================
        # Part C: 构建调度器 (Scheduler) -> [模式分歧点]
        # =========================================================
        if cfg.engine.stream == "offline":
            test_mode = str(getattr(scene_cfg.loading, "test_mode", "interpolated_file")).lower()
            # holdout_interval：测试帧为训练 sweep 内的子集索引；Scene 会用 train_container 兼任 test，不得再建单独 test 容器
            # （否则 poses_abspath 全序列 (N_train) 会与 test_images（K 张）长度不一致而报错）。
            if test_mode == "holdout_interval":
                test_container = None
                test_len = int(getattr(scene_cfg.loading, "holdout_num_samples", 40) or 0)
                if test_len <= 0:
                    test_len = Scene._resolve_holdout_num_samples(
                        len(train_container),
                        trim_each=int(getattr(scene_cfg.loading, "holdout_trim_each", 10) or 0),
                        ratio=float(getattr(scene_cfg.loading, "holdout_test_ratio", 0.2) or 0.2),
                    )
                scheduler = OfflineScheduler(
                    train_len=len(train_container),
                    test_len=test_len,
                    seed=cfg.system.seed,
                )
            else:
                test_container = SingleSweepContainer(
                    meta=dataset_meta,
                    cfg=scene_cfg,
                    device=device,
                    images_name=scene_cfg.loading.test_images_name,
                    poses_name=SceneFactory._resolve_test_poses_name(scene_cfg),
                )
                if (
                    int(train_container.meta.H) != int(test_container.meta.H)
                    or int(train_container.meta.W) != int(test_container.meta.W)
                ):
                    raise ValueError(
                        "Train/Test resolution mismatch: "
                        f"train=({train_container.meta.H},{train_container.meta.W}) vs "
                        f"test=({test_container.meta.H},{test_container.meta.W}). "
                        "Please use matching train/test image files (e.g., images.npy with test_images.npy, "
                        "or images_x2.npy with test_images_x2.npy)."
                    )
                if sensorless_required and bool(getattr(test_container, "uses_file_poses_for_geometry", True)):
                    raise RuntimeError(
                        "Sensorless experiment requires canonical generated poses for test SliceMeta, "
                        "but test_container reports file poses reach geometry."
                    )
                scheduler = OfflineScheduler(
                    train_len=len(train_container),
                    test_len=len(test_container),
                    seed=cfg.system.seed,
                )
        else:  # online / inference
            test_container = None
            scheduler = OnlineScheduler(dataset_len=len(train_container))

        # =========================================================
        # Part D: 最终组装
        # =========================================================
        scene = Scene(
            cfg=scene_cfg,
            train_container=train_container,
            scheduler=scheduler,
            bbox_computer=bbox_computer,
            gauss_sampler=gauss_sampler,
            device=device,
            test_container=test_container,
        )
        
        return scene
