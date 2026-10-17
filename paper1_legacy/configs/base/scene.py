from dataclasses import dataclass, field
from omegaconf import MISSING

from typing import List

# 反向注册 dataset meta registry 以供检查
from scene.meta.dataset_meta.registry import DATASET_META_REGISTRY

# ================= 3. Scene (Data & Strategy) =================
@dataclass
class LoadingParams:
    """去哪里找数据"""
    baseroot: str = "./data"
    dataset_dirname: str = ""
    sweep_name: str = "sweep_01"
    images_name: str = "images.npy"
    test_images_name: str = "test_images.npy"
    test_poses_name: str = "test_poses.npy"
    # Image value convention before SliceMeta normalizes to [0, 1].
    # - uint8_255: image values are already 0..255-like.
    # - spine_phantom_shifted_float: Ultra-NeRF spine phantom floats use a dark
    #   background near -0.125; convert by image = clamp(image + 0.125, 0, 1) * 255.
    image_value_mode: str = "uint8_255"
    image_nan_value: float = 0.0
    # Optional integer downsample on both image axes, preserving physical FoV.
    image_downsample_factor: int = 1
    # Optional integer downsample on the image height/axial axis only.
    axial_downsample_factor: int = 1
    # Optional frame-axis crop applied before image normalization and pose_mode.
    # frame_end is exclusive; a negative value means the full sweep end.
    frame_start: int = 0
    frame_end: int = -1
    frame_stride: int = 1
    # Optional JSON/TXT/NPY file with explicit original frame indices.
    # Applied after frame_start/frame_end/frame_stride. Empty means disabled.
    frame_indices_path: str = ""
    # Test source mode:
    # - interpolated_file: use test_images_name/test_poses_name files
    # - holdout_interval: split from train sweep by trim + interval sampling
    test_mode: str = "interpolated_file"
    holdout_trim_each: int = 10
    holdout_num_samples: int = 0
    # When holdout_num_samples <= 0, use this fraction of the full selected sweep
    # for holdout evaluation. Production comparisons use 0.2 for an 80/20 split.
    holdout_test_ratio: float = 0.2
    #: ``absolute``：每帧 ``poses.npy`` 为传感器系下绝对 ``probe→world``。
    #: ``relative``：与 ``gen_relative.py`` 一致，磁盘存 ``R_i = inv(P_0^{abs}) @ P_i^{abs}``，
    #: 第 0 帧为 **I**，后续帧为相对第 0 帧；SliceMeta.pose 仍按 ``probe→world`` 使用（世界锚在首帧）。
    #: PEM 与 UltraGS 共用 ``SingleSweepContainer``；合成数据 ``liver*`` 默认见对应 yaml。
    pose_type: str = "absolute"
    #: 若为非空字符串，则从该 **绝对路径** 加载 `(N,4,4)` 位姿数组，取代
    #: ``join(sweep_dir, poses_name)``（用于 PEM 预测的 probe→world 位姿离线写入后供 UltraGS 消费）。
    poses_abspath: str = ""
    #: Pose source used after loading the image file.
    #: - file: use poses.npy/relative_poses.npy as usual.
    #: - canonical_sweep: ignore loaded tracker poses and build a tracking-free
    #:   canonical one-sweep trajectory from frame index and elevational spacing.
    pose_mode: str = "file"
    #: Slice step for canonical_sweep. If <= 0, use DatasetMeta.delta_elevational_mm.
    canonical_sweep_delta_mm: float = -1.0
    #: Extra scale applied to canonical_sweep_delta_mm or DatasetMeta.delta_elevational_mm.
    canonical_sweep_delta_scale: float = 1.0
    #: Reference frame index for canonical_sweep. Negative means the first frame.
    canonical_sweep_ref_frame: int = 0
    #: Translation axis in canonical_sweep. Supports x/y/z and signed variants like -z.
    canonical_sweep_axis: str = "z"
    #: Spacing sequence used by canonical_sweep.
    #: - uniform: fixed frame_index * canonical_sweep_delta_mm.
    #: - file_step_norm: oracle diagnostic, cumulative adjacent file-pose translation norm.
    #: - file_axis_projection: oracle diagnostic, cumulative adjacent file-pose translation projected on the canonical axis.
    canonical_sweep_spacing_mode: str = "uniform"

@dataclass
class SamplerParams:
    """采样器行为策略"""
    num_init: int = 300000
    num_new: int = 5000 # 都进入 num_to_sample

    scales_init: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3]) # mm
    quats_init: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0]) # wxyz
    
    uniform_ratio: float = 0.3
    intensity_threshold: float = 0.05
    jitter_radius: float = 0.2

@dataclass
class SchedulerParams:
    # 保留字段以兼容旧配置；offline 现改为显式 test_images/test_poses 文件加载。
    test_ratio: float = 0.2

@dataclass
class GridParams:
    """网格与包围盒策略"""
    tile_size: float = 4.0
    bbox_margin: float = 0.05

@dataclass
class SceneConfig:
    """Scene 完整配置"""
    dataset_name: str = MISSING
    
    loading: LoadingParams = field(default_factory=LoadingParams)
    sampler: SamplerParams = field(default_factory=SamplerParams)
    grid: GridParams = field(default_factory=GridParams)
    scheduler: SchedulerParams = field(default_factory=SchedulerParams)
