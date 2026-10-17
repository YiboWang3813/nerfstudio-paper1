"""
FiMA 离线 PEM +（fork）PoseRefiner / PoseWash 配置。

- ``fima_component``：``pem`` | ``refiner``，与 ``engine.task``（train/test）组合成四阶段。
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FiMAPEMConfig:
    #: ``multiframe``：原 FiMA 多帧窗；``adjacent_pair``：双帧输入，预测相邻帧右乘变换 T（轻量 CNN）。
    input_mode: str = "multiframe"
    #: ``standard``：原滑窗采样；``phase_a_full`` / ``phase_a_local_window``：阶段 A 协议（仅 ``adjacent_pair``）。
    train_protocol: str = "standard"
    #: 阶段 A 局部窗：沿**全局时间轴**连续 8 帧且全在 train 内的窗；stride 控制重叠（stride=4 → 重叠 4）。
    phase_a_local_window: int = 8
    phase_a_local_stride: int = 4
    #: 阶段 A 局部窗训练：每步随机采样的窗个数 B。
    phase_a_windows_per_batch: int = 4
    #: 阶段 A 局部窗训练单步最多前向的边数（7×B 可能较大时截断）。
    phase_a_max_pair_forward: int = 56

    window_size: int = 10
    #: ``last_frame_rel0`` -> (B, 6)；``window_frames_rel0`` -> (B, N-1, 6)；
    #: ``window_edges`` -> (B, N-1, 6)，预测窗口内每条相邻边，最贴近 FiMA 原文的序列输出。
    #: ``adjacent_pair`` 时忽略多帧布局。
    pem_output_layout: str = "last_frame_rel0"
    d_model: int = 128
    num_mixer_layers: int = 4
    stem_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    #: Learnable early image downsample inside FiMA. ``1`` keeps legacy behavior.
    input_downsample_stride: int = 1
    input_downsample_channels: int = 32
    dropout: float = 0.0
    #: True 时优先 ``mamba_ssm.Mamba``，否则退化为 TransformerEncoderLayer
    prefer_mamba: bool = True
    #: True 时若 ``mamba_ssm`` 不可用则直接报错，避免 paper-style FiMA 静默退化。
    require_mamba: bool = False
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-2
    lr_decay_steps: int = 0
    lr_decay_gamma: float = 0.5
    batch_size: int = 4
    #: 非空时加载 UltraGS 导出 occ：``{glob_frame_idx:06d}.npy``，``(H,W)`` float32；缺失则用 0。
    gs_occ_cache_dir: str = ""
    #: True 时仍强制双通道 `[灰度, occ]`（仅 ``occ_fusion_mode=concat``）；occ 在无 cache 时全 0，用于 B1 首轮与后续 stem 形状一致。
    occ_always_two_channels: bool = False
    #: ``concat``：``[灰度, OCC]`` 通道拼接，``SpatialStem(in_ch=2)``；``weighted``：单通道 ``灰度 * clamp(1+scale*O)`` 再进 stem。
    occ_fusion_mode: str = "concat"
    occ_weight_scale: float = 0.6
    occ_weight_clamp_lo: float = 0.35
    occ_weight_clamp_hi: float = 1.35
    #: When true, FiMA also predicts two log-variance scalars per pose:
    #: translation and rotation uncertainty in the supervised output space.
    predict_uncertainty: bool = False
    uncertainty_logvar_min: float = -8.0
    uncertainty_logvar_max: float = 4.0
    #: Inference strategy for ``window_edges``. ``single`` preserves the legacy
    #: deterministic window choice; ``mean`` averages every valid window-context
    #: prediction for the same global adjacent edge.
    infer_window_edge_aggregation: str = "single"
    #: Export strategy for ``window_edges`` pose priors. ``chain`` composes
    #: adjacent edges from frame 0 to N-1. ``local_stitch`` composes each
    #: overlapping window locally, aligns it to previous overlap frames, then
    #: fuses per-frame pose candidates.
    export_pose_mode: str = "chain"
    #: Stride, in frames, for ``export_pose_mode=local_stitch``.
    infer_stitch_stride: int = 4


@dataclass
class FiMALossConfig:
    use_l1: bool = True
    use_mse: bool = False
    use_pearson: bool = True
    lambda_l1: float = 1.0
    lambda_mse: float = 1.0
    lambda_pearson: float = 0.1
    #: ``per_channel``：对 6 维分别算 Pearson 再平均；``batch_flat``：展平 (B,6) 成一条序列
    pearson_mode: str = "per_channel"
    #: True 时用训练 targets 的逐通道均值/标准差计算监督 loss；模型输出仍保持真实 mm/rad 单位。
    standardize_targets: bool = False
    target_std_min: float = 1.0e-6
    target_std_min_trans: float = 1.0e-2
    target_std_min_rot: float = 1.0e-4
    use_heteroscedastic_nll: bool = False
    lambda_heteroscedastic: float = 1.0
    #: Train window_edges in both temporal directions. The reversed pass uses
    #: inverse adjacent transforms as supervision.
    use_bidirectional_edges: bool = False
    lambda_reverse: float = 1.0
    #: Cycle loss between forward edge predictions and aligned reverse-edge
    #: predictions: T_fwd(i->i+1) @ T_rev(i+1->i) should be identity.
    use_edge_cycle: bool = False
    lambda_cycle: float = 0.1


@dataclass
class FiMASequenceConfig:
    val_ratio: float = 0.1
    #: val 帧划分可复现
    split_seed: int = 42
    #: 左侧 pad 策略：``repeat_edge`` | ``zero``
    pad_mode: str = "repeat_edge"
    #: ``auto``：保持旧逻辑，优先跟随 Scene holdout；``random_frame``：忽略 Scene holdout，
    #: 按 val_ratio 随机抽帧；``block_holdout``：把验证帧合并成少量连续块，减少对训练连续窗的切碎。
    split_protocol: str = "auto"
    #: True 时若 ``Scene`` 已启用 ``holdout_interval``（提供 ``train_indices``/``test_indices``），
    #: FiMA 与 GS 使用同一留出划分，且训练窗不跨过 test 帧（避免泄漏）。
    respect_scene_holdout: bool = True
    #: ``block_holdout`` 总验证帧数。
    block_holdout_num_samples: int = 20
    #: ``block_holdout`` 单个连续验证块长度；最后一个块可被截断以匹配总验证帧数。
    block_holdout_block_size: int = 5
    #: ``block_holdout`` 两端裁掉的候选区域。
    block_holdout_trim_each: int = 10


@dataclass
class PoseRefinerConfig:
    """Fork 语义与 ultrags.PoseRefinerConfig 对齐（独立 dataclass 避免循环 import）。"""

    pose_dim: int = 16
    num_frequencies: int = 6
    encoder_hidden_dim: int = 256
    encoder_layers: int = 2
    bottleneck_dim: int = 128
    decoder_hidden_dim: int = 256
    decoder_layers: int = 2
    lr: float = 1e-4
    max_delta_rot_deg: float = 0.3
    max_delta_trans_mm: float = 0.05


@dataclass
class PoseWashConfig:
    enable: bool = True
    min_improve: float = 0.0
    use_abs_improve: bool = False
    rollback_model_on_reject: bool = True
    num_pose_wash_steps: int = 4


@dataclass
class PoseGtDeltaLossConfig:
    rot_weight: float = 1.0
    trans_weight: float = 1.0


@dataclass
class FiMAConfig:
    name: str = "fima"
    #: ``pem``：姿态估计网络；``refiner``：冻结 PEM 后的 PoseRefiner+Wash
    fima_component: str = "pem"
    pem: FiMAPEMConfig = field(default_factory=FiMAPEMConfig)
    pem_loss: FiMALossConfig = field(default_factory=FiMALossConfig)
    sequence: FiMASequenceConfig = field(default_factory=FiMASequenceConfig)
    pose_refiner: PoseRefinerConfig = field(default_factory=PoseRefinerConfig)
    pose_wash: PoseWashConfig = field(default_factory=PoseWashConfig)
    pose_gt_delta: PoseGtDeltaLossConfig = field(default_factory=PoseGtDeltaLossConfig)
    #: 可选：绝对路径或相对于 cwd；用于 refiner 阶段加载 PEM
    pem_checkpoint_path: str = ""
    refiner_checkpoint_path: str = ""
