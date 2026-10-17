from dataclasses import dataclass, field
from typing import List


@dataclass
class RenderConfig:
    k_sigma: float = 3.0
    fixed_bg_color: float = 0.0
    slab_thickness_ratio: float = 0.85
    jitter_radius_ratio: float = 0.25
    mapper_sort_by_depth: bool = True
    shader_compositing_mode: str = "alpha"
    #: which renderer implementation to use: "legacy" or "integrated"
    which_render: str = "legacy"
    #: ultrags_renderer input checks; disable for max performance
    enable_input_checks: bool = False
    #: use CUDA batched mapper/shader (B, H, W) in one launch (offline integrated path)
    batched_kernel: bool = False
    #: number of training slices per render call when batched_kernel is True
    batch_size: int = 1
    #: reserved for finite datasets; offline infinite train ignores partial batches
    drop_last_batch: bool = True
    #: intermediate tensors collection level: "none" | "light" | "full"
    intermediates_level: str = "none"


@dataclass
class GaussContainerConfig:
    """高斯参数与优化器初值（Offline / Online 共用）。"""
    @dataclass
    class LRConfig:
        # [修改] 提高 Means LR，让点能动起来
        means: float = 1.6e-4  # 原 5e-5 太小
        # [关键修改] 大幅提高 Scales LR，否则 Split 后很难收缩
        scales: float = 5e-3  # 原 1e-5 是导致 Scale 卡死的元凶
        quats: float = 1e-3
        opacities: float = 5e-2  # 提高 Opacity LR，让它能快速在实/虚之间切换
        intensities: float = 1e-3
    
    @dataclass
    class InitValConfig:
        opacities: float = 0.05
        intensities: float = 0.5

    lr: LRConfig = field(default_factory=LRConfig)
    init_val: InitValConfig = field(default_factory=InitValConfig)


@dataclass
class OfflineConfig:
    """仅 Offline 训练使用的子配置（密度控制等）。"""
    @dataclass
    class LocalWindowConfig:
        @dataclass
        class JointOptConfig:
            enable: bool = True
            lambda_seam: float = 0.05
            num_overlap_samples: int = 1

        enable: bool = False
        window_size_frames: int = 16
        overlap_k_frames: int = 4
        gauss_expand_m_frames: int = 8
        use_gauss_subset: bool = True
        # <=0 means use all windows each iteration.
        num_windows_per_iter: int = 0
        # <=0 means only rebuild cache when topology changes (e.g., DC prune/densify).
        subset_cache_rebuild_every: int = 0
        # <=0 disables periodic full-gaussian correction step.
        global_refresh_every: int = 0
        joint_opt: JointOptConfig = field(default_factory=JointOptConfig)

    @dataclass
    class DensityControlConfig:
        @dataclass
        class DCEnableConfig:
            # 确保这两个是 True
            densify: bool = True
            prune: bool = True
        
        @dataclass
        class DCTimingConfig:
            start_iter: int = 500
            end_iter: int = 25_000  # 建议延长到训练结束前不久，例如 25000
            freq: int = 100  # [修改] 500 太慢了，建议 100 步一次，让优化更灵敏

        @dataclass
        class DCThresholdsConfig:
            # [修正] 你的实际 max grad 只有 8e-5，原 4e-4 太高了
            # 建议设为 0.00002 (2e-5) 甚至更低，先让点长出来！
            densify_grad: float = 6e-6  # 大放水

            # [致命问题修正]
            # 你的 Log 里 Scale max=0.35。如果这里设 1.2，永远不会 Split。
            # 设为 0.3 或者 0.4，强迫那些“稍大”的球分裂成小球。
            split_scale: float = 0.6

            # Prune 阈值
            prune_opacity: float = 5e-3  # [修改] 提高到 0.02，0.005 太低了，留下了太多幽灵点
            prune_intensity: float = 0.01

        enable: DCEnableConfig = field(default_factory=DCEnableConfig)
        timing: DCTimingConfig = field(default_factory=DCTimingConfig)
        thresholds: DCThresholdsConfig = field(default_factory=DCThresholdsConfig)

    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)
    local_window: LocalWindowConfig = field(default_factory=LocalWindowConfig)


@dataclass
class StreamSeedConfig:
    """Online 流式 concat 播种：legacy（固定帧频+num_new）或 occupancy+残差 门控。"""

    #: ``legacy``：沿用 ``seed_every_frames`` + ``scene.sampler.num_new``
    #: ``occupancy_guided``：首帧可全像素；后续帧用全池预渲染得 occupancy 与 diff，再分位+Cap 选像素
    mode: str = "legacy"
    #: 首帧（``stream_frame_id==0`` 且池为空）是否按 H*W 逐像素各采一条高斯
    first_frame_dense: bool = False
    #: 组合分数：对 ``lack = 1 - occupancy``（``occupancy`` 来自 ``render.occupancy_from_intermediates``）与 diff 的权重，将归一化到和为 1
    occupancy_lack_weight: float = 0.7
    diff_weight: float = 0.3
    #: 取 ``score >= quantile(score_flat, score_quantile)`` 的像素（如 0.9 ≈ 高分前约 10%）
    score_quantile: float = 0.90
    #: 单次 concat 最大条数（首帧 dense 时不生效）
    max_new_cap: int = 15_000
    #: 对 ``|pred-gt|`` 作高斯平滑的 sigma（像素）；0 关闭
    diff_blur_sigma: float = 0.0
    #: 是否将播种决策关键统计写入仓库根目录 ``debug_new_seed_strategy.txt``
    debug_enabled: bool = False


@dataclass
class OnlineConfig:
    @dataclass
    class LocalWindowManagerConfig:
        window_size_frames: int = 10
        #: 与 offline 对齐的总优化步 T；0 表示使用 engine.iterations
        total_opt_steps: int = 0
        ratio_front: float = 0.5
        ratio_rear: float = 0.5

    @dataclass
    class GaussianStateManagerConfig:
        #: ``off`` | ``membership`` | ``spatial_chunk`` | ``slice_identity``
        mode: str = "off"
        @dataclass
        class MembershipConfig:
            membership_k: int = 5
            #: 子集局部性 l；0 表示全程全量高斯
            sub_locality_l: int = 0

        @dataclass
        class SpatialChunkConfig:
            #: 空间分桶厚度（mm），越小越局部，越大越平滑
            chunk_size_mm: float = 1.0
            #: 以当前切片中心为基准，激活带半宽（mm）
            active_half_width_mm: float = 3.0
            #: 为避免边界 popping，附加尺度余量：max_scale * 该系数
            scale_margin_ratio: float = 1.0

        @dataclass
        class SliceIdentityConfig:
            #: 若 |gauss_slice_id - current_frame_id| >= exclude_num_ids，则排除该高斯
            exclude_num_ids: int = 15

        membership: MembershipConfig = field(default_factory=MembershipConfig)
        spatial_chunk: SpatialChunkConfig = field(default_factory=SpatialChunkConfig)
        slice_identity: SliceIdentityConfig = field(default_factory=SliceIdentityConfig)

    @dataclass
    class DensityControlConfig:
        """Online 流式密度控制：梯度统计、clone/split/prune，以及可选的近邻融合合并。"""

        @dataclass
        class DCEnableConfig:
            densify: bool = False
            prune: bool = False

        @dataclass
        class DCTimingConfig:
            #: 从第几个全局 GS 步开始累积梯度并允许触发 DC（含 warm-up）
            start_gs_step: int = 500
            #: 上限步；0 表示不限制（与 Offline end_iter=0 语义对齐时可单独约定）
            end_gs_step: int = 0
            #: 每多少个全局 GS 步执行一次 prune+densify；0 表示永不触发
            freq_gs_steps: int = 100

        @dataclass
        class DCThresholdsConfig:
            densify_grad: float = 6e-6
            split_scale: float = 0.6
            prune_opacity: float = 5e-3
            prune_intensity: float = 0.01

        enable: DCEnableConfig = field(default_factory=DCEnableConfig)
        timing: DCTimingConfig = field(default_factory=DCTimingConfig)
        thresholds: DCThresholdsConfig = field(default_factory=DCThresholdsConfig)
        #: DC 步末尾是否执行近邻合并（融合+删冗），缓解 split/clone 膨胀
        merge_after_dc: bool = False
        #: 每多少 **stream 帧** 在帧末额外执行一次合并（0=关闭）；与 merge_after_dc 二选一或降频
        merge_every_frames: int = 0
        #: 近邻判定距离（mm）
        merge_dist_thresh_mm: float = 0.02
        #: 参与合并的最低激活不透明度门槛（max(oi,oj) 低于则跳过该对）
        merge_opacity_thresh: float = 0.01
        #: 单次合并最多处理的候选 pair 条数
        merge_max_pairs: int = 100_000

    @dataclass
    class PoseRefinerConfig:
        pose_dim: int = 16
        num_frequencies: int = 6
        encoder_hidden_dim: int = 256
        encoder_layers: int = 2
        bottleneck_dim: int = 128
        decoder_hidden_dim: int = 256
        decoder_layers: int = 2
        lr: float = 1e-4
        # Hard caps for residual motion per forward pass (micro-adjustment mode).
        max_delta_rot_deg: float = 0.0
        max_delta_trans_mm: float = 0.05

    @dataclass
    class NoisePoseConfig:
        rot_std_deg: float = 1.0
        trans_std_mm: float = 1.0

    @dataclass
    class PoseWashGateConfig:
        # 接受-拒绝机制：仅当 pose-wash 目标下降超过阈值才接受 refined。
        enable: bool = True
        min_improve: float = 0.0
        # 若为 True，使用 |improve| 与阈值比较。
        use_abs_improve: bool = False
        # 若拒绝，是否回滚 PoseRefiner 本帧更新（恢复到帧开始参数）。
        rollback_model_on_reject: bool = True

    """
    Online Streaming (LocalWindow + GlobalPool) 的最小可行配置。
    说明：
    - 步数由 LWM 预算表 + ratio_front/ratio_rear 决定（见 docs/methods/baselines/ultrags_baseline/online_update_spec.md）
    - seed_every_frames: 每隔多少帧从当前帧像素做一次增量采样（1=每帧；0=不采样）；增密主要靠 DC
    - save_every_frames: 按帧触发保存（0/负数=关闭）；流式渲染评测由 ``cfg.engine.stream_eval_*`` 控制
    - log_scalars_every_gs_step / console_log_every_gs_step / console_frame_summary_interval /
      console_print_eval_metrics: 流式训练细粒度日志（见 ScalarManager.print_online_gs_step）
    - lwm/gsm；近邻合并超参在 ``density_control``（与 clone/split/prune 同属 DC）
    - noise_pose/pose_refiner: case B 配置
    - stream_seed: 见 ``StreamSeedConfig``；``occupancy_guided`` 时仍尊重 ``seed_every_frames``（0=每帧可播）
    """

    seed_every_frames: int = 1
    save_every_frames: int = 50
    #: Render/GS optimization stage loss names. Empty means use cfg.loss.needed_losses.
    render_loss_names: List[str] = field(default_factory=list)
    #: Pose-wash stage loss names. Empty means use cfg.loss.needed_losses.
    pose_wash_loss_names: List[str] = field(default_factory=list)

    #: 每个 GS_OPTIM 步是否写入标量（wandb/EMA）；数据多时可关
    log_scalars_every_gs_step: bool = False
    #: 每个 GS_OPTIM（及 POSE_WASH 内步）是否在控制台打印各 loss 分项
    console_log_every_gs_step: bool = False
    #: 每处理完一帧后打印 ``[UltraGS-Online] Frame ...`` 摘要的间隔；1=每帧，0=关闭
    console_frame_summary_interval: int = 10
    #: eval 触发时是否调用 ``ScalarManager.print_test`` 打印 PSNR/SSIM 等到控制台
    console_print_eval_metrics: bool = False

    # A: gt, B: noise, C: pem
    mode: str = "A"
    #: POSE_WASH：仅训练 Phi_p 的迭代步数（文档 num_pose_refinement）
    num_pose_refinement: int = 1

    lwm: LocalWindowManagerConfig = field(default_factory=LocalWindowManagerConfig)
    gsm: GaussianStateManagerConfig = field(default_factory=GaussianStateManagerConfig)
    stream_seed: StreamSeedConfig = field(default_factory=StreamSeedConfig)
    density_control: DensityControlConfig = field(default_factory=DensityControlConfig)
    pose_refiner: PoseRefinerConfig = field(default_factory=PoseRefinerConfig)
    noise_pose: NoisePoseConfig = field(default_factory=NoisePoseConfig)
    pose_wash_gate: PoseWashGateConfig = field(default_factory=PoseWashGateConfig)


@dataclass
class PemGsAblationConfig:
    """PEM ↔ UltraGS 消融编排（见 ``tmp_pem_gs_ablation_b0_b1_callbacks.py``）。"""

    #: ``off`` | ``b0`` | ``b1``
    mode: str = "off"
    #: 供子进程 PEM 使用的 FiMA yaml（相对仓库根目录或绝对路径）
    fima_config: str = "preproduction/baseline_reproduction/fima/paper_edges"
    pem_iterations: int = 2500
    #: 0 表示与 ``pem_iterations`` 相同（保证末尾 iteration 写 checkpoint）
    pem_save_every: int = 0
    #: ``b1``：每轮子进程 UltraGS 训练步数
    alternating_gs_iterations: int = 2500
    #: ``b1``：外层交替轮数（含首轮无 occ 的 PEM）
    num_rounds: int = 2
    #: ``b1`` 必填：UltraGS Hydra experiment alias，用于子进程 GS 与 occupancy 导出。
    ultrags_driver_yaml: str = ""


@dataclass
class ModelConfig:
    name: str = "ultrags"

    render: RenderConfig = field(default_factory=RenderConfig)
    gauss_container: GaussContainerConfig = field(default_factory=GaussContainerConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    pem_gs_ablation: PemGsAblationConfig = field(default_factory=PemGsAblationConfig)
