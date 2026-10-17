import torch

# CUDA 算子 已经编译到 Conda 环境了
from ellipsoid_mapper import EllipsoidMapper, EllipsoidMapperConfig, SliceGeometry
from ultrags_shader import ShaderConfig, Shader

# 显式声明类型 内源
from .config import ModelConfig
from .gauss_container import GaussContainer

# 显式声明类型 外源
from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta


def occupancy_from_intermediates(
    intermediates: dict,
    H: int,
    W: int,
) -> torch.Tensor:
    """由 Shader 的 ``pixel_T``（透射率）得到 occupancy：``1 - T``（与 α 合成一致）。"""
    if intermediates is None or "pixel_T" not in intermediates:
        raise KeyError("occupancy_from_intermediates 需要 return_intermediates=True 且 intermediates 含 pixel_T")
    T = intermediates["pixel_T"].view(H, W).clamp(0.0, 1.0)
    return 1.0 - T


def render(
    cfg: ModelConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta, 
    container: GaussContainer,
    active_global_indices: torch.Tensor = None,
    is_jitter: bool = False,
    return_intermediates: bool = False
):
    # --- 1. 准备数据 ---
    means = container.means
    scales = container.scales
    quats = container.quats
    opacities = container.opacities
    intensities = container.intensities

    # 语义约定：
    # - 入参 active_global_indices: 全局高斯“索引列表”（可选），不是 0/1 mask。
    # - mapper/shader 的 pixel_gauss_indices: 相对“当前参与渲染高斯数组”的局部索引（CSR 扁平）。
    # - active_global_indices: 当前参与渲染高斯数组在全局容器中的索引映射。
    if active_global_indices is not None:
        active_global_indices = active_global_indices.to(device=means.device, dtype=torch.long)
        means = means[active_global_indices]
        scales = scales[active_global_indices]
        quats = quats[active_global_indices]
        opacities = opacities[active_global_indices]
        intensities = intensities[active_global_indices]

    pixel_pos = slice_meta.pixel_pos
    probe_origin = slice_meta.probe_origin

    # jitter逻辑 只在train时给pixel_pos和probe_origin施加等量同方向的jitter
    if is_jitter:
        # 提取配置中的抖动半径 (比如 0.2 mm)
        jitter_radius = cfg.render.jitter_radius_ratio * dataset_meta.delta_elevational_mm
        
        # 生成 [-jitter_radius, +jitter_radius] 的均匀随机偏移标量
        jitter_scalar = (torch.rand(1, device=pixel_pos.device) - 0.5) * 2.0 * jitter_radius
        
        # 算出物理偏移向量
        jitter_vec = slice_meta.vec_elevational * jitter_scalar
        
        # 【核心操作】：同步且等量地移动像素阵列和探头原点！
        pixel_pos = (pixel_pos + jitter_vec).contiguous()
        probe_origin = (probe_origin + jitter_vec).contiguous()

    # --- 2. 执行映射 (Mapper) ---
    (pixel_gauss_counts, pixel_gauss_offsets, pixel_gauss_indices) = \
    EllipsoidMapper.map(
        config=EllipsoidMapperConfig(
            H=dataset_meta.H,
            W=dataset_meta.W,
            lateral_spacing_mm_per_pix=dataset_meta.delta_lateral_mm,
            axial_spacing_mm_per_pix=dataset_meta.delta_axial_mm,
            slab_half_thickness_mm=cfg.render.slab_thickness_ratio * dataset_meta.delta_elevational_mm,
            axial_near_mm=dataset_meta.near_mm,
            k_sigma=cfg.render.k_sigma,
            lateral_col_offset=(float(dataset_meta.W - 1) * 0.5 if dataset_meta.center_lateral else 0.0),
            sort_by_depth=bool(getattr(cfg.render, "mapper_sort_by_depth", True)),
        ),
        geometry=SliceGeometry(
            origin_world=probe_origin,
            axis_lateral_world=slice_meta.vec_lateral,
            axis_axial_world=slice_meta.vec_axial,
            axis_elevational_world=slice_meta.vec_elevational,
        ),
        pixel_positions_world=pixel_pos,
        gauss_means_world=means,
        gauss_scales=scales,
        gauss_quats=quats,
    )

    # --- 3. 着色 (Shader) ---
    shader_cfg = ShaderConfig(
        H=dataset_meta.H,
        W=dataset_meta.W,
        fixed_bg_color=cfg.render.fixed_bg_color,
        compositing_mode=str(getattr(cfg.render, "shader_compositing_mode", "alpha")),
    )
    shader = Shader(shader_cfg)
    
    pred_image, depth_image, intermediates = shader.forward(
        gauss_means=means,
        gauss_scales=scales,
        gauss_quats=quats,
        gauss_opacities=opacities,
        gauss_intensities=intensities,

        pixel_positions=pixel_pos,
        probe_origin=probe_origin,
        probe_vec_elevational=slice_meta.vec_elevational,

        pixel_gauss_offsets=pixel_gauss_offsets,
        pixel_gauss_counts=pixel_gauss_counts,
        pixel_gauss_indices=pixel_gauss_indices,
        
        return_intermediates=return_intermediates
    )

    # --- 4. 纯数据打包 ---
    if return_intermediates:
        intermediates["pixel_gauss_counts"] = pixel_gauss_counts
        intermediates["num_pixel_gauss_pairs"] = len(pixel_gauss_indices)
        H, W = int(dataset_meta.H), int(dataset_meta.W)
        T_hw = intermediates["pixel_T"].view(H, W)
        intermediates["transmittance_image"] = T_hw
        intermediates["occupancy_image"] = (1.0 - T_hw.clamp(0.0, 1.0))
        if "pixel_gauss_indices" in intermediates:
            gauss_local_indices = intermediates["pixel_gauss_indices"].to(dtype=torch.long)
            seen_gauss_local_indices = torch.unique(gauss_local_indices)
            if active_global_indices is not None:
                seen_gauss_global_indices = active_global_indices[seen_gauss_local_indices]
            else:
                seen_gauss_global_indices = seen_gauss_local_indices
            intermediates["seen_gauss_global_indices"] = seen_gauss_global_indices
    
    return pred_image, depth_image, intermediates
