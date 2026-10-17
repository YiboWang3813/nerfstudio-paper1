import torch
from common.scene.geometry.slice_meta import compute_slice_meta

# 导入底层模块化算子
from cov3d import compute_cov3d
from slice_mapping import build_pixel_gauss_mapping
from .rasterizer.rasterizer_s2s import rasterize_s2s
from .gaussian_model import GaussianModel 

def render(
    args,
    pixel_positions: torch.Tensor,
    pc: GaussianModel,  # GaussianModel
    index: int = -1,
):
    # 1. 提取切片几何元数据 (从 Scene 逻辑中获取)
    meta = compute_slice_meta(
        pixel_positions,
        H=args.image_height,
        W=args.image_width,
        slice_thickness_h=args.slice_thickness_h,
        beam_width_r=args.beam_width_r,
        axial_res_h=args.axial_res_h,
        index=index,
    )

    # 2. 核心算子 A：计算协方差逆矩阵 (Cov3D)
    # 返回 sigma_invs: [N, 9]
    sigma_invs, _, _ = compute_cov3d(pc.scales, pc.quats)

    # 3. 核心算子 B：构建像素与高斯的动态映射 (Mapping)
    # MappingResult 包含 pixel_offsets, pixel_counts, pixel_indices
    mapping_res = build_pixel_gauss_mapping(
        pixel_positions=pixel_positions,
        ray_origins=meta["ray_origins"],
        ray_dirs=meta["ray_dirs"],
        gauss_means=pc.means,
        gauss_scales=pc.scales,
        gauss_quats=pc.quats,
        slice_origin=meta["slice_origin"],
        slice_normal=meta["slice_normal"],
        img_width=args.image_width,
        slice_thickness_h=meta["slice_thickness_h"],
        beam_width_r=meta["beam_width_r"],
        axial_res_h=meta["axial_res_h"]
    )

    # 4. 核心算子 C：执行 S2S 专用光栅化渲染
    # 内部会自动触发 RasterizerDebugHook
    intensity_map = rasterize_s2s(
        pixel_positions=pixel_positions,
        means=pc.means,
        sigma_invs=sigma_invs,
        opacities=pc.opacities,
        intensities=pc.intensities,
        mapping=mapping_res
    )

    # 5. 调整形状返回 [H, W]
    return intensity_map.view(args.image_height, args.image_width)