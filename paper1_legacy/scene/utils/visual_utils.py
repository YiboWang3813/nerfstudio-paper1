import os 
import numpy as np
import torch
from plyfile import PlyData, PlyElement

def save_debug_ply(path, gaussians_dict, colors=None, normalize_for_vis=True):
    """
    Args:
        normalize_for_vis (bool): 是否强制归一化到 [-1, 1] 空间以便于 Visualization。
                                  (不会修改原始数据，只影响导出的 PLY)
    """
    print(f"[Debug] Saving PLY to {path} ...")

    # 0. 确保输出目录存在
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    # 1. 提取数据 (Clone 出来防止修改原数据)
    xyz = gaussians_dict["means"].detach().cpu().numpy().copy()
    scales = gaussians_dict["scales"].detach().cpu().numpy().copy()
    rots = gaussians_dict["quats"].detach().cpu().numpy().copy() # (w, x, y, z)
    
    N = xyz.shape[0]
    
    # ============================================================
    # [关键修复] 归一化到 [-1, 1] 视觉空间
    # ============================================================
    if normalize_for_vis:
        # 计算包围盒中心
        center = np.mean(xyz, axis=0)
        # 计算最大跨度
        dist = np.max(np.abs(xyz - center))
        if dist < 1e-6: dist = 1.0
        
        print(f"[Debug] Visual Normalize: Center={center}, MaxDist={dist}")
        print(f"[Debug] Original Z-range: {xyz[:, 2].min():.2f} ~ {xyz[:, 2].max():.2f} (mm)")
        
        # 居中并缩放位置
        xyz = (xyz - center) / dist
        # 缩放物理尺寸 (Scale 也要除以同样的系数，保持相对比例)
        scales = scales / dist
    
    # ============================================================
    # [关键检查] 颜色范围安全检查
    # ============================================================
    if colors is not None:
        c = colors.detach().cpu().numpy()
        if c.ndim == 1: c = c[:, None]
        if c.shape[1] == 1: c = np.repeat(c, 3, axis=1) 
        
        # 检查是否误传了 0-255 的数据
        if c.max() > 1.1:
            print("[Warning] Colors seem to be in 0-255 range. Normalizing to 0-1.")
            c = c / 255.0
        
        # 裁切防止越界
        c = np.clip(c, 0.0, 1.0)
    else:
        # 默认给个醒目的颜色（例如青色），防止灰色在黑底里看不清
        c = np.tile(np.array([[0.0, 1.0, 1.0]]), (N, 1))

    # SH DC 转换
    f_dc = (c - 0.5) / 0.28209479177387814

    # Opacity (Logit)
    # 稍微提高一点不透明度方便观察 (0.3)
    opacities = np.ones((N, 1)) * 0.3
    opacities_logit = np.log(opacities / (1 - opacities))

    # Scales (Log)
    # 防止 log(0)
    scales_log = np.log(np.maximum(scales, 1e-8))

    # 构建 dtype
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    elements = np.empty(N, dtype=dtype)
    
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = np.zeros(N)
    elements['ny'] = np.zeros(N)
    elements['nz'] = np.zeros(N)
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opacities_logit.flatten()
    elements['scale_0'] = scales_log[:, 0]
    elements['scale_1'] = scales_log[:, 1]
    elements['scale_2'] = scales_log[:, 2]
    elements['rot_0'] = rots[:, 0]
    elements['rot_1'] = rots[:, 1]
    elements['rot_2'] = rots[:, 2]
    elements['rot_3'] = rots[:, 3]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"[Debug] PLY saved with Normalization: {path}")