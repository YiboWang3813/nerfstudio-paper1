import torch 
import plyfile 
import numpy as np 

from typing import NamedTuple, List 


class BasicPointCloud(NamedTuple):
    points: np.array  # position 
    colors: np.array  # color 
    normals: np.array  # normal 


def load_ply(path):
    # Read point cloud data from path 
    plydata = plyfile.PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # (num_points, 3)
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # (num_points, 3)
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T  # (num_points, 3)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def save_ply(pcd: BasicPointCloud, path: str): 
    vertex = np.array(
        list(zip(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2],
                pcd.normals[:, 0], pcd.normals[:, 1], pcd.normals[:, 2],
                pcd.colors[:, 0], pcd.colors[:, 1], pcd.colors[:, 2])),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    ply_el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([ply_el], text=False).write(path)
    print(f"[BasicPointCloud] Saved {len(pcd.points)} points to {path}")


# def merge_and_deduplicate_pointclouds(pcd_list: List["BasicPointCloud"], voxel_size: float = 0.8):
#     """
#     将多个 BasicPointCloud 合并为一个，并进行全局体素下采样去重。

#     Args:
#         pcd_list (List[BasicPointCloud]): 各个 sweep 的点云列表
#         voxel_size (float): 体素网格大小 (单位 mm)，默认 0.8mm 左右
    
#     Returns:
#         BasicPointCloud: 合并后、去重后的点云
#     """
#     if len(pcd_list) == 0:
#         raise ValueError("Empty point cloud list for merging.")

#     # === 拼接所有 sweep 点云 ===
#     all_points = np.vstack([pc.points for pc in pcd_list])
#     all_colors = np.vstack([pc.colors for pc in pcd_list])
#     all_normals = np.vstack([pc.normals for pc in pcd_list])
#     print(f"[Merge] Before dedup: {len(all_points):,} points")

#     # === 全局体素下采样去重 ===
#     coords = np.floor(all_points / voxel_size)
#     _, unique_idx = np.unique(coords, axis=0, return_index=True)

#     dedup_points = all_points[unique_idx]
#     dedup_colors = all_colors[unique_idx]
#     dedup_normals = all_normals[unique_idx]

#     print(f"[Merge] After dedup: {len(dedup_points):,} points (voxel={voxel_size} mm)")

#     return BasicPointCloud(
#         points=dedup_points,
#         colors=dedup_colors,
#         normals=dedup_normals
#     )


# def deduplicate_pointclouds(pcd_raw: BasicPointCloud, voxel_size=0.8): 
#     points = pcd_raw.points 
#     colors = pcd_raw.colors 
#     normals = pcd_raw.normals 

#     if len(points) == 0: 
#         raise ValueError("Empty point cloud for deduplicating.") 

#     coords = np.floor(points / voxel_size)
#     _, unique_idx = np.unique(coords, axis=0, return_index=True)

#     dedup_points = points[unique_idx] 
#     dedup_colors = colors[unique_idx] 
#     dedup_normals = normals[unique_idx] 

#     return BasicPointCloud(points=dedup_points, colors=dedup_colors, normals=dedup_normals) 


# def deduplicate_pointclouds(points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, voxel_size=0.8):
#     """
#     points: (N,3) torch.float32
#     colors: (N,3) torch.uint8
#     normals: (N,3) torch.float32
#     返回 BasicPointCloud，内部用 numpy（为了 save_ply），但本函数完全不做 numpy 判断。
#     """

#     device = points.device

#     # STEP 1 — voxel 坐标 (N,3)
#     coords = torch.floor(points / voxel_size).to(torch.int64)

#     # STEP 2 — shift 为非负
#     min_vals = coords.min(dim=0).values
#     coords_shift = coords - min_vals

#     # STEP 3 — 构成 1D hash
#     xmax = coords_shift[:, 0].max() + 1
#     ymax = coords_shift[:, 1].max() + 1

#     hash_vals = coords_shift[:, 0] + xmax * (coords_shift[:, 1] + ymax * coords_shift[:, 2])

#     # STEP 4 — unique（GPU 支持）
#     _, uniq_idx = torch.unique(hash_vals, sorted=True, return_indices=True)

#     # STEP 5 — 取 unique 点（仍为 torch）
#     pts_d = points[uniq_idx]
#     col_d = colors[uniq_idx]
#     nor_d = normals[uniq_idx]

#     # STEP 6 — 返回 numpy（因为 BasicPointCloud 和 save_ply 需要）
#     return BasicPointCloud(
#         points=pts_d.cpu().numpy(),
#         colors=col_d.cpu().numpy(),
#         normals=nor_d.cpu().numpy()
#     )


# def save_ply_data_prep(pts, intens, out_ply_path): 
#     intens_norm = (intens - intens.min()) / (intens.max() - intens.min() + 1e-12) 
#     colors = (np.repeat(intens_norm[:, None], 3, axis=1) * 255).astype(np.uint8)
#     normals = np.zeros_like(pts) 

#     vertex = np.empty(pts.shape[0], dtype=[
#         ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
#     ])
#     vertex['x'], vertex['y'], vertex['z'] = pts.T
#     vertex['nx'], vertex['ny'], vertex['nz'] = normals.T
#     vertex['red'], vertex['green'], vertex['blue'] = colors.T

#     ply_el = plyfile.PlyElement.describe(vertex, 'vertex')
#     plyfile.PlyData([ply_el], text=False).write(out_ply_path)  # binary format
#     print(f'Saved {len(pts)} points to {out_ply_path}')
