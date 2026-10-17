import torch
from . import SliceMeta
from ..dataset_meta import DatasetMeta

class ConvexSliceMeta(SliceMeta):
    def __init__(self, dataset_meta: DatasetMeta, device: torch.device):
        super().__init__(dataset_meta, device)
        self.radius_mm = dataset_meta.radius_mm
        self.fov_deg = dataset_meta.fov_deg
        self.near_mm = dataset_meta.near_mm
        self.far_mm = dataset_meta.far_mm

    def _compute_local_geometry(self, H, W):
        # 1. 极坐标网格生成
        grid_h = torch.arange(H, device=self.device).float()
        grid_w = torch.arange(W, device=self.device).float()
        arr_h, arr_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        fov_rad = torch.deg2rad(torch.tensor(self.fov_deg, device=self.device))
        d_theta = fov_rad / W
        depth_step = (self.far_mm - self.near_mm) / H

        # 角度范围 [-FOV/2, FOV/2]
        angles = (arr_w - (W - 1) / 2.0) * d_theta
        # 半径范围 [R+near, R+far]
        radii = self.radius_mm + self.near_mm + arr_h * depth_step

        local_x = radii * torch.sin(angles)
        local_y = torch.zeros_like(local_x)
        local_z = radii * torch.cos(angles)
        
        ones = torch.ones_like(local_x)
        pts_local = torch.stack([local_x, local_y, local_z, ones], dim=-1).view(-1, 4)

        # 2. 射线原点 (在探头表面，即 r = radius_mm)
        surf_x = self.radius_mm * torch.sin(angles)
        surf_z = self.radius_mm * torch.cos(angles)
        ray_o_local = torch.stack([surf_x, local_y, surf_z, ones], dim=-1).view(-1, 4)

        return pts_local, ray_o_local