import torch
from . import SliceMeta
from ..dataset_meta import DatasetMeta

class LinearSliceMeta(SliceMeta):
    def __init__(self, dataset_meta: DatasetMeta, device: torch.device):
        super().__init__(dataset_meta, device)
        self.center_lateral = dataset_meta.center_lateral
        self.near_mm = dataset_meta.near_mm
        self.far_mm = dataset_meta.far_mm
        self.width_mm = dataset_meta.width_mm

    def _compute_local_geometry(self, H, W):
        # 1. 矩形网格生成
        grid_h = torch.arange(H, device=self.device).float()
        grid_w = torch.arange(W, device=self.device).float()
        arr_h, arr_w = torch.meshgrid(grid_h, grid_w, indexing='ij') 

        axial_spacing = (self.far_mm - self.near_mm) / H
        lateral_spacing = self.width_mm / W
        
        w_offset = (W - 1) * 0.5 if self.center_lateral else 0.0

        # Lateral -> X, Axial -> Y, Elevational -> Z
        local_x = (arr_w - w_offset) * lateral_spacing
        local_y = arr_h * axial_spacing + self.near_mm  
        local_z = torch.zeros_like(local_x)             
        
        ones = torch.ones_like(local_x)
        # 探头坐标系下 所有像素的坐标
        # z为0 因为一张slice认为没有厚度 
        pts_local = torch.stack([local_x, local_y, local_z, ones], dim=-1).view(-1, 4)

        # 探头坐标系下 所有像素所在射线的原点坐标
        # 一条射线上的像素具有相同的原点坐标
        # y = 0 因为射线的原点在最上面从0开始 z = 0 因为同理射线原点也在slice平面上没有厚度
        ray_o_local = torch.stack([local_x, torch.zeros_like(local_y), local_z, ones], dim=-1).view(-1, 4)

        return pts_local, ray_o_local