import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ..dataset_meta import DatasetMeta


class SliceMeta(ABC):
    """统一 SliceMeta：同时持有一帧 slice 的元信息和行为。"""

    def __init__(self, dataset_meta: DatasetMeta, device: torch.device) -> None:
        self.dataset_meta: DatasetMeta = dataset_meta
        self.device: torch.device = device

        self.image: Optional[torch.Tensor] = None
        self.pose: Optional[torch.Tensor] = None
        self.frame_id: Optional[int] = None

        self.probe_origin: Optional[torch.Tensor] = None
        self.vec_lateral: Optional[torch.Tensor] = None
        self.vec_axial: Optional[torch.Tensor] = None
        self.vec_elevational: Optional[torch.Tensor] = None

        self.pixel_pos: Optional[torch.Tensor] = None
        self.pixel_int: Optional[torch.Tensor] = None

        self.ray_origins: Optional[torch.Tensor] = None
        self.ray_dirs: Optional[torch.Tensor] = None

        self.grid_t_vals: Optional[torch.Tensor] = None
        self.grid_ray_origins: Optional[torch.Tensor] = None
        self.grid_ray_dirs: Optional[torch.Tensor] = None

        # replace_pose 依赖的局部几何模板
        self.pts_local_h: Optional[torch.Tensor] = None
        self.ray_o_local_h: Optional[torch.Tensor] = None

    @abstractmethod
    def _compute_local_geometry(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回探头局部坐标模板: (pts_local_h, ray_o_local_h)，shape 均为 (N, 4)。"""

    @torch.no_grad()
    def initialize_from_image_pose(self, image: torch.Tensor, pose: torch.Tensor, frame_id: int) -> "SliceMeta":
        image_f = image.to(self.device).float() / 255.0
        pose_f = pose.to(self.device).float()
        H, W = image_f.shape

        pts_local_h, ray_o_local_h = self._compute_local_geometry(H, W)
        self.pts_local_h = pts_local_h.to(device=self.device, dtype=pose_f.dtype)
        self.ray_o_local_h = ray_o_local_h.to(device=self.device, dtype=pose_f.dtype)

        self.image = image_f
        self.frame_id = int(frame_id)
        self._rebuild_from_pose(pose_f)
        return self

    @torch.no_grad()
    def replace_pose(self, pose: torch.Tensor) -> "SliceMeta":
        """无梯度版本：用于常规在线路径（避免意外构图/显存开销）。"""
        return self.replace_pose_differentiable(pose)

    def replace_pose_differentiable(self, pose: torch.Tensor) -> "SliceMeta":
        """可微版本：用于需要将渲染误差反传到 pose 的场景（如 Pose Wash）。"""
        if self.image is None or self.pts_local_h is None or self.ray_o_local_h is None:
            raise RuntimeError("replace_pose() 前需要先 initialize_from_image_pose()")
        pose_f = pose.to(device=self.device, dtype=self.pose.dtype)
        out = self.__class__(self.dataset_meta, self.device)
        out.image = self.image
        out.frame_id = self.frame_id
        out.pts_local_h = self.pts_local_h
        out.ray_o_local_h = self.ray_o_local_h
        out._rebuild_from_pose(pose_f)
        return out

    def _rebuild_from_pose(self, pose: torch.Tensor) -> None:
        H, W = self.image.shape
        R = pose[:3, :3]
        t = pose[:3, 3]

        self.pose = pose
        self.probe_origin = t
        self.vec_lateral = F.normalize(R[:, 0], dim=0)
        self.vec_axial = F.normalize(R[:, 1], dim=0)
        self.vec_elevational = F.normalize(R[:, 2], dim=0)

        pts_world = (pose @ self.pts_local_h.t()).t()[:, :3]
        ray_o_world = (pose @ self.ray_o_local_h.t()).t()[:, :3]

        self.pixel_pos = pts_world
        self.pixel_int = self.image.view(-1)

        pixel_pos_grid = pts_world.view(H, W, 3)
        self.grid_ray_origins = ray_o_world.view(H, W, 3)

        diff_vec = pixel_pos_grid - self.grid_ray_origins
        self.grid_t_vals = torch.linalg.norm(diff_vec, dim=-1)
        self.grid_ray_dirs = diff_vec / (self.grid_t_vals.unsqueeze(-1) + 1e-8)

        mask_zero = self.grid_t_vals < 1e-6
        if mask_zero.any():
            global_axial = self.vec_axial.view(1, 1, 3).expand(H, W, 3)
            self.grid_ray_dirs[mask_zero] = global_axial[mask_zero]

        self.ray_origins = self.grid_ray_origins[0, :, :].contiguous()
        self.ray_dirs = self.grid_ray_dirs[0, :, :].contiguous()
