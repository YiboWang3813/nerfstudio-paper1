import math
from dataclasses import dataclass

import torch

from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta

from .configs import RenderConfig


@dataclass
class UltraGSVirtualCamera:
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    znear: float
    zfar: float
    world_view_transform: torch.Tensor
    projection_matrix: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor


def _projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float, device: torch.device) -> torch.Tensor:
    tan_half_y = math.tan(fov_y / 2.0)
    tan_half_x = math.tan(fov_x / 2.0)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right

    p = torch.zeros((4, 4), dtype=torch.float32, device=device)
    z_sign = 1.0
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p.transpose(0, 1).contiguous()


def _world_view_from_slice(slice_meta: SliceMeta) -> torch.Tensor:
    pose = slice_meta.pose.to(dtype=torch.float32)
    c2w = pose.clone()
    # Match the NeRF/OpenGL-to-rasterizer convention used by the released code.
    c2w[:3, 1:3] *= -1.0
    w2c = torch.linalg.inv(c2w)
    return w2c.transpose(0, 1).contiguous()


def _world_view_from_slice_plane(
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
    camera_distance_mm: float,
) -> torch.Tensor:
    device = slice_meta.pose.device
    dtype = torch.float32
    lateral = torch.nn.functional.normalize(slice_meta.vec_lateral.to(device=device, dtype=dtype), dim=0)
    axial = torch.nn.functional.normalize(slice_meta.vec_axial.to(device=device, dtype=dtype), dim=0)
    elev = torch.nn.functional.normalize(slice_meta.vec_elevational.to(device=device, dtype=dtype), dim=0)

    mid_axial = 0.5 * (float(dataset_meta.near_mm) + float(dataset_meta.far_mm))
    mid_lateral = 0.0
    if not bool(getattr(dataset_meta, "center_lateral", True)):
        mid_lateral = 0.5 * float(dataset_meta.W - 1) * float(dataset_meta.delta_lateral_mm)
    distance = max(float(camera_distance_mm), 1.0)
    center = (
        slice_meta.probe_origin.to(device=device, dtype=dtype)
        + lateral * mid_lateral
        + axial * mid_axial
        - elev * distance
    )

    view = torch.eye(4, dtype=dtype, device=device)
    view[:3, 0] = lateral
    view[:3, 1] = axial
    view[:3, 2] = elev
    view[3, 0] = -torch.dot(center, lateral)
    view[3, 1] = -torch.dot(center, axial)
    view[3, 2] = -torch.dot(center, elev)
    return view.contiguous()


def _physical_fovs(dataset_meta: DatasetMeta, camera_distance_mm: float):
    if str(dataset_meta.probe_type) != "linear":
        raise ValueError(
            "ultrags_baseline render.camera_mode=slice_plane_physical currently supports "
            f"linear probes only, got {dataset_meta.probe_type!r}."
        )
    distance = max(float(camera_distance_mm), 1.0)
    half_width = 0.5 * float(dataset_meta.width_mm)
    half_depth = 0.5 * (float(dataset_meta.far_mm) - float(dataset_meta.near_mm))
    return 2.0 * math.atan(half_width / distance), 2.0 * math.atan(half_depth / distance)


def build_virtual_camera(
    cfg: RenderConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
) -> UltraGSVirtualCamera:
    device = slice_meta.pose.device
    camera_mode = str(getattr(cfg, "camera_mode", "released_pose_fov"))
    if camera_mode == "released_pose_fov":
        fov_x = math.radians(float(cfg.fov_x_deg))
        fov_y = math.radians(float(cfg.fov_y_deg))
        world_view = _world_view_from_slice(slice_meta)
    elif camera_mode == "slice_plane_physical":
        fov_x, fov_y = _physical_fovs(dataset_meta, float(cfg.physical_camera_distance_mm))
        world_view = _world_view_from_slice_plane(dataset_meta, slice_meta, float(cfg.physical_camera_distance_mm))
    else:
        raise ValueError(
            "ultrags_baseline render.camera_mode must be 'released_pose_fov' or "
            f"'slice_plane_physical', got {camera_mode!r}."
        )
    proj = _projection_matrix(float(cfg.znear), float(cfg.zfar), fov_x, fov_y, device)
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    camera_center = world_view.inverse()[3, :3]
    return UltraGSVirtualCamera(
        image_height=int(dataset_meta.H),
        image_width=int(dataset_meta.W),
        FoVx=fov_x,
        FoVy=fov_y,
        znear=float(cfg.znear),
        zfar=float(cfg.zfar),
        world_view_transform=world_view,
        projection_matrix=proj,
        full_proj_transform=full,
        camera_center=camera_center,
    )
