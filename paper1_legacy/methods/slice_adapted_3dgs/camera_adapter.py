import math
from dataclasses import dataclass

import torch

from scene.meta.dataset_meta import DatasetMeta
from scene.meta.slice_meta import SliceMeta

from .configs import RenderConfig


@dataclass
class SliceAdaptedCamera:
    image_height: int
    image_width: int
    half_width_mm: float
    half_height_mm: float
    focal_depth_mm: float
    world_view_transform: torch.Tensor
    camera_center: torch.Tensor


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


def _physical_extents(dataset_meta: DatasetMeta):
    if str(dataset_meta.probe_type) != "linear":
        raise ValueError(
            "slice_adapted_3dgs render.camera_mode=slice_plane_physical currently supports "
            f"linear probes only, got {dataset_meta.probe_type!r}."
        )
    half_width = 0.5 * float(dataset_meta.width_mm)
    half_height = 0.5 * (float(dataset_meta.far_mm) - float(dataset_meta.near_mm))
    return half_width, half_height


def build_slice_camera(
    cfg: RenderConfig,
    dataset_meta: DatasetMeta,
    slice_meta: SliceMeta,
) -> SliceAdaptedCamera:
    camera_mode = str(getattr(cfg, "camera_mode", "slice_plane_physical"))
    if camera_mode != "slice_plane_physical":
        raise ValueError(
            "slice_adapted_3dgs currently supports render.camera_mode='slice_plane_physical', "
            f"got {camera_mode!r}."
        )

    distance = max(float(cfg.physical_camera_distance_mm), 1.0)
    half_width, half_height = _physical_extents(dataset_meta)
    world_view = _world_view_from_slice_plane(dataset_meta, slice_meta, distance)
    camera_center = world_view.inverse()[3, :3]
    return SliceAdaptedCamera(
        image_height=int(dataset_meta.H),
        image_width=int(dataset_meta.W),
        half_width_mm=half_width,
        half_height_mm=half_height,
        focal_depth_mm=distance,
        world_view_transform=world_view,
        camera_center=camera_center,
    )


def pixel_grid(camera: SliceAdaptedCamera, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(
        -camera.half_height_mm,
        camera.half_height_mm,
        int(camera.image_height),
        device=device,
        dtype=dtype,
    )
    xs = torch.linspace(
        -camera.half_width_mm,
        camera.half_width_mm,
        int(camera.image_width),
        device=device,
        dtype=dtype,
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def physical_fovs_for_debug(camera: SliceAdaptedCamera):
    fov_x = 2.0 * math.atan(camera.half_width_mm / max(camera.focal_depth_mm, 1.0))
    fov_y = 2.0 * math.atan(camera.half_height_mm / max(camera.focal_depth_mm, 1.0))
    return fov_x, fov_y
