from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import SliceMeta


def _ensure_buffer(
    buffers: Dict[str, torch.Tensor],
    key: str,
    shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    buf = buffers.get(key)
    if buf is None or tuple(buf.shape) != tuple(shape) or buf.device != device or buf.dtype != dtype:
        buf = torch.empty(shape, device=device, dtype=dtype)
        buffers[key] = buf
    return buf


def stack_slice_metas(
    slice_metas: List["SliceMeta"],
    buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Stack single-frame SliceMeta objects into batched tensors.

    When ``buffers`` is provided, tensors are written into preallocated storage via ``copy_``
    to reduce per-iteration allocation overhead in offline batch training.
    """
    if not slice_metas:
        raise ValueError("stack_slice_metas: empty slice_metas")

    device = slice_metas[0].pixel_pos.device
    dtype = slice_metas[0].pixel_pos.dtype
    batch_size = len(slice_metas)

    if buffers is None:
        pixel_pos = torch.stack([m.pixel_pos.to(device=device, dtype=dtype) for m in slice_metas], dim=0)
        probe_origin = torch.stack([m.probe_origin.to(device=device, dtype=dtype) for m in slice_metas], dim=0)
        vec_lateral = torch.stack([m.vec_lateral.to(device=device, dtype=dtype) for m in slice_metas], dim=0)
        vec_axial = torch.stack([m.vec_axial.to(device=device, dtype=dtype) for m in slice_metas], dim=0)
        vec_elevational = torch.stack([m.vec_elevational.to(device=device, dtype=dtype) for m in slice_metas], dim=0)
        image = torch.stack([m.image.to(device=device) for m in slice_metas], dim=0)
    else:
        pixel_pos = _ensure_buffer(
            buffers, "pixel_pos", torch.Size([batch_size, *slice_metas[0].pixel_pos.shape]), device, dtype
        )
        probe_origin = _ensure_buffer(
            buffers, "probe_origin", torch.Size([batch_size, *slice_metas[0].probe_origin.shape]), device, dtype
        )
        vec_lateral = _ensure_buffer(
            buffers, "vec_lateral", torch.Size([batch_size, *slice_metas[0].vec_lateral.shape]), device, dtype
        )
        vec_axial = _ensure_buffer(
            buffers, "vec_axial", torch.Size([batch_size, *slice_metas[0].vec_axial.shape]), device, dtype
        )
        vec_elevational = _ensure_buffer(
            buffers, "vec_elevational", torch.Size([batch_size, *slice_metas[0].vec_elevational.shape]), device, dtype
        )
        image = _ensure_buffer(
            buffers, "image", torch.Size([batch_size, *slice_metas[0].image.shape]), device, slice_metas[0].image.dtype
        )
        for i, m in enumerate(slice_metas):
            pixel_pos[i].copy_(m.pixel_pos)
            probe_origin[i].copy_(m.probe_origin)
            vec_lateral[i].copy_(m.vec_lateral)
            vec_axial[i].copy_(m.vec_axial)
            vec_elevational[i].copy_(m.vec_elevational)
            image[i].copy_(m.image)

    pixel_int = None
    if slice_metas[0].pixel_int is not None:
        if buffers is None:
            pixel_int = torch.stack([m.pixel_int.to(device=device) for m in slice_metas], dim=0)
        else:
            pixel_int = _ensure_buffer(
                buffers,
                "pixel_int",
                torch.Size([batch_size, *slice_metas[0].pixel_int.shape]),
                device,
                slice_metas[0].pixel_int.dtype,
            )
            for i, m in enumerate(slice_metas):
                pixel_int[i].copy_(m.pixel_int)

    pose = None
    if slice_metas[0].pose is not None:
        if buffers is None:
            pose = torch.stack([m.pose.to(device=device) for m in slice_metas], dim=0)
        else:
            pose = _ensure_buffer(
                buffers, "pose", torch.Size([batch_size, *slice_metas[0].pose.shape]), device, slice_metas[0].pose.dtype
            )
            for i, m in enumerate(slice_metas):
                pose[i].copy_(m.pose)

    frame_ids = [int(m.frame_id) for m in slice_metas if m.frame_id is not None]

    return {
        "pixel_pos": pixel_pos,
        "probe_origin": probe_origin,
        "vec_lateral": vec_lateral,
        "vec_axial": vec_axial,
        "vec_elevational": vec_elevational,
        "image": image,
        "pixel_int": pixel_int,
        "pose": pose,
        "frame_ids": frame_ids,
    }
