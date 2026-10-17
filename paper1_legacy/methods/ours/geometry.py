"""Pose helpers for ``ours_renderer`` (world millimetres)."""

import torch


def pose_probe_world_to_normalized(
    pose_probe_to_world: torch.Tensor,
    scene_center_world: torch.Tensor,
    scene_scale_world: torch.Tensor,
) -> torch.Tensor:
    """
    Legacy helper for **checkpoint migration**: convert probe→world mm pose to the old
    normalized-world convention used before the mm refactor.

    ``pose_probe_to_world``: (4,4) probe→world mm.

    Returns a pose with ``[:3,:3] = R/s`` and ``[:3,3] = (t-c)/s`` so that
    ``means_norm = (means_mm - c) / s`` matches previous training when paired with this pose.
    """
    if pose_probe_to_world.shape != (4, 4):
        raise ValueError(f"pose must be (4,4), got {tuple(pose_probe_to_world.shape)}")
    s = scene_scale_world.reshape(()).to(device=pose_probe_to_world.device, dtype=pose_probe_to_world.dtype)
    c = scene_center_world.to(device=pose_probe_to_world.device, dtype=pose_probe_to_world.dtype)
    out = torch.eye(4, device=pose_probe_to_world.device, dtype=pose_probe_to_world.dtype)
    out[:3, :3] = pose_probe_to_world[:3, :3] / s
    out[:3, 3] = (pose_probe_to_world[:3, 3] - c) / s
    return out
