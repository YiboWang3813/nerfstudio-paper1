"""Scene center/scale from GT probe translations (world mm) — checkpoint migration only."""

from typing import Tuple

import torch


def scene_center_scale_from_pose_translations(
    translations_world: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        translations_world: (N, 3) probe origins ``pose[:3,3]`` in world mm.
        margin: multiply max extent (e.g. 1.15).

    Returns:
        center (3,), scale scalar tensor (>0).
    """
    if translations_world.ndim != 2 or translations_world.shape[1] != 3:
        raise ValueError(f"Expected (N,3), got {tuple(translations_world.shape)}")
    c = translations_world.mean(dim=0)
    extent = (translations_world - c.unsqueeze(0)).abs().max()
    s = (extent * float(margin)).clamp(min=1e-3)
    return c, s.reshape(())
