from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from scene.meta.slice_meta import SliceMeta


@dataclass(frozen=True)
class PatchSpec:
    top: int
    left: int
    height: int
    width: int


def random_patch_spec(height: int, width: int, patch_height: int, patch_width: int) -> PatchSpec:
    ph = min(int(patch_height), int(height))
    pw = min(int(patch_width), int(width))
    max_top = max(0, int(height) - ph)
    max_left = max(0, int(width) - pw)
    top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
    left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
    return PatchSpec(top=top, left=left, height=ph, width=pw)


def extract_slice_grid(
    slice_meta: SliceMeta,
    patch: Optional[PatchSpec] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return world points, per-pixel ray distances, and target image as HxW tensors."""
    if slice_meta.image is None or slice_meta.pixel_pos is None or slice_meta.grid_t_vals is None:
        raise RuntimeError("SliceMeta must be initialized before UltraNeRF rendering.")

    image = slice_meta.image
    h, w = image.shape
    points = slice_meta.pixel_pos.view(h, w, 3)
    depths = slice_meta.grid_t_vals.view(h, w)

    if patch is None:
        return points, depths, image

    sl_h = slice(patch.top, patch.top + patch.height)
    sl_w = slice(patch.left, patch.left + patch.width)
    return points[sl_h, sl_w], depths[sl_h, sl_w], image[sl_h, sl_w]
