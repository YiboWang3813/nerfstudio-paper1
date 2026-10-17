import torch

from scene.meta.dataset_meta import DatasetMeta


def pixel_positions_for_renderer(pixel_pos: torch.Tensor, dataset_meta: DatasetMeta) -> torch.Tensor:
    w = int(dataset_meta.W)
    h = int(dataset_meta.H)
    if pixel_pos.dim() == 2:
        return pixel_pos.view(h, w, 3).permute(1, 0, 2).contiguous()
    if pixel_pos.shape[0] == h and pixel_pos.shape[1] == w:
        return pixel_pos.permute(1, 0, 2).contiguous()
    return pixel_pos.contiguous()
