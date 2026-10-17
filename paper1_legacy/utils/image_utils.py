import torch 
import imageio 
import numpy as np 


def save_image(fname, img):
    """img: torch.Tensor / float numpy [0,1] / uint8 numpy [0,255]。

    注意：``make_diff`` 等返回的 uint8 若再走 ``clip(0,1)``，会把 2–255 全压成 1，
    再乘 255 后几乎整图变白、只剩 0 处为黑（伪「白底黑点」），与真实差异图不符。
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if np.issubdtype(img.dtype, np.uint8):
        out = img
    else:
        img = np.clip(img.astype(np.float64), 0.0, 1.0)
        out = (img * 255.0).astype(np.uint8)
    imageio.imwrite(fname, out)


def make_diff(pred, gt):
    """绝对差异图归一化"""
    diff = np.abs(pred - gt)
    d = diff / (diff.max() + 1e-8)
    d = (d * 255).astype(np.uint8)
    return d


def concat_side_by_side(pred, gt):
    """拼图：GT | Pred"""
    h, w = gt.shape
    canvas = np.zeros((h, w * 2), dtype=np.uint8)
    canvas[:, :w] = (gt * 255).astype(np.uint8)
    canvas[:, w:] = (pred * 255).astype(np.uint8)
    return canvas