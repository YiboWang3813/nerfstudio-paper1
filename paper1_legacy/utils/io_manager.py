"""
IOManager: 统一负责 checkpoint 的加载/保存、图像保存到本地、.ply 保存到本地。
"""
import os
from typing import Any, List, Optional, Tuple

import torch

from .image_utils import save_image as _save_image


def _to_numpy(x: Any):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x


def _write_gauss_ply(path: str, means, scales, quats, opacities, intensities) -> None:
    """将高斯参数写入标准 3DGS PLY 格式。所有输入为 numpy 或 tensor。"""
    import numpy as np
    from plyfile import PlyData, PlyElement

    means = _to_numpy(means)
    scales = _to_numpy(scales)
    quats = _to_numpy(quats)
    opacities = _to_numpy(opacities)
    intensities = _to_numpy(intensities)
    N = means.shape[0]

    colors = np.stack([intensities, intensities, intensities], axis=-1)
    f_dc = (colors - 0.5) / 0.28209479177387814
    o = np.clip(opacities, 1e-4, 1.0 - 1e-4)
    opacities_logit = np.log(o / (1.0 - o))
    scales_log = np.log(np.maximum(scales, 1e-8))

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    elements = np.empty(N, dtype=dtype)
    elements["x"], elements["y"], elements["z"] = means[:, 0], means[:, 1], means[:, 2]
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0
    elements["f_dc_0"] = f_dc[:, 0]
    elements["f_dc_1"] = f_dc[:, 1]
    elements["f_dc_2"] = f_dc[:, 2]
    elements["opacity"] = opacities_logit.flatten() if opacities_logit.ndim > 1 else opacities_logit
    elements["scale_0"], elements["scale_1"], elements["scale_2"] = scales_log[:, 0], scales_log[:, 1], scales_log[:, 2]
    elements["rot_0"], elements["rot_1"], elements["rot_2"], elements["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


class IOManager:
    """
    专门负责：
    - 加载 / 保存 checkpoint
    - 保存图像到本地
    - 保存 .ply 到本地
    """

    def __init__(
        self,
        output_dir: str,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.device = device
        self.checkpoint_dir = checkpoint_dir or os.path.join(output_dir, "checkpoints")

    def save_checkpoint(
        self,
        ckpt_items: List[Tuple[str, Any]],
        iteration: int,
    ) -> None:
        """按 ckpt_items 将各对象保存到 checkpoint_dir。"""
        ckpt_dir = self.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        for name, obj in ckpt_items:
            path = os.path.join(ckpt_dir, f"iteration_{iteration}_{name}.pth")
            if hasattr(obj, "save"):
                obj.save(path)
            elif isinstance(obj, torch.nn.Module):
                torch.save(obj.state_dict(), path)
            elif isinstance(obj, torch.optim.Optimizer):
                torch.save(obj.state_dict(), path)
            else:
                torch.save(obj, path)
        print(f"[IOManager] Checkpoint saved at iteration {iteration}.")

    def load_checkpoint(
        self,
        ckpt_items: List[Tuple[str, Any]],
        resume_value: str,
    ) -> int:
        """
        从 checkpoint_dir 加载。resume_value 为迭代步数字符串。
        返回加载到的 iteration，无法解析或未配置时返回 0。
        """
        if not resume_value:
            return 0
        try:
            iter_num = int(resume_value)
        except (ValueError, TypeError):
            print(f"[IOManager] Invalid resume value '{resume_value}', expected iteration number.")
            return 0
        if iter_num <= 0:
            return 0
        ckpt_dir = self.checkpoint_dir
        for name, obj in ckpt_items:
            path = os.path.join(ckpt_dir, f"iteration_{iter_num}_{name}.pth")
            if os.path.exists(path):
                if hasattr(obj, "load"):
                    obj.load(path)
                else:
                    blob = torch.load(path, map_location=self.device)
                    if hasattr(obj, "load_state_dict"):
                        if isinstance(blob, dict):
                            if "state_dict" in blob and isinstance(blob["state_dict"], dict):
                                blob = blob["state_dict"]
                        elif isinstance(blob, torch.nn.Module):
                            blob = blob.state_dict()
                        elif isinstance(blob, torch.optim.Optimizer):
                            blob = blob.state_dict()
                        obj.load_state_dict(blob)
                print(f"  -> Loaded {name}")
            else:
                print(f"  -> [Warning] Checkpoint {path} not found.")
        return iter_num

    def save_image(self, path: str, img: Any) -> None:
        """保存单张图像到 path。img 可为 tensor [H,W] 或 numpy。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        _save_image(path, img)

    def save_images(self, dir_path: str, pred, gt, diff=None) -> None:
        """将 pred、gt 及可选的 diff 保存到 dir_path 下 pred.png、gt.png、diff.png。"""
        os.makedirs(dir_path, exist_ok=True)
        _save_image(os.path.join(dir_path, "pred.png"), pred)
        _save_image(os.path.join(dir_path, "gt.png"), gt)
        if diff is not None:
            _save_image(os.path.join(dir_path, "diff.png"), diff)

    def save_ply(self, path: str, container: Any) -> None:
        """
        将高斯容器保存为 .ply。container 需有 .means, .scales, .quats, .opacities, .intensities。
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        means = getattr(container, "means", None)
        scales = getattr(container, "scales", None)
        quats = getattr(container, "quats", None)
        opacities = getattr(container, "opacities", None)
        intensities = getattr(container, "intensities", None)
        if any(x is None for x in (means, scales, quats, opacities, intensities)):
            raise ValueError("container must have means, scales, quats, opacities, intensities")
        _write_gauss_ply(path, means, scales, quats, opacities, intensities)
        N = _to_numpy(means).shape[0]
        print(f"[IOManager] Saved {N} gaussians to {path}.")
