import torch

from core.runner.debug.debug_entry import DebugEntry
from core.runner.debug.debug_scope import DebugScope

from .gauss_container import GaussContainer

from typing import Dict

class ImageDebugHook:
    @staticmethod
    def on_images(pred, gt):
        if not DebugScope.enabled(): return
        
        DebugScope.push("debug/img/pred", DebugEntry.image(pred))
        DebugScope.push("debug/img/gt", DebugEntry.image(gt))

class GaussContainerDebugHook:
    @staticmethod
    @torch.no_grad()
    def on_forward(container: GaussContainer):
        if not DebugScope.enabled(): return

        # 只负责采集，不负责计算
        for name in container._param_dict.keys():
            val = container._param_dict[name].value
            if val is not None:
                # 声明：这是一个分布数据
                DebugScope.push(
                    f"debug/gauss/{name}/val",
                    DebugEntry.dist(val)
                )

    @staticmethod
    @torch.no_grad()
    def on_backward(container: GaussContainer):
        if not DebugScope.enabled(): return

        for name in container._param_dict.keys():
            raw_param = container._param_dict[name].tensor
            if raw_param is not None and raw_param.grad is not None:
                # 声明：这是一个梯度数据
                DebugScope.push(
                    f"debug/gauss/{name}/grad", 
                    DebugEntry.grad(raw_param.grad)
                )

class RendererDebugHook:
    @staticmethod
    @torch.no_grad()
    def on_forward(**kwargs: Dict):
        if not DebugScope.enabled(): return
        
        # 1. Pixel Counts: 既要统计也要热力图
        if "pixel_counts" in kwargs: 
            DebugScope.push(
                "debug/renderer/pixel_counts",
                DebugEntry.dist(kwargs["pixel_counts"])
            )
        if "pixel_counts_2d" in kwargs:
            DebugScope.push(
                "debug/renderer/pixel_counts_heatmap",
                DebugEntry.heatmap(kwargs["pixel_counts_2d"], cmap="jet")
            )

        # 2. Pixel Indices: 只要一个数值 统计总共多少对
        if "n_total_pairs" in kwargs:
            DebugScope.push(
                "debug/renderer/n_total_pairs",
                DebugEntry.scalar(kwargs["n_total_pairs"])
            )

        # 3. 各种状态变量 
        choices = ["pixel_accum", "pixel_T", "pixel_contrib"]
        for choice in choices: 
            if choice in kwargs: 
                DebugScope.push(
                    "debug/renderer/{}".format(choice),
                    DebugEntry.dist(kwargs[choice])
                )