import torch

from common.debug.debug_scope import DebugScope
from common.debug.debug_utils import (
    range_stats,
    count_stats,
    grad_stats,
)


class RasterizerDebugHook:
    """
    Debug hook for Gaussian Rasterizer (S2S).

    NOTE:
    - d_gps / e_gps are pixel-gauss pairs (can be millions).
    - We subsample them before range_stats to avoid OOM / quantile crash.
    """

    # 最大允许参与 quantile 的元素数
    MAX_STATS_ELEMS = 200_000   # 经验值：安全 + 统计足够稳定

    @staticmethod
    def _subsample(t: torch.Tensor, max_elems: int):
        """
        Uniform subsample a 1D tensor if too large.
        """
        if t.numel() <= max_elems:
            return t

        # 均匀步进采样（确定性，debug 友好）
        step = t.numel() // max_elems
        return t.flatten()[::step]

    @staticmethod
    @torch.no_grad()
    def on_forward(
        *,
        pixel_counts: torch.Tensor,
        pixel_indices: torch.Tensor,
        d_gps: torch.Tensor,
        e_gps: torch.Tensor,
    ):
        if not DebugScope.enabled():
            return

        stats = {}

        # --------------------------------------------------
        # pixel-level stats（安全，保持原样）
        # --------------------------------------------------
        stats.update(
            count_stats(
                pixel_counts,
                "debug/rasterizer/pixel_counts"
            )
        )

        stats["debug/rasterizer/n_pairs"] = int(
            pixel_indices.numel()
        )

        # --------------------------------------------------
        # pair-level stats（SUBSAMPLED）
        # --------------------------------------------------
        d_gps_s = RasterizerDebugHook._subsample(
            d_gps, RasterizerDebugHook.MAX_STATS_ELEMS
        )
        e_gps_s = RasterizerDebugHook._subsample(
            e_gps, RasterizerDebugHook.MAX_STATS_ELEMS
        )

        stats.update(
            range_stats(
                d_gps_s,
                "debug/rasterizer/d_gps"
            )
        )

        stats.update(
            range_stats(
                e_gps_s,
                "debug/rasterizer/e_gps"
            )
        )

        DebugScope.update(stats)

    @staticmethod
    @torch.no_grad()
    def on_backward(
        *,
        grad_image: torch.Tensor,
    ):
        if not DebugScope.enabled():
            return

        stats = {}

        # grad_image 是 pixel-level（H*W），安全
        stats.update(
            grad_stats(
                grad_image,
                "debug/rasterizer/grad/image"
            )
        )

        DebugScope.update(stats)
