import time

import torch

from .config import ModelConfig
from .render import render as render_legacy
from .render_integrated import render as render_integrated


class _DummyDatasetMeta:
    def __init__(self, H: int, W: int):
        self.H = H
        self.W = W
        # 这些值只影响几何尺度，不影响纯速度对比结论
        self.delta_lateral_mm = 0.2
        self.delta_axial_mm = 0.2
        self.delta_elevational_mm = 0.4
        self.near_mm = 0.0


class _DummySliceMeta:
    def __init__(self, H: int, W: int, device: torch.device):
        P = H * W
        self.pixel_pos = torch.randn(P, 3, device=device)
        self.probe_origin = torch.zeros(3, device=device)
        self.vec_lateral = torch.tensor([1.0, 0.0, 0.0], device=device)
        self.vec_axial = torch.tensor([0.0, 1.0, 0.0], device=device)
        self.vec_elevational = torch.tensor([0.0, 0.0, 1.0], device=device)


class _DummyGaussContainer:
    def __init__(self, G: int, device: torch.device):
        self.means = torch.randn(G, 3, device=device)
        self.scales = torch.rand(G, 3, device=device) * 0.5 + 0.1
        self.quats = torch.randn(G, 4, device=device)
        self.opacities = torch.rand(G, device=device)
        self.intensities = torch.rand(G, device=device)


def _time_it(fn, *, n_iters: int = 10) -> float:
    # CUDA 下需要同步，避免异步执行影响计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / n_iters


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = 200_000
    H, W = 256, 256

    print(f"[debug_speed] device={device}, G={G}, H={H}, W={W}")

    cfg = ModelConfig()
    dataset_meta = _DummyDatasetMeta(H, W)
    slice_meta = _DummySliceMeta(H, W, device=device)
    container = _DummyGaussContainer(G, device=device)

    # 预热，触发 kernel 编译等一次性开销
    print("[debug_speed] warmup...")
    for _ in range(2):
        render_legacy(
            cfg,
            dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )
        render_integrated(
            cfg,
            dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )

    n_iters = 10
    print(f"[debug_speed] measuring {n_iters} iters each...")

    def _call_legacy():
        render_legacy(
            cfg,
            dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )

    def _call_integrated():
        render_integrated(
            cfg,
            dataset_meta,
            slice_meta,
            container,
            is_jitter=False,
            return_intermediates=False,
        )

    t_legacy = _time_it(_call_legacy, n_iters=n_iters)
    t_integrated = _time_it(_call_integrated, n_iters=n_iters)

    print(f"[debug_speed] legacy render:     {t_legacy:.3f} ms/iter")
    print(f"[debug_speed] integrated render: {t_integrated:.3f} ms/iter")
    if t_integrated > 0:
        print(f"[debug_speed] ratio (integrated / legacy): {t_integrated / t_legacy:.3f}")


if __name__ == "__main__":
    main()

