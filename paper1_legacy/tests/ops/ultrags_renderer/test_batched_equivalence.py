"""
GPU equivalence check:
serial `UltragsRenderer` over B slices vs `UltragsBatchedRenderer`.

Run:
    pytest tests/ops/ultrags_renderer/test_batched_equivalence.py -q
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PKG_ROOT = Path(__file__).resolve().parents[3] / "ops" / "ultrags_renderer"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

pytest.importorskip("ultrags_renderer._C", reason="ultrags_renderer CUDA extension not built")

from ultrags_renderer import (  # noqa: E402
    BatchSliceGeometry,
    SliceGeometry,
    UltragsBatchedRenderer,
    UltragsRenderConfig,
    UltragsRenderer,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_matches_serial_forward_backward():
    device = torch.device("cuda")
    B, H, W, G = 2, 24, 24, 192
    P = H * W

    cfg = UltragsRenderConfig(
        H=H,
        W=W,
        lateral_spacing_mm_per_pix=0.2,
        axial_spacing_mm_per_pix=0.2,
        slab_half_thickness_mm=0.2,
        axial_near_mm=0.0,
        k_sigma=2.0,
        mapper_sort_by_depth=True,
        fixed_bg_color=0.0,
        compositing_mode="alpha",
        enable_input_checks=False,
    )

    torch.manual_seed(0)
    means = torch.randn(G, 3, device=device, dtype=torch.float32, requires_grad=True)
    scales = (torch.rand(G, 3, device=device, dtype=torch.float32) * 0.25 + 0.05).requires_grad_(True)
    quats = torch.randn(G, 4, device=device, dtype=torch.float32)
    quats = (quats / quats.norm(dim=-1, keepdim=True)).requires_grad_(True)
    opacities = torch.sigmoid(torch.randn(G, device=device, dtype=torch.float32)).requires_grad_(True)
    intensities = torch.sigmoid(torch.randn(G, device=device, dtype=torch.float32)).requires_grad_(True)

    pixel_pos_list = [torch.randn(P, 3, device=device, dtype=torch.float32) for _ in range(B)]
    geom_list = [
        SliceGeometry(
            origin_world=torch.randn(3, device=device, dtype=torch.float32),
            axis_lateral_world=torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32),
            axis_axial_world=torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32),
            axis_elevational_world=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),
        )
        for _ in range(B)
    ]

    serial_renderer = UltragsRenderer(cfg).to(device)
    serial_imgs = []
    for b in range(B):
        img, _, _ = serial_renderer(
            means,
            scales,
            quats,
            opacities,
            intensities,
            pixel_pos_list[b],
            geom_list[b],
            return_intermediates=False,
        )
        serial_imgs.append(img)

    batched_renderer = UltragsBatchedRenderer(cfg).to(device)
    batch_pos = torch.stack(pixel_pos_list, dim=0)
    batch_geom = BatchSliceGeometry(
        origin_world=torch.stack([g.origin_world for g in geom_list], dim=0),
        axis_lateral_world=torch.stack([g.axis_lateral_world for g in geom_list], dim=0),
        axis_axial_world=torch.stack([g.axis_axial_world for g in geom_list], dim=0),
        axis_elevational_world=torch.stack([g.axis_elevational_world for g in geom_list], dim=0),
    )
    batch_img, _, _ = batched_renderer(
        means,
        scales,
        quats,
        opacities,
        intensities,
        batch_pos,
        batch_geom,
        return_intermediates=False,
    )

    for b in range(B):
        torch.testing.assert_close(batch_img[b], serial_imgs[b], rtol=1e-4, atol=1e-4)

    serial_loss = sum(im.sum() for im in serial_imgs)
    serial_loss.backward(retain_graph=True)
    serial_means_grad = means.grad.detach().clone()

    means.grad.zero_()
    scales.grad.zero_()
    quats.grad.zero_()
    opacities.grad.zero_()
    intensities.grad.zero_()

    batch_loss = batch_img.sum()
    batch_loss.backward()

    torch.testing.assert_close(means.grad, serial_means_grad, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_with_active_mask_matches_serial_subsets():
    device = torch.device("cuda")
    B, H, W, G = 3, 16, 16, 128
    P = H * W

    cfg = UltragsRenderConfig(
        H=H,
        W=W,
        lateral_spacing_mm_per_pix=0.2,
        axial_spacing_mm_per_pix=0.2,
        slab_half_thickness_mm=0.2,
        axial_near_mm=0.0,
        k_sigma=2.0,
        mapper_sort_by_depth=True,
        fixed_bg_color=0.0,
        compositing_mode="alpha",
        enable_input_checks=False,
    )

    torch.manual_seed(1)
    means = torch.randn(G, 3, device=device, dtype=torch.float32, requires_grad=True)
    scales = (torch.rand(G, 3, device=device, dtype=torch.float32) * 0.25 + 0.05).requires_grad_(True)
    quats = torch.randn(G, 4, device=device, dtype=torch.float32)
    quats = (quats / quats.norm(dim=-1, keepdim=True)).requires_grad_(True)
    opacities = torch.sigmoid(torch.randn(G, device=device, dtype=torch.float32)).requires_grad_(True)
    intensities = torch.sigmoid(torch.randn(G, device=device, dtype=torch.float32)).requires_grad_(True)

    pixel_pos_list = [torch.randn(P, 3, device=device, dtype=torch.float32) for _ in range(B)]
    geom_list = [
        SliceGeometry(
            origin_world=torch.randn(3, device=device, dtype=torch.float32),
            axis_lateral_world=torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32),
            axis_axial_world=torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32),
            axis_elevational_world=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),
        )
        for _ in range(B)
    ]

    # Per-sample active subsets
    active_sets = []
    mask = torch.zeros((B, G), device=device, dtype=torch.uint8)
    for b in range(B):
        idx = torch.randperm(G, device=device)[: (G // 2)]
        active_sets.append(idx)
        mask[b, idx] = 1

    serial_renderer = UltragsRenderer(cfg).to(device)
    serial_imgs = []
    for b in range(B):
        idx = active_sets[b]
        img, _, _ = serial_renderer(
            means[idx],
            scales[idx],
            quats[idx],
            opacities[idx],
            intensities[idx],
            pixel_pos_list[b],
            geom_list[b],
            return_intermediates=False,
        )
        serial_imgs.append(img)

    batched_renderer = UltragsBatchedRenderer(cfg).to(device)
    batch_pos = torch.stack(pixel_pos_list, dim=0)
    batch_geom = BatchSliceGeometry(
        origin_world=torch.stack([g.origin_world for g in geom_list], dim=0),
        axis_lateral_world=torch.stack([g.axis_lateral_world for g in geom_list], dim=0),
        axis_axial_world=torch.stack([g.axis_axial_world for g in geom_list], dim=0),
        axis_elevational_world=torch.stack([g.axis_elevational_world for g in geom_list], dim=0),
    )
    batch_img, _, _ = batched_renderer(
        means,
        scales,
        quats,
        opacities,
        intensities,
        batch_pos,
        batch_geom,
        active_gauss_mask_batched=mask,
        return_intermediates=False,
    )

    for b in range(B):
        torch.testing.assert_close(batch_img[b], serial_imgs[b], rtol=1e-4, atol=1e-4)
