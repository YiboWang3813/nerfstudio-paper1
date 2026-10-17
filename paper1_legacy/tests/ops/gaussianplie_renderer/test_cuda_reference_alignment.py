import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

OPS_PATH = Path(__file__).resolve().parents[3] / "ops" / "gaussianplie_renderer"
if str(OPS_PATH) not in sys.path:
    sys.path.insert(0, str(OPS_PATH))

from gaussianplie_renderer import GaussianPlieRenderConfig, GaussianPlieRenderer
from gaussianplie_renderer.grid_geometry import build_slice_grid_pack
from gaussianplie_renderer.reference import render_reference


def _case(device):
    H, W = 7, 7
    lateral_mm = 0.2
    axial_mm = 0.2
    near_mm = -0.6
    sigma_z = 0.4
    cfg = GaussianPlieRenderConfig(
        H=H,
        W=W,
        lateral_spacing_mm_per_pix=lateral_mm,
        axial_spacing_mm_per_pix=axial_mm,
        axial_near_mm=near_mm,
        sigma_z_mm=sigma_z,
        radius_sigma=8.0,
        opacity_cull_threshold=0.0,
        det_eps=1.0e-6,
        cov_eps=1.0e-6,
    )
    meta = SimpleNamespace(
        H=H,
        W=W,
        delta_lateral_mm=lateral_mm,
        delta_axial_mm=axial_mm,
        delta_elevational_mm=sigma_z,
        near_mm=near_mm,
        center_lateral=True,
        probe_type="linear",
    )
    renderer = GaussianPlieRenderer(cfg).to(device)
    renderer.set_slice_grid_geometry(build_slice_grid_pack(meta, device=device))

    means = torch.tensor(
        [[0.0, 0.0, 0.2], [0.2, 0.1, -0.1]],
        device=device,
        dtype=torch.float32,
    )
    scales = torch.tensor(
        [[0.35, 0.40, 0.45], [0.30, 0.50, 0.40]],
        device=device,
        dtype=torch.float32,
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    alphas = torch.tensor([0.4, 0.25], device=device, dtype=torch.float32)
    pose = torch.eye(4, device=device, dtype=torch.float32)

    rows, cols = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (cols - 0.5 * (W - 1)) * lateral_mm
    y = rows * axial_mm + near_mm
    xy_grid = torch.stack([x, y], dim=-1)
    return renderer, means, scales, quats, alphas, pose, xy_grid, sigma_z


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for renderer/reference alignment.")
def test_cuda_forward_matches_focus_reference_for_single_alpha():
    device = torch.device("cuda")
    renderer, means, scales, quats, alphas, pose, xy_grid, sigma_z = _case(device)

    cuda_image = renderer(means, scales, quats, alphas, pose)
    ref_image = render_reference(
        means.double(),
        scales.double(),
        quats.double(),
        alphas.double(),
        pose.double(),
        xy_grid.double(),
        sigma_z=sigma_z,
        det_eps=1.0e-6,
        cov_eps=1.0e-6,
    ).float()

    assert torch.allclose(cuda_image, ref_image, rtol=2.0e-4, atol=2.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for renderer additive alignment.")
def test_cuda_forward_is_additive_in_single_alpha_channel():
    device = torch.device("cuda")
    renderer, means, scales, quats, alphas, pose, _xy_grid, _sigma_z = _case(device)

    one = renderer(means[:1], scales[:1], quats[:1], alphas[:1], pose)
    two = renderer(means[:1].repeat(2, 1), scales[:1].repeat(2, 1), quats[:1].repeat(2, 1), alphas[:1].repeat(2), pose)

    assert torch.allclose(two, one * 2.0, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for renderer gradient checks.")
def test_cuda_backward_alpha_matches_reference_autograd():
    device = torch.device("cuda")
    renderer, means, scales, quats, alphas, pose, xy_grid, sigma_z = _case(device)
    alphas = alphas.clone().requires_grad_(True)
    weights = torch.linspace(0.5, 1.5, renderer.config.H * renderer.config.W, device=device).reshape(
        renderer.config.H, renderer.config.W
    )

    loss = (renderer(means, scales, quats, alphas, pose) * weights).sum()
    loss.backward()
    analytic = alphas.grad.detach().clone()

    ref_alphas = alphas.detach().double().requires_grad_(True)
    ref_loss = (
        render_reference(
            means.detach().double(),
            scales.detach().double(),
            quats.detach().double(),
            ref_alphas,
            pose.detach().double(),
            xy_grid.detach().double(),
            sigma_z=sigma_z,
            det_eps=1.0e-6,
            cov_eps=1.0e-6,
        )
        * weights.double()
    ).sum()
    ref_loss.backward()
    expected = ref_alphas.grad.float()

    assert torch.allclose(analytic, expected, rtol=2.0e-3, atol=2.0e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for renderer gradient checks.")
def test_cuda_backward_scale_includes_determinant_path():
    device = torch.device("cuda")
    renderer, means, scales, quats, alphas, pose, _xy_grid, _sigma_z = _case(device)
    scales = scales.clone().requires_grad_(True)
    weights = torch.linspace(1.5, 0.5, renderer.config.H * renderer.config.W, device=device).reshape(
        renderer.config.H, renderer.config.W
    )

    loss = (renderer(means, scales, quats, alphas, pose) * weights).sum()
    loss.backward()
    analytic = scales.grad[0, 0].detach()

    eps = 2.0e-3
    with torch.no_grad():
        plus = scales.detach().clone()
        minus = scales.detach().clone()
        plus[0, 0] += eps
        minus[0, 0] -= eps
        loss_plus = (renderer(means, plus, quats, alphas, pose) * weights).sum()
        loss_minus = (renderer(means, minus, quats, alphas, pose) * weights).sum()
        fd = (loss_plus - loss_minus) / (2.0 * eps)

    assert torch.isfinite(analytic)
    assert torch.allclose(analytic, fd, rtol=7.5e-2, atol=7.5e-2)
